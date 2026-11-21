"""``disconnect()`` stops the WhatsApp bridge on Windows, and keeps track of a bridge it could not stop.

The bridge runs without a window, so ``taskkill /T`` without ``/F`` cannot close it: taskkill exits 128
("can only be terminated forcefully") or hangs until its 10 s timeout. ``disconnect()`` tried that first,
the error skipped the forced stop, and ``bridge.pid`` was unlinked anyway: the bridge kept its port and the
WhatsApp session, and nothing identified it on the next start. A hang also outlasted the gateway's
disconnect budget before ``/F`` could run.
"""
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


def _adapter(session_path, proc):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._shutting_down = False
    adapter._session_path = session_path
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._poll_task = None
    adapter._http_session = None
    adapter._release_platform_lock = MagicMock()
    adapter._mark_disconnected = MagicMock()
    adapter._bridge_process = proc
    return adapter


def _windowless_child():
    """A child started like the bridge: its own process group and no console window."""
    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway

    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        creationflags=windows_detach_flags_without_breakaway(),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _reap(child):
    """Stop the child's whole tree and wait for it, whatever the test did."""
    if child.poll() is None:
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, check=False)
        child.wait(10)


@pytest.mark.platforms("windows")
def test_taskkill_without_f_cannot_stop_a_windowless_child():
    """Why disconnect() skips the soft stop on Windows."""
    from plugins.platforms.whatsapp.adapter import _terminate_bridge_process

    child = _windowless_child()
    try:
        with pytest.raises((OSError, subprocess.TimeoutExpired)):
            _terminate_bridge_process(child, force=False)
        assert child.poll() is None
    finally:
        _reap(child)


@pytest.mark.asyncio
@pytest.mark.platforms("windows")
async def test_disconnect_stops_a_windowless_bridge_within_the_gateway_budget(tmp_path):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    child = _windowless_child()
    try:
        (tmp_path / "bridge.pid").write_text(str(child.pid), encoding="utf-8")
        adapter = _adapter(tmp_path, child)
        timeout = runner._adapter_disconnect_timeout_secs()

        assert await runner._await_adapter_cleanup_with_timeout(adapter.disconnect(), timeout)
        assert child.poll() is not None, "disconnect() left the bridge running"
        assert not (tmp_path / "bridge.pid").exists()
        assert adapter._bridge_process is None
    finally:
        _reap(child)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forced_stop_error", [None, OSError("taskkill exited 1"), subprocess.TimeoutExpired("taskkill", 10)]
)
async def test_bridge_that_outlives_disconnect_stays_owned_with_its_pidfile(tmp_path, forced_stop_error):
    """A bridge the forced stop did not end keeps bridge.pid across disconnects, so the next connect() can reap it."""
    proc = MagicMock(pid=4242, returncode=None)
    proc.poll.return_value = None
    (tmp_path / "bridge.pid").write_text("4242", encoding="utf-8")
    adapter = _adapter(tmp_path, proc)
    forced = []

    def stop(_proc, *, force):
        forced.append(force)
        if force and forced_stop_error:
            raise forced_stop_error

    with patch("plugins.platforms.whatsapp.adapter._terminate_bridge_process", side_effect=stop), \
         patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock):
        await adapter.disconnect()
        await adapter.disconnect()

    assert (tmp_path / "bridge.pid").exists()
    assert adapter._bridge_process is proc
    assert forced.count(True) == 2, "each disconnect() retries the forced stop"
