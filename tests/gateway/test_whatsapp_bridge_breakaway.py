"""WhatsApp bridge spawn must fall back when the job object forbids breakaway.

Regression for a scheduled-task gateway (Task Scheduler runs the gateway in a
job object without JOB_OBJECT_LIMIT_BREAKAWAY_OK): the adapter's bridge Popen
carries CREATE_BREAKAWAY_FROM_JOB, which Windows rejects with WinError 5
(access denied). Without a fallback the WhatsApp adapter can never start its
Node bridge from a scheduled-task gateway — every reconnect attempt fails,
forever. hermes_cli.gateway_windows._spawn_detached established the canonical
fallback: catch the OSError, retry without the breakaway flag.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.test_whatsapp_connect import _make_adapter

# CREATE_BREAKAWAY_FROM_JOB (0x01000000) in the Windows creationflags.
_BREAKAWAY_BIT = 0x01000000


def _no_breakaway_flags() -> int:
    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway
    return windows_detach_flags_without_breakaway()


@pytest.mark.platforms("windows")
class TestBridgeSpawnBreakawayFallback:
    """connect() retries the bridge spawn without breakaway on WinError 5.

    Faking sys.platform on Linux could not reach the real Windows creationflags
    path; this runs on the Windows CI job instead (convention of
    tests/gateway/test_restart_drain.py).
    """

    @pytest.mark.asyncio
    async def test_connect_falls_back_without_breakaway(self):
        adapter = _make_adapter()

        mock_proc = MagicMock()
        created = []

        def _popen_side_effect(argv, **kwargs):
            creationflags = kwargs.get("creationflags", 0)
            created.append(creationflags)
            if creationflags & _BREAKAWAY_BIT:
                # Real Popen on Windows carries .winerror; constructing with
                # PermissionError(None, msg, None, 5) reproduces that shape.
                raise PermissionError(None, "Access is denied", None, 5)
            return mock_proc

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter.find_node_executable", return_value="node"), \
             patch.object(adapter, "_ensure_bridge_deps", return_value=True), \
             patch.object(adapter, "_reuse_running_bridge", new_callable=AsyncMock, return_value=False), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("subprocess.Popen", side_effect=_popen_side_effect) as mock_popen, \
             patch("builtins.open", return_value=MagicMock()), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter._write_bridge_pidfile"), \
             patch.object(adapter, "_wait_for_bridge", new_callable=AsyncMock, return_value=True):
            result = await adapter.connect()

        # First spawn requested breakaway and was refused; the retry dropped
        # CREATE_BREAKAWAY_FROM_JOB — and the bridge started.
        assert mock_popen.call_count == 2
        assert _BREAKAWAY_BIT & created[0], "first attempt must request breakaway"
        assert not (_BREAKAWAY_BIT & created[1]), "retry must drop CREATE_BREAKAWAY_FROM_JOB"
        assert created[1] == _no_breakaway_flags()
        assert result is True
        assert adapter._bridge_process is mock_proc

    @pytest.mark.asyncio
    async def test_other_oserror_not_retried(self):
        """Only access-denied (winerror 5) is retried; other OSErrors surface
        as a connect failure without a second spawn."""
        adapter = _make_adapter()

        def _popen_side_effect(argv, **kwargs):
            raise OSError("disk quota exceeded")

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter.find_node_executable", return_value="node"), \
             patch.object(adapter, "_ensure_bridge_deps", return_value=True), \
             patch.object(adapter, "_reuse_running_bridge", new_callable=AsyncMock, return_value=False), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("subprocess.Popen", side_effect=_popen_side_effect) as mock_popen, \
             patch("builtins.open", return_value=MagicMock()), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter._write_bridge_pidfile"), \
             patch.object(adapter, "_wait_for_bridge", new_callable=AsyncMock, return_value=True):
            result = await adapter.connect()

        assert result is False
        assert mock_popen.call_count == 1, "non-winerror-5 OSError must not be retried"