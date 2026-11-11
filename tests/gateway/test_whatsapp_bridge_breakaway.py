"""WhatsApp bridge spawn must fall back when the job object forbids breakaway.

Regression for a scheduled-task gateway (Task Scheduler runs the gateway in a
job object without JOB_OBJECT_LIMIT_BREAKAWAY_OK): the bridge spawn carries
CREATE_BREAKAWAY_FROM_JOB, which Windows rejects with WinError 5 (access
denied). Without a fallback the WhatsApp adapter can never start its Node
bridge from a scheduled-task gateway — every reconnect attempt fails, forever.
hermes_cli.gateway_windows._spawn_detached established the canonical fallback:
catch the OSError, retry without the breakaway flag.
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


def _bridge_not_reusable(adapter):
    """Mock _reuse_running_bridge as 'nothing to adopt' while keeping its entry
    contract: the real method resets the foreign-session and probe-timeout
    markers on entry, and connect() reads both right after the call. A bare
    AsyncMock skips the reset and connect() then dies on unset attributes."""
    async def _not_reusable(bridge_path):
        adapter._foreign_bridge_session = None
        adapter._bridge_probe_timed_out = False
        return False
    return patch.object(adapter, "_reuse_running_bridge", new=_not_reusable)


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
        calls = []

        def _spawn_owned_side_effect(argv, **kwargs):
            creationflags = kwargs.get("creationflags", 0)
            created.append(creationflags)
            calls.append((argv, kwargs))
            if creationflags & _BREAKAWAY_BIT:
                # spawn_owned is a thin Popen wrapper on Windows; a real Popen
                # there carries .winerror, which PermissionError(None, msg,
                # None, 5) reproduces.
                raise PermissionError(None, "Access is denied", None, 5)
            return mock_proc

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter.find_node_executable", return_value="node"), \
             patch.object(adapter, "_ensure_bridge_deps", return_value=True), \
             _bridge_not_reusable(adapter), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned",
                   side_effect=_spawn_owned_side_effect) as mock_spawn, \
             patch("builtins.open", return_value=MagicMock()), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter._write_bridge_pidfile"), \
             patch.object(adapter, "_wait_for_bridge", new_callable=AsyncMock, return_value=True):
            result = await adapter.connect()

        # First spawn requested breakaway and was refused; the retry dropped
        # CREATE_BREAKAWAY_FROM_JOB — and the bridge started.
        assert mock_spawn.call_count == 2
        assert _BREAKAWAY_BIT & created[0], "first attempt must request breakaway"
        assert not (_BREAKAWAY_BIT & created[1]), "retry must drop CREATE_BREAKAWAY_FROM_JOB"
        assert created[1] == _no_breakaway_flags()
        # The retry is the same child: argv and the ownership/piping kwargs survive.
        assert calls[1][0] == calls[0][0]
        for key in ("purpose", "stdin", "stdout", "stderr", "env"):
            assert calls[1][1][key] == calls[0][1][key], f"retry dropped {key}"
        assert result is True
        assert adapter._bridge_process is mock_proc

    @pytest.mark.asyncio
    async def test_other_oserror_not_retried(self):
        """Only access-denied (winerror 5) is retried; other OSErrors surface
        as a connect failure without a second spawn."""
        adapter = _make_adapter()

        def _spawn_owned_side_effect(argv, **kwargs):
            raise OSError("disk quota exceeded")

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter.find_node_executable", return_value="node"), \
             patch.object(adapter, "_ensure_bridge_deps", return_value=True), \
             _bridge_not_reusable(adapter), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned",
                   side_effect=_spawn_owned_side_effect) as mock_spawn, \
             patch("builtins.open", return_value=MagicMock()), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter._write_bridge_pidfile"), \
             patch.object(adapter, "_wait_for_bridge", new_callable=AsyncMock, return_value=True):
            result = await adapter.connect()

        assert result is False
        assert mock_spawn.call_count == 1, "non-winerror-5 OSError must not be retried"
