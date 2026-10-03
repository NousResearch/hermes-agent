"""Regression tests: a logged-out WhatsApp session must not respawn the bridge forever.

The Baileys bridge exits 1 when WhatsApp ends the session — an explicit logout,
or the linked device removed from the phone. Both arrive as status 401, and
Baileys' documented lifecycle treats that code as terminal
(``shouldReconnect = statusCode !== DisconnectReason.loggedOut``).

The gateway sees only the child's exit code, where that code is
indistinguishable from a crash, so ``_check_managed_bridge_exit`` marked every
exit ``retryable=True``. The gateway's reconnect watcher retries retryable
platforms forever (30s → 300s backoff) and drops non-retryable ones, so a
session that could never work again was re-spawned indefinitely — 300+ kill and
restart cycles over three days in one reported incident, with no re-pair signal
anywhere the user could see (#80088).

The bridge now records the terminal reason in ``bridge-exit.json`` beside the
session, and the adapter classifies that exit as a non-retryable fatal carrying
re-pair instructions. The crash path keeps its behaviour: an exit nobody
explained stays retryable.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


class _ExitedProc:
    """Stand-in for a finished bridge subprocess."""

    def __init__(self, code: int):
        self.returncode = code
        self._code = code

    def poll(self):
        return self._code


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter(session_path: Path, bridge_script: str = "/tmp/test-bridge.js"):
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19877
    adapter._bridge_script = bridge_script
    adapter._session_path = session_path
    adapter._bridge_process = None
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._reply_prefix = None
    adapter._send_read_receipts = False
    adapter._dm_policy = adapter._group_policy = "pairing"
    adapter._allow_from = adapter._group_allow_from = set()
    adapter._running = False
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = None
    return adapter


def _record_exit(session_path: Path, payload) -> None:
    """Write the record the bridge leaves behind, verbatim or as JSON."""
    session_path.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (session_path / "bridge-exit.json").write_text(text, encoding="utf-8")


async def _exit(adapter, code: int = 1):
    adapter._bridge_process = _ExitedProc(code)
    return await adapter._check_managed_bridge_exit()


class TestTerminalExit:
    @pytest.mark.asyncio
    async def test_logged_out_exit_is_not_retryable(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        _record_exit(tmp_path, {"reason": "logged_out", "statusCode": 401})

        message = await _exit(adapter)

        assert adapter.fatal_error_retryable is False, (
            "a logged-out session cannot be revived by another reconnect; a retryable fatal "
            "is what makes the gateway respawn the bridge forever"
        )
        assert adapter.fatal_error_code == "whatsapp_logged_out"
        assert "was ended from the phone" in message
        assert "hermes whatsapp" in message, "the fatal error must tell the user how to re-pair"

    @pytest.mark.asyncio
    async def test_terminal_exit_is_reported_even_for_a_clean_exit_code(self, tmp_path):
        """The reason decides, not the number: the bridge exits 1, but the classification must not depend on that."""
        adapter = _make_adapter(tmp_path)
        _record_exit(tmp_path, {"reason": "logged_out", "statusCode": 401})

        await _exit(adapter, code=0)

        assert adapter.fatal_error_retryable is False


class TestCrashPathUnchanged:
    @pytest.mark.asyncio
    async def test_exit_without_a_record_stays_retryable(self, tmp_path):
        adapter = _make_adapter(tmp_path)

        message = await _exit(adapter, code=-9)

        assert adapter.fatal_error_retryable is True
        assert adapter.fatal_error_code == "whatsapp_bridge_exited"
        assert "-9" in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "record",
        [
            "{not json",
            "",
            '{"reason": "something_new", "statusCode": 999}',
            '["logged_out"]',
            '{"statusCode": 401}',
        ],
        ids=["malformed", "empty", "unknown-reason", "not-an-object", "missing-reason"],
    )
    async def test_unusable_record_is_a_crash(self, tmp_path, record):
        """An unreadable or unrecognized record must not be trusted into a terminal classification."""
        adapter = _make_adapter(tmp_path)
        _record_exit(tmp_path, record)

        await _exit(adapter)

        assert adapter.fatal_error_retryable is True
        assert adapter.fatal_error_code == "whatsapp_bridge_exited"

    @pytest.mark.asyncio
    async def test_intentional_shutdown_still_wins(self, tmp_path):
        """A clean exit during shutdown is not a failure, whatever the stale record says."""
        adapter = _make_adapter(tmp_path)
        _record_exit(tmp_path, {"reason": "logged_out", "statusCode": 401})
        adapter._shutting_down = True

        result = await _exit(adapter, code=0)

        assert result is None
        assert adapter.has_fatal_error is False


class TestRecordIsClearedBeforeSpawn:
    @pytest.mark.asyncio
    async def test_a_stale_record_cannot_classify_the_next_process_exit(self, tmp_path):
        """The record describes one process, so it must not outlive it into the next bridge's exit.

        The two halves are one contract: the real spawn preparation clears the record, and an
        exit that follows with no record of its own stays a retryable crash instead of
        inheriting the previous session's terminal reason.
        """
        bridge_dir = tmp_path / "whatsapp-bridge"
        bridge_dir.mkdir()
        (bridge_dir / "bridge.js").write_text("// bridge\n", encoding="utf-8")
        (bridge_dir / "package.json").write_text('{"name": "bridge"}\n', encoding="utf-8")
        session = tmp_path / "session"
        _record_exit(session, {"reason": "logged_out", "statusCode": 401})
        (session / "creds.json").write_text("{}", encoding="utf-8")

        from plugins.platforms.whatsapp.adapter import _file_content_hash

        node_modules = bridge_dir / "node_modules"
        node_modules.mkdir()
        (node_modules / ".hermes-pkg-hash").write_text(_file_content_hash(bridge_dir / "package.json"), encoding="utf-8")

        adapter = _make_adapter(session, bridge_script=str(bridge_dir / "bridge.js"))
        proc = MagicMock()
        proc.poll.return_value = None

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch.object(adapter, "_wait_for_bridge", new_callable=AsyncMock, return_value=True), \
             patch.object(adapter, "_attach_to_bridge"), \
             patch.object(adapter, "_mark_connected"), \
             patch.object(adapter, "_wire_plugin_handlers"), \
             patch("subprocess.Popen", return_value=proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        mock_popen.assert_called_once()
        assert not (session / "bridge-exit.json").exists(), (
            "a record left by the previous bridge would classify the new process's crash as terminal"
        )

        # The spawned bridge dies without recording anything of its own. The previous
        # session's record is gone, so this exit is an ordinary crash and must stay retryable.
        adapter._bridge_process = _ExitedProc(1)

        message = await adapter._check_managed_bridge_exit()

        assert adapter.fatal_error_retryable is True, (
            "the previous session's terminal record classified an unrelated exit; the bridge "
            "would be dropped as permanently unusable without a re-pair that never happened"
        )
        assert adapter.fatal_error_code == "whatsapp_bridge_exited"
        assert "exited unexpectedly" in message

    def test_clear_is_idempotent(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _clear_bridge_exit, _read_bridge_exit_reason

        _clear_bridge_exit(tmp_path)  # nothing recorded yet
        _record_exit(tmp_path, {"reason": "logged_out", "statusCode": 401})
        assert _read_bridge_exit_reason(tmp_path) == "logged_out"

        _clear_bridge_exit(tmp_path)

        assert _read_bridge_exit_reason(tmp_path) is None
