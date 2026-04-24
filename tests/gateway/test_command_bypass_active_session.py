"""Regression tests: slash commands must bypass the base adapter's active-session guard.

When an agent is running, the base adapter's Level 1 guard in
handle_message() intercepts all incoming messages and queues them as
pending.  Certain commands (/stop, /new, /reset, /approve, /deny,
/status) must bypass this guard and be dispatched directly to the gateway
runner — otherwise they are queued as user text and either:
  - leak into the conversation as agent input (/stop, /new), or
  - deadlock (/approve, /deny — agent blocks on Event.wait)

These tests verify that the bypass works at the adapter level and that
the safety net in _run_agent discards leaked command text.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubAdapter(BasePlatformAdapter):
    """Concrete adapter with abstract methods stubbed out."""

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    """Create a minimal adapter for testing the active-session guard."""
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.TELEGRAM)
    adapter.sent_responses = []

    async def _mock_handler(event):
        cmd = event.get_command()
        return f"handled:{cmd}" if cmd else f"handled:text:{event.text}"

    adapter._message_handler = _mock_handler

    async def _mock_send_retry(chat_id, content, **kwargs):
        adapter.sent_responses.append(content)

    adapter._send_with_retry = _mock_send_retry
    return adapter


def _make_event(text="/stop", chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _session_key(chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"
    )
    return build_session_key(source)


# ---------------------------------------------------------------------------
# Tests: commands bypass Level 1 when session is active
# ---------------------------------------------------------------------------


class TestCommandBypassActiveSession:
    """Commands that must bypass the active-session guard."""

    @pytest.mark.asyncio
    async def test_stop_bypasses_guard(self):
        """/stop must be dispatched directly, not queued."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/stop"))

        assert sk not in adapter._pending_messages, (
            "/stop was queued as a pending message instead of being dispatched"
        )
        assert any("handled:stop" in r for r in adapter.sent_responses), (
            "/stop response was not sent back to the user"
        )

    @pytest.mark.asyncio
    async def test_new_bypasses_guard(self):
        """/new must be dispatched directly, not queued."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/new"))

        assert sk not in adapter._pending_messages
        assert any("handled:new" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_reset_bypasses_guard(self):
        """/reset (alias for /new) must be dispatched directly."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/reset"))

        assert sk not in adapter._pending_messages
        assert any("handled:reset" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_approve_bypasses_guard(self):
        """/approve must bypass (deadlock prevention)."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/approve"))

        assert sk not in adapter._pending_messages
        assert any("handled:approve" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_deny_bypasses_guard(self):
        """/deny must bypass (deadlock prevention)."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/deny"))

        assert sk not in adapter._pending_messages
        assert any("handled:deny" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_status_bypasses_guard(self):
        """/status must bypass so it returns a system response."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/status"))

        assert sk not in adapter._pending_messages
        assert any("handled:status" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_agents_bypasses_guard(self):
        """/agents must bypass so active-task queries don't interrupt runs."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/agents"))

        assert sk not in adapter._pending_messages
        assert any("handled:agents" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_tasks_alias_bypasses_guard(self):
        """/tasks alias must bypass active-session guard too."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/tasks"))

        assert sk not in adapter._pending_messages
        assert any("handled:tasks" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_background_bypasses_guard(self):
        """/background must bypass so it spawns a parallel task, not an interrupt."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/background summarize HN"))

        assert sk not in adapter._pending_messages, (
            "/background was queued as a pending message instead of being dispatched"
        )
        assert any("handled:background" in r for r in adapter.sent_responses), (
            "/background response was not sent back to the user"
        )

    @pytest.mark.asyncio
    async def test_steer_bypasses_guard(self):
        """/steer must bypass the Level-1 active-session guard so it reaches
        the gateway runner's /steer handler and injects into the running
        agent instead of being queued as user text for the next turn.
        """
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/steer also check auth.log"))

        assert sk not in adapter._pending_messages, (
            "/steer was queued as a pending message instead of being dispatched"
        )
        assert any("handled:steer" in r for r in adapter.sent_responses), (
            "/steer response was not sent back to the user"
        )

    @pytest.mark.asyncio
    async def test_help_bypasses_guard(self):
        """/help must bypass so it is not silently dropped as pending slash text."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/help"))

        assert sk not in adapter._pending_messages, (
            "/help was queued as a pending message instead of being dispatched"
        )
        assert any("handled:help" in r for r in adapter.sent_responses), (
            "/help response was not sent back to the user"
        )

    @pytest.mark.asyncio
    async def test_update_bypasses_guard(self):
        """/update must bypass so it is not discarded by the pending-command safety net."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/update"))

        assert sk not in adapter._pending_messages, (
            "/update was queued as a pending message instead of being dispatched"
        )
        assert any("handled:update" in r for r in adapter.sent_responses), (
            "/update response was not sent back to the user"
        )

    @pytest.mark.asyncio
    async def test_queue_bypasses_guard(self):
        """/queue must bypass so it can queue without interrupting."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/queue follow up"))

        assert sk not in adapter._pending_messages, (
            "/queue was queued as a pending message instead of being dispatched"
        )
        assert any("handled:queue" in r for r in adapter.sent_responses), (
            "/queue response was not sent back to the user"
        )


# ---------------------------------------------------------------------------
# Tests: non-bypass-set commands (no dedicated Level-2 handler) also bypass
# instead of interrupting + being discarded.  Regression for the Discord
# ghost-slash-command bug where /model, /reasoning, /voice, /insights, /title,
# /resume, /retry, /undo, /compress, /usage, /reload-mcp,
# /sethome, /reset silently interrupted the running agent.
# ---------------------------------------------------------------------------


class TestAllResolvableCommandsBypassGuard:
    """Every recognized slash command must bypass the Level-1 active-session
    guard. Without this, commands the user fires mid-run interrupt the agent
    AND get silently discarded by the slash-command safety net (zero-char
    response)."""

    @pytest.mark.parametrize(
        "command_text,canonical",
        [
            ("/model claude-sonnet-4", "model"),
            ("/model", "model"),
            ("/reasoning high", "reasoning"),
            ("/personality default", "personality"),
            ("/voice on", "voice"),
            ("/insights 7", "insights"),
            ("/title my session", "title"),
            ("/resume yesterday", "resume"),
            ("/retry", "retry"),
            ("/undo", "undo"),
            ("/compress", "compress"),
            ("/usage", "usage"),
            ("/reload-mcp", "reload-mcp"),
            ("/sethome", "sethome"),
        ],
    )
    @pytest.mark.asyncio
    async def test_command_bypasses_guard(self, command_text, canonical):
        """Any resolvable slash command bypasses instead of being queued."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event(command_text))

        assert sk not in adapter._pending_messages, (
            f"{command_text} was queued as pending — it should bypass the guard"
        )
        assert len(adapter.sent_responses) > 0, (
            f"{command_text} produced no response — it should be dispatched, "
            "not silently discarded"
        )

    def test_should_bypass_returns_true_for_every_registered_command(self):
        """Spot-check: the commands previously-broken on Discord all bypass."""
        from hermes_cli.commands import should_bypass_active_session

        for cmd in (
            "model", "reasoning", "personality", "voice", "insights", "title",
            "resume", "retry", "undo", "compress", "usage",
            "reload-mcp", "sethome", "reset",
        ):
            assert should_bypass_active_session(cmd) is True, (
                f"/{cmd} must bypass the active-session guard"
            )

    def test_should_bypass_returns_false_for_unknown(self):
        """Unknown words don't bypass — they get queued as user text."""
        from hermes_cli.commands import should_bypass_active_session

        assert should_bypass_active_session("foobar") is False
        assert should_bypass_active_session(None) is False
        assert should_bypass_active_session("") is False
        # A file path split on whitespace: '/path/to/file.py' -> 'path/to/file.py'
        assert should_bypass_active_session("path/to/file.py") is False


# ---------------------------------------------------------------------------
# Tests: non-bypass messages still get queued
# ---------------------------------------------------------------------------


class TestNonBypassStillQueued:
    """Regular messages and unknown commands must be queued, not dispatched."""

    @pytest.mark.asyncio
    async def test_regular_text_queued(self):
        """Plain text while agent is running must be queued as pending."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("hello world"))

        assert sk in adapter._pending_messages, (
            "Regular text was not queued — it should be pending"
        )
        assert len(adapter.sent_responses) == 0, (
            "Regular text should not produce a direct response"
        )

    @pytest.mark.asyncio
    async def test_unknown_command_queued(self):
        """Unknown /commands must be queued, not dispatched."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/foobar"))

        assert sk in adapter._pending_messages
        assert len(adapter.sent_responses) == 0

    @pytest.mark.asyncio
    async def test_file_path_not_treated_as_command(self):
        """A message like '/path/to/file' must not bypass the guard."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/path/to/file.py"))

        assert sk in adapter._pending_messages
        assert len(adapter.sent_responses) == 0


# ---------------------------------------------------------------------------
# Tests: no active session — commands go through normally
# ---------------------------------------------------------------------------


class TestNoActiveSessionNormalDispatch:
    """When no agent is running, messages spawn a background task normally."""

    @pytest.mark.asyncio
    async def test_stop_when_no_session_active(self):
        """/stop without an active session spawns a background task
        (the Level 2 handler will return 'No active task')."""
        adapter = _make_adapter()
        sk = _session_key()

        # No active session — _active_sessions is empty
        assert sk not in adapter._active_sessions

        await adapter.handle_message(_make_event("/stop"))

        # Should have gone through the normal path (background task spawned)
        # and NOT be in _pending_messages (that's the queued-during-active path)
        assert sk not in adapter._pending_messages


# ---------------------------------------------------------------------------
# Tests: safety net in _run_agent discards command text from pending queue
# ---------------------------------------------------------------------------


class TestPendingCommandSafetyNet:
    """The safety net in gateway/run.py _run_agent must discard command text
    that leaks into the pending queue via interrupt_message fallback."""

    def test_stop_command_detected(self):
        """resolve_command must recognize /stop so the safety net can
        discard it."""
        from hermes_cli.commands import resolve_command

        assert resolve_command("stop") is not None
        assert resolve_command("stop").name == "stop"

    def test_new_command_detected(self):
        from hermes_cli.commands import resolve_command

        assert resolve_command("new") is not None
        assert resolve_command("new").name == "new"

    def test_reset_alias_detected(self):
        from hermes_cli.commands import resolve_command

        assert resolve_command("reset") is not None
        assert resolve_command("reset").name == "new"  # alias

    def test_unknown_command_not_detected(self):
        from hermes_cli.commands import resolve_command

        assert resolve_command("foobar") is None

    def test_file_path_not_detected_as_command(self):
        """'/path/to/file' should not resolve as a command."""
        from hermes_cli.commands import resolve_command

        # The safety net splits on whitespace and takes the first word
        # after stripping '/'.  For '/path/to/file', that's 'path/to/file'.
        assert resolve_command("path/to/file") is None


# ---------------------------------------------------------------------------
# Tests: bypass with @botname suffix (Telegram-style)
# ---------------------------------------------------------------------------


class TestBypassWithBotnameSuffix:
    """Telegram appends @botname to commands. The bypass must still work."""

    @pytest.mark.asyncio
    async def test_stop_with_botname(self):
        """/stop@MyHermesBot must bypass the guard."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/stop@MyHermesBot"))

        assert sk not in adapter._pending_messages, (
            "/stop@MyHermesBot was queued instead of bypassing"
        )
        assert any("handled:stop" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_new_with_botname(self):
        """/new@MyHermesBot must bypass the guard."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/new@MyHermesBot"))

        assert sk not in adapter._pending_messages
        assert any("handled:new" in r for r in adapter.sent_responses)
