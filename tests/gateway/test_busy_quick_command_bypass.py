"""Tests for exec-type quick commands bypassing the busy-session queue (#25783).

When the agent is mid-turn, ``/<name>`` quick commands of type ``exec`` should
run inline against the gateway's subprocess machinery instead of getting
queued/interrupted and replayed to the LLM as raw text.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can import gateway code without heavy deps
# ---------------------------------------------------------------------------
import sys, types

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)


def _make_event(text="/note hello world", chat_id="123", platform_val="telegram"):
    source = SessionSource(
        platform=MagicMock(value=platform_val),
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )


def _make_runner(quick_commands=None):
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = {"quick_commands": quick_commands or {}}
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._reply_anchor_for_event = lambda _e: "msg1"
    runner._thread_metadata_for_source = lambda _src, _anchor=None: {}
    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter(platform_val="telegram"):
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_val)
    return adapter


class _FakeProc:
    def __init__(self, stdout=b"note saved\n", stderr=b""):
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecQuickCommandBypassesBusySession:
    """An exec quick command must run inline even when the agent is busy."""

    @pytest.mark.asyncio
    async def test_exec_quick_command_runs_inline_during_busy_session(self):
        runner, _sentinel = _make_runner(
            quick_commands={"note": {"type": "exec", "command": "bash daily-note.sh hello"}}
        )
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="/note remember to buy milk")
        sk = build_session_key(event.source)

        running_agent = MagicMock()
        runner._running_agents[sk] = running_agent
        runner.adapters[event.source.platform] = adapter

        fake_proc = _FakeProc(stdout=b"note saved\n")
        with patch("gateway.run.asyncio.create_subprocess_shell", new=AsyncMock(return_value=fake_proc)) as mock_spawn, \
             patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        # The subprocess actually ran (the bug was that it didn't).
        mock_spawn.assert_called_once()
        # The exec output was delivered to the user.
        adapter._send_with_retry.assert_called_once()
        call_kwargs = adapter._send_with_retry.call_args.kwargs
        assert call_kwargs.get("content") == "note saved"
        # The running agent was NOT touched — no interrupt, no queue.
        running_agent.interrupt.assert_not_called()
        running_agent.steer.assert_not_called()
        mock_merge.assert_not_called()
        assert adapter._pending_messages == {}

    @pytest.mark.asyncio
    async def test_non_exec_command_falls_through_to_queue_logic(self):
        # Regression guard: only ``exec`` quick commands bypass the queue.
        # Plain user messages and ``alias`` quick commands keep their normal
        # busy semantics (queue + ack), so the agent still hears them.
        runner, _sentinel = _make_runner(
            quick_commands={"hello": {"type": "alias", "target": "/help"}}
        )
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="just a normal follow-up")
        sk = build_session_key(event.source)

        running_agent = MagicMock()
        runner._running_agents[sk] = running_agent
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.asyncio.create_subprocess_shell", new=AsyncMock()) as mock_spawn, \
             patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        # No subprocess for a plain message.
        mock_spawn.assert_not_called()
        # Normal queue semantics applied.
        mock_merge.assert_called_once()
        # User got the queue ack.
        adapter._send_with_retry.assert_called_once()
        ack_content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Queued for the next turn" in ack_content

    @pytest.mark.asyncio
    async def test_alias_quick_command_is_not_short_circuited(self):
        # ``alias`` rewrites into other commands and ultimately reaches the
        # agent. It must NOT be intercepted by the busy bypass — otherwise
        # ``/foo`` aliased to ``/help`` would silently disappear during a
        # busy session.
        runner, _sentinel = _make_runner(
            quick_commands={"foo": {"type": "alias", "target": "help"}}
        )
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="/foo")
        sk = build_session_key(event.source)

        running_agent = MagicMock()
        runner._running_agents[sk] = running_agent
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.asyncio.create_subprocess_shell", new=AsyncMock()) as mock_spawn, \
             patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        mock_spawn.assert_not_called()
        # Alias falls through to the queue path so the agent eventually sees it.
        mock_merge.assert_called_once()

    @pytest.mark.asyncio
    async def test_exec_quick_command_blocked_during_drain(self):
        # Symmetric with the non-busy ``_handle_message`` flow: when the
        # gateway is draining, the user gets the draining message instead of
        # the exec subprocess running. The exec bypass is positioned AFTER
        # the draining branch in ``_handle_active_session_busy_message`` for
        # exactly this reason.
        runner, _sentinel = _make_runner(
            quick_commands={"note": {"type": "exec", "command": "echo hi"}}
        )
        runner._busy_input_mode = "interrupt"
        runner._draining = True
        runner._status_action_gerund = lambda: "restarting"
        runner._queue_during_drain_enabled = lambda: False
        adapter = _make_adapter()

        event = _make_event(text="/note while draining")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.asyncio.create_subprocess_shell", new=AsyncMock()) as mock_spawn:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        mock_spawn.assert_not_called()
        ack = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "restarting" in ack

    @pytest.mark.asyncio
    async def test_helper_returns_none_for_non_exec_command(self):
        # The helper drives both the cold path (``_handle_message``) and the
        # busy path (``_handle_active_session_busy_message``). It must
        # cleanly return None for non-matching inputs so callers fall
        # through instead of swallowing the message.
        runner, _ = _make_runner(
            quick_commands={"note": {"type": "exec", "command": "echo hi"}}
        )
        assert await runner._try_execute_exec_quick_command(None) is None
        assert await runner._try_execute_exec_quick_command("") is None
        assert await runner._try_execute_exec_quick_command("not-registered") is None

        # Wrong type → still None so the cold path's alias/unsupported
        # branches handle it.
        runner_alias, _ = _make_runner(
            quick_commands={"foo": {"type": "alias", "target": "/help"}}
        )
        assert await runner_alias._try_execute_exec_quick_command("foo") is None

    @pytest.mark.asyncio
    async def test_helper_reports_missing_exec_command_string(self):
        runner, _ = _make_runner(
            quick_commands={"broken": {"type": "exec"}}  # no "command" key
        )
        output = await runner._try_execute_exec_quick_command("broken")
        assert output == "Quick command '/broken' has no command defined."
