"""Regression test: pre_gateway_dispatch must run on the busy path.

Background:

The idle path (no active agent) invokes the ``pre_gateway_dispatch``
plugin hook inside ``GatewayRunner._handle_message`` so plugins like
``gateway-policy`` (handover, listen-only) can return
``{"action": "skip"}`` to silence an inbound message.

The busy path (agent already running for the same ``session_key``) used
to bypass that hook entirely: ``BasePlatform.handle_message`` short-
circuited into ``_handle_active_session_busy_message`` which sent its
own ack ("⚡ Interrupting current task...") and returned, so plugins
never got a chance to silence the message.  Symptom: during an active
handover, owner-typed asides triggered the busy ack instead of being
suppressed.

Fix: ``_handle_active_session_busy_message`` now consults
``pre_gateway_dispatch`` first.  A ``skip`` short-circuits with no ack
and no queue.  Other return shapes (rewrite/allow/None) and hook
exceptions fall through to existing behavior — only ``skip`` is
honored mid-run because the only plugin that emits it (gateway-policy)
is what we need to support, and rewriting an event we're about to
interrupt with is racy.

Style mirrors ``tests/gateway/test_busy_session_ack.py`` which already
exercises ``_handle_active_session_busy_message`` directly.
"""
import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so we can import gateway code without heavy deps
# (mirrors test_busy_session_ack.py)
# ---------------------------------------------------------------------------

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

from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text="hello", chat_id="123", platform_val="whatsapp"):
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


def _make_runner():
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter(platform_val="whatsapp"):
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_val)
    return adapter


def _wire_running_agent(runner, sk, adapter):
    """Install a running agent so the method takes the busy path."""
    agent = MagicMock()
    agent.get_activity_summary.return_value = {
        "api_call_count": 1,
        "max_iterations": 60,
        "current_tool": None,
        "last_activity_ts": time.time(),
        "last_activity_desc": "api",
        "seconds_since_activity": 0.1,
    }
    runner._running_agents[sk] = agent
    runner._running_agents_ts[sk] = time.time() - 5
    runner.adapters[adapter.platform] = adapter
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBusyPathPreGatewayDispatch:
    """``_handle_active_session_busy_message`` consults the hook first."""

    @pytest.mark.asyncio
    async def test_skip_short_circuits_no_ack_no_queue(self):
        """Hook returning skip => return True, no ack, no merge into pending."""
        runner, _ = _make_runner()
        adapter = _make_adapter()

        event = _make_event(text="owner aside during handover")
        sk = build_session_key(event.source)
        agent = _wire_running_agent(runner, sk, adapter)
        runner.adapters[event.source.platform] = adapter

        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"action": "skip", "reason": "handover_active"}],
        ) as mock_invoke, patch(
            "gateway.run.merge_pending_message_event"
        ) as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True, "skip must report 'handled'"
        adapter._send_with_retry.assert_not_called(), "no busy-ack on skip"
        mock_merge.assert_not_called(), "skip must not enqueue the event"
        agent.interrupt.assert_not_called(), "skip must not interrupt the run"
        assert sk not in runner._busy_ack_ts, "skip should not stamp ack-cooldown"

        # Idle-path contract: signature must match exactly.
        mock_invoke.assert_called_once()
        args, kwargs = mock_invoke.call_args
        assert args[0] == "pre_gateway_dispatch"
        assert kwargs["event"] is event
        assert kwargs["gateway"] is runner
        assert kwargs["session_store"] is runner.session_store

    @pytest.mark.asyncio
    async def test_no_plugin_results_falls_through_to_existing_behavior(self):
        """Empty / None hook results => existing busy-ack behavior preserved."""
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="ping")
        sk = build_session_key(event.source)
        agent = _wire_running_agent(runner, sk, adapter)
        runner.adapters[event.source.platform] = adapter

        for hook_return in ([], None):
            adapter._send_with_retry.reset_mock()
            agent.interrupt.reset_mock()
            runner._busy_ack_ts.clear()

            with patch(
                "hermes_cli.plugins.invoke_hook", return_value=hook_return
            ):
                result = await runner._handle_active_session_busy_message(event, sk)

            assert result is True
            adapter._send_with_retry.assert_called_once()
            content = adapter._send_with_retry.call_args.kwargs.get("content", "")
            assert "Interrupting" in content
            agent.interrupt.assert_called_once_with("ping")

    @pytest.mark.asyncio
    async def test_hook_exception_falls_through_with_warning(self, caplog):
        """Hook raising must not break the busy path; busy-ack still sent."""
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="boom")
        sk = build_session_key(event.source)
        agent = _wire_running_agent(runner, sk, adapter)
        runner.adapters[event.source.platform] = adapter

        with patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=RuntimeError("plugin crashed"),
        ):
            with caplog.at_level("WARNING", logger="gateway.run"):
                result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        adapter._send_with_retry.assert_called_once()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Interrupting" in content
        agent.interrupt.assert_called_once_with("boom")
        assert any(
            "pre_gateway_dispatch (busy path) invocation failed" in rec.message
            for rec in caplog.records
        ), "exception must be logged so it's debuggable"

    @pytest.mark.asyncio
    async def test_rewrite_and_allow_fall_through_unchanged(self):
        """Rewrite/allow are NOT honored on the busy path; behavior unchanged."""
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="original text")
        sk = build_session_key(event.source)
        agent = _wire_running_agent(runner, sk, adapter)
        runner.adapters[event.source.platform] = adapter

        for hook_return in (
            [{"action": "rewrite", "text": "REPLACED"}],
            [{"action": "allow"}],
        ):
            adapter._send_with_retry.reset_mock()
            agent.interrupt.reset_mock()
            runner._busy_ack_ts.clear()

            with patch(
                "hermes_cli.plugins.invoke_hook", return_value=hook_return
            ):
                result = await runner._handle_active_session_busy_message(event, sk)

            assert result is True
            adapter._send_with_retry.assert_called_once()
            content = adapter._send_with_retry.call_args.kwargs.get("content", "")
            assert "Interrupting" in content
            assert "REPLACED" not in content, (
                "rewrite must not actually rewrite mid-run on the busy path"
            )
            agent.interrupt.assert_called_once_with("original text"), (
                "interrupt payload must come from the original event, not the rewrite"
            )

    @pytest.mark.asyncio
    async def test_skip_works_on_draining_path_too(self):
        """Skip should also short-circuit the draining branch."""
        runner, _ = _make_runner()
        runner._draining = True
        runner._queue_during_drain_enabled = lambda: False
        runner._status_action_gerund = lambda: "restarting"

        adapter = _make_adapter()
        event = _make_event(text="aside during drain")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"action": "skip", "reason": "handover_active"}],
        ):
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        adapter._send_with_retry.assert_not_called(), (
            "skip must suppress the drain-status ack too"
        )
