"""
Tests for interactive approval + stop on the /api/sessions/{id}/chat/stream path.

The session chat stream reaches the agent through ``_run_agent`` and, before
this change, registered nothing in the shared run maps — so guarded tools
failed closed with no ``approval.request`` event, and there was no way to stop
a turn.  This wires the streaming session turn into the same run maps the
``/v1/runs`` control endpoints already use, reusing them unchanged:

- ``_run_agent`` gains ``approval_notify_callback`` (register/unregister the
  gateway approval notifier for the turn) and ``run_id`` (publish the live
  ``AIAgent`` in ``_active_run_agents[run_id]`` so ``/v1/runs/{id}/stop`` can
  interrupt it).
- ``_handle_session_chat_stream`` defines an ``_approval_notify`` closure that
  emits ``approval.request`` on the SSE stream, registers the run in
  ``_run_approval_sessions`` / ``_run_statuses`` / ``_active_run_tasks`` so
  ``POST /v1/runs/{run_id}/approval`` and ``/stop`` resolve, and cleans the
  registrations up when the turn ends.

Covers:
- _run_agent publishes / cleans up _active_run_agents[run_id]
- _run_agent registers / unregisters the approval callback
- the session _approval_notify closure redacts + shapes approval.request
- streaming session turn emits approval.request and populates the shared maps
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms.api_server import APIServerAdapter
from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _make_mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "ok",
        "messages": [],
        "api_calls": 1,
    }
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_id = "sess-1"
    return agent


# ---------------------------------------------------------------------------
# _run_agent run_id registration (stop support)
# ---------------------------------------------------------------------------

class TestRunAgentRunIdRegistration:
    """run_id makes the live agent reachable by POST /v1/runs/{id}/stop."""

    @pytest.mark.asyncio
    async def test_agent_registered_during_run_and_cleaned_up_after(self):
        adapter = _make_adapter()
        agent = _make_mock_agent()
        seen_during_run = {}

        def _during_run(*args, **kwargs):
            # While run_conversation executes, the agent must be reachable by
            # the stop handler via _active_run_agents[run_id].
            seen_during_run["agent"] = adapter._active_run_agents.get("run-xyz")
            return {"final_response": "ok", "messages": [], "api_calls": 1}

        agent.run_conversation.side_effect = _during_run

        with (
            patch.object(adapter, "_create_agent", return_value=agent),
            patch.object(adapter, "_bind_api_server_session", return_value=[]),
            patch("gateway.session_context.clear_session_vars"),
        ):
            await adapter._run_agent(
                user_message="hi",
                conversation_history=[],
                session_id="sess-1",
                run_id="run-xyz",
            )

        assert seen_during_run["agent"] is agent
        # Cleaned up once the turn ends so the map does not leak.
        assert "run-xyz" not in adapter._active_run_agents

    @pytest.mark.asyncio
    async def test_no_registration_without_run_id(self):
        adapter = _make_adapter()
        agent = _make_mock_agent()
        with (
            patch.object(adapter, "_create_agent", return_value=agent),
            patch.object(adapter, "_bind_api_server_session", return_value=[]),
            patch("gateway.session_context.clear_session_vars"),
        ):
            await adapter._run_agent(
                user_message="hi", conversation_history=[], session_id="sess-1",
            )
        assert adapter._active_run_agents == {}


# ---------------------------------------------------------------------------
# _run_agent approval callback registration
# ---------------------------------------------------------------------------

class TestRunAgentApprovalCallback:
    """approval_notify_callback is registered for the turn and torn down."""

    @pytest.mark.asyncio
    async def test_registers_and_unregisters_with_session_key(self):
        adapter = _make_adapter()
        registered, unregistered = {}, {}
        agent = _make_mock_agent()

        with (
            patch.object(adapter, "_create_agent", return_value=agent),
            patch.object(adapter, "_bind_api_server_session", return_value=[]),
            patch("gateway.session_context.clear_session_vars"),
            patch("tools.approval.register_gateway_notify",
                  side_effect=lambda k, cb: registered.update(key=k, cb=cb)),
            patch("tools.approval.unregister_gateway_notify",
                  side_effect=lambda k: unregistered.update(key=k)),
            patch("tools.approval.set_current_session_key", return_value=MagicMock()),
            patch("tools.approval.reset_current_session_key"),
        ):
            notify_cb = MagicMock()
            await adapter._run_agent(
                user_message="t",
                conversation_history=[],
                session_id="sess-1",
                gateway_session_key="gkey-1",
                approval_notify_callback=notify_cb,
            )

        # Prefers the gateway session key, then session_id, then task id.
        assert registered["key"] == "gkey-1"
        assert registered["cb"] is notify_cb
        assert unregistered["key"] == "gkey-1"

    @pytest.mark.asyncio
    async def test_no_registration_when_callback_is_none(self):
        adapter = _make_adapter()
        registered = {}
        agent = _make_mock_agent()
        with (
            patch.object(adapter, "_create_agent", return_value=agent),
            patch.object(adapter, "_bind_api_server_session", return_value=[]),
            patch("gateway.session_context.clear_session_vars"),
            patch("tools.approval.register_gateway_notify",
                  side_effect=lambda k, cb: registered.update(key=k)),
        ):
            await adapter._run_agent(
                user_message="t", conversation_history=[], session_id="sess-1",
            )
        assert registered == {}


# ---------------------------------------------------------------------------
# The session _approval_notify closure shape
# ---------------------------------------------------------------------------

class TestSessionApprovalNotifyClosure:
    """Reproduces the closure defined in _handle_session_chat_stream."""

    def _make_closure(self, enqueue, message_id="msg-1"):
        def _approval_notify(approval_data):
            event = dict(approval_data or {})
            if "command" in event:
                from gateway.run import _redact_approval_command
                event["command"] = _redact_approval_command(event.get("command"))
            event.update({
                "message_id": message_id,
                "choices": ["once", "session", "always", "deny"],
            })
            enqueue("approval.request", event)
        return _approval_notify

    def test_emits_approval_request_with_choices(self):
        captured = []
        cb = self._make_closure(lambda name, ev: captured.append((name, ev)))
        cb({"command": "rm -rf /tmp/x", "description": "delete dir"})
        assert len(captured) == 1
        name, event = captured[0]
        assert name == "approval.request"
        assert event["choices"] == ["once", "session", "always", "deny"]
        assert event["description"] == "delete dir"
        assert event["message_id"] == "msg-1"

    def test_routes_command_through_redaction(self):
        captured = []
        cb = self._make_closure(lambda name, ev: captured.append((name, ev)))
        with patch("gateway.run._redact_approval_command",
                   return_value="<redacted>") as red:
            cb({"command": "curl -H 'Authorization: Bearer sk-secret' x"})
        red.assert_called_once()
        assert captured[0][1]["command"] == "<redacted>"


# ---------------------------------------------------------------------------
# End-to-end streaming session turn
# ---------------------------------------------------------------------------

class TestSessionChatStreamControl:
    """A streaming session turn emits approval.request and registers the run
    in the shared maps the runs-API control endpoints read."""

    @pytest.mark.asyncio
    async def test_stream_emits_approval_and_registers_run(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        adapter = _make_adapter()
        app = web.Application()
        app.router.add_post(
            "/api/sessions/{session_id}/chat/stream",
            adapter._handle_session_chat_stream,
        )

        captured = {}

        async def _mock_run_agent(**kwargs):
            # The handler must hand us the approval notifier and the run_id,
            # and must have populated the shared maps the control endpoints
            # read *before* the turn runs.
            captured["has_notify"] = kwargs.get("approval_notify_callback") is not None
            captured["run_id"] = kwargs.get("run_id")
            captured["approval_sessions"] = dict(adapter._run_approval_sessions)
            captured["statuses"] = dict(adapter._run_statuses)
            captured["active_tasks"] = dict(adapter._active_run_tasks)
            cb = kwargs.get("approval_notify_callback")
            if cb:
                cb({"command": "echo hi", "description": "run echo"})
                await asyncio.sleep(0.05)
            return (
                {"final_response": "done", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent),
                patch.object(adapter, "_get_existing_session_or_404",
                             return_value=(object(), None)),
                patch.object(adapter, "_conversation_history_for_session",
                             return_value=[]),
                patch.object(adapter, "_turn_transcript_messages", return_value=[]),
            ):
                resp = await cli.post(
                    "/api/sessions/sess-1/chat/stream",
                    json={"message": "please run echo"},
                )
                assert resp.status == 200
                body = await resp.text()

        run_id = captured["run_id"]
        assert run_id and run_id.startswith("run_")
        # Wired into _run_agent.
        assert captured["has_notify"] is True
        # Approval + stop endpoints have what they need *during* the turn.
        assert captured["approval_sessions"].get(run_id) == "sess-1"
        assert run_id in captured["statuses"]
        assert run_id in captured["active_tasks"]
        # The approval event reached the SSE wire with resolvable choices.
        assert "event: approval.request" in body
        assert "choices" in body
        assert "run.started" in body and "run.completed" in body
        # Registrations are cleaned up after the turn (no leaks).
        await asyncio.sleep(0.05)
        assert run_id not in adapter._run_approval_sessions
        assert run_id not in adapter._active_run_tasks
        assert run_id not in adapter._active_run_agents
