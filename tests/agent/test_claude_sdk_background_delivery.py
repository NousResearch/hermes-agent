"""Background result delivery wiring — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""


import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from tests.agent.claude_sdk_fakes import (
    _make_turn,
    _make_agent,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestBackgroundDeliveryWiring:
    """Runtime glue: the session's delivery callback enqueues an
    sdk_background_result event for the gateway watcher's direct outbound
    send, config-gated."""

    def _spy_kwargs(self, monkeypatch):

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input, **kw):
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(
            "agent.transports.claude_agent_sdk_session.ClaudeAgentSdkSession",
            SpySession,
        )
        return captured

    def test_flag_on_wires_callback_and_queue_event(self, monkeypatch):
        import hermes_cli.config as cfg
        from tools.process_registry import process_registry

        captured = self._spy_kwargs(monkeypatch)
        events = []

        class _FakeQueue:
            def put(self, evt):
                events.append(evt)

        monkeypatch.setattr(process_registry, "completion_queue", _FakeQueue())
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: "gw-key-7"
        )
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        # Opt-in flag (upstream-conservative default is OFF).
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": True}}
            },
            raising=False,
        )

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent.session_id = "sess-bg-1"
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        callback = captured.get("on_unsolicited_result")
        assert callback is not None, "flag defaults ON — callback must be wired"
        callback(["Research landed — writing up.", "background answer text"])
        assert len(events) == 1
        evt = events[0]
        # Direct-outbound event: the payload burst rides UNJOINED (each text
        # becomes its own outbound message) and no model-facing directive is
        # prepended — on a direct send it would leak to the user.
        assert evt["type"] == "sdk_background_result"
        assert evt["payloads"] == [
            "Research landed — writing up.", "background answer text",
        ]
        assert not any("[USER IS WAITING" in p for p in evt["payloads"])
        assert evt["session_key"] == "gw-key-7"
        assert evt["parent_session_id"] == "sess-bg-1"
        assert "delegation_id" not in evt

    def test_bg_parent_resolved_at_delivery_time_after_rotation(
        self, monkeypatch,
    ):
        # P0.g: the SDK session outlives hermes session rotations. The old
        # code snapshotted parent_session_id/session_key at SDK-session
        # CREATION, so a completion firing after rotation carried the dead
        # parent — the gateway classified it permanently gone and dropped
        # it. The callback must resolve the parent AT DELIVERY TIME, with
        # the creation-time snapshot only as a fallback for the SDK-loop
        # thread where the session-key contextvar is unset.
        import hermes_cli.config as cfg
        from tools.process_registry import process_registry

        captured = self._spy_kwargs(monkeypatch)
        events = []

        class _FakeQueue:
            def put(self, evt):
                events.append(evt)

        monkeypatch.setattr(process_registry, "completion_queue", _FakeQueue())
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: "gw-key-7"
        )
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": True}}
            },
            raising=False,
        )

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent.session_id = "sess-before"
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        callback = captured.get("on_unsolicited_result")
        assert callback is not None

        # Hermes rotates the session between turns; the completion fires on
        # the SDK loop thread where the contextvar reads empty.
        agent.session_id = "sess-after-rotation"
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: ""
        )
        callback(["late background report"])
        assert len(events) == 1
        assert events[0]["parent_session_id"] == "sess-after-rotation"
        # Empty live key -> creation-time snapshot fallback keeps the route.
        assert events[0]["session_key"] == "gw-key-7"

        # A live, non-empty contextvar read wins over the snapshot.
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: "gw-key-LIVE"
        )
        callback(["second late report"])
        assert len(events) == 2
        assert events[1]["session_key"] == "gw-key-LIVE"
        assert events[1]["parent_session_id"] == "sess-after-rotation"

    def test_flag_off_leaves_callback_unwired(self, monkeypatch):
        import hermes_cli.config as cfg

        captured = self._spy_kwargs(monkeypatch)
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": False}}
            },
            raising=False,
        )
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert captured.get("on_unsolicited_result") is None
