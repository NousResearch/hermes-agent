"""Native Gateway lifecycle → Buzz edge-translation tests."""

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, TurnRunner
from gateway.session import SessionSource
from gateway.turn_observer import GatewayTurnObserver
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter
_nostr_auth = _buzz_mod._load_nostr_auth()


class _CaptureRoute:
    def __init__(self, events):
        self.events = events

    def on_turn_lifecycle(self, event):
        self.events.append(event)
        return True


def _observer(loop, *, is_new_session=True, events=None):
    return GatewayTurnObserver(
        platform="buzz",
        profile="default",
        channel_id="channel-1",
        session_id="session-1",
        triggering_event_id="a" * 64,
        is_new_session=is_new_session,
        route=_CaptureRoute(events) if events is not None else object(),
        loop=loop,
        is_current=lambda: True,
    )


def _capture_lifecycle():
    return []


@pytest.mark.asyncio
async def test_generic_lifecycle_preserves_session_novelty_and_terminal_order(
    monkeypatch,
):
    events = _capture_lifecycle()
    observer = _observer(
        asyncio.get_running_loop(), is_new_session=True, events=events
    )

    assert observer.start(liveness_interval=0) is True
    assert observer.session_resolved() is True
    assert observer.finish({"final_response": "done"}) is True
    assert observer.finish({"failed": True}) is False

    assert [event.phase for event in events] == [
        "turn_started",
        "session_resolved",
        "turn_finished",
    ]
    assert events[1].is_new_session is True
    assert events[2].outcome == "success"
    assert len({event.turn_id for event in events}) == 1


@pytest.mark.asyncio
async def test_liveness_starts_with_turn_and_stops_at_terminal(monkeypatch):
    events = _capture_lifecycle()
    observer = _observer(asyncio.get_running_loop(), events=events)

    observer.start(liveness_interval=0.005)
    await asyncio.sleep(0.012)
    observer.finish({"final_response": "done"})
    count_after_finish = len(events)
    await asyncio.sleep(0.012)

    assert any(event.phase == "turn_liveness" for event in events)
    assert len(events) == count_after_finish
    assert events[-1].phase == "turn_finished"


@pytest.mark.asyncio
async def test_tool_events_are_allowlisted_and_omit_sensitive_content(monkeypatch):
    events = _capture_lifecycle()
    observer = _observer(asyncio.get_running_loop(), events=events)
    observer.start(liveness_interval=0)

    secret_command = "curl -H 'Authorization: Bearer token' /private/path"
    secret_result = json.dumps({"exit_code": 1, "output": "private-error"})
    observer.tool_started("call-1", "terminal", {"command": secret_command})
    observer.tool_finished(
        "call-1", "terminal", {"command": secret_command}, secret_result
    )

    tool_events = [event for event in events if event.phase.startswith("tool_")]
    assert [event.tool_status for event in tool_events] == ["executing", "failed"]
    assert [event.tool_call_id for event in tool_events] == ["tool-1", "tool-1"]
    serialized = json.dumps([asdict(event) for event in tool_events])
    assert secret_command not in serialized
    assert secret_result not in serialized
    assert "private-error" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_proxy_gateway_turn_emits_terminal_without_masking_error(fail):
    events = _capture_lifecycle()
    route = _CaptureRoute(events)
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._get_proxy_url = lambda: "http://proxy.invalid"
    runner._adapter_for_source = lambda _source: route
    runner._is_session_run_current = lambda *_args: True

    async def run_proxy(**_kwargs):
        if fail:
            raise RuntimeError("proxy root cause")
        return {"final_response": "done", "completed": True}

    runner._run_agent_via_proxy = run_proxy
    source = SessionSource(
        platform=Platform("buzz"),
        user_id="owner",
        chat_id="channel-1",
        user_name="owner",
        chat_type="channel",
    )

    if fail:
        with pytest.raises(RuntimeError, match="proxy root cause"):
            await runner._run_agent(
                "hello", "", [], source, "session-1", "buzz:channel-1"
            )
    else:
        result = await runner._run_agent(
            "hello", "", [], source, "session-1", "buzz:channel-1"
        )
        assert result["final_response"] == "done"

    assert [event.phase for event in events] == [
        "turn_started",
        "session_resolved",
        "turn_finished",
    ]
    assert events[-1].outcome == ("failed" if fail else "success")


@pytest.mark.asyncio
async def test_gateway_wrapper_reports_early_inner_setup_failure():
    events = _capture_lifecycle()
    route = _CaptureRoute(events)
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._adapter_for_source = lambda _source: route
    runner._is_session_run_current = lambda *_args: True

    async def fail_during_setup(*_args, **_kwargs):
        raise RuntimeError("early setup root cause")

    runner._run_agent_inner = fail_during_setup
    source = SessionSource(
        platform=Platform("buzz"),
        user_id="owner",
        chat_id="channel-1",
        user_name="owner",
        chat_type="channel",
    )

    with pytest.raises(RuntimeError, match="early setup root cause"):
        await runner._run_agent("hello", "", [], source, "session-1")

    assert [event.phase for event in events] == [
        "turn_started",
        "session_resolved",
        "turn_finished",
    ]
    assert events[-1].outcome == "failed"


@pytest.mark.asyncio
async def test_post_execution_cancellation_does_not_rewrite_latched_success():
    events = _capture_lifecycle()
    route = _CaptureRoute(events)
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._adapter_for_source = lambda _source: route
    runner._is_session_run_current = lambda *_args: True

    async def cancel_after_execution(*_args, **kwargs):
        kwargs["turn_observer"].finish(
            {"final_response": "done", "completed": True}
        )
        raise asyncio.CancelledError

    runner._run_agent_inner = cancel_after_execution
    source = SessionSource(
        platform=Platform("buzz"),
        user_id="owner",
        chat_id="channel-1",
        user_name="owner",
        chat_type="channel",
    )

    with pytest.raises(asyncio.CancelledError):
        await runner._run_agent("hello", "", [], source, "session-1")

    terminal = [event for event in events if event.phase == "turn_finished"]
    assert len(terminal) == 1
    assert terminal[0].outcome == "success"


@pytest.mark.parametrize(
    ("result", "exception_type", "outcome"),
    [
        ({"final_response": "done"}, None, "success"),
        ({"failed": True}, None, "failed"),
        ({"interrupted": True}, None, "cancelled"),
        ({"final_response": "done"}, RuntimeError, "failed"),
        (None, None, "failed"),
    ],
)
@pytest.mark.asyncio
async def test_terminal_outcome_matrix(monkeypatch, result, exception_type, outcome):
    events = _capture_lifecycle()
    observer = _observer(asyncio.get_running_loop(), events=events)
    observer.start(liveness_interval=0)
    observer.finish(result, exception_type=exception_type)

    terminal = [event for event in events if event.phase == "turn_finished"]
    assert len(terminal) == 1
    assert terminal[0].outcome == outcome


@pytest.mark.asyncio
async def test_inactivity_timeout_failed_result_reports_timed_out(monkeypatch):
    events = _capture_lifecycle()
    observer = _observer(asyncio.get_running_loop(), events=events)
    observer.start(liveness_interval=0)

    observer.finish({"failed": True}, timed_out=True)

    terminal = [event for event in events if event.phase == "turn_finished"]
    assert [event.outcome for event in terminal] == ["timed_out"]


@pytest.mark.asyncio
async def test_propagated_task_cancellation_reports_cancelled(monkeypatch):
    events = _capture_lifecycle()
    observer = _observer(asyncio.get_running_loop(), events=events)
    observer.start(liveness_interval=0)

    observer.finish(None, exception_type=asyncio.CancelledError)

    terminal = [event for event in events if event.phase == "turn_finished"]
    assert [event.outcome for event in terminal] == ["cancelled"]


@pytest.mark.asyncio
async def test_no_listener_keeps_observer_and_liveness_disabled(monkeypatch):
    observer = _observer(asyncio.get_running_loop())

    assert observer.start(liveness_interval=0.001) is False
    await asyncio.sleep(0.005)
    assert observer.finish({"final_response": "done"}) is False


def _buzz_adapter():
    from gateway.config import PlatformConfig

    owner = _nostr_auth.public_key_hex("00" * 31 + "02")
    return BuzzAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "relay_url": "https://test.relay",
                "activity_owner_pubkey": owner,
            },
        )
    )


@pytest.mark.parametrize(
    ("phase", "fields", "expected_kind"),
    [
        ("turn_started", {"triggering_event_id": "a" * 64}, "turn_started"),
        ("session_resolved", {"is_new_session": True}, "session_resolved"),
        ("turn_liveness", {}, "turn_liveness"),
        (
            "tool_started",
            {
                "tool_call_id": "call-1",
                "tool_name": "terminal",
                "tool_status": "executing",
            },
            "acp_read",
        ),
        (
            "tool_finished",
            {
                "tool_call_id": "call-1",
                "tool_name": "terminal",
                "tool_status": "failed",
            },
            "acp_read",
        ),
        ("turn_finished", {"outcome": "success"}, "turn_completed"),
        ("turn_finished", {"outcome": "failed"}, "turn_error"),
        ("turn_finished", {"outcome": "cancelled"}, "turn_error"),
        ("turn_finished", {"outcome": "timed_out"}, "turn_error"),
    ],
)
def test_buzz_edge_translates_neutral_lifecycle_only(
    monkeypatch, phase, fields, expected_kind
):
    adapter = _buzz_adapter()
    captured = []
    monkeypatch.setattr(
        adapter,
        "_enqueue_activity",
        lambda kind, **kwargs: captured.append((kind, kwargs)) or True,
    )
    event_fields = {
        "phase": phase,
        "channel_id": "channel-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "started_at": "2026-08-04T00:00:00.000Z",
        "triggering_event_id": None,
        "is_new_session": None,
        "tool_call_id": None,
        "tool_name": None,
        "tool_status": None,
        "outcome": None,
    }
    event_fields.update(fields)
    event = SimpleNamespace(**event_fields)

    assert _buzz_mod._handle_gateway_turn_lifecycle(event=event, route=adapter)
    assert captured[0][0] == expected_kind
    serialized = json.dumps(captured[0][1])
    assert "Authorization" not in serialized
    assert "/private/" not in serialized

    if phase == "session_resolved":
        assert captured[0][1]["payload"]["isNewSession"] is True
    if phase.startswith("tool_"):
        update = captured[0][1]["payload"]["params"]["update"]
        assert update["rawInput"] == {}
        assert update["toolCallId"] == "call-1"
    if phase == "turn_finished" and fields["outcome"] != "success":
        assert captured[0][1]["payload"]["status"] == fields["outcome"]


def test_buzz_edge_ignores_non_buzz_routes():
    event = SimpleNamespace(phase="turn_started")
    assert not _buzz_mod._handle_gateway_turn_lifecycle(
        event=event, route=SimpleNamespace()
    )


@pytest.mark.asyncio
async def test_structured_callbacks_wire_for_active_observer_and_preserve_existing(
    monkeypatch,
):
    calls = []
    events = _capture_lifecycle()
    observer = _observer(asyncio.get_running_loop(), events=events)
    assert observer.active is False

    context = SimpleNamespace(_voice_ack_guild=[None])
    runner = TurnRunner(SimpleNamespace(), context, observer)

    def prior_start(*_args):
        calls.append(("prior-start", ()))
        raise RuntimeError("must not suppress observer")

    def prior_finish(*_args):
        calls.append(("prior-finish", ()))

    agent = SimpleNamespace(
        tool_start_callback=prior_start,
        tool_complete_callback=prior_finish,
    )
    observer.start(liveness_interval=0)
    runner.wire_structured_tool_callbacks(agent)

    assert callable(agent.tool_start_callback)
    assert callable(agent.tool_complete_callback)
    agent.tool_start_callback("call-1", "terminal", {"secret": "value"})
    agent.tool_complete_callback("call-1", "terminal", {}, "result")

    assert [name for name, _ in calls] == ["prior-start", "prior-finish"]
    assert [event.phase for event in events] == [
        "turn_started",
        "tool_started",
        "tool_finished",
    ]


def test_inactive_observer_does_not_wrap_existing_tool_callbacks():
    loop = asyncio.new_event_loop()
    try:
        observer = _observer(loop, events=[])
        context = SimpleNamespace(_voice_ack_guild=[None])
        runner = TurnRunner(SimpleNamespace(), context, observer)
        prior_start = lambda *_args: None
        prior_finish = lambda *_args: None
        agent = SimpleNamespace(
            tool_start_callback=prior_start,
            tool_complete_callback=prior_finish,
        )

        runner.wire_structured_tool_callbacks(agent)

        assert agent.tool_start_callback is prior_start
        assert agent.tool_complete_callback is prior_finish
    finally:
        loop.close()


def test_gateway_core_has_no_platform_activity_transport_coupling():
    """Core emits neutral lifecycle events but never knows Buzz wire semantics."""

    import inspect
    import gateway.run as gateway_run
    import gateway.turn_context as turn_context

    source = inspect.getsource(gateway_run.TurnRunner) + inspect.getsource(
        turn_context.TurnContext
    )
    for forbidden in (
        "publish_activity",
        "activity_owner_pubkey",
        "acp_read",
        "session/update",
        "Nostr",
        "Buzz activity",
    ):
        assert forbidden not in source
