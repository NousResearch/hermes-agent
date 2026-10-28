"""A mid-turn route change republishes ``session.info`` to a local viewer (#78613).

The TUI and Desktop status bars read ``session.info.model``, which the gateway otherwise publishes
only at turn end. ``local_viewer_events.wire`` installs ``route_change_callback`` for a local
viewer's turn and ``restore`` puts the previous value back on the shared cached agent. A
messaging route never wires the sinks, so its agent keeps no callback and publishes nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.route_binding import notify_route_change
from gateway import local_viewer_events
from gateway.config import Platform
from gateway.run_turn_progress import local_viewer_turn
from gateway.run_turn_runner import TurnRunner


def _turn(platform, published, agent):
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=platform, chat_id='chat'),
                                mute_notification_reply=False, session_id='s', inbound_message_id=None)
    turn._runner = SimpleNamespace(config=None)
    authority = SimpleNamespace(sessions={'s': SimpleNamespace(tool_progress_mode=None, show_reasoning=False)},
                                profile_id='default', agent=lambda ref: agent)
    turn._approval_owner = (authority, 's', 1)
    turn._publish_execution = lambda kind, payload: published.append((kind, payload)) or True
    return turn


def _agent():
    return SimpleNamespace(model='claude-opus-5-5', provider='anthropic', route_change_callback=None,
                           tool_gen_callback=None, _print_fn=None, tools=[], _cached_system_prompt='')


@pytest.fixture(autouse=True)
def _session_info_reads_the_cached_agent(monkeypatch):
    # local_session_info's model is the cached agent's ``model``; the policy fields are not under test.
    monkeypatch.setattr('gateway.session_local.local_session_info',
                        lambda authority, ref: {'model': authority.agent(ref).model, 'lazy': False})
    monkeypatch.setattr(local_viewer_events, '_config_default_launch', lambda: False)


def _swap_to_fallback(agent):
    agent.model, agent.provider = 'global.anthropic.claude-opus-5-5', 'bedrock'
    return notify_route_change(agent)


def test_a_local_viewer_route_change_publishes_one_session_info_on_the_new_route():
    published, agent = [], _agent()
    previous_callback = object()
    agent.route_change_callback = previous_callback
    turn = _turn(Platform.LOCAL, published, agent)
    assert local_viewer_turn(turn)
    previous = local_viewer_events.wire(turn, agent)
    assert _swap_to_fallback(agent) is True
    infos = [payload for kind, payload in published if kind == 'session.info']
    assert len(infos) == 1
    assert infos[0]['model'] == 'global.anthropic.claude-opus-5-5' and infos[0]['provider'] == 'bedrock'
    local_viewer_events.restore(agent, previous)
    assert agent.route_change_callback is previous_callback


def test_a_messaging_turn_installs_no_callback_and_publishes_nothing():
    published, agent = [], _agent()
    turn = _turn(Platform.TELEGRAM, published, agent)
    # TurnRunner wires the local-viewer sinks only behind this gate.
    assert not local_viewer_turn(turn)
    assert agent.route_change_callback is None
    assert _swap_to_fallback(agent) is True
    assert published == []


def test_a_messaging_turns_wiring_installs_no_route_change_callback():
    # Runs the real per-turn wiring for a Telegram turn, with the same minimal fakes as
    # tests/gateway/test_display_null_turn_wiring.py::_wire, so a callback installed outside the
    # local-viewer gate fails here.
    ctx = SimpleNamespace(
        progress_callback=None, native_tool_start_callback=None, voice_ack_callback=None,
        _voice_ack_guild=[None], voice_turn=False, _native_slack_task_cards=False,
        native_tool_complete_callback=None, _step_callback_sync=None,
        _hooks_ref=SimpleNamespace(loaded_hooks=[]), _status_callback_sync=None,
        _event_callback_sync=None, _status_adapter=None, session_key="", user_config={},
        source=SimpleNamespace(platform="telegram"), mute_notification_reply=False,
        _thinking_enabled=False, agent_holder=[None], tools_holder=[None], process_task_id=None,
        process_baseline=None, run_generation=0,
    )
    holder = SimpleNamespace(
        _ctx=ctx, _approval_owner=None,
        _runner=SimpleNamespace(_service_tier=None, _consume_pending_turn_sidecar_notes=lambda key: []),
        _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
        _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
        _clarify_callback_sync=lambda *a, **k: None, _notice_callback_sync=lambda *a, **k: None,
        _attach_session_title_callback=lambda agent, ctx: None,
    )
    agent = SimpleNamespace(route_change_callback=None)
    TurnRunner._wire_turn_agent_callbacks(holder, agent, {}, None, None, None, False)
    assert agent.route_change_callback is None


def test_a_route_change_in_a_one_turn_admission_evicts_nothing_and_names_the_new_route():
    # ``/model X --once`` (a model_override admission) that falls back mid-turn: the mid-turn publish
    # must not run publish_settled_info's one-turn eviction or its policy-route model.
    from gateway.session_cli_turn import cli_turn_scope
    published, agent, evicted = [], _agent(), []
    turn = _turn(Platform.LOCAL, published, agent)
    turn._ctx.inbound_message_id = 'm1'
    turn._runner = SimpleNamespace(config=None, _evict_cached_agent=lambda route, **kw: evicted.append(route))
    live = turn._approval_owner[0].sessions['s']
    live.route, live.source = 'route-s', turn._ctx.source
    previous = local_viewer_events.wire(turn, agent)
    with cli_turn_scope('m1', {'model_override': 'gpt-4o'}):
        assert _swap_to_fallback(agent) is True
    local_viewer_events.restore(agent, previous)
    assert evicted == []
    assert [p['model'] for kind, p in published if kind == 'session.info'] == ['global.anthropic.claude-opus-5-5']
