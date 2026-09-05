"""Focused contracts for cache-preserving GPT-6 Astra effort transitions."""

import threading
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _preflight_codex_input_items,
)
from agent.turn_iteration_prep import _reset_astra_segment, _stage_astra_configuration_update
from agent.transports.codex import ResponsesApiTransport, is_astra_reasoning_cache_eligible


_DIRECT = "https://api.openai.com/v1"
_UPDATE_HIGH = {"type": "configuration_update", "reasoning": {"effort": "high"}}


def _user(text, update=None, base=None):
    msg = {"role": "user", "content": text}
    if update is not None:
        msg["_astra_configuration_update"] = update
    if base is not None:
        msg["_astra_reasoning_base_effort"] = base
    return msg


def test_configuration_update_is_inserted_before_marked_user_and_replayed_from_metadata():
    history = [
        _user("first", {"type": "configuration_update", "reasoning": {"effort": "low"}}),
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "second", "display_metadata": {"astra_configuration_update": _UPDATE_HIGH}},
    ]
    items = _chat_messages_to_responses_input(history, astra_configuration_updates=True)
    assert [item.get("type", item.get("role")) for item in items] == [
        "configuration_update", "user", "assistant", "configuration_update", "user"
    ]
    assert items[0] == {"type": "configuration_update", "reasoning": {"effort": "low"}}
    assert items[3] == _UPDATE_HIGH


def test_configuration_updates_never_leak_to_non_astra_conversion():
    history = [_user("first", _UPDATE_HIGH, base="low")]
    assert _chat_messages_to_responses_input(history) == [{"role": "user", "content": "first"}]


def test_latest_base_marker_starts_a_fresh_compatible_segment():
    history = [
        _user("old", _UPDATE_HIGH, base="low"),
        {"role": "assistant", "content": "done"},
        _user("fresh", base="medium"),
        {"role": "assistant", "content": "new"},
        _user("next", {"type": "configuration_update", "reasoning": {"effort": "xhigh"}}),
    ]
    items = _chat_messages_to_responses_input(history, astra_configuration_updates=True)
    assert _UPDATE_HIGH not in items
    assert items[-2:] == [
        {"type": "configuration_update", "reasoning": {"effort": "xhigh"}},
        {"role": "user", "content": "next"},
    ]


def test_configuration_update_preflight_is_strict_and_ladder_bounded():
    assert _preflight_codex_input_items([_UPDATE_HIGH]) == [_UPDATE_HIGH]
    with pytest.raises(ValueError):
        _preflight_codex_input_items([{"type": "configuration_update", "reasoning": {"effort": "ultra"}}])
    with pytest.raises(ValueError):
        _preflight_codex_input_items([{"type": "configuration_update", "reasoning": {"effort": "high"}, "id": "x"}])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": ""},
        {"base_url": "https://sub.api.openai.com/v1"},
        {"auth_mode": "oauth"},
        {"api_mode": "chat_completions"},
        {"model": "gpt-5.6"},
        {"is_subagent": True},
        {"platform": "subagent"},
        {"delegate_depth": 1},
        {"compression_checkpoint_required": True},
    ],
)
def test_astra_update_eligibility_excludes_incompatible_routes(kwargs):
    args = {
        "model": "gpt-6-astra", "base_url": _DIRECT, "api_mode": "codex_responses", "api_key": "sk-test",
        "auth_mode": "api_key", "provider": "openai",
    }
    args.update(kwargs)
    assert not is_astra_reasoning_cache_eligible(**args)


def test_astra_effort_transition_keeps_top_level_base_and_cache_key():
    transport = ResponsesApiTransport()
    state = {}
    common = {
        "base_url": _DIRECT, "api_mode": "codex_responses", "api_key": "sk-test", "auth_mode": "api_key",
        "provider": "openai", "session_id": "session-1", "astra_state": state,
    }
    first = transport.build_kwargs(
        model="gpt-6-astra", messages=[_user("hello")], reasoning_config={"effort": "low"}, **common
    )
    second = transport.build_kwargs(
        model="gpt-6-astra", messages=[_user("hello"), {"role": "assistant", "content": "ok"}, _user("next", _UPDATE_HIGH)],
        reasoning_config={"effort": "high"}, **common
    )
    assert first["reasoning"]["effort"] == "low"
    assert second["reasoning"]["effort"] == "low"
    assert second["input"][-2:] == [_UPDATE_HIGH, {"role": "user", "content": "next"}]
    assert state == {"base_effort": "low", "effective_effort": "high"}
    assert first["prompt_cache_key"] == second["prompt_cache_key"]

    third = transport.build_kwargs(
        model="gpt-6-astra",
        messages=[
            _user("hello"), {"role": "assistant", "content": "ok"},
            _user("next", _UPDATE_HIGH), {"role": "assistant", "content": "done"},
            _user("again", {"type": "configuration_update", "reasoning": {"effort": "low"}}),
        ],
        reasoning_config={"effort": "low"}, **common
    )
    assert third["reasoning"]["effort"] == "low"
    assert third["input"][-3:] == [
        {"role": "assistant", "content": "done"},
        {"type": "configuration_update", "reasoning": {"effort": "low"}},
        {"role": "user", "content": "again"},
    ]
    assert third["input"][-1] == {"role": "user", "content": "again"}
    assert state == {"base_effort": "low", "effective_effort": "low"}
    assert third["prompt_cache_key"] == first["prompt_cache_key"]


def _astra_agent(**overrides):
    values = {
        "model": "gpt-6-astra", "base_url": _DIRECT, "api_mode": "codex_responses",
        "api_key": "fixture-key", "auth_mode": "api_key", "provider": "openai",
        "is_subagent": False, "platform": "cli", "_delegate_depth": 0,
        "compression_checkpoint_required": False,
        "reasoning_config": {"effort": "low"}, "_astra_reasoning_state": {"base_effort": "low"},
        "_astra_base_effort": "low", "_astra_effective_effort": "high",
        "_astra_pending_configuration_update": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_compaction_reset_stages_only_for_an_ordinary_post_boundary_user():
    agent = _astra_agent()
    _reset_astra_segment(agent)
    assert agent._astra_reasoning_state == {}
    assert agent._astra_base_effort is None
    assert agent._astra_effective_effort is None
    assert agent._astra_pending_configuration_update == "high"

    nudge = {"role": "user", "content": "continue", "_length_continuation_nudge": True}
    _stage_astra_configuration_update(agent, [nudge])
    assert "_astra_configuration_update" not in nudge
    assert agent._astra_pending_configuration_update == "high"

    ordinary = {"role": "user", "content": "new turn"}
    _stage_astra_configuration_update(agent, [ordinary])
    assert ordinary["_astra_configuration_update"] == {
        "type": "configuration_update", "reasoning": {"effort": "high"}
    }
    assert ordinary["_astra_reasoning_base_effort"] == "low"
    assert agent._astra_pending_configuration_update is None


def test_requested_effort_change_is_derived_from_history_after_resume():
    agent = _astra_agent(
        reasoning_config={"effort": "high"}, _astra_reasoning_state={},
        _astra_base_effort=None, _astra_effective_effort=None,
    )
    messages = [
        _user("first", base="low"),
        {"role": "assistant", "content": "ok"},
        _user("next"),
    ]
    _stage_astra_configuration_update(agent, messages)
    assert messages[-1]["_astra_configuration_update"] == _UPDATE_HIGH
    assert agent._astra_base_effort == "low"
    assert agent._astra_effective_effort == "high"


def test_initial_and_noop_effort_record_base_without_an_update():
    agent = _astra_agent(
        reasoning_config={"effort": "medium"}, _astra_reasoning_state={},
        _astra_base_effort=None, _astra_effective_effort=None,
        _astra_pending_configuration_update=None,
    )
    messages = [_user("first")]
    _stage_astra_configuration_update(agent, messages)
    assert messages[-1]["_astra_reasoning_base_effort"] == "medium"
    assert "_astra_configuration_update" not in messages[-1]


def test_message_sidecar_round_trip_preserves_base_update_and_existing_metadata(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session(session_id="astra-sidecar", source="cli", model="gpt-6-astra")
        db.append_message(
            "astra-sidecar", "user", "next", display_metadata={"existing": "kept"},
        )
        assert db.set_latest_user_display_metadata(
            "astra-sidecar", "next", {
                "astra_reasoning_base_effort": "low",
                "astra_configuration_update": _UPDATE_HIGH,
            },
        ) == 1
        restored = db.get_messages("astra-sidecar")[-1]
        assert restored["display_metadata"] == {
            "existing": "kept",
            "astra_reasoning_base_effort": "low",
            "astra_configuration_update": _UPDATE_HIGH,
        }
        assert _chat_messages_to_responses_input(
            [restored], astra_configuration_updates=True,
        ) == [_UPDATE_HIGH, {"role": "user", "content": "next"}]
    finally:
        db.close()


def test_non_astra_boundary_reset_drops_astra_pending_state():
    agent = _astra_agent(model="gpt-5.6", _astra_pending_configuration_update="high")
    _reset_astra_segment(agent)
    assert agent._astra_pending_configuration_update is None


def test_effort_only_gateway_change_keeps_cached_agent_and_arms_marker():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    session_key = "fixture-session"
    agent = _astra_agent(_astra_effective_effort="low")
    state = runner._session_state(session_key)
    state.conversation.base_effort = "low"
    state.conversation.effective_effort = "low"
    state.conversation.reasoning_change_requested = True
    runner._resolve_session_reasoning_config = lambda **_kwargs: {"effort": "high"}
    runner._agent_cache[session_key] = (agent, "fixture-signature")

    runner._evict_cached_agent(session_key)

    assert runner._agent_cache[session_key][0] is agent
    assert agent._astra_pending_configuration_update == "high"
    assert state.conversation.pending_configuration_update == "high"
    assert state.conversation.reasoning_change_requested is False


def test_reasoning_change_reason_does_not_leak_when_no_agent_is_cached():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    state = runner._session_state("empty-session")
    state.conversation.reasoning_change_requested = True

    runner._evict_cached_agent("empty-session")

    assert state.conversation.reasoning_change_requested is False


def test_gateway_replay_binds_update_to_exact_duplicate_user_row():
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "same"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "same"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "same", "display_metadata": {
            "astra_reasoning_base_effort": "low",
            "astra_configuration_update": _UPDATE_HIGH,
        }},
    ]

    replay, _ = _build_gateway_agent_history(history)

    users = [message for message in replay if message["role"] == "user"]
    assert "_astra_configuration_update" not in users[0]
    assert "_astra_configuration_update" not in users[1]
    assert users[2]["_astra_reasoning_base_effort"] == "low"
    assert users[2]["_astra_configuration_update"] == _UPDATE_HIGH


def test_manual_compaction_reset_survives_cached_agent_eviction():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    session_key = "manual-compress"
    agent = _astra_agent(
        _astra_base_effort="low", _astra_effective_effort="low",
        _astra_pending_configuration_update="high",
    )
    runner._agent_cache[session_key] = (agent, "fixture-signature")

    runner._arm_astra_segment_reset(session_key)

    conversation = runner._session_state(session_key).conversation
    assert conversation.astra_force_new_segment is True
    assert conversation.base_effort == "low"
    assert conversation.effective_effort == "high"
    assert conversation.pending_configuration_update == "high"
