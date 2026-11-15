"""Per-request ProviderProfile hooks on the request-build hot path: bounded calls, one failure policy.

A host-claiming plugin profile is asked about every request (under ``plugins.isolation: host`` each
ask is an IPC round trip). Its answers must not cost O(history) per request, and a hook that raises
must leave the session on the generic answer from then on, so request bytes never flip between turns.
"""

import copy
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

HOST = "https://api.hookpolicy.test/v1"
CARRIER = "hermes.native_assistant"


@pytest.fixture
def make_profile(request, monkeypatch):
    import providers
    from agent import provider_route_hooks
    from providers.base import ProviderProfile

    monkeypatch.setattr(provider_route_hooks, "_WARNED", set(), raising=False)

    def make(**hooks):
        cls = type("HookPolicyProfile", (ProviderProfile,), hooks)
        profile = cls(name="hookpolicy", base_url=HOST, hostnames=("hookpolicy.test",),
                      message_carrier_opaque="hp_opaque", message_carrier_text="hp_text")
        providers.list_providers()  # discovery first, so the fake survives it
        providers.register_provider(profile)
        return profile

    def _cleanup():
        providers._REGISTRY.pop("hookpolicy", None)
        providers._SOURCES.pop("hookpolicy", None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    return make


def _agent():
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        return AIAgent(api_key="k", base_url=HOST, provider="custom", api_mode="chat_completions", model="m1",
                       quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
                       reasoning_config={"enabled": True, "effort": "high"})


def _history(turns, *, text_every=0, text_at=()):
    hist = [{"role": "system", "content": "s"}]
    for i in range(turns):
        record = {"type": CARRIER, "hp_opaque": f"OPQ{i}"}
        if (text_every and i % text_every == 0) or i in text_at:
            record["hp_text"] = f"T{i}"
        hist += [{"role": "user", "content": f"u{i}"},
                 {"role": "assistant", "content": f"a{i}", "reasoning_details": [record]}]
    return hist + [{"role": "user", "content": "next"}]


def _wire(agent, hist):
    kwargs = agent._build_api_kwargs(copy.deepcopy(hist))
    return kwargs["messages"], kwargs.get("extra_body", {}).get("reasoning")


def _replay(self, values, *, model, base_url):
    return {k: values[k] for k in ("hp_opaque", "hp_text") if values.get(k)}


def test_a_failed_hook_warns_per_session_and_is_asked_again_after_a_reset(make_profile, caplog):
    from agent.provider_route_hooks import ask_profile

    calls = []

    def flaky(self, *, base_url, model="", user_initiated=False):
        calls.append(model)
        raise RuntimeError("plugin bug")

    profile = make_profile(request_headers=flaky)
    first, second = _agent(), _agent()
    with caplog.at_level(logging.WARNING, logger="agent.provider_route_hooks"):
        for agent in (first, first, second):
            assert ask_profile(profile, "request_headers", {}, base_url=HOST, model="m", scope=agent) == {}
        first.reset_session_state()
        assert ask_profile(profile, "request_headers", {}, base_url=HOST, model="m", scope=first) == {}
    assert len(calls) == 3  # first once (sticky), second once, first again after /new
    warned = [r for r in caplog.records if "request_headers hook failed" in r.getMessage()]
    assert len(warned) == 3  # one per session, not one per process


def test_long_history_asks_the_replay_hook_once_per_carrier_key_set(make_profile):
    calls = []

    def replay_message_carriers(self, values, *, model, base_url):
        calls.append(tuple(sorted(values)))
        return _replay(self, values, model=model, base_url=base_url)

    make_profile(replay_message_carriers=replay_message_carriers)
    agent = _agent()
    hist = _history(50, text_every=5)
    messages, _ = _wire(agent, hist)
    assert len(calls) == 2  # two key sets ({opaque}, {opaque, text}), not 50 messages
    replayed = [(m.get("hp_opaque"), m.get("hp_text")) for m in messages if m["role"] == "assistant"]
    assert replayed == [(f"OPQ{i}", f"T{i}" if i % 5 == 0 else None) for i in range(50)]
    _wire(agent, hist)
    assert len(calls) == 2  # the next request asks nothing new


def test_failing_reasoning_hook_keeps_the_session_on_the_generic_answer(make_profile, caplog):
    state = {"fail": True, "calls": 0}

    def reasoning_payload(self, model, reasoning_config, *, base_url):
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("plugin host went away")
        return {"effort": "high"}

    make_profile(reasoning_payload=reasoning_payload, replay_message_carriers=_replay)
    agent = _agent()
    hist = _history(3)
    def request():  # the wire plus the two reasoning gates built from this hook
        return _wire(agent, hist), agent._supports_reasoning_extra_body(), agent._endpoint_reasoning_payload()

    with caplog.at_level(logging.WARNING, logger="agent.provider_route_hooks"):
        first = request()
        state["fail"] = False  # the hook recovers; the session must not flip back mid-conversation
        later = [request() for _ in range(2)]
    assert first[1:] == (False, None)
    assert first[0][0][2]["hp_opaque"] == "OPQ0"  # the other hooks of the profile keep working
    assert all(json.dumps(r, sort_keys=True) == json.dumps(first, sort_keys=True) for r in later)
    assert state["calls"] == 1
    assert len([r for r in caplog.records if "reasoning_payload" in r.getMessage()]) == 1
    # A new session asks again (the failure is the agent's, not the process's).
    assert _agent()._endpoint_reasoning_payload() == {"effort": "high"}


def test_replay_hook_failing_on_a_later_call_never_flips_bytes_after_that_turn(make_profile, caplog):
    calls = []

    def replay_message_carriers(self, values, *, model, base_url):
        calls.append(1)
        if len(calls) >= 2:
            raise RuntimeError("plugin bug")
        return _replay(self, values, model=model, base_url=base_url)

    make_profile(replay_message_carriers=replay_message_carriers)
    agent = _agent()
    hist = _history(4)
    first, _ = _wire(agent, hist)
    assert first[2]["hp_opaque"] == "OPQ0"
    hist = _history(4, text_at=(3,))  # a new key set late in history: the hook is asked again and fails
    with caplog.at_level(logging.WARNING, logger="agent.provider_route_hooks"):
        second, _ = _wire(agent, hist)
        third, _ = _wire(agent, hist)
    assert second == third
    assert not any("hp_opaque" in m or "hp_text" in m for m in second)  # generic: carriers stripped
    assert len([r for r in caplog.records if "replay_message_carriers" in r.getMessage()]) == 1
