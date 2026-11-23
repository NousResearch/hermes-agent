"""MoA advisors and the aggregator clamp the agent's ``auto`` cache tier per destination exactly
like the main loop: ``auto`` on a Bedrock route resolves to 5m, an explicit ``1h`` pin keeps 1h.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import moa_loop

_BEDROCK = {"provider": "bedrock", "model": "anthropic.claude-sonnet-4", "base_url": "",
            "api_mode": "anthropic_messages"}


def _ttls(obj) -> set:
    found = set()
    if isinstance(obj, dict):
        if isinstance(obj.get("cache_control"), dict):
            found.add(obj["cache_control"].get("ttl", "5m"))
        for value in obj.values():
            found |= _ttls(value)
    elif isinstance(obj, list):
        for value in obj:
            found |= _ttls(value)
    return found


def _agent(auto: bool):
    return SimpleNamespace(_cache_disabled=False, _cache_ttl="1h", _cache_ttl_auto=auto,
                           _cached_system_prompt_static=None, _interrupt_requested=False)


@pytest.mark.parametrize(("auto", "expected"), [(True, {"5m"}), (False, {"1h"})])
def test_moa_bedrock_advisor_and_aggregator_mirror_main_path_ttl(monkeypatch, auto, expected):
    from agent.prompt_caching import effective_cache_ttl

    # The main loop's answer for the same agent state is the contract.
    assert {effective_cache_ttl("1h", provider="bedrock", model=_BEDROCK["model"], auto=auto)} == expected
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: dict(_BEDROCK))
    monkeypatch.setattr(moa_loop, "_trim_messages_for_reference", lambda messages, *a, **k: messages)
    monkeypatch.setattr(moa_loop, "_slot_request_headers", lambda runtime, agent: None)
    sent = []
    monkeypatch.setattr(moa_loop, "call_llm", lambda **kw: sent.append(kw["messages"]) or SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="advice", tool_calls=[]))], usage=None))
    monkeypatch.setattr(moa_loop, "_price_reference_response", lambda *a, **k: (None, None, None, None))
    agent = _agent(auto)
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
               {"role": "user", "content": "q2"}]

    moa_loop._run_references_parallel([{"provider": "bedrock", "model": _BEDROCK["model"]}], history, agent=agent)
    assert sent and _ttls(sent[0]) == expected

    facade = moa_loop.MoAChatCompletions("default", agent=agent)
    planned, _ = facade._plan_aggregator_cache(
        [{"role": "system", "content": "sys"}, *history], None, None, dict(_BEDROCK))
    assert _ttls(planned) == expected
