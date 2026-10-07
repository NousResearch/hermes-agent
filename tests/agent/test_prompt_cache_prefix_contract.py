"""Prompt-cache prefix contract on caching routes, through a real AIAgent and a recording fake provider.

Provider prompt caches key on the request content up to each ``cache_control`` breakpoint, so every
request in a session must re-send the previous request's content unchanged and only append; a
request that rewrites earlier content re-bills the whole prefix. Marker placement is NOT part of the
key: the rolling breakpoint window moves every request, and a single text part with a marker is
the same content as the plain string it decorates (production telemetry: plain sessions reuse the
full previous prefix on 99.5% of consecutive calls). Splitting one string into several parts is a
content change.

The cells mirror what real sessions do on the two cache layouts Hermes emits (OpenAI-wire envelope
markers: OpenRouter / Nous Portal / custom relays; native Anthropic Messages content-block markers)
and were chosen from per-call cache telemetry (``agent.log`` ``cache=R/T write=W``): consecutive
calls in one session reuse the whole previous prefix except where these cells say otherwise.

Regression pinned here (#133715): a skill turn's first user message must go out as the same
[scaffold, tail] parts on every request, marked or not. It used to be split only while it sat inside
the last-N marker window (#81867), then re-sent as a plain string once the tool round pushed it out,
so request 3 re-wrote the whole skill body — the ~20k-token re-write seen on call 2 of skill-invoked
sessions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG, canon, prefix_breaks
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

SKILL_BODY = "Follow these steps carefully. " * 300


def _cache_content(value: Any) -> Any:
    """What the cache keys on: markers dropped, a lone text part folded back to its string."""
    if isinstance(value, list):
        value = [_cache_content(v) for v in value]
        if len(value) == 1 and isinstance(value[0], dict) and set(value[0]) == {"type", "text"}:
            return value[0]["text"]
        return value
    if isinstance(value, dict):
        return {k: _cache_content(v) for k, v in value.items() if k != "cache_control"}
    return value


def _prefix_breaks(requests: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """``prefix_breaks`` over what the cache keys on, after asserting markers reached the wire."""
    assert any("cache_control" in canon(r) for r in requests), "caching route sent no cache_control markers"
    return prefix_breaks([_cache_content(r) for r in requests])


@pytest.fixture
def caching_route():
    """A custom OpenAI-wire relay declared ``prompt_caching: true`` -> envelope cache layout."""
    home = Path(os.environ["HERMES_HOME"])
    skill = home / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: demo\ndescription: Demo skill.\n---\n\n# Demo\n\n{SKILL_BODY}\n")
    srv = FakeLLMServer([]).__enter__()
    route = (f"custom_providers:\n  - name: cacherelay\n    base_url: {srv.base_url}\n    api_key: sk-fake-e2e\n"
             "    models:\n      fake-model:\n        prompt_caching: true\n")
    write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + route)
    yield srv, home
    srv.__exit__(None, None, None)


def _agent(srv, home):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    agent = AIAgent(provider="custom:cacherelay", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                    session_db=SessionDB(db_path=home / "state.db"), session_id="cache-contract", quiet_mode=True,
                    platform="cli")
    assert (agent._use_prompt_caching, agent._use_native_cache_layout) == (True, False)
    return agent


def test_tool_rounds_and_follow_up_turns_only_append(caching_route):
    srv, home = caching_route
    srv.push(ToolCall("terminal", {"command": "echo one"}), ToolCall("terminal", {"command": "echo two"}),
             Text("first done"), ToolCall("terminal", {"command": "echo three"}), Text("second done"))
    agent = _agent(srv, home)
    history: list[dict[str, Any]] = []
    for text in ("run two commands", "and one more"):
        history = agent.run_conversation(text, conversation_history=history, task_id="t")["messages"]
    agent.close()
    main = srv.main_requests()
    assert len(main) == 5
    assert not _prefix_breaks(main), _prefix_breaks(main)


def test_skill_turn_keeps_its_scaffold_bytes_across_the_tool_loop(caching_route):
    from agent.skill_commands import build_skill_invocation_message, scan_skill_commands

    srv, home = caching_route
    srv.push(ToolCall("terminal", {"command": "echo one"}), ToolCall("terminal", {"command": "echo two"}),
             Text("done"))
    scan_skill_commands()
    message = build_skill_invocation_message("/demo", "do the thing", task_id="t")
    assert message and SKILL_BODY.strip() in message
    agent = _agent(srv, home)
    agent.run_conversation(message, conversation_history=[], task_id="t")
    agent.close()
    main = srv.main_requests()
    assert len(main) == 3
    assert not _prefix_breaks(main), f"skill scaffold re-sent with different bytes: {_prefix_breaks(main)}"
