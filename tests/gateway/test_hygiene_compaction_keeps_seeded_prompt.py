"""Gateway hygiene / gateway /compress must not persist a reduced-toolset prompt over a live session.

Both paths compact with a detached ``AIAgent(enabled_toolsets=["memory"])`` seeded with the session's
stored prompt (``_seed_hygiene_system_prompt``). Compaction rebuilds the prompt at the commit boundary
(#98426) and the commit persists the result, so without the seed being honoured the session row lost the
skills index / provider blocks and every later fresh agent restored those degraded bytes verbatim.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import conversation_compression as cc
from gateway.run import _seed_hygiene_system_prompt

STORED = "You are Hermes.\n\n## Skills (mandatory)\n<available_skills>\n- a: b\n</available_skills>"
DEGRADED = "You are Hermes."


def _agent(**over):
    built = []

    def _build(system_message=None):
        built.append(system_message)
        return DEGRADED

    def _invalidate():
        agent._cached_system_prompt = None

    fields = dict(_cached_system_prompt=None, session_id="sess-1",
                  _build_system_prompt=_build, _invalidate_system_prompt=_invalidate)
    fields.update(over)
    agent = SimpleNamespace(**fields)
    return agent, built


def _rebuild(agent):
    with patch.object(cc, "_refresh_agent_tool_definitions", return_value=False), \
         patch("agent.system_prompt.reconstruct_static_prefix"):
        return cc._rebuild_system_prompt_at_boundary(agent, "")


def test_seeded_hygiene_agent_keeps_the_stored_prompt_at_the_commit_boundary():
    agent, built = _agent()
    assert _seed_hygiene_system_prompt(agent, {"system_prompt": STORED}) is True

    result = _rebuild(agent)

    assert result is STORED
    assert agent._cached_system_prompt is STORED
    assert built == []

    empty, built_empty = _agent()
    assert _seed_hygiene_system_prompt(empty, None) is False
    assert _rebuild(empty) == ""
    assert built_empty == []


def test_live_agent_compaction_still_rebuilds():
    agent, built = _agent(_cached_system_prompt=STORED)

    assert _rebuild(agent) == DEGRADED
    assert built == [""]


@pytest.mark.parametrize("kind, platform", [("hygiene", None), ("manual", "telegram")])
def test_compression_agent_refreshes_auto_ttl_after_context_engine_start(monkeypatch, kind, platform):
    import asyncio
    from unittest.mock import AsyncMock
    import hermes_cli.config
    from agent.context_engine import ContextEngine
    from gateway.run import GatewayRunner, _GATEWAY_HYGIENE_PLATFORM

    starts = []

    class Engine(ContextEngine):
        name = "cache-ttl-test"

        def update_from_response(self, usage):
            pass

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, current_tokens=None):
            return messages

        def on_session_start(self, session_id, **kwargs):
            starts.append(kwargs["platform"])

    cfg = {"context": {"engine": "cache-ttl-test"}}
    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: cfg)
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: cfg)
    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value=None))

    runtime = dict(api_key="fixture-key", provider="openrouter", base_url="https://openrouter.ai/api/v1")
    if platform:
        runtime["platform"] = platform
    with (
        patch("plugins.context_engine.load_context_engine", return_value=Engine()),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        if kind == "hygiene":
            agent, _ = asyncio.run(runner._hmwa_hygiene_build_agent(
                "fixture-model", runtime, SimpleNamespace(session_id="hygiene-cache")))
        else:
            agent = asyncio.run(runner._build_manual_compression_agent("manual-cache", "fixture-model", runtime))
    try:
        assert starts == [platform or "cli"]
        assert agent.platform == _GATEWAY_HYGIENE_PLATFORM
        assert agent._cache_ttl == "5m"
    finally:
        agent.close()
