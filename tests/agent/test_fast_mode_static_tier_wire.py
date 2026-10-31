"""A static ``agent.service_tier`` (fast / ultrafast) must reach the wire on every agent builder.

The CLI and gateway turn routes pin the tier into ``request_overrides``; builders that only set
``agent.service_tier`` (``hermes serve`` / TUI ``_make_agent``, ``hermes -z``, the api_server
per-request tier) used to send nothing. ``effective_request_overrides`` now resolves an unpinned
static tier through the same route gate as ``/fast``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import fast_mode
from hermes_cli.models import resolve_fast_mode_overrides

OPENAI = dict(provider="openai", base_url="https://api.openai.com/v1", api_mode="codex_responses")
CODEX = dict(provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses")
PROXY = dict(provider="custom", base_url="http://127.0.0.1:8000/v1", api_mode="chat_completions")
ANTHROPIC = dict(provider="anthropic", base_url="https://api.anthropic.com", api_mode="anthropic_messages")


def _agent(tier, model, route, request_overrides=None):
    return SimpleNamespace(service_tier=tier, model=model, request_overrides=request_overrides or {}, **route)


@pytest.mark.parametrize("tier,model,route", [
    ("ultrafast", "gpt-6-astra", OPENAI),
    ("ultrafast", "gpt-6-astra", CODEX),
    ("priority", "gpt-5.5", OPENAI),
    ("ultrafast", "gpt-5.5", OPENAI),       # no Ultrafast on this model: nothing, never priority
    ("priority", "gpt-5.5", PROXY),         # proxies never see the param
    ("ultrafast", "gpt-6-astra", PROXY),
    ("priority", "claude-opus-5", ANTHROPIC),
])
def test_unpinned_static_tier_matches_the_fast_gate(tier, model, route):
    expected = resolve_fast_mode_overrides(model, provider=route["provider"], base_url=route["base_url"], tier=tier)
    got = fast_mode.effective_request_overrides(_agent(tier, model, route, {"extra_body": {"keep": 1}}))
    assert got == {"extra_body": {"keep": 1}, **(expected or {})}


@pytest.mark.parametrize("pinned", [{"service_tier": "priority"}, {"speed": "fast"}])
def test_a_pinned_tier_is_left_alone(pinned):
    agent = _agent("ultrafast", "gpt-6-astra", OPENAI, dict(pinned))
    assert fast_mode.effective_request_overrides(agent) == pinned
    assert agent.request_overrides == pinned  # never mutated


@pytest.mark.parametrize("tier", [None, "auto", "cold"])
def test_normal_and_closed_windows_add_nothing(tier):
    agent = _agent(tier, "gpt-6-astra", OPENAI)
    agent._fast_until = 0.0
    assert fast_mode.effective_request_overrides(agent) == {}


def test_real_agent_built_with_only_service_tier_sends_it(monkeypatch):
    """The serve/TUI, `-z` and api_server constructor shape: service_tier set, request_overrides not."""
    import run_agent

    agent = run_agent.AIAgent(model="gpt-6-astra", api_key="test-key", quiet_mode=True, skip_context_files=True,
                              skip_memory=True, session_db=MagicMock(), enabled_toolsets=["terminal"],
                              service_tier="ultrafast", **CODEX)
    assert agent.request_overrides == {}
    assert agent._build_api_kwargs([{"role": "user", "content": "hi"}])["service_tier"] == "ultrafast"


def test_oneshot_builds_the_agent_with_the_configured_tier(monkeypatch):
    import run_agent

    import hermes_cli.config as config
    import hermes_cli.mcp_startup as mcp_startup
    import hermes_cli.oneshot as oneshot
    import hermes_cli.runtime_provider as runtime_provider

    cfg = {"model": {"default": "gpt-6-astra", "provider": "openai-codex"}, "agent": {"service_tier": "ultrafast"}}
    monkeypatch.setattr(config, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(runtime_provider, "resolve_runtime_with_fallback", lambda *a, **k: (
        {"provider": "openai-codex", "api_mode": "codex_responses", "base_url": CODEX["base_url"],
         "api_key": "test-key"}, None))
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: MagicMock())
    monkeypatch.setattr(oneshot, "_load_resume_target", lambda db, r: (None, [], None))
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **k: None)
    seen = {}

    class _Stop(Exception):
        pass

    class _Capture(run_agent.AIAgent):
        def run_conversation(self, *a, **k):
            seen["wire"] = self._build_api_kwargs([{"role": "user", "content": "hi"}]).get("service_tier")
            raise _Stop

    monkeypatch.setattr(run_agent, "AIAgent", _Capture)
    with pytest.raises(_Stop):
        oneshot._run_agent("hi", toolsets="terminal", use_config_toolsets=False)
    assert seen["wire"] == "ultrafast"
