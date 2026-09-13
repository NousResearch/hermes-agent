"""Unit coverage for the background-review aux-model selector + routed digest.

Covers the two behaviors this change adds:
  • _resolve_review_runtime — auto/same-model → not routed (main model, warm
    cache); a configured different model → routed with resolved credentials.
  • _digest_history — compact replay used ONLY on the routed path (recent tail
    verbatim + a digest of older turns), preserving role alternation.

Pure-function / config-driven; no live model calls.
"""
from copy import deepcopy
from functools import partial
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent import background_review as br
from agent.background_review import build_cache_parity_fork
from run_agent import AIAgent


def _msg(role, content, tool_calls=None):
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


# ---------------------------------------------------------------------------
# _resolve_review_runtime — the aux-model selector
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, provider="openai-codex", model="gpt-5.5"):
        self.provider = provider
        self.model = model
        self._credential_pool: Any = None
        self.request_overrides = {}
        self.max_tokens: int | None = None

    def _current_main_runtime(self):
        return {
            "api_key": "parent-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_app_server",
        }


def test_routing_auto_inherits_parent_and_downgrades_codex_app_server():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {"provider": "auto", "model": ""}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"
    assert rt["model"] == "gpt-5.5"
    assert rt["api_mode"] == "codex_responses"  # downgraded so agent-loop tools dispatch


def test_routing_to_different_model_marks_routed_and_resolves_credentials():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    fake_rp = {
        "provider": "openrouter", "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions",
        "credential_pool": "routed-pool",
        "request_overrides": {"extra_body": {"store": False}},
        "max_output_tokens": 2048,
    }
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_rp):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is True
    assert rt["provider"] == "openrouter"
    assert rt["model"] == "google/gemini-3-flash-preview"
    assert rt["api_key"] == "or-key"
    assert rt["credential_pool"] == "routed-pool"
    assert rt["request_overrides"] == {"extra_body": {"store": False}}
    assert rt.get("max_tokens") is None


def test_unrouted_runtime_keeps_parent_pool_and_overrides():
    agent = _FakeAgent()
    agent._credential_pool = "parent-pool"
    agent.request_overrides = {"service_tier": "priority"}
    agent.max_tokens = 4096
    with patch("hermes_cli.config.load_config", return_value={}), patch("hermes_cli.config.load_config_readonly", return_value={}):
        rt = br._resolve_review_runtime(agent)
    assert rt["credential_pool"] == "parent-pool"
    assert rt["request_overrides"] == {"service_tier": "priority"}
    assert rt["max_tokens"] == 4096


def test_routing_same_model_as_parent_is_not_routed():
    agent = _FakeAgent(provider="openrouter", model="anthropic/claude-opus-4.8")
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "anthropic/claude-opus-4.8",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False  # same model/provider → keep full-replay path


def test_routing_resolution_failure_falls_back_to_parent():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=RuntimeError("boom")):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"


# ---------------------------------------------------------------------------
# _digest_history — routed-path compact replay
# ---------------------------------------------------------------------------

def test_digest_under_tail_returns_full():
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert br._digest_history(msgs, tail=24) == msgs


def test_digest_collapses_old_keeps_tail_verbatim():
    msgs = []
    for i in range(60):
        msgs.append(_msg("user", f"u{i} " + "x" * 50))
        msgs.append(_msg("assistant", f"a{i} " + "y" * 50))
    out = br._digest_history(msgs, tail=10)
    # First message is the synthetic digest (user role → alternation preserved).
    assert out[0]["role"] == "user"
    assert out[0]["content"].startswith("[Earlier conversation digest")
    # Recent tail preserved verbatim.
    assert out[-1] == msgs[-1]
    assert len(out) == 11  # 1 digest + 10 tail


def test_digest_does_not_open_tail_on_a_tool_message():
    msgs = []
    for i in range(40):
        msgs.append(_msg("user", "u" + "x" * 50))
        msgs.append(_msg("assistant", "", tool_calls=[
            {"function": {"name": "terminal", "arguments": "{}"}}]))
        msgs.append({"role": "tool", "content": "result " + "w" * 50})
    out = br._digest_history(msgs, tail=2)
    # The verbatim tail (after the digest) must not begin on a bare tool message.
    assert out[1]["role"] != "tool"


def test_digest_records_tool_names_in_arc():
    old = [
        _msg("user", "do the thing"),
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "skill_view", "arguments": "{}"}},
            {"function": {"name": "patch", "arguments": "{}"}}]),
    ]
    msgs = old + [_msg("user", f"tail{i}") for i in range(30)]
    out = br._digest_history(msgs, tail=10)
    digest = out[0]["content"]
    assert "USER: do the thing" in digest
    assert "tools: skill_view, patch" in digest


# ---------------------------------------------------------------------------
# Cost / configurability controls (issue #87250)
# ---------------------------------------------------------------------------

def test_enabled_defaults_true():
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert br.load_background_review_settings()[0] is True


def test_enabled_false_disables_automatic_review():
    cfg = {"auxiliary": {"background_review": {"enabled": False}}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert br.load_background_review_settings()[0] is False


@pytest.fixture
def review_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "agent": {"environment_probe": False},
        "custom_providers": [
            {"name": "tier-test", "model": "gpt-5.5", "base_url": "https://tier.example.test/v1",
             "extra_body": {"service_tier": "priority", "speed": "fast", "provider_marker": {"items": ["reloaded"]}}},
            {"name": "review-tier", "model": "gpt-4.1", "base_url": "https://review.example.test/v1",
             "extra_body": {"service_tier": "flex", "provider_marker": {"items": ["routed"]}}},
        ],
    }
    # Native loaders and native custom-provider merge read this isolated config.
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    agents = []
    with (
        patch("agent.process_bootstrap.OpenAI"),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
    ):
        parent = AIAgent(
            provider="custom", model="gpt-5.5", api_mode="chat_completions",
            api_key="fixture-key", base_url="https://tier.example.test/v1",
            quiet_mode=True, skip_memory=True, skip_context_files=True,
            request_overrides={"extra_body": {"store": False}},
        )
        parent._active_fallback_service_tier_override = "normal"
        agents.append(parent)
        yield parent, agents
        for agent in reversed(agents):
            agent.shutdown_memory_provider()
            agent.close()


@pytest.mark.parametrize("policy, expected", [("normal", None), (None, "priority")])
@pytest.mark.parametrize("restore", [False, True])
def test_real_inherited_child_reloads_defaults_but_retains_route_policy(review_runtime, policy, expected, restore):
    parent, agents = review_runtime
    parent._active_fallback_service_tier_override = policy
    before = deepcopy(parent.request_overrides)
    fork, runtime, routed = build_cache_parity_fork(parent, {}, max_iterations=1)
    agents.append(fork)
    assert isinstance(fork, AIAgent)
    assert routed is False
    # This survives descriptor stripping only because init really reloads defaults.
    assert fork.request_overrides["extra_body"]["service_tier"] == "priority"
    assert fork.request_overrides["extra_body"]["provider_marker"] == {"items": ["reloaded"]}
    if restore:
        fork._fallback_chain = [{"provider": "custom", "model": "gpt-4.1",
                                 "base_url": "https://review.example.test/v1", "api_mode": "chat_completions"}]
        client = MagicMock(api_key="child-key", base_url="https://review.example.test/v1")
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "gpt-4.1")):
            assert fork._try_activate_fallback() is True
        assert fork._active_fallback_service_tier_override is None
        assert fork._restore_primary_runtime() is True
        assert fork.model == parent.model
        assert fork.base_url == parent.base_url
    wire = fork._build_api_kwargs([{"role": "user", "content": "fixture"}])
    assert wire["extra_body"].get("service_tier") == expected
    assert wire["extra_body"].get("speed") == (None if policy == "normal" else "fast")
    assert wire["extra_body"]["store"] is False
    assert runtime["request_overrides"]["extra_body"].get("service_tier") == expected
    wire["extra_body"]["provider_marker"]["items"].append("wire-only")
    assert parent.request_overrides == before
    assert parent._active_fallback_service_tier_override == policy


def test_real_independently_routed_child_keeps_its_own_tier(review_runtime):
    parent, agents = review_runtime
    before = deepcopy(parent.request_overrides)
    fork, _, routed = build_cache_parity_fork(parent, {
        "provider": "custom", "model": "gpt-4.1",
        "base_url": "https://review.example.test/v1", "api_key": "review-fixture-key",
    }, max_iterations=1)
    agents.append(fork)
    assert routed is True
    assert fork.model == "gpt-4.1"
    wire = fork._build_api_kwargs([{"role": "user", "content": "fixture"}])
    assert wire["extra_body"]["service_tier"] == "flex"
    assert wire["extra_body"]["provider_marker"] == {"items": ["routed"]}
    assert parent.request_overrides == before
    assert parent._active_fallback_service_tier_override == "normal"


@pytest.mark.parametrize("parent_policy, child_policy, expected", [
    ("normal", None, "priority"), (None, "normal", None),
])
def test_real_child_init_fallback_keeps_its_own_policy(review_runtime, parent_policy, child_policy, expected):
    parent, agents = review_runtime
    parent.provider, parent.model = "alibaba-coding-plan", "qwen3.6-plus"
    parent.api_key = parent.base_url = ""
    parent._active_fallback_service_tier_override = parent_policy
    before = deepcopy(parent.request_overrides)
    fallback = {"provider": "custom", "model": "gpt-5.5", "service_tier_override": child_policy}
    client = MagicMock(api_key="fixture-key", base_url="https://tier.example.test/v1",
                       _custom_headers={}, default_headers={}, _default_headers={})

    def resolve_client(provider, **kwargs):
        return (client, "gpt-5.5") if provider == "custom" else (None, None)

    # Inject only the optional fallback argument. The native fork factory, real
    # AIAgent constructor, config loader, merge, activation and builder all run.
    with (
        patch("run_agent.AIAgent", new=partial(AIAgent, fallback_model=fallback)),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve_client),
    ):
        fork, _, routed = build_cache_parity_fork(parent, {}, max_iterations=1)
    agents.append(fork)
    assert isinstance(fork, AIAgent)
    assert routed is False
    assert fork._fallback_activated is True
    for restored in (False, True):
        if restored:
            assert fork._restore_primary_runtime() is True
        assert fork.provider == "custom"
        assert fork._active_fallback_service_tier_override == child_policy
        wire = fork._build_api_kwargs([{"role": "user", "content": "fixture"}])
        assert wire["extra_body"].get("service_tier") == expected
        assert wire["extra_body"]["provider_marker"] == {"items": ["reloaded"]}
    assert parent.request_overrides == before
    assert parent._active_fallback_service_tier_override == parent_policy
