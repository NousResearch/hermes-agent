"""Unit coverage for the background-review aux-model selector + routed digest.

Covers the two behaviors this change adds:
  • _resolve_review_runtime — auto/same-model → not routed (main model, warm
    cache); a configured different model → routed with resolved credentials.
  • _digest_history — compact replay used ONLY on the routed path (recent tail
    verbatim + a digest of older turns), preserving role alternation.

Pure-function / config-driven; no live model calls.
"""
from typing import Any
from unittest.mock import patch

from agent import background_review as br


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


def test_unrouted_runtime_applies_aux_service_tier_on_openrouter():
    agent = _FakeAgent(provider="openrouter", model="openai/gpt-5")
    agent._current_main_runtime = lambda: {
        "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }
    cfg = {"auxiliary": {"background_review": {"service_tier": "flex"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch(
        "hermes_cli.config.load_config_readonly", return_value=cfg
    ):
        rt = br._resolve_review_runtime(agent)
    assert rt["request_overrides"].get("service_tier") == "flex"
    assert rt["routed"] is False


def test_routed_runtime_applies_aux_service_tier_on_openrouter():
    agent = _FakeAgent()
    cfg = {
        "auxiliary": {
            "background_review": {
                "provider": "openrouter",
                "model": "google/gemini-3-flash-preview",
                "service_tier": "priority",
            }
        }
    }
    routed = {
        "provider": "openrouter",
        "model": "google/gemini-3-flash-preview",
        "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "credential_pool": None,
        "request_overrides": {},
        "max_output_tokens": 2048,
        "command": None,
        "args": [],
    }
    with patch("hermes_cli.config.load_config", return_value=cfg), patch(
        "hermes_cli.config.load_config_readonly", return_value=cfg
    ), patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=routed):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is True
    assert rt["request_overrides"].get("service_tier") == "priority"


def test_aux_service_tier_ignored_on_first_party_review_runtime():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {"service_tier": "flex"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch(
        "hermes_cli.config.load_config_readonly", return_value=cfg
    ):
        rt = br._resolve_review_runtime(agent)
    assert "service_tier" not in rt["request_overrides"]
    assert "speed" not in rt["request_overrides"]


def test_aux_service_tier_priority_ignored_on_first_party_review_runtime():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {"service_tier": "priority"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch(
        "hermes_cli.config.load_config_readonly", return_value=cfg
    ):
        rt = br._resolve_review_runtime(agent)
    assert "service_tier" not in rt["request_overrides"]
    assert "speed" not in rt["request_overrides"]


def test_apply_aux_honors_slot_extra_body_over_slot_tier():
    from hermes_cli.models import apply_aux_service_tier_overrides

    out = apply_aux_service_tier_overrides(
        {},
        {"service_tier": "flex", "extra_body": {"service_tier": "priority"}},
        model="openai/gpt-5",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        task="background_review",
    )
    assert out.get("service_tier") == "priority"


def test_unrouted_runtime_applies_slot_extra_body_service_tier():
    agent = _FakeAgent(provider="openrouter", model="openai/gpt-5")
    agent._current_main_runtime = lambda: {
        "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }
    cfg = {
        "auxiliary": {
            "background_review": {
                "service_tier": "flex",
                "extra_body": {"service_tier": "priority"},
            }
        }
    }
    with patch("hermes_cli.config.load_config", return_value=cfg), patch(
        "hermes_cli.config.load_config_readonly", return_value=cfg
    ):
        rt = br._resolve_review_runtime(agent)
    assert rt["request_overrides"].get("service_tier") == "priority"


def test_background_review_final_api_kwargs_slot_beats_global(monkeypatch):
    """auxiliary.background_review.service_tier wins over global agent.service_tier on the wire."""
    from run_agent import AIAgent

    cfg = {
        "agent": {"service_tier": "priority", "service_tier_overrides": {}},
        "auxiliary": {"background_review": {"service_tier": "flex"}},
    }
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    parent = AIAgent(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        model="openai/gpt-5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        platform="cli",
    )
    fork = None
    try:
        fork, _rt, _routed = br.build_cache_parity_fork(
            parent, cfg["auxiliary"]["background_review"], max_iterations=2,
        )
        kwargs = fork._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs.get("service_tier") == "flex"
        assert fork._service_tier_session_pinned is True
    finally:
        if fork is not None:
            fork.close()
        parent.close()


def test_background_review_final_api_kwargs_extra_body_beats_slot(monkeypatch):
    from run_agent import AIAgent

    task = {"service_tier": "flex", "extra_body": {"service_tier": "priority"}}
    cfg = {
        "agent": {"service_tier": "flex", "service_tier_overrides": {}},
        "auxiliary": {"background_review": task},
    }
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    parent = AIAgent(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        model="openai/gpt-5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        platform="cli",
    )
    fork = None
    try:
        fork, _rt, _routed = br.build_cache_parity_fork(parent, task, max_iterations=2)
        kwargs = fork._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs.get("service_tier") == "priority"
    finally:
        if fork is not None:
            fork.close()
        parent.close()


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
