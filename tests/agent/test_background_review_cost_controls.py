"""Unit coverage for the background-review aux-model selector + routed digest.

Covers the two behaviors this change adds:
  • _resolve_review_runtime — auto/same-model → not routed (main model, warm
    cache); a configured different model → routed with resolved credentials.
  • _digest_history — compact replay used ONLY on the routed path (recent tail
    verbatim + a digest of older turns), preserving role alternation.

Pure-function / config-driven; no live model calls.
"""
import time
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


def test_unrouted_fork_uses_pending_primary_snapshot_when_fallback_active():
    """#138979: a fork spawned while a fallback is active must run on the primary
    runtime the next turn restores to — not keep billing the fallback provider for
    the whole review."""
    agent = _FakeAgent(provider="anthropic", model="claude-haiku-4.5")  # live = fallback
    agent._current_main_runtime = lambda: {  # live runtime is the one-off fallback
        "provider": "anthropic", "model": "claude-haiku-4.5",
        "api_key": "fallback-key", "base_url": "https://api.anthropic.com",
        "api_mode": "messages",
    }
    agent._fallback_activated = True
    agent._primary_runtime = {
        "provider": "openai-codex", "model": "gpt-5.5",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "primary-key", "api_mode": "codex_app_server",
    }
    with patch("hermes_cli.config.load_config", return_value={}), patch("hermes_cli.config.load_config_readonly", return_value={}):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"
    assert rt["model"] == "gpt-5.5"
    assert rt["api_key"] == "primary-key"
    assert rt["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert rt["api_mode"] == "codex_responses"  # snapshot api_mode, downgrade still applied


def test_unrouted_fork_with_unusable_snapshot_keeps_live_runtime_fail_open():
    """Fail-open: fallback active but no usable primary snapshot → keep the live
    runtime exactly as before; never skip or block the review."""
    agent = _FakeAgent(provider="anthropic", model="claude-haiku-4.5")
    agent._fallback_activated = True
    agent._primary_runtime = {}  # unusable snapshot
    with patch("hermes_cli.config.load_config", return_value={}), patch("hermes_cli.config.load_config_readonly", return_value={}):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "anthropic"
    assert rt["model"] == "claude-haiku-4.5"
    assert rt["api_key"] == "parent-key"


# ---------------------------------------------------------------------------
# #138979 follow-ups — gates, pin precedence, all-or-nothing shaping, /btw scope
# ---------------------------------------------------------------------------

def _fallback_agent() -> _FakeAgent:
    """Live runtime is the one-off fallback; the primary snapshot is complete and
    carries its own request shaping."""
    agent = _FakeAgent(provider="anthropic", model="claude-haiku-4.5")
    agent._current_main_runtime = lambda: {
        "provider": "anthropic", "model": "claude-haiku-4.5",
        "api_key": "fallback-key", "base_url": "https://api.anthropic.com",
        "api_mode": "messages",
    }
    agent._fallback_activated = True
    agent._primary_runtime = {
        "provider": "openai-codex", "model": "gpt-5.5",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "primary-key", "api_mode": "codex_app_server",
        "request_overrides": {"extra_body": {"store": False}},
        "reasoning_config": {"effort": "low"},
    }
    return agent


def _resolve(agent, cfg=None, reset_gate=(False, None, False), **kwargs):
    cfg = cfg or {}
    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("agent.agent_runtime_helpers._primary_reset_gate_blocks", return_value=reset_gate):
        return br._resolve_review_runtime(agent, **kwargs)


def test_pending_primary_skipped_while_primary_in_429_cooldown():
    """#138979: a snapshot the next turn would NOT restore (primary still in 429
    cooldown) is not the pending primary. The fork keeps the live fallback runtime
    (fail-open: the review still runs) instead of aiming at a known-unavailable
    primary and failing every turn of the fallback window."""
    agent = _fallback_agent()
    agent._rate_limited_until = time.monotonic() + 3600
    rt = _resolve(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "anthropic"
    assert rt["model"] == "claude-haiku-4.5"
    assert rt["api_key"] == "fallback-key"


def test_pending_primary_skipped_when_primary_entitlement_rejected():
    """Same gate discipline as restore_primary_runtime: an entitlement-rejected
    (or known non-chat) primary is not restored next turn, so the fork stays live."""
    agent = _fallback_agent()
    agent._entitlement_rejected_models = {("openai-codex", "gpt-5.5")}
    rt = _resolve(agent)
    assert rt["provider"] == "anthropic"
    assert rt["model"] == "claude-haiku-4.5"


def test_pending_primary_skipped_when_reset_gate_blocks():
    """The pool reset gate (primary benched until a future resets_at) also vetoes
    the snapshot, mirroring restore_primary_runtime."""
    agent = _fallback_agent()
    rt = _resolve(agent, reset_gate=(True, None, False))
    assert rt["provider"] == "anthropic"
    assert rt["model"] == "claude-haiku-4.5"


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




# ---------------------------------------------------------------------------
# Cost / configurability controls (issue #87250)
# ---------------------------------------------------------------------------



def test_enabled_false_disables_automatic_review():
    cfg = {"auxiliary": {"background_review": {"enabled": False}}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert br.load_background_review_settings()[0] is False


def test_unresolvable_review_provider_falls_back_with_visible_warning(caplog):
    """The fork silently ran on the main model with only a debug line (#116055): the fallback must
    name the configured provider and reason at WARNING and reach the agent's user-visible warning rail."""
    import logging

    agent = _FakeAgent()
    emitted = []
    agent._emit_warning = emitted.append
    cfg = {"auxiliary": {"background_review": {"provider": "no-such-provider", "model": "review-model"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        with caplog.at_level(logging.WARNING, logger="agent.background_review"):
            rt = br._resolve_review_runtime(agent)
            br._resolve_review_runtime(agent)

    assert rt["routed"] is False and rt["model"] == "gpt-5.5"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings and all("no-such-provider" in w and "review-model" in w for w in warnings)
    assert len(emitted) == 1 and "no-such-provider" in emitted[0]  # once per agent on the user rail
