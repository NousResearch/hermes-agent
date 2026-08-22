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

import pytest

from agent import background_review as br
from hermes_cli import runtime_provider as _runtime_provider  # noqa: F401


def _msg(role, content, tool_calls=None):
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


# ---------------------------------------------------------------------------
# _resolve_review_runtime — the aux-model selector
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(
        self,
        provider="openai-codex",
        model="gpt-5.5",
        requested_provider=None,
    ):
        self.provider = provider
        self.requested_provider = requested_provider or provider
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
    assert rt["max_tokens"] == 2048


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


@pytest.mark.parametrize(
    "task_provider",
    ("litellm-local", "custom:litellm-local"),
)
def test_routing_same_custom_alias_inherits_live_parent_credential(task_provider):
    """A named custom alias resolving to ``custom`` is still the parent route.

    In a multiplex process, config interpolation may have captured the default
    profile's process-global key before this profile's scope was installed.
    The review must recognize the requested alias and inherit the live parent
    credential instead of explicitly forwarding that stale expanded value.
    """
    from agent import secret_scope

    agent = _FakeAgent(
        provider="custom",
        requested_provider="litellm-local",
        model="MiniMax-M3",
    )
    task = {
        "provider": task_provider,
        "model": "MiniMax-M3",
        "api_key": "wrong-default-profile-key",
        "key_env": "LITELLM_MASTER_KEY",
    }
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(
        {"LITELLM_MASTER_KEY": "scoped-profile-key"}
    )
    try:
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as resolve_runtime, patch(
            "agent.secret_scope.get_secret"
        ) as get_secret:
            rt = br._resolve_review_runtime(agent, task)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)

    assert rt["routed"] is False
    assert rt["requested_provider"] == "litellm-local"
    assert rt["api_key"] == "parent-key"
    resolve_runtime.assert_not_called()
    get_secret.assert_not_called()


def test_distinct_review_provider_uses_scoped_key_env_credential():
    """A different provider remains routed even when its model name matches."""
    from agent import secret_scope

    agent = _FakeAgent(
        provider="custom",
        requested_provider="litellm-local",
        model="MiniMax-M3",
    )
    task = {
        "provider": "other-proxy",
        "model": "MiniMax-M3",
        "api_key": "wrong-default-profile-key",
        "key_env": "LITELLM_MASTER_KEY",
    }
    routed = {
        "provider": "custom",
        "requested_provider": "other-proxy",
        "model": "MiniMax-M3",
        "api_key": "scoped-profile-key",
        "base_url": "http://127.0.0.1:4001/v1",
        "api_mode": "chat_completions",
    }
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(
        {"LITELLM_MASTER_KEY": "scoped-profile-key"}
    )
    try:
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=routed,
        ) as resolve_runtime:
            rt = br._resolve_review_runtime(agent, task)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)

    assert rt["routed"] is True
    assert rt["requested_provider"] == "other-proxy"
    assert rt["model"] == "MiniMax-M3"
    resolve_runtime.assert_called_once_with(
        requested="other-proxy",
        target_model="MiniMax-M3",
        explicit_api_key="scoped-profile-key",
        explicit_base_url=None,
    )


def test_distinct_review_provider_real_resolution_uses_profile_scope(
    tmp_path,
    monkeypatch,
):
    """Exercise config expansion through the real custom-provider resolver."""
    from agent import secret_scope

    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
providers:
  other-proxy:
    api: http://127.0.0.1:4001/v1
    key_env: LITELLM_MASTER_KEY
    default_model: MiniMax-M3
auxiliary:
  background_review:
    provider: other-proxy
    model: MiniMax-M3
    api_key: ${LITELLM_MASTER_KEY}
    key_env: LITELLM_MASTER_KEY
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("LITELLM_MASTER_KEY", "wrong-default-profile-key")
    agent = _FakeAgent(
        provider="custom",
        requested_provider="litellm-local",
        model="MiniMax-M3",
    )

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(
        {"LITELLM_MASTER_KEY": "scoped-profile-key"}
    )
    try:
        rt = br._resolve_review_runtime(agent)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)

    assert rt["routed"] is True
    assert rt["provider"] == "custom"
    assert rt["requested_provider"] == "other-proxy"
    assert rt["model"] == "MiniMax-M3"
    assert rt["api_key"] == "scoped-profile-key"
    assert rt["base_url"] == "http://127.0.0.1:4001/v1"


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
        assert br.is_background_review_enabled() is True


def test_enabled_false_disables_automatic_review():
    cfg = {"auxiliary": {"background_review": {"enabled": False}}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert br.is_background_review_enabled() is False
