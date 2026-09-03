"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import chat_completion_helpers
from agent.error_classifier import FailoverReason
from run_agent import AIAgent, _pool_may_recover_from_rate_limit


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None



    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"


    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (FailoverReason.auth, "authentication failed"),
        (FailoverReason.billing, "billing or quota exhausted"),
        (FailoverReason.rate_limit, "rate limit"),
        (FailoverReason.upstream_rate_limit, "upstream model rate limit"),
        (FailoverReason.overloaded, "provider overloaded"),
        (FailoverReason.server_error, "provider server error"),
        (FailoverReason.timeout, "request timeout"),
        (FailoverReason.model_not_found, "model not found"),
        (
            FailoverReason.egress_policy_blocked,
            "local egress policy blocked the request",
        ),
        (FailoverReason.unknown, "provider failure"),
    ],
)
def test_fallback_reason_text_is_operator_friendly(reason, expected):
    assert chat_completion_helpers._fallback_reason_text(reason) == expected


def test_fallback_reason_text_defaults_when_reason_is_missing():
    assert chat_completion_helpers._fallback_reason_text(None) == "provider failure"


def test_egress_policy_skips_remote_fallbacks_and_uses_loopback():
    """Unsafe remote payloads must go directly to a local fallback.

    The firewall has already rejected the current request, so retrying the
    same payload against another remote provider only creates noisy false
    provider failures and cannot succeed.
    """
    agent = _make_agent(
        fallback_model=[
            {
                "provider": "openai-codex",
                "model": "gpt-5.5",
                "base_url": "https://chatgpt.com/backend-api/codex",
            },
            {
                "provider": "ollama-launch",
                "model": "hermes-review-fast:latest",
                "base_url": "http://127.0.0.1:11434/v1",
            },
        ]
    )
    clients = [_mock_client(base_url="http://127.0.0.1:11434/v1")]
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(clients[0], "hermes-review-fast:latest"),
    ) as resolve_client:
        assert (
            agent._try_activate_fallback(FailoverReason.egress_policy_blocked)
            is True
        )

    assert agent.provider == "ollama-launch"
    assert agent.model == "hermes-review-fast:latest"
    assert agent._fallback_index == 2
    resolve_client.assert_called_once()


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    def test_records_user_visible_switch_with_reason(self):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5.2"},
        )
        agent.model = "gpt-5.6-sol"
        agent.provider = "openai-codex"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url="https://api.z.ai/v1"), "glm-5.2"),
        ):
            assert agent._try_activate_fallback(FailoverReason.rate_limit) is True

        expected = (
            "⚠️ Model fallback: gpt-5.6-sol via openai-codex unavailable "
            "(rate limit); using glm-5.2 via zai."
        )
        assert agent._pending_fallback_notice == [expected]
        assert agent._retry_status_buffer[-1] == ("status", expected)

    def test_records_sequential_switches_in_order(self):
        agent = _make_agent(
            fallback_model=[
                {"provider": "zai", "model": "glm-5.2"},
                {"provider": "deepseek", "model": "deepseek-v4-flash"},
            ],
        )
        agent.model = "gpt-5.6-sol"
        agent.provider = "openai-codex"
        clients = [
            _mock_client(base_url="https://api.z.ai/v1"),
            _mock_client(base_url="https://api.deepseek.com/v1"),
        ]
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(clients[0], "glm-5.2"), (clients[1], "deepseek-v4-flash")],
        ):
            assert agent._try_activate_fallback(FailoverReason.rate_limit) is True
            assert agent._try_activate_fallback(FailoverReason.overloaded) is True

        assert agent._pending_fallback_notice == [
            "⚠️ Model fallback: gpt-5.6-sol via openai-codex unavailable "
            "(rate limit); using glm-5.2 via zai.",
            "⚠️ Model fallback: glm-5.2 via zai unavailable "
            "(provider overloaded); using deepseek-v4-flash via deepseek.",
        ]
    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_resolves_key_env_for_fallback_provider(self):
        fbs = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example/v1",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(
                        base_url="https://fallback.example/v1",
                        api_key="env-secret",
                    ),
                    "fallback-model",
                ),
            ) as mock_rpc,
        ):
            assert agent._try_activate_fallback() is True
            assert mock_rpc.call_args.kwargs["explicit_api_key"] == "env-secret"


    def test_nous_anthropic_fallback_uses_the_messages_wire(self):
        """Portal Claude fallbacks must not stay on chat_completions.

        ``resolve_provider_client`` still returns an OpenAI client for Nous;
        activation has to re-derive api_mode from the model and rebuild the
        Anthropic client — otherwise the turn POSTs /chat/completions.
        """
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [
            {
                "provider": "nous",
                "model": "anthropic/claude-opus-4.8",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        rebuilt = {"count": 0}

        def _fake_build(api_key, base_url, timeout=None, **kwargs):
            rebuilt["count"] += 1
            rebuilt["api_key"] = api_key
            rebuilt["base_url"] = base_url
            return MagicMock(name="anthropic-client")

        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url=portal, api_key="portal-jwt"),
                    "anthropic/claude-opus-4.8",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=_fake_build,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "anthropic_messages"
        assert agent.provider == "nous"
        assert agent.model == "anthropic/claude-opus-4.8"
        assert agent.client is None
        assert rebuilt["count"] == 1
        assert rebuilt["api_key"] == "portal-jwt"
        assert rebuilt["base_url"] == portal
        assert agent._anthropic_client is not None

    def test_nous_non_anthropic_fallback_stays_on_chat_completions(self):
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [{"provider": "nous", "model": "hermes-4-405b"}]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url=portal, api_key="portal-jwt"),
                    "hermes-4-405b",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=AssertionError("must not build Anthropic client"),
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "chat_completions"
        assert agent.client is not None


# ── Pool-rotation vs fallback gating (#11314) ────────────────────────────


def _pool(n_entries: int, has_available: bool = True):
    """Make a minimal credential-pool stand-in for rotation-room checks."""
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


class TestPoolRotationRoom:
    def test_none_pool_returns_false(self):
        assert _pool_may_recover_from_rate_limit(None) is False







# ── Skip-self dedup (#22548) ───────────────────────────────────────────────


class TestFallbackChainDedup:
    """A fallback chain entry that resolves to the current provider/model
    (or the same custom-provider base_url) must be skipped, not retried.
    Otherwise a misconfigured chain or two custom_providers entries pointing
    at the same shim loop the same failure. See issue #22548."""

    def test_skips_entry_matching_current_provider_and_model(self):
        """Chain has [same-as-current, real-fallback]; activate must skip
        the first and use the second."""
        fbs = [
            # First entry == current state. Should be skipped.
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
            # Second entry: real fallback.
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        # Stub out resolve_provider_client so we can assert which entry was
        # actually used — return a MagicMock client tagged with the provider.
        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # The first entry was skipped — only the second reached resolve.
        assert called == [("zai", "glm-4.7")], (
            f"expected fallback to skip same-state entry, got call order: {called}"
        )


    def test_returns_false_when_only_self_matching_entries(self):
        """A chain with only self-matching entries exhausts to False."""
        fbs = [
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_not_called()

    def test_allows_xai_api_fallback_from_xai_oauth_same_host_model(self):
        """xai-oauth and xai share api.x.ai but use different credentials.

        A spending-limit 403 on OAuth must still be able to fall over to the
        API-key provider even when both entries use the same model slug and
        base URL.  Blind base_url+model dedup incorrectly skipped that path.
        """
        fbs = [
            {
                "provider": "xai",
                "model": "grok-4.5",
                "base_url": "https://api.x.ai/v1",
            },
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "xai-oauth"
        agent.model = "grok-4.5"
        agent.base_url = "https://api.x.ai/v1"

        called = []

        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(base_url="https://api.x.ai/v1"), model

        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                ok = agent._try_activate_fallback()

        assert ok is True
        assert called == [("xai", "grok-4.5")]
        assert agent.provider == "xai"
        assert agent.model == "grok-4.5"


# ── extra_body re-resolution on fallback activation (#75091) ─────────────


class TestFallbackExtraBodyReResolution:
    """Fallback activation must re-resolve extra_body key-scoped.

    The old provider's custom_providers-contributed extra_body keys are
    stale on the new backend and must be dropped; caller-provided
    request_overrides keys must survive; the fallback provider's own
    extra_body must be merged in (salvage of #75139).
    """

    OLD_URL = "https://old-llm.example.com/v1"
    FB_URL = "https://fb-llm.example.com/v1"

    def _agent_with_custom_providers(self, caller_extra_body=None):
        agent = _make_agent(
            fallback_model={
                "provider": "custom:fbprov",
                "model": "fb-model",
                "base_url": self.FB_URL,
            },
        )
        agent.provider = "custom"
        agent.model = "old-model"
        agent.base_url = self.OLD_URL
        agent._custom_providers = [
            {
                "name": "oldprov",
                "base_url": self.OLD_URL,
                "extra_body": {"enable_thinking": True, "old_only": 1},
            },
            {
                "provider_key": "fbprov",
                "base_url": self.FB_URL,
                "extra_body": {"top_k": 20},
            },
        ]
        # Simulate the init-time merge: provider extra_body + caller keys
        # (caller wins on conflict — agent_init._merge_custom_provider_extra_body).
        merged = {"enable_thinking": True, "old_only": 1}
        merged.update(caller_extra_body or {})
        agent.request_overrides = {"extra_body": merged}
        return agent

    def _activate(self, agent):
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url=self.FB_URL), "fb-model"),
        ), patch(
            "agent.model_metadata.get_model_context_length",
            return_value=128_000,
        ):
            assert agent._try_activate_fallback() is True

    def test_stale_provider_keys_removed_and_new_provider_merged(self):
        agent = self._agent_with_custom_providers()
        self._activate(agent)
        eb = agent.request_overrides.get("extra_body") or {}
        # Old provider's contributed keys are gone.
        assert "enable_thinking" not in eb
        assert "old_only" not in eb
        # Fallback provider's own extra_body is applied.
        assert eb.get("top_k") == 20

    def test_caller_override_keys_survive_fallback(self):
        agent = self._agent_with_custom_providers(
            caller_extra_body={"reasoning": {"effort": "high"}, "enable_thinking": False},
        )
        self._activate(agent)
        eb = agent.request_overrides.get("extra_body") or {}
        # Pure caller key survives untouched.
        assert eb.get("reasoning") == {"effort": "high"}
        # Caller redefined a key the old provider also set (caller won at
        # init: False != True) — the caller's value must survive key-scoped
        # removal.
        assert eb.get("enable_thinking") is False
        # But the key the old provider alone contributed is dropped.
        assert "old_only" not in eb
        assert eb.get("top_k") == 20

    def test_non_extra_body_overrides_untouched(self):
        agent = self._agent_with_custom_providers()
        agent.request_overrides["temperature"] = 0.2
        self._activate(agent)
        assert agent.request_overrides.get("temperature") == 0.2
