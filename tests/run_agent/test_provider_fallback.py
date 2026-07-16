"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent, _pool_may_recover_from_rate_limit


def _make_agent(
    fallback_model=None,
    *,
    fallback_auto_activate=True,
    fallback_selection_interactive=False,
    clarify_callback=None,
    **agent_kwargs,
):
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
            fallback_auto_activate=fallback_auto_activate,
            fallback_selection_interactive=fallback_selection_interactive,
            clarify_callback=clarify_callback,
            **agent_kwargs,
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

    def test_single_dict_backwards_compat(self):
        fb = {"provider": "openai", "model": "gpt-4o"}
        agent = _make_agent(fallback_model=fb)
        assert agent._fallback_chain == [fb]
        assert agent._fallback_model == fb

    def test_list_of_providers(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 2
        assert agent._fallback_model == fbs[0]

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

    def test_empty_list(self):
        agent = _make_agent(fallback_model=[])
        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


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

    def test_second_fallback_works(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._try_activate_fallback() is True
            assert agent.model == "glm-4.7"
            assert agent._fallback_index == 2

    def test_all_exhausted_returns_false(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False

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

    def test_anthropic_host_custom_provider_uses_anthropic_messages(self):
        """A custom provider on the native api.anthropic.com host (no
        "/anthropic" path suffix, name != "anthropic") must resolve to the
        anthropic_messages wire protocol — not default to chat_completions,
        which POSTs /v1/chat/completions and 404s. Mirrors the primary-path
        determine_api_mode() host check."""
        fbs = [
            {
                "provider": "cron-anthropic",
                "model": "claude-sonnet-4-6",
                "base_url": "https://api.anthropic.com",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url="https://api.anthropic.com"),
                    "claude-sonnet-4-6",
                ),
            ),
            patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.api_mode == "anthropic_messages"


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

    def test_single_credential_returns_false(self):
        """With one credential that just 429'd, rotation has nowhere to go.

        The pool may still report has_available() True once cooldown expires,
        but retrying against the same entry will hit the same daily-quota
        429 and burn the retry budget.  Must fall back.
        """
        assert _pool_may_recover_from_rate_limit(_pool(1)) is False

    def test_single_credential_in_cooldown_returns_false(self):
        assert _pool_may_recover_from_rate_limit(_pool(1, has_available=False)) is False

    def test_two_credentials_available_returns_true(self):
        """With >1 credentials and at least one available, rotate instead of fallback."""
        assert _pool_may_recover_from_rate_limit(_pool(2)) is True

    def test_multiple_credentials_all_in_cooldown_returns_false(self):
        """All credentials cooling down — fall back rather than wait."""
        assert _pool_may_recover_from_rate_limit(_pool(3, has_available=False)) is False

    def test_many_credentials_available_returns_true(self):
        assert _pool_may_recover_from_rate_limit(_pool(10)) is True


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

    def test_skips_entry_matching_current_base_url_and_model(self):
        """Two custom_providers entries pointing at the same shim URL
        with the same model should dedup even if their provider names differ."""
        fbs = [
            # Different provider name but same shim URL + model — same backend.
            {"provider": "claude-cli-alt", "model": "claude-opus-4.7",
             "base_url": "http://127.0.0.1:7891/v1"},
            # Real different fallback.
            {"provider": "openrouter", "model": "anthropic/claude-opus-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "claude-cli"
        agent.model = "claude-opus-4.7"
        agent.base_url = "http://127.0.0.1:7891/v1"

        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # Same shim/base_url+model entry skipped, second one used.
        assert called == [("openrouter", "anthropic/claude-opus-4.7")], (
            f"expected base_url-aware dedup, got call order: {called}"
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


class TestManualFallbackSelection:
    def test_manual_mode_with_empty_effective_chain_does_not_prompt(self):
        clarify = MagicMock(return_value="should not be used")
        fbs = [{"provider": "openrouter", "model": "z-ai/glm-4.7"}]
        agent = _make_agent(
            fallback_model=fbs,
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=clarify,
        )
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        clarify.assert_not_called()
        mock_resolve.assert_not_called()

    def test_manual_mode_uses_only_selected_route(self):
        fbs = [
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "base_url": "https://api.anthropic.com",
            },
            {
                "provider": "openrouter",
                "model": "anthropic/claude-opus-4.1",
                "base_url": "https://openrouter.ai/api/v1",
            },
        ]
        selected_label = (
            "Continue with anthropic/claude-opus-4.1 via openrouter "
            "(https://openrouter.ai)"
        )
        clarify = MagicMock(return_value="2")
        agent = _make_agent(
            fallback_model=fbs,
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=clarify,
        )

        called = []

        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model, kwargs.get("explicit_base_url")))
            return _mock_client(base_url=kwargs.get("explicit_base_url") or ""), model

        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                ok = agent._try_activate_fallback()

        assert ok is True
        clarify.assert_called_once_with(
            "Primary model failed. Select a fallback route for this turn:",
            [
                "Continue with claude-sonnet-4-6 via anthropic (https://api.anthropic.com)",
                selected_label,
            ],
        )
        assert called == [
            ("openrouter", "anthropic/claude-opus-4.1", "https://openrouter.ai/api/v1")
        ]
        assert agent.provider == "openrouter"
        assert agent.model == "anthropic/claude-opus-4.1"

    @pytest.mark.parametrize(
        "response",
        [
            "",
            "[user did not respond within 2m]",
            "[clarify prompt could not be delivered]",
            "not a valid choice",
            "0",
            "2",
        ],
    )
    def test_manual_mode_unmatched_response_fails_closed(self, response):
        agent = _make_agent(
            fallback_model=[{"provider": "openai", "model": "gpt-4o"}],
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=MagicMock(return_value=response),
        )

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_not_called()

    def test_manual_selection_state_resets_for_next_turn(self):
        agent = _make_agent(
            fallback_model=[{"provider": "openai", "model": "gpt-4o"}],
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=MagicMock(),
        )
        agent._fallback_manual_attempted = True
        agent._fallback_manual_selected_index = 0

        agent._reset_turn_scoped_fallback_state()

        assert agent._fallback_manual_attempted is False
        assert agent._fallback_manual_selected_index is None

    def test_two_turn_rate_limit_restores_primary_and_prompts_again(self):
        """Manual consent is one turn and run_conversation re-arms it next turn."""

        class RateLimitError(Exception):
            status_code = 429

            def __init__(self):
                super().__init__("Error code: 429 - rate limit exceeded")
                self.response = SimpleNamespace(headers={})
                self.body = {"error": {"message": "rate limit exceeded"}}

        def response(text):
            message = SimpleNamespace(content=text, tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(
                choices=[choice], model="fallback/model", usage=None
            )

        clarify = MagicMock(side_effect=["1", "1"])
        agent = _make_agent(
            fallback_model=[{"provider": "openai", "model": "gpt-4o"}],
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=clarify,
            provider="custom",
            model="primary-model",
        )
        agent._api_max_retries = 1
        calls = []

        def api_call(_kwargs):
            calls.append((agent.provider, agent.model))
            if len(calls) in {1, 3}:
                raise RateLimitError()
            return response(f"fallback turn {len(calls) // 2}")

        fallback_client = _mock_client()
        with (
            patch.object(agent, "_interruptible_api_call", side_effect=api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(fallback_client, "gpt-4o"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda model, provider: model,
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=200000,
            ),
        ):
            first = agent.run_conversation("turn one")
            second = agent.run_conversation(
                "turn two", conversation_history=first["messages"]
            )

        assert first["completed"] is True
        assert second["completed"] is True
        assert calls == [
            ("custom", "primary-model"),
            ("openai", "gpt-4o"),
            ("custom", "primary-model"),
            ("openai", "gpt-4o"),
        ]
        assert clarify.call_count == 2

    def test_manual_mode_supports_callback_attached_after_init(self):
        label = "Continue with gpt-4o via openai"
        agent = _make_agent(
            fallback_model=[{"provider": "openai", "model": "gpt-4o"}],
            fallback_auto_activate=False,
            fallback_selection_interactive=None,
            clarify_callback=None,
        )
        agent.clarify_callback = MagicMock(return_value=label)

        assert agent._has_pending_fallback() is True
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "gpt-4o"),
        ):
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                assert agent._try_activate_fallback() is True

    def test_manual_mode_noninteractive_fails_closed_even_with_callback(self):
        clarify = MagicMock(
            return_value="Continue with gpt-4o via openai"
        )
        agent = _make_agent(
            fallback_model=[{"provider": "openai", "model": "gpt-4o"}],
            fallback_auto_activate=False,
            fallback_selection_interactive=False,
            clarify_callback=clarify,
        )

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        clarify.assert_not_called()
        mock_resolve.assert_not_called()

    def test_manual_mode_selected_route_failure_does_not_cascade(self):
        fbs = [
            {"provider": "broken", "model": "bad-route"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        clarify = MagicMock(return_value="Continue with bad-route via broken")
        agent = _make_agent(
            fallback_model=fbs,
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=clarify,
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),
        ) as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_called_once()

    def test_manual_mode_never_cascades_after_selected_route_activation(self):
        label = "Continue with gpt-4o via openai"
        clarify = MagicMock(return_value=label)
        agent = _make_agent(
            fallback_model=[
                {"provider": "openai", "model": "gpt-4o"},
                {"provider": "zai", "model": "glm-4.7"},
            ],
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=clarify,
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "gpt-4o"),
        ) as mock_resolve:
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                assert agent._try_activate_fallback() is True
                assert agent._try_activate_fallback() is False

        clarify.assert_called_once()
        mock_resolve.assert_called_once()

    def test_manual_mode_redacts_base_url_credentials_from_choices(self):
        safe_label = (
            "Continue with gpt-4o via custom (https://api.example)"
        )
        clarify = MagicMock(return_value=safe_label)
        agent = _make_agent(
            fallback_model=[
                {
                    "provider": "custom",
                    "model": "gpt-4o",
                    "base_url": "https://user:password@api.example/token-secret/v1?api_key=secret",
                }
            ],
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=clarify,
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "gpt-4o"),
        ):
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                assert agent._try_activate_fallback() is True

        choices = clarify.call_args.args[1]
        assert choices == [safe_label]
        assert "password" not in choices[0]
        assert "secret" not in choices[0]

    def test_manual_mode_renders_ipv6_origin_safely(self):
        safe_label = (
            "Continue with gpt-4o via custom (https://[2001:db8::1]:8443)"
        )
        clarify = MagicMock(return_value=safe_label)
        agent = _make_agent(
            fallback_model=[
                {
                    "provider": "custom",
                    "model": "gpt-4o",
                    "base_url": (
                        "https://user:password@[2001:db8::1]:8443/secret/v1"
                    ),
                }
            ],
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=clarify,
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "gpt-4o"),
        ):
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                assert agent._try_activate_fallback() is True

        assert clarify.call_args.args[1] == [safe_label]

    def test_manual_mode_fails_closed_when_choice_limit_exceeded(self):
        clarify = MagicMock(return_value="should not be used")
        agent = _make_agent(
            fallback_model=[
                {"provider": f"provider-{idx}", "model": f"model-{idx}"}
                for idx in range(5)
            ],
            fallback_auto_activate=False,
            fallback_selection_interactive=True,
            clarify_callback=clarify,
        )

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        clarify.assert_not_called()
        mock_resolve.assert_not_called()
