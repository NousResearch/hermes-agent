"""Regression tests for the ``auto`` → main-model-first policy.

Prior to this change, aggregator users (OpenRouter / Nous Portal) had aux
tasks routed through a cheap provider-side default (Gemini Flash) while
non-aggregator users got their main model.  This made behavior inconsistent
and surprising — users picked Claude but got Gemini Flash summaries.

The current policy: ``auto`` means "use my main chat model" for every user,
regardless of provider type.  Explicit per-task overrides in ``config.yaml``
(``auxiliary.<task>.provider``) still win.  The cheap fallback chain only
runs when the main provider has no working client.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch



# ── Text aux tasks — _resolve_auto ──────────────────────────────────────────


class TestResolveAutoMainFirst:
    """_resolve_auto() must prefer main provider + main model for every user."""

    def test_openrouter_main_uses_main_model_for_aux(self, monkeypatch):
        """OpenRouter main user → aux uses their picked OR model, not Gemini Flash."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        with patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="anthropic/claude-sonnet-4.6",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-sonnet-4.6")

            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto()

        assert client is mock_client
        assert model == "anthropic/claude-sonnet-4.6"
        # Verify it asked resolve_provider_client for the MAIN provider+model,
        # not a fallback-chain provider
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.args[0] == "openrouter"
        assert mock_resolve.call_args.args[1] == "anthropic/claude-sonnet-4.6"

    def test_nous_main_uses_main_model_for_aux(self, monkeypatch):
        """Nous Portal main user → aux uses their picked Nous model, not free-tier MiMo."""
        # No OPENROUTER_API_KEY → ensures if main failed we'd fall to chain
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="anthropic/claude-opus-4.6",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-opus-4.6")

            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto()

        assert client is mock_client
        assert model == "anthropic/claude-opus-4.6"
        assert mock_resolve.call_args.args[0] == "nous"

    def test_non_aggregator_main_still_uses_main(self, monkeypatch):
        """Non-aggregator main (DeepSeek) → unchanged behavior, main model used."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="deepseek",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="deepseek-chat",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "deepseek-chat")

            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto()

        assert client is mock_client
        assert model == "deepseek-chat"
        assert mock_resolve.call_args.args[0] == "deepseek"

    def test_main_unavailable_falls_through_to_chain(self, monkeypatch):
        """Main provider with no working client → fall back to aux chain."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        chain_client = MagicMock()
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="anthropic",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),  # main provider has no client
        ), patch(
            "agent.auxiliary_client._try_openrouter",
            return_value=(chain_client, "google/gemini-3-flash-preview"),
        ):
            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto()

        assert client is chain_client
        assert model == "google/gemini-3-flash-preview"

    def test_no_main_config_uses_chain_directly(self):
        """No main provider configured → skip step 1, use chain (no regression)."""
        chain_client = MagicMock()
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="",
        ), patch(
            "agent.auxiliary_client._try_openrouter",
            return_value=(chain_client, "google/gemini-3-flash-preview"),
        ):
            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto()

        assert client is chain_client

    def test_runtime_override_wins_over_config(self, monkeypatch):
        """main_runtime kwarg overrides config-read main provider/model."""
        with patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="config-model",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_resolve.return_value = (MagicMock(), "runtime-model")

            from agent.auxiliary_client import _resolve_auto

            _resolve_auto(main_runtime={
                "provider": "anthropic",
                "model": "runtime-model",
                "base_url": "",
                "api_key": "",
                "api_mode": "",
            })

        # Runtime override wins
        assert mock_resolve.call_args.args[0] == "anthropic"
        assert mock_resolve.call_args.args[1] == "runtime-model"

    def test_runtime_base_url_passed_for_named_api_key_provider(self):
        """Named API-key providers inherit the live session endpoint for aux work."""
        token_plan_url = "https://token-plan-sgp.xiaomimimo.com/v1"
        with patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="config-model",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_resolve.return_value = (MagicMock(), "mimo-v2.5-pro")

            from agent.auxiliary_client import _resolve_auto

            _resolve_auto(main_runtime={
                "provider": "xiaomi",
                "model": "mimo-v2.5-pro",
                "base_url": token_plan_url,
                "api_key": "tp-test-key",
                "api_mode": "chat_completions",
            })

        assert mock_resolve.call_args.args[0] == "xiaomi"
        assert mock_resolve.call_args.args[1] == "mimo-v2.5-pro"
        assert mock_resolve.call_args.kwargs["explicit_base_url"] == token_plan_url
        assert mock_resolve.call_args.kwargs["explicit_api_key"] == "tp-test-key"
        assert mock_resolve.call_args.kwargs["api_mode"] == "chat_completions"


# ── Vision — resolve_vision_provider_client ─────────────────────────────────


class TestResolveVisionMainFirst:
    """Vision auto-detection prefers the main provider first."""

    def test_openrouter_main_vision_uses_main_model(self, monkeypatch):
        """OpenRouter main with vision-capable model → aux vision uses main model."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="anthropic/claude-sonnet-4.6",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve, patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ):
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-sonnet-4.6")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "openrouter"
        assert client is mock_client
        assert model == "anthropic/claude-sonnet-4.6"
        # Verify it did NOT call the strict vision backend for OpenRouter
        # (which would have used a cheap gemini-flash-preview default)
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.args[0] == "openrouter"
        assert mock_resolve.call_args.args[1] == "anthropic/claude-sonnet-4.6"
        assert mock_resolve.call_args.kwargs.get("is_vision") is True

    def test_nous_main_vision_uses_paid_nous_vision_backend(self):
        """Paid Nous main → aux vision uses the dedicated Nous vision backend."""
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="openai/gpt-5",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            return_value=(MagicMock(), "google/gemini-3-flash-preview"),
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "nous"
        assert client is not None
        assert model == "google/gemini-3-flash-preview"

    def test_nous_main_vision_uses_free_tier_nous_vision_backend(self):
        """Free-tier Nous main → aux vision uses MiMo omni, not the text main model."""
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="xiaomi/mimo-v2-pro",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            return_value=(MagicMock(), "xiaomi/mimo-v2-omni"),
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "nous"
        assert client is not None
        assert model == "xiaomi/mimo-v2-omni"

    def test_exotic_provider_with_vision_override_preserved(self):
        """xiaomi → mimo-v2.5 override still wins over main_model."""
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="xiaomi",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="mimo-v2-pro",  # text model
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve, patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ):
            mock_resolve.return_value = (MagicMock(), "mimo-v2.5")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "xiaomi"
        # Should use mimo-v2.5 (vision override), not mimo-v2-pro (text main)
        assert mock_resolve.call_args.args[1] == "mimo-v2.5"
        assert mock_resolve.call_args.kwargs.get("is_vision") is True

    def test_copilot_vision_sets_vision_header(self, monkeypatch):
        """Copilot vision requests include the header required for vision routing."""
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghu_test-token")

        captured = {}

        def fake_headers(*, is_agent_turn=False, is_vision=False):
            captured["is_agent_turn"] = is_agent_turn
            captured["is_vision"] = is_vision
            return {"Copilot-Vision-Request": "true"} if is_vision else {}

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="copilot",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="configured-copilot-model",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.OpenAI",
        ) as mock_openai, patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "copilot",
                "api_key": "copilot-api-token",
                "base_url": "https://api.githubcopilot.com",
            },
        ), patch(
            "hermes_cli.copilot_auth.copilot_request_headers",
            side_effect=fake_headers,
        ):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "copilot"
        assert client is mock_client
        assert model == "configured-copilot-model"
        assert captured == {"is_agent_turn": True, "is_vision": True}
        assert mock_openai.call_args.kwargs["default_headers"]["Copilot-Vision-Request"] == "true"

    def test_text_copilot_does_not_set_vision_header(self, monkeypatch):
        """Text Copilot requests keep the vision-only header off."""
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghu_test-token")

        captured = {}

        def fake_headers(*, is_agent_turn=False, is_vision=False):
            captured["is_agent_turn"] = is_agent_turn
            captured["is_vision"] = is_vision
            return {"Copilot-Vision-Request": "true"} if is_vision else {}

        with patch(
            "agent.auxiliary_client.OpenAI",
        ) as mock_openai, patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "copilot",
                "api_key": "copilot-api-token",
                "base_url": "https://api.githubcopilot.com",
            },
        ), patch(
            "hermes_cli.copilot_auth.copilot_request_headers",
            side_effect=fake_headers,
        ):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("copilot", "gpt-5-mini")

        assert client is mock_client
        assert model == "gpt-5-mini"
        assert captured == {"is_agent_turn": True, "is_vision": False}
        assert "default_headers" not in mock_openai.call_args.kwargs

    def test_main_unavailable_vision_falls_through_to_aggregators(self):
        """Main provider fails → fall back to OpenRouter/Nous strict backends."""
        fallback_client = MagicMock()
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="deepseek",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="deepseek-chat",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),
        ), patch(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            return_value=(fallback_client, "google/gemini-3-flash-preview"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert client is fallback_client
        assert provider in {"openrouter", "nous"}

    def test_explicit_provider_override_still_wins(self):
        """Explicit config override bypasses main-first policy."""
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="anthropic/claude-opus-4.6",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("nous", None, None, None, None),  # explicit override
        ), patch(
            "agent.auxiliary_client._resolve_strict_vision_backend"
        ) as mock_strict:
            mock_strict.return_value = (MagicMock(), "nous-default-model")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        # Explicit "nous" override → uses strict backend, NOT main model path
        assert provider == "nous"
        mock_strict.assert_called_once_with("nous", None)


# ── Constant cleanup ────────────────────────────────────────────────────────


def test_aggregator_providers_constant_removed():
    """The dead _AGGREGATOR_PROVIDERS constant should no longer live in the module.

    Removed when the main-first policy made the aggregator-skip guard obsolete.
    """
    import agent.auxiliary_client as aux_mod

    assert not hasattr(aux_mod, "_AGGREGATOR_PROVIDERS"), (
        "_AGGREGATOR_PROVIDERS was removed when _resolve_auto stopped "
        "treating aggregators specially. If you re-added it, the main-first "
        "policy may have regressed."
    )


# ── provider+model matched-pair resolution (mid-session route swap) ──────────


class TestResolveAutoProviderModelPairing:
    """``_resolve_auto`` must resolve (provider, model) as a MATCHED PAIR.

    Regression for the mid-session route-swap bug: the caller's live
    ``main_runtime`` carried the NEW provider (``openai-codex``) while the
    process-global ``_RUNTIME_MAIN_MODEL`` still held the OLD model
    (``claude-opus-4-8``). Two independent ``or`` fallbacks crossed them into
    a Codex route + an Opus model id, producing a hard upstream
    ``400: 'claude-opus-4-8' is not supported when using Codex``. The pair
    must come from the SAME source.
    """

    def test_stale_runtime_model_not_crossed_onto_global_provider(self):
        """A stale runtime model (no provider) must NOT cross onto the global provider.

        Faithful repro of the observed 400: after a mid-session route swap the
        process-globals correctly moved to the NEW pair (openai-codex / gpt-5.5),
        but the caller's ``main_runtime`` still carried only a STALE model
        (claude-opus-4-8) with no provider. The old two-independent-``or`` logic
        crossed the stale Opus model onto the global Codex provider →
        ``400: 'claude-opus-4-8' is not supported when using Codex``. With the
        matched-pair fix the absent runtime provider means BOTH fields come from
        the (correct) global pair.
        """
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openai-codex",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="gpt-5.5",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "gpt-5.5")

            from agent.auxiliary_client import _resolve_auto

            # Runtime dict carries ONLY the stale model, no provider.
            client, model = _resolve_auto(main_runtime={"model": "claude-opus-4-8"})

        assert client is mock_client
        # MUST resolve the global pair (codex + gpt-5.5); the stale Opus model
        # must NOT be crossed onto the Codex provider.
        assert mock_resolve.call_args.args[0] == "openai-codex"
        assert mock_resolve.call_args.args[1] == "gpt-5.5"
        assert mock_resolve.call_args.args[1] != "claude-opus-4-8"

    def test_no_runtime_provider_falls_back_to_global_pair(self):
        """With no runtime provider, BOTH provider and model come from the globals."""
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="anthropic/claude-sonnet-4.6",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-sonnet-4.6")

            from agent.auxiliary_client import _resolve_auto

            # Empty runtime dict → must use the global pair, not cross sources.
            client, model = _resolve_auto(main_runtime={})

        assert client is mock_client
        assert mock_resolve.call_args.args[0] == "openrouter"
        assert mock_resolve.call_args.args[1] == "anthropic/claude-sonnet-4.6"

    def test_runtime_provider_without_model_does_not_borrow_global_model(self):
        """A runtime provider with a blank model must NOT borrow the global model.

        Borrowing would re-introduce the cross-source pairing. With no model to
        pair to the runtime provider, Step-1 is skipped (main_model blank) and
        resolution falls through to the fallback chain — never a mismatched
        provider+model client.
        """
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="claude-pool",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus-4-8",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve, patch(
            "agent.auxiliary_client._get_provider_chain", return_value=[],
        ):
            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto(
                main_runtime={"provider": "openai-codex", "model": ""}
            )

        # Step-1 skipped (no model paired to the runtime provider); the global
        # Opus model was NOT borrowed onto the codex provider.
        mock_resolve.assert_not_called()
        assert client is None


class TestResolveProviderClientAutoRuntimeModel:
    """The outer resolver must not reintroduce stale main-config models."""

    def test_auto_provider_does_not_let_stale_config_model_override_runtime_model(self):
        """``provider=auto`` uses ``_resolve_auto``'s runtime pair, not ``_read_main_model``.

        Regression for a production compression failure after mid-session
        fallback: ``_resolve_auto(main_runtime=...)`` correctly selected the
        Codex runtime pair (openai-codex / gpt-5.5), but
        ``resolve_provider_client`` had already filled ``model`` from stale
        process/config state (claude-opus-4-8) and then returned Codex client +
        stale Opus model. Codex rejected that wire shape with HTTP 400.
        """
        codex_client = MagicMock()

        with patch(
            "agent.auxiliary_client._get_aux_model_for_provider", return_value=None,
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus-4-8",
        ), patch(
            "agent.auxiliary_client._resolve_auto", return_value=(codex_client, "gpt-5.5"),
        ) as mock_resolve_auto:
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client(
                "auto",
                None,
                False,
                main_runtime={"provider": "openai-codex", "model": "gpt-5.5"},
            )

        assert client is codex_client
        mock_resolve_auto.assert_called_once_with(
            main_runtime={"provider": "openai-codex", "model": "gpt-5.5"}
        )
        assert model == "gpt-5.5"
        assert model != "claude-opus-4-8"

    def test_auto_provider_preserves_explicit_model_override(self):
        """A real caller-supplied model still wins over the auto-resolved model."""
        codex_client = MagicMock()

        with patch(
            "agent.auxiliary_client._resolve_auto", return_value=(codex_client, "gpt-5.5"),
        ):
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client(
                "auto",
                "gpt-5.4-mini",
                False,
                main_runtime={"provider": "openai-codex", "model": "gpt-5.5"},
            )

        assert client is codex_client
        assert model == "gpt-5.4-mini"
