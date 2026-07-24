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

    def test_moa_main_resolves_aux_to_aggregator(self, monkeypatch, tmp_path):
        """MoA main user → aux runs on the aggregator slot, NOT the preset name.

        provider='moa'/model='opus-gpt' would otherwise send the preset name
        'opus-gpt' as the model id and 400 ("not a valid model ID"). Aux tasks
        don't need the reference fan-out — they use the aggregator (the preset's
        acting model). The virtual moa://local base_url + placeholder key must
        be dropped so the aggregator resolves via its own provider credentials.
        """
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "moa": {
                        "default_preset": "opus-gpt",
                        "presets": {
                            "opus-gpt": {
                                "enabled": True,
                                "reference_models": [{"provider": "openrouter", "model": "openai/gpt-5.5"}],
                                "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                            }
                        },
                    }
                }
            )
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        with patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve, patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-opus-4.8")

            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto(
                main_runtime={
                    "provider": "moa",
                    "model": "opus-gpt",
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                },
                task="title_generation",
            )

        assert client is mock_client
        # Resolved to the aggregator's real provider+model, not the preset name.
        assert mock_resolve.call_args.args[0] == "openrouter"
        assert mock_resolve.call_args.args[1] == "anthropic/claude-opus-4.8"
        # The virtual moa://local endpoint must not be forwarded as the
        # aggregator's base_url.
        assert mock_resolve.call_args.kwargs.get("explicit_base_url") in (None, "")

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

    def test_main_unavailable_uses_task_fallback_chain_before_builtin_chain(self):
        """Auto aux resolution honors auxiliary.<task>.fallback_chain before built-ins."""
        task_client = MagicMock()
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nvidia",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="qwen/qwen3.5-122b-a10b",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),  # main provider has no client
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(task_client, "task-free-model", "fallback_chain[0](openrouter)"),
        ) as mock_task_chain, patch(
            "agent.auxiliary_client._try_main_fallback_chain",
        ) as mock_main_chain, patch(
            "agent.auxiliary_client._try_openrouter",
        ) as mock_openrouter:
            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto(task="title_generation")

        assert client is task_client
        assert model == "task-free-model"
        mock_task_chain.assert_called_once_with(
            "title_generation", "nvidia", reason="main provider unavailable")
        mock_main_chain.assert_not_called()
        mock_openrouter.assert_not_called()

    def test_main_unavailable_uses_main_fallback_chain_before_builtin_chain(self):
        """Auto aux resolution honors top-level fallback_providers before built-ins."""
        main_fallback_client = MagicMock()
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nvidia",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="qwen/qwen3.5-122b-a10b",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),  # main provider has no client
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ), patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            return_value=(main_fallback_client, "inclusionai/ring-2.6-1t:free", "openrouter"),
        ) as mock_main_chain, patch(
            "agent.auxiliary_client._try_openrouter",
        ) as mock_openrouter:
            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto(task="title_generation")

        assert client is main_fallback_client
        assert model == "inclusionai/ring-2.6-1t:free"
        mock_main_chain.assert_called_once_with(
            "title_generation", "nvidia", reason="main provider unavailable")
        mock_openrouter.assert_not_called()

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

    def test_resolve_provider_auto_returns_runtime_model_not_stale_config_default(self):
        """Blank auto aux requests must not pair a stale config model with live fallback provider."""
        runtime_client = MagicMock()
        with patch(
            "agent.auxiliary_client._read_main_model",
            return_value="claude-opus-4-8",
        ) as mock_read_main_model, patch(
            "agent.auxiliary_client._resolve_auto",
            return_value=(runtime_client, "gpt-5.5"),
        ) as mock_resolve_auto:
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client(
                "auto",
                main_runtime={
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "base_url": "",
                    "api_key": "",
                    "api_mode": "codex_responses",
                },
            )

        assert client is runtime_client
        assert model == "gpt-5.5"
        mock_read_main_model.assert_not_called()
        mock_resolve_auto.assert_called_once()

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


class TestResolveAutoCustomProviderNoRuntime:
    """Bare `model.provider: custom` + `model.base_url` with no live agent
    runtime and no set_runtime_main() override (e.g. _resolve_auto() called
    from a standalone script/tool context — background review, an approval
    classifier, a CLI utility, ...) must still resolve Step 1 from config.

    Regression: main_provider/main_model already fell back to
    _read_main_provider()/_read_main_model() here, but runtime_base_url/
    runtime_api_key had no equivalent config.yaml fallback — only an explicit
    main_runtime dict or a process-local _RUNTIME_MAIN_* override populated
    them. A bare-custom main (a local llama.cpp/vLLM/Ollama/LM Studio server
    configured directly in model.base_url, not via a named custom_providers
    entry) resolved main_provider="custom" correctly but left
    explicit_base_url=None, so resolve_provider_client("custom", ...,
    explicit_base_url=None) could never produce a client — Step 1 silently
    failed and fell through to the Step 3 chain (OpenRouter -> Nous ->
    local/custom -> api-key), which has no way to discover a bare-custom
    endpoint either, so every auxiliary task (compression, session_search,
    smart/Auto Mode approval, ...) failed with "no provider available" on
    this class of setup. Mirrors TestResolveVisionCustomProvider's
    test_custom_main_no_runtime_falls_back_to_configured_endpoint, which
    already worked correctly for vision via _resolve_custom_runtime().
    """

    def test_bare_custom_main_no_runtime_resolves_from_config(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="custom",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="Qwen3.5-4B-UD-Q4_K_XL.gguf",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="http://127.0.0.1:8090/v1",
        ), patch(
            "agent.auxiliary_client._read_main_api_key", return_value="",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "Qwen3.5-4B-UD-Q4_K_XL.gguf")

            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto()

        assert client is mock_client
        assert model == "Qwen3.5-4B-UD-Q4_K_XL.gguf"
        assert mock_resolve.call_args.args[0] == "custom"
        assert mock_resolve.call_args.kwargs.get("explicit_base_url") == "http://127.0.0.1:8090/v1"

    def test_custom_prefixed_main_no_runtime_also_resolves_from_config(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "")

        with patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="custom:my-local-server",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="local-model",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="http://127.0.0.1:9000/v1",
        ), patch(
            "agent.auxiliary_client._read_main_api_key", return_value="",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "local-model")

            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto()

        assert client is mock_client
        assert mock_resolve.call_args.kwargs.get("explicit_base_url") == "http://127.0.0.1:9000/v1"

    def test_live_runtime_base_url_still_wins_over_config(self, monkeypatch):
        """An explicit main_runtime dict must NOT be overridden by the new
        config fallback — the live runtime is always more authoritative."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="custom",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="config-model",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="http://127.0.0.1:9999/v1",
        ) as mock_read_base_url, patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "live-model")

            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto(main_runtime={
                "provider": "custom",
                "model": "live-model",
                "base_url": "https://live.example/v1",
                "api_key": "sk-live",
            })

        assert mock_resolve.call_args.kwargs.get("explicit_base_url") == "https://live.example/v1"
        mock_read_base_url.assert_not_called()

    def test_non_custom_provider_unaffected(self, monkeypatch):
        """The new fallback must only engage for provider == custom(:*) — a
        non-custom main with no runtime must behave exactly as before."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="anthropic/claude-opus-4.8",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
        ) as mock_read_base_url, patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-opus-4.8")

            from agent.auxiliary_client import _resolve_auto

            client, model = _resolve_auto()

        assert client is mock_client
        assert mock_resolve.call_args.kwargs.get("explicit_base_url") is None
        mock_read_base_url.assert_not_called()


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


# ── Vision — custom provider endpoint credential passthrough ────────────────


class TestResolveVisionCustomProvider:
    """Custom-endpoint mains must forward base_url/api_key to Step 1.

    Regression: a ``custom:<name>`` main provider resolves to the bare
    runtime provider id ``"custom"``.  ``resolve_provider_client("custom")``
    has no built-in endpoint, so without forwarding the live base_url/api_key
    it returns ``(None, None)`` and vision falls through to OpenRouter / Nous,
    which an offline / aggregator-less user has never configured — breaking
    vision entirely with ``No LLM provider configured for task=vision
    provider=auto``.  The fix recovers the live endpoint that
    ``set_runtime_main()`` recorded for the turn.
    """

    def test_custom_main_forwards_runtime_endpoint(self, monkeypatch):
        """custom main with recorded runtime endpoint → Step 1 builds a client."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "https://my.endpoint.example/v1")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "sk-runtime-key")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "anthropic_messages")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="custom",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus-4-8",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "claude-opus-4-8")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "custom"
        assert client is mock_client
        assert model == "claude-opus-4-8"
        # The endpoint credentials recorded for the turn MUST be forwarded,
        # otherwise resolve_provider_client("custom") returns (None, None).
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs.get("explicit_base_url") == "https://my.endpoint.example/v1"
        assert kwargs.get("explicit_api_key") == "sk-runtime-key"
        assert kwargs.get("is_vision") is True

    def test_custom_prefixed_main_forwards_runtime_endpoint(self, monkeypatch):
        """A ``custom:<name>`` provider id also forwards the runtime endpoint."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "https://named.example/v1")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "sk-named")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "")

        with patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="custom:copilot-gateway",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus-4-8",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "claude-opus-4-8")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "custom:copilot-gateway"
        assert client is mock_client
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs.get("explicit_base_url") == "https://named.example/v1"
        assert kwargs.get("explicit_api_key") == "sk-named"
        assert kwargs.get("is_vision") is True

    def test_custom_main_no_runtime_falls_back_to_configured_endpoint(self, monkeypatch):
        """No recorded runtime endpoint → resolve the configured custom endpoint."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="custom",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus-4-8",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._resolve_custom_runtime",
            return_value=("https://configured.example/v1", "sk-configured", "chat_completions"),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "claude-opus-4-8")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert client is mock_client
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs.get("explicit_base_url") == "https://configured.example/v1"
        assert kwargs.get("explicit_api_key") == "sk-configured"


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
