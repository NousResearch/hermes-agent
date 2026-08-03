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




    @staticmethod
    def _stub_nous_portal(seen: dict):
        """Stub the Nous network boundary, keeping the resolution chain real.

        Returns a ``_try_nous`` replacement that answers with the Portal's
        tier-aware slots: a vision model for ``vision=True``, the text chat
        default otherwise.
        """
        nous_client = MagicMock()
        nous_client.api_key = "jwt-test"
        nous_client.base_url = "https://inference-api.nousresearch.com/v1"

        def fake_try_nous(vision=False):
            seen["vision"] = vision
            return nous_client, (
                "stepfun/step-3.7-flash:free" if vision else "tencent/hy3:free"
            )

        return nous_client, fake_try_nous

    def test_nous_main_vision_uses_portal_pick_not_text_chat_model(self):
        """Nous main → vision runs the Portal's vision slot, not the chat model.

        A Nous chat default is routinely text-only (e.g. a ``:free`` chat SKU).
        Letting it reach the vision lane means the image goes to a model that
        cannot accept one and the Portal 404s. Only the Nous network boundary
        is stubbed — the strict vision backend, the provider router, and its
        missing-model pre-fill all run for real, because that pre-fill is where
        the chat model used to clobber the Portal's pick.
        """
        seen: dict = {}
        nous_client, fake_try_nous = self._stub_nous_portal(seen)

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="tencent/hy3:free",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._try_nous", side_effect=fake_try_nous,
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "nous"
        assert client is nous_client
        assert seen["vision"] is True
        assert model == "stepfun/step-3.7-flash:free"

    def test_nous_main_vision_honours_explicit_vision_model(self):
        """An explicit auxiliary.vision.model still overrides the Portal pick."""
        seen: dict = {}
        _nous_client, fake_try_nous = self._stub_nous_portal(seen)

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="tencent/hy3:free",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "qwen/qwen3-vl-8b-instruct", None, None, None),
        ), patch(
            "agent.auxiliary_client._try_nous", side_effect=fake_try_nous,
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, _client, model = resolve_vision_provider_client()

        assert provider == "nous"
        assert model == "qwen/qwen3-vl-8b-instruct"

    def test_nous_explicit_vision_provider_also_skips_chat_model(self):
        """``auxiliary.vision.provider: nous`` takes the same Portal pick.

        The explicit-provider branch reaches the strict vision backend with no
        model too, so it has to resolve the same way the auto branch does.
        """
        seen: dict = {}
        nous_client, fake_try_nous = self._stub_nous_portal(seen)

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="tencent/hy3:free",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("nous", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._try_nous", side_effect=fake_try_nous,
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "nous"
        assert client is nous_client
        assert model == "stepfun/step-3.7-flash:free"

    def test_nous_text_aux_still_uses_main_chat_model(self):
        """The vision carve-out must not leak into text aux resolution.

        Text auxiliary work on a Nous main deliberately keeps the user's chat
        model rather than dropping to the Portal's cheap default.
        """
        seen: dict = {}
        _nous_client, fake_try_nous = self._stub_nous_portal(seen)

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="tencent/hy3:free",
        ), patch(
            "agent.auxiliary_client._try_nous", side_effect=fake_try_nous,
        ):
            from agent.auxiliary_client import resolve_provider_client

            _client, model = resolve_provider_client("nous")

        assert model == "tencent/hy3:free"

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
