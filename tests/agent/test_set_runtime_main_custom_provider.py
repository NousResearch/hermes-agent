"""Regression test: set_runtime_main() must pass base_url/api_key/api_mode
so that _resolve_auto() can route custom: providers in Step 1.

Fixes https://github.com/NousResearch/hermes-agent/issues/34777
"""
import pytest
from unittest.mock import patch, MagicMock


def _get_globals(mod):
    """Read runtime globals without triggering redaction."""
    return {
        "provider": mod._RUNTIME_MAIN_PROVIDER,
        "model": mod._RUNTIME_MAIN_MODEL,
        "base_url": mod._RUNTIME_MAIN_BASE_URL,
        "cred": mod._RUNTIME_MAIN_API_KEY,  # renamed to avoid redaction
        "api_mode": mod._RUNTIME_MAIN_API_MODE,
    }


class TestSetRuntimeMainCustomProvider:
    """set_runtime_main must propagate base_url/api_key/api_mode for custom providers."""

    def test_globals_stored(self):
        """set_runtime_main stores all five fields in process-local globals."""
        import agent.auxiliary_client as mod

        mod.clear_runtime_main()
        try:
            mod.set_runtime_main(
                "custom:my-router",
                "glm-5.1",
                base_url="https://my-server.example.com/v1",
                api_key="sk-test-key",
                api_mode="chat_completions",
            )
            g = _get_globals(mod)
            assert g["provider"] == "custom:my-router"
            assert g["model"] == "glm-5.1"
            assert g["base_url"] == "https://my-server.example.com/v1"
            assert g["cred"] == "sk-test-key"
            assert g["api_mode"] == "chat_completions"
        finally:
            mod.clear_runtime_main()

    def test_clear_resets_all_globals(self):
        """clear_runtime_main resets all five globals to empty."""
        import agent.auxiliary_client as mod

        mod.set_runtime_main(
            "custom:x", "m",
            base_url="https://x.example.com",
            api_key="sk-abc",
            api_mode="chat_completions",
        )
        mod.clear_runtime_main()
        g = _get_globals(mod)
        for v in g.values():
            assert v == "", f"Expected empty, got {v!r}"

    def test_resolve_auto_uses_globals_for_custom_provider(self):
        """_resolve_auto reads base_url/api_key from globals when main_runtime is None."""
        import agent.auxiliary_client as mod

        mod.clear_runtime_main()
        try:
            mod.set_runtime_main(
                "custom:test-router",
                "test-model",
                base_url="https://custom-endpoint.example.com/v1",
                api_key="sk-test-123",
            )

            with patch.object(mod, "resolve_provider_client") as mock_resolve:
                mock_resolve.return_value = (MagicMock(), "test-model")
                client, resolved = mod._resolve_auto(main_runtime=None)

                mock_resolve.assert_called_once()
                call_args = mock_resolve.call_args
                assert call_args[0][0] == "custom"
                assert call_args[1]["explicit_base_url"] == "https://custom-endpoint.example.com/v1"
                assert call_args[1]["explicit_api_key"] == "sk-test-123"
        finally:
            mod.clear_runtime_main()

    def test_explicit_main_runtime_takes_precedence(self):
        """When main_runtime dict has values, globals are NOT used."""
        import agent.auxiliary_client as mod

        mod.clear_runtime_main()
        try:
            mod.set_runtime_main(
                "custom:router-a",
                "model-a",
                base_url="https://from-global.example.com",
                api_key="sk-global",
            )

            with patch.object(mod, "resolve_provider_client") as mock_resolve:
                mock_resolve.return_value = (MagicMock(), "model-b")
                main_rt = {
                    "provider": "custom:router-b",
                    "model": "model-b",
                    "base_url": "https://from-dict.example.com",
                    "api_key": "sk-dict",
                }
                mod._resolve_auto(main_runtime=main_rt)

                call_args = mock_resolve.call_args[1]
                assert call_args["explicit_base_url"] == "https://from-dict.example.com"
                assert call_args["explicit_api_key"] == "sk-dict"
        finally:
            mod.clear_runtime_main()

    def test_backward_compatible_defaults(self):
        """Calling set_runtime_main with only positional args still works."""
        import agent.auxiliary_client as mod

        mod.clear_runtime_main()
        try:
            mod.set_runtime_main("openrouter", "gpt-4o")
            g = _get_globals(mod)
            assert g["provider"] == "openrouter"
            assert g["model"] == "gpt-4o"
            assert g["base_url"] == ""
            assert g["cred"] == ""
            assert g["api_mode"] == ""
        finally:
            mod.clear_runtime_main()


class TestResolveAutoCustomEndToEnd:
    """End-to-end routing assertions — build a *real* client (no mock on
    resolve_provider_client) and verify the auxiliary auto-detect chain lands
    on the user's custom endpoint instead of falling through to the aggregator
    chain.  These guard the actual user-visible symptom in #34777 (aux tasks
    silently routed to a fallback provider) rather than just the wiring.
    """

    @staticmethod
    def _client_base_url(client):
        for chain in (("base_url",), ("_client", "base_url")):
            obj = client
            try:
                for attr in chain:
                    obj = getattr(obj, attr)
                return str(obj)
            except AttributeError:
                continue
        return None

    def test_config_less_custom_endpoint_routes_via_global(self, tmp_path, monkeypatch):
        """custom:<name> with NO config entry: the live base_url carried by
        set_runtime_main() must build a real client at that endpoint — not
        fall through to Step 2 (the regression in #34777)."""
        import agent.auxiliary_client as mod

        # Hermetic: no aggregator creds, no stale OPENAI_BASE_URL.
        for var in ("OPENROUTER_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY",
                    "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  default: glm-5.1\n"
            "  provider: 'custom:ephemeral'\n"
            "  base_url: ''\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        mod.clear_runtime_main()
        try:
            mod.set_runtime_main(
                "custom:ephemeral",
                "glm-5.1",
                base_url="https://ephemeral.live/v1",
                api_key="sk-live",
            )
            client, resolved = mod.resolve_provider_client("auto", None)
            assert client is not None, (
                "config-less custom endpoint fell through to Step 2 — "
                "the #34777 bug is back"
            )
            assert resolved == "glm-5.1"
            base = self._client_base_url(client)
            assert base and base.rstrip("/") == "https://ephemeral.live/v1"
        finally:
            mod.clear_runtime_main()

    def test_named_custom_with_config_entry_still_routes(self, tmp_path, monkeypatch):
        """Regression guard: custom:<name> WITH a custom_providers entry must
        still resolve to that entry's endpoint.  An earlier competing fix
        collapsed the provider to bare ``custom`` before resolution, which
        broke the named-custom branch and returned None here."""
        import agent.auxiliary_client as mod

        for var in ("OPENROUTER_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY",
                    "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  default: glm-5.1\n"
            "  provider: 'custom:openclaw'\n"
            "  base_url: ''\n"
            "custom_providers:\n"
            "  - name: openclaw\n"
            "    base_url: 'https://withcfg.example/v1'\n"
            "    model: glm-5.1\n"
            "    api_key: cfg-key\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # No live base_url carried — resolution must come from config alone,
        # via the named-custom branch in resolve_provider_client.
        mod.clear_runtime_main()
        try:
            mod.set_runtime_main("custom:openclaw", "glm-5.1")
            client, resolved = mod.resolve_provider_client("auto", None)
            assert client is not None
            base = self._client_base_url(client)
            assert base and base.rstrip("/") == "https://withcfg.example/v1"
        finally:
            mod.clear_runtime_main()


class TestResolveVisionUsesRuntimeGlobals:
    """Vision auto-detect must bridge the same runtime globals as _resolve_auto.

    Regression for the sibling gap of #35259: set_runtime_main() records a
    custom:<name> main provider as bare "custom" plus base_url/api_key, but
    resolve_vision_provider_client()'s auto branch did not forward those
    credentials, so vision_analyze failed with
    "No LLM provider configured for task=vision provider=auto".
    """

    def test_vision_auto_forwards_runtime_credentials(self):
        """The vision auto branch passes explicit_base_url/api_key/api_mode
        from the runtime globals to resolve_provider_client."""
        import agent.auxiliary_client as mod

        mod.clear_runtime_main()
        try:
            mod.set_runtime_main(
                "custom",
                "claude-opus-4-7",
                base_url="https://gateway.example.com/v1",
                api_key="sk-vision-123",
                api_mode="anthropic_messages",
            )

            with patch.object(mod, "resolve_provider_client") as mock_resolve, \
                    patch.object(mod, "_main_model_supports_vision", return_value=True):
                mock_resolve.return_value = (MagicMock(), "claude-opus-4-7")
                provider, client, _model = mod.resolve_vision_provider_client(provider="auto")

                assert client is not None
                mock_resolve.assert_called_once()
                kwargs = mock_resolve.call_args[1]
                assert kwargs["explicit_base_url"] == "https://gateway.example.com/v1"
                assert kwargs["explicit_api_key"] == "sk-vision-123"
                assert kwargs["api_mode"] == "anthropic_messages"
                assert kwargs["is_vision"] is True
        finally:
            mod.clear_runtime_main()
