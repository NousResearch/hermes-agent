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
