"""Regression tests for dict entries in provider models lists.

Issue #57405: Model selector crashes with
'``dict`` object has no attribute ``lower``' when a provider's
``models:`` config is a list of dicts (e.g. ``[{id: "gpt-4"}]``)
instead of a flat list of strings.

The models list is consumed by :func:`build_models_payload` which calls
``m.lower()`` on every entry for deduplication.  If a dict slips through,
the Desktop model selector and CLI ``hermes model`` both crash.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# model_switch: list_authenticated_providers
# ---------------------------------------------------------------------------


class TestListAuthenticatedProvidersDictModels:
    """Section 3 (providers:) and Section 4 (custom_providers:) must
    extract the ``id`` field from dict entries instead of appending
    the raw dict to the models list."""

    def test_providers_dict_with_list_of_dicts(self) -> None:
        """providers: models is a list of dicts — extract id strings."""
        from hermes_cli.model_switch import list_authenticated_providers

        user_providers = {
            "my-proxy": {
                "base_url": "https://proxy.example.com/v1",
                "models": [
                    {"id": "model-alpha", "context_length": 128_000},
                    {"id": "model-beta", "context_length": 16_384},
                ],
            }
        }

        rows = list_authenticated_providers(
            current_provider="my-proxy",
            current_model="model-alpha",
            current_base_url="https://proxy.example.com/v1",
            user_providers=user_providers,
            custom_providers=[],
        )

        proxy_rows = [r for r in rows if r.get("slug") == "my-proxy"]
        assert proxy_rows, f"my-proxy not found in rows: {[r['slug'] for r in rows]}"
        models = proxy_rows[0]["models"]
        assert all(isinstance(m, str) for m in models), (
            f"Non-string model entries: {[type(m).__name__ for m in models]}"
        )
        assert "model-alpha" in models
        assert "model-beta" in models

    def test_providers_dict_with_string_list(self) -> None:
        """providers: models is a flat list of strings — still works."""
        from hermes_cli.model_switch import list_authenticated_providers

        user_providers = {
            "my-proxy": {
                "base_url": "https://proxy.example.com/v1",
                "models": ["model-x", "model-y"],
            }
        }

        rows = list_authenticated_providers(
            current_provider="my-proxy",
            current_model="model-x",
            current_base_url="https://proxy.example.com/v1",
            user_providers=user_providers,
            custom_providers=[],
        )

        proxy_rows = [r for r in rows if r.get("slug") == "my-proxy"]
        assert proxy_rows
        models = proxy_rows[0]["models"]
        assert "model-x" in models
        assert "model-y" in models

    def test_providers_dict_with_keyed_dict(self) -> None:
        """providers: models is a dict keyed by model id — still works."""
        from hermes_cli.model_switch import list_authenticated_providers

        user_providers = {
            "my-proxy": {
                "base_url": "https://proxy.example.com/v1",
                "models": {
                    "model-p": {"context_length": 128_000},
                    "model-q": {"context_length": 16_384},
                },
            }
        }

        rows = list_authenticated_providers(
            current_provider="my-proxy",
            current_model="model-p",
            current_base_url="https://proxy.example.com/v1",
            user_providers=user_providers,
            custom_providers=[],
        )

        proxy_rows = [r for r in rows if r.get("slug") == "my-proxy"]
        assert proxy_rows
        models = proxy_rows[0]["models"]
        assert "model-p" in models
        assert "model-q" in models

    def test_custom_providers_list_of_dicts(self) -> None:
        """custom_providers: models is a list of dicts — extract id strings."""
        from hermes_cli.model_switch import list_authenticated_providers

        custom_providers = [
            {
                "name": "My Ollama",
                "base_url": "http://localhost:11434/v1",
                "models": [
                    {"id": "llama-4", "context_length": 8192},
                    {"id": "qwen-3", "context_length": 32768},
                ],
            }
        ]

        rows = list_authenticated_providers(
            current_provider="",
            current_model="",
            current_base_url="",
            user_providers={},
            custom_providers=custom_providers,
        )

        ollama_rows = [r for r in rows if "ollama" in r.get("slug", "").lower()]
        assert ollama_rows, f"No ollama row found: {[r['slug'] for r in rows]}"
        models = ollama_rows[0]["models"]
        assert all(isinstance(m, str) for m in models), (
            f"Non-string model entries: {[type(m).__name__ for m in models]}"
        )
        assert "llama-4" in models
        assert "qwen-3" in models

    def test_dict_entries_without_id_fall_through(self) -> None:
        """Dicts without an 'id' key are silently skipped."""
        from hermes_cli.model_switch import list_authenticated_providers

        user_providers = {
            "my-proxy": {
                "base_url": "https://proxy.example.com/v1",
                "models": [
                    {"name": "no-id-field"},
                    "valid-model",
                ],
            }
        }

        rows = list_authenticated_providers(
            current_provider="my-proxy",
            current_model="valid-model",
            current_base_url="https://proxy.example.com/v1",
            user_providers=user_providers,
            custom_providers=[],
        )

        proxy_rows = [r for r in rows if r.get("slug") == "my-proxy"]
        assert proxy_rows
        models = proxy_rows[0]["models"]
        assert all(isinstance(m, str) for m in models)
        assert "valid-model" in models
        # The dict without 'id' should be skipped (not crash, not leak)
        assert len(models) == 1


# ---------------------------------------------------------------------------
# inventory: build_models_payload dedup
# ---------------------------------------------------------------------------


class TestBuildModelsPayloadDictModels:
    """build_models_payload dedup must not crash when models contain dicts."""

    def test_build_payload_with_dict_models(self) -> None:
        """End-to-end: build_models_payload survives dict entries."""
        from hermes_cli.inventory import build_models_payload, load_picker_context

        ctx = load_picker_context()
        # Should not raise AttributeError on .lower()
        payload = build_models_payload(
            ctx,
            include_unconfigured=True,
            picker_hints=True,
            canonical_order=True,
        )
        assert "providers" in payload
        for provider in payload["providers"]:
            for m in provider.get("models", []):
                assert isinstance(m, str), (
                    f"Non-string model in {provider['slug']}: "
                    f"{type(m).__name__} = {repr(m)}"
                )
