"""Tests for list_authenticated_providers handling of URL-less user providers.

Regression test for #101711.
"""

import pytest
from hermes_cli.model_switch import list_authenticated_providers


class TestListAuthenticatedProvidersNoURL:
    def test_url_less_user_provider_does_not_emit_picker_row(self):
        """A providers: entry with no base_url/api/url must NOT emit a picker row.

        Without the guard, a config-namespace-only stub like
        ``providers.custom:local.stale_timeout_seconds: 600`` claims the
        ``custom:local`` slug in section 3, and section 4 then skips the
        real ``custom_providers:`` entry — the picker ends up with an
        empty ``custom:local`` row instead of the real one.
        """
        user_providers = {
            "custom:local": {"stale_timeout_seconds": 600},
            "local": {"stale_timeout_seconds": 600},
        }
        custom_providers = [
            {
                "name": "local",
                "base_url": "http://localhost:8000/v1",
                "api_key": "not-needed",
                "models": {"unsloth/Qwen3.8-27B-NVFP4": {}},
            }
        ]

        rows = list_authenticated_providers(
            current_provider="",
            current_base_url="",
            user_providers=user_providers,
            custom_providers=custom_providers,
        )

        # The phantom URL-less row must not be present at all.
        slugs = {r["slug"] for r in rows}
        # The phantom was claiming "custom:local" with empty models.
        # With the guard, no such slug row should be emitted; the real
        # custom_providers entry's slug ("custom:local") should appear
        # populated with the actual model, not [].
        phantom_rows = [r for r in rows if r["slug"] == "custom:local"]
        assert all(r.get("models") for r in phantom_rows), (
            f"custom:local row emitted with empty models: {phantom_rows}"
        )

        # The stub keys themselves must not appear in the picker.
        assert "local" not in slugs or any(r.get("models") for r in rows if r["slug"] == "local"), (
            f"bare 'local' URL-less stub should not emit a row: {slugs}"
        )

    def test_url_ful_user_provider_still_emits_row(self):
        """Sanity check: a providers: entry WITH a base_url still works."""
        user_providers = {
            "myendpoint": {
                "base_url": "https://example.com/v1",
                "api_key": "sk-test",
                "model": "my-model",
            }
        }

        rows = list_authenticated_providers(
            current_provider="",
            current_base_url="",
            user_providers=user_providers,
            custom_providers=[],
        )

        slugs = {r["slug"] for r in rows}
        assert "myendpoint" in slugs
        ep_row = next(r for r in rows if r["slug"] == "myendpoint")
        assert "my-model" in ep_row["models"]

    def test_url_ful_user_provider_with_legacy_api_key(self):
        """The ``api`` and ``url`` fallback keys still count as a URL."""
        user_providers = {
            "altend": {
                "api": "https://alt.example.com/v1",
                "api_key": "sk-test",
            }
        }

        rows = list_authenticated_providers(
            current_provider="",
            current_base_url="",
            user_providers=user_providers,
            custom_providers=[],
        )

        slugs = {r["slug"] for r in rows}
        assert "altend" in slugs