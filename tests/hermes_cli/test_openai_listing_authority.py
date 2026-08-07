"""Live-listing authority on official OpenAI hosts (model switching).

Coatue data-residency report, issue 2: on an API key whose project only
grants a subset of models, ``/model`` accepted a curated-catalog model that
the live ``/v1/models`` listing (which is access-scoped and authoritative on
official OpenAI hosts) did not contain — manufacturing a selection that is
guaranteed to 400 at first use.

Contract: for ``openai``/``openai-api`` against an official OpenAI host
(canonical or data-residency regional), a successful live listing is
authoritative — a model absent from it is rejected, not soft-accepted from
the curated catalog. Custom OpenAI-compatible proxies and other providers
keep the #46850 curated-fallback behavior (their listings are often
incomplete).
"""

from __future__ import annotations

from unittest.mock import patch as mock_patch

import pytest

from hermes_cli import models as models_mod


def _validate(requested: str, base_url: str, live: list[str], provider: str = "openai-api"):
    with mock_patch.object(models_mod, "fetch_api_models", return_value=live):
        return models_mod.validate_requested_model(
            requested,
            provider=provider,
            api_key="sk-test",
            base_url=base_url,
        )


class TestOfficialHostAuthority:
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "https://us.api.openai.com/v1",
            "https://eu.api.openai.com/v1",
        ],
    )
    def test_absent_model_rejected_on_official_hosts(self, base_url):
        result = _validate("gpt-5.5", base_url, live=["gpt-5.6-terra"])
        assert result["accepted"] is False

    def test_present_model_accepted_on_regional_host(self):
        result = _validate("gpt-5.6-terra", "https://us.api.openai.com/v1", live=["gpt-5.6-terra"])
        assert result["accepted"] is True


class TestCuratedFallbackPreserved:
    def test_custom_proxy_keeps_curated_fallback(self):
        # gpt-5.5 is in the curated openai catalog; a custom proxy's listing
        # may be incomplete, so the #46850 soft-accept stays.
        result = _validate("gpt-5.5", "https://proxy.corp.test/v1", live=["some-other-model"])
        assert result["accepted"] is True
        assert "curated catalog" in (result.get("message") or "")
