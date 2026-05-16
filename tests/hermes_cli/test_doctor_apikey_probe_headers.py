"""Regression tests for ``_build_apikey_probe_headers`` in ``hermes_cli.doctor``.

Covers #26623: the doctor connectivity probe used to send
``Authorization: Bearer <key>`` to every API-key provider, but Google AI
Studio (Gemini) requires ``x-goog-api-key`` and rejects Bearer with HTTP
401 — causing ``hermes doctor`` to falsely report a valid
``GOOGLE_API_KEY`` as invalid.
"""

import pytest

from hermes_cli.doctor import _build_apikey_probe_headers


GEMINI_DEFAULT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
)


class TestApikeyProbeHeaders:
    # ── Gemini detection paths ────────────────────────────────────────

    def test_gemini_detected_by_default_url(self):
        headers = _build_apikey_probe_headers(
            pname="gemini",
            base="",
            default_url=GEMINI_DEFAULT_URL,
            key="abc123",
        )
        assert headers["x-goog-api-key"] == "abc123"
        assert "Authorization" not in headers

    def test_gemini_detected_by_overridden_base_url(self):
        # User points GEMINI_BASE_URL at their own Gemini-compat proxy.
        headers = _build_apikey_probe_headers(
            pname="gemini",
            base="https://generativelanguage.googleapis.com/v1",
            default_url=None,
            key="abc123",
        )
        assert headers["x-goog-api-key"] == "abc123"
        assert "Authorization" not in headers

    @pytest.mark.parametrize(
        "pname", ["gemini", "Gemini", "GEMINI", "google", "Google", "Google / Gemini"]
    )
    def test_gemini_detected_by_name(self, pname):
        # Even when neither base nor default_url is set, the provider name
        # tells us this is Gemini and Bearer would 401.
        headers = _build_apikey_probe_headers(
            pname=pname,
            base="",
            default_url=None,
            key="abc123",
        )
        assert headers["x-goog-api-key"] == "abc123"
        assert "Authorization" not in headers

    # ── Non-Gemini providers keep Bearer ─────────────────────────────

    def test_openrouter_uses_bearer(self):
        headers = _build_apikey_probe_headers(
            pname="OpenRouter",
            base="",
            default_url="https://openrouter.ai/api/v1/models",
            key="sk-or-test",
        )
        assert headers["Authorization"] == "Bearer sk-or-test"
        assert "x-goog-api-key" not in headers

    def test_deepseek_uses_bearer(self):
        headers = _build_apikey_probe_headers(
            pname="DeepSeek",
            base="",
            default_url="https://api.deepseek.com/v1/models",
            key="sk-deepseek-test",
        )
        assert headers["Authorization"] == "Bearer sk-deepseek-test"

    def test_kimi_uses_bearer(self):
        headers = _build_apikey_probe_headers(
            pname="Kimi / Moonshot",
            base="https://api.moonshot.ai/v1",
            default_url="https://api.moonshot.ai/v1/models",
            key="sk-kimi-test",
        )
        assert headers["Authorization"] == "Bearer sk-kimi-test"

    # ── Defensive cases ──────────────────────────────────────────────

    def test_substring_match_does_not_false_positive(self):
        # ``google`` substring in a hostile host must not be classified as
        # Gemini — base_url_host_matches uses real URL parsing for this.
        headers = _build_apikey_probe_headers(
            pname="not-gemini",
            base="https://evil.com/generativelanguage.googleapis.com/v1",
            default_url=None,
            key="abc123",
        )
        assert headers["Authorization"] == "Bearer abc123"
        assert "x-goog-api-key" not in headers

    def test_empty_pname_and_no_urls_defaults_to_bearer(self):
        headers = _build_apikey_probe_headers(
            pname="",
            base="",
            default_url=None,
            key="abc123",
        )
        assert headers["Authorization"] == "Bearer abc123"

    def test_user_agent_always_set(self):
        # Both auth styles must include a User-Agent so the request isn't
        # served a generic Python-httpx UA on providers that gate on it.
        gemini = _build_apikey_probe_headers(
            pname="gemini",
            base="",
            default_url=GEMINI_DEFAULT_URL,
            key="abc",
        )
        bearer = _build_apikey_probe_headers(
            pname="DeepSeek",
            base="",
            default_url="https://api.deepseek.com/v1/models",
            key="abc",
        )
        assert gemini.get("User-Agent")
        assert bearer.get("User-Agent")
