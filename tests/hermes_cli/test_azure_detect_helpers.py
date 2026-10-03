"""Behavioral tests for the pure helper functions in hermes_cli.azure_detect.

The public `detect()` function makes live HTTP calls and is not tested here.
These tests cover the pure parsing/routing helpers that have no I/O.
"""

import pytest

from hermes_cli.azure_detect import (
    DetectionResult,
    _apply_auth_headers,
    _extract_model_ids,
    _looks_like_anthropic_path,
    _resolve_credential,
    _strip_trailing_v1,
)
from urllib import request as urllib_request


# ── _resolve_credential ───────────────────────────────────────────────────────

class TestResolveCredential:
    def test_api_key_string(self):
        token, mode = _resolve_credential("sk-test-key")
        assert token == "sk-test-key"
        assert mode == "api_key"

    def test_empty_api_key_returns_none(self):
        token, mode = _resolve_credential("")
        assert token is None
        assert mode == "api_key"

    def test_none_api_key_returns_none(self):
        token, mode = _resolve_credential(None)
        assert token is None
        assert mode == "api_key"

    def test_callable_token_provider(self):
        token, mode = _resolve_credential("ignored", token_provider=lambda: "bearer-jwt-123")
        assert token == "bearer-jwt-123"
        assert mode == "entra_id"

    def test_callable_api_key(self):
        token, mode = _resolve_credential(lambda: "from-callable")
        assert token == "from-callable"
        assert mode == "entra_id"

    def test_token_provider_wins_over_api_key_string(self):
        token, mode = _resolve_credential("sk-plain", token_provider=lambda: "entra-token")
        assert token == "entra-token"
        assert mode == "entra_id"

    def test_failing_callable_returns_none_entra(self):
        def bad_provider():
            raise RuntimeError("no token")

        token, mode = _resolve_credential("", token_provider=bad_provider)
        assert token is None
        assert mode == "entra_id"


# ── _apply_auth_headers ───────────────────────────────────────────────────────

class TestApplyAuthHeaders:
    def _new_req(self):
        return urllib_request.Request("https://example.azure.com/", method="GET")

    def test_api_key_sends_both_headers(self):
        req = self._new_req()
        _apply_auth_headers(req, "my-key", "api_key")
        assert req.get_header("Api-key") == "my-key"
        assert req.get_header("Authorization") == "Bearer my-key"

    def test_entra_id_sends_only_bearer(self):
        req = self._new_req()
        _apply_auth_headers(req, "jwt-token", "entra_id")
        assert req.get_header("Api-key") is None
        assert req.get_header("Authorization") == "Bearer jwt-token"

    def test_no_token_sets_no_headers(self):
        req = self._new_req()
        _apply_auth_headers(req, None, "api_key")
        assert req.get_header("Api-key") is None
        assert req.get_header("Authorization") is None

    def test_empty_token_sets_no_headers(self):
        req = self._new_req()
        _apply_auth_headers(req, "", "api_key")
        assert req.get_header("Api-key") is None


# ── _looks_like_anthropic_path ────────────────────────────────────────────────

class TestLooksLikeAnthropicPath:
    def test_path_ends_with_anthropic(self):
        assert _looks_like_anthropic_path(
            "https://myresource.services.ai.azure.com/models/anthropic"
        ) is True

    def test_path_contains_anthropic_segment(self):
        assert _looks_like_anthropic_path(
            "https://myresource.azure.com/anthropic/v1"
        ) is True

    def test_openai_style_url_is_false(self):
        assert _looks_like_anthropic_path(
            "https://myresource.openai.azure.com/openai/v1"
        ) is False

    def test_case_insensitive(self):
        assert _looks_like_anthropic_path(
            "https://example.azure.com/ANTHROPIC"
        ) is True

    def test_plain_host_no_path_is_false(self):
        assert _looks_like_anthropic_path("https://example.azure.com") is False

    def test_anthropic_in_hostname_only_is_false(self):
        # "anthropic" appears only in the host, not the path
        assert _looks_like_anthropic_path(
            "https://anthropic.azure.com/openai/v1"
        ) is False


# ── _strip_trailing_v1 ────────────────────────────────────────────────────────

class TestStripTrailingV1:
    def test_strips_v1_no_slash(self):
        assert _strip_trailing_v1("https://host.azure.com/openai/v1") == "https://host.azure.com/openai"

    def test_strips_v1_with_trailing_slash(self):
        assert _strip_trailing_v1("https://host.azure.com/openai/v1/") == "https://host.azure.com/openai"

    def test_no_v1_unchanged(self):
        url = "https://host.azure.com/openai"
        assert _strip_trailing_v1(url) == url

    def test_v1_mid_path_not_stripped(self):
        # /v1/ in the middle should not be stripped
        url = "https://host.azure.com/openai/v1/models"
        assert _strip_trailing_v1(url) == url


# ── _extract_model_ids ────────────────────────────────────────────────────────

class TestExtractModelIds:
    def test_standard_openai_shape(self):
        payload = {
            "data": [
                {"id": "gpt-5.4", "object": "model"},
                {"id": "gpt-5.4-mini", "object": "model"},
            ]
        }
        assert _extract_model_ids(payload) == ["gpt-5.4", "gpt-5.4-mini"]

    def test_empty_data_list(self):
        assert _extract_model_ids({"data": []}) == []

    def test_missing_data_key(self):
        assert _extract_model_ids({"object": "list"}) == []

    def test_non_dict_payload(self):
        assert _extract_model_ids([]) == []  # type: ignore[arg-type]

    def test_item_missing_id_skipped(self):
        payload = {"data": [{"object": "model"}, {"id": "good-model"}]}
        assert _extract_model_ids(payload) == ["good-model"]

    def test_falls_back_to_model_key(self):
        payload = {"data": [{"model": "fallback-slug"}]}
        assert _extract_model_ids(payload) == ["fallback-slug"]

    def test_falls_back_to_name_key(self):
        payload = {"data": [{"name": "name-slug"}]}
        assert _extract_model_ids(payload) == ["name-slug"]


# ── DetectionResult defaults ──────────────────────────────────────────────────

def test_detection_result_defaults():
    r = DetectionResult()
    assert r.api_mode is None
    assert r.models == []
    assert r.hostname == ""
    assert r.reason == ""
    assert r.models_probe_ok is False
    assert r.is_anthropic is False
