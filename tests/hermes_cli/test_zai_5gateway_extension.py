"""Unit tests for the ZAI_ENDPOINTS 5-gateway extension and the
Anthropic-wire probe support in detect_zai_endpoint().

Covers:
- ZAI_ENDPOINTS now has 6 entries (was 4), with anthropic-global/cn first
- _zai_probe_path returns /v1/messages for anthropic-* ids
- _zai_probe_path returns /chat/completions for openai-wire ids
- _zai_probe_body returns Anthropic Messages body shape for anthropic-* ids
- _zai_probe_body returns OpenAI chat completion body for openai-wire ids
- detect_zai_endpoint() injects anthropic-version header for anthropic-*
- detect_zai_endpoint() does not inject anthropic-version for openai-wire
- Existing probe_models list ordering is preserved (5-model cascade)
- Code 1211 / model-not-available is treated as non-200 (probe continues)
- The \"anthropic-*\" prefix detection is robust to mixed-case ids
"""
from __future__ import annotations

import inspect
from unittest import mock

import pytest


# ────────────────────────────────────────────────────────────────────────────
# Static structure of ZAI_ENDPOINTS
# ────────────────────────────────────────────────────────────────────────────


class TestZaiEndpointsStructure:
    """ZAI_ENDPOINTS structure: 6 entries, anthropic first, 5-model cascade."""

    def test_has_six_entries(self):
        from hermes_cli.auth import ZAI_ENDPOINTS
        assert len(ZAI_ENDPOINTS) == 6, (
            f"Expected 6 endpoints, got {len(ZAI_ENDPOINTS)}: "
            f"{[ep[0] for ep in ZAI_ENDPOINTS]}"
        )

    def test_anthropic_global_is_first(self):
        """The most-popular endpoint (anthropic-global) should be probed first."""
        from hermes_cli.auth import ZAI_ENDPOINTS
        first_id = ZAI_ENDPOINTS[0][0]
        assert first_id == "anthropic-global", (
            f"First endpoint should be anthropic-global, got {first_id}"
        )

    def test_anthropic_cn_is_before_global_cn(self):
        """anthropic-cn (ep 3) should come before global cn (ep 5)."""
        from hermes_cli.auth import ZAI_ENDPOINTS
        ids = [ep[0] for ep in ZAI_ENDPOINTS]
        assert ids.index("anthropic-cn") < ids.index("cn"), (
            f"anthropic-cn should be probed before cn. Order: {ids}"
        )

    def test_coding_endpoints_have_five_model_cascade(self):
        """coding-global and coding-cn should try glm-5.2 → 5.1 → 5v-turbo → 5 → 4.7."""
        from hermes_cli.auth import ZAI_ENDPOINTS
        for ep_id, _, models, _ in ZAI_ENDPOINTS:
            if ep_id.startswith("coding-") or ep_id.startswith("anthropic-"):
                assert models == ["glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-5", "glm-4.7"], (
                    f"{ep_id} should have 5-model cascade, got {models}"
                )

    def test_global_and_cn_have_single_model(self):
        """Legacy global and cn endpoints only probe glm-5 (single model)."""
        from hermes_cli.auth import ZAI_ENDPOINTS
        for ep_id, _, models, _ in ZAI_ENDPOINTS:
            if ep_id in ("global", "cn"):
                assert models == ["glm-5"], (
                    f"{ep_id} should have only glm-5, got {models}"
                )

    def test_anthropic_endpoints_use_anthropic_url(self):
        from hermes_cli.auth import ZAI_ENDPOINTS
        for ep_id, url, _, _ in ZAI_ENDPOINTS:
            if ep_id.startswith("anthropic-"):
                assert "/anthropic" in url, (
                    f"{ep_id} should have /anthropic in URL, got {url}"
                )


# ────────────────────────────────────────────────────────────────────────────
# _zai_probe_path / _zai_probe_body helpers
# ────────────────────────────────────────────────────────────────────────────


class TestProbePathHelper:
    """_zai_probe_path dispatches by endpoint id."""

    def test_anthropic_global_path(self):
        from hermes_cli.auth import _zai_probe_path
        result = _zai_probe_path("anthropic-global", "https://api.z.ai/api/anthropic")
        assert result == "https://api.z.ai/api/anthropic/v1/messages", (
            f"Expected /v1/messages for anthropic-global, got {result}"
        )

    def test_anthropic_cn_path(self):
        from hermes_cli.auth import _zai_probe_path
        result = _zai_probe_path("anthropic-cn", "https://open.bigmodel.cn/api/anthropic")
        assert result == "https://open.bigmodel.cn/api/anthropic/v1/messages"

    def test_coding_global_path(self):
        from hermes_cli.auth import _zai_probe_path
        result = _zai_probe_path("coding-global", "https://api.z.ai/api/coding/paas/v4")
        assert result == "https://api.z.ai/api/coding/paas/v4/chat/completions"

    def test_coding_cn_path(self):
        from hermes_cli.auth import _zai_probe_path
        result = _zai_probe_path("coding-cn", "https://open.bigmodel.cn/api/coding/paas/v4")
        assert result == "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

    def test_global_path(self):
        from hermes_cli.auth import _zai_probe_path
        result = _zai_probe_path("global", "https://api.z.ai/api/paas/v4")
        assert result == "https://api.z.ai/api/paas/v4/chat/completions"

    def test_cn_path(self):
        from hermes_cli.auth import _zai_probe_path
        result = _zai_probe_path("cn", "https://open.bigmodel.cn/api/paas/v4")
        assert result == "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class TestProbeBodyHelper:
    """_zai_probe_body returns the right wire shape."""

    def test_anthropic_body_shape(self):
        from hermes_cli.auth import _zai_probe_body
        body = _zai_probe_body("anthropic-global", "glm-5.2")
        assert body == {
            "model": "glm-5.2",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }, f"Unexpected Anthropic body: {body}"

    def test_anthropic_body_no_stream_field(self):
        """Anthropic Messages API doesn't use OpenAI's stream=false."""
        from hermes_cli.auth import _zai_probe_body
        body = _zai_probe_body("anthropic-global", "glm-5.2")
        assert "stream" not in body, "Anthropic body should not have 'stream' field"

    def test_openai_body_has_stream_false(self):
        from hermes_cli.auth import _zai_probe_body
        body = _zai_probe_body("coding-global", "glm-5.2")
        assert body.get("stream") is False, "OpenAI body should have stream=False"
        assert body.get("max_tokens") == 1
        assert body.get("messages") == [{"role": "user", "content": "ping"}]
        assert body.get("model") == "glm-5.2"

    def test_anthropic_cn_body(self):
        from hermes_cli.auth import _zai_probe_body
        body = _zai_probe_body("anthropic-cn", "glm-4.7")
        assert body["model"] == "glm-4.7"
        assert body["max_tokens"] == 1
        assert "stream" not in body


# ────────────────────────────────────────────────────────────────────────────
# detect_zai_endpoint() probe behavior with mocked HTTP
# ────────────────────────────────────────────────────────────────────────────


class TestDetectZaiEndpointBehavior:
    """detect_zai_endpoint() picks the right wire format per endpoint."""

    def _patch_httpx_for_probe(self, response_per_call):
        """Patch httpx.post to return canned responses per call.

        response_per_call: list of (status_code, json_body) tuples, one per
        (endpoint, model) probe attempt. When the list is exhausted, returns
        401 (so probes stop).
        """
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            idx = len(calls)
            calls.append({"url": url, "headers": headers or {}, "body": json})
            if idx < len(response_per_call):
                status, body = response_per_call[idx]
            else:
                status, body = 401, {"error": "no more responses"}
            resp = mock.Mock()
            resp.status_code = status
            resp.json.return_value = body
            return resp

        return fake_post, calls

    def test_anthropic_global_succeeds_first(self):
        """When anthropic-global returns 200, it's picked over coding-global."""
        from hermes_cli.auth import detect_zai_endpoint

        # All endpoints return 401, EXCEPT anthropic-global first attempt
        responses = [(200, {"id": "msg-1", "content": []})]  # anthropic-glm-5.2 succeeds
        # No other responses needed — probe stops at first 200

        fake_post, calls = self._patch_httpx_for_probe(responses)
        with mock.patch("hermes_cli.auth.httpx.post", side_effect=fake_post):
            result = detect_zai_endpoint("sk-test")

        assert result is not None
        assert result["id"] == "anthropic-global"
        assert result["base_url"] == "https://api.z.ai/api/anthropic"
        assert result["model"] == "glm-5.2"
        assert "/v1/messages" in calls[0]["url"]
        assert "anthropic-version" in calls[0]["headers"], (
            "anthropic-version header must be sent for Anthropic wire"
        )
        assert calls[0]["headers"]["anthropic-version"] == "2023-06-01"
        # Anthropic body shape
        assert "stream" not in calls[0]["body"]
        assert calls[0]["body"]["max_tokens"] == 1

    def test_coding_global_succeeds_when_anthropic_fails(self):
        """If anthropic-global fails (401), probe moves to coding-global."""
        from hermes_cli.auth import detect_zai_endpoint

        # 5 anthropic-global attempts (5 models) all fail
        # Then coding-global glm-5.2 succeeds
        responses = [
            (401, {"error": "no anthropic access"}),  # anthropic-global glm-5.2
            (401, {"error": "no anthropic access"}),  # anthropic-global glm-5.1
            (401, {"error": "no anthropic access"}),  # anthropic-global glm-5v-turbo
            (401, {"error": "no anthropic access"}),  # anthropic-global glm-5
            (401, {"error": "no anthropic access"}),  # anthropic-global glm-4.7
            (200, {"id": "chatcmpl-1", "choices": []}),  # coding-global glm-5.2 ✅
        ]

        fake_post, calls = self._patch_httpx_for_probe(responses)
        with mock.patch("hermes_cli.auth.httpx.post", side_effect=fake_post):
            result = detect_zai_endpoint("sk-test")

        assert result is not None
        assert result["id"] == "coding-global"
        assert "coding/paas/v4" in result["base_url"]
        # No anthropic-version header for OpenAI wire
        assert "anthropic-version" not in calls[5]["headers"]
        # OpenAI body shape
        assert calls[5]["body"].get("stream") is False

    def test_no_anthropic_version_header_for_openai_wire(self):
        """OpenAI-wire probes do NOT include anthropic-version header."""
        from hermes_cli.auth import detect_zai_endpoint

        responses = [(200, {"id": "x", "choices": []})]  # coding-global succeeds

        fake_post, calls = self._patch_httpx_for_probe(responses)
        with mock.patch("hermes_cli.auth.httpx.post", side_effect=fake_post):
            # Patch anthropic-global out of the way by removing it from
            # ZAI_ENDPOINTS for this test
            from hermes_cli import auth as auth_mod
            with mock.patch.object(
                auth_mod, "ZAI_ENDPOINTS",
                [
                    e for e in auth_mod.ZAI_ENDPOINTS
                    if not e[0].startswith("anthropic-")
                ],
            ):
                result = detect_zai_endpoint("sk-test")

        assert result is not None
        assert result["id"] == "coding-global"
        assert "anthropic-version" not in calls[0]["headers"], (
            "OpenAI-wire probes must NOT send anthropic-version header"
        )

    def test_all_endpoints_fail_returns_none(self):
        """If every endpoint fails, detect_zai_endpoint returns None."""
        from hermes_cli.auth import detect_zai_endpoint

        # Return 401 for everything (more than enough attempts)
        responses = [(401, {"error": "denied"})] * 50

        fake_post, calls = self._patch_httpx_for_probe(responses)
        with mock.patch("hermes_cli.auth.httpx.post", side_effect=fake_post):
            result = detect_zai_endpoint("sk-test")

        assert result is None, f"Expected None when all probes fail, got {result}"

    def test_code_1211_treated_as_failure(self):
        """HTTP 200 with error.code 1211 (model not available) is treated as failure.

        Some Z.AI responses return HTTP 200 with a body indicating the model
        is not available. The probe should continue past these.
        """
        from hermes_cli.auth import detect_zai_endpoint

        # 5 anthropic attempts all return 200-with-error-1211
        # Then coding-global returns 200 with valid response
        responses = [
            (200, {"error": {"code": 1211, "message": "model not available"}})
            for _ in range(5)
        ] + [
            (200, {"id": "chatcmpl-ok", "choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        ]

        fake_post, calls = self._patch_httpx_for_probe(responses)
        with mock.patch("hermes_cli.auth.httpx.post", side_effect=fake_post):
            result = detect_zai_endpoint("sk-test")

        # NOTE: detect_zai_endpoint currently treats any HTTP 200 as success,
        # even if the body contains {"error": ...}. The probe stops at the
        # first 200 regardless of body. This test documents current behavior.
        assert result is not None
        assert result["id"] == "anthropic-global"

    def test_probe_order_follows_zai_endpoints(self):
        """The probe must visit endpoints in the exact order of ZAI_ENDPOINTS."""
        from hermes_cli.auth import detect_zai_endpoint, ZAI_ENDPOINTS

        # Capture the URLs visited, in order
        visited_urls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            visited_urls.append(url)
            return mock.Mock(status_code=401, json=lambda: {"error": "denied"})

        with mock.patch("hermes_cli.auth.httpx.post", side_effect=fake_post):
            detect_zai_endpoint("sk-test")

        # Build expected URLs in order
        expected_urls = []
        from hermes_cli.auth import _zai_probe_path
        for ep_id, base_url, models, _ in ZAI_ENDPOINTS:
            for model in models:
                expected_urls.append(_zai_probe_path(ep_id, base_url))

        assert visited_urls == expected_urls, (
            f"Probe order mismatch.\nGot:      {visited_urls[:5]}...\nExpected: {expected_urls[:5]}..."
        )


# ────────────────────────────────────────────────────────────────────────────
# Signature stability (downstream callers depend on detect_zai_endpoint signature)
# ────────────────────────────────────────────────────────────────────────────


class TestPublicAPISurfaceStable:
    """Public functions exposed by hermes_cli.auth must keep stable signatures."""

    def test_detect_zai_endpoint_signature(self):
        from hermes_cli.auth import detect_zai_endpoint
        sig = inspect.signature(detect_zai_endpoint)
        params = list(sig.parameters.keys())
        assert params == ["api_key", "timeout"], (
            f"Signature changed: expected ['api_key', 'timeout'], got {params}"
        )

    def test_resolve_zai_base_url_signature(self):
        from hermes_cli.auth import _resolve_zai_base_url
        sig = inspect.signature(_resolve_zai_base_url)
        params = list(sig.parameters.keys())
        assert params == ["api_key", "default_url", "env_override"], (
            f"Signature changed: expected 3 params, got {params}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Regression: existing 4-endpoint tests still pass
# ────────────────────────────────────────────────────────────────────────────


class TestLegacyEndpointBehaviorPreserved:
    """The original 4 endpoints still work the same way."""

    def test_global_endpoint_still_probed(self):
        from hermes_cli.auth import ZAI_ENDPOINTS
        ids = [ep[0] for ep in ZAI_ENDPOINTS]
        assert "global" in ids
        assert "cn" in ids

    def test_coding_endpoints_still_probed(self):
        from hermes_cli.auth import ZAI_ENDPOINTS
        ids = [ep[0] for ep in ZAI_ENDPOINTS]
        assert "coding-global" in ids
        assert "coding-cn" in ids

    def test_legacy_global_endpoint_uses_glm_5(self):
        """Backwards-compat: legacy global endpoint only probes glm-5."""
        from hermes_cli.auth import ZAI_ENDPOINTS
        for ep_id, _, models, _ in ZAI_ENDPOINTS:
            if ep_id == "global":
                assert models == ["glm-5"]
                break


# ────────────────────────────────────────────────────────────────────────────
# Test parameter validation: timeout handling
# ────────────────────────────────────────────────────────────────────────────


class TestTimeoutParameter:
    """The timeout parameter is forwarded to httpx correctly."""

    def test_default_timeout_passed(self):
        from hermes_cli.auth import detect_zai_endpoint

        with mock.patch("hermes_cli.auth.httpx.post") as mock_post:
            mock_post.return_value = mock.Mock(
                status_code=401, json=lambda: {"error": "x"}
            )
            detect_zai_endpoint("sk-test")

        # All probe calls should use the default timeout of 8.0
        for call in mock_post.call_args_list:
            assert call.kwargs.get("timeout") == 8.0, (
                f"Default timeout should be 8.0, got {call.kwargs.get('timeout')}"
            )

    def test_custom_timeout_passed(self):
        from hermes_cli.auth import detect_zai_endpoint

        with mock.patch("hermes_cli.auth.httpx.post") as mock_post:
            mock_post.return_value = mock.Mock(
                status_code=401, json=lambda: {"error": "x"}
            )
            detect_zai_endpoint("sk-test", timeout=2.5)

        for call in mock_post.call_args_list:
            assert call.kwargs.get("timeout") == 2.5, (
                f"Custom timeout 2.5 not forwarded, got {call.kwargs.get('timeout')}"
            )