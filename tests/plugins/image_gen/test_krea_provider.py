#!/usr/bin/env python3
"""Tests for Krea image generation provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Ensure KREA_API_KEY is set for all tests."""
    monkeypatch.setenv("KREA_API_KEY", "test-key-12345")


def _completed_job(url: str = "https://krea.cdn/img.png") -> dict:
    return {
        "job_id": "00000000-0000-0000-0000-000000000abc",
        "status": "completed",
        "created_at": "2026-05-27T00:00:00Z",
        "completed_at": "2026-05-27T00:00:30Z",
        "result": {"urls": [url]},
    }


def _submit_response(job_id: str = "00000000-0000-0000-0000-000000000abc"):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "job_id": job_id,
        "status": "queued",
        "created_at": "2026-05-27T00:00:00Z",
        "completed_at": None,
        "result": None,
    }
    return resp


def _poll_response(body: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = body
    return resp


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestKreaImageGenProvider:
    def test_name(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        assert KreaImageGenProvider().name == "krea"

    def test_display_name(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        assert KreaImageGenProvider().display_name == "Krea"

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("KREA_API_KEY", "sk-test")
        from plugins.image_gen.krea import KreaImageGenProvider

        assert KreaImageGenProvider().is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("KREA_API_KEY", raising=False)
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        # No direct key AND no managed gateway → unavailable.
        monkeypatch.setattr(krea_mod, "_managed_krea_gateway_ready", lambda: False)
        assert KreaImageGenProvider().is_available() is False

    def test_is_available_via_managed_gateway_without_key(self, monkeypatch):
        monkeypatch.delenv("KREA_API_KEY", raising=False)
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        # No direct key but the managed Nous gateway is ready → available.
        monkeypatch.setattr(krea_mod, "_managed_krea_gateway_ready", lambda: True)
        assert KreaImageGenProvider().is_available() is True

    def test_list_models(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        models = KreaImageGenProvider().list_models()
        ids = {m["id"] for m in models}
        assert {"krea-2-medium", "krea-2-large"} <= ids
        # Each entry carries the picker fields the registry expects.
        for m in models:
            assert m["display"]
            assert m["speed"]
            assert m["strengths"]
            assert m["price"]

    def test_default_model_is_medium(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        assert KreaImageGenProvider().default_model() == "krea-2-medium"

    def test_get_setup_schema(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        schema = KreaImageGenProvider().get_setup_schema()
        assert schema["name"] == "Krea"
        assert schema["badge"] == "paid"
        env_vars = schema["env_vars"]
        assert len(env_vars) == 1
        assert env_vars[0]["key"] == "KREA_API_KEY"
        assert "krea.ai" in env_vars[0]["url"]


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_default(self):
        from plugins.image_gen.krea import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "krea-2-medium"
        assert meta["path"] == "medium"

    def test_env_override_large(self, monkeypatch):
        monkeypatch.setenv("KREA_IMAGE_MODEL", "krea-2-large")
        from plugins.image_gen.krea import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "krea-2-large"
        assert meta["path"] == "large"

    def test_env_override_unknown_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("KREA_IMAGE_MODEL", "krea-2-xxl-fake")
        from plugins.image_gen.krea import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "krea-2-medium"

    def test_creativity_default(self):
        from plugins.image_gen.krea import _resolve_creativity

        assert _resolve_creativity(None) == "medium"

    def test_creativity_valid(self):
        from plugins.image_gen.krea import _resolve_creativity

        assert _resolve_creativity("HIGH") == "high"
        assert _resolve_creativity(" raw ") == "raw"

    def test_creativity_invalid(self):
        from plugins.image_gen.krea import _resolve_creativity

        assert _resolve_creativity("ultra") == "medium"


# ---------------------------------------------------------------------------
# Generate — main flow
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("KREA_API_KEY", raising=False)
        from plugins.image_gen.krea import KreaImageGenProvider

        result = KreaImageGenProvider().generate(prompt="test")
        assert result["success"] is False
        assert "KREA_API_KEY" in result["error"]
        assert result["error_type"] == "auth_required"

    def test_empty_prompt(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        result = KreaImageGenProvider().generate(prompt="   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_successful_generation(self):
        """Happy path: submit → one poll → completed → URL downloaded."""
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job("https://krea.cdn/result.png"))

        with patch("plugins.image_gen.krea.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", return_value=poll) as mock_get, \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/krea_krea-2-medium_test.png"),
             ) as mock_save, \
             patch("plugins.image_gen.krea.time.sleep"):  # skip real waits
            result = KreaImageGenProvider().generate(prompt="A cinematic lamp")

        assert result["success"] is True
        assert result["image"] == "/tmp/krea_krea-2-medium_test.png"
        assert result["provider"] == "krea"
        assert result["model"] == "krea-2-medium"
        assert result["aspect_ratio"] == "landscape"
        assert result["job_id"] == "00000000-0000-0000-0000-000000000abc"
        assert result["resolution"] == "1K"
        assert result["creativity"] == "medium"
        # Submit hit the medium endpoint
        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/generate/image/krea/krea-2/medium")
        # Poll hit /jobs/{job_id}
        poll_url = mock_get.call_args[0][0]
        assert "/jobs/00000000-0000-0000-0000-000000000abc" in poll_url
        # URL was materialised once
        mock_save.assert_called_once()

    def test_large_model_routes_to_large_endpoint(self, monkeypatch):
        monkeypatch.setenv("KREA_IMAGE_MODEL", "krea-2-large")
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch("plugins.image_gen.krea.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            KreaImageGenProvider().generate(prompt="test")

        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/generate/image/krea/krea-2/large")

    def test_aspect_ratio_mapping(self):
        """Hermes 'square' must map to Krea '1:1' in the wire payload."""
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch("plugins.image_gen.krea.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            KreaImageGenProvider().generate(prompt="test", aspect_ratio="square")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["aspect_ratio"] == "1:1"
        assert payload["resolution"] == "1K"

    def test_auth_header(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch("plugins.image_gen.krea.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            KreaImageGenProvider().generate(prompt="test")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key-12345"
        assert headers["Content-Type"] == "application/json"

    def test_passthrough_seed_styles_moodboards(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch("plugins.image_gen.krea.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            KreaImageGenProvider().generate(
                prompt="test",
                seed=42,
                styles=[{"id": "lora-1", "strength": 0.7}],
                moodboards=[{"url": "https://x.com/mood.png"}, {"url": "https://x.com/mood2.png"}],
                image_style_references=[{"url": f"https://x.com/{i}.png"} for i in range(15)],
                creativity="high",
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["seed"] == 42
        assert payload["styles"] == [{"id": "lora-1", "strength": 0.7}]
        assert len(payload["moodboards"]) == 1  # capped at 1
        assert len(payload["image_style_references"]) == 10  # capped at 10
        assert payload["creativity"] == "high"

    def test_string_style_references_converted_to_objects(self):
        """Krea requires {url, strength} objects; bare URL strings must be
        converted (a string yields a 422 'Expected object, received string')."""
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch("plugins.image_gen.krea.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            KreaImageGenProvider().generate(
                prompt="test",
                image_style_references=[
                    "https://x.com/a.png",
                    {"url": "https://x.com/b.png", "strength": 1.2},
                ],
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["image_style_references"] == [
            {"url": "https://x.com/a.png", "strength": 0.6},
            {"url": "https://x.com/b.png", "strength": 1.2},
        ]

    def test_unknown_kwargs_ignored(self):
        """Forward-compat: unknown kwargs must not break generate()."""
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch("plugins.image_gen.krea.requests.post", return_value=submit), \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(
                prompt="test",
                fictional_param="should be ignored",
                num_images=4,
            )

        assert result["success"] is True


# ---------------------------------------------------------------------------
# Generate — error paths
# ---------------------------------------------------------------------------


class TestGenerateErrors:
    def test_submit_http_error(self):
        import requests as req_lib
        from plugins.image_gen.krea import KreaImageGenProvider

        resp = req_lib.Response()
        resp.status_code = 401
        resp._content = b'{"error": {"message": "Invalid API key"}}'
        resp.headers["Content-Type"] = "application/json"
        resp.raise_for_status = MagicMock(
            side_effect=req_lib.HTTPError(response=resp)
        )

        with patch("plugins.image_gen.krea.requests.post", return_value=resp):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]
        assert "Invalid API key" in result["error"]

    def test_submit_timeout(self):
        import requests as req_lib
        from plugins.image_gen.krea import KreaImageGenProvider

        with patch(
            "plugins.image_gen.krea.requests.post", side_effect=req_lib.Timeout()
        ):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_submit_connection_error(self):
        import requests as req_lib
        from plugins.image_gen.krea import KreaImageGenProvider

        with patch(
            "plugins.image_gen.krea.requests.post",
            side_effect=req_lib.ConnectionError("dns nope"),
        ):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "connection_error"

    def test_submit_missing_job_id(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        bad_submit = MagicMock()
        bad_submit.status_code = 200
        bad_submit.raise_for_status = MagicMock()
        bad_submit.json.return_value = {"status": "queued"}

        with patch("plugins.image_gen.krea.requests.post", return_value=bad_submit):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "invalid_response"
        assert "job_id" in result["error"]

    def test_job_failed(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        failed = {
            "job_id": "abc",
            "status": "failed",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {"error": "NSFW content"},
        }

        submit = _submit_response()
        with patch("plugins.image_gen.krea.requests.post", return_value=submit), \
             patch(
                 "plugins.image_gen.krea.requests.get",
                 return_value=_poll_response(failed),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "NSFW" in result["error"]

    def test_job_cancelled(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        cancelled = {
            "job_id": "abc",
            "status": "cancelled",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {},
        }

        with patch("plugins.image_gen.krea.requests.post", return_value=_submit_response()), \
             patch(
                 "plugins.image_gen.krea.requests.get",
                 return_value=_poll_response(cancelled),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "cancelled"

    def test_completed_but_missing_urls(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        completed_empty = {
            "job_id": "abc",
            "status": "completed",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {"urls": []},
        }

        with patch("plugins.image_gen.krea.requests.post", return_value=_submit_response()), \
             patch(
                 "plugins.image_gen.krea.requests.get",
                 return_value=_poll_response(completed_empty),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_url_download_failure_falls_back_to_bare_url(self):
        """Mirror of xAI behaviour — if local cache fails, return the URL."""
        import requests as req_lib
        from plugins.image_gen.krea import KreaImageGenProvider

        url = "https://krea.cdn/expired-soon.png"
        submit = _submit_response()
        poll = _poll_response(_completed_job(url))

        with patch("plugins.image_gen.krea.requests.post", return_value=submit), \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 side_effect=req_lib.HTTPError("404"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is True
        assert result["image"] == url

    def test_polling_picks_up_completed_at_with_unknown_status(self):
        """``completed_at`` set + unrecognised pending status → still terminal."""
        from plugins.image_gen.krea import KreaImageGenProvider

        # Use a status value that is NOT in our terminal set ("intermediate-complete")
        # but with completed_at populated — Krea's spec says completed_at is the
        # canonical terminal marker.
        oddball = {
            "job_id": "abc",
            "status": "intermediate-complete",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {"urls": ["https://krea.cdn/done.png"]},
        }

        with patch("plugins.image_gen.krea.requests.post", return_value=_submit_response()), \
             patch(
                 "plugins.image_gen.krea.requests.get",
                 return_value=_poll_response(oddball),
             ), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is True


class TestPollRetryPolicy:
    """Polling fail-fast on permanent 4xx, retry on transient 5xx/429."""

    def _http_error_response(self, status: int):
        import requests as req_lib

        resp = req_lib.Response()
        resp.status_code = status
        resp._content = b'{"error": "boom"}'
        resp.headers["Content-Type"] = "application/json"
        resp.raise_for_status = MagicMock(
            side_effect=req_lib.HTTPError(response=resp)
        )
        return resp

    def test_poll_fails_fast_on_401(self):
        """Auth failure mid-poll should not wait the 180s deadline."""
        from plugins.image_gen.krea import KreaImageGenProvider

        bad_poll = self._http_error_response(401)

        with patch("plugins.image_gen.krea.requests.post", return_value=_submit_response()), \
             patch("plugins.image_gen.krea.requests.get", return_value=bad_poll) as mock_get, \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]
        # One call — no retry on permanent auth failure.
        assert mock_get.call_count == 1

    def test_poll_fails_fast_on_404(self):
        """Missing job (404) should surface immediately, not retry for 180s."""
        from plugins.image_gen.krea import KreaImageGenProvider

        bad_poll = self._http_error_response(404)

        with patch("plugins.image_gen.krea.requests.post", return_value=_submit_response()), \
             patch("plugins.image_gen.krea.requests.get", return_value=bad_poll) as mock_get, \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "404" in result["error"]
        assert mock_get.call_count == 1

    def test_poll_fails_fast_on_403(self):
        """Billing/permission failure (403) should not retry."""
        from plugins.image_gen.krea import KreaImageGenProvider

        bad_poll = self._http_error_response(403)

        with patch("plugins.image_gen.krea.requests.post", return_value=_submit_response()), \
             patch("plugins.image_gen.krea.requests.get", return_value=bad_poll) as mock_get, \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert mock_get.call_count == 1

    def test_poll_retries_on_503_then_succeeds(self):
        """Transient 5xx should retry and eventually surface a completion."""
        from plugins.image_gen.krea import KreaImageGenProvider

        flaky = self._http_error_response(503)
        good = _poll_response(_completed_job("https://krea.cdn/ok.png"))

        with patch("plugins.image_gen.krea.requests.post", return_value=_submit_response()), \
             patch(
                 "plugins.image_gen.krea.requests.get",
                 side_effect=[flaky, flaky, good],
             ) as mock_get, \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is True
        assert mock_get.call_count == 3

    def test_poll_retries_on_429(self):
        """Rate-limit (429) is in the retryable set."""
        from plugins.image_gen.krea import KreaImageGenProvider

        rate_limited = self._http_error_response(429)
        good = _poll_response(_completed_job("https://krea.cdn/ok.png"))

        with patch("plugins.image_gen.krea.requests.post", return_value=_submit_response()), \
             patch(
                 "plugins.image_gen.krea.requests.get",
                 side_effect=[rate_limited, good],
             ) as mock_get, \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is True
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Managed Nous gateway path
# ---------------------------------------------------------------------------


def _managed_cfg(
    origin: str = "https://krea-gateway.example.com",
    token: str = "nous-tok-abc",
):
    from types import SimpleNamespace

    return SimpleNamespace(
        vendor="krea",
        gateway_origin=origin,
        nous_user_token=token,
        managed_mode=True,
    )


class TestManagedGateway:
    def test_managed_submit_uses_gateway_origin_and_nous_token(self, monkeypatch):
        """Managed mode submits to the gateway origin with the Nous token."""
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        # Even with a direct key present, an active managed gateway wins.
        monkeypatch.setattr(krea_mod, "_resolve_managed_krea_gateway", lambda: _managed_cfg())

        submit = _submit_response()
        poll = _poll_response(_completed_job())
        with patch("plugins.image_gen.krea.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", return_value=poll) as mock_get, \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="A managed lamp")

        assert result["success"] is True
        post_url = mock_post.call_args[0][0]
        assert post_url == (
            "https://krea-gateway.example.com/generate/image/krea/krea-2/medium"
        )
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer nous-tok-abc"
        # Idempotency key drives the gateway's per-generation billing boundary.
        assert headers["x-idempotency-key"]
        # Poll is bound to the same gateway + Nous token.
        poll_url = mock_get.call_args[0][0]
        assert poll_url.startswith("https://krea-gateway.example.com/jobs/")
        poll_headers = mock_get.call_args.kwargs["headers"]
        assert poll_headers["Authorization"] == "Bearer nous-tok-abc"

    def test_managed_available_without_direct_key(self, monkeypatch):
        """No KREA_API_KEY but an active gateway → generate proceeds (no auth_required)."""
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        monkeypatch.delenv("KREA_API_KEY", raising=False)
        monkeypatch.setattr(krea_mod, "_resolve_managed_krea_gateway", lambda: _managed_cfg())

        submit = _submit_response()
        poll = _poll_response(_completed_job())
        with patch("plugins.image_gen.krea.requests.post", return_value=submit), \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is True

    def test_managed_4xx_returns_actionable_remediation(self, monkeypatch):
        import requests as req_lib
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        monkeypatch.setattr(krea_mod, "_resolve_managed_krea_gateway", lambda: _managed_cfg())

        resp = req_lib.Response()
        resp.status_code = 402
        resp._content = b'{"error": {"message": "out of credits"}}'
        resp.headers["Content-Type"] = "application/json"
        resp.raise_for_status = MagicMock(side_effect=req_lib.HTTPError(response=resp))

        with patch("plugins.image_gen.krea.requests.post", return_value=resp):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "402" in result["error"]
        assert "Nous Subscription Krea gateway" in result["error"]
        assert "KREA_API_KEY" in result["error"]

    def test_managed_429_concurrency_hint(self, monkeypatch):
        import requests as req_lib
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        monkeypatch.setattr(krea_mod, "_resolve_managed_krea_gateway", lambda: _managed_cfg())

        resp = req_lib.Response()
        resp.status_code = 429
        resp._content = b'{"error": {"message": "maximum number of concurrent jobs"}}'
        resp.headers["Content-Type"] = "application/json"
        resp.raise_for_status = MagicMock(side_effect=req_lib.HTTPError(response=resp))

        with patch("plugins.image_gen.krea.requests.post", return_value=resp):
            result = KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert "429" in result["error"]
        assert "concurrency" in result["error"].lower()

    def test_managed_blocks_styles(self, monkeypatch):
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        monkeypatch.setattr(krea_mod, "_resolve_managed_krea_gateway", lambda: _managed_cfg())

        with patch("plugins.image_gen.krea.requests.post") as mock_post:
            result = KreaImageGenProvider().generate(
                prompt="test",
                styles=[{"id": "lora-1"}],
            )

        assert result["success"] is False
        assert result["error_type"] == "unsupported_argument"
        assert "LoRA" in result["error"] or "styles" in result["error"]
        # Never hit the network with an unsupported tier.
        mock_post.assert_not_called()

    def test_managed_blocks_moodboards(self, monkeypatch):
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        monkeypatch.setattr(krea_mod, "_resolve_managed_krea_gateway", lambda: _managed_cfg())

        with patch("plugins.image_gen.krea.requests.post") as mock_post:
            result = KreaImageGenProvider().generate(
                prompt="test",
                moodboards=[{"url": "https://x.com/m.png"}],
            )

        assert result["success"] is False
        assert result["error_type"] == "unsupported_argument"
        assert "moodboard" in result["error"].lower()
        mock_post.assert_not_called()


class TestExplicitModelOverride:
    def test_model_kwarg_overrides_config(self, monkeypatch):
        """An explicit ``model`` kwarg (managed routing) wins over config/default."""
        from plugins.image_gen.krea import _resolve_model

        model_id, meta = _resolve_model("krea-2-large")
        assert model_id == "krea-2-large"
        assert meta["path"] == "large"

    def test_turbo_routes_to_medium_turbo_endpoint(self):
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())
        with patch("plugins.image_gen.krea.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(prompt="test", model="krea-2-medium-turbo")

        assert result["success"] is True
        assert result["model"] == "krea-2-medium-turbo"
        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/generate/image/krea/krea-2/medium-turbo")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register(self):
        from plugins.image_gen.krea import KreaImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_image_gen_provider.assert_called_once()
        provider = mock_ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, KreaImageGenProvider)
        assert provider.name == "krea"
