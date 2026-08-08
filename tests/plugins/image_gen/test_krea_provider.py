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

    def test_env_override_large(self, monkeypatch):
        monkeypatch.setenv("KREA_IMAGE_MODEL", "krea-2-large")
        from plugins.image_gen.krea import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "krea-2-large"
        assert meta["path"] == "large"


    def test_creativity_default(self):
        from plugins.image_gen.krea import _resolve_creativity

        assert _resolve_creativity(None) == "medium"


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
            result = KreaImageGenProvider().generate(prompt="A cinematic lamp", upscale=False)

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
            KreaImageGenProvider().generate(prompt="test", aspect_ratio="square", upscale=False)

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
                upscale=False,
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
                upscale=False,
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
            result = KreaImageGenProvider().generate(prompt="A managed lamp", upscale=False)

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
            result = KreaImageGenProvider().generate(prompt="test", model="krea-2-medium-turbo", upscale=False)

        assert result["success"] is True
        assert result["model"] == "krea-2-medium-turbo"
        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/generate/image/krea/krea-2/medium-turbo")


# ---------------------------------------------------------------------------
# Upscale pass (Krea Enhance)
# ---------------------------------------------------------------------------


class TestUpscalePass:
    def _run_generate(self, *, upscale, enhance_job, model=None):
        """Drive generate() with sequenced post/get mocks.

        Sequence: generation submit POST → generation poll GET; then (when
        upscale fires) enhance submit POST → enhance poll GET.
        """
        from plugins.image_gen.krea import KreaImageGenProvider

        gen_submit = _submit_response()
        gen_poll = _poll_response(_completed_job("https://krea.cdn/native.png"))
        enh_submit = _submit_response("00000000-0000-0000-0000-00000000e0e0")
        enh_poll = _poll_response(enhance_job) if enhance_job else None

        posts = [gen_submit, enh_submit]
        gets = [gen_poll] + ([enh_poll] if enh_poll else [])

        kwargs = {"prompt": "a lamp", "upscale": upscale}
        if model is not None:
            kwargs["model"] = model

        with patch("plugins.image_gen.krea.requests.post", side_effect=posts) as mock_post, \
             patch("plugins.image_gen.krea.requests.get", side_effect=gets) as mock_get, \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 side_effect=lambda url, prefix: Path(f"/tmp/{url.rsplit('/', 1)[-1]}"),
             ), \
             patch("plugins.image_gen.krea.time.sleep"):
            result = KreaImageGenProvider().generate(**kwargs)
        return result, mock_post, mock_get

    def test_upscale_routes_through_enhance_endpoint(self):
        enhance_job = {
            "job_id": "00000000-0000-0000-0000-00000000e0e0",
            "status": "completed",
            "created_at": "2026-05-27T00:00:00Z",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {"urls": ["https://krea.cdn/enhanced.png"]},
        }
        result, mock_post, _ = self._run_generate(upscale=True, enhance_job=enhance_job)

        assert result["success"] is True
        assert result["upscaled"] is True
        assert result["upscale_factor"] == 2
        assert result["image"].endswith("enhanced.png")
        # Second POST hit the Enhance endpoint with the native image + factor.
        assert mock_post.call_count == 2
        enh_url = mock_post.call_args_list[1][0][0]
        assert enh_url.endswith("/generate/enhance/krea/enhance")
        enh_payload = mock_post.call_args_list[1].kwargs["json"]
        assert enh_payload["image_url"] == "https://krea.cdn/native.png"
        assert enh_payload["image_scaling_factor"] == 2
        assert enh_payload["prompt"] == "a lamp"

    def test_upscale_failure_falls_back_to_native(self):
        failed_job = {
            "job_id": "00000000-0000-0000-0000-00000000e0e0",
            "status": "failed",
            "created_at": "2026-05-27T00:00:00Z",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": None,
        }
        result, mock_post, _ = self._run_generate(upscale=True, enhance_job=failed_job)

        assert result["success"] is True
        assert result["upscaled"] is False
        assert result["image"].endswith("native.png")
        assert mock_post.call_count == 2  # enhance attempted, fell back

    def test_medium_upscales_by_default(self):
        """krea-2-medium is 1.5K native — the Enhance pass defaults ON."""
        enhance_job = {
            "job_id": "00000000-0000-0000-0000-00000000e0e0",
            "status": "completed",
            "created_at": "2026-05-27T00:00:00Z",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {"urls": ["https://krea.cdn/enhanced.png"]},
        }
        result, mock_post, _ = self._run_generate(upscale=None, enhance_job=enhance_job)

        assert result["success"] is True
        assert result["upscaled"] is True
        assert result["image"].endswith("enhanced.png")
        assert mock_post.call_count == 2

    def test_large_skips_upscale_by_default(self):
        """krea-2-large is 2K native — no automatic Enhance pass."""
        result, mock_post, _ = self._run_generate(
            upscale=None, enhance_job=None, model="krea-2-large",
        )

        assert result["success"] is True
        assert result["upscaled"] is False
        assert result["image"].endswith("native.png")
        assert mock_post.call_count == 1  # only the generation submit

    def test_explicit_false_disables_default(self):
        """Explicit upscale=False wins over medium's default-on."""
        result, mock_post, _ = self._run_generate(upscale=False, enhance_job=None)

        assert result["success"] is True
        assert result["upscaled"] is False
        assert result["image"].endswith("native.png")
        assert mock_post.call_count == 1


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
