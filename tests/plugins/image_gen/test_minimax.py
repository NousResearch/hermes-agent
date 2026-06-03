"""Tests for the MiniMax Text-to-Image plugin in plugins/image_gen/minimax/.

Covers:
- ABC contract (name, is_available, list_models, get_setup_schema)
- Aspect ratio mapping (landscape/square/portrait → 16:9/1:1/9:16)
- Payload construction (model, prompt, response_format, n, etc.)
- API error handling (4xx, 5xx, base_resp status_code != 0, empty response)
- Base64 response decoding via save_b64_image
- URL response fallback (defensive, with cache helper)
- Prompt truncation at the 1500-char cap
- Prompt validation (empty prompt rejected)
- API key validation (auth_required)
- Auth header is Bearer

Mirrors the structure of tests/tools/test_tts_minimax.py.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Wipe the key env var so tests don't pick up the developer's key."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("MINIMAX_IMAGE_BASE_URL", raising=False)


@pytest.fixture
def minimax_api_key(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    return "test-minimax-key"


@pytest.fixture
def mock_requests_post():
    """Patch requests.post — the plugin does ``import requests`` at module
    level, so we patch the module-level ``requests.post``."""
    with patch("requests.post") as post:
        yield post


def _mock_response(json_body=None, raw_bytes=None, status_code=200):
    """Build a MagicMock stand-in for a requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if json_body is not None:
        resp.json.return_value = json_body
        resp.text = ""
    else:
        resp.json.side_effect = ValueError("not json")
        resp.text = raw_bytes.decode("utf-8", errors="ignore") if raw_bytes else ""
    resp.content = raw_bytes or b""
    resp.raise_for_status = MagicMock()
    # iter_content is what save_url_image uses — yield the raw bytes in
    # one chunk so the file lands on disk correctly when the test
    # simulates a URL response.
    if raw_bytes:
        resp.iter_content = MagicMock(return_value=iter([raw_bytes]))
    else:
        resp.iter_content = MagicMock(return_value=iter([b""]))
    # Content-Type default for URL helper (save_url_image inspects it)
    resp.headers = {"Content-Type": "image/jpeg" if raw_bytes else "application/json"}
    return resp


def _success_b64_body(b64_payload: str = "aGVsbG8td29ybGQ=") -> dict:
    """Build a MiniMax success response with one base64 image."""
    return {
        "id": "req-abc-123",
        "data": {"image_base64": [b64_payload]},
        "metadata": {"success_count": "1", "failed_count": "0"},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


class TestMiniMaxImageGenProviderContract:
    def test_name_is_minimax(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.name == "minimax"

    def test_display_name(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.display_name == "MiniMax"

    def test_is_available_false_when_key_missing(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.is_available() is False

    def test_is_available_true_when_key_present(self, minimax_api_key):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.is_available() is True

    def test_list_models_returns_image_01(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        models = provider.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "image-01"
        assert "display" in models[0]
        assert "speed" in models[0]

    def test_default_model_is_image_01(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.default_model() == "image-01"

    def test_get_setup_schema_declares_api_key(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        schema = provider.get_setup_schema()
        assert schema["name"] == "MiniMax"
        assert schema["badge"] == "paid"
        env_keys = [v["key"] for v in schema["env_vars"]]
        assert "MINIMAX_API_KEY" in env_keys


# ---------------------------------------------------------------------------
# Aspect ratio mapping
# ---------------------------------------------------------------------------


class TestAspectRatioMapping:
    @pytest.mark.parametrize(
        "hermes_aspect, minimax_aspect",
        [
            ("landscape", "16:9"),
            ("square", "1:1"),
            ("portrait", "9:16"),
        ],
    )
    def test_canonical_aspect_ratios(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post,
        hermes_aspect, minimax_aspect
    ):
        """The 3-value Hermes enum must map to the documented MiniMax
        ratios — landscape→16:9, square→1:1, portrait→9:16."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate("A test", aspect_ratio=hermes_aspect)

        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["aspect_ratio"] == minimax_aspect

    def test_invalid_aspect_falls_back_to_landscape(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate("A test", aspect_ratio="bogus")

        # resolve_aspect_ratio coerces invalid values to "landscape"
        # (the ABC default), then we map landscape → 16:9.
        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["aspect_ratio"] == "16:9"


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


class TestPayloadConstruction:
    def test_default_payload_uses_base64_response_format(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        """We always request base64 to dodge the 24h URL expiry."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate("A test")

        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["response_format"] == "base64"

    def test_payload_includes_model_prompt_n(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate("Hello world")

        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["model"] == "image-01"
        assert payload["prompt"] == "Hello world"
        assert payload["n"] == 1

    def test_auth_header_is_bearer(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate("test")

        headers = mock_requests_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-minimax-key"
        assert headers["Content-Type"] == "application/json"

    def test_endpoint_url(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider, DEFAULT_BASE_URL

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate("test")

        called_url = mock_requests_post.call_args[0][0]
        assert called_url == DEFAULT_BASE_URL

    def test_optional_seed_forwarded(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate("test", seed=42)

        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["seed"] == 42

    def test_optional_prompt_optimizer_forwarded(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate("test", prompt_optimizer=True)

        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["prompt_optimizer"] is True

    def test_base_url_override_via_kwarg(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        provider.generate(
            "test",
            base_url="https://proxy.example.com/v1/image_generation",
        )

        called_url = mock_requests_post.call_args[0][0]
        assert called_url == "https://proxy.example.com/v1/image_generation"


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_prompt_returns_invalid_argument(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        result = provider.generate("")

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        # The dispatch must never have been attempted
        mock_requests_post.assert_not_called()

    def test_whitespace_only_prompt_treated_as_empty(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        result = provider.generate("   \n  ")

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key_returns_auth_required(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        # Note: NOT setting minimax_api_key fixture
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is False
        assert result["error_type"] == "auth_required"
        assert "MINIMAX_API_KEY" in result["error"]
        mock_requests_post.assert_not_called()

    def test_prompt_truncated_to_max_length(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        """Prompts longer than 1500 chars are clamped. The API rejects
        longer inputs, so we surface the truncation in the response
        extra field for transparency."""
        from plugins.image_gen.minimax import (
            MAX_PROMPT_LENGTH,
            MiniMaxImageGenProvider,
        )

        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body()
        )
        provider = MiniMaxImageGenProvider()
        long_prompt = "x" * 2000
        result = provider.generate(long_prompt)

        payload = mock_requests_post.call_args.kwargs["json"]
        assert len(payload["prompt"]) == MAX_PROMPT_LENGTH
        # The success path records the truncation in the extra field
        assert result["success"] is True
        assert result.get("prompt_truncated") is True


# ---------------------------------------------------------------------------
# API error handling
# ---------------------------------------------------------------------------


class TestApiErrorHandling:
    def test_http_4xx_returns_api_error(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            status_code=401, raw_bytes=b"Unauthorized"
        )
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]

    def test_http_5xx_returns_api_error(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            status_code=500, raw_bytes=b"Internal Server Error"
        )
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "500" in result["error"]

    def test_base_resp_status_code_nonzero_returns_api_error(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        """MiniMax uses a base_resp envelope; status_code != 0 is a
        logical failure (e.g. insufficient balance → 1004)."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body={
                "id": "req-xyz",
                "data": {},
                "base_resp": {
                    "status_code": 1004,
                    "status_msg": "insufficient balance",
                },
            }
        )
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "1004" in result["error"]
        assert "insufficient balance" in result["error"]

    def test_non_json_response_returns_api_error(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            status_code=200, raw_bytes=b"<html>some html</html>"
        )
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "non-JSON" in result["error"]

    def test_empty_data_returns_empty_response(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        """Success envelope but no image_base64 / image_urls — usually a
        content-policy block or upstream glitch."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.return_value = _mock_response(
            json_body={
                "id": "req-empty",
                "data": {},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_network_error_returns_network_error(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        import requests

        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_requests_post.side_effect = requests.ConnectionError("boom")
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is False
        assert result["error_type"] == "network_error"


# ---------------------------------------------------------------------------
# Success path: b64 response
# ---------------------------------------------------------------------------


class TestB64Response:
    def test_b64_response_saves_to_cache(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        # 1x1 red PNG (base64-encoded) so save_b64_image has real bytes
        red_dot_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlE"
            "QRVR4nGP8//8/AwAI/AL+XJ/PIgAAAABJRU5ErkJggg=="
        )
        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body(red_dot_b64)
        )
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is True
        assert result["provider"] == "minimax"
        assert result["model"] == "image-01"
        assert result["prompt"] == "test"
        # image should be a file path under HERMES_HOME/cache/images/
        assert result["image"] and "/" in result["image"]
        assert result["image"].endswith(".png")
        # The file should exist on disk and contain the decoded bytes
        with open(result["image"], "rb") as f:
            decoded = f.read()
        assert decoded == base64.b64decode(red_dot_b64)

    def test_b64_response_records_metadata_in_extra(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        red_dot_b64 = "aGVsbG8="
        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body(red_dot_b64)
        )
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test", aspect_ratio="square")

        assert result["success"] is True
        assert result["minimax_aspect_ratio"] == "1:1"
        assert result["request_id"] == "req-abc-123"
        assert result["metadata"]["success_count"] == "1"
        assert result["aspect_ratio"] == "square"


# ---------------------------------------------------------------------------
# Success path: URL response (defensive — we always send base64)
# ---------------------------------------------------------------------------


class TestUrlResponseFallback:
    def test_url_response_caches_locally(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        """If the API ever returns image_urls instead of image_base64,
        we must cache the bytes locally — the URL expires in 24h."""
        import requests as real_requests

        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        url_body = {
            "id": "req-url",
            "data": {"image_urls": ["https://cdn.example.com/img.jpeg"]},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_requests_post.return_value = _mock_response(json_body=url_body)
        # Stub the URL fetch (the image_gen_provider.save_url_image
        # helper does its own GET)
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        with patch.object(
            real_requests, "get", return_value=_mock_response(
                status_code=200, raw_bytes=fake_png
            )
        ):
            provider = MiniMaxImageGenProvider()
            result = provider.generate("test")

        assert result["success"] is True
        # Should be a local cached path, not the bare 24h-expire URL
        assert result["image"].endswith(".jpg") or result["image"].endswith(".png")
        assert "cdn.example.com" not in result["image"]


# ---------------------------------------------------------------------------
# Plugin registration surface
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    def test_register_wires_provider(self):
        """register(ctx) must call ctx.register_image_gen_provider with
        an instance whose name is 'minimax'."""
        from plugins.image_gen.minimax import (
            MiniMaxImageGenProvider,
            register,
        )

        ctx = MagicMock()
        register(ctx)
        assert ctx.register_image_gen_provider.call_count == 1
        provider = ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, MiniMaxImageGenProvider)
        assert provider.name == "minimax"


# ---------------------------------------------------------------------------
# Magic-byte sniffer
# ---------------------------------------------------------------------------


class TestSniffImageExtension:
    def test_jpeg_magic(self):
        from plugins.image_gen.minimax import _sniff_image_extension

        # JPEG: FF D8 FF
        assert _sniff_image_extension(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "jpg"

    def test_png_magic(self):
        from plugins.image_gen.minimax import _sniff_image_extension

        # PNG: 89 50 4E 47 0D 0A 1A 0A
        assert _sniff_image_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) == "png"

    def test_webp_magic(self):
        from plugins.image_gen.minimax import _sniff_image_extension

        # RIFF + 4 bytes + WEBP
        assert (
            _sniff_image_extension(b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"VP8 ")
            == "webp"
        )

    def test_gif87a_magic(self):
        from plugins.image_gen.minimax import _sniff_image_extension

        assert _sniff_image_extension(b"GIF87a" + b"\x00" * 32) == "gif"

    def test_gif89a_magic(self):
        from plugins.image_gen.minimax import _sniff_image_extension

        assert _sniff_image_extension(b"GIF89a" + b"\x00" * 32) == "gif"

    def test_unknown_format_falls_back_to_png(self):
        from plugins.image_gen.minimax import _sniff_image_extension

        # BMP / TIFF / etc. — fall back to png to match save_b64_image's default
        assert _sniff_image_extension(b"BM" + b"\x00" * 32) == "png"

    def test_empty_bytes_falls_back_to_png(self):
        from plugins.image_gen.minimax import _sniff_image_extension

        assert _sniff_image_extension(b"") == "png"


# ---------------------------------------------------------------------------
# End-to-end: real JPEG from the API gets the right extension
# ---------------------------------------------------------------------------


class TestRealImageExtensionDetection:
    def test_b64_response_detects_jpeg_from_magic(
        self, tmp_path, monkeypatch, minimax_api_key, mock_requests_post
    ):
        """If the API returns JPEG bytes (as it does today for image-01),
        the saved file must end in .jpg — not .png. The gateway keys
        off the extension when handing the file to downstream tools."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        # 1x1 red JPEG (the smallest valid JPEG — 134 bytes)
        red_jpeg_b64 = (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEB"
            "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAf/AABEI"
            "AAEAAQMBAQAAAAAAAAAAAAAAGAUDBwgCAQAKA//EABUBAQEAAAAAAAAAAAAAAAAB"
            "AQD/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwBVAAH/2Q=="
        )
        mock_requests_post.return_value = _mock_response(
            json_body=_success_b64_body(red_jpeg_b64)
        )
        provider = MiniMaxImageGenProvider()
        result = provider.generate("test")

        assert result["success"] is True
        assert result["image"].endswith(".jpg"), (
            f"expected .jpg extension, got {result['image']!r} "
            f"(save_b64_image defaulting to .png despite JPEG content)"
        )
