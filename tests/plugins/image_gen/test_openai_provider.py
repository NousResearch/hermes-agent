"""Tests for the bundled OpenAI image_gen plugin (gpt-image-2, three tiers)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.openai as openai_plugin


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_response(*, b64=None, url=None, revised_prompt=None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return openai_plugin.OpenAIImageGenProvider()


def _patched_openai(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return patch.dict("sys.modules", {"openai": fake_openai})


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:


    def test_picker_matches_resolvable_catalog(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert set(ids) == set(provider.models)
        assert provider.default_model() in ids



# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert openai_plugin.OpenAIImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        assert openai_plugin.OpenAIImageGenProvider().is_available() is True


# ── Model resolution ────────────────────────────────────────────────────────


class TestModelResolution:

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2-high")
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"


    def test_config_openai_model(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"model": "gpt-image-2-low"}}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-low"
        assert meta["quality"] == "low"


# ── Endpoint / credential routing ───────────────────────────────────────────


class TestEndpointConfig:
    """``image_gen.openai.base_url`` / ``key_env`` reach the client and its request (#65309, #97928,
    #13798); the project header is blanked (#60748); custom endpoints bypass system proxies (#64888)."""

    def test_config_base_url_and_key_env_reach_client_and_availability(self, monkeypatch, tmp_path):
        import yaml
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("IMAGE_GATEWAY_TOKEN", "gateway-token")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {"openai": {
            "base_url": "http://localhost:18081/v1/", "key_env": "IMAGE_GATEWAY_TOKEN"}}}))
        provider = openai_plugin.OpenAIImageGenProvider()
        assert provider.is_available() is True  # same resolver as generate(); no OPENAI_API_KEY needed
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        with _patched_openai(fake_client):
            assert provider.generate("a cat")["success"] is True
            kwargs = __import__("sys").modules["openai"].OpenAI.call_args.kwargs
        assert kwargs["base_url"] == "http://localhost:18081/v1"
        assert kwargs["api_key"] == "gateway-token"
        assert kwargs["default_headers"]["OpenAI-Project"] == ""
        kwargs["http_client"].close()

    def test_custom_model_id_passes_through_without_quality(self, monkeypatch, tmp_path):
        """A non-catalog ``image_gen.openai.model`` reaches the gateway verbatim as ``model`` and no
        ``quality`` is sent (gateways reject unknown enum values); a stale top-level ``image_gen.model``
        from another provider never passes through (#97928)."""
        import yaml
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {
            "model": "fal-ai/flux-2/klein/9b", "openai": {"model": "custom-image-model"}}}))
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        with _patched_openai(fake_client):
            result = openai_plugin.OpenAIImageGenProvider().generate("a cat")
        assert result["success"] is True and result["model"] == "custom-image-model"
        request = fake_client.images.generate.call_args.kwargs
        assert request["model"] == "custom-image-model" and "quality" not in request
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {"model": "fal-ai/flux-2/klein/9b"}}))
        assert openai_plugin._resolve_model()[0] == "gpt-image-2-medium"

    def test_named_custom_endpoint_supplies_base_url_and_key(self, monkeypatch, tmp_path):
        """``image_gen.openai.provider: <name>`` inherits that ``providers:`` entry's base_url and
        key_env when ``base_url``/``key_env`` are unset; explicit values still win (#83080)."""
        import yaml
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("MY_GW_KEY", "gw-token")
        monkeypatch.setenv("IMAGE_ONLY_KEY", "image-token")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "providers": {"my-gateway": {"name": "My Gateway", "api": "https://gw.example/v1", "key_env": "MY_GW_KEY"}},
            "image_gen": {"openai": {"provider": "my-gateway"}}}))
        assert openai_plugin._resolve_endpoint() == ("https://gw.example/v1", "gw-token")
        assert openai_plugin.OpenAIImageGenProvider().is_available() is True
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "providers": {"my-gateway": {"name": "My Gateway", "api": "https://gw.example/v1", "key_env": "MY_GW_KEY"}},
            "image_gen": {"openai": {"provider": "my-gateway", "key_env": "IMAGE_ONLY_KEY"}}}))
        assert openai_plugin._resolve_endpoint() == ("https://gw.example/v1", "image-token")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {"openai": {"provider": "no-such"}}}))
        assert openai_plugin._resolve_endpoint() == ("", "")

    def test_custom_base_url_ignores_system_proxy(self, monkeypatch, tmp_path):
        """httpx only sees macOS system proxies via ``getproxies()`` (bound at import in
        ``httpx._utils``; ExceptionsList dropped): with a system proxy visible and no proxy env var,
        ``generate()`` must hand ``openai.OpenAI`` a client with no ``HTTPProxy`` mount, while a plain
        ``httpx.Client()`` under the same conditions (control) does pick the proxy up (#64888)."""
        import httpx
        import yaml
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
                    "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(
            {"image_gen": {"openai": {"base_url": "http://localhost:18081/v1"}}}))
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        sys_proxy = {"http": "http://sysproxy:3128", "https": "http://sysproxy:3128"}

        def proxy_mounts(client):
            return [m for m in client._mounts.values()
                    if type(getattr(m, "_pool", None)).__name__ == "HTTPProxy"]

        with patch("httpx._utils.getproxies", return_value=sys_proxy), _patched_openai(fake_client):
            with httpx.Client() as control:
                assert len(proxy_mounts(control)) == 2  # the fake system proxy IS visible to httpx
            assert openai_plugin.OpenAIImageGenProvider().generate("a cat")["success"] is True
            http_client = __import__("sys").modules["openai"].OpenAI.call_args.kwargs["http_client"]
        assert isinstance(http_client, httpx.Client)
        assert proxy_mounts(http_client) == []
        http_client.close()


# ── Generate ────────────────────────────────────────────────────────────────


class TestSourceImageLoading:
    def test_load_image_bytes_blocks_credential_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        with pytest.raises(ValueError, match="credential store"):
            openai_plugin._load_image_bytes(str(auth_json))


    def test_load_image_bytes_allows_legit_local_image(self, tmp_path, monkeypatch):
        """Negative control: a legitimate local image path is NOT blocked and
        loads normally — proves the guard doesn't over-fire on everything."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")

        data, name = openai_plugin._load_image_bytes(str(img))
        assert data == b"\x89PNG\r\n\x1a\nfake-image-bytes"
        assert name == "pic.png"


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = openai_plugin.OpenAIImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_b64_saves_to_cache(self, provider, tmp_path):
        png_bytes = bytes.fromhex(_PNG_HEX)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "openai"
        assert result["quality"] == "medium"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        # All tiers hit the single underlying API model.
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "medium"
        assert call_kwargs["size"] == "1536x1024"
        # gpt-image-2 rejects response_format — we must NOT send it.
        assert "response_format" not in call_kwargs

    @pytest.mark.parametrize("has_image", [True, False])
    def test_token_usage_reaches_session_accounting(self, provider, has_image):
        """gpt-image bills per token: the Images API ``usage`` block lands as one
        ``image_generation`` row keyed on the API model, not the Hermes tier label — also
        when the billed HTTP 200 carries no image data."""
        from agent import aux_accounting

        recorded = []

        class _DB:
            def record_auxiliary_usage(self, *args, **kwargs):
                recorded.append((args, kwargs))

        response = _fake_response(b64=_b64_png())
        if not has_image:
            response.data = []
        response.usage = SimpleNamespace(input_tokens=23, output_tokens=1056, total_tokens=1079)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = response
        token = aux_accounting.set_accounting_context(_DB(), "sess-1")
        try:
            with _patched_openai(fake_client):
                result = provider.generate("a cat", aspect_ratio="landscape")
        finally:
            aux_accounting.reset_accounting_context(token)

        assert result["success"] is has_image
        if not has_image:
            assert result["error_type"] == "empty_response"
        ((session_id, task), kwargs), = recorded
        assert (session_id, task) == ("sess-1", "image_generation")
        assert (kwargs["model"], kwargs["billing_provider"]) == ("gpt-image-2", "openai")
        assert (kwargs["input_tokens"], kwargs["output_tokens"]) == (23, 1056)

    @pytest.mark.parametrize("api_model,quality", [
        ("gpt-image-2", quality) for quality in ("low", "medium", "high")
    ] + [
        (model, quality)
        for model in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst")
        for quality in ("auto", "low", "medium", "high", "xhigh", "max")
    ])
    @pytest.mark.parametrize("editing", [False, True])
    def test_selection_reaches_image_request(
        self, provider, monkeypatch, tmp_path, api_model, quality, editing
    ):
        import yaml

        tier = api_model if quality == "auto" else f"{api_model}-{quality}"
        monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "image_gen": {"openai": {"model": tier}}
        }))
        source = tmp_path / "source.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))
        fake_client = MagicMock()
        call = fake_client.images.edit if editing else fake_client.images.generate
        call.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", image_url=str(source) if editing else None)

        assert result["success"] is True
        assert result["model"] == tier
        assert result["quality"] == quality
        assert call.call_args.kwargs["quality"] == quality
        assert call.call_args.kwargs["model"] == api_model
        assert "response_format" not in call.call_args.kwargs
        assert Path(result["image"]).read_bytes() == bytes.fromhex(_PNG_HEX)

    @pytest.mark.parametrize("aspect,expected_size", [
        ("landscape", "1536x1024"),
        ("square", "1024x1024"),
        ("portrait", "1024x1536"),
    ])
    def test_aspect_ratio_mapping(self, provider, aspect, expected_size):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            provider.generate("a cat", aspect_ratio=aspect)

        assert fake_client.images.generate.call_args.kwargs["size"] == expected_size

    def test_revised_prompt_passed_through(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=_b64_png(), revised_prompt="A photo of a cat",
        )

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["revised_prompt"] == "A photo of a cat"


    def test_url_response_is_cached_locally(self, provider):
        """OpenAI URL response (if API ever returns one) is cached locally.

        Pre-fix this asserted the bare URL passed through; symmetric to the
        xAI #26942 fix.  Even though gpt-image-2 returns b64 today, every
        ``image_gen`` provider must guarantee the gateway gets a stable
        file path so ephemeral signed URLs can't expire mid-flight.
        """
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png",
        )

        with _patched_openai(fake_client), patch(
            "plugins.image_gen._common.save_url_image",
            return_value=Path("/tmp/openai_gpt-image-2_20260524_000000_deadbeef.png"),
        ) as mock_save_url:
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"].startswith("/")
        assert "example.com" not in result["image"]
        mock_save_url.assert_called_once()

