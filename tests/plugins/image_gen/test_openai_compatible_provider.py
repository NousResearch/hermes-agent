from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


PLUGIN_PATH = (
    Path(__file__).resolve().parents[3]
    / "plugins"
    / "image_gen"
    / "openai-compatible"
    / "__init__.py"
)


def _load_plugin_module():
    spec = importlib.util.spec_from_file_location(
        "test_openai_compatible_image_plugin", PLUGIN_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(body)


class _Response(requests.Response):
    def __init__(self, payload=None, *, status_code=200, json_error=None):
        super().__init__()
        self._payload = payload
        self.status_code = status_code
        self._json_error = json_error
        self._content = b"ok"

    def json(self, **kwargs):
        if self._json_error:
            raise self._json_error
        return self._payload


@pytest.fixture
def configured_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "test-key")
    _write_config(
        tmp_path,
        "image_gen:\n"
        "  provider: openai-compatible\n"
        "  model: vendor/image-model\n"
        "  openai_compatible:\n"
        "    display_name: Local Images\n"
        "    base_url: http://example.test/v1\n"
        "    size_by_aspect:\n"
        "      square: 512x512\n",
    )
    return tmp_path


class TestOpenAICompatibleImageProvider:
    def test_registers_one_fixed_provider(self, configured_home):
        module = _load_plugin_module()
        registered = []
        module.register(SimpleNamespace(register_image_gen_provider=registered.append))

        assert len(registered) == 1
        assert registered[0].name == "openai-compatible"
        assert registered[0].display_name == "Local Images"
        assert registered[0].is_available() is True

    def test_real_discovery_and_configured_handler_dispatch(
        self, monkeypatch, configured_home
    ):
        seen = {}
        encoded = base64.b64encode(b"discovered-image").decode("ascii")

        def fake_post(url, **kwargs):
            seen.update({"url": url, **kwargs})
            return _Response({"data": [{"b64_json": encoded}]})

        monkeypatch.setattr(requests, "post", fake_post)

        from agent import image_gen_registry
        from hermes_cli.plugins import _ensure_plugins_discovered
        from tools import image_generation_tool

        image_gen_registry._reset_for_tests()
        try:
            _ensure_plugins_discovered(force=True)
            provider = image_gen_registry.get_provider("openai-compatible")
            assert provider is not None

            payload = json.loads(
                image_generation_tool._handle_image_generate(
                    {"prompt": "draw local cat", "aspect_ratio": "square"}
                )
            )
            assert payload["success"] is True
            assert payload["provider"] == "openai-compatible"
            assert payload["model"] == "vendor/image-model"
            assert Path(payload["image"]).read_bytes() == b"discovered-image"
            assert seen["url"] == "http://example.test/v1/images/generations"
        finally:
            image_gen_registry._reset_for_tests()

    def test_dispatch_prefers_provider_scoped_model(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "test-key")
        _write_config(
            tmp_path,
            "image_gen:\n"
            "  provider: openai-compatible\n"
            "  model: stale/global-model\n"
            "  openai_compatible:\n"
            "    model: scoped/provider-model\n"
            "    base_url: http://example.test/v1\n",
        )
        seen = {}

        def fake_post(url, **kwargs):
            seen.update({"url": url, **kwargs})
            encoded = base64.b64encode(b"scoped-model-image").decode("ascii")
            return _Response({"data": [{"b64_json": encoded}]})

        monkeypatch.setattr(requests, "post", fake_post)

        from agent import image_gen_registry
        from hermes_cli.plugins import _ensure_plugins_discovered
        from tools import image_generation_tool

        image_gen_registry._reset_for_tests()
        try:
            _ensure_plugins_discovered(force=True)
            payload = json.loads(
                image_generation_tool._handle_image_generate(
                    {"prompt": "draw scoped cat", "aspect_ratio": "square"}
                )
            )
            assert payload["success"] is True
            assert payload["model"] == "scoped/provider-model"
            assert seen["json"]["model"] == "scoped/provider-model"
        finally:
            image_gen_registry._reset_for_tests()

    def test_unselected_credentials_do_not_activate_provider(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "unused-key")
        _write_config(tmp_path, "image_gen:\n  model: vendor/image-model\n")

        from tools import image_generation_tool

        assert (
            image_generation_tool._dispatch_to_plugin_provider("draw cat", "square")
            is None
        )

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("http://example.test", "http://example.test/v1/images/generations"),
            ("http://example.test/v1", "http://example.test/v1/images/generations"),
            (
                "https://example.test/root/v1/",
                "https://example.test/root/v1/images/generations",
            ),
            (
                "https://example.test/v1/images/generations",
                "https://example.test/v1/images/generations",
            ),
        ],
    )
    def test_endpoint_joining(self, base_url, expected):
        module = _load_plugin_module()
        assert module._endpoint(base_url) == expected

    @pytest.mark.parametrize(
        "base_url",
        [
            "",
            "example.test/v1",
            "file:///tmp/images",
            "https://user:pass@example.test/v1",
            "https://example.test/v1?tenant=a",
            "https://example.test/v1#fragment",
        ],
    )
    def test_endpoint_rejects_invalid_or_credentialed_urls(self, base_url):
        module = _load_plugin_module()
        with pytest.raises(ValueError):
            module._endpoint(base_url)

    def test_payload_omits_response_format_unless_configured(
        self, monkeypatch, configured_home
    ):
        module = _load_plugin_module()
        seen = []

        def fake_post(*args, **kwargs):
            seen.append(kwargs["json"])
            return _Response({"data": [{"b64_json": base64.b64encode(b"x").decode()}]})

        monkeypatch.setattr(module.requests, "post", fake_post)
        result = module.OpenAICompatibleImageProvider().generate("cat", "square")
        assert result["success"] is True
        assert "response_format" not in seen[-1]

        config_path = configured_home / "config.yaml"
        config_path.write_text(config_path.read_text() + "    response_format: b64_json\n")
        result = module.OpenAICompatibleImageProvider().generate("cat", "square")
        assert result["success"] is True
        assert seen[-1]["response_format"] == "b64_json"

    def test_reserved_params_fail_closed(self, monkeypatch, configured_home):
        config_path = configured_home / "config.yaml"
        config_path.write_text(
            config_path.read_text()
            + "    params:\n"
            + "      prompt: attacker-controlled replacement\n"
        )
        module = _load_plugin_module()
        monkeypatch.setattr(
            module.requests,
            "post",
            lambda *args, **kwargs: pytest.fail("network call should not occur"),
        )

        result = module.OpenAICompatibleImageProvider().generate("real prompt", "square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_config"
        assert "prompt" in result["error"]

    def test_multiple_output_count_is_rejected(self, monkeypatch, configured_home):
        config_path = configured_home / "config.yaml"
        config_path.write_text(config_path.read_text() + "    n: 2\n")
        module = _load_plugin_module()
        monkeypatch.setattr(
            module.requests,
            "post",
            lambda *args, **kwargs: pytest.fail("network call should not occur"),
        )

        result = module.OpenAICompatibleImageProvider().generate("cat", "square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_config"
        assert "n must be 1" in result["error"]

    def test_url_output_is_cached(self, monkeypatch, configured_home, tmp_path):
        module = _load_plugin_module()
        cached = tmp_path / "cached.png"
        cached.write_bytes(b"url-image")
        monkeypatch.setattr(
            module.requests,
            "post",
            lambda *args, **kwargs: _Response(
                {"data": [{"url": "https://cdn.example.test/temporary.png"}]}
            ),
        )
        seen = {}

        def fake_save(url, *, prefix):
            seen.update({"url": url, "prefix": prefix})
            return cached

        monkeypatch.setattr(module, "save_url_image", fake_save)
        result = module.OpenAICompatibleImageProvider().generate("cat", "square")

        assert result["success"] is True
        assert result["image"] == str(cached)
        assert seen["url"] == "https://cdn.example.test/temporary.png"
        assert "/" not in seen["prefix"]

    def test_unsupported_edit_is_rejected(self, configured_home):
        module = _load_plugin_module()
        result = module.OpenAICompatibleImageProvider().generate(
            "edit cat", "square", image_url="/tmp/cat.png"
        )
        assert result["error_type"] == "modality_unsupported"

    def test_redirect_is_not_followed(self, monkeypatch, configured_home):
        module = _load_plugin_module()
        seen = {}

        def fake_post(*args, **kwargs):
            seen.update(kwargs)
            return _Response({}, status_code=302)

        monkeypatch.setattr(module.requests, "post", fake_post)
        result = module.OpenAICompatibleImageProvider().generate("cat", "square")
        assert result["error_type"] == "redirect_not_allowed"
        assert seen["allow_redirects"] is False

    def test_timeout_is_structured(self, monkeypatch, configured_home):
        module = _load_plugin_module()
        monkeypatch.setattr(
            module.requests,
            "post",
            lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout()),
        )
        result = module.OpenAICompatibleImageProvider().generate("cat", "square")
        assert result["error_type"] == "timeout"

    def test_http_error_is_structured(self, monkeypatch, configured_home):
        module = _load_plugin_module()
        monkeypatch.setattr(
            module.requests,
            "post",
            lambda *args, **kwargs: _Response(
                {"error": {"message": "backend rejected request"}}, status_code=400
            ),
        )
        result = module.OpenAICompatibleImageProvider().generate("cat", "square")
        assert result["error_type"] == "api_error"
        assert "backend rejected request" in result["error"]

    def test_invalid_json_and_empty_output_are_structured(
        self, monkeypatch, configured_home
    ):
        module = _load_plugin_module()
        provider = module.OpenAICompatibleImageProvider()
        monkeypatch.setattr(
            module.requests,
            "post",
            lambda *args, **kwargs: _Response(json_error=ValueError("bad json")),
        )
        assert provider.generate("cat", "square")["error_type"] == "invalid_response"

        monkeypatch.setattr(
            module.requests,
            "post",
            lambda *args, **kwargs: _Response({"data": []}),
        )
        assert provider.generate("cat", "square")["error_type"] == "empty_response"

    def test_literal_config_credentials_are_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", raising=False)
        _write_config(
            tmp_path,
            "image_gen:\n"
            "  provider: openai-compatible\n"
            "  model: vendor/image-model\n"
            "  openai_compatible:\n"
            "    base_url: http://example.test/v1\n"
            "    api_key: must-not-be-read\n"
            "    headers:\n"
            "      Authorization: must-not-be-read\n",
        )
        module = _load_plugin_module()
        result = module.OpenAICompatibleImageProvider().generate("cat", "square")
        assert result["error_type"] == "auth_required"

    def test_secret_scope_is_authoritative(self, monkeypatch):
        from agent import secret_scope

        module = _load_plugin_module()
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "process-global-key")
        secret_scope.set_multiplex_active(True)
        try:
            token = secret_scope.set_secret_scope(
                {"OPENAI_COMPATIBLE_IMAGE_API_KEY": "profile-a-key"}
            )
            try:
                assert module._read_api_key() == "profile-a-key"
            finally:
                secret_scope.reset_secret_scope(token)

            token = secret_scope.set_secret_scope(
                {"OPENAI_COMPATIBLE_IMAGE_API_KEY": "profile-b-key"}
            )
            try:
                assert module._read_api_key() == "profile-b-key"
            finally:
                secret_scope.reset_secret_scope(token)

            token = secret_scope.set_secret_scope({})
            try:
                assert module._read_api_key() == ""
            finally:
                secret_scope.reset_secret_scope(token)
        finally:
            secret_scope.set_multiplex_active(False)

    def test_unscoped_secret_read_fails_closed_under_multiplex(self, monkeypatch):
        from agent import secret_scope

        module = _load_plugin_module()
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "process-global-key")
        secret_scope.set_multiplex_active(True)
        try:
            with pytest.raises(secret_scope.UnscopedSecretError):
                module._read_api_key()
        finally:
            secret_scope.set_multiplex_active(False)

    def test_switch_to_profile_without_key_does_not_reuse_previous_key(
        self, monkeypatch, tmp_path
    ):
        from agent import secret_scope

        first = tmp_path / "profile-a"
        second = tmp_path / "profile-b"
        first.mkdir()
        second.mkdir()
        (first / ".env").write_text("OPENAI_COMPATIBLE_IMAGE_API_KEY=profile-a-key\n")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "process-global-key")
        module = _load_plugin_module()
        secret_scope.set_multiplex_active(True)
        try:
            token = secret_scope.set_secret_scope(
                secret_scope.build_profile_secret_scope(first)
            )
            try:
                assert module._read_api_key() == "profile-a-key"
            finally:
                secret_scope.reset_secret_scope(token)

            token = secret_scope.set_secret_scope(
                secret_scope.build_profile_secret_scope(second)
            )
            try:
                assert module._read_api_key() == ""
            finally:
                secret_scope.reset_secret_scope(token)
        finally:
            secret_scope.set_multiplex_active(False)

    def test_setup_schema_uses_fixed_profile_secret(self, configured_home):
        module = _load_plugin_module()
        schema = module.OpenAICompatibleImageProvider().get_setup_schema()
        assert schema["env_vars"] == [
            {
                "key": "OPENAI_COMPATIBLE_IMAGE_API_KEY",
                "prompt": "OpenAI-compatible image endpoint bearer token",
            }
        ]
