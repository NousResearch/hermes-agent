"""Fresh-profile setup through the same picker used by hermes tools/setup."""

import base64
from pathlib import Path
import json

import pytest
import requests

from agent import image_gen_registry
from hermes_cli.config import load_config, save_config
from hermes_cli.plugins import _ensure_plugins_discovered
from hermes_cli import tools_config_providers as setup


@pytest.mark.parametrize("endpoint,model", [("", ""), ("https://images.example.test", "")])
def test_skipped_required_settings_do_not_report_ready(monkeypatch, tmp_path, capsys, endpoint, model):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "existing-token")
    image_gen_registry._reset_for_tests()
    try:
        _ensure_plugins_discovered(force=True)
        row = next(row for row in setup._plugin_image_gen_providers()
                   if row["image_gen_plugin_name"] == "openai-compatible")
        config = load_config()
        assert setup.provider_readiness_status(row, config) == "needs_setup"
        answers = iter([endpoint, model])
        monkeypatch.setattr(setup, "_prompt", lambda *a, **kw: next(answers))
        setup._configure_provider(row, config)
        save_config(config)
        assert setup.provider_readiness_status(row, config) == "needs_setup"
        assert not image_gen_registry.get_provider("openai-compatible").is_available()
        assert "OpenAI-compatible configured!" not in capsys.readouterr().out
    finally:
        image_gen_registry._reset_for_tests()


@pytest.mark.parametrize("scoped_model,expected_model", [
    (None, "vendor/image-model"),
    ("", "vendor/image-model"),
    ("   ", "vendor/image-model"),
    (42, "vendor/image-model"),
    ("  scoped/image-model  ", "scoped/image-model"),
])
def test_documented_model_config_agrees_across_generation_and_setup(
    monkeypatch, tmp_path, scoped_model, expected_model
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "profile-token")
    # The specification's documented global-model configuration, not a schema snapshot.
    config = {"image_gen": {
        "provider": "openai-compatible",
        "model": "vendor/image-model",
        "openai_compatible": {
            "display_name": "Local Image Service",
            "base_url": "http://localhost:30122/v1",
            "size_by_aspect": {"square": "512x512", "landscape": "1024x576", "portrait": "576x1024"},
            "params": {"num_inference_steps": 8, "guidance_scale": 1.0},
            "timeout": 600,
        },
    }}
    if scoped_model is not None:
        config["image_gen"]["openai_compatible"]["model"] = scoped_model
    save_config(config)
    image_gen_registry._reset_for_tests()
    try:
        _ensure_plugins_discovered(force=True)
        provider = image_gen_registry.get_provider("openai-compatible")
        row = next(row for row in setup._plugin_image_gen_providers()
                   if row["image_gen_plugin_name"] == provider.name)
        seen = {}

        def post(url, **kwargs):
            seen.update(kwargs)
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({"data": [{"b64_json": base64.b64encode(b"image").decode()}]}).encode()
            return response

        monkeypatch.setattr(requests, "post", post)
        assert provider.is_available()
        assert provider.generate("cat")["success"] is True
        assert seen["json"]["model"] == expected_model
        assert setup.provider_readiness_status(row, config) == "ready"

        # A valid model must not trigger a second free-text model/endpoint/secret prompt.
        monkeypatch.setattr(setup, "_prompt", lambda *a, **kw: pytest.fail("configuration is already complete"))
        monkeypatch.setattr("hermes_cli.tools_config._prompt_choice", lambda *a, **kw: 0)
        setup._configure_provider(row, config)
        save_config(config)
        saved = load_config()
        assert saved["image_gen"]["openai_compatible"].get("model") == scoped_model
        assert setup.provider_readiness_status(row, saved) == "ready"
        assert provider.is_available()
        assert provider.default_model() == expected_model
    finally:
        image_gen_registry._reset_for_tests()


def test_empty_profile_setup_can_generate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", raising=False)
    image_gen_registry._reset_for_tests()
    try:
        _ensure_plugins_discovered(force=True)
        provider = image_gen_registry.get_provider("openai-compatible")
        assert not provider.is_available()
        assert not provider.list_models()
        row = next(row for row in setup._plugin_image_gen_providers()
                   if row["image_gen_plugin_name"] == provider.name)
        answers = iter(["setup-token", "https://images.example.test/v1", "vendor/setup-model"])
        monkeypatch.setattr(setup, "_prompt", lambda *a, **kw: next(answers))
        config = load_config()
        setup._configure_provider(row, config)
        save_config(config)

        # Load what setup actually persisted, not an injected config/provider stub.
        assert provider.is_available()
        assert provider.default_model() == "vendor/setup-model"
        assert any(model["id"] == provider.default_model() for model in provider.list_models())
        assert "setup-token" not in (tmp_path / "config.yaml").read_text()
        assert "setup-token" in (tmp_path / ".env").read_text()
        seen = {}

        def post(url, **kwargs):
            seen.update(url=url, **kwargs)
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({"data": [{"b64_json": base64.b64encode(b"image").decode()}]}).encode()
            return response

        monkeypatch.setattr("requests.post", post)
        from tools.image_generation_tool import _handle_image_generate

        result = json.loads(_handle_image_generate({"prompt": "cat", "aspect_ratio": "square"}))
        assert result["success"] is True
        assert Path(result["image"]).read_bytes() == b"image"
        assert seen["url"] == "https://images.example.test/v1/images/generations"
        assert seen["json"]["model"] == "vendor/setup-model"
        assert seen["headers"]["Authorization"] == "Bearer setup-token"
    finally:
        image_gen_registry._reset_for_tests()
