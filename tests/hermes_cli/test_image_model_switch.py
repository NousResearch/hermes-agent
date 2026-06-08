from __future__ import annotations

import yaml

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


class _CatalogProvider(ImageGenProvider):
    def __init__(self, name: str, models: list[str], default: str | None = None):
        self._name = name
        self._models = models
        self._default = default or (models[0] if models else None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name.title()

    def list_models(self):
        return [{"id": mid, "display": mid.upper()} for mid in self._models]

    def default_model(self):
        return self._default

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        raise AssertionError("switch tests should not generate images")


def setup_function():
    image_gen_registry._reset_for_tests()


def teardown_function():
    image_gen_registry._reset_for_tests()


def test_cli_process_command_routes_image_model(monkeypatch):
    import cli as cli_module

    calls = []
    instance = object.__new__(cli_module.HermesCLI)
    monkeypatch.setattr(
        cli_module.HermesCLI,
        "_handle_image_model_switch",
        lambda self, command: calls.append(command),
    )

    assert cli_module.HermesCLI.process_command(instance, "/image_model openai/gpt-image-2-medium") is True
    assert calls == ["/image_model openai/gpt-image-2-medium"]


def test_apply_image_model_switch_persists_provider_and_model(monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.image_model_switch import apply_image_model_switch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "image_gen:\n"
        "  provider: fal\n"
        "  model: flux-dev\n"
    )
    image_gen_registry.register_provider(_CatalogProvider("fal", ["flux-dev"]))
    image_gen_registry.register_provider(_CatalogProvider("openai", ["gpt-image-2-low", "gpt-image-2-medium"]))
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

    result = apply_image_model_switch("gpt-image-2-medium --provider openai")

    assert result.success is True
    assert result.provider == "openai"
    assert result.model == "gpt-image-2-medium"
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["image_gen"]["provider"] == "openai"
    assert cfg["image_gen"]["model"] == "gpt-image-2-medium"
    assert cfg["image_gen"]["openai"]["model"] == "gpt-image-2-medium"


def test_apply_image_model_switch_accepts_provider_slash_model(monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.image_model_switch import apply_image_model_switch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("image_gen:\n  provider: fal\n")
    image_gen_registry.register_provider(_CatalogProvider("fal", ["flux-dev"]))
    image_gen_registry.register_provider(_CatalogProvider("xai", ["grok-imagine-image-quality"]))
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

    result = apply_image_model_switch("xai/grok-imagine-image-quality")

    assert result.success is True
    assert result.provider == "xai"
    assert result.model == "grok-imagine-image-quality"


def test_apply_image_model_switch_reports_available_providers_for_unknown_provider(monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.image_model_switch import apply_image_model_switch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("image_gen:\n  provider: fal\n")
    image_gen_registry.register_provider(_CatalogProvider("fal", ["flux-dev"]))
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

    result = apply_image_model_switch("--provider missing")

    assert result.success is False
    assert "missing" in result.message
    assert "fal" in result.message


def test_format_image_model_status_lists_current_and_usage(monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.image_model_switch import format_image_model_status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "image_gen:\n"
        "  provider: openai\n"
        "  openai:\n"
        "    model: gpt-image-2-medium\n"
    )
    image_gen_registry.register_provider(_CatalogProvider("openai", ["gpt-image-2-low", "gpt-image-2-medium"]))
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

    message = format_image_model_status()

    assert "openai" in message
    assert "gpt-image-2-medium" in message
    assert "/image_model" in message


def test_apply_image_model_switch_clears_stale_global_model_for_model_less_provider(monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.image_model_switch import apply_image_model_switch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "image_gen:\n"
        "  provider: openai\n"
        "  model: stale-model\n"
    )
    image_gen_registry.register_provider(_CatalogProvider("openai", ["stale-model"]))
    image_gen_registry.register_provider(_CatalogProvider("noop", [], default=""))
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

    result = apply_image_model_switch("noop")

    assert result.success is True
    assert result.provider == "noop"
    assert result.model == ""
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["image_gen"]["provider"] == "noop"
    assert "model" not in cfg["image_gen"]
    assert "noop" not in cfg["image_gen"]


def test_apply_image_model_switch_reports_managed_config_error(monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.image_model_switch import apply_image_model_switch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED", "homebrew")
    for subdir in ("cron", "sessions", "logs", "memories"):
        (tmp_path / subdir).mkdir()
    (tmp_path / "config.yaml").write_text("image_gen:\n  provider: fal\n")
    image_gen_registry.register_provider(_CatalogProvider("openai", ["gpt-image-2-medium"]))
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

    result = apply_image_model_switch("openai/gpt-image-2-medium")

    assert result.success is False
    assert "managed by Homebrew" in result.message
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["image_gen"]["provider"] == "fal"
