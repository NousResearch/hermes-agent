"""A newly shipped xAI model is usable before third-party catalogs catch up."""
from contextlib import contextmanager

import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def _profile(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_offline_declared_model_reaches_context_vision_and_picker_consumers(tmp_path, monkeypatch):
    from agent import model_metadata as metadata, models_dev
    from agent.image_routing import decide_image_input_mode
    from hermes_cli.inventory import _apply_capabilities
    from providers import get_provider_profile

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda **kwargs: {})
    monkeypatch.setattr(metadata, "fetch_model_metadata", lambda **kwargs: {})
    declared = get_provider_profile("xai").model_capabilities["grok-4.7"]
    expected = declared["context_window"]
    for provider in ("xai", "xai-oauth", "x-ai"):
        for model in ("grok-4.7", "GROK-4.7", "x-ai/grok-4.7"):
            caps = models_dev.get_model_capabilities(provider, model)
            assert caps.context_window == expected
            assert caps.supports_vision and caps.supports_tools and caps.supports_reasoning
            assert caps.max_output_tokens is None
            assert metadata.get_model_context_length(model, "https://api.x.ai/v1", provider=provider) == expected
            assert decide_image_input_mode(provider, model, {"model": {"provider": provider, "default": model}}) == "native"
    rows = [{"slug": "xai", "models": ["grok-4.7", "grok-4.70"]}]
    _apply_capabilities(rows)
    assert rows[0]["capabilities"]["grok-4.7"]["reasoning"] is True
    for neighbor in ("grok-4.70", "grok-4.8", "grok-4.7-fast", "custom-grok-4.7"):
        assert models_dev.get_model_capabilities("xai", neighbor) is None
        assert decide_image_input_mode("xai", neighbor, {}) == "text"
        assert metadata.get_model_context_length(neighbor, "https://api.x.ai/v1", provider="xai") != expected
    assert models_dev.get_model_capabilities("openai", "grok-4.7") is None
    assert metadata.get_model_context_length("grok-4.6", "https://api.x.ai/v1", provider="xai") == 500_000


def test_stale_context_repair_and_explicit_limits_stay_profile_scoped(tmp_path, monkeypatch):
    from agent import model_metadata as metadata, models_dev
    from hermes_cli.config import load_config_readonly
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / "A", tmp_path / "B"]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda **kwargs: {})
    monkeypatch.setattr(metadata, "fetch_model_metadata", lambda **kwargs: {})
    (homes[1] / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "xai", "default": "grok-4.6"},
        "model_overrides": {"xai": {"grok-4.7": {"context_window": 96_000, "supports_vision": False}}}}), encoding="utf-8")
    base = "https://api.x.ai/v1"
    for home in homes:
        with _profile(home):
            metadata.save_context_length("grok-4.7", base, 256_000)
    for home, expected, vision in ((homes[0], 500_000, True), (homes[1], 96_000, False), (homes[0], 500_000, True)):
        with _profile(home):
            assert metadata.get_model_context_length("grok-4.7", base, provider="xai") == expected
            caps = models_dev.get_model_capabilities("xai", "grok-4.7")
            assert (caps.context_window, caps.supports_vision) == (expected, vision)
            assert metadata.get_model_context_length("grok-4.7", base, provider="xai", config_context_length=72_000) == 72_000
    with _profile(homes[1]):
        assert metadata.get_cached_context_length("grok-4.7", base) == 256_000
        assert load_config_readonly()["model"]["default"] == "grok-4.6"
    with _profile(homes[0]):
        assert metadata.get_cached_context_length("grok-4.7", base) != 256_000
        configured = [{"name": "xai", "base_url": base, "models": {"grok-4.7": {"context_length": 120_000}}}]
        assert metadata.get_model_context_length("grok-4.7", base, provider="xai", custom_providers=configured) == 120_000
