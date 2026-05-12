from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_MODELS


def test_deepseek_models_include_v4_variants():
    models = _PROVIDER_MODELS["deepseek"]

    assert "deepseek-v4-pro" in models
    assert "deepseek-v4-flash" in models


def test_deepseek_provider_description_mentions_v4_models():
    entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "deepseek")

    assert "V4 Pro" in entry.tui_desc
    assert "V4 Flash" in entry.tui_desc
    assert "V3" not in entry.tui_desc
