"""Tests for three model-switch bug fixes.

1. list_authenticated_providers() includes models from the `models` dict
   in custom_providers entries (not just the single `model` field).

2. set_runtime_model() propagates the switch to auxiliary tasks so that
   title generation, compression, etc. use the same model as the session.
"""

from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fix 1: picker detects models from the `models` dict
# ---------------------------------------------------------------------------


def _find_provider(results, name):
    return next((p for p in results if p["name"] == name), None)


def test_list_authenticated_providers_includes_models_dict():
    """Models listed in the `models` dict of a custom_providers entry must
    appear in the picker, not just the single `model` field."""
    from hermes_cli.model_switch import list_authenticated_providers

    custom_providers = [
        {
            "name": "LM Studio",
            "base_url": "http://localhost:1234/v1",
            "model": "vendor/model-a",
            "models": {
                "vendor/model-a": {},
                "vendor/model-b": {},
                "vendor/model-c": {},
            },
        }
    ]

    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        results = list_authenticated_providers(custom_providers=custom_providers)

    provider = _find_provider(results, "LM Studio")
    assert provider is not None
    assert "vendor/model-a" in provider["models"]
    assert "vendor/model-b" in provider["models"]
    assert "vendor/model-c" in provider["models"]
    assert provider["total_models"] == 3


def test_list_authenticated_providers_deduplicates_model_field():
    """The `model` field must not appear twice when it is also in `models`."""
    from hermes_cli.model_switch import list_authenticated_providers

    custom_providers = [
        {
            "name": "Local",
            "base_url": "http://localhost:1234/v1",
            "model": "vendor/model-a",
            "models": {
                "vendor/model-a": {},
                "vendor/model-b": {},
            },
        }
    ]

    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        results = list_authenticated_providers(custom_providers=custom_providers)

    provider = _find_provider(results, "Local")
    assert provider is not None
    models = provider["models"]
    assert models.count("vendor/model-a") == 1
    assert len(models) == 2


def test_list_authenticated_providers_no_models_dict():
    """Entries without a `models` dict still work — only `model` field shown."""
    from hermes_cli.model_switch import list_authenticated_providers

    custom_providers = [
        {
            "name": "Local Single",
            "base_url": "http://localhost:1234/v1",
            "model": "vendor/model-a",
        }
    ]

    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        results = list_authenticated_providers(custom_providers=custom_providers)

    provider = _find_provider(results, "Local Single")
    assert provider is not None
    assert provider["models"] == ["vendor/model-a"]
    assert provider["total_models"] == 1


# ---------------------------------------------------------------------------
# Fix 2: set_runtime_model() propagates switch to auxiliary tasks
# ---------------------------------------------------------------------------


def test_set_runtime_model_overrides_read_main_model():
    """After set_runtime_model(), _read_main_model() returns the override,
    not the value from config.yaml."""
    import agent.auxiliary_client as aux

    aux._runtime_model_override = ""

    with patch("hermes_cli.config.load_config", return_value={
        "model": {"default": "config/default-model"}
    }):
        assert aux._read_main_model() == "config/default-model"

        aux.set_runtime_model("switched/model", "custom", "http://localhost:1234/v1")
        assert aux._read_main_model() == "switched/model"

    aux._runtime_model_override = ""


def test_set_runtime_model_empty_clears_override():
    """set_runtime_model('') clears the override so config.yaml is used again."""
    import agent.auxiliary_client as aux

    aux.set_runtime_model("some/model")
    assert aux._runtime_model_override == "some/model"

    aux.set_runtime_model("")
    assert aux._runtime_model_override == ""

    aux._runtime_model_override = ""


def test_set_runtime_model_stores_provider_and_base_url():
    """set_runtime_model() stores provider and base_url overrides too."""
    import agent.auxiliary_client as aux

    aux.set_runtime_model("m/model", "custom", "http://localhost:9999/v1")
    assert aux._runtime_provider_override == "custom"
    assert aux._runtime_base_url_override == "http://localhost:9999/v1"

    aux._runtime_model_override = ""
    aux._runtime_provider_override = ""
    aux._runtime_base_url_override = ""
