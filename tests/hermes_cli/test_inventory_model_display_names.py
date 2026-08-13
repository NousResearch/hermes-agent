"""Tests for attach_custom_model_display_names (hermes_cli.inventory).

Pure-unit: the function is exercised directly with a payload + ConfigContext,
so no provider probe / network is involved. Pins that a custom provider row
gets a model_display_names map sourced from the config models dict `name`
field, and that built-in rows are left untouched.
"""
from hermes_cli.inventory import ConfigContext, attach_custom_model_display_names


def _ctx(custom_providers):
    return ConfigContext(
        current_provider="",
        current_model="",
        current_base_url="",
        user_providers={},
        custom_providers=custom_providers,
    )


def test_attaches_for_custom_provider():
    ctx = _ctx(
        [
            {
                "name": "Lab",
                "base_url": "https://lab/v1",
                "models": {"a": {"name": "Alpha"}, "b": {}},
            }
        ]
    )
    payload = {"providers": [{"slug": "custom:lab", "models": ["a", "b"]}]}

    attach_custom_model_display_names(payload, ctx)

    assert payload["providers"][0]["model_display_names"] == {"a": "Alpha"}


def test_builtin_provider_untouched():
    ctx = _ctx([])
    payload = {"providers": [{"slug": "openai", "models": ["gpt-4o"]}]}

    attach_custom_model_display_names(payload, ctx)

    assert "model_display_names" not in payload["providers"][0]


def test_normalized_name_match():
    ctx = _ctx(
        [
            {
                "name": "My Lab",
                "models": {"x": {"name": "X Label"}},
            }
        ]
    )
    payload = {"providers": [{"slug": "custom:my-lab", "models": ["x"]}]}

    attach_custom_model_display_names(payload, ctx)

    assert payload["providers"][0]["model_display_names"] == {"x": "X Label"}
