"""``/model <name>`` must only match MoA presets the user actually configured.

``DEFAULT_CONFIG["moa"]["presets"]`` ships an enabled preset literally named
``default``, and ``switch_model`` resolved MoA preset names from
``load_config()``, which merges DEFAULT_CONFIG. On a stock install
``/model default`` therefore matched that shipped preset and silently switched
the session into MoA mode, routing every later turn through the
reference-model fan-out.

``inventory.raw_config_has_enabled_moa_preset()`` already draws this line for
the model pickers ("the DEFAULT_CONFIG preset is not a user choice"); these
tests pin it for the switch path too.

Note that reading from ``read_raw_config()`` is not sufficient on its own:
``normalize_moa_config({})`` synthesizes a ``default`` preset for an empty
config, so the gate is what actually does the work.

The ``_switch`` helper patches both config seams so each test means the same
thing before and after the fix: ``read_raw_config`` returns the user's raw
config (what the fixed code reads), and ``load_config`` returns that config
merged over the real shipped ``DEFAULT_CONFIG["moa"]`` (what the unfixed code
reads). ``test_user_configured_preset_still_routes_into_moa`` is a regression
guard and passes on unfixed code too; the other two demonstrate the bug and
fail without the fix.
"""

import copy

from unittest.mock import patch

import pytest
from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.model_switch import switch_model


_USER_RAW_CONFIG = {
    "moa": {
        "presets": {
            "review": {
                "enabled": True,
                "reference_models": [
                    {"provider": "openai-codex", "model": "gpt-5.5"},
                ],
                "aggregator": {
                    "provider": "openrouter",
                    "model": "anthropic/claude-opus-4.8",
                },
            },
        },
    },
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: {
            "accepted": True, "persist": True, "recognized": True, "message": None,
        },
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None,
    )
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_catalog.get_catalog", lambda *a, **k: {})


def _merged_config(raw_config):
    """What ``load_config()`` would return: user config over DEFAULT_CONFIG.

    Only the ``moa`` section matters to the code under test. Using the real
    shipped DEFAULT_CONFIG keeps the pre-fix behaviour honest instead of
    hand-writing a stand-in for the ``default`` preset.
    """
    moa = copy.deepcopy(DEFAULT_CONFIG.get("moa") or {})
    user_moa = (raw_config or {}).get("moa") or {}
    presets = dict(moa.get("presets") or {})
    presets.update(user_moa.get("presets") or {})
    moa.update({k: v for k, v in user_moa.items() if k != "presets"})
    moa["presets"] = presets
    return {"moa": moa}


def _switch(raw_input, raw_config):
    with patch("hermes_cli.config.read_raw_config", return_value=raw_config), \
         patch("hermes_cli.config.load_config", return_value=_merged_config(raw_config)):
        return switch_model(
            raw_input=raw_input,
            current_provider="openrouter",
            current_model="anthropic/claude-opus-4.8",
            user_providers={},
            custom_providers=[],
        )


def test_stock_config_does_not_route_model_default_into_moa():
    """The shipped DEFAULT_CONFIG preset is not a user-selectable switch target."""
    result = _switch("default", {})

    assert result.target_provider != "moa"


def test_user_configured_preset_still_routes_into_moa():
    """A preset the user did write in config.yaml keeps working (pre- and post-fix)."""
    result = _switch("review", _USER_RAW_CONFIG)

    assert result.target_provider == "moa"
    assert result.new_model == "review"


def test_model_default_does_not_match_when_user_configured_a_different_preset():
    """Having *some* MoA config must not resurrect the shipped "default" preset."""
    result = _switch("default", _USER_RAW_CONFIG)

    assert result.target_provider != "moa"
