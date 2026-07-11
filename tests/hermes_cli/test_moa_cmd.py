from hermes_cli.moa_cmd import _print_config


def _config(*, provider="moa", model="full", legacy_active=""):
    return {
        "model": {"provider": provider, "default": model},
        "moa": {
            "default_preset": "full",
            "active_preset": legacy_active,
            "presets": {
                "full": {
                    "reference_models": [{"provider": "zai", "model": "glm-5.2"}],
                    "aggregator": {"provider": "zai", "model": "glm-5.2"},
                }
            },
        },
    }


def test_print_config_reports_main_model_moa_preset_as_active(capsys):
    _print_config(_config())

    assert "Active in config: full" in capsys.readouterr().out


def test_print_config_reports_off_when_main_provider_is_not_moa(capsys):
    _print_config(_config(provider="zai", model="glm-5.2"))

    assert "Active in config: (off)" in capsys.readouterr().out


def test_print_config_keeps_legacy_active_preset_compatibility(capsys):
    _print_config(_config(provider="zai", model="glm-5.2", legacy_active="full"))

    assert "Active in config: full (legacy)" in capsys.readouterr().out
