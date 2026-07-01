from types import SimpleNamespace

from hermes_cli import moa_cmd


def test_cmd_moa_activate_persists_active_preset(monkeypatch, capsys):
    saved = {}
    config = {
        "moa": {
            "default_preset": "default",
            "presets": {
                "default": {},
                "review": {"reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}]},
            },
        }
    }

    monkeypatch.setattr(moa_cmd, "load_config", lambda: config)
    monkeypatch.setattr(moa_cmd, "save_config", lambda cfg: saved.update(cfg))

    moa_cmd.cmd_moa(SimpleNamespace(moa_command="activate", name="review"))

    assert saved["moa"]["active_preset"] == "review"
    out = capsys.readouterr().out
    assert "Activated MoA preset for /moa: review" in out
    assert "Effective for /moa: review" in out


def test_cmd_moa_deactivate_clears_active_preset(monkeypatch, capsys):
    saved = {}
    config = {
        "moa": {
            "default_preset": "default",
            "active_preset": "review",
            "presets": {
                "default": {},
                "review": {"reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}]},
            },
        }
    }

    monkeypatch.setattr(moa_cmd, "load_config", lambda: config)
    monkeypatch.setattr(moa_cmd, "save_config", lambda cfg: saved.update(cfg))

    moa_cmd.cmd_moa(SimpleNamespace(moa_command="deactivate"))

    assert saved["moa"]["active_preset"] == ""
    out = capsys.readouterr().out
    assert "Deactivated MoA preset override" in out
    assert "Effective for /moa: default" in out
