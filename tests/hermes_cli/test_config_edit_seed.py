"""``hermes config edit`` on a home with no config.yaml seeds one. The seed must not pin a display value over
any messaging platform's own default (the gateway loader merges no DEFAULT_CONFIG, so every written key is explicit)."""
import pytest


@pytest.mark.parametrize("seed", ["template", "no-template"])
def test_config_edit_seed_keeps_every_platform_display_default(tmp_path, monkeypatch, seed):
    import hermes_cli.config as cfg
    from gateway.display_config import _PLATFORM_DEFAULTS, resolve_display_setting, resolve_tool_progress
    from gateway.run import _load_gateway_config

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("EDITOR", "true")
    monkeypatch.setattr(cfg.subprocess, "run", lambda *a, **k: None)
    if seed == "no-template":
        monkeypatch.setattr(cfg, "get_project_root", lambda: tmp_path / "no-checkout")

    cfg.edit_config()

    config_path = home / "config.yaml"
    assert config_path.exists()  # the resolution checks below would pass on a missing file
    seeded = _load_gateway_config(config_path)
    tier_keys = {key for tier in _PLATFORM_DEFAULTS.values() for key in tier}
    for platform in _PLATFORM_DEFAULTS:
        assert resolve_tool_progress(seeded, platform) == resolve_tool_progress({}, platform), platform
        for key in tier_keys:
            assert resolve_display_setting(seeded, platform, key) == resolve_display_setting({}, platform, key), (
                platform, key)
