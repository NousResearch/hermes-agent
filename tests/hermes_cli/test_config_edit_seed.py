"""Every writer that seeds or first-configures config.yaml must leave each messaging platform's display defaults
alone (#121230). The gateway loader merges no DEFAULT_CONFIG, so any written global ``display.<key>`` beats every
platform tier (e.g. Telegram/Slack tool_progress ``off`` -> ``all``, QQBot show_reasoning ``False`` -> ``True``)."""
import pytest

from tests.gateway.test_display_config import assert_keeps_platform_display_defaults


def _config_edit(tmp_path, monkeypatch, cfg, *, template):
    """`hermes config edit` on a home with no config.yaml (seeds via seed_config_file, like `doctor --fix`)."""
    monkeypatch.setenv("EDITOR", "true")
    monkeypatch.setattr(cfg.subprocess, "run", lambda *a, **k: None)
    if not template:
        monkeypatch.setattr(cfg, "get_project_root", lambda: tmp_path / "no-checkout")
    cfg.edit_config()


def _setup_agent_enter(tmp_path, monkeypatch, cfg):
    """`hermes setup agent` on a fresh home, pressing Enter (the offered default) on every prompt."""
    import hermes_cli.setup as setup

    monkeypatch.setattr(setup, "prompt", lambda question, default=None, *a, **k: default or "")
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *a, **k: False)
    setup.setup_agent_settings(cfg.load_config())


def _apply_default_agent_settings(tmp_path, monkeypatch, cfg):
    """Quick and full first-time setup."""
    from hermes_cli.setup import _apply_default_agent_settings

    _apply_default_agent_settings(cfg.load_config())


def _blank_slate(tmp_path, monkeypatch, cfg):
    from hermes_cli.setup_quick import _blank_slate_minimize_config

    config = cfg.load_config()
    _blank_slate_minimize_config(config)
    cfg.save_config(config)


SEEDERS = {
    "config-edit-template": lambda *a: _config_edit(*a, template=True),
    "config-edit-no-template": lambda *a: _config_edit(*a, template=False),
    "setup-agent-enter": _setup_agent_enter,
    "apply-default-agent-settings": _apply_default_agent_settings,
    "blank-slate": _blank_slate,
}


@pytest.mark.parametrize("seeder", list(SEEDERS))
def test_every_seeder_keeps_every_platform_display_default(tmp_path, monkeypatch, seeder):
    import hermes_cli.config as cfg
    from gateway.run import _load_gateway_config

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    SEEDERS[seeder](tmp_path, monkeypatch, cfg)

    config_path = home / "config.yaml"
    assert config_path.exists()  # the resolution checks below would pass on a missing file
    assert_keeps_platform_display_defaults(_load_gateway_config(config_path))
