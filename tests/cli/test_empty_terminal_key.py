"""Tests: empty/null ``terminal:`` YAML key in config.yaml crashes load_cli_config.

Regression test for #58277 — when a profile's config.yaml contains an empty
``terminal:`` line, YAML parses it as ``None``.  The old code
``defaults.get("terminal", {})`` returns ``None`` because the key *exists*
(just with a null value), and ``"backend" in None`` raises TypeError.
"""

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return home


def _load_cli_config(home):
    import cli

    cli._hermes_home = home
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    return cli.load_cli_config()


def test_empty_terminal_key_does_not_crash(homes):
    """Empty ``terminal:`` (YAML null) must not crash load_cli_config()."""
    home = homes

    # Simulate the exact user config that triggers #58277
    (home / "config.yaml").write_text(
        "model:\n  provider: deepseek\n  default: deepseek-v4-flash\n"
        "terminal:\n"
        "toolsets:\n  - kanban\n",
        encoding="utf-8",
    )

    cfg = _load_cli_config(home)
    terminal = cfg.get("terminal")
    # Should be a dict (fallback), not None
    assert isinstance(terminal, dict), (
        f"terminal config should be a dict, got {type(terminal).__name__}: {terminal!r}"
    )


def test_terminal_config_preserved_when_present(homes):
    """Normal non-null terminal config still works."""
    home = homes
    (home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n  cwd: /workspace\n",
        encoding="utf-8",
    )

    cfg = _load_cli_config(home)
    terminal = cfg.get("terminal", {})
    assert isinstance(terminal, dict)
    assert "env_type" in terminal or "backend" in terminal
