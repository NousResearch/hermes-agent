"""Invariants for ``hermes_cli.config_effective.load_user_config_effective`` — the one loader every
defaults-free config reader (gateway runtime, TUI gateway, cron, ``hermes send`` bridge, doctor,
bootstrap modules) goes through."""
import textwrap

import pytest
import yaml


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("FIXTURE_USER_KEY", "user-secret")
    monkeypatch.setenv("FIXTURE_MANAGED_URL", "https://managed.example")
    _reset_caches()
    return home, managed


def _reset_caches():
    import hermes_cli.config as cfg
    from hermes_cli import config_effective, managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()
    config_effective._LAST_GOOD_USER_RAW.clear()
    managed_scope.invalidate_managed_cache()


def _write(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    _reset_caches()


USER_YAML = """
    model:
      name: user/model
      api_key: ${FIXTURE_USER_KEY}
    provider: custom
    display:
      skin: user-skin
    """
MANAGED_YAML = """
    model:
      base_url: ${FIXTURE_MANAGED_URL}
    display:
      skin: managed-skin
    """


def test_effective_is_user_plus_managed_plus_env_with_no_defaults(homes):
    """Contract as a fixture: given user config.yaml X, managed overlay Y and env Z, the effective
    dict is exactly this literal — ``${VAR}`` expanded on both layers, managed keys winning,
    root ``provider`` migrated under ``model``, and no DEFAULT_CONFIG key introduced (a missing
    key stays missing). Per-message gateway reads (and the system prompt built from them) are
    pinned by this shape, not by re-running the implementation's primitives."""
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.config_effective import load_user_config_effective

    home, managed = homes
    _write(home / "config.yaml", USER_YAML)
    _write(managed / "config.yaml", MANAGED_YAML)

    effective = load_user_config_effective(home / "config.yaml")

    assert effective == {
        "model": {
            "default": "user/model",
            "provider": "custom",
            "api_key": "user-secret",
            "base_url": "https://managed.example",
        },
        "display": {"skin": "managed-skin"},
    }
    assert "agent" in DEFAULT_CONFIG  # would be present if defaults had been merged


def test_broken_yaml_serves_last_good_and_fail_closed_raises(homes):
    """A torn mid-edit write must not silently drop user overrides: the fail-open path serves the last
    successfully parsed user file through the same pipeline; ``fail_closed`` surfaces the error to
    callers that keep their own last-good state."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    _write(home / "config.yaml", USER_YAML)
    good = load_user_config_effective(home / "config.yaml")

    (home / "config.yaml").write_text("model: [unterminated", encoding="utf-8")
    _reset_caches_keep_last_good()

    assert load_user_config_effective(home / "config.yaml") == good
    with pytest.raises(yaml.YAMLError):  # the type _refresh_fallback_model's own last-good path keys on
        load_user_config_effective(home / "config.yaml", fail_closed=True)


def test_good_backup_is_written_only_for_the_active_home(homes, tmp_path):
    """Reading ANOTHER profile's config (doctor, TUI cwd lookup) is a read: it must not create
    ``backups/config/`` inside that profile. The active home keeps the last-good copy."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    other = tmp_path / "other-profile"
    other.mkdir()
    _write(home / "config.yaml", USER_YAML)
    _write(other / "config.yaml", USER_YAML)

    load_user_config_effective(other / "config.yaml")
    load_user_config_effective(home / "config.yaml")

    assert not (other / "backups").exists()
    assert list((home / "backups" / "config").glob("config.yaml.good.*"))


def _reset_caches_keep_last_good():
    import hermes_cli.config as cfg
    from hermes_cli import config_effective

    cfg._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()


def test_raw_cache_hit_replaces_last_good_and_backup(homes):
    """Regression for #109876: a snapshot accepted through the shared raw cache (refreshed by
    another raw-config caller such as ``read_raw_config()``) must become the last-known-good AND
    the newest durable good backup — so a later torn write recovers the NEWER accepted policy,
    never the first-ever snapshot (older ``approvals``, providers, routing)."""
    import time

    from hermes_cli import config
    from hermes_cli.config_backups import load_newest_good_backup
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    cfg = home / "config.yaml"
    _write(cfg, "approvals:\n  deny: [old]\n")
    assert load_user_config_effective(cfg)["approvals"]["deny"] == ["old"]
    assert load_newest_good_backup(cfg)["approvals"]["deny"] == ["old"]

    time.sleep(1.1)  # good copies are named at second granularity; keep the v1 and v2 copies distinct
    cfg.write_text("approvals:\n  deny: [new]\n", encoding="utf-8")  # the file moves on, caches stay
    assert config.read_raw_config()["approvals"]["deny"] == ["new"]
    assert load_user_config_effective(cfg)["approvals"]["deny"] == ["new"]  # served off the raw-cache hit
    assert load_newest_good_backup(cfg)["approvals"]["deny"] == ["new"]

    cfg.write_text("approvals: [unterminated\n", encoding="utf-8")
    _reset_caches_keep_last_good()
    assert load_user_config_effective(cfg)["approvals"]["deny"] == ["new"]


def test_good_backup_matches_active_home_by_inode(homes):
    """Regression for #109876: the active-home gate matches by inode, not lexical Path equality
    — reading through a symlinked alias of the active config.yaml still publishes the good copy
    UNDER THE ACTIVE NAME (a fresh process recovering through the canonical path finds it),
    while another profile's file still never creates backups/ inside that profile."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    _write(home / "config.yaml", USER_YAML)
    alias = home / "alias-config.yaml"
    alias.symlink_to(home / "config.yaml")

    load_user_config_effective(alias)

    assert list((home / "backups" / "config").glob("config.yaml.good.*"))
    assert not list((home / "backups" / "config").glob("alias-config.yaml.*"))
