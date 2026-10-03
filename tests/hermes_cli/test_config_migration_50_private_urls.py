"""Migration 49→50: ``browser.allow_private_urls`` no longer lifts the global private-address guard.

While ``security.allow_private_urls`` was unset the browser key used to stand in for it, so web,
vision and media fetches reached private addresses too. The step writes that effective opt-out to the
security key, where Settings → Safety shows it; an explicit security value is never overwritten.
Driven through the real ladder against a temp home.
"""

import os
from unittest.mock import patch

import pytest
import hermes_yaml as yaml


def _write(home, cfg):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _read(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))


def _step(home):
    from hermes_cli.config_migrations import run_migrations

    results = {"env_added": [], "config_added": [], "warnings": []}
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        run_migrations(49, results, quiet=True)
    return results


def _global_allow_private(home):
    from tools.url_safety import _resolve_allow_private_urls

    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        return _resolve_allow_private_urls()


@pytest.mark.parametrize(
    "security",
    [None, {}, {"redact_secrets": False}, {"allow_private_urls": None}],
    ids=["security-null", "security-empty", "security-key-unset", "security-key-null"],
)
def test_browser_opt_out_becomes_an_explicit_security_opt_out(tmp_path, monkeypatch, security):
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    home = tmp_path / "home"
    _write(home, {"_config_version": 49, "browser": {"allow_private_urls": True}, "security": security})
    assert _global_allow_private(home) is False  # the browser key alone no longer lifts the guard

    results = _step(home)

    raw = _read(home)
    assert raw["security"]["allow_private_urls"] is True
    assert raw["browser"]["allow_private_urls"] is True
    if security and "redact_secrets" in security:
        assert raw["security"]["redact_secrets"] is False
    assert results["config_added"] == [
        "security.allow_private_urls=true (carried over from browser.allow_private_urls)"]
    assert _global_allow_private(home) is True  # same effective policy as before the update


def test_missing_security_section_is_created(tmp_path):
    home = tmp_path / "home"
    _write(home, {"_config_version": 49, "browser": {"allow_private_urls": "yes"}})

    _step(home)

    assert _read(home)["security"] == {"allow_private_urls": True}


@pytest.mark.parametrize("value", [False, True, "false"])
def test_explicit_security_value_is_kept(tmp_path, value):
    home = tmp_path / "home"
    _write(home, {"_config_version": 49, "browser": {"allow_private_urls": True},
                  "security": {"allow_private_urls": value}})
    before = (home / "config.yaml").read_text(encoding="utf-8")

    results = _step(home)

    assert (home / "config.yaml").read_text(encoding="utf-8") == before
    assert results["config_added"] == []


@pytest.mark.parametrize(
    "browser",
    [{"allow_private_urls": False}, {"allow_private_urls": "false"}, {"headed": True}, None],
    ids=["false", "string-false", "key-absent", "section-absent"],
)
def test_no_browser_opt_out_writes_nothing(tmp_path, browser):
    home = tmp_path / "home"
    cfg = {"_config_version": 49, "security": {"redact_secrets": False}}
    if browser is not None:
        cfg["browser"] = browser
    _write(home, cfg)
    before = (home / "config.yaml").read_text(encoding="utf-8")

    results = _step(home)

    assert (home / "config.yaml").read_text(encoding="utf-8") == before
    assert results["config_added"] == []


def test_step_is_idempotent(tmp_path):
    home = tmp_path / "home"
    _write(home, {"_config_version": 49, "browser": {"allow_private_urls": True}})
    _step(home)
    after_first = (home / "config.yaml").read_text(encoding="utf-8")

    second = _step(home)

    assert (home / "config.yaml").read_text(encoding="utf-8") == after_first
    assert second["config_added"] == []


def test_unversioned_config_gets_the_step_and_the_current_stamp(tmp_path):
    """A never-stamped file (seeded from the template, then ``hermes config set``) relied on the fallback too."""
    from hermes_cli.config import migrate_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    home = tmp_path / "home"
    _write(home, {"browser": {"allow_private_urls": True}})
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        migrate_config(interactive=False, quiet=True)

    raw = _read(home)
    assert raw["security"]["allow_private_urls"] is True
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"] == 50


def test_settings_save_keeps_the_migrated_key_and_an_explicit_off(tmp_path):
    """Desktop/dashboard saves send the defaulted record back and strip default-equal values that are
    not already on disk: the migrated ``true`` differs from the default, and turning the Safety switch
    off afterwards is kept as an explicit ``false``."""
    from hermes_cli.config import load_config, save_config

    home = tmp_path / "home"
    _write(home, {"_config_version": 49, "browser": {"allow_private_urls": True}})
    _step(home)
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        save_config(load_config())
        assert _read(home)["security"]["allow_private_urls"] is True

        cfg = load_config()
        cfg["security"]["allow_private_urls"] = False
        save_config(cfg)
    raw = _read(home)
    assert raw["security"]["allow_private_urls"] is False
    assert raw["browser"]["allow_private_urls"] is True


def test_named_profile_is_migrated_by_update(monkeypatch, tmp_path):
    """``hermes update`` migrates every sibling profile's config.yaml through the same ladder."""
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.update_cmd as update_cmd
    import hermes_constants

    active = tmp_path / "profiles" / "active"
    sibling = tmp_path / "profiles" / "lan"
    _write(active, {"_config_version": 50})
    _write(sibling, {"_config_version": 49, "browser": {"allow_private_urls": True}})
    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: tmp_path / "profiles")
    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: tmp_path / "no-such-default-home")
    monkeypatch.setattr(hermes_constants, "get_process_hermes_home", lambda: active)

    migrated = update_cmd._migrate_sibling_profile_configs()

    assert ("lan", 49, 50) in migrated
    assert _read(sibling)["security"]["allow_private_urls"] is True
    assert "security" not in _read(active)
