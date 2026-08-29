"""Fleet-wide config migration (#91277 Phase 2 — #20438/#54926/#79048 class).

`hermes update` migrated only the active profile's config.yaml; sibling
profiles silently drifted config versions until their gateway hit a config
the new code couldn't read. `_migrate_sibling_profile_configs()` runs the
same non-interactive safe migration for every sibling home, scoped via the
context-local HERMES_HOME override.

These tests use REAL config files on disk and the REAL migration pipeline —
only the profile-root location is pointed at tmp_path.
"""

import yaml
from pathlib import Path

import hermes_cli.update_cmd as update_cmd


def _write_profile(root: Path, name: str, version: int) -> Path:
    home = root / name
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"_config_version": version, "model": {"provider": "nous"}}),
        encoding="utf-8",
    )
    return home


def _latest_version() -> int:
    from hermes_cli.config import DEFAULT_CONFIG

    return int(DEFAULT_CONFIG["_config_version"])


def _setup(monkeypatch, tmp_path, active_home: Path):
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: tmp_path / "profiles")
    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "get_process_hermes_home", lambda: active_home
    )
    monkeypatch.setattr(
        update_cmd, "_reload_config_modules", lambda: None
    )  # module reload is orthogonal here; the real one re-imports from disk


def test_sibling_behind_is_migrated_on_disk(monkeypatch, tmp_path):
    active = _write_profile(tmp_path / "profiles", "active", _latest_version())
    sibling = _write_profile(tmp_path / "profiles", "research", 12)
    _setup(monkeypatch, tmp_path, active)

    migrated = update_cmd._migrate_sibling_profile_configs()

    names = [m[0] for m in migrated]
    assert "research" in names
    entry = next(m for m in migrated if m[0] == "research")
    assert entry[1] == 12 and entry[2] == _latest_version()
    # REAL file on disk carries the new version
    on_disk = yaml.safe_load((sibling / "config.yaml").read_text())
    assert on_disk["_config_version"] == _latest_version()
    # and user settings survived
    assert on_disk["model"]["provider"] == "nous"


def test_active_profile_is_skipped(monkeypatch, tmp_path):
    active = _write_profile(tmp_path / "profiles", "active", 12)
    _setup(monkeypatch, tmp_path, active)

    migrated = update_cmd._migrate_sibling_profile_configs()

    assert "active" not in [m[0] for m in migrated]
    # active home untouched (the caller's own migration handles it)
    on_disk = yaml.safe_load((active / "config.yaml").read_text())
    assert on_disk["_config_version"] == 12


def test_current_sibling_untouched(monkeypatch, tmp_path):
    active = _write_profile(tmp_path / "profiles", "active", _latest_version())
    sibling = _write_profile(tmp_path / "profiles", "work", _latest_version())
    _setup(monkeypatch, tmp_path, active)
    before = (sibling / "config.yaml").read_bytes()

    migrated = update_cmd._migrate_sibling_profile_configs()

    assert migrated == []
    assert (sibling / "config.yaml").read_bytes() == before


def test_unconfigured_profile_skipped(monkeypatch, tmp_path):
    active = _write_profile(tmp_path / "profiles", "active", _latest_version())
    bare = tmp_path / "profiles" / "empty"
    bare.mkdir()
    _setup(monkeypatch, tmp_path, active)

    assert update_cmd._migrate_sibling_profile_configs() == []
    assert not (bare / "config.yaml").exists()


def test_one_broken_profile_does_not_block_others(monkeypatch, tmp_path):
    active = _write_profile(tmp_path / "profiles", "active", _latest_version())
    broken_home = tmp_path / "profiles" / "broken"
    broken_home.mkdir()
    (broken_home / "config.yaml").write_text(":\nnot yaml: [", encoding="utf-8")
    _write_profile(tmp_path / "profiles", "healthy", 12)
    _setup(monkeypatch, tmp_path, active)

    migrated = update_cmd._migrate_sibling_profile_configs()

    assert [m[0] for m in migrated] == ["healthy"]


def test_override_is_reset_after_run(monkeypatch, tmp_path):
    """The ContextVar override must not leak past the sweep."""
    from hermes_constants import get_hermes_home_override

    active = _write_profile(tmp_path / "profiles", "active", _latest_version())
    _write_profile(tmp_path / "profiles", "research", 12)
    _setup(monkeypatch, tmp_path, active)

    before = get_hermes_home_override()
    update_cmd._migrate_sibling_profile_configs()
    assert get_hermes_home_override() == before
