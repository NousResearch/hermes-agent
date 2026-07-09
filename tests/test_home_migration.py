"""Phase 6: backward-compatible ~/.hermes -> ~/.ht-ai-agent data-dir migration.

The on-disk default flips to the HT-branded ~/.ht-ai-agent, but an existing
legacy ~/.hermes is honored in place until `maybe_migrate_home()` moves it —
atomically, all-or-nothing, with a back-compat symlink so anything still
referencing the old path keeps working. Nothing runs at import time and nothing
touches real directories under tests.
"""

import os

import pytest

import hermes_constants
from hermes_constants import (
    _get_platform_default_hermes_home,
    _legacy_default_home,
    _migrate_legacy_home,
    _new_default_home,
    _provision_fresh_home,
    maybe_migrate_home,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake POSIX home with no HERMES_HOME/HT_HOME override in effect."""
    monkeypatch.setattr(hermes_constants.sys, "platform", "linux")
    monkeypatch.setattr(hermes_constants.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HT_HOME", raising=False)
    return tmp_path


class TestDefaultResolution:
    def test_fresh_install_uses_new(self, home):
        assert _get_platform_default_hermes_home() == home / ".ht-ai-agent"

    def test_existing_legacy_honored_in_place(self, home):
        (home / ".hermes").mkdir()
        assert _get_platform_default_hermes_home() == home / ".hermes"

    def test_new_wins_when_both_exist(self, home):
        (home / ".hermes").mkdir()
        (home / ".ht-ai-agent").mkdir()
        assert _get_platform_default_hermes_home() == home / ".ht-ai-agent"


class TestMigrateLegacyHome:
    def test_migrates_populated_dir_with_backsymlink(self, home):
        legacy = home / ".hermes"
        new = home / ".ht-ai-agent"
        legacy.mkdir()
        (legacy / "config.yaml").write_text("model: x\n")
        (legacy / "sessions").mkdir()

        assert _migrate_legacy_home(legacy, new) is True
        # Data moved to the new location.
        assert (new / "config.yaml").read_text() == "model: x\n"
        assert (new / "sessions").is_dir()
        # Back-compat symlink left at the old path, pointing at the new one.
        assert legacy.is_symlink()
        assert legacy.resolve() == new.resolve()
        # Old references still resolve through the symlink.
        assert (legacy / "config.yaml").read_text() == "model: x\n"

    def test_noop_when_new_already_exists(self, home):
        legacy = home / ".hermes"
        new = home / ".ht-ai-agent"
        legacy.mkdir()
        (legacy / "a").write_text("1")
        new.mkdir()

        assert _migrate_legacy_home(legacy, new) is False
        # Legacy left untouched (not a symlink, still real).
        assert legacy.is_dir() and not legacy.is_symlink()
        assert (legacy / "a").read_text() == "1"

    def test_noop_when_legacy_missing(self, home):
        assert _migrate_legacy_home(home / ".hermes", home / ".ht-ai-agent") is False
        assert not (home / ".ht-ai-agent").exists()

    def test_noop_when_legacy_empty(self, home):
        legacy = home / ".hermes"
        legacy.mkdir()
        assert _migrate_legacy_home(legacy, home / ".ht-ai-agent") is False
        assert legacy.is_dir() and not legacy.is_symlink()
        assert not (home / ".ht-ai-agent").exists()

    def test_noop_when_legacy_is_symlink(self, home):
        # An already-migrated legacy (symlink) must not be re-migrated.
        target = home / "elsewhere"
        target.mkdir()
        (target / "x").write_text("1")
        legacy = home / ".hermes"
        legacy.symlink_to(target)
        assert _migrate_legacy_home(legacy, home / ".ht-ai-agent") is False

    def test_rolls_back_when_symlink_fails(self, home, monkeypatch):
        legacy = home / ".hermes"
        new = home / ".ht-ai-agent"
        legacy.mkdir()
        (legacy / "config.yaml").write_text("keep\n")

        # Simulate a platform where symlink creation is not permitted.
        def _boom(*a, **k):
            raise OSError("symlink not permitted")

        monkeypatch.setattr(hermes_constants.Path, "symlink_to", _boom)

        assert _migrate_legacy_home(legacy, new) is False
        # Rolled back: legacy still holds the data, new does not exist.
        assert legacy.is_dir() and not legacy.is_symlink()
        assert (legacy / "config.yaml").read_text() == "keep\n"
        assert not new.exists()


class TestProvisionFreshHome:
    def test_creates_new_and_legacy_symlink(self, home):
        new = home / ".ht-ai-agent"
        legacy = home / ".hermes"

        assert _provision_fresh_home(new, legacy) is True
        assert new.is_dir()
        # Legacy path is a symlink to the new home, so hardcoded ~/.hermes
        # fallbacks resolve to the same directory.
        assert legacy.is_symlink()
        assert legacy.resolve() == new.resolve()
        (new / "config.yaml").write_text("k: v\n")
        assert (legacy / "config.yaml").read_text() == "k: v\n"

    def test_noop_when_new_exists(self, home):
        new = home / ".ht-ai-agent"
        new.mkdir()
        assert _provision_fresh_home(new, home / ".hermes") is False
        assert not (home / ".hermes").exists()

    def test_noop_when_legacy_exists(self, home):
        legacy = home / ".hermes"
        legacy.mkdir()
        assert _provision_fresh_home(home / ".ht-ai-agent", legacy) is False

    def test_new_created_even_if_symlink_fails(self, home, monkeypatch):
        def _boom(*a, **k):
            raise OSError("symlink not permitted")

        monkeypatch.setattr(hermes_constants.Path, "symlink_to", _boom)
        new = home / ".ht-ai-agent"
        assert _provision_fresh_home(new, home / ".hermes") is True
        assert new.is_dir()
        assert not (home / ".hermes").exists()  # bridge skipped, no split-dir stub


class TestMaybeMigrateHome:
    def test_skips_under_pytest(self, home):
        # PYTEST_CURRENT_TEST is set while a test runs, so this always skips.
        legacy = home / ".hermes"
        legacy.mkdir()
        (legacy / "a").write_text("1")
        assert maybe_migrate_home() is False
        assert legacy.is_dir() and not legacy.is_symlink()

    def test_skips_when_hermes_home_override_set(self, home, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(home / "custom"))
        legacy = home / ".hermes"
        legacy.mkdir()
        (legacy / "a").write_text("1")
        assert maybe_migrate_home() is False
        assert legacy.is_dir() and not legacy.is_symlink()

    def test_skips_when_ht_home_override_set(self, home, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("HT_HOME", str(home / "custom"))
        legacy = home / ".hermes"
        legacy.mkdir()
        (legacy / "a").write_text("1")
        assert maybe_migrate_home() is False

    def test_skips_when_skip_flag_set(self, home, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("HT_SKIP_HOME_MIGRATION", "1")
        legacy = home / ".hermes"
        legacy.mkdir()
        (legacy / "a").write_text("1")
        assert maybe_migrate_home() is False

    def test_migrates_existing_when_unguarded(self, home, monkeypatch):
        # Clear every guard to exercise the real migration path.
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("HT_SKIP_HOME_MIGRATION", raising=False)
        legacy = _legacy_default_home()
        new = _new_default_home()
        legacy.mkdir()
        (legacy / "config.yaml").write_text("model: x\n")

        assert maybe_migrate_home() is True
        assert (new / "config.yaml").read_text() == "model: x\n"
        assert legacy.is_symlink() and legacy.resolve() == new.resolve()

    def test_provisions_fresh_when_unguarded(self, home, monkeypatch):
        # Fresh install: neither home exists → create new + legacy symlink.
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("HT_SKIP_HOME_MIGRATION", raising=False)
        legacy = _legacy_default_home()
        new = _new_default_home()

        assert maybe_migrate_home() is True
        assert new.is_dir()
        assert legacy.is_symlink() and legacy.resolve() == new.resolve()

    def test_noop_when_already_migrated(self, home, monkeypatch):
        # New already exists (already migrated) → nothing to do.
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        _new_default_home().mkdir()
        assert maybe_migrate_home() is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
