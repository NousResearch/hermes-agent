"""#132515: managed-scope ``skills.external_dirs`` must reach skill discovery.

``agent.skill_utils`` deliberately bypasses ``hermes_cli.config`` (import-cycle guard),
which used to mean the managed overlay never entered the skill-discovery view: the key
was pinned, enforced as admin-immutable by the config pipeline, and silently inert.
These tests pin the merged behaviour of ``_load_raw_config()``/``get_external_skills_dirs()``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.skill_utils import (
    _external_dirs_cache_clear,
    _raw_config_cache_clear,
    get_external_skills_dirs,
)
from hermes_cli.managed_scope import invalidate_managed_cache


@pytest.fixture
def scopes(tmp_path, monkeypatch):
    """Isolated HERMES_HOME plus a managed scope; returns (home, managed_dir)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_caches()
    yield home, managed
    _reset_caches()


def _reset_caches() -> None:
    _raw_config_cache_clear()
    _external_dirs_cache_clear()
    invalidate_managed_cache()


def _write_managed_external_dirs(managed: Path, external: Path) -> None:
    (managed / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external}\n",
        encoding="utf-8",
    )
    invalidate_managed_cache()


def _bump_mtime(path: Path) -> None:
    stat = path.stat()
    future = stat.st_atime + 10
    os.utime(path, (future, future))


def test_managed_pin_overrides_profile_value(scopes):
    home, managed = scopes
    profile_external = home / "profile_skills"
    profile_external.mkdir()
    managed_external = home / "admin_skills"
    managed_external.mkdir()
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {profile_external}\n",
        encoding="utf-8",
    )
    _write_managed_external_dirs(managed, managed_external)

    # Leaf-level merge: the admin pin wins, the profile's own dir is replaced.
    assert get_external_skills_dirs() == [managed_external.resolve()]


def test_managed_dir_scanned_without_profile_skills_block(scopes):
    home, managed = scopes
    managed_external = home / "admin_skills"
    managed_external.mkdir()
    (home / "config.yaml").write_text("model:\n  default: x/y\n", encoding="utf-8")
    _write_managed_external_dirs(managed, managed_external)

    assert get_external_skills_dirs() == [managed_external.resolve()]


def test_managed_dir_scanned_with_no_profile_config_at_all(scopes):
    """A fully managed deployment has no profile config.yaml; the pin must still scan."""
    home, managed = scopes
    managed_external = home / "admin_skills"
    managed_external.mkdir()
    _write_managed_external_dirs(managed, managed_external)

    assert get_external_skills_dirs() == [managed_external.resolve()]


def test_managed_edit_invalidates_cached_dirs(scopes, tmp_path):
    home, managed = scopes
    first = home / "admin_first"
    first.mkdir()
    second = home / "admin_second"
    second.mkdir()
    # An unrelated profile file keeps the cache key non-None, so the second read
    # can only see the managed edit through the managed signature in the key.
    (home / "config.yaml").write_text("model:\n  default: x/y\n", encoding="utf-8")
    _write_managed_external_dirs(managed, first)
    assert get_external_skills_dirs() == [first.resolve()]

    # Rewrite the managed file; bump mtime so coarse-granularity filesystems see it.
    _write_managed_external_dirs(managed, second)
    _bump_mtime(managed / "config.yaml")

    assert get_external_skills_dirs() == [second.resolve()]


def test_malformed_managed_file_fails_open_to_profile_value(scopes):
    home, managed = scopes
    profile_external = home / "profile_skills"
    profile_external.mkdir()
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {profile_external}\n",
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text("skills: [unclosed", encoding="utf-8")
    invalidate_managed_cache()

    assert get_external_skills_dirs() == [profile_external.resolve()]
