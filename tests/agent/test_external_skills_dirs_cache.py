"""Guards for ``get_external_skills_dirs`` mtime-based memo.

``get_external_skills_dirs()`` is called once per skill during banner
construction and tool registration — on a typical install that's 120+
calls.  Without caching, each call re-reads + YAML-parses the full
config.yaml (~85ms each, 10+ seconds total).  This test pins the
behavior: first call parses, subsequent calls return cached result,
cache invalidates when config.yaml's mtime changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import skill_utils
from agent.skill_utils import (
    _external_dirs_cache_clear,
    get_external_skills_dirs,
)


@pytest.fixture
def hermes_home_with_config(tmp_path, monkeypatch):
    """Isolated ``~/.hermes/`` with a config.yaml referencing one external dir."""
    home = tmp_path / ".hermes"
    home.mkdir()
    external = tmp_path / "external_skills"
    external.mkdir()

    config = home / "config.yaml"
    config.write_text(
        "skills:\n"
        f"  external_dirs:\n"
        f"    - {external}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _external_dirs_cache_clear()
    yield home, external, config
    _external_dirs_cache_clear()






def test_cache_invalidates_on_mtime_change(hermes_home_with_config):
    """A config.yaml edit invalidates the cache on the next call."""
    _home, external, config = hermes_home_with_config
    other = external.parent / "other_skills"
    other.mkdir()

    # Prime cache with original contents.
    first = get_external_skills_dirs()
    assert first == [external.resolve()]

    # Rewrite config; bump mtime forward explicitly so filesystems with
    # coarse mtime granularity still register the change on fast test
    # systems.
    config.write_text(
        "skills:\n"
        f"  external_dirs:\n"
        f"    - {other}\n",
        encoding="utf-8",
    )
    stat = config.stat()
    future = stat.st_atime + 10
    os.utime(config, (future, future))

    second = get_external_skills_dirs()
    assert second == [other.resolve()]






def test_cache_key_is_per_config_path(tmp_path, monkeypatch):
    """Two different HERMES_HOMEs keep separate cache entries."""
    home_a = tmp_path / "home_a" / ".hermes"
    home_a.mkdir(parents=True)
    ext_a = tmp_path / "ext_a"
    ext_a.mkdir()
    (home_a / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {ext_a}\n", encoding="utf-8"
    )

    home_b = tmp_path / "home_b" / ".hermes"
    home_b.mkdir(parents=True)
    ext_b = tmp_path / "ext_b"
    ext_b.mkdir()
    (home_b / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {ext_b}\n", encoding="utf-8"
    )

    _external_dirs_cache_clear()

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert get_external_skills_dirs() == [ext_a.resolve()]

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert get_external_skills_dirs() == [ext_b.resolve()]

    # And switching back still works — both entries coexist in the cache.
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert get_external_skills_dirs() == [ext_a.resolve()]


# ---------------------------------------------------------------------------
# Managed Scope (#90040): skills.external_dirs declared in the managed config
# replaces the profile's list and must be discovered, with the managed file's
# signature keying the cache.
# ---------------------------------------------------------------------------


@pytest.fixture
def managed_scope_home(tmp_path, monkeypatch):
    """Isolated profile home + managed dir with a shared skill directory."""
    home = tmp_path / "home"
    home.mkdir()
    shared = tmp_path / "shared"
    (shared / "demo").mkdir(parents=True)
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "skills:\n"
        f"  external_dirs:\n"
        f"    - {shared}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _external_dirs_cache_clear()
    yield home, shared, managed
    _external_dirs_cache_clear()


def test_managed_only_external_dirs_discovered(managed_scope_home):
    """A managed-only skills.external_dirs is discovered even when the
    profile config declares no dirs."""
    _home, shared, _managed = managed_scope_home
    assert get_external_skills_dirs() == [shared.resolve()]


def test_managed_only_works_without_profile_config_file(managed_scope_home):
    """The profile home has NO config.yaml; managed scope alone drives it."""
    _home, shared, _managed = managed_scope_home
    from agent.skill_utils import get_config_path

    assert not get_config_path().exists()
    assert get_external_skills_dirs() == [shared.resolve()]


def test_managed_list_replaces_profile_list(managed_scope_home):
    """Managed Scope semantics: the managed list replaces the profile's
    list (same overlay rule as apply_managed_overlay for list leaves)."""
    home, shared, _managed = managed_scope_home
    profile_only = home / "profile_only"
    profile_only.mkdir()
    (home / "config.yaml").write_text(
        "skills:\n"
        f"  external_dirs:\n"
        f"    - {profile_only}\n",
        encoding="utf-8",
    )
    dirs = get_external_skills_dirs()
    assert dirs == [shared.resolve()]
    assert profile_only.resolve() not in dirs


def test_managed_edit_invalidates_cache(managed_scope_home):
    """Editing the managed config.yaml invalidates the discovery cache."""
    _home, shared, managed = managed_scope_home
    assert get_external_skills_dirs() == [shared.resolve()]

    other = managed.parent / "other_shared"
    other.mkdir()
    cfg = managed / "config.yaml"
    cfg.write_text(
        "skills:\n"
        f"  external_dirs:\n"
        f"    - {other}\n",
        encoding="utf-8",
    )
    stat = cfg.stat()
    future = stat.st_atime + 10
    os.utime(cfg, (future, future))

    assert get_external_skills_dirs() == [other.resolve()]
