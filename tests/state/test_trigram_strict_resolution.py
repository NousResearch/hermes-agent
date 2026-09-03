"""Strict fail-closed config resolution for sessions.trigram_fts (P1 Tension A).

A quarantined trigram index must never be silently re-enabled because the
config *source* became unreadable in a way that looks like "absent":
dangling symlinks, non-mapping YAML roots, unreadable/special files, or a
managed overlay that fails open. Every such case resolves to
``ConfigResolutionError`` → ``None`` → the durable on-disk quarantine marker
stays authoritative.

These tests exercise the public ``resolve_effective_config_value`` resolver
and real disposable file trees (no monkeypatched ``_trigram_enabled``).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

import hermes_cli.managed_scope as managed_scope
from hermes_cli.config import (
    ConfigResolutionError,
    resolve_effective_config_value,
)
from hermes_state import SessionDB
from hermes_state_common import FTS_TRIGRAM_STALE_KEY

_FTS_TRIGRAM_TRIGGERS = (
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)


def _trigger_names(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }


def _has_marker(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM state_meta WHERE key = ? LIMIT 1",
            (FTS_TRIGRAM_STALE_KEY,),
        ).fetchone()
        is not None
    )


def _write_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_caches():
    managed_scope.invalidate_managed_cache()
    yield
    managed_scope.invalidate_managed_cache()


def _quarantined_db(tmp_path: Path, name: str = "state.db") -> Path:
    """Fresh trigram DB, seeded, then deliberately quarantined via false."""
    path = tmp_path / name
    db = SessionDB(db_path=path)
    if not db._trigram_available:
        db.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    db.create_session("root", source="cli")
    db.append_message("root", role="user", content="交付状态正常 root-word")
    db.close()
    _write_config(path.parent / "config.yaml", "sessions:\n  trigram_fts: false\n")
    disabled = SessionDB(db_path=path)
    assert disabled._trigram_available is False
    disabled.close()
    assert _has_marker(_raw_connect(path))
    return path


def _raw_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# ── 1-3. Quarantined DB + broken user config source ─────────────────────


def test_quarantined_db_user_dangling_symlink_stays_quarantined(tmp_path):
    """Dangling user config symlink must resolve as an ERROR (marker wins),
    never as "absent → default true" which would re-enable the index."""
    path = _quarantined_db(tmp_path)
    cfg = path.parent / "config.yaml"
    cfg.unlink()
    outside = tmp_path / "does-not-exist.yaml"
    cfg.symlink_to(outside)

    # Resolver level: strict error, not a silent default.
    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)

    # State level: marker honored, triggers stay down, storage untouched.
    db = SessionDB(db_path=path)
    try:
        assert db._trigram_enabled is False
        assert db._trigram_available is False
        conn = _raw_connect(path)
        try:
            assert _has_marker(conn)
            names = _trigger_names(conn)
            for trig in _FTS_TRIGRAM_TRIGGERS:
                assert trig not in names
        finally:
            conn.close()
    finally:
        db.close()


def test_quarantined_db_non_mapping_user_yaml_stays_quarantined(tmp_path):
    path = _quarantined_db(tmp_path)
    cfg = path.parent / "config.yaml"
    for broken in ("[]\n", "- a\n- b\n", "5\n", "'just a scalar'\n"):
        _write_config(cfg, broken)
        with pytest.raises(ConfigResolutionError):
            resolve_effective_config_value(
                cfg, "sessions", "trigram_fts", default=True
            )
    db = SessionDB(db_path=path)
    try:
        assert db._trigram_enabled is False
        assert db._trigram_available is False
        assert _has_marker(_raw_connect(path))
    finally:
        db.close()


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="chmod 000 is readable for root")
def test_quarantined_db_unreadable_user_config_stays_quarantined(tmp_path):
    path = _quarantined_db(tmp_path)
    cfg = path.parent / "config.yaml"
    cfg.chmod(0o000)
    try:
        with pytest.raises(ConfigResolutionError):
            resolve_effective_config_value(
                cfg, "sessions", "trigram_fts", default=True
            )
    finally:
        cfg.chmod(0o644)


def test_quarantined_db_special_user_path_stays_quarantined(tmp_path):
    """A directory (or any non-regular file) at the config path is an error."""
    path = _quarantined_db(tmp_path)
    cfg = path.parent / "config.yaml"
    cfg.unlink()
    cfg.mkdir()
    try:
        with pytest.raises(ConfigResolutionError):
            resolve_effective_config_value(
                cfg, "sessions", "trigram_fts", default=True
            )
        db = SessionDB(db_path=path)
        try:
            assert db._trigram_enabled is False
            assert _has_marker(_raw_connect(path))
        finally:
            db.close()
    finally:
        cfg.rmdir()


# ── 4. Malformed user YAML control (was already fail-closed) ────────────


def test_malformed_user_yaml_control_remains_fail_closed(tmp_path):
    path = _quarantined_db(tmp_path)
    _write_config(path.parent / "config.yaml", "sessions: [unclosed\n")
    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(
            path.parent / "config.yaml", "sessions", "trigram_fts", default=True
        )
    db = SessionDB(db_path=path)
    try:
        assert db._trigram_enabled is False
    finally:
        db.close()


# ── 5-6. Managed overlay strictness ─────────────────────────────────────


@pytest.fixture
def managed_dir(tmp_path, monkeypatch):
    d = tmp_path / "managed"
    d.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(d))
    return d


def _quarantined_with_absent_user_config(tmp_path: Path) -> Path:
    path = _quarantined_db(tmp_path)
    (path.parent / "config.yaml").unlink()
    return path


def test_quarantined_db_malformed_managed_yaml_stays_quarantined(
    tmp_path, managed_dir
):
    path = _quarantined_with_absent_user_config(tmp_path)
    _write_config(managed_dir / "config.yaml", "sessions: [unclosed\n")
    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(
            path.parent / "config.yaml", "sessions", "trigram_fts", default=True
        )
    db = SessionDB(db_path=path)
    try:
        assert db._trigram_enabled is False
        assert _has_marker(_raw_connect(path))
    finally:
        db.close()


@pytest.mark.parametrize(
    "managed_body",
    [
        "[]\n",                       # non-mapping root
        "5\n",                        # scalar root
        "sessions: [unclosed\n",      # malformed YAML
    ],
    ids=["non-mapping-list", "scalar", "malformed"],
)
def test_quarantined_db_broken_managed_variants_stay_quarantined(
    tmp_path, managed_dir, managed_body
):
    path = _quarantined_with_absent_user_config(tmp_path)
    _write_config(managed_dir / "config.yaml", managed_body)
    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(
            path.parent / "config.yaml", "sessions", "trigram_fts", default=True
        )


def test_quarantined_db_dangling_managed_symlink_stays_quarantined(
    tmp_path, managed_dir
):
    path = _quarantined_with_absent_user_config(tmp_path)
    link = managed_dir / "config.yaml"
    link.symlink_to(tmp_path / "managed-missing.yaml")
    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(
            path.parent / "config.yaml", "sessions", "trigram_fts", default=True
        )


# ── 7. Non-strict prewarm cannot poison strict resolution ───────────────


def test_non_strict_prewarm_cannot_bypass_strict_failure(tmp_path, managed_dir):
    path = _quarantined_db(tmp_path)
    cfg = path.parent / "config.yaml"
    # Malformed managed overlay + dangling user symlink.
    _write_config(managed_dir / "config.yaml", "[]\n")
    cfg.unlink()
    cfg.symlink_to(tmp_path / "nope.yaml")

    # Non-strict prewarm of BOTH caches (fail-open paths).
    assert managed_scope.load_managed_config() == {}
    from hermes_cli.config import read_user_config_raw

    assert read_user_config_raw(cfg) == {}

    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)

    db = SessionDB(db_path=path)
    try:
        assert db._trigram_enabled is False
        assert _has_marker(_raw_connect(path))
    finally:
        db.close()


# ── 8. Cache provenance / invalidation ──────────────────────────────────


def test_strict_cache_invalidates_on_atomic_replace(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, "sessions:\n  trigram_fts: true\n")
    assert (
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)
        is True
    )
    # Atomic replace with an opt-out file (new inode, possibly same mtime ns).
    staged = tmp_path / "config.yaml.new"
    _write_config(staged, "sessions:\n  trigram_fts: false\n")
    os.replace(staged, cfg)
    assert (
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)
        is False
    )


def test_strict_cache_invalidates_on_chmod_and_rewrite(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, "sessions:\n  trigram_fts: true\n")
    assert (
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)
        is True
    )
    # chmod-restrict, then rewrite under it: a stale cache would keep
    # answering True off the old bytes, so the resolver must see the new
    # identity (permission bits are part of the strict signature). The
    # rewrite itself cannot open a 0o400 file for writing as non-root —
    # relax, write, re-restrict, and also prove a chmod-000 read fails
    # closed rather than serving the cached True.
    cfg.chmod(0o644)
    _write_config(cfg, "sessions:\n  trigram_fts: false\n")
    cfg.chmod(0o400)
    assert (
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)
        is False
    )
    cfg.chmod(0o000)
    try:
        if getattr(os, "geteuid", lambda: 1)() != 0:
            with pytest.raises(ConfigResolutionError):
                resolve_effective_config_value(
                    cfg, "sessions", "trigram_fts", default=True
                )
    finally:
        cfg.chmod(0o644)


def test_managed_repair_flips_strict_result(tmp_path, managed_dir):
    """A changed/removed/repaired managed source invalidates the strict
    resolved value — managed provenance is part of the cache signature."""
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, "sessions:\n  trigram_fts: true\n")

    # Broken managed source → strict error (nothing cached as a value).
    _write_config(managed_dir / "config.yaml", "[]\n")
    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)

    # Repaired managed source forces trigram off — user true is overridden.
    _write_config(managed_dir / "config.yaml", "sessions:\n  trigram_fts: false\n")
    assert (
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)
        is False
    )

    # Managed source removed → back to the user value.
    (managed_dir / "config.yaml").unlink()
    assert (
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)
        is True
    )


def test_valid_managed_config_overrides_user_value(tmp_path, managed_dir):
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, "sessions:\n  trigram_fts: true\n")
    _write_config(managed_dir / "config.yaml", "sessions:\n  trigram_fts: false\n")
    assert (
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)
        is False
    )


# ── 9. Profile isolation ────────────────────────────────────────────────


def test_two_profiles_do_not_share_strict_results(tmp_path):
    prof_a = tmp_path / "alpha"
    prof_b = tmp_path / "beta"
    prof_a.mkdir()
    prof_b.mkdir()
    _write_config(prof_a / "config.yaml", "sessions:\n  trigram_fts: false\n")
    # Profile B has no config at all.
    assert (
        resolve_effective_config_value(
            prof_a / "config.yaml", "sessions", "trigram_fts", default=True
        )
        is False
    )
    assert (
        resolve_effective_config_value(
            prof_b / "config.yaml", "sessions", "trigram_fts", default=True
        )
        is True
    )
    # And the first answer is still cached correctly afterwards.
    assert (
        resolve_effective_config_value(
            prof_a / "config.yaml", "sessions", "trigram_fts", default=True
        )
        is False
    )


def test_two_profiles_broken_vs_absent_no_cross_leak(tmp_path):
    prof_a = tmp_path / "alpha"
    prof_b = tmp_path / "beta"
    prof_a.mkdir()
    prof_b.mkdir()
    cfg_a = prof_a / "config.yaml"
    cfg_a.symlink_to(tmp_path / "missing.yaml")
    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(cfg_a, "sessions", "trigram_fts", default=True)
    assert (
        resolve_effective_config_value(
            prof_b / "config.yaml", "sessions", "trigram_fts", default=True
        )
        is True
    )


# ── 10. Genuinely absent config defaults true ───────────────────────────


def test_absent_optional_config_defaults_true_without_quarantine(tmp_path):
    cfg = tmp_path / "config.yaml"
    assert not cfg.exists()
    assert (
        resolve_effective_config_value(cfg, "sessions", "trigram_fts", default=True)
        is True
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._trigram_enabled is True
        conn = _raw_connect(tmp_path / "state.db")
        try:
            # No quarantine fabricated from an absent optional config.
            assert not _has_marker(conn)
        finally:
            conn.close()
    finally:
        db.close()


def test_genuinely_absent_vs_dangling_symlink_differ(tmp_path):
    absent = tmp_path / "absent" / "config.yaml"
    dangling = tmp_path / "dangling" / "config.yaml"
    dangling.parent.mkdir()
    dangling.symlink_to(tmp_path / "nowhere.yaml")
    assert (
        resolve_effective_config_value(
            absent, "sessions", "trigram_fts", default=True
        )
        is True
    )
    with pytest.raises(ConfigResolutionError):
        resolve_effective_config_value(
            dangling, "sessions", "trigram_fts", default=True
        )
