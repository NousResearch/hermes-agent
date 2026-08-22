"""Current-main contracts for profile-owned trigram FTS configuration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB


def _write_config(home: Path, enabled: bool) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"sessions:\n  trigram_fts: {'true' if enabled else 'false'}\n",
        encoding="utf-8",
    )
    return home / "state.db"


def _object_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone() is not None


def test_explicit_db_path_uses_adjacent_profile_config(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    wrong_home = tmp_path / "wrong-home"
    db_path = _write_config(profile, False)
    _write_config(wrong_home, True)

    monkeypatch.setenv("HERMES_HOME", str(wrong_home))
    monkeypatch.setenv("HERMES_TRIGRAM_FTS", "1")  # stale legacy carrier

    db = SessionDB(db_path=db_path)
    try:
        assert db._conn is not None
        assert db._trigram_available is False
        assert not _object_exists(db._conn, "messages_fts_trigram")
        assert _object_exists(db._conn, "messages_fts")
    finally:
        db.close()


def test_cli_sessiondb_honors_yaml_without_gateway_bridge(tmp_path, monkeypatch):
    home = tmp_path / "cli-profile"
    db_path = _write_config(home, False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TRIGRAM_FTS", raising=False)

    db = SessionDB(db_path=db_path)
    try:
        assert db._conn is not None
        assert db._trigram_available is False
        assert not _object_exists(db._conn, "messages_fts_trigram")
    finally:
        db.close()


def test_profile_config_honors_env_expansion_and_managed_overlay(
    tmp_path, monkeypatch
):
    profile = tmp_path / "expanded-profile"
    profile.mkdir()
    db_path = profile / "state.db"
    (profile / "config.yaml").write_text(
        "sessions:\n  trigram_fts: ${TRIGRAM_SWITCH}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRIGRAM_SWITCH", "false")

    expanded = SessionDB(db_path=db_path)
    try:
        assert expanded._trigram_available is False
    finally:
        expanded.close()

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )
    (profile / "config.yaml").write_text(
        "sessions:\n  trigram_fts: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    managed_db = SessionDB(db_path=db_path)
    try:
        assert managed_db._trigram_available is False
    finally:
        managed_db.close()
        managed_scope.invalidate_managed_cache()


def test_read_only_profile_does_not_serve_disabled_stale_trigram(
    tmp_path, monkeypatch
):
    home = tmp_path / "readonly-profile"
    db_path = _write_config(home, True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))

    writer = SessionDB(db_path=db_path)
    writer.create_session("s1", source="test")
    writer.append_message("s1", role="user", content="프로젝트 관리")
    writer.close()

    _write_config(home, False)
    reader = SessionDB(db_path=db_path, read_only=True)
    try:
        assert reader._fts_enabled is True
        assert reader._trigram_available is False
        assert reader.search_messages("관리")
    finally:
        reader.close()


def test_read_only_reenable_waits_for_writable_stale_rebuild(tmp_path, monkeypatch):
    home = tmp_path / "readonly-reenable"
    db_path = _write_config(home, True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))

    enabled = SessionDB(db_path=db_path)
    enabled.create_session("s1", source="test")
    enabled.append_message("s1", role="user", content="before quarantine")
    enabled.close()

    _write_config(home, False)
    disabled = SessionDB(db_path=db_path)
    disabled.append_message(
        "s1", role="user", content="DURING_DISABLED_UNIQUE 관리"
    )
    disabled.close()

    _write_config(home, True)
    reader = SessionDB(db_path=db_path, read_only=True)
    try:
        assert reader._trigram_available is False
        assert len(reader.search_messages("DURING_DISABLED_UNIQUE")) == 1
    finally:
        reader.close()

    rebuilt = SessionDB(db_path=db_path)
    try:
        assert rebuilt._trigram_available is True
        assert len(rebuilt.search_messages("DURING_DISABLED_UNIQUE")) == 1
    finally:
        rebuilt.close()


def test_read_only_during_reenable_rebuild_stays_on_fallback(
    tmp_path, monkeypatch
):
    home = tmp_path / "reenable-barrier"
    db_path = _write_config(home, True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))

    enabled = SessionDB(db_path=db_path)
    enabled.create_session("s1", source="test")
    enabled.append_message("s1", role="user", content="before barrier")
    enabled.close()

    _write_config(home, False)
    disabled = SessionDB(db_path=db_path)
    disabled.append_message("s1", role="user", content="BARRIER_UNIQUE 관리")
    disabled.close()
    _write_config(home, True)

    from hermes_state_schema import SessionSchemaMixin

    original = SessionSchemaMixin._rebuild_fts_indexes
    observed = {}

    def _probe_then_rebuild(cursor, *, include_trigram=True):
        reader = SessionDB(db_path=db_path, read_only=True)
        try:
            observed["available"] = reader._trigram_available
            observed["hits"] = len(reader.search_messages("BARRIER_UNIQUE"))
        finally:
            reader.close()
        return original(cursor, include_trigram=include_trigram)

    monkeypatch.setattr(
        SessionSchemaMixin,
        "_rebuild_fts_indexes",
        staticmethod(_probe_then_rebuild),
    )
    rebuilt = SessionDB(db_path=db_path)
    try:
        assert observed == {"available": False, "hits": 1}
        assert rebuilt._trigram_available is True
        assert len(rebuilt.search_messages("BARRIER_UNIQUE")) == 1
    finally:
        rebuilt.close()


def test_fts_stale_recovery_keeps_disabled_trigram_quarantined(
    tmp_path, monkeypatch
):
    home = tmp_path / "stale-recovery"
    db_path = _write_config(home, True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))

    enabled = SessionDB(db_path=db_path)
    enabled.create_session("s1", source="test")
    enabled.append_message("s1", role="user", content="stale recovery needle")
    assert enabled._conn is not None
    enabled._conn.execute(
        "INSERT INTO state_meta (key, value) VALUES ('fts_stale', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = '1'"
    )
    enabled._conn.commit()
    enabled.close()

    _write_config(home, False)
    recovered = SessionDB(db_path=db_path)
    try:
        assert recovered._conn is not None
        assert recovered._trigram_available is False
        trigger_count = recovered._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'messages_fts_trigram%'"
        ).fetchone()[0]
        assert trigger_count == 0
        assert recovered._conn.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is not None
        assert len(recovered.search_messages("needle")) == 1
    finally:
        recovered.close()


def test_optimize_storage_retires_disabled_trigram_without_deleting_messages(
    tmp_path, monkeypatch
):
    home = tmp_path / "optimize-profile"
    db_path = _write_config(home, True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))

    enabled = SessionDB(db_path=db_path)
    enabled.create_session("s1", source="test")
    enabled.append_message("s1", role="user", content="canonical message")
    enabled.close()

    _write_config(home, False)
    disabled = SessionDB(db_path=db_path)
    try:
        assert disabled._conn is not None
        assert _object_exists(disabled._conn, "messages_fts_trigram")
        assert disabled.fts_optimize_available() is True
        result = disabled.optimize_fts_storage(vacuum=False)
        assert result["ok"] is True
        assert not _object_exists(disabled._conn, "messages_fts_trigram")
        assert not _object_exists(disabled._conn, "messages_fts_trigram_src")
        assert disabled._conn.execute(
            "SELECT content FROM messages WHERE session_id = 's1'"
        ).fetchone()[0] == "canonical message"
        assert _object_exists(disabled._conn, "messages_fts")
    finally:
        disabled.close()

    _write_config(home, True)
    reenabled = SessionDB(db_path=db_path)
    try:
        assert reenabled._conn is not None
        assert reenabled._trigram_available is True
        assert _object_exists(reenabled._conn, "messages_fts_trigram")
    finally:
        reenabled.close()


def test_config_resolution_failure_preserves_existing_trigram_quarantine(
    tmp_path, monkeypatch
):
    home = tmp_path / "resolution-failure"
    db_path = _write_config(home, True)

    enabled = SessionDB(db_path=db_path)
    enabled.create_session("s1", source="test")
    enabled.append_message("s1", role="user", content="canonical survives")
    enabled.close()

    _write_config(home, False)
    quarantined = SessionDB(db_path=db_path)
    try:
        assert quarantined._conn is not None
        assert quarantined._trigram_available is False
        assert quarantined._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is not None
    finally:
        quarantined.close()

    from hermes_cli import config as config_module

    def _resolution_failed(*args, **kwargs):
        raise config_module.ConfigResolutionError("deliberate test failure")

    monkeypatch.setattr(
        config_module,
        "resolve_effective_config_value",
        _resolution_failed,
    )
    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened._conn is not None
        assert reopened._trigram_enabled is False
        assert reopened._trigram_available is False
        assert reopened._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is not None
        trigger_count = reopened._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'messages_fts_trigram%'"
        ).fetchone()[0]
        assert trigger_count == 0
        assert reopened.search_messages("canonical")
    finally:
        reopened.close()


def test_config_resolution_failure_defaults_on_only_without_quarantine(
    tmp_path, monkeypatch
):
    from hermes_cli import config as config_module

    def _resolution_failed(*args, **kwargs):
        raise config_module.ConfigResolutionError("deliberate test failure")

    monkeypatch.setattr(
        config_module,
        "resolve_effective_config_value",
        _resolution_failed,
    )
    db = SessionDB(db_path=tmp_path / "fresh" / "state.db")
    try:
        assert db._conn is not None
        assert db._trigram_enabled is True
        assert db._trigram_available is True
    finally:
        db.close()


def test_state_trigram_gate_uses_only_public_config_resolver(tmp_path, monkeypatch):
    from hermes_cli import config as config_module
    from hermes_state import _trigram_fts_enabled_from_config

    monkeypatch.setattr(
        config_module,
        "resolve_effective_config_value",
        lambda *args, **kwargs: False,
    )

    def _private_helper_must_not_escape(*args, **kwargs):
        raise AssertionError("state layer reached a private config helper")

    for name in (
        "_deep_merge",
        "_expand_env_vars",
        "_normalize_max_turns_config",
        "_normalize_root_model_keys",
        "read_user_config_raw",
    ):
        monkeypatch.setattr(config_module, name, _private_helper_must_not_escape)

    assert _trigram_fts_enabled_from_config(tmp_path / "state.db") is False


def test_repeated_sessiondb_opens_reuse_effective_config_cache(
    tmp_path, monkeypatch
):
    from hermes_cli import config as config_module

    home = tmp_path / "cached-profile"
    db_path = _write_config(home, False)
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    config_module._CONFIG_RESOLUTION_FAILURE_CACHE.clear()

    calls = 0
    original = config_module.fast_safe_load

    def _counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(config_module, "fast_safe_load", _counting_load)

    first = SessionDB(db_path=db_path)
    first.close()
    second = SessionDB(db_path=db_path)
    second.close()

    assert calls == 1


def test_repeated_sessiondb_opens_without_config_run_pipeline_once(
    tmp_path, monkeypatch
):
    from hermes_cli import config as config_module

    home = tmp_path / "missing-config-profile"
    home.mkdir()
    db_path = home / "state.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "missing-managed"))
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    config_module._CONFIG_RESOLUTION_FAILURE_CACHE.clear()

    calls = 0
    original = config_module._normalize_root_model_keys

    def _counting_normalizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        config_module,
        "_normalize_root_model_keys",
        _counting_normalizer,
    )

    first = SessionDB(db_path=db_path)
    first.close()
    second = SessionDB(db_path=db_path)
    second.close()

    assert calls == 1


def test_malformed_managed_config_preserves_existing_trigram_quarantine(
    tmp_path, monkeypatch
):
    from hermes_cli import config as config_module
    from hermes_cli import managed_scope

    profile = tmp_path / "managed-failure-profile"
    managed = tmp_path / "managed"
    profile.mkdir()
    managed.mkdir()
    db_path = profile / "state.db"
    (profile / "config.yaml").write_text(
        "sessions:\n  trigram_fts: true\n",
        encoding="utf-8",
    )
    managed_path = managed / "config.yaml"
    managed_path.write_text(
        "sessions:\n  trigram_fts: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()

    enabled = SessionDB(db_path=db_path)
    enabled.close()

    managed_path.write_text(
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )
    managed_scope.invalidate_managed_cache()
    disabled = SessionDB(db_path=db_path)
    try:
        assert disabled._conn is not None
        assert disabled._trigram_available is False
        assert disabled._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is not None
    finally:
        disabled.close()

    managed_path.write_text("sessions: [\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    managed_scope.invalidate_managed_cache()
    # Reproduce the dangerous ordering: a general fail-open read caches the
    # user-only value before SessionDB asks for strict resolution.
    assert config_module.load_config_readonly()["sessions"]["trigram_fts"] is True
    unresolved = SessionDB(db_path=db_path)
    try:
        assert unresolved._conn is not None
        assert unresolved._trigram_enabled is False
        assert unresolved._trigram_available is False
        assert unresolved._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is not None
        trigger_count = unresolved._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'messages_fts_trigram%'"
        ).fetchone()[0]
        assert trigger_count == 0
    finally:
        unresolved.close()


def _assert_hardening_quarantine(db: SessionDB) -> None:
    assert db._conn is not None
    assert db._trigram_enabled is False
    assert db._trigram_available is False
    assert db._conn.execute(
        "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
    ).fetchone() is not None
    trigger_count = db._conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' "
        "AND name LIKE 'messages_fts_trigram%'"
    ).fetchone()[0]
    assert trigger_count == 0


def _establish_user_hardening_quarantine(home: Path) -> Path:
    db_path = _write_config(home, True)
    enabled = SessionDB(db_path=db_path)
    enabled.close()

    _write_config(home, False)
    disabled = SessionDB(db_path=db_path)
    try:
        _assert_hardening_quarantine(disabled)
    finally:
        disabled.close()
    return db_path


def _establish_managed_hardening_quarantine(
    profile: Path,
    managed_path: Path,
) -> Path:
    db_path = _write_config(profile, True)
    managed_path.write_text(
        "sessions:\n  trigram_fts: true\n",
        encoding="utf-8",
    )
    enabled = SessionDB(db_path=db_path)
    enabled.close()

    managed_path.write_text(
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    disabled = SessionDB(db_path=db_path)
    try:
        _assert_hardening_quarantine(disabled)
    finally:
        disabled.close()
    return db_path


@pytest.mark.parametrize(
    "invalid_yaml",
    (
        "false\n",
        "[]\n",
        "sessions: false\n",
        "sessions: []\n",
        "sessions: null\n",
    ),
)
def test_wrong_shape_user_config_preserves_existing_trigram_quarantine(
    tmp_path,
    monkeypatch,
    invalid_yaml,
):
    from hermes_cli import config as config_module

    home = tmp_path / "wrong-shape-user"
    db_path = _establish_user_hardening_quarantine(home)
    (home / "config.yaml").write_text(invalid_yaml, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    config_module.load_config_readonly()

    unresolved = SessionDB(db_path=db_path)
    try:
        _assert_hardening_quarantine(unresolved)
    finally:
        unresolved.close()


@pytest.mark.parametrize(
    "invalid_yaml",
    (
        "false\n",
        "[]\n",
        "sessions: false\n",
        "sessions: []\n",
        "sessions: null\n",
    ),
)
def test_wrong_shape_managed_config_preserves_existing_trigram_quarantine(
    tmp_path,
    monkeypatch,
    invalid_yaml,
):
    from hermes_cli import config as config_module
    from hermes_cli import managed_scope

    profile = tmp_path / "wrong-shape-managed-profile"
    managed = tmp_path / "wrong-shape-managed"
    profile.mkdir()
    managed.mkdir()
    managed_path = managed / "config.yaml"
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    managed_scope.invalidate_managed_cache()
    db_path = _establish_managed_hardening_quarantine(profile, managed_path)

    managed_path.write_text(invalid_yaml, encoding="utf-8")
    managed_scope.invalidate_managed_cache()
    config_module.load_config_readonly()
    unresolved = SessionDB(db_path=db_path)
    try:
        _assert_hardening_quarantine(unresolved)
    finally:
        unresolved.close()


@pytest.mark.linux_only
def test_dangling_user_config_symlink_preserves_quarantine_and_recovers(
    tmp_path,
    monkeypatch,
):
    from hermes_cli import config as config_module

    home = tmp_path / "dangling-user"
    db_path = _establish_user_hardening_quarantine(home)
    config_path = home / "config.yaml"
    missing_target = home / "deployed-config.yaml"
    config_path.unlink()
    config_path.symlink_to(missing_target)
    monkeypatch.setenv("HERMES_HOME", str(home))
    config_module.load_config_readonly()

    unresolved = SessionDB(db_path=db_path)
    try:
        _assert_hardening_quarantine(unresolved)
    finally:
        unresolved.close()

    missing_target.write_text(
        "sessions:\n  trigram_fts: true\n",
        encoding="utf-8",
    )
    repaired = SessionDB(db_path=db_path)
    try:
        assert repaired._trigram_enabled is True
        assert repaired._trigram_available is True
    finally:
        repaired.close()


@pytest.mark.linux_only
def test_dangling_managed_config_symlink_preserves_quarantine_and_recovers(
    tmp_path,
    monkeypatch,
):
    from hermes_cli import config as config_module
    from hermes_cli import managed_scope

    profile = tmp_path / "dangling-managed-profile"
    managed = tmp_path / "dangling-managed"
    profile.mkdir()
    managed.mkdir()
    managed_path = managed / "config.yaml"
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    managed_scope.invalidate_managed_cache()
    db_path = _establish_managed_hardening_quarantine(profile, managed_path)

    missing_target = managed / "deployed-config.yaml"
    managed_path.unlink()
    managed_path.symlink_to(missing_target)
    managed_scope.invalidate_managed_cache()
    config_module.load_config_readonly()

    unresolved = SessionDB(db_path=db_path)
    try:
        _assert_hardening_quarantine(unresolved)
    finally:
        unresolved.close()

    missing_target.write_text(
        "sessions:\n  trigram_fts: true\n",
        encoding="utf-8",
    )
    repaired = SessionDB(db_path=db_path)
    try:
        assert repaired._trigram_enabled is True
        assert repaired._trigram_available is True
    finally:
        repaired.close()
