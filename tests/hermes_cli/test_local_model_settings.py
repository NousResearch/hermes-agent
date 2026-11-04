"""Behaviour tests for the profile-scoped local model settings store.

These exercise real persistence (SQLite under a temporary profile root),
compare-and-swap revisions, cross-process concurrency and typed validation.
No source reading, no mocks of the storage layer.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli.local_runtime import model_settings as ms

REPO_ROOT = Path(__file__).resolve().parents[2]

FINGERPRINT = "a" * 64


def _key(engine: str = "llama.cpp", workflow: str = "chat") -> ms.SettingsKey:
    return ms.SettingsKey(model_fingerprint=FINGERPRINT, engine=engine, workflow=workflow)


def _schema() -> ms.EngineSchema:
    return ms.EngineSchema(
        engine="llama.cpp",
        workflow="chat",
        schema_version=ms.STORE_SCHEMA_VERSION,
        settings=(
            ms.SettingSpec(key="temperature", kind="float", minimum=0.0, maximum=2.0,
                           supports_auto=True),
            ms.SettingSpec(key="top_k", kind="int", minimum=0, maximum=400),
            ms.SettingSpec(key="min_p", kind="float", minimum=0.0, maximum=1.0),
            ms.SettingSpec(key="n_ctx", kind="int", minimum=512, apply="load"),
            ms.SettingSpec(key="flash_attn", kind="bool", apply="load"),
            ms.SettingSpec(key="chat_tag", kind="enum", choices=("none", "chatml"),
                           supports_auto=True, apply="load"),
        ),
    )


# ---------------------------------------------------------------------------
# SettingsKey identity validation


def test_settings_key_accepts_canonical_identity():
    key = _key()
    assert key.to_dict() == {
        "model_fingerprint": FINGERPRINT,
        "engine": "llama.cpp",
        "workflow": "chat",
    }
    assert ms.SettingsKey.from_dict(key.to_dict()) == key
    assert len({key, _key()}) == 1  # frozen dataclass: hashable + value equality


@pytest.mark.parametrize(
    ("fingerprint", "engine", "workflow"),
    [
        ("A" * 64, "llama.cpp", "chat"),          # uppercase hex rejected
        ("abc", "llama.cpp", "chat"),             # short fingerprint
        ("a" * 63, "llama.cpp", "chat"),          # not exactly 64
        ("g" * 64, "llama.cpp", "chat"),           # not hex
        (FINGERPRINT, "", "chat"),                 # empty engine
        (FINGERPRINT, "Llama.cpp", "chat"),        # uppercase engine
        (FINGERPRINT, "llama cpp", "chat"),        # space in engine
        (FINGERPRINT, "llama.cpp", "../evil"),     # path-ish workflow
        (FINGERPRINT, "llama.cpp", "chat;rm"),     # shell-ish workflow
    ],
)
def test_settings_key_rejects_malformed_identity(fingerprint, engine, workflow):
    with pytest.raises(ms.InvalidSettingsKeyError):
        ms.SettingsKey(model_fingerprint=fingerprint, engine=engine, workflow=workflow)


# ---------------------------------------------------------------------------
# Absent-key snapshot: no file creation, revision 0, empty overrides


def test_absent_read_returns_inherit_snapshot_without_touching_disk(tmp_path):
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    snapshot = store.read(_key())
    assert snapshot.revision == 0
    assert snapshot.overrides == {}
    assert snapshot.key == _key()
    assert snapshot.schema_version == ms.STORE_SCHEMA_VERSION
    assert snapshot.to_dict() == {
        "key": _key().to_dict(),
        "schema_version": ms.STORE_SCHEMA_VERSION,
        "revision": 0,
        "overrides": {},
    }
    # Reading an absent store must not create the database (or its directory).
    assert not list(root.rglob("*.sqlite3"))
    assert not root.exists() or not list(root.rglob("*"))


# ---------------------------------------------------------------------------
# Write / read / persistence across "restart" (fresh store instance)


def test_write_then_restart_roundtrips_overrides_and_revision(tmp_path):
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    saved = store.write(_key(), {"temperature": 0.2, "top_k": 40}, expected_revision=0)
    assert saved.revision == 1

    reopened = ms.ModelSettingsStore(root)
    snapshot = reopened.read(_key())
    assert snapshot.revision == 1
    assert snapshot.overrides == {"temperature": 0.2, "top_k": 40}
    assert reopened.read(_key()) is not snapshot  # fresh read, same data


def test_explicit_zero_is_persisted_and_roundtrips(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(_key(), {"temperature": 0.0, "top_k": 0}, expected_revision=0)
    snapshot = store.read(_key())
    assert snapshot.overrides["temperature"] == 0
    assert isinstance(snapshot.overrides["temperature"], float)
    assert snapshot.overrides["top_k"] == 0
    assert "temperature" in snapshot.overrides  # present, not dropped as falsy


def test_auto_sentinel_roundtrips_and_serializes_tagged(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(_key(), {"temperature": ms.AUTO, "top_k": 5}, expected_revision=0)
    snapshot = store.read(_key())
    assert snapshot.overrides["temperature"] is ms.AUTO
    payload = json.loads(json.dumps(snapshot.to_dict()))
    assert payload["overrides"]["temperature"] == {"auto": True}
    assert payload["overrides"]["top_k"] == 5


# ---------------------------------------------------------------------------
# Compare-and-swap revisions


def test_write_replaces_complete_override_set(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(_key(), {"temperature": 0.5, "top_k": 40}, expected_revision=0)
    saved = store.write(_key(), {"min_p": 0.1}, expected_revision=1)
    assert saved.revision == 2
    # Complete replacement: keys absent from the new payload are gone.
    assert saved.overrides == {"min_p": 0.1}


def test_stale_expected_revision_raises_conflict_with_both_revisions(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(_key(), {"top_k": 1}, expected_revision=0)
    store.write(_key(), {"top_k": 2}, expected_revision=1)
    with pytest.raises(ms.RevisionConflictError) as excinfo:
        store.write(_key(), {"top_k": 3}, expected_revision=0)
    assert excinfo.value.expected_revision == 0
    assert excinfo.value.current_revision == 2
    # The failed write left the store untouched.
    assert store.read(_key()).overrides == {"top_k": 2}


def test_write_with_nonzero_expected_on_absent_row_is_conflict(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    with pytest.raises(ms.RevisionConflictError) as excinfo:
        store.write(_key(), {"top_k": 1}, expected_revision=3)
    assert excinfo.value.current_revision == 0


def test_negative_expected_revision_rejected(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    with pytest.raises(ms.ModelSettingsError):
        store.write(_key(), {"top_k": 1}, expected_revision=-1)


# ---------------------------------------------------------------------------
# Reset semantics


def test_reset_all_leaves_empty_overrides_at_strictly_newer_revision(tmp_path):
    """Reset deletes the overrides, never the revision history (docs §3)."""
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(_key(), {"temperature": 0.5}, expected_revision=0)
    reset = store.reset(_key(), expected_revision=1)
    assert reset.revision == 2  # monotonic: newer than the pre-reset revision
    assert reset.overrides == {}
    # The row survives: the revision history is intact, not rewound to 0.
    assert store.read(_key()).revision == 2
    assert store.read(_key()).overrides == {}
    # The next save continues from the reset revision.
    again = store.write(_key(), {"temperature": 1.5}, expected_revision=2)
    assert again.revision == 3
    assert again.overrides == {"temperature": 1.5}


def test_stale_pre_create_writer_cannot_clobber_after_reset(tmp_path):
    """ABA guard: a reader from revision 0 before the create must lose after
    create+reset; the revision must never come back down to 0."""
    store = ms.ModelSettingsStore(tmp_path / "profile")
    stale_revision = store.read(_key()).revision
    assert stale_revision == 0
    store.write(_key(), {"top_k": 1}, expected_revision=stale_revision)
    store.reset(_key(), expected_revision=1)
    with pytest.raises(ms.RevisionConflictError) as excinfo:
        store.write(_key(), {"top_k": 99}, expected_revision=stale_revision)
    assert excinfo.value.current_revision == 2
    # A pre-reset CAS (holding revision 1) is rejected too.
    with pytest.raises(ms.RevisionConflictError):
        store.reset(_key(), expected_revision=1, keys=["top_k"])
    assert store.read(_key()).overrides == {}
    # Successive resets keep climbing (strictly monotonic revision).
    second_reset = store.reset(_key(), expected_revision=2)
    assert second_reset.revision == 3


def test_reset_selected_keys_only_removes_those_keys(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(
        _key(), {"temperature": 0.5, "top_k": 40, "min_p": 0.05}, expected_revision=0
    )
    reset = store.reset(_key(), expected_revision=1, keys=["top_k"])
    assert reset.revision == 2
    assert reset.overrides == {"temperature": 0.5, "min_p": 0.05}


def test_reset_selected_with_stale_revision_is_conflict(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(_key(), {"top_k": 40}, expected_revision=0)
    with pytest.raises(ms.RevisionConflictError):
        store.reset(_key(), expected_revision=0, keys=["top_k"])


def test_reset_absent_with_expected_zero_is_noop(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    snapshot = store.reset(_key(), expected_revision=0)
    assert snapshot.revision == 0
    assert snapshot.overrides == {}
    # No-op means truly no-op: no database, no directory.
    assert not store.path.exists()
    assert not store.path.parent.exists()


# ---------------------------------------------------------------------------
# Profile / engine / workflow isolation


def test_two_profile_roots_do_not_bleed(tmp_path):
    store_a = ms.ModelSettingsStore(tmp_path / "profile-a")
    store_b = ms.ModelSettingsStore(tmp_path / "profile-b")
    store_a.write(_key(), {"temperature": 0.7}, expected_revision=0)
    assert store_b.read(_key()).revision == 0
    assert store_b.read(_key()).overrides == {}


def test_engine_and_workflow_scopes_are_isolated(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(_key(engine="llama.cpp"), {"top_k": 40}, expected_revision=0)
    store.write(_key(engine="mlx"), {"top_k": 7}, expected_revision=0)
    store.write(_key(workflow="image"), {"top_k": 3}, expected_revision=0)
    assert store.read(_key(engine="llama.cpp")).overrides == {"top_k": 40}
    assert store.read(_key(engine="mlx")).overrides == {"top_k": 7}
    assert store.read(_key(workflow="image")).overrides == {"top_k": 3}
    assert store.read(_key(workflow="video")).revision == 0


def test_different_fingerprints_are_isolated(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    store.write(_key(), {"top_k": 40}, expected_revision=0)
    other = ms.SettingsKey(model_fingerprint="b" * 64, engine="llama.cpp", workflow="chat")
    assert store.read(other).revision == 0


# ---------------------------------------------------------------------------
# Corruption / unsupported stored schema


def _raw_db(root: Path) -> sqlite3.Connection:
    matches = sorted(root.rglob("*.sqlite3"))
    assert matches, "store database was never created"
    return sqlite3.connect(matches[0])


def test_corrupt_override_json_raises_corruption_error(tmp_path):
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    store.write(_key(), {"top_k": 40}, expected_revision=0)
    conn = _raw_db(root)
    conn.execute("UPDATE model_settings SET overrides_json = '{not json'")
    conn.commit()
    conn.close()
    with pytest.raises(ms.SettingsCorruptionError):
        store.read(_key())


def test_stored_value_of_wrong_shape_raises_corruption_error(tmp_path):
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    store.write(_key(), {"top_k": 40}, expected_revision=0)
    conn = _raw_db(root)
    conn.execute(
        "UPDATE model_settings SET overrides_json = ?",
        (json.dumps({"top_k": [1, 2]}),),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ms.SettingsCorruptionError):
        store.read(_key())


def test_future_stored_schema_version_is_rejected(tmp_path):
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    store.write(_key(), {"top_k": 40}, expected_revision=0)
    conn = _raw_db(root)
    conn.execute("UPDATE model_settings SET schema_version = ?",
                 (ms.STORE_SCHEMA_VERSION + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(ms.UnsupportedSchemaVersionError):
        store.read(_key())


def test_non_finite_stored_float_raises_corruption_error(tmp_path):
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    store.write(_key(), {"top_k": 40}, expected_revision=0)
    conn = _raw_db(root)
    conn.execute(
        "UPDATE model_settings SET overrides_json = ?",
        ('{"top_k": NaN}',),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ms.SettingsCorruptionError):
        store.read(_key())


def test_non_database_file_raises_domain_error_everywhere(tmp_path):
    """A file that is not SQLite at all must fail closed as corruption on
    every path — never a raw sqlite error, never treated as absent."""
    root = tmp_path / "profile"
    db = root / "local_runtime" / "model_settings.sqlite3"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is definitely not a sqlite3 database file " * 32)
    store = ms.ModelSettingsStore(root)
    with pytest.raises(ms.SettingsCorruptionError):
        store.read(_key())
    with pytest.raises(ms.SettingsCorruptionError):
        store.write(_key(), {"top_k": 1}, expected_revision=0)
    with pytest.raises(ms.SettingsCorruptionError):
        store.reset(_key(), expected_revision=0)


def test_renamed_table_is_future_schema_not_absence(tmp_path):
    """A database whose tables this build does not recognise (e.g. migrated
    by a future build) must never read as 'absent settings' and must not be
    silently repopulated with a fresh revision-1 table."""
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    store.write(_key(), {"top_k": 40}, expected_revision=0)
    conn = _raw_db(root)
    conn.execute("ALTER TABLE model_settings RENAME TO model_settings_v2")
    conn.commit()
    conn.close()
    with pytest.raises(ms.SettingsCorruptionError):
        store.read(_key())
    with pytest.raises(ms.SettingsCorruptionError):
        store.write(_key(), {"top_k": 2}, expected_revision=0)
    # The future data is untouched, not overwritten by a recreated table.
    conn = _raw_db(root)
    assert conn.execute("SELECT revision FROM model_settings_v2").fetchone() == (1,)
    conn.close()


def test_missing_expected_column_raises_corruption_error(tmp_path):
    """A model_settings table whose columns do not match this build's layout
    (future or damaged schema) is a domain error, not a raw OperationalError."""
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    store.write(_key(), {"top_k": 40}, expected_revision=0)
    conn = _raw_db(root)
    conn.execute("ALTER TABLE model_settings DROP COLUMN overrides_json")
    conn.commit()
    conn.close()
    with pytest.raises(ms.SettingsCorruptionError):
        store.read(_key())


def test_read_never_recreates_database_deleted_under_it(tmp_path, monkeypatch):
    """Path race: the file exists when checked, then vanishes. A read must
    fail as a domain error WITHOUT recreating the database file."""
    root = tmp_path / "profile"
    store = ms.ModelSettingsStore(root)
    store.write(_key(), {"top_k": 1}, expected_revision=0)
    db = store.path
    real_connect = sqlite3.connect

    def racing_connect(*args, **kwargs):
        db.unlink()  # simulates deletion right after the exists() check
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(ms.sqlite3, "connect", racing_connect)
    with pytest.raises(ms.ModelSettingsError):
        store.read(_key())
    monkeypatch.undo()
    assert not db.exists()  # the read recreated nothing


# ---------------------------------------------------------------------------
# Typed validation via the trusted adapter schema (service seam)


def test_service_save_validates_against_adapter_schema(tmp_path):
    service = ms.ModelSettingsService(ms.ModelSettingsStore(tmp_path / "profile"))
    snapshot = service.save(_key(), _schema(), {"temperature": 0.6}, expected_revision=0)
    assert snapshot.revision == 1
    assert snapshot.overrides == {"temperature": 0.6}


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": "hot"},          # wrong type
        {"temperature": True},           # bool is not a number
        {"temperature": float("nan")},   # NaN rejected
        {"temperature": float("inf")},   # infinity rejected
        {"temperature": 5.0},            # above maximum
        {"temperature": -0.1},           # below minimum
        {"top_k": 1.5},                  # float for int kind
        {"top_k": True},                 # bool-as-int rejected
        {"top_k": -1},                   # below minimum
        {"flash_attn": 1},               # int is not a bool
        {"chat_tag": "bogus"},           # outside enum choices
        {"chat_tag": 0},                 # enum wants a string, zero is not one
    ],
)
def test_service_save_rejects_invalid_values(tmp_path, overrides):
    service = ms.ModelSettingsService(ms.ModelSettingsStore(tmp_path / "profile"))
    with pytest.raises(ms.InvalidSettingValueError):
        service.save(_key(), _schema(), overrides, expected_revision=0)


def test_service_save_rejects_auto_where_not_supported(tmp_path):
    service = ms.ModelSettingsService(ms.ModelSettingsStore(tmp_path / "profile"))
    with pytest.raises(ms.InvalidSettingValueError):
        service.save(_key(), _schema(), {"top_k": ms.AUTO}, expected_revision=0)


def test_service_save_rejects_unknown_settings(tmp_path):
    service = ms.ModelSettingsService(ms.ModelSettingsStore(tmp_path / "profile"))
    with pytest.raises(ms.UnknownSettingError):
        service.save(_key(), _schema(), {"no_such_setting": 1}, expected_revision=0)


def test_service_save_rejects_schema_for_wrong_engine(tmp_path):
    service = ms.ModelSettingsService(ms.ModelSettingsStore(tmp_path / "profile"))
    wrong = ms.EngineSchema(
        engine="mlx", workflow="chat",
        schema_version=ms.STORE_SCHEMA_VERSION, settings=(),
    )
    with pytest.raises(ms.ScopeMismatchError):
        service.save(_key(engine="llama.cpp"), wrong, {"top_k": 1}, expected_revision=0)


def test_service_save_rejects_future_schema_version(tmp_path):
    service = ms.ModelSettingsService(ms.ModelSettingsStore(tmp_path / "profile"))
    future = ms.EngineSchema(
        engine="llama.cpp", workflow="chat",
        schema_version=ms.STORE_SCHEMA_VERSION + 1, settings=(),
    )
    with pytest.raises(ms.UnsupportedSchemaVersionError):
        service.save(_key(), future, {}, expected_revision=0)


def test_schema_construction_rejects_nonsense():
    with pytest.raises(ValueError):
        ms.SettingSpec(key="x", kind="percentile")
    with pytest.raises(ValueError):
        ms.SettingSpec(key="x", kind="enum")  # enum without choices
    with pytest.raises(ValueError):
        ms.SettingSpec(key="x", kind="float", minimum=2.0, maximum=1.0)
    with pytest.raises(ValueError):
        ms.SettingSpec(key="x", kind="int", apply="whenever")
    with pytest.raises(ValueError):
        ms.SettingSpec(key="not a key!", kind="int")


# ---------------------------------------------------------------------------
# Concurrency: in-process threads and real separate processes


def test_concurrent_threads_cas_only_one_winner(tmp_path):
    store = ms.ModelSettingsStore(tmp_path / "profile")
    results: list[str] = []
    barrier = threading.Barrier(8)

    def worker(index: int) -> None:
        barrier.wait()
        try:
            store.write(_key(), {"top_k": index}, expected_revision=0)
        except ms.RevisionConflictError:
            results.append("conflict")
        else:
            results.append("ok")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("ok") == 1
    assert results.count("conflict") == 7
    assert store.read(_key()).revision == 1


_WORKER_SCRIPT = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from pathlib import Path

from hermes_cli.local_runtime.model_settings import (
    ModelSettingsStore,
    RevisionConflictError,
    SettingsKey,
)

root = Path(sys.argv[2])
key = SettingsKey(model_fingerprint=sys.argv[3], engine="llama.cpp", workflow="chat")
store = ModelSettingsStore(root)
try:
    snapshot = store.write(key, {"top_k": 7}, expected_revision=0)
except RevisionConflictError as exc:
    print(json.dumps({"status": "conflict", "current": exc.current_revision}))
else:
    print(json.dumps({"status": "ok", "revision": snapshot.revision}))
"""


def test_concurrent_processes_cas_only_one_winner(tmp_path):
    root = tmp_path / "profile"
    procs = [
        subprocess.run(
            [sys.executable, "-c", _WORKER_SCRIPT, str(REPO_ROOT), str(root), FINGERPRINT],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        for _ in range(4)
    ]
    for proc in procs:
        assert proc.returncode == 0, proc.stderr
    statuses = [json.loads(proc.stdout)["status"] for proc in procs]
    assert statuses.count("ok") == 1
    assert statuses.count("conflict") == 3
    assert ms.ModelSettingsStore(root).read(_key()).revision == 1


def test_persisted_data_visible_from_fresh_process(tmp_path):
    root = tmp_path / "profile"
    ms.ModelSettingsStore(root).write(_key(), {"temperature": 0.25}, expected_revision=0)
    reader = subprocess.run(
        [
            sys.executable, "-c",
            "import json,sys; sys.path.insert(0, sys.argv[1]);"
            "from pathlib import Path;"
            "from hermes_cli.local_runtime.model_settings import ModelSettingsStore, SettingsKey;"
            "k = SettingsKey(sys.argv[3], 'llama.cpp', 'chat');"
            "s = ModelSettingsStore(Path(sys.argv[2])).read(k);"
            "print(json.dumps({'rev': s.revision, 'ov': dict(s.overrides)}))",
            str(REPO_ROOT), str(root), FINGERPRINT,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert reader.returncode == 0, reader.stderr
    payload = json.loads(reader.stdout)
    assert payload == {"rev": 1, "ov": {"temperature": 0.25}}
