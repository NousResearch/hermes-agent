from __future__ import annotations

import importlib
import os
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

import gateway
import gateway.state_db_authority._install as authority_install
import hermes_state
from gateway.config import GatewayConfig
from gateway.state_db_authority import (
    IntegrityVerdict,
    StateDBGenerationConflictError,
    StateDBIntegrityError,
    gateway_state_db_authority_snapshot,
    sqlite_read_only_uri,
    verify_state_db_integrity,
)


@pytest.fixture(autouse=True)
def _proof_lifetime_is_test_bounded():
    assert gateway_state_db_authority_snapshot() == {}
    yield
    assert gateway_state_db_authority_snapshot() == {}


def _healthy_db(path: Path) -> Path:
    db = hermes_state.SessionDB(db_path=path)
    db.close()
    return path


def _corrupt_file(path: Path) -> Path:
    path.write_bytes(b"this is not a sqlite database")
    return path


def test_gateway_import_is_passive_and_reload_safe():
    installed = hermes_state.SessionDB

    assert getattr(installed, "_gateway_state_db_authority_wrapped", False) is True
    assert gateway_state_db_authority_snapshot() == {}

    importlib.reload(gateway)

    assert hermes_state.SessionDB is installed
    assert gateway_state_db_authority_snapshot() == {}


def test_read_only_session_db_never_claims_writer_authority(tmp_path):
    path = _healthy_db(tmp_path / "state.db")

    db = hermes_state.SessionDB(db_path=path, read_only=True)
    try:
        assert gateway_state_db_authority_snapshot() == {}
    finally:
        db.close()


def test_first_writer_proof_is_shared_only_while_handles_are_live(tmp_path):
    path = tmp_path / "state.db"

    first = hermes_state.SessionDB(db_path=path)
    first_snapshot = gateway_state_db_authority_snapshot()[str(path.resolve())]
    second = hermes_state.SessionDB(db_path=path)
    second_snapshot = gateway_state_db_authority_snapshot()[str(path.resolve())]

    assert first_snapshot["proof_id"] == second_snapshot["proof_id"]
    assert second_snapshot["holders"] == 2

    second.close()
    assert gateway_state_db_authority_snapshot()[str(path.resolve())]["holders"] == 1

    first.close()
    assert gateway_state_db_authority_snapshot() == {}


def test_closed_generation_is_reverified_before_same_path_reopens(tmp_path):
    path = _healthy_db(tmp_path / "state.db")
    _corrupt_file(path)

    with pytest.raises(StateDBIntegrityError, match="writer admission refused"):
        hermes_state.SessionDB(db_path=path)

    assert gateway_state_db_authority_snapshot() == {}


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows prevents replacing an open SQLite file",
)
def test_live_generation_replacement_refuses_split_brain(tmp_path):
    path = tmp_path / "state.db"
    live = hermes_state.SessionDB(db_path=path)
    replacement = _healthy_db(tmp_path / "replacement.db")
    os.replace(replacement, path)

    try:
        with pytest.raises(StateDBGenerationConflictError, match="split-brain"):
            hermes_state.SessionDB(db_path=path)
    finally:
        live.close()

    # Once the old generation's last handle closes, the replacement must prove
    # itself independently and may then become the sole admitted generation.
    reopened = hermes_state.SessionDB(db_path=path)
    reopened.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows prevents replacing the admission anchor target",
)
def test_tracked_connect_rejects_replacement_before_schema_initialization(
    tmp_path, monkeypatch
):
    path = _healthy_db(tmp_path / "state.db")
    replacement = _healthy_db(tmp_path / "replacement.db")
    replacement_bytes = replacement.read_bytes()
    real_connect = authority_install.ORIGINAL_TRACKED_CONNECT

    def replace_then_connect(database, *args, **kwargs):
        os.replace(replacement, path)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(
        authority_install,
        "ORIGINAL_TRACKED_CONNECT",
        replace_then_connect,
    )

    with pytest.raises(StateDBGenerationConflictError, match="during writer connect"):
        hermes_state.SessionDB(db_path=path)

    # The generation guard fires immediately after sqlite3_connect and before
    # SessionDB can run journal/schema initialization against the replacement.
    assert path.read_bytes() == replacement_bytes


def test_reserved_uri_characters_never_redirect_probe_to_sibling(tmp_path):
    path = _corrupt_file(tmp_path / "state?profile#one.db")
    uri = sqlite_read_only_uri(path)

    assert "%3F" in uri and "%23" in uri
    report = verify_state_db_integrity(path)

    assert report.verdict is IntegrityVerdict.CORRUPT
    assert report.may_open_writer is False
    assert not (tmp_path / "state").exists()


def test_torn_btree_is_rejected_by_canonical_full_probe(tmp_path):
    path = _healthy_db(tmp_path / "state.db")
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        conn.executemany(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            [(f"sid-{index}", "gateway", float(index)) for index in range(1000)],
        )

    raw = bytearray(path.read_bytes())
    assert len(raw) > page_size * 5
    raw[page_size * 4 : page_size * 5] = b"\xff" * page_size
    path.write_bytes(raw)

    report = verify_state_db_integrity(path)

    assert report.verdict is IntegrityVerdict.CORRUPT
    assert report.checked == "canonical_full"
    assert report.problems
    assert report.may_open_writer is False


def test_absence_and_object_type_are_not_healthy_verdicts(tmp_path):
    absent = verify_state_db_integrity(tmp_path / "absent.db")
    directory = tmp_path / "directory.db"
    directory.mkdir()
    unsupported = verify_state_db_integrity(directory)

    assert absent.verdict is IntegrityVerdict.ABSENT
    assert absent.may_open_writer is False
    assert unsupported.verdict is IntegrityVerdict.UNSUPPORTED_OBJECT
    assert unsupported.may_open_writer is False


def test_session_store_propagates_integrity_refusal_and_does_not_cache_it(
    tmp_path, monkeypatch
):
    path = _corrupt_file(tmp_path / "state.db")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", path)

    store = object.__new__(gateway.SessionStore)
    store._db_handles = {}
    store._db_handles_lock = threading.Lock()

    for _ in range(2):
        with pytest.raises(StateDBIntegrityError):
            store._open_session_db_for_active_scope()
        assert path.resolve() not in store._db_handles


def test_normal_session_store_gets_an_admitted_handle(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", path)

    store = gateway.SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        proof = getattr(store._db, "_gateway_state_db_admission", None)
        assert proof is not None
        assert proof.path == path.resolve()
    finally:
        store.close_all_db_handles()
