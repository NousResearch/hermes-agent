from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

import gateway  # noqa: F401  -- installs the gateway state.db authority
import gateway.state_db_authority._authority as authority_mod
from gateway.state_db_authority import (
    IntegrityVerdict,
    StateDBIntegrityError,
    gateway_state_db_authority_snapshot,
    install_gateway_state_db_authority,
    verify_state_db_integrity,
)


GATEWAY_SESSION_DB = install_gateway_state_db_authority()


def test_existing_sqlite_without_session_schema_is_materialized_and_reverified(
    tmp_path: Path,
) -> None:
    """A healthy shared SQLite generation may predate the session schema."""
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE shared_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO shared_metadata (key, value) VALUES (?, ?)",
            ("owner", "gateway-independent"),
        )

    before = verify_state_db_integrity(path)
    assert before.verdict is IntegrityVerdict.SCHEMA_INCOMPLETE
    assert before.checked == "canonical_schema"
    assert before.may_open_writer is False

    db = GATEWAY_SESSION_DB(db_path=path)
    try:
        proof = db._gateway_state_db_admission
        assert proof.report.verdict is IntegrityVerdict.VERIFIED
        assert proof.report.checked == "canonical_full"
        assert (
            db._conn.execute(
                "SELECT value FROM shared_metadata WHERE key = 'owner'"
            ).fetchone()[0]
            == "gateway-independent"
        )
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name IN ('sessions', 'messages')"
            ).fetchall()
        }
        assert tables == {"sessions", "messages"}
        identity = proof.identity
    finally:
        db.close()

    after = verify_state_db_integrity(path)
    assert after.verdict is IntegrityVerdict.VERIFIED
    assert after.identity == identity
    assert gateway_state_db_authority_snapshot() == {}


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows prevents replacing the bootstrap database while handles may exist",
)
def test_first_run_replacement_after_bootstrap_must_prove_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement generation cannot inherit bootstrap health."""
    path = tmp_path / "state.db"
    replacement = tmp_path / "replacement.db"
    replacement.write_bytes(b"not a sqlite database")

    real_bootstrap = authority_mod.AUTHORITY._bootstrap

    def bootstrap_then_replace(instance, bootstrap_path, original_init):
        real_bootstrap(instance, bootstrap_path, original_init)
        os.replace(replacement, bootstrap_path)

    monkeypatch.setattr(
        authority_mod.AUTHORITY,
        "_bootstrap",
        bootstrap_then_replace,
    )

    with pytest.raises(StateDBIntegrityError, match="writer admission refused"):
        GATEWAY_SESSION_DB(db_path=path)

    assert gateway_state_db_authority_snapshot() == {}
