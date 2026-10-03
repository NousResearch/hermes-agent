"""Scratch-only proofs for the fail-open _append_event signing hunk (t_22d998fd).

Never writes ~/.hermes/audit/kanban-event-signatures.db. Sidecar is a tempfile.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_event_signing as signing

LIVE_SIDECAR = "/home/frank/.hermes/audit/kanban-event-signatures.db"


def _write_test_key(td: Path, ident: str = "alice") -> tuple[str, str, str]:
    priv = Ed25519PrivateKey.generate()
    key_path = td / f"{ident}-signing"
    key_path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    pub = (
        priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        .decode("ascii")
    )
    identity = f"{ident}@hermes-fleet"
    allowed = td / "allowed_signers"
    allowed.write_text(
        f'{identity} namespaces="git" ssh-ed25519 {pub.split()[1]}\n',
        encoding="utf-8",
    )
    return identity, str(key_path), str(allowed)


def _scratch_board(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    db_path = tmp_path / "kanban" / "boards" / "scratch-t22d998fd" / "kanban.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = kb.connect(db_path)
    return db_path, conn


def test_append_event_sidecar_verify_good(tmp_path, monkeypatch):
    sidecar = str(tmp_path / "kanban-event-signatures.db")
    assert os.path.abspath(sidecar) != os.path.abspath(LIVE_SIDECAR)
    identity, key_path, allowed = _write_test_key(tmp_path)

    monkeypatch.setattr(signing, "DEFAULT_SIDECAR", sidecar)
    monkeypatch.setattr(
        signing, "resolve_signing_key", lambda profile=None: (identity, key_path)
    )

    db_path, conn = _scratch_board(tmp_path)
    try:
        with kb.write_txn(conn):
            tid = kb.create_task(conn, title="t_22d998fd sign proof")
        row = conn.execute(
            "SELECT id, task_id, run_id, kind, payload, created_at "
            "FROM task_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        event_id = int(row["id"])
        sc = sqlite3.connect(sidecar)
        try:
            sig_row = sc.execute(
                "SELECT signer, signature, content FROM event_signatures "
                "WHERE board=? AND event_id=?",
                ("scratch-t22d998fd", event_id),
            ).fetchone()
        finally:
            sc.close()
        assert sig_row is not None, "sidecar row missing — live sidecar must not be used"
        status = signing.verify_signature(
            sig_row[2], sig_row[1], sig_row[0], allowed
        )
        assert status == "GOOD"
        counts = signing.verify_sidecar(
            str(db_path), sidecar, allowed, board="scratch-t22d998fd"
        )
        assert counts.get("GOOD", 0) >= 1
        assert counts.get("BAD", 0) == 0
        assert counts.get("UNTRUSTED", 0) == 0
    finally:
        conn.close()
    assert not os.path.samefile(sidecar, LIVE_SIDECAR) if os.path.exists(LIVE_SIDECAR) else True


def test_append_event_fail_open_on_signing_exception(tmp_path, monkeypatch):
    sidecar = str(tmp_path / "kanban-event-signatures.db")
    identity, key_path, _allowed = _write_test_key(tmp_path, ident="bob")

    monkeypatch.setattr(signing, "DEFAULT_SIDECAR", sidecar)
    monkeypatch.setattr(
        signing, "resolve_signing_key", lambda profile=None: (identity, key_path)
    )

    def _boom(*_a, **_k):
        raise RuntimeError("forced signing exception t_22d998fd")

    monkeypatch.setattr(signing, "sign_event_payload", _boom)

    _db_path, conn = _scratch_board(tmp_path)
    try:
        with kb.write_txn(conn):
            tid = kb.create_task(conn, title="t_22d998fd fail-open")
        n_before = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=?", (tid,)
        ).fetchone()[0]
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "heartbeat", {"probe": "fail-open"})
        n_after = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=?", (tid,)
        ).fetchone()[0]
        assert n_after == n_before + 1
        kinds = [
            r[0]
            for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,)
            )
        ]
        assert "heartbeat" in kinds
        if os.path.isfile(sidecar):
            sc = sqlite3.connect(sidecar)
            try:
                n_sig = sc.execute("SELECT COUNT(*) FROM event_signatures").fetchone()[0]
            finally:
                sc.close()
            assert n_sig == 0
    finally:
        conn.close()
