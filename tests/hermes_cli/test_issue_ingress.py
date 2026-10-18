"""Durable Issue 22 receipts and pending intake never dispatch work."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import issue_ingress as ingress


def make_intent(*, nonce="abcdefghijklmnop", body_sha256="a" * 64,
                comment_id=1, kind="PING", expires_at=1200):
    return ingress.AuthenticatedIntent(
        ingress._ISSUE_SEAL,
        repository="ISVGitHub/autonomous-entity",
        repository_id=1165823465,
        app_id=5343540,
        installation_id=172036721,
        issue_number=22,
        issue_state="open",
        comment_id=comment_id,
        actor_id=548464,
        actor_type="User",
        created_at=1000,
        kind=kind,
        nonce=nonce,
        expires_at=expires_at,
        body_sha256=body_sha256,
    )


def connect():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_ping_receipt_is_durable_and_idempotent_without_task_binding():
    conn = connect()
    try:
        first = ingress.admit_intent(conn, intent=make_intent(), now=1100)
        second = ingress.admit_intent(conn, intent=make_intent(comment_id=2), now=1101)
        assert first.verdict == second.verdict == "ACK"
        assert first.kind == second.kind == "PING"
        assert conn.execute("SELECT COUNT(*) FROM issue22_control_receipts").fetchone()[0] == 1
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='tasks'").fetchone() is None
    finally:
        conn.close()


def test_task_intake_is_bounded_idempotent_and_waits_for_explicit_binding(monkeypatch):
    conn = connect()
    try:
        pins = ingress.TrustPins(
            "ISVGitHub/autonomous-entity", 1165823465, 5343540, 172036721, 548464,
        )
        body = '{"kind":"TASK","nonce":"abcdefghijklmnop","expires_at":1200,"task":"review the release"}'
        rows = [{
            "id": 81,
            "user": {"id": 548464, "type": "User"},
            "created_at": "1970-01-01T00:16:40Z",
            "updated_at": "1970-01-01T00:16:40Z",
            "body": body,
        }]

        class FakeAuth:
            def list_issue_comments(self, owner, repo, number, *, per_page):
                assert (owner, repo, number, per_page) == ("ISVGitHub", "autonomous-entity", 22, 100)
                return rows

        monkeypatch.setattr(
            ingress, "_fetch_issue",
            lambda _path: (type("Cfg", (), {"owner": "ISVGitHub", "repo": "autonomous-entity"})(),
                           pins, {}, {"state": "open"}, FakeAuth()),
        )
        monkeypatch.setattr(ingress, "validate_ingress_config", lambda _path: pins)
        intent = ingress.read_authenticated_intent("/tmp/pinned.json", comment_id=81, now=1100)
        assert intent.kind == "TASK"
        first = ingress.admit_intent(conn, intent=intent, now=1100)
        replay = ingress.admit_intent(conn, intent=intent, now=1101)
        assert first.verdict == replay.verdict == "ACK"
        assert first.kind == replay.kind == "TASK"
        row = conn.execute(
            "SELECT comment_id,actor_id,state,expires_at "
            "FROM issue22_pending_intakes"
        ).fetchone()
        assert tuple(row) == (81, 548464, "awaiting_binding", 1200)
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='tasks'").fetchone() is None
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='issue22_control_receipts'").fetchone() is None
        assert ingress.read_pending_intake_task(
            conn, config_path=Path("/tmp/pinned.json"),
            nonce="abcdefghijklmnop",
        ) == "review the release"
        rows[0]["body"] = body.replace("review the release", "different instruction")
        with pytest.raises(ingress.AdmissionRefused, match="PENDING_COMMENT_CHANGED"):
            ingress.read_pending_intake_task(
                conn, config_path=Path("/tmp/pinned.json"),
                nonce="abcdefghijklmnop",
            )
        kwargs = {
            "repository_id": intent.repository_id, "issue_number": intent.issue_number,
            "nonce": intent.nonce, "result_comment_id": 900,
        }
        ingress.record_control_result(conn, **kwargs)
        ingress.record_control_result(conn, **kwargs)
        row = conn.execute(
            "SELECT state,result_comment_id FROM issue22_pending_intakes"
        ).fetchone()
        assert tuple(row) == ("awaiting_binding", 900)
        with pytest.raises(ingress.AdmissionRefused, match="RESULT_COMMENT_CONFLICT"):
            ingress.record_control_result(conn, **{**kwargs, "result_comment_id": 901})
    finally:
        conn.close()


def test_pending_task_nonce_cannot_be_reused_for_different_body():
    conn = connect()
    try:
        ingress.admit_intent(
            conn, intent=make_intent(kind="TASK", body_sha256="a" * 64), now=1100,
        )
        with pytest.raises(ingress.AdmissionRefused, match="NONCE_REUSED_WITH_DIFFERENT_BODY"):
            ingress.admit_intent(
                conn, intent=make_intent(kind="TASK", body_sha256="b" * 64), now=1100,
            )
    finally:
        conn.close()


def test_nonce_reuse_with_different_body_is_rejected():
    conn = connect()
    try:
        ingress.admit_intent(conn, intent=make_intent(), now=1100)
        with pytest.raises(ingress.AdmissionRefused, match="NONCE_REUSED_WITH_DIFFERENT_BODY"):
            ingress.admit_intent(conn, intent=make_intent(body_sha256="b" * 64), now=1100)
    finally:
        conn.close()


def test_expired_intent_is_rejected_before_receipt():
    conn = connect()
    try:
        with pytest.raises(ingress.AdmissionRefused, match="INTENT_EXPIRED"):
            ingress.admit_intent(conn, intent=make_intent(), now=1200)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='issue22_control_receipts'"
        ).fetchone() is None
    finally:
        conn.close()


def test_result_comment_link_is_durable_and_cannot_be_replaced():
    conn = connect()
    try:
        ingress.admit_intent(conn, intent=make_intent(), now=1100)
        kwargs = {
            "repository_id": 1165823465, "issue_number": 22,
            "nonce": "abcdefghijklmnop", "result_comment_id": 900,
        }
        ingress.record_control_result(conn, **kwargs)
        ingress.record_control_result(conn, **kwargs)
        row = conn.execute(
            "SELECT result_comment_id FROM issue22_control_receipts"
        ).fetchone()
        assert row["result_comment_id"] == 900
        with pytest.raises(ingress.AdmissionRefused, match="RESULT_COMMENT_CONFLICT"):
            ingress.record_control_result(conn, **{**kwargs, "result_comment_id": 901})
    finally:
        conn.close()
