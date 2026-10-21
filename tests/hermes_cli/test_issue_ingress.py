"""Durable Issue 22 receipts and pending intake never dispatch work."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_cli import issue_delivery, issue_ingress as ingress


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("1", True), ("false", False), ("0", False)])
def test_issue_ingress_env_opt_in_is_explicit_and_can_be_disabled(monkeypatch, raw, expected):
    from github_bridge.config import _from_env
    monkeypatch.setenv("GITHUB_BRIDGE_ISSUE_INGRESS_ENABLED", raw)
    assert _from_env()["issue_ingress_enabled"] is expected


def test_issue_ingress_env_opt_in_rejects_ambiguous_values(monkeypatch):
    from github_bridge.config import ConfigInvalid, _from_env
    monkeypatch.setenv("GITHUB_BRIDGE_ISSUE_INGRESS_ENABLED", "sometimes")
    with pytest.raises(ConfigInvalid, match="must be a boolean"):
        _from_env()


def make_intent(*, nonce="abcdefghijklmnop", body_sha256="a" * 64,
                comment_id=1, kind="PING", expires_at=1200, task_text=None,
                result_files=()):
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
        task_text=task_text,
        result_files=result_files,
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
        body = ('{"kind":"TASK","nonce":"abcdefghijklmnop","expires_at":1200,'
                '"task":"review the release","artifacts":["ARCH001_H1_REUSE_INDEX.json"]}')
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
        assert intent.result_files == ("ARCH001_H1_REUSE_INDEX.json",)
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


def test_executor_binding_resolves_profile_without_creating_or_pinning_session(monkeypatch, tmp_path):
    profile = tmp_path / "profiles" / "issue22-dispatch"
    profile.mkdir(parents=True)
    from types import SimpleNamespace
    config_path = tmp_path / "source.json"
    monkeypatch.setattr(ingress, "load_config", lambda _path: SimpleNamespace(
        issue_ingress_executor_profile="issue22-dispatch"))
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env", lambda _name: str(profile),
    )

    assert ingress._issue22_executor(config_path) == "issue22-dispatch"


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


def test_task_result_links_pending_row_when_control_receipts_also_exist():
    conn = connect()
    try:
        ingress.admit_intent(conn, intent=make_intent(nonce="ping_nonce_123456"), now=1100)
        ingress.admit_intent(
            conn, intent=make_intent(nonce="task_nonce_123456", kind="TASK", comment_id=81),
            now=1100,
        )
        ingress.record_control_result(
            conn, repository_id=1165823465, issue_number=22,
            nonce="task_nonce_123456", result_comment_id=900,
        )
        ingress.record_control_result(
            conn, repository_id=1165823465, issue_number=22,
            nonce="ping_nonce_123456", result_comment_id=901,
        )
        assert conn.execute(
            "SELECT result_comment_id FROM issue22_pending_intakes WHERE nonce=?",
            ("task_nonce_123456",),
        ).fetchone()[0] == 900
        assert conn.execute(
            "SELECT result_comment_id FROM issue22_control_receipts WHERE nonce=?",
            ("ping_nonce_123456",),
        ).fetchone()[0] == 901
    finally:
        conn.close()


def test_pending_task_binds_without_live_lease_and_waits_ready_for_executor(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import kanban_db as kb
    kb.create_board("missions")
    conn = ingress.kanban_db_connect.connect(board="missions")
    pins = ingress.TrustPins(
        "ISVGitHub/autonomous-entity", 1165823465, 5343540, 172036721, 548464,
    )
    now = int(ingress.time.time())
    intent = make_intent(
        nonce="task_nonce_123456", comment_id=81, kind="TASK",
        expires_at=now + 300, task_text="Review the release\nCheck the release notes.",
    )
    ingress.admit_intent(conn, intent=intent, now=now)
    monkeypatch.setattr(ingress, "validate_ingress_config", lambda _path: pins)
    active_intent = {"value": intent}
    monkeypatch.setattr(
        ingress, "read_authenticated_intent", lambda *a, **kw: active_intent["value"],
    )
    monkeypatch.setattr(ingress, "_issue22_executor", lambda *_args: "grandloop")

    try:
        bound = ingress.bind_next_pending_task(
            conn, config_path=Path("/tmp/pinned.json"), now=now,
        )
        row = conn.execute(
            "SELECT dispatch_state,task_id,executor_profile "
            "FROM issue22_pending_intakes WHERE nonce=?", (intent.nonce,),
        ).fetchone()
        task = conn.execute(
            "SELECT status,assignee,session_id,max_runtime_seconds,max_retries "
            "FROM tasks WHERE id=?", (row["task_id"],),
        ).fetchone()
        assert bound == {"verdict": "ACK", "state": "queued", "task_id": row["task_id"]}
        assert tuple(row) == ("queued", bound["task_id"], "grandloop")
        assert "session_id" not in {column[1] for column in conn.execute("PRAGMA table_info(issue22_pending_intakes)")}
        assert tuple(task) == ("ready", "grandloop", None, 900, 1)
        assert ingress.issue22_task_is_active(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind='github_task_binding'",
        ).fetchone()[0] == 1
        second = make_intent(
            nonce="task_nonce_second1", comment_id=83, kind="TASK",
            expires_at=now + 300, task_text="must remain pending while busy",
        )
        ingress.admit_intent(conn, intent=second, now=now)
        active_intent["value"] = second
        # BUSY leaves the second intake pending; it is not consumed or dropped.
        assert ingress.bind_next_pending_task(
            conn, config_path=Path("/tmp/pinned.json"), now=now,
        ) == {"verdict": "BUSY", "reason": "TASK_EXECUTOR_OCCUPIED"}
        pending = conn.execute(
            "SELECT dispatch_state,task_id FROM issue22_pending_intakes WHERE nonce=?",
            (second.nonce,),
        ).fetchone()
        assert tuple(pending) == ("awaiting_binding", None)
        # Simulate a lost admission ACK/repeated poll: neither task is duplicated.
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    finally:
        conn.close()


def test_issue22_tasks_are_claimed_by_only_one_concurrent_gateway_dispatcher(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import kanban_db as kb
    kb.create_board("missions")
    conn = ingress.kanban_db_connect.connect(board="missions")
    pins = ingress.TrustPins(
        "ISVGitHub/autonomous-entity", 1165823465, 5343540, 172036721, 548464,
    )
    now = int(ingress.time.time())
    intent = make_intent(
        nonce="task_nonce_concurrent", comment_id=82, kind="TASK",
        expires_at=now + 300, task_text="one durable task",
    )
    ingress.admit_intent(conn, intent=intent, now=now)
    monkeypatch.setattr(ingress, "validate_ingress_config", lambda _path: pins)
    monkeypatch.setattr(ingress, "read_authenticated_intent", lambda *a, **kw: intent)
    monkeypatch.setattr(ingress, "_issue22_executor", lambda *_args: "issue22-dispatch")
    bound = ingress.bind_next_pending_task(conn, config_path=Path("/tmp/pinned.json"), now=now)
    task_id = bound["task_id"]
    conn.close()

    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch

    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(dispatch._kbw, "resolve_workspace", lambda *_a, **_kw: str(tmp_path / "worker-ws"))
    (tmp_path / "worker-ws").mkdir()
    called = []
    called_lock = Lock()
    barrier = __import__("threading").Barrier(2)

    def spawn(_task, _workspace, board=None):
        with called_lock:
            called.append(_task.id)
        return 72001

    def run_dispatch():
        with kbc.connect(board="missions") as db:
            barrier.wait(timeout=5)
            return dispatch.dispatch_once(db, board="missions", spawn_fn=spawn)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _n: run_dispatch(), range(2)))
    with kbc.connect(board="missions") as db:
        task = kb.get_task(db, task_id)
        assert task.status == "running"
        assert task.worker_pid == 72001
    assert called == [task_id]


def test_expired_pending_task_is_never_bound_or_dispatched(monkeypatch):
    conn = connect()
    pins = ingress.TrustPins(
        "ISVGitHub/autonomous-entity", 1165823465, 5343540, 172036721, 548464,
    )
    intent = make_intent(
        nonce="expired_task_nonce", kind="TASK", expires_at=1200,
        task_text="must remain inert",
    )
    ingress.admit_intent(conn, intent=intent, now=1100)
    monkeypatch.setattr(ingress, "validate_ingress_config", lambda _path: pins)
    monkeypatch.setattr(ingress, "_issue22_executor", lambda *_args: "grandloop")
    monkeypatch.setattr(
        ingress, "read_authenticated_intent",
        lambda *args, **kwargs: pytest.fail("expired task must not be re-fetched for dispatch"),
    )
    try:
        assert ingress.bind_next_pending_task(
            conn, config_path=Path("/tmp/pinned.json"), now=1200,
        ) is None
        row = conn.execute(
            "SELECT dispatch_state,task_id FROM issue22_pending_intakes WHERE nonce=?",
            (intent.nonce,),
        ).fetchone()
        assert tuple(row) == ("awaiting_binding", None)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'",
        ).fetchone() is None
    finally:
        conn.close()
