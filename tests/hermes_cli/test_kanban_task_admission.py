"""HERMES-ORCH-001B: deterministic admission and claim gates.

Covers:
- pure admission rejection matrix (exact reason codes)
- opt-in enforcement (contracts mode) vs enforce-all / off
- ready-transition gate (recompute_ready / promote_task)
- atomic claim_task revalidation before task_runs / PID
- invalidation after ready is caught at claim
- legacy unenforced tasks still promote and claim
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


FULL_SHA = "e9b8ae6be137abead6d19ed8a67c523f8c527096"
SHORT_SHA = "e9b8ae6"


def _valid_contract(**overrides):
    base = {
        "version": 1,
        "scope": "ORCH-001B admission gates only",
        "allowed_files": [
            "hermes_cli/kanban_db.py",
            "tests/hermes_cli/test_kanban_task_admission.py",
        ],
        "forbidden_files": ["hermes_cli/main.py"],
        "base_commit": FULL_SHA,
        "required_evidence": ["pytest output", "commit SHA"],
        "required_commands": [
            "scripts/run_tests.sh tests/hermes_cli/test_kanban_task_admission.py -q"
        ],
        "allow_child_creation": False,
        "forbidden_git_actions": [
            "push",
            "merge",
            "amend",
            "reset",
            "clean",
            "restore",
            "stash",
        ],
        "notification_verified": True,
    }
    base.update(overrides)
    return base


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_ADMISSION_ENFORCE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _fresh_conn(board: str = "default"):
    path = kb.kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return kb.connect(board=board)


def _subscribe(conn, task_id: str) -> None:
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="12345",
        thread_id="topic-1",
        user_id="u1",
        notifier_profile="default",
    )


def _event_kinds(conn, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [r["kind"] for r in rows]


def _latest_payload(conn, task_id: str, kind: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()
    assert row is not None, f"no event kind={kind!r}"
    return json.loads(row["payload"] or "{}")


def _run_count(conn, task_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM task_runs WHERE task_id = ?",
        (task_id,),
    ).fetchone()["c"]


# ---------------------------------------------------------------------------
# Pure validator matrix
# ---------------------------------------------------------------------------


class TestEvaluateAdmission:
    def test_legacy_no_contract_admitted_by_default(self):
        d = kb.evaluate_admission(
            contract=None,
            workspace_kind="scratch",
            workspace_path=None,
            has_notification_subscription=False,
        )
        assert d.admitted is True
        assert d.enforced is False
        assert d.codes == []

    def test_missing_contract_under_enforce_all(self):
        d = kb.evaluate_admission(
            contract=None,
            workspace_kind="scratch",
            workspace_path=None,
            has_notification_subscription=True,
            enforce_mode=kb.ADMISSION_ENFORCE_ALL,
        )
        assert d.admitted is False
        assert d.enforced is True
        assert d.codes == [kb.ADMISSION_REASON_MISSING_CONTRACT]

    def test_invalid_full_base_sha(self):
        d = kb.evaluate_admission(
            contract=kb.normalize_task_contract(
                _valid_contract(base_commit=SHORT_SHA)
            ),
            workspace_kind="scratch",
            workspace_path=None,
            has_notification_subscription=True,
        )
        assert d.admitted is False
        assert kb.ADMISSION_REASON_INVALID_BASE_COMMIT in d.codes

    def test_missing_required_evidence(self):
        d = kb.evaluate_admission(
            contract=kb.normalize_task_contract(
                _valid_contract(required_evidence=[])
            ),
            workspace_kind="scratch",
            workspace_path=None,
            has_notification_subscription=True,
        )
        assert d.admitted is False
        assert kb.ADMISSION_REASON_MISSING_REQUIRED_EVIDENCE in d.codes

    def test_missing_workspace_data_for_dir(self):
        d = kb.evaluate_admission(
            contract=kb.normalize_task_contract(_valid_contract()),
            workspace_kind="dir",
            workspace_path=None,
            has_notification_subscription=True,
        )
        assert d.admitted is False
        assert kb.ADMISSION_REASON_MISSING_WORKSPACE_DATA in d.codes

    def test_missing_notification_subscription(self):
        d = kb.evaluate_admission(
            contract=kb.normalize_task_contract(_valid_contract()),
            workspace_kind="scratch",
            workspace_path=None,
            has_notification_subscription=False,
        )
        assert d.admitted is False
        assert kb.ADMISSION_REASON_MISSING_NOTIFICATION_SUBSCRIPTION in d.codes

    def test_valid_contract_admitted(self):
        d = kb.evaluate_admission(
            contract=kb.normalize_task_contract(_valid_contract()),
            workspace_kind="dir",
            workspace_path="/tmp/ws",
            has_notification_subscription=True,
        )
        assert d.admitted is True
        assert d.enforced is True
        assert d.codes == []

    def test_enforce_off_ignores_problems(self):
        d = kb.evaluate_admission(
            contract=kb.normalize_task_contract(
                _valid_contract(base_commit=SHORT_SHA, required_evidence=[])
            ),
            workspace_kind="worktree",
            workspace_path=None,
            has_notification_subscription=False,
            enforce_mode=kb.ADMISSION_ENFORCE_OFF,
        )
        assert d.admitted is True
        assert d.enforced is False


# ---------------------------------------------------------------------------
# Ready + claim gates
# ---------------------------------------------------------------------------


class TestAdmissionGates:
    def test_legacy_task_promotes_and_claims(self, isolated_home):
        with _fresh_conn() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(
                conn, title="child", assignee="a", parents=[parent]
            )
            assert kb.get_task(conn, child).status == "todo"
            kb.claim_task(conn, parent)
            assert kb.complete_task(conn, parent, result="ok")
            # complete_task calls recompute_ready; legacy child becomes ready.
            assert kb.get_task(conn, child).status == "ready"
            claimed = kb.claim_task(conn, child, claimer="host:1")
            assert claimed is not None
            assert claimed.status == "running"
            assert _run_count(conn, child) == 1

    def test_create_with_contract_stays_todo_without_subscription(
        self, isolated_home
    ):
        with _fresh_conn() as conn:
            tid = kb.create_task(
                conn,
                title="contracted",
                contract=_valid_contract(),
            )
            task = kb.get_task(conn, tid)
            assert task.status == "todo"
            assert "admission_rejected" in _event_kinds(conn, tid)
            payload = _latest_payload(conn, tid, "admission_rejected")
            assert (
                kb.ADMISSION_REASON_MISSING_NOTIFICATION_SUBSCRIPTION
                in payload["reasons"]
            )

    def test_missing_contract_prevents_ready_when_enforce_all(
        self, isolated_home
    ):
        with _fresh_conn() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(
                conn, title="no-contract", assignee="a", parents=[parent]
            )
            kb.claim_task(conn, parent)
            assert kb.complete_task(conn, parent, result="ok")
            # Force back to todo then recompute under enforce=all.
            conn.execute(
                "UPDATE tasks SET status='todo' WHERE id=?", (child,)
            )
            conn.commit()
            n = kb.recompute_ready(
                conn, enforce_mode=kb.ADMISSION_ENFORCE_ALL
            )
            assert n == 0
            assert kb.get_task(conn, child).status == "todo"
            payload = _latest_payload(conn, child, "admission_rejected")
            assert payload["reasons"] == [kb.ADMISSION_REASON_MISSING_CONTRACT]
            assert payload["phase"] == "ready"

    def test_invalid_base_sha_blocks_recompute_ready(self, isolated_home):
        with _fresh_conn() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(
                conn,
                title="bad-sha",
                assignee="a",
                parents=[parent],
                contract=_valid_contract(base_commit=SHORT_SHA),
            )
            _subscribe(conn, child)
            kb.claim_task(conn, parent)
            assert kb.complete_task(conn, parent, result="ok")
            assert kb.get_task(conn, child).status == "todo"
            payload = _latest_payload(conn, child, "admission_rejected")
            assert kb.ADMISSION_REASON_INVALID_BASE_COMMIT in payload["reasons"]
            assert _run_count(conn, child) == 0

    def test_missing_evidence_blocks_promote(self, isolated_home):
        with _fresh_conn() as conn:
            tid = kb.create_task(
                conn,
                title="no-evidence",
                contract=_valid_contract(required_evidence=[]),
            )
            _subscribe(conn, tid)
            ok, err = kb.promote_task(conn, tid, actor="tester")
            assert ok is False
            assert "admission rejected" in (err or "")
            assert kb.get_task(conn, tid).status == "todo"
            payload = _latest_payload(conn, tid, "admission_rejected")
            assert (
                kb.ADMISSION_REASON_MISSING_REQUIRED_EVIDENCE
                in payload["reasons"]
            )

    def test_missing_workspace_data_blocks_ready(self, isolated_home):
        with _fresh_conn() as conn:
            tid = kb.create_task(
                conn,
                title="dir-no-path",
                workspace_kind="dir",
                workspace_path=None,
                contract=_valid_contract(),
            )
            _subscribe(conn, tid)
            # Ensure status is todo and parents-satisfied.
            conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (tid,))
            conn.commit()
            n = kb.recompute_ready(conn)
            assert n == 0
            payload = _latest_payload(conn, tid, "admission_rejected")
            assert (
                kb.ADMISSION_REASON_MISSING_WORKSPACE_DATA in payload["reasons"]
            )

    def test_subscription_required_for_ready(self, isolated_home):
        with _fresh_conn() as conn:
            tid = kb.create_task(
                conn,
                title="needs-sub",
                contract=_valid_contract(),
            )
            assert kb.get_task(conn, tid).status == "todo"
            ok, _ = kb.promote_task(conn, tid, actor="tester")
            assert ok is False
            _subscribe(conn, tid)
            ok, err = kb.promote_task(conn, tid, actor="tester")
            assert ok is True, err
            assert kb.get_task(conn, tid).status == "ready"

    def test_claim_rejects_invalidation_after_ready_no_run_row(
        self, isolated_home
    ):
        with _fresh_conn() as conn:
            tid = kb.create_task(
                conn,
                title="claim-gate",
                contract=_valid_contract(),
            )
            _subscribe(conn, tid)
            ok, err = kb.promote_task(conn, tid, actor="tester")
            assert ok is True, err
            assert kb.get_task(conn, tid).status == "ready"

            # Invalidate after ready: wipe notification subscription.
            kb.remove_notify_sub(
                conn,
                task_id=tid,
                platform="telegram",
                chat_id="12345",
                thread_id="topic-1",
            )

            claimed = kb.claim_task(conn, tid, claimer="host:claim")
            assert claimed is None
            task = kb.get_task(conn, tid)
            assert task.status == "todo"
            assert task.worker_pid is None
            assert task.current_run_id is None
            assert _run_count(conn, tid) == 0
            assert "claimed" not in _event_kinds(conn, tid)
            payload = _latest_payload(conn, tid, "claim_rejected")
            assert payload["reason"] == "admission"
            assert payload["phase"] == "claim"
            assert (
                kb.ADMISSION_REASON_MISSING_NOTIFICATION_SUBSCRIPTION
                in payload["reasons"]
            )

    def test_claim_rejects_when_base_sha_invalidated(self, isolated_home):
        with _fresh_conn() as conn:
            tid = kb.create_task(
                conn,
                title="sha-flip",
                contract=_valid_contract(),
            )
            _subscribe(conn, tid)
            assert kb.promote_task(conn, tid, actor="tester")[0] is True

            # Corrupt stored base_commit after ready (structural write bypass).
            bad = _valid_contract(base_commit=SHORT_SHA)
            blob = json.dumps(bad, sort_keys=True, separators=(",", ":"))
            conn.execute(
                "UPDATE tasks SET contract = ? WHERE id = ?", (blob, tid)
            )
            conn.commit()

            assert kb.claim_task(conn, tid, claimer="host:2") is None
            assert _run_count(conn, tid) == 0
            payload = _latest_payload(conn, tid, "claim_rejected")
            assert kb.ADMISSION_REASON_INVALID_BASE_COMMIT in payload["reasons"]

    def test_happy_path_contract_ready_and_claim(self, isolated_home):
        with _fresh_conn() as conn:
            tid = kb.create_task(
                conn,
                title="happy",
                workspace_kind="dir",
                workspace_path=str(isolated_home / "ws"),
                contract=_valid_contract(),
            )
            _subscribe(conn, tid)
            assert kb.promote_task(conn, tid, actor="tester")[0] is True
            claimed = kb.claim_task(conn, tid, claimer="host:ok")
            assert claimed is not None
            assert claimed.status == "running"
            assert _run_count(conn, tid) == 1
            assert "claimed" in _event_kinds(conn, tid)

    def test_evaluate_task_admission_reads_db_state(self, isolated_home):
        with _fresh_conn() as conn:
            tid = kb.create_task(
                conn,
                title="db-eval",
                contract=_valid_contract(base_commit=SHORT_SHA),
            )
            d = kb.evaluate_task_admission(conn, tid)
            assert d.admitted is False
            assert kb.ADMISSION_REASON_INVALID_BASE_COMMIT in d.codes
            assert (
                kb.ADMISSION_REASON_MISSING_NOTIFICATION_SUBSCRIPTION
                in d.codes
            )
