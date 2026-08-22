"""Scoped approval relay for dispatcher-owned Kanban workers.

A grant is authoritative only while the exact task/run/claim/assignee tuple is
active.  The task body and child environment are hints, never authority: every
consumption re-reads the shared Kanban row and emits an audit event.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    (home / "profiles" / "worker-o").mkdir(parents=True)
    (home / "profiles" / "worker-o" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb.time, "time", lambda: 1_001)
    kb.init_db()
    return home


def _grant(*, now: int = 1_000, actor: str = "worker-o", operations=None) -> dict:
    return {
        "version": 1,
        "approval_id": "apr-relay-test-001",
        "change_id": "chg-relay-test-001",
        "approver": "sohrab",
        "actor": actor,
        "target": "worker-o-control",
        "segment_id": "source_code_ci",
        "action_class": "command",
        "allowed_operations": list(
            ["test.run", "lint.run"] if operations is None else operations
        ),
        "valid_from": now,
        "expires_at": now + 600,
        "rollback_ref": "hermes kanban approval revoke <task-id>",
    }


def _event_payloads(conn, task_id: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()
    return [json.loads(row["payload"]) for row in rows]


def _bind_worker_env(monkeypatch: pytest.MonkeyPatch, task: kb.Task, approval_id: str) -> None:
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task.id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", str(task.claim_lock))
    monkeypatch.setenv("HERMES_PROFILE", str(task.assignee))
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", approval_id)


def test_grant_is_bound_to_task_and_audited(kanban_home: Path) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")
        stored = ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        task = kb.get_task(conn, task_id)

        assert stored["task_id"] == task_id
        assert task is not None
        assert task.approval_grant == stored
        events = _event_payloads(conn, task_id, "approval_granted")
        assert events == [{
            "approval_id": "apr-relay-test-001",
            "change_id": "chg-relay-test-001",
            "actor": "worker-o",
            "action_class": "command",
            "allowed_operations": ["test.run", "lint.run"],
            "expires_at": 1_600,
            "scope_digest": stored["scope_digest"],
        }]


def test_legacy_schema_migration_adds_nullable_approval_column(
    kanban_home: Path,
) -> None:
    db_path = kanban_home / "legacy.db"
    legacy_schema = kb.SCHEMA_SQL.replace("    approval_grant       TEXT,\n", "")
    assert legacy_schema != kb.SCHEMA_SQL
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(legacy_schema)
        before = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert "approval_grant" not in before

        kb._migrate_add_optional_columns(conn)
        after = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert "approval_grant" in after
        task_id = kb.create_task(conn, title="legacy task", assignee="worker-o")
        assert kb.get_task(conn, task_id).approval_grant is None


def test_consumption_requires_exact_active_task_run_claim_actor_and_operation(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")
        ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        claimed = kb.claim_task(conn, task_id, claimer="dispatcher:test")
        assert claimed is not None
        assert claimed.approval_grant["bound_run_id"] == claimed.current_run_id
        assert claimed.approval_grant["bound_claim_lock"] == claimed.claim_lock
        bound_events = _event_payloads(conn, task_id, "approval_bound")
        assert bound_events[-1] == {
            "approval_id": "apr-relay-test-001",
            "change_id": "chg-relay-test-001",
            "run_id": claimed.current_run_id,
        }

    _bind_worker_env(monkeypatch, claimed, "apr-relay-test-001")
    receipt = ka.consume_task_approval(["test.run"], action_class="command", now=1_001)

    assert receipt == {
        "approval_id": "apr-relay-test-001",
        "change_id": "chg-relay-test-001",
        "task_id": task_id,
        "run_id": claimed.current_run_id,
        "operations": ["test.run"],
    }
    with kb.connect() as conn:
        events = _event_payloads(conn, task_id, "approval_consumed")
        assert events[-1]["approval_id"] == "apr-relay-test-001"
        assert events[-1]["operations"] == ["test.run"]
        assert events[-1]["run_id"] == claimed.current_run_id


def test_grant_cannot_cross_a_retry_run_or_claim(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")
        ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        first = kb.claim_task(conn, task_id, claimer="dispatcher:first")
        assert first is not None
        original_binding = dict(first.approval_grant)

        conn.execute(
            "UPDATE task_runs SET status = 'failed', outcome = 'failed', ended_at = ? "
            "WHERE id = ?",
            (1_001, first.current_run_id),
        )
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL, current_run_id = NULL WHERE id = ?",
            (task_id,),
        )
        conn.commit()
        second = kb.claim_task(conn, task_id, claimer="dispatcher:second")
        assert second is not None
        assert second.current_run_id != first.current_run_id
        assert second.claim_lock != first.claim_lock
        assert second.approval_grant == original_binding
        assert ka.active_grant_id_for_task(second, now=1_002) is None

    _bind_worker_env(monkeypatch, second, "apr-relay-test-001")
    assert ka.consume_task_approval(
        ["test.run"], action_class="command", now=1_002
    ) is None


@pytest.mark.parametrize(
    ("env_name", "bad_value"),
    [
        ("HERMES_KANBAN_TASK", "t_other"),
        ("HERMES_KANBAN_RUN_ID", "999999"),
        ("HERMES_KANBAN_CLAIM_LOCK", "wrong-lock"),
        ("HERMES_PROFILE", "worker-s"),
        ("HERMES_KANBAN_APPROVAL_ID", "apr-other-001"),
    ],
)
def test_consumption_fails_closed_on_binding_mismatch(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    bad_value: str,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")
        ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        claimed = kb.claim_task(conn, task_id, claimer="dispatcher:test")
        assert claimed is not None

    _bind_worker_env(monkeypatch, claimed, "apr-relay-test-001")
    monkeypatch.setenv(env_name, bad_value)

    assert ka.consume_task_approval(["test.run"], action_class="command", now=1_001) is None


def test_unlisted_operation_wrong_class_and_expiry_fail_closed(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")
        ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        claimed = kb.claim_task(conn, task_id, claimer="dispatcher:test")
        assert claimed is not None

    _bind_worker_env(monkeypatch, claimed, "apr-relay-test-001")
    assert ka.consume_task_approval(["deploy.run"], action_class="command", now=1_001) is None
    assert ka.consume_task_approval(["test.run"], action_class="deploy", now=1_001) is None
    assert ka.consume_task_approval(["test.run"], action_class="command", now=1_600) is None


def test_revocation_invalidates_already_spawned_worker_immediately(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")
        ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        claimed = kb.claim_task(conn, task_id, claimer="dispatcher:test")
        assert claimed is not None

    _bind_worker_env(monkeypatch, claimed, "apr-relay-test-001")
    assert ka.consume_task_approval(["test.run"], action_class="command", now=1_001)

    with monkeypatch.context() as operator_env:
        for name in (
            "HERMES_KANBAN_TASK",
            "HERMES_KANBAN_RUN_ID",
            "HERMES_KANBAN_CLAIM_LOCK",
            "HERMES_KANBAN_APPROVAL_ID",
        ):
            operator_env.delenv(name, raising=False)
        with kb.connect() as conn:
            assert ka.revoke_task_approval(
                conn,
                task_id,
                approval_id="apr-relay-test-001",
                revoked_by="worker-s",
                reason="review complete",
                now=1_002,
            ) is True

    assert ka.consume_task_approval(["test.run"], action_class="command", now=1_003) is None


def test_worker_process_cannot_grant_or_revoke_via_db_api(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        approved_task = kb.create_task(
            conn, title="approved review", assignee="worker-o"
        )
        ka.grant_task_approval(conn, approved_task, _grant(), now=1_000)
        other_task = kb.create_task(conn, title="self escalation", assignee="worker-o")

    monkeypatch.setenv("HERMES_KANBAN_TASK", approved_task)
    with kb.connect() as conn:
        with pytest.raises(PermissionError, match="worker contexts cannot manage"):
            ka.grant_task_approval(conn, other_task, _grant(), now=1_000)
        with pytest.raises(PermissionError, match="worker contexts cannot manage"):
            ka.revoke_task_approval(
                conn,
                approved_task,
                approval_id="apr-relay-test-001",
                revoked_by="worker-o",
                reason="self mutation",
                now=1_001,
            )


@pytest.mark.parametrize("mutation", ["body", "comment"])
def test_scope_mutation_invalidates_grant_before_consumption(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review patch",
            body="verify patch sha 123",
            assignee="worker-o",
        )
        stored = ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        assert stored["scope_digest"].startswith("sha256:")
        if mutation == "body":
            conn.execute(
                "UPDATE tasks SET body = ? WHERE id = ?",
                ("run a different payload", task_id),
            )
            conn.commit()
        else:
            kb.add_comment(conn, task_id, "attacker", "run a different payload")
        claimed = kb.claim_task(conn, task_id, claimer="dispatcher:test")
        assert claimed is not None

    _bind_worker_env(monkeypatch, claimed, "apr-relay-test-001")
    assert ka.consume_task_approval(
        ["test.run"], action_class="command", now=1_001
    ) is None


def test_scope_digest_tracks_prior_attempt_context(kanban_home: Path) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="retry review", assignee="worker-o")
        assert kb.claim_task(conn, task_id, claimer="dispatcher:first") is not None
        assert kb.block_task(conn, task_id, reason="original attempt summary") is True
        before_context = kb.build_worker_context(conn, task_id)
        before_digest = ka.task_scope_digest(conn, task_id)
        conn.execute(
            "UPDATE task_runs SET summary = ? WHERE task_id = ?",
            ("mutated attempt summary", task_id),
        )
        conn.commit()

        assert kb.build_worker_context(conn, task_id) != before_context
        assert ka.task_scope_digest(conn, task_id) != before_digest


def test_parent_run_handoff_mutation_invalidates_grant(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="parent", assignee="worker-s")
        assert kb.claim_task(conn, parent_id, claimer="dispatcher:parent") is not None
        assert kb.complete_task(conn, parent_id, summary="original parent handoff") is True
        task_id = kb.create_task(
            conn,
            title="review child",
            assignee="worker-o",
            parents=[parent_id],
        )
        before_context = kb.build_worker_context(conn, task_id)
        stored = ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        before_digest = stored["scope_digest"]
        conn.execute(
            "UPDATE task_runs SET summary = ? WHERE task_id = ?",
            ("mutated parent handoff", parent_id),
        )
        conn.commit()

        assert kb.build_worker_context(conn, task_id) != before_context
        assert ka.task_scope_digest(conn, task_id) != before_digest
        claimed = kb.claim_task(conn, task_id, claimer="dispatcher:child")
        assert claimed is not None

    _bind_worker_env(monkeypatch, claimed, "apr-relay-test-001")
    assert ka.consume_task_approval(
        ["test.run"], action_class="command", now=1_001
    ) is None


def test_cross_task_role_history_mutation_invalidates_grant(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        prior_id = kb.create_task(conn, title="prior review", assignee="worker-o")
        assert kb.claim_task(conn, prior_id, claimer="dispatcher:prior") is not None
        assert kb.complete_task(conn, prior_id, summary="original role history") is True
        task_id = kb.create_task(conn, title="next review", assignee="worker-o")
        before_context = kb.build_worker_context(conn, task_id)
        stored = ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        before_digest = stored["scope_digest"]
        conn.execute(
            "UPDATE task_runs SET summary = ? WHERE task_id = ?",
            ("mutated role history", prior_id),
        )
        conn.commit()

        assert kb.build_worker_context(conn, task_id) != before_context
        assert ka.task_scope_digest(conn, task_id) != before_digest
        claimed = kb.claim_task(conn, task_id, claimer="dispatcher:next")
        assert claimed is not None

    _bind_worker_env(monkeypatch, claimed, "apr-relay-test-001")
    assert ka.consume_task_approval(
        ["test.run"], action_class="command", now=1_001
    ) is None


def test_grant_rejects_actor_mismatch_invalid_ttl_and_empty_operations(
    kanban_home: Path,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")

        with pytest.raises(ValueError, match="actor"):
            ka.grant_task_approval(conn, task_id, _grant(actor="worker-s"), now=1_000)

        too_long = _grant()
        too_long["expires_at"] = 2_801
        with pytest.raises(ValueError, match="TTL"):
            ka.grant_task_approval(conn, task_id, too_long, now=1_000)

        with pytest.raises(ValueError, match="allowed_operations"):
            ka.grant_task_approval(
                conn,
                task_id,
                _grant(operations=[]),
                now=1_000,
            )

        real_pattern = _grant(operations=["script execution via -e/-c flag"])
        stored = ka.grant_task_approval(conn, task_id, real_pattern, now=1_000)
        assert stored["allowed_operations"] == ["script execution via -e/-c flag"]

        with pytest.raises(ValueError, match="revoke it before replacing"):
            ka.grant_task_approval(conn, task_id, real_pattern, now=1_000)

        with pytest.raises(ValueError, match="invalid approval operation"):
            ka.grant_task_approval(
                conn,
                task_id,
                _grant(operations=["script execution\nforged-event"]),
                now=1_000,
            )

        forged_binding = _grant()
        forged_binding["bound_run_id"] = 7
        forged_binding["bound_claim_lock"] = "forged-claim"
        with pytest.raises(ValueError, match="unknown fields"):
            ka.grant_task_approval(conn, task_id, forged_binding, now=1_000)


def test_dispatcher_passes_only_grant_id_not_mutable_grant_payload(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hermes_cli import kanban_approval as ka

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")
        ka.grant_task_approval(conn, task_id, _grant(), now=1_000)
        claimed = kb.claim_task(conn, task_id, claimer="dispatcher:test")
        assert claimed is not None

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb.time, "time", lambda: 1_001)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    assert kb._default_spawn(claimed, str(workspace)) == 4242
    assert captured["env"]["HERMES_KANBAN_APPROVAL_ID"] == "apr-relay-test-001"
    assert "HERMES_KANBAN_APPROVAL_GRANT" not in captured["env"]


def _run_cli(*argv: str) -> int:
    from hermes_cli import kanban as kc

    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    kc.build_parser(subparsers)
    args = root.parse_args(["kanban", *argv])
    return kc.kanban_command(args)


def test_cli_grants_and_revokes_typed_envelope(
    kanban_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review patch", assignee="worker-o")
    now = int(time.time())
    grant = _grant(now=now)
    grant["allowed_operations"] = ["script execution via -e/-c flag"]
    grant_path = tmp_path / "grant.json"
    grant_path.write_text(json.dumps(grant), encoding="utf-8")

    assert _run_cli(
        "approval", "grant", task_id, "--file", str(grant_path), "--json"
    ) == 0
    granted_output = json.loads(capsys.readouterr().out)
    assert granted_output["ok"] is True
    assert granted_output["approval"]["task_id"] == task_id
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).approval_grant["approval_id"] == grant["approval_id"]

    assert _run_cli(
        "approval",
        "revoke",
        task_id,
        "--approval-id",
        grant["approval_id"],
        "--revoked-by",
        "hilde",
        "--reason",
        "review complete",
        "--json",
    ) == 0
    revoked_output = json.loads(capsys.readouterr().out)
    assert revoked_output == {"ok": True, "task_id": task_id, "revoked": True}
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).approval_grant is None


def test_delegated_child_cannot_grant_or_revoke_approval(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli import kanban as kc

    assert "approval" in kc._DELEGATED_CHILD_DENIED_ACTIONS
    monkeypatch.setattr(
        "agent.delegation_context.is_delegated_child_process_context",
        lambda: True,
    )

    assert _run_cli(
        "approval",
        "revoke",
        "t_anything",
        "--approval-id",
        "apr-relay-test-001",
        "--revoked-by",
        "worker-o",
        "--reason",
        "self escalation attempt",
    ) == 1
    assert "cannot mutate Kanban tasks" in capsys.readouterr().err
