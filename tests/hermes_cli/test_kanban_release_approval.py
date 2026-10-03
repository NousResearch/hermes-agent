"""Release approval saga contracts: stale/replay/race/task binding and argv execution."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


def _present(
    conn,
    task_id: str,
    *,
    release: str = "2026.09.18-01+abc",
    message: str = "m1",
    chat_id: str = "chat-1",
    thread_id: str = "",
    actor_id: str = "armin-1",
):
    from hermes_cli.kanban_release_approval import present_release
    return present_release(
        conn,
        task_id=task_id,
        release_id=release,
        manifest_sha256="a" * 64,
        manual_test_cases_digest="b" * 64,
        source_commit="c" * 40,
        bundle_sha256="d" * 64,
        artifact_digests={"php": "e" * 64, "react": "f" * 64, "caddy": "1" * 64},
        workflow_status="Auf Dev zur Prüfung",
        active_dev_release_id=release,
        platform="telegram",
        chat_id=chat_id,
        thread_id=thread_id,
        actor_id=actor_id,
        presented_message_id=message,
        previous_release_id="2026.09.17-02+old",
        rollback_available=True,
    )


def _adapter(tmp_path: Path, calls: Path) -> list[str]:
    script = tmp_path / "promotion adapter ; safe.py"
    script.write_text(
        "import json,os,sys\n"
        f"p={str(calls)!r}\n"
        "trusted=json.load(sys.stdin)\n"
        "open(p,'a',encoding='utf-8').write(json.dumps({'argv':sys.argv[1:],'trusted':trusted,'secret_present':'HERMES_RELEASE_TEST_SECRET' in os.environ})+'\\n')\n"
        "a=sys.argv; rid=a[a.index('--release-id')+1]; task=a[a.index('--task-id')+1]; manifest=a[a.index('--manifest-sha256')+1]\n"
        "op=a[a.index('--operation-key')+1]\n"
        "print(json.dumps({'ok':True,'contract_version':'cuto-hermes-release/v1','operation_key':op,'task_id':task,'release_id':rid,'manifest_sha256':manifest,'manual_test_cases_sha256':'b'*64,'source_commit':'c'*40,'workflow_run_id':'123','bundle_sha256':'d'*64,'artifacts':{'php':{'sha256':'e'*64},'react':{'sha256':'f'*64},'caddy':{'sha256':'1'*64}},'dev':{'result':'success','active_release_id':rid},"
        "'test':{'result':'success','active_release_id':rid},'production':{'result':'success','active_release_id':rid},"
        "'previous_release_id':'2026.09.17-02+old','rollback_available':True,'database_restore_attempted':False}))\n"
    )
    return [sys.executable, str(script), "literal ; $(not-shell)"]


def _approve(conn, argv, *, message="m1", now=100):
    from hermes_cli.kanban_release_approval import ApprovalContext, process_approval
    return process_approval(
        conn,
        text="freigegeben",
        context=ApprovalContext(
            platform="telegram", chat_id="chat-1", thread_id="", actor_id="armin-1",
            reply_to_message_id=message,
        ),
        promotion_argv=argv,
        now=now,
    )


def _receipt(command, **changes):
    release_id = command[command.index("--release-id") + 1]
    receipt = {
        "ok": True,
        "contract_version": "cuto-hermes-release/v1",
        "operation_key": command[command.index("--operation-key") + 1],
        "task_id": command[command.index("--task-id") + 1],
        "release_id": release_id,
        "manifest_sha256": command[command.index("--manifest-sha256") + 1],
        "manual_test_cases_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "workflow_run_id": "123",
        "bundle_sha256": "d" * 64,
        "artifacts": {
            "php": {"sha256": "e" * 64},
            "react": {"sha256": "f" * 64},
            "caddy": {"sha256": "1" * 64},
        },
        "dev": {"result": "success", "active_release_id": release_id},
        "test": {"result": "success", "active_release_id": release_id},
        "production": {"result": "success", "active_release_id": release_id},
        "previous_release_id": "2026.09.17-02+old",
        "rollback_available": True,
        "database_restore_attempted": False,
    }
    receipt.update(changes)
    return receipt


def _receipt_adapter(monkeypatch, mutate):
    def adapter(command, **_kwargs):
        receipt = _receipt(command)
        mutate(receipt)
        return type("Completed", (), {"stdout": json.dumps(receipt)})()

    monkeypatch.setattr("hermes_cli.kanban_release_approval.subprocess.run", adapter)


def test_release_approval_is_bound_and_executes_literal_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RELEASE_TEST_SECRET", "must-not-reach-adapter")
    kb.init_db()
    calls = tmp_path / "calls.jsonl"
    with connect() as conn:
        task = kb.create_task(conn, title="Release")
        gate = _present(conn, task)
        result = _approve(conn, _adapter(tmp_path, calls))
        assert result.ok and result.classification == "success"
        assert result.release_id == gate.release_id
        assert result.active_release_id == gate.release_id
        assert result.previous_release_id == "2026.09.17-02+old"
        assert result.rollback_available is True
        assert result.dev_result == result.test_result == result.production_result == "success"
        call = json.loads(calls.read_text().splitlines()[0])
        argv = call["argv"]
        assert argv[:2] == ["literal ; $(not-shell)", "promote"]
        assert argv[argv.index("--task-id") + 1] == task
        assert argv[argv.index("--release-id") + 1] == gate.release_id
        assert "armin-1" not in argv and "b" * 64 not in argv
        assert call["trusted"]["actor_subject"] == "armin-1"
        assert call["trusted"]["manual_test_cases_sha256"] == "b" * 64
        assert call["trusted"]["contract_version"] == "cuto-hermes-release/v1"
        assert call["trusted"]["approval_id"].startswith("hermes:")
        assert call["secret_present"] is False
        assert len(calls.read_text().splitlines()) == 1


def test_stale_release_replay_and_task_switch_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    calls = tmp_path / "calls.jsonl"
    argv = _adapter(tmp_path, calls)
    with connect() as conn:
        old_task = kb.create_task(conn, title="Old")
        _present(conn, old_task, message="old-message")
        new_task = kb.create_task(conn, title="New")
        _present(conn, new_task, release="2026.09.18-02+def", message="new-message")

        switched = _approve(conn, argv, message="old-message")
        assert not switched.ok and switched.classification == "stale_approval"
        assert switched.active_release_id == "2026.09.18-02+def"
        assert switched.previous_release_id is None
        assert switched.rollback_available is None
        accepted = _approve(conn, argv, message="new-message")
        assert accepted.ok
        replay = _approve(conn, argv, message="new-message", now=101)
        assert not replay.ok and replay.classification == "replay"
        assert len(calls.read_text().splitlines()) == 1


def test_approved_release_replay_after_task_switch_reports_current_route_state(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    calls = tmp_path / "calls.jsonl"
    argv = _adapter(tmp_path, calls)
    with connect() as conn:
        old_task = kb.create_task(conn, title="Old")
        old_gate = _present(
            conn,
            old_task,
            release="release-old",
            message="old-message",
        )
        accepted = _approve(conn, argv, message="old-message")
        assert accepted.ok

        new_task = kb.create_task(conn, title="New")
        _present(
            conn,
            new_task,
            release="release-new",
            message="new-message",
        )

        replay = _approve(conn, argv, message="old-message", now=101)
        assert not replay.ok and replay.classification == "replay"
        assert replay.release_id == old_gate.release_id
        assert replay.active_release_id == "release-new"
        assert replay.previous_release_id is None
        assert replay.rollback_available is None
        assert replay.dev_result == replay.test_result == replay.production_result == "unknown"
        assert "Aktiv: release-new" in replay.user_message()
        assert "Aktiv: release-old" not in replay.user_message()
        assert "Vorgänger: unbekannt" in replay.user_message()
        assert "Rollback verfügbar: unbekannt" in replay.user_message()
        assert len(calls.read_text().splitlines()) == 1


def test_manifest_dev_state_and_workflow_status_are_rechecked(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    calls = tmp_path / "calls.jsonl"
    with connect() as conn:
        task = kb.create_task(conn, title="Release")
        gate = _present(conn, task)
        conn.execute("UPDATE kanban_release_gates SET active_dev_release_id=? WHERE id=?", ("replacement", gate.gate_id))
        conn.commit()
        stale = _approve(conn, _adapter(tmp_path, calls))
        assert not stale.ok and stale.classification == "stale_approval"
        assert not calls.exists()

        other = kb.create_task(conn, title="Wrong state")
        gate2 = _present(conn, other, message="m2")
        conn.execute("UPDATE kanban_release_gates SET workflow_status=? WHERE id=?", ("Draft", gate2.gate_id))
        conn.commit()
        wrong = _approve(conn, _adapter(tmp_path, calls), message="m2")
        assert not wrong.ok and wrong.classification == "stale_approval"
        assert not calls.exists()


def test_non_exact_text_and_changed_current_release_state_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    calls = tmp_path / "calls.jsonl"
    argv = _adapter(tmp_path, calls)
    with connect() as conn:
        task = kb.create_task(conn, title="Release")
        gate = _present(conn, task)
        # The protocol token is byte-exact; gateways must not normalize it.
        from hermes_cli.kanban_release_approval import ApprovalContext, process_approval, update_release_state
        context = ApprovalContext("telegram", "chat-1", "", "armin-1", "m1")
        whitespace = process_approval(conn, text=" freigegeben", context=context,
                                      promotion_argv=argv, now=100)
        assert not whitespace.ok and whitespace.classification == "not_approval"
        update_release_state(conn, task_id=task, workflow_status="Auf Dev zur Prüfung",
                             active_dev_release_id="replacement",
                             manifest_sha256="c" * 64)
        stale = _approve(conn, argv, now=101)
        assert not stale.ok and stale.classification == "stale_approval"
        assert stale.release_id == gate.release_id
        assert stale.active_release_id == "replacement"
        assert stale.previous_release_id is None
        assert stale.rollback_available is None
        assert stale.dev_result == stale.test_result == stale.production_result == "unknown"
        message = stale.user_message()
        assert "Dev: unbekannt. Test: unbekannt. Prod: unbekannt." in message
        assert "Aktiv: replacement" in message
        assert "Vorgänger: unbekannt" in message
        assert "Rollback verfügbar: unbekannt" in message
        assert "not_started" not in message
        assert not calls.exists()


def test_expired_promoting_claim_resumes_with_same_operation_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    calls = tmp_path / "calls.jsonl"
    with connect() as conn:
        task = kb.create_task(conn, title="Interrupted promotion")
        gate = _present(conn, task)
        from hermes_cli.kanban_release_approval import _operation_key
        row = conn.execute("SELECT * FROM kanban_release_gates WHERE id=?", (gate.gate_id,)).fetchone()
        conn.execute(
            "INSERT INTO kanban_release_sagas "
            "(operation_key,gate_id,state,owner_token,lease_expires,created_at,updated_at) "
            "VALUES (?,?,'promoting','dead-worker',1,1,1)",
            (_operation_key(row), gate.gate_id),
        )
        conn.execute("UPDATE kanban_release_gates SET gate_status='consuming' WHERE id=?", (gate.gate_id,))
        conn.commit()
        operation_key = _operation_key(row)
        result = _approve(conn, _adapter(tmp_path, calls), now=100)
        assert result.ok and result.classification == "success"
        argv = json.loads(calls.read_text().splitlines()[0])["argv"]
        assert argv[1] == "resume"
        assert argv[argv.index("--operation-key") + 1] == operation_key
        saga = conn.execute(
            "SELECT state,operation_key,adapter_receipt FROM kanban_release_sagas WHERE gate_id=?",
            (gate.gate_id,),
        ).fetchone()
        assert (saga["state"], saga["operation_key"]) == ("succeeded", operation_key)
        assert json.loads(saga["adapter_receipt"])["release_id"] == gate.release_id


def test_present_release_cannot_replace_claimed_task_from_another_route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = kb.init_db()
    adapter_started = threading.Event()
    release_adapter = threading.Event()
    calls = []

    def blocked_adapter(command, **_kwargs):
        calls.append(command)
        adapter_started.set()
        assert release_adapter.wait(timeout=5)
        receipt = _receipt(command)
        return type("Completed", (), {"stdout": json.dumps(receipt)})()

    monkeypatch.setattr("hermes_cli.kanban_release_approval.subprocess.run", blocked_adapter)
    with connect(db) as conn:
        task = kb.create_task(conn, title="Release race")
        old_gate = _present(conn, task, release="release-old", message="old-message")

    results = []

    def approve_old():
        with connect(db) as conn:
            results.append(_approve(conn, ["adapter"], message="old-message", now=100))

    thread = threading.Thread(target=approve_old)
    thread.start()
    assert adapter_started.wait(timeout=5)
    try:
        with connect(db) as conn:
            with pytest.raises(RuntimeError, match="promotion is in progress"):
                _present(
                    conn,
                    task,
                    release="release-new",
                    message="new-message",
                    chat_id="chat-2",
                    thread_id="thread-2",
                    actor_id="armin-2",
                )
            current = conn.execute(
                "SELECT gate_status FROM kanban_release_gates WHERE id=?", (old_gate.gate_id,)
            ).fetchone()
            assert current["gate_status"] == "consuming"
            state = conn.execute(
                "SELECT active_dev_release_id FROM kanban_release_state WHERE task_id=?", (task,)
            ).fetchone()
            assert state["active_dev_release_id"] == "release-old"
            from hermes_cli.kanban_release_approval import update_release_state
            with pytest.raises(RuntimeError, match="promotion is in progress"):
                update_release_state(
                    conn,
                    task_id=task,
                    workflow_status="Auf Dev zur Prüfung",
                    active_dev_release_id="release-new",
                    manifest_sha256="c" * 64,
                )
    finally:
        release_adapter.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(calls) == 1
    assert calls[0][calls[0].index("--release-id") + 1] == "release-old"
    assert results[0].ok is True


def test_failure_user_message_is_nontechnical_and_reports_release_state():
    from hermes_cli.kanban_release_approval import ApprovalResult

    result = ApprovalResult(
        ok=False,
        classification="stale_after_promotion",
        release_id="release-1",
        dev_result="success",
        test_result="failed",
        production_result="not_started",
        active_release_id="release-0",
        previous_release_id="release-prev",
        rollback_available=True,
    )

    message = result.user_message()
    assert message.startswith("Fehler: ")
    assert "stale_after_promotion" not in message
    assert "Release-ID: release-1" in message
    assert "Dev: success" in message
    assert "Test: failed" in message
    assert "Prod: not_started" in message
    assert "Aktiv: release-0" in message
    assert "Vorgänger: release-prev" in message
    assert "Rollback verfügbar: ja" in message


def test_ambiguous_adapter_failure_stays_consuming_and_retries_as_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    commands = []

    def flaky_adapter(command, **_kwargs):
        commands.append(command)
        if len(commands) == 1:
            raise OSError("adapter connection lost")
        receipt = _receipt(command)
        return type("Completed", (), {"stdout": json.dumps(receipt)})()

    monkeypatch.setattr("hermes_cli.kanban_release_approval.subprocess.run", flaky_adapter)
    with connect() as conn:
        task = kb.create_task(conn, title="Ambiguous adapter")
        gate = _present(conn, task)
        first = _approve(conn, ["adapter"], now=100)
        assert not first.ok and first.classification == "adapter_failed"
        assert first.active_release_id is None
        assert first.previous_release_id is None
        assert first.rollback_available is None
        assert first.dev_result == first.test_result == first.production_result == "unknown"
        message = first.user_message()
        assert "Aktiv: unbekannt" in message
        assert "Vorgänger: unbekannt" in message
        assert "Rollback verfügbar: unbekannt" in message
        assert "not_started" not in message
        persisted = conn.execute(
            "SELECT g.gate_status,s.state,s.operation_key FROM kanban_release_gates g "
            "JOIN kanban_release_sagas s ON s.gate_id=g.id WHERE g.id=?",
            (gate.gate_id,),
        ).fetchone()
        assert (persisted["gate_status"], persisted["state"]) == ("consuming", "promoting")

        second = _approve(conn, ["adapter"], now=101)
        assert second.ok
        assert [command[1] for command in commands] == ["promote", "resume"]
        assert (
            commands[0][commands[0].index("--operation-key") + 1]
            == commands[1][commands[1].index("--operation-key") + 1]
            == persisted["operation_key"]
        )


def test_success_without_rollback_evidence_reports_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()

    def adapter_without_rollback(command, **_kwargs):
        receipt = _receipt(command)
        receipt.pop("previous_release_id")
        receipt.pop("rollback_available")
        return type("Completed", (), {"stdout": json.dumps(receipt)})()

    monkeypatch.setattr(
        "hermes_cli.kanban_release_approval.subprocess.run", adapter_without_rollback
    )
    with connect() as conn:
        task = kb.create_task(conn, title="No rollback evidence")
        _present(conn, task)
        result = _approve(conn, ["adapter"], now=100)

    assert not result.ok and result.classification == "adapter_failed"
    assert result.previous_release_id is None
    assert result.rollback_available is None
    assert "Vorgänger: unbekannt" in result.user_message()
    assert "Rollback verfügbar: unbekannt" in result.user_message()


def test_verified_test_failure_stops_before_production_and_is_resumable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    calls = []

    def adapter(command, **kwargs):
        trusted = json.loads(kwargs["input"])
        calls.append((command, trusted))
        successful = len(calls) == 2
        receipt = _receipt(command)
        if not successful:
            previous = receipt["previous_release_id"]
            receipt["test"] = {"result": "failed", "active_release_id": previous}
            receipt["production"] = {"result": "not_started", "active_release_id": previous}
        return type("Completed", (), {"stdout": json.dumps(receipt)})()

    monkeypatch.setattr("hermes_cli.kanban_release_approval.subprocess.run", adapter)
    with connect() as conn:
        task = kb.create_task(conn, title="Test smoke failure")
        gate = _present(conn, task)
        first = _approve(conn, ["adapter"], now=100)
        assert not first.ok and first.classification == "promotion_failed"
        assert first.test_result == "failed"
        assert first.production_result == "not_started"
        state = conn.execute(
            "SELECT g.gate_status,s.state,s.error_class FROM kanban_release_gates g "
            "JOIN kanban_release_sagas s ON s.gate_id=g.id WHERE g.id=?",
            (gate.gate_id,),
        ).fetchone()
        assert tuple(state) == ("consuming", "failed", "promotion_failed")

        second = _approve(conn, ["adapter"], now=101)
        assert second.ok
        assert [call[0][1] for call in calls] == ["promote", "resume"]
        assert calls[0][1] == calls[1][1]


def test_verified_production_failure_reports_compensated_active_release(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()

    def adapter(command, **_kwargs):
        receipt = _receipt(command)
        receipt["production"] = {
            "result": "failed",
            "active_release_id": receipt["previous_release_id"],
        }
        return type("Completed", (), {"stdout": json.dumps(receipt)})()

    monkeypatch.setattr("hermes_cli.kanban_release_approval.subprocess.run", adapter)
    with connect() as conn:
        task = kb.create_task(conn, title="Production smoke compensation")
        _present(conn, task)
        result = _approve(conn, ["adapter"], now=100)

    assert not result.ok and result.classification == "promotion_failed"
    assert result.production_result == "failed"
    assert result.active_release_id == "2026.09.17-02+old"
    assert result.previous_release_id == "2026.09.17-02+old"
    assert result.rollback_available is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.update({
            "test": {"result": "failed", "active_release_id": "release-old"},
        }),
        lambda receipt: receipt.update({
            "production": {
                "result": "failed",
                "active_release_id": receipt["release_id"],
            },
        }),
        lambda receipt: receipt.update({
            "manifest_sha256": "9" * 64,
            "manual_test_cases_sha256": "8" * 64,
            "artifacts": {"php": {"sha256": "7" * 64}},
            "previous_release_id": "chosen-by-caller",
            "database_restore_attempted": True,
        }),
    ],
    ids=(
        "test-failed-production-success",
        "production-failed-candidate-still-active",
        "unbound-identities-and-database-restore",
    ),
)
def test_contradictory_or_unbound_receipts_fail_closed(
    tmp_path, monkeypatch, mutate
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    _receipt_adapter(monkeypatch, mutate)
    with connect() as conn:
        task = kb.create_task(conn, title="Invalid receipt")
        gate = _present(conn, task)
        result = _approve(conn, ["adapter"], now=100)
        persisted = conn.execute(
            "SELECT g.gate_status,s.state,s.error_class,s.adapter_receipt "
            "FROM kanban_release_gates g JOIN kanban_release_sagas s ON s.gate_id=g.id "
            "WHERE g.id=?",
            (gate.gate_id,),
        ).fetchone()

    assert not result.ok and result.classification == "adapter_failed"
    assert tuple(persisted) == ("consuming", "promoting", "adapter", None)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.update({"unexpected_security_claim": True}),
        lambda receipt: receipt["production"].update({"compensated": True}),
        lambda receipt: receipt["artifacts"]["php"].update({"path": "/replacement"}),
    ],
    ids=("top-level", "target", "artifact"),
)
def test_unknown_receipt_security_fields_fail_closed(tmp_path, monkeypatch, mutate):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    _receipt_adapter(monkeypatch, mutate)
    with connect() as conn:
        task = kb.create_task(conn, title="Unknown receipt field")
        _present(conn, task)
        result = _approve(conn, ["adapter"], now=100)

    assert not result.ok and result.classification == "adapter_failed"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.update({"contract_version": "cuto-hermes-release/v2"}),
        lambda receipt: receipt.update({"operation_key": "release-approval:" + "4" * 64}),
        lambda receipt: receipt.update({"task_id": "replacement-task"}),
        lambda receipt: receipt.update({"release_id": "replacement-release"}),
        lambda receipt: receipt.update({"manifest_sha256": "9" * 64}),
        lambda receipt: receipt.update({"manual_test_cases_sha256": "8" * 64}),
        lambda receipt: receipt.update({"source_commit": "7" * 40}),
        lambda receipt: receipt.update({"bundle_sha256": "6" * 64}),
        lambda receipt: receipt["artifacts"]["php"].update({"sha256": "5" * 64}),
        lambda receipt: receipt.update({"previous_release_id": "caller-selected"}),
        lambda receipt: receipt.update({"database_restore_attempted": True}),
    ],
    ids=(
        "contract-version", "operation-key", "task", "release", "manifest",
        "manual-tests", "source-commit", "bundle", "artifact", "previous-release",
        "database-restore",
    ),
)
def test_receipt_identity_substitution_and_restore_claims_fail_closed(
    tmp_path, monkeypatch, mutate
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    _receipt_adapter(monkeypatch, mutate)
    with connect() as conn:
        task = kb.create_task(conn, title="Receipt identity")
        _present(conn, task)
        result = _approve(conn, ["adapter"], now=100)

    assert not result.ok and result.classification == "adapter_failed"


def test_parallel_double_delivery_invokes_adapter_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = kb.init_db()
    calls = tmp_path / "calls.jsonl"
    with connect(db) as conn:
        task = kb.create_task(conn, title="Race")
        _present(conn, task)

    barrier = threading.Barrier(2)
    results = []
    def run():
        with connect(db) as conn:
            barrier.wait()
            results.append(_approve(conn, _adapter(tmp_path, calls)))
    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(result.ok for result in results) == 1
    assert {result.classification for result in results} <= {"success", "in_progress", "replay"}
    assert len(calls.read_text().splitlines()) == 1
