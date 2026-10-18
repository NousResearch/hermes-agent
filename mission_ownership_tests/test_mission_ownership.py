import hashlib
import os
import sqlite3
import sys
import subprocess
import threading
import types
from pathlib import Path

import pytest

from hermes_cli import mission_ownership as mo


def db(path):
    conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    mo.ensure_schema(conn)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,status TEXT,assignee TEXT,started_at INTEGER,current_run_id INTEGER,
            claim_lock TEXT,claim_expires INTEGER,worker_pid INTEGER,worker_started_at INTEGER);
        CREATE TABLE IF NOT EXISTS task_runs(id INTEGER PRIMARY KEY,status TEXT,worker_pid INTEGER,worker_started_at INTEGER,
            ended_at INTEGER,outcome TEXT,error TEXT);
    """)
    return conn


def native_db(path):
    """Initialize an isolated board with Hermes' real Kanban schema."""
    from hermes_cli.kanban_db_connect import connect
    conn = connect(db_path=path)
    conn.row_factory = sqlite3.Row
    mo.ensure_schema(conn)
    return conn


def contract(kind="implementation"):
    return {"kind": kind, "outcome": "fix the runtime failure", "acceptance": ["runtime change", "focused checks pass"]}


def admit(conn, key="telegram:chat:1:msg:1", owner="session-a"):
    return mo.admit(conn, ingress_key=key, request="FULL original request and acceptance details", contract=contract(), owner_id=owner)


def test_admission_is_durable_non_dispatchable_and_preserves_full_request(tmp_path):
    conn = db(tmp_path / "board.db")
    mission_id, created = admit(conn)
    row = mo.get(conn, mission_id)
    assert created is True
    assert row["original_request"] == "FULL original request and acceptance details"
    assert row["contract"] == contract()
    assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_duplicate_delivery_is_idempotent_but_distinct_objectives_are_not(tmp_path):
    conn = db(tmp_path / "board.db")
    first, _created = admit(conn)
    again, created_again = admit(conn)
    independent, other_created = admit(conn, "telegram:chat:1:msg:2", "session-a")
    assert (again, created_again) == (first, False)
    assert other_created is True and independent != first
    assert conn.execute("SELECT count(*) FROM missions").fetchone()[0] == 2


def test_concurrent_duplicate_admission_creates_one_mission(tmp_path):
    path = tmp_path / "board.db"
    setup = db(path)
    setup.close()
    barrier = threading.Barrier(2)
    results = []

    def deliver():
        conn = db(path)
        barrier.wait()
        results.append(admit(conn, "telegram:chat:1:msg:duplicate", "same-session"))
        conn.close()

    workers = [threading.Thread(target=deliver) for _ in range(2)]
    for worker in workers: worker.start()
    for worker in workers: worker.join()
    assert results[0][0] == results[1][0]
    assert sum(created for _mid, created in results) == 1
    check = db(path)
    assert check.execute("SELECT count(*) FROM missions").fetchone()[0] == 1


def test_handoff_is_atomic_and_preserves_progress_children_and_recovery(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    conn.execute("INSERT INTO tasks(id,status,assignee) VALUES ('t_worker','triage','fred')")
    conn.execute("INSERT INTO tasks(id,status,assignee) VALUES ('t_child','ready','hermes')")
    generation = mo.handoff(
        conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
        expected_generation=1, new_owner_kind="task", new_owner_id="t_worker",
        progress={"done": ["scoped"], "next": "implement"}, child_ids=["t_child"],
        recovery={"checkpoint": "git:abc123", "resume": "rerun focused suite"},
    )
    row = mo.get(conn, mid)
    assert generation == 2
    assert (row["owner_kind"], row["owner_id"]) == ("task", "t_worker")
    assert row["progress"]["next"] == "implement"
    assert row["children"] == ["t_child"]
    assert row["recovery"]["checkpoint"] == "git:abc123"
    assert conn.execute("SELECT status FROM tasks WHERE id='t_worker'").fetchone()[0] == "ready"


def test_old_owner_is_fenced_after_transfer(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    conn.execute("INSERT INTO tasks(id,status,assignee) VALUES ('t_worker','triage','fred')")
    mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
               expected_generation=1, new_owner_kind="task", new_owner_id="t_worker", progress={})
    assert mo.authorize(conn, owner_kind="session", owner_id="session-a") is False
    assert mo.authorize(conn, owner_kind="task", owner_id="t_worker") is True
    with pytest.raises(mo.StaleOwnerError):
        mo.record_progress(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1, progress={"unauthorized": True})


def test_child_task_does_not_transfer_root_ownership(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    conn.execute("INSERT INTO tasks(id,status,assignee) VALUES ('t_specialist','ready','hermes')")
    mo.record_progress(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                       progress={"delegated": True}, child_ids=["t_specialist"])
    row = mo.get(conn, mid)
    assert (row["owner_kind"], row["owner_id"]) == ("session", "session-a")
    assert row["children"] == ["t_specialist"]


def test_interrupted_handoff_rolls_back_without_two_owners(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    conn.execute("INSERT INTO tasks(id,status,assignee) VALUES ('t_worker','triage','fred')")
    conn.commit()
    conn.execute("CREATE TRIGGER fail_handoff BEFORE INSERT ON mission_history WHEN NEW.kind='handoff' BEGIN SELECT RAISE(ABORT,'simulated crash'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                   expected_generation=1, new_owner_kind="task", new_owner_id="t_worker", progress={})
    conn.execute("DROP TRIGGER fail_handoff")
    row = mo.get(conn, mid)
    assert (row["owner_kind"], row["owner_id"], row["owner_generation"]) == ("session", "session-a", 1)
    assert mo.authorize(conn, owner_kind="session", owner_id="session-a") is True
    assert mo.authorize(conn, owner_kind="task", owner_id="t_worker") is False


def test_active_tool_lease_blocks_handoff_then_releases_on_completion(tmp_path, monkeypatch):
    path = tmp_path / "board.db"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    setup = native_db(path)
    mid, _ = admit(setup)
    from hermes_cli import kanban_db as kb
    task_id = kb.create_task(setup, title="Continue mission", body="handoff", assignee="fred", triage=True)
    setup.close()
    started = threading.Event()
    finish = threading.Event()
    errors = []

    def run_long_tool():
        try:
            with mo.execution_scope(session_id="session-a", tool_name="terminal"):
                started.set()
                finish.wait(timeout=5)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_long_tool)
    worker.start()
    assert started.wait(timeout=2)
    conn = native_db(path)
    with pytest.raises(mo.MissionError, match="not quiescent"):
        mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                   expected_generation=1, new_owner_kind="task", new_owner_id=task_id, progress={})
    finish.set()
    worker.join(timeout=3)
    assert not worker.is_alive() and errors == []
    assert conn.execute("SELECT count(*) FROM mission_execution_leases").fetchone()[0] == 0
    assert mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                      expected_generation=1, new_owner_kind="task", new_owner_id=task_id, progress={}) == 2


def test_shared_executor_holds_lease_around_long_running_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "board.db"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "fred")
    conn = native_db(path)
    mid, _ = admit(conn)
    from hermes_cli import kanban_db as kb
    task_id = kb.create_task(conn, title="Continue", body="handoff", assignee="fred", triage=True)
    conn.close()

    from types import SimpleNamespace
    from agent import relay_tools
    from hermes_cli import middleware
    from agent import tool_executor

    class Guardrails:
        def before_call(self, *_args):
            return SimpleNamespace(allows_execution=True)

    class Agent:
        session_id = "session-a"
        _tool_guardrails = Guardrails()

    monkeypatch.setattr(relay_tools, "execute",
                        lambda _name, args, downstream, **_kw: (downstream(args), args))
    monkeypatch.setattr(middleware, "apply_tool_request_middleware",
                        lambda _name, args, **_kw: SimpleNamespace(payload=args, trace=[]))
    monkeypatch.setattr(middleware, "run_tool_execution_middleware",
                        lambda _name, args, downstream, **_kw: downstream(args))
    monkeypatch.setattr(tool_executor, "_pre_tool_block", lambda _agent, _ref: (None, {}))
    monkeypatch.setattr(tool_executor, "_context_pruned_argument_paths", lambda *_args: [])
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_args: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat",
                        lambda _agent, _name, execute: execute())
    from agent import terminal_approval_batch
    monkeypatch.setattr(terminal_approval_batch, "prepare_current_terminal", lambda _ref: None)

    started = threading.Event()
    finish = threading.Event()
    output = []

    def long_tool(_args):
        started.set()
        finish.wait(timeout=5)
        return "finished"

    def dispatch():
        output.append(tool_executor._run_agent_tool_execution_middleware(
            Agent(), function_name="terminal", function_args={"command": "sleep"},
            effective_task_id="turn-local-id", tool_call_id="call-1", execute=long_tool,
        ))

    worker = threading.Thread(target=dispatch)
    worker.start()
    assert started.wait(timeout=2)
    check = native_db(path)
    assert check.execute("SELECT tool_name FROM mission_execution_leases").fetchone()[0] == "terminal"
    with pytest.raises(mo.MissionError, match="not quiescent"):
        mo.handoff(check, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                   expected_generation=1, new_owner_kind="task", new_owner_id=task_id, progress={})
    finish.set()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert output[0].result == "finished"
    assert check.execute("SELECT count(*) FROM mission_execution_leases").fetchone()[0] == 0
    assert mo.handoff(check, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                      expected_generation=1, new_owner_kind="task", new_owner_id=task_id, progress={}) == 2


def test_interrupted_tool_releases_lease_but_crash_left_lease_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "board.db"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    conn = native_db(path)
    mid, _ = admit(conn)
    with pytest.raises(KeyboardInterrupt):
        with mo.execution_scope(session_id="session-a", tool_name="execute_code"):
            raise KeyboardInterrupt()
    assert conn.execute("SELECT count(*) FROM mission_execution_leases").fetchone()[0] == 0

    # Abrupt worker death bypasses the tool finally block; the committed lease
    # survives and blocks transfer/recovery because process-tree liveness is
    # unproven.
    child_env = os.environ.copy()
    child_env["HERMES_HOME"] = str(hermes_home)
    child_env["HERMES_KANBAN_DB"] = str(path)
    child_code = (
        "import os; from hermes_cli import mission_ownership as m; "
        "scope=m.execution_scope(session_id='session-a', tool_name='terminal'); "
        "scope.__enter__(); os._exit(31)"
    )
    crashed = subprocess.run([sys.executable, "-c", child_code], cwd=Path(__file__).parents[1],
                             env=child_env, check=False, timeout=10)
    assert crashed.returncode == 31
    assert conn.execute("SELECT count(*) FROM mission_execution_leases WHERE mission_id=?", (mid,)).fetchone()[0] == 1
    with pytest.raises(mo.MissionError, match="not quiescent"):
        mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                   expected_generation=1, new_owner_kind="session", new_owner_id="session-b", progress={})
    with pytest.raises(mo.MissionError, match="not quiescent"):
        mo.recover(conn, mission_id=mid, expected_generation=1, new_owner_kind="session",
                   new_owner_id="session-b", recovery_actor="operator", now=10**12)


def test_user_correction_versions_contract_and_preserves_original_request(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    changed = {"kind": "implementation", "outcome": "also support CLI", "acceptance": ["CLI path", "gateway path"]}
    version = mo.update_contract(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                                 expected_generation=1, correction="Please include CLI too", contract=changed)
    row = mo.get(conn, mid)
    assert version == 2 and row["contract_version"] == 2
    assert row["original_request"] == "FULL original request and acceptance details"
    assert row["contract"]["outcome"] == changed["outcome"]
    assert row["contract"]["user_updates"] == ["Please include CLI too"]
    assert row["contract"]["acceptance"][-1] == "Honor this user correction: Please include CLI too"
    code = tmp_path / "control.py"
    code.write_text("# fixture\n")
    with pytest.raises(mo.ContractError, match="stale contract"):
        mo.complete(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                    contract_version=1, validation_id="v_stale")


def test_identical_prompts_from_distinct_requests_do_not_attach_across_sessions(tmp_path):
    conn = db(tmp_path / "board.db")
    text = "Please implement mission ownership in the Hermes system and test the repository."
    first = mo.ensure_turn_mission(conn, session_id="telegram-session", text=text, ingress_id="delivery:telegram:1")
    second = mo.ensure_turn_mission(conn, session_id="cli-session", text=text, ingress_id="turn:cli:2")
    assert first["id"] != second["id"]
    assert first["owner_id"] == "telegram-session"
    assert second["owner_id"] == "cli-session"


def test_delivery_identity_rejects_changed_content_and_fallback_keys_are_per_turn(tmp_path):
    conn = db(tmp_path / "board.db")
    first_key = mo.ingress_key(session_id="cli-session", platform="cli", delivery_id=None, turn_id="turn-1")
    assert first_key == mo.ingress_key(
        session_id="cli-session", platform="cli", delivery_id=None, turn_id="turn-1",
    )
    assert first_key != mo.ingress_key(
        session_id="cli-session", platform="cli", delivery_id=None, turn_id="turn-2",
    )
    mo.ensure_turn_mission(conn, session_id="telegram-session", text="Please implement a Hermes task.",
                           ingress_id="delivery:message-1")
    with pytest.raises(mo.MissionError, match="reused for different request"):
        mo.ensure_turn_mission(conn, session_id="telegram-session", text="Please implement a different Hermes task.",
                               ingress_id="delivery:message-1")


def test_october_16_forensic_request_admits_one_non_runnable_mission(tmp_path):
    conn = db(tmp_path / "board.db")
    request = (
        "I've been looking through raw/intake/web/ in X and noticed some of the newer webpage "
        "captures aren't particularly useful. There's a lot of unnecessary navigation, unrelated "
        "content and messy formatting instead of clean, readable Markdown. Can you investigate "
        "what's happening and get this working properly? I want the sources we collect to be useful "
        "for both reading and future knowledge processing."
    )
    ingress = mo.ingress_key(session_id="telegram:20261016", platform="telegram", delivery_id="message-77")
    mission = mo.ensure_turn_mission(conn, session_id="telegram:20261016", text=request, ingress_id=ingress)
    assert mission["contract"]["kind"] == "implementation"
    assert mission["original_request"] == request
    replay = mo.ensure_turn_mission(conn, session_id="telegram:20261016", text=request, ingress_id=ingress)
    assert replay["id"] == mission["id"]
    assert conn.execute("SELECT count(*) FROM missions").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_gateway_telegram_admission_deduplicates_delivery_id_without_text_merging(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from contextlib import nullcontext

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(hermes_home / "kanban.db"))
    from hermes_cli import lifecycle
    from gateway import run_inbound_consumer
    from plugins.mission_admission import _admit_message
    monkeypatch.setattr(run_inbound_consumer, "_routed_scope", lambda *_args: nullcontext())
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "post_gateway_admission")

    async def invoke(_name, **kwargs):
        return [_admit_message(**kwargs)]

    monkeypatch.setattr(lifecycle, "ainvoke_hook", invoke)

    class Source:
        platform = SimpleNamespace(value="telegram")

        def to_dict(self):
            return {"chat_id": "chat-1"}

    class Runner:
        config = SimpleNamespace(multiplex_profiles=False)

    text = "Please implement mission ownership in the Hermes system and test the repository."
    event = SimpleNamespace(message_id="telegram-message-100", text=text)
    handled, reply = asyncio.run(run_inbound_consumer.run_post_admission_hook(
        Runner(), event, Source(), "telegram:chat-1",
    ))
    assert (handled, reply) == (False, None)
    again, _ = asyncio.run(run_inbound_consumer.run_post_admission_hook(
        Runner(), event, Source(), "telegram:chat-1",
    ))
    assert again is False
    with connect_board_at(hermes_home) as conn:
        assert conn.execute("SELECT count(*) FROM missions").fetchone()[0] == 1
        first_id = conn.execute("SELECT id FROM missions").fetchone()[0]
        conn.execute("UPDATE missions SET status='completed' WHERE id=?", (first_id,))
        conn.commit()

    # A genuinely new Telegram message with identical text is a fresh request.
    event2 = SimpleNamespace(message_id="telegram-message-101", text=text)
    asyncio.run(run_inbound_consumer.run_post_admission_hook(
        Runner(), event2, Source(), "telegram:chat-1",
    ))
    with connect_board_at(hermes_home) as conn:
        assert conn.execute("SELECT count(*) FROM missions").fetchone()[0] == 2


def test_user_message_steering_is_preserved_in_current_contract(tmp_path):
    conn = db(tmp_path / "board.db")
    text = "Please implement mission ownership in the Hermes system and test the repository."
    mission = mo.ensure_turn_mission(conn, session_id="session-a", text=text)
    changed = mo.ensure_turn_mission(conn, session_id="session-a", text="Also include CLI support.")
    assert changed["id"] == mission["id"]
    assert changed["original_request"] == text
    assert changed["contract_version"] == 2
    assert changed["contract"]["user_updates"] == ["Also include CLI support."]
    assert changed["contract"]["acceptance"][-1] == "Honor this user correction: Also include CLI support."


def test_one_session_cannot_open_a_second_root_but_substantial_correction_stays_in_contract(tmp_path):
    conn = db(tmp_path / "board.db")
    first = mo.ensure_turn_mission(
        conn, session_id="session-a",
        text="Please implement mission ownership in the Hermes system and test the repository.",
    )
    corrected = mo.ensure_turn_mission(
        conn, session_id="session-a",
        text="Correction: also implement CLI support and synchronize persistent mission state.",
    )
    assert corrected["id"] == first["id"]
    assert corrected["contract_version"] == 2
    with pytest.raises(mo.MissionError, match="already owns active mission"):
        mo.ensure_turn_mission(
            conn, session_id="session-a",
            text="Please create an unrelated billing dashboard in the analytics repository.",
        )
    assert conn.execute("SELECT count(*) FROM missions").fetchone()[0] == 1


def test_unrelated_followup_does_not_mutate_mission_contract(tmp_path):
    conn = db(tmp_path / "board.db")
    text = "Please implement mission ownership in the Hermes system and test the repository."
    mission = mo.ensure_turn_mission(conn, session_id="session-a", text=text)
    result = mo.ensure_turn_mission(conn, session_id="session-a", text="What time is sunset today?")
    assert result["id"] == mission["id"]
    assert result["contract_version"] == 1
    assert result["contract"]["user_updates"] == []


def test_stale_or_document_only_completion_is_refused(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    with pytest.raises(mo.ContractError, match="stale contract"):
        mo.complete(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                    contract_version=0, validation_id="v_fake")
    with pytest.raises(mo.ContractError, match="trusted validation receipt"):
        mo.complete(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                    contract_version=1, validation_id="v_model_fabricated")
    assert mo.get(conn, mid)["status"] == "active"


def test_valid_receipt_completes_and_requires_existing_code_artifact(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    code = "mission_ownership_tests/test_mission_ownership.py"
    from hermes_cli import mission_validation
    receipt = mission_validation.validate(
        conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
        contract_version=1, artifact_paths=[code], reviewer_identity="session:session-a",
    )
    mo.complete(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                contract_version=1, validation_id=receipt["validation_id"])
    assert mo.get(conn, mid)["status"] == "completed"
    with pytest.raises(mo.StaleOwnerError):
        mo.complete(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                    contract_version=1, validation_id=receipt["validation_id"])


def test_stale_artifact_hash_and_fake_empty_check_are_refused(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    code = "mission_ownership_tests/test_mission_ownership.py"
    from hermes_cli import mission_validation
    receipt = mission_validation.validate(
        conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
        contract_version=1, artifact_paths=[code], reviewer_identity="session:session-a",
    )
    path = Path(code)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# changed after validation\n")
        with pytest.raises(mo.ContractError, match="artifact is missing or stale"):
            mo.complete(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                        contract_version=1, validation_id=receipt["validation_id"])
    finally:
        path.write_bytes(original)
    assert mo.get(conn, mid)["status"] == "active"


def test_two_concurrent_handoffs_have_at_most_one_winner(tmp_path):
    path = tmp_path / "board.db"
    root = db(path)
    mid, _ = admit(root)
    root.executemany("INSERT INTO tasks(id,status,assignee) VALUES (?,?,?)",
                     [("worker-a", "triage", "fred"), ("worker-b", "triage", "fred")])
    root.commit()
    root.close()
    results = []
    barrier = threading.Barrier(2)

    def transfer(target):
        conn = db(path)
        barrier.wait()
        try:
            g = mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                           expected_generation=1, new_owner_kind="task", new_owner_id=target, progress={})
            results.append(("ok", target, g))
        except (mo.StaleOwnerError, sqlite3.OperationalError):
            results.append(("lost", target, None))
        finally:
            conn.close()

    threads = [threading.Thread(target=transfer, args=(name,)) for name in ("worker-a", "worker-b")]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    winner = mo.get(db(path), mid)
    assert sum(result[0] == "ok" for result in results) == 1
    assert winner["owner_id"] in {"worker-a", "worker-b"}
    assert winner["owner_generation"] == 2


def test_explicit_recovery_after_worker_death_is_fenced_by_generation(tmp_path):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    conn.execute("INSERT INTO tasks(id,status,assignee) VALUES ('t_worker','triage','fred')")
    generation = mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                            expected_generation=1, new_owner_kind="task", new_owner_id="t_worker", progress={},
                            recovery={"resume": "inspect last run"})
    with pytest.raises(mo.MissionError, match="lease has not expired"):
        mo.recover(conn, mission_id=mid, expected_generation=generation, new_owner_kind="session",
                   new_owner_id="session-b", recovery_actor="operator", now=1)
    new_generation = mo.recover(conn, mission_id=mid, expected_generation=generation, new_owner_kind="session",
                                new_owner_id="session-b", recovery_actor="operator", now=10**12)
    row = mo.get(conn, mid)
    assert new_generation == 3
    assert (row["owner_kind"], row["owner_id"]) == ("session", "session-b")
    assert mo.authorize_actor(conn, session_id="session-a") is False


def test_recovery_waits_for_live_task_worker_then_parks_dead_worker(tmp_path, monkeypatch):
    conn = db(tmp_path / "board.db")
    mid, _ = admit(conn)
    conn.execute("INSERT INTO tasks(id,status,assignee) VALUES ('t_worker','triage','fred')")
    generation = mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="session-a",
                            expected_generation=1, new_owner_kind="task", new_owner_id="t_worker", progress={})
    conn.execute("UPDATE tasks SET status='running',current_run_id=77,worker_pid=? WHERE id='t_worker'", (os.getpid(),))
    conn.execute("INSERT INTO task_runs(id,status,worker_pid,worker_started_at) VALUES (77,'running',?,NULL)", (os.getpid(),))
    conn.commit()
    dispatch = types.ModuleType("hermes_cli.kanban_db_dispatch")
    dispatch._worker_alive = lambda pid, _started: int(pid) == os.getpid()
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db_dispatch", dispatch)
    with pytest.raises(mo.MissionError, match="worker is still alive"):
        mo.recover(conn, mission_id=mid, expected_generation=generation, new_owner_kind="session",
                   new_owner_id="session-b", recovery_actor="operator", now=10**12)
    conn.execute("UPDATE task_runs SET worker_pid=999999999 WHERE id=77")
    conn.commit()
    mo.recover(conn, mission_id=mid, expected_generation=generation, new_owner_kind="session",
               new_owner_id="session-b", recovery_actor="operator", now=10**12)
    assert tuple(conn.execute("SELECT status,worker_pid FROM tasks WHERE id='t_worker'").fetchone()) == ("blocked", None)
    assert tuple(conn.execute("SELECT status,outcome FROM task_runs WHERE id=77").fetchone()) == ("crashed", "crashed")


def test_explicit_mission_resume_transfers_only_the_named_mission(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    db_path = hermes_home / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    conn = native_db(db_path)
    mid, _ = admit(conn)
    conn.execute("UPDATE missions SET owner_expires=0 WHERE id=?", (mid,))
    conn.commit()

    from tools.mission_tools import _resume
    result = _resume({"mission_id": mid, "generation": 1}, session_id="desktop-session")
    assert result["ok"] is True and result["generation"] == 2
    with native_db(db_path) as check:
        assert mo.authorize_actor(check, session_id="session-a") is False
        assert mo.authorize_actor(check, session_id="desktop-session") is True
        assert mo.get(check, mid)["owner_id"] == "desktop-session"


def test_mission_completion_waits_for_recorded_native_child_tasks(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    db_path = hermes_home / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    conn = native_db(db_path)
    mid, _ = admit(conn)
    from hermes_cli import kanban_db as kb
    child = kb.create_task(conn, title="Child verification", body="complete child", assignee="hermes")
    mo.record_progress(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                       progress={"next": "wait for child"}, child_ids=[child])

    from hermes_cli import mission_validation
    receipt = mission_validation.validate(
        conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
        contract_version=1, artifact_paths=["mission_ownership_tests/test_mission_ownership.py"],
        reviewer_identity="session:session-a",
    )
    with pytest.raises(mo.ContractError, match="child task must exist and be done or archived"):
        mo.complete(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                    contract_version=1, validation_id=receipt["validation_id"])
    assert mo.get(conn, mid)["status"] == "active"
    assert kb.complete_task(conn, child, summary="Child complete") is True
    mo.complete(conn, mission_id=mid, owner_kind="session", owner_id="session-a", generation=1,
                contract_version=1, validation_id=receipt["validation_id"])


def test_core_turn_admission_and_tool_fence_ignore_optional_hook_failures(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(hermes_home / "kanban.db"))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "fred")
    from agent.turn_context import _admit_fred_mission

    class Agent:
        session_id = "telegram-session"

    request = "Please implement mission ownership in the Hermes system and test the repository."
    context = _admit_fred_mission(Agent(), request)
    assert "mission_id=m_" in context
    with connect_board() as conn:
        mission = mo.list_active(conn)[0]
        generation = mo.handoff(
            conn, mission_id=mission["id"], expected_owner_kind="session",
            expected_owner_id="telegram-session", expected_generation=1,
            new_owner_kind="session", new_owner_id="cli-session", progress={"next": "resume"},
        )
    assert generation == 2

    # The core fence runs before optional hooks, even when a plugin throws or
    # the caller explicitly skips plugin hooks.
    from hermes_cli import plugins
    from model_tools import _CallIds, _pre_dispatch_guards
    called = []

    def throwing_hook(*_args, **_kwargs):
        called.append(True)
        raise RuntimeError("simulated hook crash")

    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", throwing_hook)
    for skip_hook in (False, True):
        _args, blocked = _pre_dispatch_guards(
            "write_file", {"path": str(tmp_path / "unauthorized.py")}, skip_hook,
            _CallIds(session_id="telegram-session"), [],
        )
        assert blocked is not None and blocked[1] == "mission_owner_stale"
    assert called == []  # core denial occurred before the throwing plugin

    import hermes_cli.kanban_db_connect as kb_connect
    monkeypatch.setattr(kb_connect, "connect_closing", lambda: (_ for _ in ()).throw(OSError("board unavailable")))
    _args, blocked = _pre_dispatch_guards(
        "write_file", {"path": str(tmp_path / "denied.py")}, True,
        _CallIds(session_id="fresh-fred-session"), [],
    )
    assert blocked is not None and blocked[1] == "mission_authority_error"


def test_shared_tool_executor_fences_inline_tools_after_handoff(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(hermes_home / "kanban.db"))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "fred")
    from agent.tool_executor import _ManagedToolResult, _ToolCallRef, _dispatch_authorized_once
    from agent import tool_executor

    class Agent:
        session_id = "old-session"

    with connect_board_at(hermes_home) as conn:
        mid, _ = mo.admit(conn, ingress_key="telegram:1:2:3", request="implement this", contract=contract(), owner_id="old-session")
        from hermes_cli import kanban_db as kb
        new_task = kb.create_task(conn, title="Continue mission", body="handoff", assignee="fred", triage=True)
        mo.handoff(conn, mission_id=mid, expected_owner_kind="session", expected_owner_id="old-session",
                   expected_generation=1, new_owner_kind="task", new_owner_id=new_task, progress={})

    events = []
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call",
                        lambda *args, **kwargs: events.append(kwargs))
    state = _ManagedToolResult(None, {}, [], False, False)
    called = []
    result = _dispatch_authorized_once(
        Agent(), state, _ToolCallRef("delegate_task", {}, "", "call-1", []),
        execute=lambda _args: called.append(True), scope_block=None, display_index=None,
        begin_execution=None, authorization_gate=None,
    )
    assert state.blocked is True
    assert called == []
    assert "no longer owns" in result
    assert events and events[0]["error_type"] == "mission_owner_stale"


def test_foreground_background_handoff_and_completion_guard_use_native_kanban(tmp_path, monkeypatch):
    import re

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(hermes_home / "kanban.db"))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "fred")
    from agent.turn_context import _admit_fred_mission
    from tools.mission_tools import _complete, _handoff
    from hermes_cli import kanban_db as kb

    class Agent:
        session_id = "foreground-session"

    request = "Please implement mission ownership in the Hermes system and test the repository."
    context = _admit_fred_mission(Agent(), request)
    mission_id = re.search(r"mission_id=(m_[a-f0-9]+)", context).group(1)
    handoff_result = _handoff({
        "mission_id": mission_id, "generation": 1, "new_owner_kind": "task",
        "target_title": "Continue mission ownership implementation",
        "target_body": "Implement the accepted runtime contract.", "target_assignee": "fred",
        "progress": {"completed": ["scoped"], "next": "implement and verify"},
        "child_ids": [], "recovery": {"resume": "rerun the focused suite"},
    }, session_id="foreground-session")
    task_id = handoff_result["owner"][1]
    assert handoff_result["ok"] is True

    with connect_board_at(hermes_home) as conn:
        assert mo.authorize_actor(conn, session_id="foreground-session") is False
        assert mo.authorize_actor(conn, task_id=task_id) is True
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert "original_request" in task.body and request in task.body
        with pytest.raises(mo.MissionError, match="trusted validation receipt first"):
            kb.complete_task(conn, task_id, summary="A documentation note was updated")

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    from hermes_cli import mission_validation
    with connect_board_at(hermes_home) as conn:
        receipt = mission_validation.validate(
            conn, mission_id=mission_id, owner_kind="task", owner_id=task_id,
            generation=2, contract_version=1,
            artifact_paths=["mission_ownership_tests/test_mission_ownership.py"],
            reviewer_identity=f"task:{task_id}",
        )
    complete_result = _complete({"mission_id": mission_id, "generation": 2,
                                 "contract_version": 1, "validation_id": receipt["validation_id"]})
    assert complete_result["status"] == "completed"
    with connect_board_at(hermes_home) as conn:
        assert kb.get_task(conn, task_id).status == "done"
        assert mo.get(conn, mission_id)["status"] == "completed"


def test_failed_root_card_handoff_rolls_back_card_and_mission_transfer(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(hermes_home / "kanban.db"))
    with connect_board_at(hermes_home) as conn:
        mission_id, _ = admit(conn, owner="foreground-session")

    from tools import mission_tools

    def fail_transfer(*_args, **_kwargs):
        raise mo.MissionError("simulated handoff failure")

    monkeypatch.setattr(mo, "handoff", fail_transfer)
    with pytest.raises(mo.MissionError, match="simulated handoff failure"):
        mission_tools._handoff({
            "mission_id": mission_id,
            "generation": 1,
            "new_owner_kind": "task",
            "target_title": "Continue mission",
            "target_body": "Use the recorded contract.",
            "target_assignee": "fred",
            "progress": {},
        }, session_id="foreground-session")

    with connect_board_at(hermes_home) as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
        mission = mo.get(conn, mission_id)
        assert (mission["owner_kind"], mission["owner_id"], mission["owner_generation"]) == (
            "session", "foreground-session", 1,
        )


def connect_board_at(hermes_home):
    from hermes_cli.kanban_db_connect import connect_closing
    return connect_closing(db_path=hermes_home / "kanban.db")


def connect_board():
    """Open the disposable Hermes Kanban board used by runtime integration tests."""
    from hermes_cli.kanban_db_connect import connect_closing
    return connect_closing()
