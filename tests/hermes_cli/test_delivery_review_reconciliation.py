
from hermes_cli import kanban_db_connect, kanban_db_dispatch, kanban_db_notify
"""Controller-owned review continuation, with real isolated DB/tool/dispatch paths."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tools import kanban_tools as tools

A = "a" * 40
B = "b" * 40


@pytest.fixture
def conn(tmp_path, monkeypatch):
    db = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _: True)
    monkeypatch.setattr(kanban_db_dispatch, "_memory_pressure_level", lambda: "normal")
    with kanban_db_connect.connect(db) as c:
        assert Path(c.execute("PRAGMA database_list").fetchone()[2]) == db
        yield c


def pair(conn):
    parent = kb.create_task(conn, title="Implementation", assignee="integrator")
    child = kb.create_task(conn, title="Independent review", assignee="pr-reviewer", parents=[parent])
    return parent, child


def candidate(child, head):
    return {"review_requirement": {"required": True, "owner": "orchestrator", "review_task_id": child},
            "delivery_review": {"head": head,
            "evidence": {"artifact": "https://github.com/example/collector/pull/3340",
                         "checks": [{"command": "focused test", "result": "passed"}],
                         "ci_head": head, "ci": "success", "draft": False,
                         "proof_head": head, "proof": "passed"}}}


def submit(conn, parent, child, head):
    run = kb.claim_task(conn, parent, claimer="remote:test")
    assert run
    assert kb.complete_task(conn, parent, summary="Implementation phase; awaiting independent review",
                             metadata=candidate(child, head), expected_run_id=run.current_run_id)


def tick(conn, **kw):
    return kanban_db_dispatch.dispatch_once(conn, spawn_fn=lambda *args: None, reconcile_orphans=False, **kw)


def verdict(conn, child, head, value):
    run = kb.get_task(conn, child)
    assert run.status == "running"
    assert kb.complete_task(conn, child, expected_run_id=run.current_run_id,
        summary=f"{value} exact candidate", metadata={"delivery_review": {
            "head": head, "verdict": value, "findings": "Fix ordering" if value == "BLOCK" else ""}})


def test_done_review_corrected_head_without_cross_card_authority(conn, monkeypatch):
    parent, child = pair(conn)
    origin = dict(task_id=parent, platform="discord", chat_id="fixture", thread_id="origin")
    kanban_db_notify.add_notify_sub(conn, **origin)
    submit(conn, parent, child, A)
    # Handoff is consumed even with no capacity; no user completion/requeue call.
    assert not tick(conn, max_spawn=0).spawned
    assert kb.get_task(conn, parent).status == "done"  # phase, not acceptance
    assert kb.get_task(conn, child).status == "ready"
    assert [x[0] for x in tick(conn).spawned] == [child]
    verdict(conn, child, A, "BLOCK")
    assert not tick(conn, max_spawn=0).spawned
    assert kb.get_task(conn, parent).status == "ready"
    # Same implementation, same reviewer. Worker still cannot reopen another card.
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    monkeypatch.setattr(tools, "_is_dispatcher_owned_worker", lambda: True)
    denied = json.loads(tools._handle_complete({"task_id": child, "summary": "forbidden"}))
    assert "refusing to mutate" in str(denied)
    assert kb.get_task(conn, child).status == "done"
    submit(conn, parent, child, B)
    assert [x[0] for x in tick(conn).spawned] == [child]
    assert not tick(conn).spawned
    verdict(conn, child, B, "PASS")
    for _ in range(3):
        assert not tick(conn).spawned
    events = kb.list_events(conn, parent)
    assert len([e for e in events if e.kind == "delivery_accepted"]) == 1
    assert len(kb.list_tasks(conn)) == 2
    result_event = [e for e in kb.list_events(conn, child) if e.kind == "completed"][-1]
    assert result_event.payload["completion_kind"] == "delivery_review_result"
    assert result_event.payload["ready"] is False
    assert len(kanban_db_notify.list_notify_subs(conn, parent)) == 1
    kinds = ("delivery_phase_completed", "delivery_changes_requested", "delivery_accepted")
    notices = kanban_db_notify.claim_unseen_events_for_sub(conn, **origin, kinds=kinds)[2]
    assert len(notices) == 4
    assert not kanban_db_notify.claim_unseen_events_for_sub(conn, **origin, kinds=kinds)[2]


@pytest.mark.parametrize("hold", ["blocked", "triage", "crashed", "exhausted"])
def test_controller_never_revives_parked_reviewer(conn, hold):
    parent, child = pair(conn)
    submit(conn, parent, child, A)
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET status=?, consecutive_failures=?, last_failure_error=? WHERE id=?",
                     ("blocked" if hold in {"crashed", "exhausted"} else hold,
                      3 if hold == "exhausted" else 0, hold, child))
        kb._append_event(conn, child, "gave_up", {"reason": hold})
    for _ in range(3):
        assert not tick(conn).spawned
    assert kb.get_task(conn, child).status in {"blocked", "triage"}
    assert not any(e.kind == "delivery_accepted" for e in kb.list_events(conn, parent))


@pytest.mark.parametrize("invalid", ["ci", "draft", "proof", "stale_review", "scientific_block"])
def test_no_acceptance_without_exact_gates(conn, invalid):
    parent, child = pair(conn)
    data = candidate(child, A)
    evidence = data["delivery_review"]["evidence"]
    if invalid == "ci":
        evidence["ci"] = "failure"
    if invalid == "draft":
        evidence["draft"] = True
    if invalid == "proof":
        evidence["proof_head"] = B
    run = kb.claim_task(conn, parent, claimer="remote:test")
    assert kb.complete_task(conn, parent, metadata=data, expected_run_id=run.current_run_id)
    tick(conn)
    verdict(conn, child, B if invalid == "stale_review" else A,
            "BLOCK" if invalid == "scientific_block" else "PASS")
    for _ in range(3):
        tick(conn, max_spawn=0)
    assert not any(e.kind == "delivery_accepted" for e in kb.list_events(conn, parent))


def test_same_sha_cannot_replenish_review_or_rework(conn):
    parent, child = pair(conn)
    submit(conn, parent, child, A)
    tick(conn)
    verdict(conn, child, A, "BLOCK")
    tick(conn, max_spawn=0)
    submit(conn, parent, child, A)
    for _ in range(3):
        assert not tick(conn).spawned
    assert kb.get_task(conn, child).status == "done"
    assert len([e for e in kb.list_events(conn, parent) if e.kind == "delivery_changes_requested"]) == 1


def test_missing_reviewer_metadata_never_emits_review_approved(conn):
    parent, child = pair(conn)
    submit(conn, parent, child, A)
    tick(conn)
    run = kb.get_task(conn, child)
    assert kb.complete_task(conn, child, summary="No structured verdict", expected_run_id=run.current_run_id)
    result = tick(conn)
    assert result.delivery_holds
    event = [e for e in kb.list_events(conn, child) if e.kind == "completed"][-1]
    assert event.payload["completion_kind"] == "delivery_review_result"
    assert event.payload["ready"] is False
    assert not any(e.kind == "delivery_accepted" for e in kb.list_events(conn, parent))


def test_ordinary_same_card_review_unchanged(conn):
    tid = kb.create_task(conn, title="Ordinary", assignee="integrator")
    run = kb.claim_task(conn, tid, claimer="remote:test")
    assert kb.request_review(conn, tid, reviewer="pr-reviewer", expected_run_id=run.current_run_id)
    assert [x[0] for x in tick(conn).spawned] == [tid]


@pytest.mark.parametrize("tamper", ["head", "checks", "binding"])
def test_malformed_candidate_fails_closed_with_deduped_visible_hold(conn, tamper):
    parent, child = pair(conn)
    data = candidate(child, A)
    if tamper == "head":
        data["delivery_review"]["head"] = "short"
    elif tamper == "checks":
        data["delivery_review"]["evidence"]["checks"] = []
    else:
        # A second child makes the automatic pair selection ambiguous.
        kb.create_task(conn, title="Unrelated child", assignee="qa", parents=[parent])
    run = kb.claim_task(conn, parent, claimer="remote:test")
    assert kb.complete_task(conn, parent, metadata=data, expected_run_id=run.current_run_id)
    for _ in range(3):
        result = tick(conn)
        assert not result.spawned
        assert result.delivery_holds
    assert len([e for e in kb.list_events(conn, parent) if e.kind == "delivery_review_hold"]) == 1


def test_concurrent_controllers_consume_one_generation(conn):
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.kanban_review_reconcile import reconcile
    parent, child = pair(conn)
    submit(conn, parent, child, A)
    db = Path(conn.execute("PRAGMA database_list").fetchone()[2])

    def run(_):
        with kanban_db_connect.connect(db) as other:
            return reconcile(other)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))
    assert len([e for e in kb.list_events(conn, parent) if e.kind == "delivery_phase_completed"]) == 1
    assert len([e for e in kb.list_events(conn, child) if e.kind == "delivery_review_enqueued"]) == 1
    assert [x[0] for x in tick(conn).spawned] == [child]
    assert not tick(conn).spawned


@pytest.mark.parametrize("guard", ["auth", "quota"])
def test_controller_keeps_failure_and_quota_gates(conn, guard):
    parent, child = pair(conn)
    submit(conn, parent, child, A)
    tick(conn, max_spawn=0)
    if guard == "auth":
        with kanban_db_connect.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error='401 Unauthorized' WHERE id=?", (child,))
    else:
        with kanban_db_connect.write_txn(conn):
            kb._synthesize_ended_run(conn, child, outcome="rate_limited", summary="quota held")
    result = tick(conn)
    assert not result.spawned
    assert (child, "blocker_auth" if guard == "auth" else "rate_limit_cooldown") in result.respawn_guarded


def test_collector_incident_real_cli_capacity_and_stale_review_context(conn, tmp_path):
    """PR3340 incident shape: stale PR3329 body, existing capacity, no user ping."""
    import subprocess
    import sys

    parent, child = pair(conn)
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET body='OLD: review PR3329 with Claude only' WHERE id=?", (child,))
        conn.execute("UPDATE tasks SET model_override='gpt-6-astra', provider_override='openai-codex', reasoning_effort='ultra' WHERE id=?", (child,))
    busy = kb.create_task(conn, title="Independent collector worker", assignee="default")
    kb.claim_task(conn, busy, claimer="remote:busy")
    data = candidate(child, A)
    data["delivery_review"]["evidence"]["artifact"] = "https://github.com/example/collector/pull/3340"
    run = kb.claim_task(conn, parent, claimer="remote:test")
    assert kb.complete_task(conn, parent, metadata=data, expected_run_id=run.current_run_id)
    db = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    env = {"HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "home"),
           "HERMES_KANBAN_DB": str(db), "PATH": str(tmp_path / "empty"),
           "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
    root = Path(__file__).resolve().parents[2]

    def cli(*args):
        result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", *args],
                                cwd=root, env=env, text=True, capture_output=True, timeout=40)
        assert result.returncode == 0, result.stderr + result.stdout
        return result.stdout

    held = json.loads(cli("dispatch", "--max", "1", "--dry-run", "--json"))
    assert held["capacity_hold"] == {"scope": "board", "running": 1, "limit": 1}
    assert held["spawned"] == []
    assert "total concurrency" in cli("dispatch", "--max", "1", "--dry-run")
    assert kb.get_task(conn, child).status == "ready"
    assert kb.get_task(conn, busy).status == "running"
    assert kb.complete_task(conn, busy, summary="Fixture busy worker finished")
    # Real CLI resolves profiles. Default exists without a live profile tree;
    # the candidate's profile existence is provided by an isolated profile dir.
    profile = tmp_path / "home" / "profiles" / "pr-reviewer"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model: gpt-6-astra\n")
    resumed = json.loads(cli("dispatch", "--max", "1", "--dry-run", "--json"))
    assert [x["task_id"] for x in resumed["spawned"]] == [child]
    context = kb.build_worker_context(conn, child)
    assert "gpt-6-astra" in context and "ultra" in context
    assert "supersedes inherited body/skill lane text" in context
    assert context.index("pull/3340") < context.index("OLD: review PR3329")
    assert A in context


def test_authorized_astra_uses_native_spawn_not_inherited_claude(conn, tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    tid = kb.create_task(conn, title="Authorized Astra", assignee="integrator",
                         model_override="gpt-6-astra", provider_override="openai-codex",
                         reasoning_effort="ultra")
    task = kb.get_task(conn, tid)
    monkeypatch.setattr(kanban_db_dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: (captured.append(cmd) or SimpleNamespace(pid=4245)))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kanban_db_dispatch._default_spawn(task, str(workspace))
    cmd = captured[0]
    assert cmd[0] == "hermes"
    assert cmd[cmd.index("-m") + 1] == "gpt-6-astra"
    assert cmd[cmd.index("--provider") + 1] == "openai-codex"
    assert "ultra" in cmd
