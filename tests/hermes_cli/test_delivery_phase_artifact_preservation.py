"""Exact-artifact preservation and explicit dispatch target compatibility."""
import pytest
from hermes_cli import kanban_db as kb, kanban_db_connect as dbc, kanban_db_dispatch as dispatch

@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with dbc.connect(tmp_path / "test.db") as c:
        yield c

@pytest.mark.parametrize("valid", [True, False])
def test_reopen_preserves_only_exact_artifact_pass(conn, valid):
    parent = kb.create_task(conn, title="parent", assignee="integrator")
    child = kb.create_task(conn, title="review", assignee="pr-reviewer", parents=[parent])
    assert kb.complete_task(conn, parent, summary="phase", fire_lifecycle_hook=False)
    run = kb.claim_task(conn, child, claimer="remote:test")
    assert run
    metadata = {"artifact": "repository@commit", "delivery_review": {
        "head": "a" * 40 if valid else "branch", "verdict": "PASS"}}
    assert kb.complete_task(conn, child, metadata=metadata, summary="proof", fire_lifecycle_hook=False)
    release = kb.create_task(conn, title="release", assignee="release", parents=[child])
    with conn:
        conn.execute("UPDATE tasks SET status='ready', completed_at=NULL WHERE id=?", (parent,))
    result = kb.invalidate_descendants_for_parent_reopen(conn, parent, author="operator")
    kb.recompute_ready(conn)
    assert kb.get_task(conn, release).status == "todo"
    assert kb.complete_task(conn, parent, summary="new phase", fire_lifecycle_hook=False)
    kb.recompute_ready(conn)
    assert kb.get_task(conn, release).status == "todo"
    assert kb.get_task(conn, child).status == "ready"
    assert bool(result["preserved"]) == valid
    assert kb.latest_run(conn, child).metadata == metadata
    assert kb.claim_task(conn, child, claimer="remote:test")
    assert kb.complete_task(conn, child, summary="fresh proof", fire_lifecycle_hook=False,
                            metadata={"delivery_review": {"head": "b" * 40, "verdict": "PASS"}})
    kb.recompute_ready(conn)
    assert kb.get_task(conn, release).status == "ready"

@pytest.mark.parametrize("role,phase,kind,held", [
    ("open_pr_remediation", "merged_checkout", "merged_checkout", True),
    ("post_merge_validation", "merged_checkout", "merged_checkout", False),
    ("open_pr_remediation", "pr_head", "pr_head", False),
    ("post_merge_validation", "merged_checkout", "pr_head", True),
])
def test_dispatch_explicit_role_target_compatibility(conn, role, phase, kind, held):
    task = kb.create_task(conn, title="typed work", assignee="integrator")
    assert kb.complete_task(conn, task, summary="handoff", fire_lifecycle_hook=False,
        metadata={"delivery_dispatch": {"role": role, "phase": phase, "assignee": "integrator",
            "validation_target": {"kind": kind, "commit": "a" * 40}}})
    assert bool(dispatch._delivery_role_hold(conn, task, "integrator")) == held
    assert dispatch._delivery_role_hold(conn, task, "other-worker")

def test_untyped_legacy_work_is_not_inferred_from_prose(conn):
    task = kb.create_task(conn, title="post merge PR", assignee="integrator")
    assert dispatch._delivery_role_hold(conn, task, "integrator") is None

@pytest.mark.parametrize("field", ["role", "phase"])
@pytest.mark.parametrize("value", [[], {}, None])
def test_malformed_contract_holds_only_its_card(conn, monkeypatch, field, value):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _: True)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "normal")
    contract = {"role": "open_pr_remediation", "phase": "pr_head",
                "assignee": "integrator",
                "validation_target": {"kind": "pr_head", "commit": "a" * 40}}
    contract[field] = value
    bad = kb.create_task(conn, title="malformed", assignee="integrator")
    assert kb.complete_task(conn, bad, summary="handoff", fire_lifecycle_hook=False,
                            metadata={"delivery_dispatch": contract})
    with conn:
        conn.execute("UPDATE tasks SET status='ready', completed_at=NULL WHERE id=?", (bad,))
    good = kb.create_task(conn, title="valid", assignee="integrator")
    spawned = []
    result = dispatch.dispatch_once(conn, spawn_fn=lambda task, *_: spawned.append(task.id),
                                    reconcile_orphans=False, max_spawn=2)
    assert any(task == bad and "strings" in reason for task, reason in result.delivery_holds)
    assert spawned == [good]
    assert kb.get_task(conn, bad).status == "ready"
