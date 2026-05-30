import time

from gateway.dev_control.production_signals import (
    DevProductionSignalStore,
    generate_signal_report,
    run_signal_digest_sources,
    sweep_reliability_proposal_outcomes,
)
from gateway.dev_control.reliability import DevReliabilityStore
from gateway.dev_control.signal_source import ReliabilitySignalSource, SignalWindow
from gateway.dev_execution import DevExecutionStore
from gateway.subagent_events import SubagentEventStore


def _success(plan_id: str, task_id: str, category: str, when: float) -> dict:
    profile_id, risk_level = category.split("/", 1)
    return {
        "plan_id": plan_id,
        "task_id": task_id,
        "category": category,
        "profile_id": profile_id,
        "risk_level": risk_level,
        "terminal_status": "completed",
        "merged": True,
        "verification_verdict": "verified",
        "ci_state": "success",
        "code_review_verdict": "approved",
        "output_contract_score": 0.95,
        "completed_at": when,
    }


def _failure(plan_id: str, task_id: str, category: str, when: float, *, escaped: bool = False) -> dict:
    payload = _success(plan_id, task_id, category, when)
    payload.update({
        "verification_verdict": "failed",
        "output_contract_score": 0.4,
        "escaped": escaped,
        "escape_refs": [{"type": "incident", "incident_id": "inc-1"}] if escaped else [],
    })
    return payload


def _seed_trusted_category(store: DevReliabilityStore, *, now: float) -> None:
    for index in range(30):
        store.upsert_outcome(_success(
            f"trusted-plan-{index}",
            f"trusted-task-{index}",
            "workspace.docs/low",
            now - 100 + index,
        ))


def test_reliability_signal_source_clusters_weak_categories_and_excludes_trusted(tmp_path):
    now = time.time()
    reliability_store = DevReliabilityStore(tmp_path / "state.db")
    reliability_store.upsert_outcome(_failure("plan-1", "task-1", "workspace.implement/high", now - 60))
    reliability_store.upsert_outcome(_success("plan-2", "task-2", "workspace.implement/high", now - 30))
    _seed_trusted_category(reliability_store, now=now)

    result = ReliabilitySignalSource(reliability_store).fetch_clusters(SignalWindow(start=now - 3600, end=now))

    keys = {cluster["key"] for cluster in result["clusters"]}
    assert "reliability:workspace.implement/high" in keys
    assert "reliability:workspace.docs/low" not in keys
    cluster = next(item for item in result["clusters"] if item["key"] == "reliability:workspace.implement/high")
    assert cluster["evidence_refs"][0]["outcome_id"]
    assert cluster["metrics"]["tier"] == "unproven"


def test_reliability_report_creates_deduped_guardrail_tagged_proposals(tmp_path):
    now = time.time()
    db_path = tmp_path / "state.db"
    signal_store = DevProductionSignalStore(db_path)
    reliability_store = DevReliabilityStore(db_path)
    event_store = SubagentEventStore(db_path)
    reliability_store.upsert_outcome(_failure("plan-1", "task-1", "workspace.test/high", now - 60))

    first = generate_signal_report(
        signal_store=signal_store,
        event_store=event_store,
        reliability_store=reliability_store,
        source="reliability",
        window_days=7,
    )
    second = generate_signal_report(
        signal_store=signal_store,
        event_store=event_store,
        reliability_store=reliability_store,
        source="reliability",
        window_days=7,
    )

    assert first["proposals"][0]["payload"]["source"] == "reliability"
    assert first["proposals"][0]["cluster_key"] == "reliability:workspace.test/high"
    assert first["proposals"][0]["payload"]["guardrail_touching"] is True
    assert first["proposals"][0]["status"] == "proposed"
    assert first["proposals"][0]["proposal_id"] == second["proposals"][0]["proposal_id"]
    assert len(signal_store.list_proposals()) == 1


def test_promoted_reliability_proposal_measures_after_linked_plan_terminal(tmp_path):
    now = time.time()
    db_path = tmp_path / "state.db"
    signal_store = DevProductionSignalStore(db_path)
    reliability_store = DevReliabilityStore(db_path)
    event_store = SubagentEventStore(db_path)
    execution_store = DevExecutionStore(db_path)
    reliability_store.upsert_outcome(_failure("before-plan", "before-task", "workspace.implement/high", now - 6 * 86400))
    report = generate_signal_report(
        signal_store=signal_store,
        event_store=event_store,
        reliability_store=reliability_store,
        source="reliability",
        window_days=7,
    )
    proposal = report["proposals"][0]
    plan = execution_store.create_plan(
        title="Reliability fix",
        vision_brief="Improve reliability",
        tasks=[{
            "prompt": "Fix recurring verifier issue.",
            "profile_id": "workspace.implement",
            "risk_level": "high",
        }],
    )
    task_id = plan["tasks"][0]["task_id"]
    reliability_store.upsert_outcome(_success(plan["plan_id"], task_id, "workspace.implement/high", now - 60))
    with execution_store._lock, execution_store._conn:
        execution_store._conn.execute("UPDATE dev_execution_plans SET status = ?, updated_at = ? WHERE plan_id = ?", ("completed", now, plan["plan_id"]))
        execution_store._conn.execute("UPDATE dev_execution_plan_tasks SET status = ?, updated_at = ? WHERE plan_id = ?", ("completed", now, plan["plan_id"]))
    signal_store.update_proposal(proposal["proposal_id"], {
        "status": "promoted",
        "linked_plan_id": plan["plan_id"],
        "source_window": {"start": now - 7 * 86400, "end": now - 5 * 86400, "days": 1, "count": 1},
        "payload": {**proposal["payload"], "status": "promoted"},
    })

    sweep = sweep_reliability_proposal_outcomes(
        signal_store=signal_store,
        reliability_store=reliability_store,
        execution_store=execution_store,
        window_days=1,
    )

    measured = sweep["measured"][0]
    assert measured["outcome"]["before_score"] == 0.0
    assert measured["outcome"]["after_score"] == 1.0
    assert measured["outcome"]["status"] == "improved"
    assert signal_store.get_proposal(proposal["proposal_id"])["outcome"]["measured_at"]


def test_digest_runs_configured_sources_and_reliability_measurement_sweep(tmp_path):
    now = time.time()
    db_path = tmp_path / "state.db"
    signal_store = DevProductionSignalStore(db_path)
    event_store = SubagentEventStore(db_path)
    reliability_store = DevReliabilityStore(db_path)
    execution_store = DevExecutionStore(db_path)
    reliability_store.upsert_outcome(_failure("before-plan", "before-task", "workspace.implement/high", now - 6 * 86400))
    initial = generate_signal_report(
        signal_store=signal_store,
        event_store=event_store,
        reliability_store=reliability_store,
        source="reliability",
        window_days=7,
    )
    proposal = initial["proposals"][0]
    plan = execution_store.create_plan(
        title="Reliability fix",
        vision_brief="Improve reliability",
        tasks=[{
            "prompt": "Fix recurring verifier issue.",
            "profile_id": "workspace.implement",
            "risk_level": "high",
        }],
    )
    reliability_store.upsert_outcome(_success(plan["plan_id"], plan["tasks"][0]["task_id"], "workspace.implement/high", now - 60))
    with execution_store._lock, execution_store._conn:
        execution_store._conn.execute("UPDATE dev_execution_plans SET status = ?, updated_at = ? WHERE plan_id = ?", ("completed", now, plan["plan_id"]))
    signal_store.update_proposal(proposal["proposal_id"], {
        "status": "promoted",
        "linked_plan_id": plan["plan_id"],
        "source_window": {"start": now - 7 * 86400, "end": now - 5 * 86400, "days": 1, "count": 1},
        "payload": {**proposal["payload"], "status": "promoted"},
    })
    reliability_store.upsert_outcome(_failure("fresh-plan", "fresh-task", "workspace.review/high", now - 30))

    digest = run_signal_digest_sources(
        signal_store=signal_store,
        event_store=event_store,
        reliability_store=reliability_store,
        execution_store=execution_store,
        sources=["reliability"],
        window_days=1,
    )

    assert digest["advisory_only"] is True
    assert digest["summary"]["weakest_categories_targeted"]
    assert digest["summary"]["reliability_measurements"][0]["status"] == "improved"
