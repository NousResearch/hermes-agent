"""Deterministic offline regression fixture for bounded research loops.

The fixture models the destination-planning incident that motivated the P0
research controls without naming a destination in production code and without
making a network request.  It uses the real Kanban DB, dispatcher envelope,
agent override, and ToolCallGuardrailController.  Only the collection results
and their logical latency are synthetic.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping
from unittest.mock import patch

from agent.agent_init import _apply_display_config
from agent.tool_guardrails import (
    RESEARCH_BUDGET_ENV,
    RESEARCH_COLLECTION_STATE,
    RESEARCH_INTENT_FIELD,
    RESEARCH_MODE_ENV,
    RESEARCH_SYNTHESIS_STATE,
    RESEARCH_TERMINAL_STATE,
    ToolCallGuardrailController,
)
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_failure import (
    FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS,
    checkpoint_evidence,
    classify_failure,
    failure_code,
    read_checkpoint,
    write_research_checkpoint,
)
from tools import kanban_tools


FIXTURE_VERSION = "p0e-offline-v2"
WORKER_PID = 48123
BASELINE_TIMEOUT_SECONDS = 30.0
SYNTHESIS_LATENCY_SECONDS = 0.5

# This is the exact typed payload used by the native task boundary.  The
# labels are intentionally generic so the fixture proves an intent budget,
# rather than a destination-specific exception.
TASK_POLICY: dict[str, Any] = {
    "web_search_max": 4,
    "browser_extract_max": 4,
    "repeated_intent_max": 2,
    "collection_deadline_seconds": 10,
    "synthesis_reserve_seconds": 2,
    "collection_tools": ["web_search", "web_extract"],
}

_RUNTIME_ENV_KEYS = (
    "HERMES_HOME",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_RESEARCH_BUDGET",
    "HERMES_KANBAN_RESEARCH_MODE",
    "HERMES_KANBAN_CHECKPOINT",
    "HERMES_KANBAN_CHECKPOINT_DIR",
    "HERMES_DELEGATED_CHILD_CONTEXT",
    "HERMES_PROFILE",
    "TERMINAL_CWD",
)


@dataclass(frozen=True)
class FixtureCall:
    """One synthetic collection call made by the scripted model planner."""

    ordinal: int
    tool: str
    intent: str
    latency_seconds: float

    def arguments(self) -> dict[str, Any]:
        if self.tool == "web_search":
            return {
                "query": f"fixture {self.intent} probe {self.ordinal}",
                RESEARCH_INTENT_FIELD: self.intent,
            }
        return {
            "urls": [f"fixture://{self.intent}/{self.ordinal}"],
            RESEARCH_INTENT_FIELD: self.intent,
        }

    @property
    def result(self) -> str:
        return json.dumps(
            {
                "source": "offline-fixture",
                "intent": self.intent,
                "ordinal": self.ordinal,
                "fact": "synthetic evidence",
            },
            separators=(",", ":"),
        )


@dataclass
class SimulatedClock:
    """Logical clock used for deterministic runtime metrics."""

    seconds: float = 0.0

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


@dataclass
class ScenarioMetrics:
    """Auditable counters for one before/after/recovery scenario."""

    name: str
    logical_runtime_seconds: float
    collection_calls_attempted: int
    collection_calls_executed: int
    web_search_calls: int
    browser_extract_calls: int
    retry_calls: int
    blocked_collection_calls: int
    terminal_tool_calls: int
    terminal_outcome: str
    result_quality_status: str
    evidence_count: int
    preserved_evidence: bool
    guardrail_code: str | None
    guardrail_state: str | None
    recovery_mode: str | None
    lifecycle_states: list[str]
    collection_disabled_after_transition: bool
    finalization_tools_available: list[str]
    events: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scripted_calls() -> tuple[FixtureCall, ...]:
    """Return a fixed reworded-search/extract loop, with no external URLs."""
    searches = tuple(
        FixtureCall(i + 1, "web_search", "weather", 1.5)
        for i in range(10)
    )
    extracts = tuple(
        FixtureCall(i + 11, "web_extract", "hours", 3.0)
        for i in range(5)
    )
    return searches + extracts


def _budget_probe_calls(tool: str) -> tuple[FixtureCall, ...]:
    """Return distinct-intent calls that reach one independent collection cap."""
    return tuple(
        FixtureCall(i + 1, tool, f"{tool}-probe-{i + 1}", 0.25)
        for i in range(5)
    )


def _policy_from_controller(controller: ToolCallGuardrailController) -> dict[str, Any]:
    policy = controller.config.research_budget
    return {
        "web_search_max": policy.web_search_max,
        "browser_extract_max": policy.browser_extract_max,
        "repeated_intent_max": policy.repeated_intent_max,
        "collection_deadline_seconds": policy.collection_deadline_seconds,
        "synthesis_reserve_seconds": policy.synthesis_reserve_seconds,
        "collection_tools": sorted(policy.collection_tools),
    }


def _append_event(
    events: list[dict[str, Any]],
    *,
    phase: str,
    ordinal: int,
    tool: str,
    intent: str | None,
    status: str,
    decision_code: str,
    logical_runtime_seconds: float,
    retry: bool = False,
    state: str | None = None,
) -> None:
    events.append(
        {
            "phase": phase,
            "ordinal": ordinal,
            "tool": tool,
            "intent": intent,
            "status": status,
            "decision_code": decision_code,
            "state": state,
            "logical_runtime_seconds": round(logical_runtime_seconds, 3),
            "retry": retry,
        }
    )


def _finalize(
    controller: ToolCallGuardrailController,
    clock: SimulatedClock,
    events: list[dict[str, Any]],
    *,
    ordinal: int,
) -> tuple[int, list[str], bool, str | None]:
    """Run the real synthesis/finalization seam after collection stops."""
    args = {"command": "write final report from fixture evidence"}
    finalization_tools = ("terminal", "read_file", "write_file", "kanban_complete")
    available_tools = [
        tool_name
        for tool_name in finalization_tools
        if controller.before_call(tool_name, {}).allows_execution
    ]
    collection_probe = controller.before_call(
        "web_search",
        {"query": "post-synthesis collection probe", RESEARCH_INTENT_FIELD: "post-synthesis"},
    )
    if collection_probe.allows_execution:
        raise AssertionError("collection remained available after the synthesis transition")
    synthesis_state = (
        controller.research_budget_metadata or {}
    ).get("state")
    decision = controller.before_call("terminal", args)
    if not decision.allows_execution:
        raise AssertionError(f"finalization unexpectedly blocked: {decision}")
    clock.advance(SYNTHESIS_LATENCY_SECONDS)
    after = controller.after_call(
        "terminal",
        args,
        '{"status":"written","quality":"explicit_unknowns"}',
        failed=False,
    )
    _append_event(
        events,
        phase="synthesize",
        ordinal=ordinal,
        tool="terminal",
        intent=None,
        status="executed",
        decision_code=after.code,
        logical_runtime_seconds=clock.seconds,
        state=synthesis_state,
    )
    controller.mark_terminal()
    terminal_state = (controller.research_budget_metadata or {}).get("state")
    if terminal_state != RESEARCH_TERMINAL_STATE:
        raise AssertionError(f"finalization did not close the lifecycle: {terminal_state!r}")
    return 1, available_tools, True, synthesis_state


def _run_script(
    controller: ToolCallGuardrailController,
    calls: tuple[FixtureCall, ...],
    *,
    name: str,
    clock: SimulatedClock,
    finalize: bool,
    result_quality_status: str,
    timeout_seconds: float | None = None,
    preserved_evidence: bool = False,
) -> ScenarioMetrics:
    """Execute scripted calls through the real guardrail controller."""
    seen_intents: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    attempted = executed = searches = extracts = retries = blocked = 0
    guardrail_code: str | None = None
    synthesis_ordinal = 0
    lifecycle_states: list[str] = []
    initial_metadata = controller.research_budget_metadata
    if initial_metadata is not None:
        initial_state = initial_metadata.get("state")
        if initial_state in {RESEARCH_COLLECTION_STATE, RESEARCH_SYNTHESIS_STATE}:
            lifecycle_states.append(initial_state)

    for call in calls:
        attempted += 1
        prior = seen_intents.get(call.intent, 0)
        retry = prior > 0
        seen_intents[call.intent] = prior + 1
        if retry:
            retries += 1
        args = call.arguments()
        decision = controller.before_call(call.tool, args)
        if not decision.allows_execution:
            blocked += 1
            guardrail_code = decision.code
            _append_event(
                events,
                phase="collect",
                ordinal=call.ordinal,
                tool=call.tool,
                intent=call.intent,
                status="blocked",
                decision_code=decision.code,
                logical_runtime_seconds=clock.seconds,
                retry=retry,
                state=decision.state,
            )
            if decision.state == RESEARCH_SYNTHESIS_STATE and RESEARCH_SYNTHESIS_STATE not in lifecycle_states:
                lifecycle_states.append(RESEARCH_SYNTHESIS_STATE)
            break

        clock.advance(call.latency_seconds)
        after = controller.after_call(call.tool, args, call.result, failed=False)
        executed += 1
        synthesis_ordinal = call.ordinal + 1
        if call.tool == "web_search":
            searches += 1
        elif call.tool == "web_extract":
            extracts += 1
        if after.code != "allow":
            guardrail_code = after.code
        _append_event(
            events,
            phase="collect",
            ordinal=call.ordinal,
            tool=call.tool,
            intent=call.intent,
            status="executed",
            decision_code=after.code,
            logical_runtime_seconds=clock.seconds,
            retry=retry,
            state=after.state,
        )
        if after.state == RESEARCH_SYNTHESIS_STATE and RESEARCH_SYNTHESIS_STATE not in lifecycle_states:
            lifecycle_states.append(RESEARCH_SYNTHESIS_STATE)
        if timeout_seconds is not None and clock.seconds >= timeout_seconds:
            _append_event(
                events,
                phase="runtime",
                ordinal=call.ordinal,
                tool="worker",
                intent=None,
                status="timeout",
                decision_code="TIMEOUT_BEFORE_SYNTHESIS",
                logical_runtime_seconds=clock.seconds,
            )
            break

    terminal_calls = 0
    collection_disabled_after_transition = False
    finalization_tools_available: list[str] = []
    if finalize:
        (
            terminal_calls,
            finalization_tools_available,
            collection_disabled_after_transition,
            synthesis_state,
        ) = _finalize(
            controller,
            clock,
            events,
            ordinal=synthesis_ordinal or attempted + 1,
        )
        if synthesis_state == RESEARCH_SYNTHESIS_STATE and RESEARCH_SYNTHESIS_STATE not in lifecycle_states:
            lifecycle_states.append(RESEARCH_SYNTHESIS_STATE)
        if RESEARCH_TERMINAL_STATE not in lifecycle_states:
            lifecycle_states.append(RESEARCH_TERMINAL_STATE)
        terminal_outcome = "completed"
    else:
        terminal_outcome = "timeout_before_synthesis"

    metadata = controller.research_budget_metadata
    evidence_count = (
        int(metadata.get("evidence_count") or 0)
        if metadata is not None
        else executed
    )
    state = metadata.get("state") if metadata else None
    recovery_mode = metadata.get("recovery_mode") if metadata else None
    return ScenarioMetrics(
        name=name,
        logical_runtime_seconds=round(clock.seconds, 3),
        collection_calls_attempted=attempted,
        collection_calls_executed=executed,
        web_search_calls=searches,
        browser_extract_calls=extracts,
        retry_calls=retries,
        blocked_collection_calls=blocked,
        terminal_tool_calls=terminal_calls,
        terminal_outcome=terminal_outcome,
        result_quality_status=result_quality_status,
        evidence_count=evidence_count,
        preserved_evidence=preserved_evidence,
        guardrail_code=guardrail_code,
        guardrail_state=state,
        recovery_mode=recovery_mode,
        lifecycle_states=lifecycle_states,
        collection_disabled_after_transition=collection_disabled_after_transition,
        finalization_tools_available=finalization_tools_available,
        events=events,
    )


@contextmanager
def _isolated_runtime(root: Path) -> Iterator[Path]:
    """Give the real Kanban modules a disposable HOME and board."""
    saved = {key: os.environ.get(key) for key in _RUNTIME_ENV_KEYS}
    for key in _RUNTIME_ENV_KEYS:
        os.environ.pop(key, None)
    hermes_home = root / ".hermes"
    db_path = root / "kanban.db"
    workspaces_root = root / "workspaces"
    hermes_home.mkdir(parents=True, exist_ok=True)
    workspaces_root.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HERMES_HOME": str(hermes_home),
            "HERMES_KANBAN_DB": str(db_path),
            "HERMES_KANBAN_WORKSPACES_ROOT": str(workspaces_root),
            "HERMES_PROFILE": "fixture",
        }
    )
    with patch.object(Path, "home", lambda: root):
        kb._INITIALIZED_PATHS.clear()
        try:
            yield root
        finally:
            kb._INITIALIZED_PATHS.clear()
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def _environment_overlay(values: Mapping[str, str]) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in (RESEARCH_BUDGET_ENV, RESEARCH_MODE_ENV)}
    for key in saved:
        os.environ.pop(key, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _active_controller(worker_env: Mapping[str, str], *, recovery: bool = False) -> ToolCallGuardrailController:
    values = {RESEARCH_BUDGET_ENV: worker_env[RESEARCH_BUDGET_ENV]}
    if recovery:
        values[RESEARCH_MODE_ENV] = worker_env[RESEARCH_MODE_ENV]
    with _environment_overlay(values):
        agent = SimpleNamespace()
        _apply_display_config(agent, {"tool_loop_guardrails": {}}, "cron")
    return agent._tool_guardrails


def _unconfigured_controller() -> ToolCallGuardrailController:
    with _environment_overlay({}):
        agent = SimpleNamespace()
        _apply_display_config(agent, {"tool_loop_guardrails": {}}, "cron")
    return agent._tool_guardrails


def _capture_worker_env(task: Any, workspace: Path, *, recovery_mode: str | None = None) -> dict[str, str]:
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = WORKER_PID

    def fake_popen(_cmd: list[str], **kwargs: Any) -> FakeProcess:
        captured.update(kwargs)
        return FakeProcess()

    with patch.object(kbd.subprocess, "Popen", side_effect=fake_popen):
        pid = kbd._default_spawn(
            task,
            str(workspace),
            recovery_mode=recovery_mode,
        )
    if pid != WORKER_PID:
        raise AssertionError(f"fixture worker envelope did not spawn: {pid!r}")
    stream = captured.get("stdout")
    if stream is not None:
        stream.close()
    env = captured.get("env")
    if not isinstance(env, dict):
        raise AssertionError("dispatcher did not provide a worker environment")
    return {
        key: str(value)
        for key, value in env.items()
        if key in {RESEARCH_BUDGET_ENV, RESEARCH_MODE_ENV}
    }


def _create_and_claim_task(root: Path) -> tuple[str, Any, dict[str, Any], Path, dict[str, str]]:
    kb.init_db()
    created = json.loads(
        kanban_tools._handle_create(
            {
                "title": "offline bounded destination research",
                "assignee": "worker",
                "research_budget": TASK_POLICY,
            }
        )
    )
    if "task_id" not in created:
        raise AssertionError(f"native task creation failed: {created}")
    task_id = created["task_id"]
    shown = json.loads(kanban_tools._handle_show({"task_id": task_id}))
    shown_policy = shown["task"]["research_budget"]
    workspace = root / "fixture-worker"
    workspace.mkdir(parents=True, exist_ok=True)
    with kbc.connect_closing() as conn:
        task = kb.claim_task(conn, task_id, claimer="offline-fixture")
    if task is None:
        raise AssertionError("fixture task was not claimable")
    worker_env = _capture_worker_env(task, workspace)
    return task_id, task, shown_policy, workspace, worker_env


def _run_recovery_path(
    root: Path,
    task_id: str,
    workspace: Path,
    task: Any,
    worker_env: Mapping[str, str],
    baseline: ScenarioMetrics,
) -> tuple[ScenarioMetrics, dict[str, Any]]:
    checkpoint = root / "checkpoint.json"
    checkpoint_budget = {
        "enabled": True,
        "evidence_count": baseline.evidence_count,
        "evidence_present": baseline.evidence_count > 0,
    }
    failure_class = classify_failure(
        outcome="timed_out",
        research_budget=checkpoint_budget,
        timed_out=True,
    )
    if failure_class != FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS:
        raise AssertionError(f"unexpected failure classification: {failure_class!r}")
    write_research_checkpoint(
        task_id=task_id,
        research_budget=checkpoint_budget,
        failure_class=failure_class,
        path=checkpoint,
    )
    if not checkpoint_evidence(checkpoint):
        raise AssertionError("fixture checkpoint did not preserve evidence")

    captured: dict[str, Any] = {}

    def spawn(_task: Any, _workspace: str, *, board: str | None = None, recovery_mode: str | None = None) -> None:
        captured.update({"board": board, "recovery_mode": recovery_mode})

    with kbc.connect_closing() as conn:
        kbd._record_task_failure(
            conn,
            task_id,
            "offline fixture timed out before synthesis",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            event_payload_extra={
                "failure_class": failure_class,
                "failure_code": failure_code(failure_class),
                "evidence_present": True,
                "checkpoint_path": str(checkpoint),
                "research_recovery": True,
            },
        )
        with patch.object(kbd, "_profile_exists_fn", lambda: None):
            dispatch_result = kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
        recovered_task = kb.get_task(conn, task_id)
        if recovered_task is None:
            raise AssertionError("recovery task disappeared")
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        run_metadata = json.loads(run["metadata"] or "{}") if run else {}

    if not dispatch_result.spawned or captured.get("recovery_mode") != "synthesis_only":
        raise AssertionError(f"dispatcher did not request synthesis-only recovery: {captured}")
    recovery_env = _capture_worker_env(
        recovered_task,
        workspace,
        recovery_mode=captured["recovery_mode"],
    )
    if recovery_env.get(RESEARCH_MODE_ENV) != "synthesis_only":
        raise AssertionError(f"recovery mode missing from worker envelope: {recovery_env}")

    controller = _active_controller(recovery_env, recovery=True)
    recovery_calls = (FixtureCall(1, "web_search", "weather", 1.5),)
    recovery = _run_script(
        controller,
        recovery_calls,
        name="recovery",
        clock=SimulatedClock(),
        finalize=True,
        result_quality_status="UNCERTAIN",
        preserved_evidence=True,
    )
    dispatch_metadata = {
        "checkpoint_evidence_present": checkpoint_evidence(checkpoint),
        "checkpoint": read_checkpoint(checkpoint),
        "failure_class": failure_class,
        "failure_code": failure_code(failure_class),
        "worker_recovery_mode": captured["recovery_mode"],
        "worker_env_recovery_mode": recovery_env.get(RESEARCH_MODE_ENV),
        "task_status_after_dispatch": recovered_task.status,
        "run_failure_class": run_metadata.get("failure_class"),
        "run_research_recovery": run_metadata.get("research_recovery"),
        "dispatch_spawned": bool(dispatch_result.spawned),
        "active_guardrail_code": recovery.guardrail_code,
        "active_guardrail_policy": _policy_from_controller(controller),
        "task_policy_still_present": worker_env.get(RESEARCH_BUDGET_ENV)
        == recovery_env.get(RESEARCH_BUDGET_ENV),
    }
    # Keep the report free of temp paths and timestamps while retaining the
    # auditable booleans and machine-facing failure fields.
    dispatch_metadata["checkpoint"] = {
        "evidence_present": bool(dispatch_metadata["checkpoint"].get("evidence_present")),
        "evidence_count": int(dispatch_metadata["checkpoint"].get("evidence_count") or 0),
        "research_recovery": bool(dispatch_metadata["checkpoint"].get("research_recovery")),
    }
    return recovery, dispatch_metadata


def _comparison(before: ScenarioMetrics, after: ScenarioMetrics) -> dict[str, Any]:
    def reduction(field: str) -> float:
        left = float(getattr(before, field))
        right = float(getattr(after, field))
        return round(left - right, 3)

    return {
        "logical_runtime_seconds_reduction": reduction("logical_runtime_seconds"),
        "web_search_calls_reduction": reduction("web_search_calls"),
        "browser_extract_calls_reduction": reduction("browser_extract_calls"),
        "retry_calls_reduction": reduction("retry_calls"),
        "before_terminal_outcome": before.terminal_outcome,
        "after_terminal_outcome": after.terminal_outcome,
        "before_result_quality_status": before.result_quality_status,
        "after_result_quality_status": after.result_quality_status,
    }


def run_fixture() -> dict[str, Any]:
    """Run the full offline P0-A/B/C/D path and return a JSON-safe report."""
    with tempfile.TemporaryDirectory(prefix="hermes-research-fixture-") as temp_dir:
        root = Path(temp_dir)
        with _isolated_runtime(root):
            task_id, task, shown_policy, workspace, worker_env = _create_and_claim_task(root)
            if worker_env.get(RESEARCH_BUDGET_ENV) is None:
                raise AssertionError("task policy was not serialized into worker envelope")
            envelope_policy = json.loads(worker_env[RESEARCH_BUDGET_ENV])

            before_controller = _unconfigured_controller()
            before = _run_script(
                before_controller,
                _scripted_calls(),
                name="before_unbounded",
                clock=SimulatedClock(),
                finalize=False,
                result_quality_status="NOT_CHECKED",
                timeout_seconds=BASELINE_TIMEOUT_SECONDS,
            )

            after_controller = _active_controller(worker_env)
            after = _run_script(
                after_controller,
                _scripted_calls(),
                name="after_bounded",
                clock=SimulatedClock(),
                finalize=True,
                result_quality_status="UNCERTAIN",
            )

            search_budget_probe = _run_script(
                _active_controller(worker_env),
                _budget_probe_calls("web_search"),
                name="web_search_budget",
                clock=SimulatedClock(),
                finalize=True,
                result_quality_status="NOT_FOUND_WITHIN_BUDGET",
            )
            extract_budget_probe = _run_script(
                _active_controller(worker_env),
                _budget_probe_calls("web_extract"),
                name="browser_extract_budget",
                clock=SimulatedClock(),
                finalize=True,
                result_quality_status="NOT_FOUND_WITHIN_BUDGET",
            )

            recovery, recovery_dispatch = _run_recovery_path(
                root,
                task_id,
                workspace,
                task,
                worker_env,
                before,
            )
            report = {
                "schema_version": 1,
                "fixture": {
                    "name": "offline_bounded_research_loop",
                    "version": FIXTURE_VERSION,
                    "external_network_calls": 0,
                    "source_mode": "synthetic_fixture_only",
                    "runtime_under_test": [
                        "native Kanban task policy persistence",
                        "dispatcher worker envelope",
                        "agent task override",
                        "ToolCallGuardrailController",
                        "synthesis-only recovery",
                    ],
                },
                "task_policy_path": {
                    "native_create_readback_matches": shown_policy == TASK_POLICY,
                    "worker_envelope_readback_matches": envelope_policy == TASK_POLICY,
                    "active_guardrail_policy": _policy_from_controller(after_controller),
                    "task_policy": TASK_POLICY,
                },
                "budget_probes": {
                    "web_search": search_budget_probe.to_dict(),
                    "browser_extract": extract_budget_probe.to_dict(),
                },
                "quality_statuses": [
                    before.result_quality_status,
                    after.result_quality_status,
                    search_budget_probe.result_quality_status,
                    recovery.result_quality_status,
                ],
                "scenarios": {
                    "before": before.to_dict(),
                    "after": after.to_dict(),
                    "recovery": recovery.to_dict(),
                },
                "comparison": _comparison(before, after),
                "recovery_dispatch": recovery_dispatch,
            }
            # Do not leak the random task id, temporary paths, or wall-clock
            # timestamps into a report intended to be repeatable.
            del task
            return report


def render_markdown(report: Mapping[str, Any]) -> str:
    scenarios = report["scenarios"]
    rows = [
        ("logical runtime (s)", "logical_runtime_seconds"),
        ("web_search calls", "web_search_calls"),
        ("browser/extract calls", "browser_extract_calls"),
        ("retry calls", "retry_calls"),
        ("collection calls blocked", "blocked_collection_calls"),
        ("terminal tool calls", "terminal_tool_calls"),
        ("terminal outcome", "terminal_outcome"),
        ("result quality", "result_quality_status"),
    ]
    lines = [
        "# Offline bounded research runtime report",
        "",
        f"Fixture: `{report['fixture']['name']}` ({report['fixture']['version']})",
        "",
        "The harness uses synthetic `fixture://` evidence only; external network calls: 0.",
        "",
        "## Before / after / recovery metrics",
        "",
        "| Metric | Before | After | Recovery |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in rows:
        values = [scenarios[name][key] for name in ("before", "after", "recovery")]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "## Policy path",
            "",
            f"- Native task readback: `{report['task_policy_path']['native_create_readback_matches']}`",
            f"- Worker envelope readback: `{report['task_policy_path']['worker_envelope_readback_matches']}`",
            f"- Active guardrail policy: `{json.dumps(report['task_policy_path']['active_guardrail_policy'], sort_keys=True)}`",
            "",
            "## Guardrail lifecycle",
            "",
            f"- Bounded lifecycle states: `{scenarios['after']['lifecycle_states']}`",
            f"- Collection disabled after transition: `{scenarios['after']['collection_disabled_after_transition']}`",
            f"- Finalization tools available: `{scenarios['after']['finalization_tools_available']}`",
            "",
            "## Independent collection caps",
            "",
            (
                f"- `web_search`: {report['budget_probes']['web_search']['web_search_calls']} executed, "
                f"cap {report['task_policy_path']['task_policy']['web_search_max']}; "
                f"status `{report['budget_probes']['web_search']['result_quality_status']}`"
            ),
            (
                f"- `web_extract`: {report['budget_probes']['browser_extract']['browser_extract_calls']} executed, "
                f"cap {report['task_policy_path']['task_policy']['browser_extract_max']}; "
                f"status `{report['budget_probes']['browser_extract']['result_quality_status']}`"
            ),
            "",
            "## Recovery path",
            "",
            f"- Failure class: `{report['recovery_dispatch']['failure_class']}`",
            f"- Checkpoint preserved evidence: `{report['recovery_dispatch']['checkpoint_evidence_present']}`",
            f"- Dispatcher recovery mode: `{report['recovery_dispatch']['worker_recovery_mode']}`",
            f"- Recovery worker envelope mode: `{report['recovery_dispatch']['worker_env_recovery_mode']}`",
            f"- Active recovery guardrail: `{report['recovery_dispatch']['active_guardrail_code']}`",
            "",
            "## Event traces",
            "",
        ]
    )
    for name in ("before", "after", "recovery"):
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(scenarios[name]["events"], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], json_path: Path, markdown_path: Path | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--markdown-output", type=Path, help="optional Markdown report path")
    args = parser.parse_args(argv)
    report = run_fixture()
    write_report(report, args.output, args.markdown_output)
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown_output) if args.markdown_output else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
