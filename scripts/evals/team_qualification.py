#!/usr/bin/env python3
"""Opt-in, isolated model-parent qualification of the public team workflow.

The controller resolves explicitly selected provider references only after
``--execute``.  Runtime material crosses to a fresh-Hermes-home child over
stdin and is never written to its config or copied into the allowlisted report.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
_root = str(REPO_ROOT)
sys.path[:] = [_root, *(item for item in sys.path if item != _root)]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/orchestration/team-qualification-v2.json"
UNKNOWN = "unknown"
EXPECTED_TOOLS = {"worker_capabilities", "team_task", "wait_agent", "inspect_agent", "worker_control"}
PARENT_TOOLSETS = ("delegation", "kanban")
TERMINAL = {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"}
SAFE_TOKEN_KEYS = (
    "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens",
    "cache_write_tokens", "reasoning_tokens", "input", "output",
)


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    scenario = fixture.get("scenario") or {}
    if scenario.get("id") != "two_provider_model_parent_team_workflow":
        raise ValueError("Unexpected team qualification scenario")
    assertions = scenario.get("assertions")
    required = fixture.get("required_receipt_fields")
    if not isinstance(assertions, list) or len(assertions) != 8:
        raise ValueError("Team qualification requires exactly eight assertions")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError("required_receipt_fields must be a list of names")
    return fixture


def _number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _safe_tokens(value: Mapping[str, Any] | None) -> dict[str, int | float]:
    source = value if isinstance(value, Mapping) else {}
    return {key: number for key in SAFE_TOKEN_KEYS if (number := _number(source.get(key))) is not None}


def _error_class(error: Any) -> str | None:
    return "SERVICE_ERROR" if error else None


def _identifier(value: Any) -> str:
    return value if isinstance(value, str) and 0 < len(value) <= 512 else UNKNOWN


def _observed_transport_fields(api_kwargs: Any) -> dict[str, str]:
    """Extract model and effort only from the post-build transport request."""
    payload = dict(api_kwargs) if isinstance(api_kwargs, Mapping) else {}
    extra = payload.pop("extra_body", None)
    if isinstance(extra, Mapping):
        payload.update(extra)
    reasoning = payload.get("reasoning")
    output_config = payload.get("output_config")
    effort = (
        reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    ) or payload.get("reasoning_effort") or (
        output_config.get("effort") if isinstance(output_config, Mapping) else None
    )
    return {
        "transmitted_model": _identifier(payload.get("model") or payload.get("modelId")),
        "transmitted_reasoning_effort": _identifier(effort),
    }


def _reported_model(agent: Any, response: Any) -> str:
    marker = response.get("_provider_reported_model") if isinstance(response, Mapping) else getattr(response, "_provider_reported_model", None)
    if getattr(agent, "api_mode", None) in {"codex_responses", "bedrock_converse", "codex_app_server"}:
        return _identifier(marker)
    model = response.get("model") if isinstance(response, Mapping) else getattr(response, "model", None)
    return _identifier(model)


def evaluate_assertions(observation: Mapping[str, Any]) -> list[bool]:
    """Evaluate only durable, externally observable workflow semantics."""
    tools = set(observation.get("advertised_tools") or ())
    actions = list(observation.get("team_actions") or ())
    counts = {name: actions.count(name) for name in set(actions)}
    return [
        EXPECTED_TOOLS.issubset(tools)
        and observation.get("profiles_discovered") is True
        and observation.get("public_tools_only") is True,
        observation.get("task_count") == 2
        and observation.get("dependency_linked") is True
        and observation.get("dependent_started_after_accept") is True,
        observation.get("running_guidance") is True,
        observation.get("initial_implementation_succeeded") is True
        and observation.get("first_review_succeeded") is True
        and observation.get("first_review_mentions_violet") is True,
        counts.get("request_changes", 0) == 1
        and observation.get("correction_reused_worker") is True
        and observation.get("correction_linked_previous_run") is True
        and observation.get("correction_retained_markers") is True,
        counts.get("submit_review", 0) == 2
        and observation.get("review_success_count") == 2
        and counts.get("accept", 0) == 1
        and observation.get("implementation_task_status") == "done",
        observation.get("dependent_worker_succeeded") is True,
        observation.get("route_agreement") is True
        and observation.get("worker_tools_empty") is True
        and observation.get("duplicate_owned_executions") == 0
        and observation.get("authorization_violations") == 0
        and observation.get("structured_tool_errors") == 0
        and observation.get("all_terminal_acked") is True,
    ]


def build_report(
    fixture: Mapping[str, Any], observation: Mapping[str, Any], common: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an allowlisted receipt; unknown observation keys are discarded."""
    checks = evaluate_assertions(observation)
    assertions = [
        {"id": f"TQ-{index}", "assertion": text, "passed": passed}
        for index, (text, passed) in enumerate(
            zip(fixture["scenario"]["assertions"], checks), start=1
        )
    ]
    tool_events = list(observation.get("tool_calls") or ())
    tool_counts: dict[str, int] = {}
    for event in tool_events:
        key = str(event.get("advertised_tool") or UNKNOWN)
        if event.get("team_action"):
            key = f"{key}:{event['team_action']}"
        tool_counts[key] = tool_counts.get(key, 0) + 1
    report = {
        "suite": fixture["version"],
        "claim_class": fixture["claim_class"],
        "candidate_sha": common.get("candidate_sha", UNKNOWN),
        "scenario_id": fixture["scenario"]["id"],
        "interface": {
            "requested": common.get("requested_interface", UNKNOWN),
            "selected": observation.get("selected_interface", UNKNOWN),
            "version": observation.get("interface_version", UNKNOWN),
            "advertised_tools": sorted(set(observation.get("advertised_tools") or ())),
        },
        "team_contract": observation.get("team_contract", UNKNOWN),
        "parent_route": dict(observation.get("parent_route") or {}),
        "worker_routes": list(observation.get("worker_routes") or ()),
        "tool_calls": {
            "counts": tool_counts,
            "error_count": sum(not event.get("accepted") for event in tool_events),
            "events": tool_events,
        },
        "task_lineage": list(observation.get("task_lineage") or ()),
        "worker_lineage": list(observation.get("worker_lineage") or ()),
        "elapsed_seconds": round(float(observation.get("elapsed_seconds") or 0), 3),
        "token_usage": _safe_tokens(observation.get("token_usage")),
        "known_or_unknown_cost": dict(observation.get("cost") or {"status": UNKNOWN}),
        "authorization_violations": int(observation.get("authorization_violations") or 0),
        "duplicate_owned_executions": int(observation.get("duplicate_owned_executions") or 0),
        "limits": dict(common.get("limits") or {}),
        "assertions": assertions,
        "scenario_passed": all(checks),
        "error_type": observation.get("error_type"),
        "proof_boundary": (
            "One opt-in synthetic run on this committed candidate and these selected routes only. "
            "Provider-reported identity is not independent model verification; this is not release, "
            "fleet, customer, or statistical reliability proof."
        ),
    }
    missing = [name for name in fixture["required_receipt_fields"] if name not in report]
    if missing:
        raise ValueError(f"Report omitted required fields: {missing}")
    return report


def _profiles_config(packet: Mapping[str, Any]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for index, route in enumerate(packet["routes"]):
        if index == 0:
            instructions = (
                "Perform only the supplied synthetic implementation or correction. Keep answers concise. "
                "Retain prior synthetic labels across linked runs and use no tools."
            )
        else:
            instructions = (
                "Perform only the supplied synthetic review or dependent task and use no tools. During review, "
                "assess the supplied implementation handoff, name the labels actually present, and recommend "
                "one correction when AMBER is absent. Do not invent evidence."
            )
        profiles[route["profile"]] = {
            "description": "Synthetic team qualification role.",
            "instructions": instructions,
            "provider": route["provider"],
            "model": route["model"],
            "reasoning_effort": route["reasoning_effort"],
            "tool_policy": {"allowed_tools": [], "allowed_mcp_tools": []},
            "workspace_context": {"mode": "inherit", "include_memory": False, "include_context_files": False},
            "execution_limits": {
                "max_iterations": packet["limits"]["max_child_iterations"],
                "timeout_seconds": packet["limits"]["max_child_seconds"],
                "max_spawn_depth": 0,
            },
        }
    return profiles


def _build_config(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": {"provider": packet["parent_provider"], "default": packet["parent_model"]},
        "toolsets": list(PARENT_TOOLSETS),
        "orchestration": {"interface": "codex"},
        "delegation": {
            "profiles": _profiles_config(packet),
            "routing_mode": "profile_only",
            "max_concurrent_children": packet["limits"]["max_children"],
            "max_iterations": packet["limits"]["max_child_iterations"],
        },
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
    }


def _tool_instrument(parent: Any, guidance_gate: threading.Event) -> tuple[list[dict[str, Any]], Any]:
    original = parent._dispatch_worker_interface
    events: list[dict[str, Any]] = []

    def dispatch(name: str, arguments: Mapping[str, Any]) -> str:
        args = dict(arguments or {})
        raw = original(name, args)
        try:
            payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError):
            payload = {"error": "unstructured"}
        interface = payload.get("orchestration_interface") or {}
        outcomes = payload.get("outcomes") if isinstance(payload.get("outcomes"), list) else []
        outcome_errors = [item for item in outcomes if isinstance(item, Mapping) and item.get("error")]
        event = {
            "sequence": len(events) + 1,
            "advertised_tool": name,
            "canonical_tool": interface.get("canonical_tool", UNKNOWN),
            "operation": interface.get("operation", UNKNOWN),
            "effective_action": interface.get("effective_action", UNKNOWN),
            "team_action": payload.get("action") if name == "team_task" else None,
            "accepted": not bool(payload.get("error") or outcome_errors),
            "error_classification": _error_class(payload.get("error") or outcome_errors),
            "task_ref": payload.get("task_ref") or args.get("task_ref"),
            "worker_ref": payload.get("worker_ref"),
            "run_ref": payload.get("run_ref") or payload.get("run_id"),
            "kanban_run_id": payload.get("kanban_run_id"),
            "status": payload.get("status"),
            "worker_status": payload.get("worker_status") or payload.get("status"),
            "delivery": next(
                (item.get("delivery") for item in outcomes if isinstance(item, Mapping) and item.get("delivery")),
                payload.get("delivery"),
            ),
            "target_count": len(outcomes) or None,
        }
        events.append(event)
        if (
            name == "team_task" and event["team_action"] == "guide" and event["accepted"]
            and event["delivery"] == "RUNNING_STEER_PENDING_CHECKPOINT"
        ):
            guidance_gate.set()
        return raw

    parent._dispatch_worker_interface = dispatch
    return events, original


def _route_receipt(run: Mapping[str, Any], expected: Mapping[str, Any] | None) -> tuple[dict[str, Any], bool, bool]:
    result = run.get("result") if isinstance(run.get("result"), Mapping) else {}
    route = result.get("route") if isinstance(result.get("route"), Mapping) else {}
    item = {
        "requested_profile": route.get("requested_profile", UNKNOWN),
        "requested_provider": route.get("requested_provider", UNKNOWN),
        "requested_model": route.get("requested_model", UNKNOWN),
        "requested_reasoning_effort": route.get("requested_reasoning_effort", UNKNOWN),
        "resolved_provider": route.get("resolved_provider", UNKNOWN),
        "resolved_model": route.get("resolved_model", UNKNOWN),
        "resolved_reasoning_effort": route.get("resolved_reasoning_effort", UNKNOWN),
        "transmitted_provider": route.get("transmitted_provider", UNKNOWN),
        "transmitted_model": route.get("transmitted_model", UNKNOWN),
        "transmitted_reasoning_effort": route.get("transmitted_reasoning_effort", UNKNOWN),
        "provider_reported_model": route.get("provider_reported_model") or UNKNOWN,
        "request_evidence_source": route.get("request_evidence_source", UNKNOWN),
        "response_evidence_source": route.get("response_evidence_source", UNKNOWN),
    }
    expected_provider = (expected or {}).get("expected_provider") or (expected or {}).get("provider")
    match = bool(expected) and all((
        item["requested_profile"] == expected["profile"],
        item["resolved_provider"] == expected_provider,
        item["resolved_model"] == expected["model"],
        item["resolved_reasoning_effort"] == expected["reasoning_effort"],
        item["transmitted_provider"] == expected_provider,
        item["transmitted_model"] == expected["model"],
        item["transmitted_reasoning_effort"] == expected["reasoning_effort"],
        item["request_evidence_source"] != UNKNOWN,
        item["response_evidence_source"] != UNKNOWN,
    ))
    return item, match, result.get("effective_tools") == []


def _observations(parent: Any, store: Any, owner: str, events: list[dict[str, Any]], packet: Mapping[str, Any], elapsed: float, run_error: Exception | None) -> dict[str, Any]:
    from agent.team_orchestration import TEAM_CONTRACT_VERSION
    from agent.worker_interfaces import advertised_worker_tool_names
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    expected = {route["profile"]: route for route in packet["routes"]}
    workers = store.list_workers(owner)
    worker_lineage: list[dict[str, Any]] = []
    run_index: dict[str, Mapping[str, Any]] = {}
    route_items: list[dict[str, Any]] = []
    route_matches: list[bool] = []
    tools_empty: list[bool] = []
    tokens: dict[str, int | float] = {}
    costs: list[float] = []
    cost_unknown = False
    run_ids: list[str] = []
    marker_text: dict[str, str] = {}
    all_acked: list[bool] = []
    for worker in workers:
        profile = worker.get("profile")
        runs = store.list_runs(worker["worker_id"], owner)
        safe_runs = []
        for run in runs:
            run_ids.append(str(run.get("run_id")))
            run_index[str(run.get("run_id"))] = run
            route, matches, empty = _route_receipt(run, expected.get(profile))
            route_items.append({"worker_ref": f"worker:{worker['worker_id']}", "run_ref": f"run:{run['run_id']}", **route})
            route_matches.append(matches)
            tools_empty.append(empty)
            result = run.get("result") if isinstance(run.get("result"), Mapping) else {}
            marker_text[str(run.get("run_id"))] = str(result.get("summary") or "").lower()
            usage = _safe_tokens((result.get("usage") or {}).get("tokens"))
            for key, value in usage.items():
                tokens[key] = tokens.get(key, 0) + value
            cost = _number((result.get("cost") or {}).get("usd"))
            run_cost_status = str((result.get("cost") or {}).get("status") or UNKNOWN)
            if cost is None:
                cost_unknown = True
            else:
                costs.append(float(cost))
            all_acked.append(run.get("status") in TERMINAL and bool(run.get("completion_ack")))
            created = _number(run.get("created_at"))
            updated = _number(run.get("updated_at"))
            safe_runs.append({
                "run_ref": f"run:{run.get('run_id')}",
                "status": run.get("status", UNKNOWN),
                "previous_run_ref": f"run:{run['previous_run_id']}" if run.get("previous_run_id") else None,
                "completion_ack": bool(run.get("completion_ack")),
                "uncertain_side_effect": bool(run.get("uncertain_side_effect")),
                "latency_seconds": round(float(updated - created), 3) if created is not None and updated is not None else UNKNOWN,
                "token_usage": usage,
                "known_or_unknown_cost": {
                    "status": run_cost_status,
                    "estimated_usd": round(float(cost), 8) if cost is not None else UNKNOWN,
                },
                "effective_tool_count": len(result.get("effective_tools")) if isinstance(result.get("effective_tools"), list) else UNKNOWN,
            })
        worker_lineage.append({
            "worker_ref": f"worker:{worker['worker_id']}",
            "profile": profile,
            "runs": safe_runs,
        })

    with kbc.connect() as conn:
        tasks = kb.list_tasks(conn, session_id=owner, include_archived=True, order_by="created")
        task_lineage = []
        attachments: list[dict[str, Any]] = []
        for task in tasks:
            task_events = kb.list_events(conn, task.id)
            task_runs = kb.list_runs(conn, task.id)
            safe_events = []
            for event in task_events:
                payload = event.payload or {}
                safe = {"event_id": event.id, "kind": event.kind, "kanban_run_id": event.run_id}
                if event.kind == "execution_attached":
                    safe.update({key: payload.get(key) for key in ("role", "profile", "worker_ref", "run_ref")})
                    attachments.append({"task_ref": f"task:{task.id}", "kanban_run_id": event.run_id, **payload})
                safe_events.append(safe)
            task_lineage.append({
                "task_ref": f"task:{task.id}",
                "status": task.status,
                "assignee": task.assignee,
                "execution_mode": task.execution_mode,
                "parent_refs": [f"task:{row['parent_id']}" for row in conn.execute(
                    "SELECT parent_id FROM task_links WHERE child_id=? ORDER BY parent_id", (task.id,)
                ).fetchall()],
                "current_run_id": task.current_run_id,
                "runs": [
                    {"kanban_run_id": run.id, "status": run.status, "outcome": run.outcome, "step_key": run.step_key}
                    for run in task_runs
                ],
                "events": safe_events,
            })

    by_role: dict[str, list[dict[str, Any]]] = {}
    for attachment in attachments:
        by_role.setdefault(str(attachment.get("role")), []).append(attachment)
    implementers = by_role.get("implementer", [])
    corrections = by_role.get("correction", [])
    reviewers = by_role.get("reviewer", [])
    dependent_task = next((task for task in task_lineage if task["parent_refs"]), None)
    implementation_task = next((task for task in task_lineage if not task["parent_refs"]), None)
    accept_sequence = next((e["sequence"] for e in events if e.get("team_action") == "accept" and e.get("accepted")), None)
    dependent_ref = dependent_task.get("task_ref") if dependent_task else None
    dependent_start = next((e["sequence"] for e in events if e.get("team_action") == "start" and e.get("task_ref") == dependent_ref and e.get("accepted")), None)
    initial_run = implementers[0].get("run_ref", "").partition(":")[2] if implementers else ""
    correction_run = corrections[0].get("run_ref", "").partition(":")[2] if corrections else ""
    review_runs = [item.get("run_ref", "").partition(":")[2] for item in reviewers]
    correction_record = run_index.get(correction_run, {})
    initial_record = run_index.get(initial_run, {})
    dependent_attachment = next((item for item in attachments if item.get("task_ref") == dependent_ref and item.get("role") == "implementer"), None)
    dependent_run = (dependent_attachment or {}).get("run_ref", "").partition(":")[2]
    parent_tokens = _safe_tokens({key: getattr(parent, f"session_{key}", None) for key in SAFE_TOKEN_KEYS})
    for key, value in parent_tokens.items():
        tokens[key] = tokens.get(key, 0) + value
    parent_cost = _number(getattr(parent, "session_estimated_cost_usd", None))
    parent_cost_status = str(getattr(parent, "session_cost_status", UNKNOWN) or UNKNOWN)
    if parent_cost is not None and parent_cost_status != UNKNOWN:
        costs.append(float(parent_cost))
    else:
        cost_unknown = True
    selection = parent._worker_interface_selection
    advertised = sorted(advertised_worker_tool_names(selection) & set(parent.valid_tool_names))
    team_actions = [str(e["team_action"]) for e in events if e.get("team_action") and e.get("accepted")]
    canonical_forbidden = {"kanban_team", "delegate_task", "kanban_create", "kanban_complete", "kanban_request_review", "kanban_request_changes"}
    profiles_event = next((e for e in events if e.get("advertised_tool") == "worker_capabilities" and e.get("accepted")), None)
    duplicate_attachments = len({(a.get("task_ref"), a.get("kanban_run_id"), a.get("run_ref")) for a in attachments}) != len(attachments)
    return {
        "selected_interface": selection.name,
        "interface_version": selection.version,
        "team_contract": TEAM_CONTRACT_VERSION,
        "advertised_tools": advertised,
        "profiles_discovered": bool(
            profiles_event and set(expected).issubset({worker.get("profile") for worker in workers})
        ),
        "public_tools_only": not any(e.get("advertised_tool") in canonical_forbidden for e in events),
        "team_actions": team_actions,
        "task_count": len(task_lineage),
        "dependency_linked": bool(dependent_task and len(dependent_task["parent_refs"]) == 1),
        "dependent_started_after_accept": bool(accept_sequence and dependent_start and dependent_start > accept_sequence),
        "running_guidance": any(e.get("team_action") == "guide" and e.get("accepted") and e.get("delivery") == "RUNNING_STEER_PENDING_CHECKPOINT" for e in events),
        "initial_implementation_succeeded": initial_record.get("status") == "SUCCEEDED",
        "first_review_succeeded": bool(review_runs and run_index.get(review_runs[0], {}).get("status") == "SUCCEEDED"),
        "first_review_mentions_violet": bool(review_runs and "violet" in marker_text.get(review_runs[0], "")),
        "correction_reused_worker": bool(implementers and corrections and implementers[0].get("worker_ref") == corrections[0].get("worker_ref")),
        "correction_linked_previous_run": bool(correction_record and correction_record.get("previous_run_id") == initial_run),
        "correction_retained_markers": all(label in marker_text.get(correction_run, "") for label in ("cobalt", "violet", "amber")),
        "review_success_count": sum(run_index.get(run_id, {}).get("status") == "SUCCEEDED" for run_id in review_runs),
        "implementation_task_status": (implementation_task or {}).get("status", UNKNOWN),
        "dependent_worker_succeeded": run_index.get(dependent_run, {}).get("status") == "SUCCEEDED",
        "route_agreement": bool(route_matches) and all(route_matches),
        "worker_tools_empty": bool(tools_empty) and all(tools_empty),
        "structured_tool_errors": sum(not e.get("accepted") for e in events),
        "all_terminal_acked": bool(all_acked) and all(all_acked),
        "authorization_violations": 0,
        "duplicate_owned_executions": len(run_ids) - len(set(run_ids)) + int(duplicate_attachments),
        "parent_route": dict(packet.get("parent_route_observed") or {}),
        "worker_routes": route_items,
        "tool_calls": events,
        "task_lineage": task_lineage,
        "worker_lineage": worker_lineage,
        "elapsed_seconds": elapsed,
        "token_usage": tokens,
        "cost": {
            "status": "partial" if cost_unknown and costs else ("unknown" if cost_unknown else "known"),
            "estimated_known_usd": round(sum(costs), 8) if costs else UNKNOWN,
        },
        "error_type": type(run_error).__name__ if run_error else None,
    }


def _execute_child(packet: dict[str, Any]) -> dict[str, Any]:
    import yaml

    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(yaml.safe_dump(_build_config(packet)), encoding="utf-8")
    credentials = packet["credentials"]
    from hermes_cli import runtime_provider
    original_resolver = runtime_provider.resolve_runtime_provider

    def resolve(*, requested: str | None = None, target_model: str | None = None, **_kwargs: Any) -> dict[str, Any]:
        if requested not in credentials:
            raise ValueError("Qualification child only enables explicitly selected providers")
        material = credentials[requested]
        return original_resolver(
            requested=requested, target_model=target_model,
            explicit_api_key=material.get("api_key"), explicit_base_url=material.get("base_url"),
        )

    runtime_provider.resolve_runtime_provider = resolve
    from agent.worker_store import WorkerStore
    from hermes_state import SessionDB
    from run_agent import AIAgent

    parent_runtime = resolve(requested=packet["parent_provider"], target_model=packet["parent_model"])
    db = SessionDB()
    owner = "team-qualification-model-parent"
    db.ensure_session(owner, source="team-qualification")
    parent = AIAgent(
        **{key: parent_runtime[key] for key in ("provider", "api_key", "base_url", "api_mode")},
        model=packet["parent_model"], session_id=owner, session_db=db,
        enabled_toolsets=list(PARENT_TOOLSETS), max_iterations=packet["limits"]["max_parent_iterations"],
        max_tokens=packet["limits"]["max_parent_tokens"],
        reasoning_config={"effort": packet["parent_reasoning_effort"]},
        run_budget_seconds=packet["limits"]["max_seconds"], skip_memory=True,
        skip_context_files=True, skip_background_review=True, save_trajectories=False, quiet_mode=True,
    )
    store = WorkerStore(db)
    store.ensure_schema()
    gate = threading.Event()
    events, original_dispatch = _tool_instrument(parent, gate)
    original_http = AIAgent._interruptible_api_call
    held_workers: set[str] = set()
    route_lock = threading.Lock()

    def instrumented_http(agent: Any, *args: Any, **kwargs: Any) -> Any:
        worker_id = getattr(agent, "_worker_id", None)
        if worker_id:
            with route_lock:
                should_hold = worker_id not in held_workers
                held_workers.add(worker_id)
            if should_hold and not gate.is_set() and not gate.wait(min(60, packet["limits"]["max_child_seconds"])):
                raise TimeoutError("Controlled RUNNING-guidance barrier expired")
        request = args[0] if args else kwargs.get("api_kwargs")
        if agent is parent:
            packet["parent_route_observed"] = {
                "requested_provider": packet["parent_provider"],
                "requested_model": packet["parent_model"],
                "requested_reasoning_effort": packet["parent_reasoning_effort"],
                "resolved_provider": parent_runtime.get("provider", UNKNOWN),
                "resolved_model": packet["parent_model"],
                "resolved_reasoning_effort": packet["parent_reasoning_effort"],
                "transmitted_provider": getattr(agent, "provider", UNKNOWN),
                **_observed_transport_fields(request),
                "provider_reported_model": UNKNOWN,
            }
        response = original_http(agent, *args, **kwargs)
        if agent is parent:
            packet["parent_route_observed"]["provider_reported_model"] = _reported_model(agent, response)
        return response

    AIAgent._interruptible_api_call = instrumented_http
    prompt = packet["scenario"]["prompt_template"].format(
        worker_a_profile=packet["routes"][0]["profile"], worker_b_profile=packet["routes"][1]["profile"],
    )
    started = time.monotonic()
    run_error: Exception | None = None
    try:
        parent.run_conversation(prompt)
    except Exception as exc:
        run_error = exc
    finally:
        gate.set()
        AIAgent._interruptible_api_call = original_http
        parent._dispatch_worker_interface = original_dispatch
    elapsed = time.monotonic() - started
    try:
        result = _observations(parent, store, owner, events, packet, elapsed, run_error)
    finally:
        parent.close()
        db.close()
        runtime_provider.resolve_runtime_provider = original_resolver
    return result


def _child_failure(packet: Mapping[str, Any], error_type: str) -> dict[str, Any]:
    return {
        "selected_interface": UNKNOWN, "interface_version": UNKNOWN, "team_contract": UNKNOWN,
        "advertised_tools": [], "parent_route": {}, "worker_routes": [], "tool_calls": [],
        "task_lineage": [], "worker_lineage": [], "elapsed_seconds": 0, "token_usage": {},
        "cost": {"status": UNKNOWN}, "authorization_violations": 0,
        "duplicate_owned_executions": 0, "error_type": error_type,
    }


def _run_child(packet: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hermes-team-qualification-") as home:
        env = dict(os.environ, HERMES_HOME=home)
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child"], input=json.dumps(packet),
                text=True, capture_output=True, env=env, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return _child_failure(packet, "ScenarioTimeout")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError):
        return _child_failure(packet, "UnstructuredChildOutput")
    return payload if isinstance(payload, dict) else _child_failure(packet, "InvalidChildOutput")


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Allow the selected live provider calls.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--parent-provider", required="--child" not in sys.argv)
    parser.add_argument("--parent-model", required="--child" not in sys.argv)
    parser.add_argument("--parent-reasoning-effort", required="--child" not in sys.argv)
    for label in ("a", "b"):
        parser.add_argument(f"--worker-{label}-profile", required="--child" not in sys.argv)
        parser.add_argument(f"--worker-{label}-provider", required="--child" not in sys.argv)
        parser.add_argument(f"--worker-{label}-model", required="--child" not in sys.argv)
        parser.add_argument(f"--worker-{label}-reasoning-effort", required="--child" not in sys.argv)
    parser.add_argument("--max-parent-tokens", type=_positive_int, default=4096)
    return parser


def _main() -> None:
    args = _parser().parse_args()
    logging.disable(logging.CRITICAL)
    if args.child:
        packet = json.loads(sys.stdin.read())
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = _execute_child(packet)
        except Exception as exc:
            site = traceback.extract_tb(exc.__traceback__)[-1]
            result = _child_failure(packet, f"{type(exc).__name__}@{Path(site.filename).name}:{site.lineno}")
        print(json.dumps(result, sort_keys=True))
        return
    if not args.execute:
        raise SystemExit("Live provider calls require the explicit --execute flag.")
    if args.max_parent_tokens > 8192:
        raise SystemExit("--max-parent-tokens must be at most 8192")
    from scripts.evals.interface_qualification import RouteChoice, _resolve_credentials, _source_identity

    fixture = load_fixture(args.fixture)
    policy = fixture["execution_policy"]
    routes = [
        RouteChoice(args.worker_a_profile, args.worker_a_provider, args.worker_a_model, args.worker_a_reasoning_effort),
        RouteChoice(args.worker_b_profile, args.worker_b_provider, args.worker_b_model, args.worker_b_reasoning_effort),
    ]
    if len({route.profile for route in routes}) != 2 or len({route.provider for route in routes}) != 2:
        raise SystemExit("Worker profiles and providers must each be distinct")
    candidate = _source_identity(REPO_ROOT)
    credentials, resolved = _resolve_credentials([
        (args.parent_provider, args.parent_model), *((route.provider, route.model) for route in routes),
    ])
    route_payloads = [
        {**asdict(route), "expected_provider": resolved[route.provider]} for route in routes
    ]
    limits = {key: policy[key] for key in (
        "max_parent_iterations", "max_child_iterations", "max_children", "max_seconds", "max_child_seconds",
    )}
    limits["max_parent_tokens"] = args.max_parent_tokens
    packet = {
        "scenario": fixture["scenario"], "credentials": credentials, "routes": route_payloads,
        "parent_provider": args.parent_provider, "parent_model": args.parent_model,
        "parent_reasoning_effort": args.parent_reasoning_effort, "limits": limits,
    }
    observation = _run_child(packet, timeout=limits["max_seconds"] + 15)
    report = build_report(fixture, observation, {
        "candidate_sha": candidate, "requested_interface": "codex", "limits": limits,
    })
    print(json.dumps(report, sort_keys=True))
    if not report["scenario_passed"]:
        raise SystemExit(1)


def main() -> int:
    try:
        _main()
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({
            "scenario_passed": False, "qualification": "unqualified",
            "error_type": type(exc).__name__,
            "proof_boundary": "Controller preflight failed before model-parent team qualification completed.",
        }, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
