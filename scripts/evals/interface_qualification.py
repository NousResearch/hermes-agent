#!/usr/bin/env python3
"""Opt-in live qualification for Hermes durable-worker presentation interfaces.

The controller resolves explicitly selected, existing provider references only
after ``--execute`` and passes the resulting runtime material to one isolated
child per scenario over stdin.  Children use fresh Hermes homes.  Output is an
allowlisted receipt: provider bodies, prompts, transcripts, and credentials are
never included.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
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
from typing import Any, Callable, Iterable, Mapping


SCENARIO_IDS = (
    "two_provider_retained_workflow",
    "invalid_profile_rejected",
    "foreign_worker_denied",
    "idle_guidance_then_followup",
)
UNKNOWN = "unknown"
REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_text = str(REPO_ROOT)
sys.path[:] = [_repo_root_text, *(item for item in sys.path if item != _repo_root_text)]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/orchestration/interface-qualification-v1.json"
SAFE_RUNTIME_KEYS = ("api_key", "base_url")
SAFE_USAGE_KEYS = (
    "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens",
    "cache_write_tokens", "reasoning_tokens",
)
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"}
GUIDANCE_DELIVERIES = {"NEXT_RUN", "RUNNING_STEER_PENDING_CHECKPOINT"}


@dataclass(frozen=True)
class RouteChoice:
    profile: str
    provider: str
    model: str
    reasoning_effort: str
    expected_provider: str | None = None


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    found = tuple(item.get("id") for item in fixture.get("scenarios", ()))
    if found != SCENARIO_IDS:
        raise ValueError(f"Expected the four {SCENARIO_IDS!r} scenarios in order; found {found!r}")
    required = fixture.get("required_receipt_fields")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError("Fixture required_receipt_fields must be a list of names")
    return fixture


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _safe_usage(value: Mapping[str, Any] | None) -> dict[str, int | float]:
    value = value if isinstance(value, Mapping) else {}
    return {key: number for key in SAFE_USAGE_KEYS if (number := _number(value.get(key))) is not None}


def _error_classification(error: Any) -> str | None:
    if not error:
        return None
    text = str(error).lower()
    if "profile" in text and any(word in text for word in ("unknown", "missing", "unavailable", "configured")):
        return "EXPECTED_INVALID_PROFILE_DENIAL"
    if any(phrase in text for phrase in (
        "foreign owner", "owned subtree", "authorized relation", "unknown worker", "outside the actor",
    )):
        return "EXPECTED_FOREIGN_OWNER_DENIAL"
    if any(phrase in text for phrase in ("unsupported", "malformed", "not part of the")):
        return "INVALID_TOOL_CALL"
    return "SERVICE_ERROR"


def _is_running_guidance(event: Mapping[str, Any]) -> bool:
    return (
        event.get("accepted") is True
        and event.get("effective_action") == "message"
        and event.get("selected_run_status") == "RUNNING"
        and event.get("delivery") == "RUNNING_STEER_PENDING_CHECKPOINT"
    )


def _scenario_checks(scenario_id: str, observation: Mapping[str, Any]) -> list[bool]:
    events = list(observation.get("events") or ())
    workers = list(observation.get("workers") or ())
    accepted = [item for item in events if item.get("accepted")]
    expected_denials = [item for item in events if str(item.get("classification") or "").startswith("EXPECTED_")]
    invalid = [item for item in events if item.get("classification") == "INVALID_TOOL_CALL"]
    if scenario_id == "two_provider_retained_workflow":
        profiles = {profile for item in accepted for profile in item.get("profiles", ())}
        providers = {item.get("provider") for item in workers if item.get("provider")}
        message_live = any(_is_running_guidance(item) for item in accepted)
        linked = any(item.get("linked_followup") and item.get("followup_avoids_labels") for item in accepted)
        all_acked = bool(workers) and all(item.get("all_terminal_acked") for item in workers)
        return [
            len(profiles) >= 2 and len(providers) >= 2,
            message_live,
            linked,
            bool(observation.get("retained_answer_match")),
            observation.get("wire_route_agreement") is True and observation.get("tool_receipts_match") is True,
            len(workers) == 2 and sum(item.get("run_count", 0) for item in workers) == 3
            and all_acked and observation.get("duplicate_owned_executions") == 0,
        ]
    if scenario_id == "invalid_profile_rejected":
        submitted = bool(events)
        visible_denial = any(item.get("classification") == "EXPECTED_INVALID_PROFILE_DENIAL" for item in expected_denials)
        return [submitted, visible_denial and not invalid, observation.get("counts_before") == observation.get("counts_after")]
    if scenario_id == "foreign_worker_denied":
        denied = any(item.get("classification") == "EXPECTED_FOREIGN_OWNER_DENIAL" for item in expected_denials)
        return [denied, observation.get("foreign_preimage_unchanged") is True, len(events) == 1 and not accepted]
    if scenario_id == "idle_guidance_then_followup":
        guidance = any(
            item.get("effective_action") == "message" and item.get("run_count_delta") == 0
            for item in accepted
        )
        linked = any(item.get("linked_followup") and item.get("run_count_delta") == 1 for item in accepted)
        new_ack = any(item.get("effective_action") == "ack" and item.get("targets_latest_run") for item in accepted)
        return [
            guidance,
            linked,
            bool(observation.get("retained_answer_match")),
            new_ack and observation.get("prior_run_replayed") is False,
        ]
    raise ValueError(f"Unknown scenario {scenario_id!r}")


def build_receipt(
    scenario: Mapping[str, Any], observation: Mapping[str, Any], common: Mapping[str, Any],
) -> dict[str, Any]:
    """Build only allowlisted evidence; unknown observation keys are discarded."""
    scenario_id = str(scenario["id"])
    checks = _scenario_checks(scenario_id, observation)
    assertion_text = list(scenario.get("assertions") or ())
    if len(checks) != len(assertion_text):
        raise ValueError(f"Assertion evaluator drift for {scenario_id}")
    events = list(observation.get("events") or ())
    invalid_count = sum(item.get("classification") == "INVALID_TOOL_CALL" for item in events)
    receipt = {
        "candidate_sha": common.get("candidate_sha", UNKNOWN),
        "scenario_id": scenario_id,
        "parent_provider": observation.get("parent_provider", common.get("parent_provider", UNKNOWN)),
        "parent_model": observation.get("parent_model", common.get("parent_model", UNKNOWN)),
        "requested_interface": common.get("requested_interface", UNKNOWN),
        "selected_interface": observation.get("selected_interface", UNKNOWN),
        "interface_version": observation.get("interface_version", UNKNOWN),
        "advertised_orchestration_tool_names": sorted(set(observation.get("advertised_tool_names") or ())),
        "observed_orchestration_tool_names": sorted({str(item.get("tool_name")) for item in events if item.get("tool_name")}),
        "guidance_delivery_evidence": [
            {
                "accepted": item.get("accepted") is True,
                "selected_run_status": item.get("selected_run_status", UNKNOWN),
                "delivery": item.get("delivery", UNKNOWN),
            }
            for item in events if item.get("effective_action") == "message"
        ],
        "task_completed": bool(observation.get("parent_completed")) and all(checks),
        "invalid_tool_calls": {
            "structured": invalid_count,
            "malformed_or_unobserved": observation.get("malformed_tool_calls", UNKNOWN),
        },
        "corrective_turns": observation.get("corrective_turns", UNKNOWN),
        "elapsed_seconds": round(float(observation.get("elapsed_seconds") or 0.0), 3),
        "token_usage": _safe_usage(observation.get("token_usage")),
        "known_or_unknown_cost": observation.get("cost", {"status": UNKNOWN}),
        "wire_route_agreement": observation.get("wire_route_agreement", UNKNOWN),
        "authorization_violations": int(observation.get("authorization_violations") or 0),
        "duplicate_owned_executions": int(observation.get("duplicate_owned_executions") or 0),
        "assertions": [
            {"assertion": text, "passed": passed}
            for text, passed in zip(assertion_text, checks)
        ],
        "limits": dict(common.get("limits") or {}),
        "error_type": observation.get("error_type"),
    }
    missing = [name for name in common["required_receipt_fields"] if name not in receipt]
    if missing:
        raise ValueError(f"Receipt omitted required fields: {missing}")
    return receipt


def run_suite(
    fixture: Mapping[str, Any],
    execute_scenario: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    common: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = [build_receipt(scenario, execute_scenario(scenario), common) for scenario in fixture["scenarios"]]
    qualified = len(receipts) == len(SCENARIO_IDS) and all(item["task_completed"] for item in receipts)
    return {
        "suite": fixture["version"],
        "claim_class": fixture["claim_class"],
        "candidate_sha": common["candidate_sha"],
        "qualification": "qualified" if qualified else "unqualified",
        "qualified": qualified,
        "receipts": receipts,
        "proof_boundary": (
            "Exact candidate and selected routes for these four scenarios only; no statistical superiority, "
            "release, runtime, fleet, or customer-readiness claim."
        ),
    }


def _counts(store: Any, owner: str) -> dict[str, int]:
    workers = store.list_workers(owner)
    return {
        "workers": len(workers),
        "runs": sum(len(store.list_runs(item["worker_id"], owner)) for item in workers),
        "messages": sum(len(store.list_messages(item["worker_id"], owner)) for item in workers),
    }


def _stable_foreign_snapshot(store: Any, owner: str, worker_id: str) -> dict[str, Any]:
    worker = store.get_worker(worker_id, owner)
    return {
        "worker": {key: worker.get(key) for key in (
            "worker_id", "owner_session_id", "profile", "config_revision", "policy",
            "history", "uncertain_side_effect",
        )},
        "runs": [
            {key: run.get(key) for key in (
                "run_id", "worker_id", "previous_run_id", "goal", "context", "status",
                "result", "completion_ack", "uncertain_side_effect",
            )}
            for run in store.list_runs(worker_id, owner)
        ],
        "messages": store.list_messages(worker_id, owner),
    }


def _target_from_args(arguments: Mapping[str, Any]) -> str | None:
    for key in ("worker_id", "target", "recipient", "task_id"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _profiles_from_args(arguments: Mapping[str, Any]) -> list[str]:
    result = []
    for item in arguments.get("tasks") or ():
        if isinstance(item, Mapping) and isinstance(item.get("profile"), str):
            result.append(item["profile"])
    for key in ("profile", "subagent_type"):
        if isinstance(arguments.get(key), str):
            result.append(arguments[key])
    return result


def _all_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_all_text(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_all_text(item) for item in value)
    return ""


def _instrument_interface(
    parent: Any, store: Any, owner: str,
    *, accepted_running_guidance: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], Callable[[], None]]:
    original = getattr(parent, "_dispatch_worker_interface", None)
    if not callable(original):
        raise RuntimeError("AIAgent._dispatch_worker_interface is unavailable; integrate the interface adapter first")
    events: list[dict[str, Any]] = []

    def dispatch(tool_name: str, arguments: Mapping[str, Any]) -> str:
        arguments = dict(arguments or {})
        target = _target_from_args(arguments)
        before_runs: list[Mapping[str, Any]] = []
        if target:
            with contextlib.suppress(Exception):
                before_runs = store.list_runs(target, owner)
        requested_run_id = arguments.get("run_id")
        selected_run = (
            next((item for item in before_runs if item.get("run_id") == requested_run_id), None)
            if isinstance(requested_run_id, str) and requested_run_id
            else (before_runs[-1] if before_runs else None)
        )
        event: dict[str, Any] = {
            "tool_name": tool_name,
            "profiles": _profiles_from_args(arguments),
            "selected_run_status": (selected_run or {}).get("status", UNKNOWN),
            "followup_avoids_labels": not any(
                label in _all_text(arguments).lower() for label in ("cobalt", "amber", "violet", "sapphire", "silver")
            ),
        }
        try:
            result = original(tool_name, arguments)
            payload = json.loads(result) if isinstance(result, str) else dict(result)
            interface = payload.get("orchestration_interface") or {}
            event.update({
                "operation": interface.get("operation"),
                "effective_action": interface.get("effective_action"),
                "accepted": not bool(payload.get("error")),
                "classification": _error_classification(payload.get("error")),
                "delivery": (
                    payload.get("delivery")
                    if payload.get("delivery") in GUIDANCE_DELIVERIES else UNKNOWN
                ),
            })
            if accepted_running_guidance is not None and _is_running_guidance(event):
                accepted_running_guidance()
        except Exception as exc:
            event.update({"accepted": False, "classification": _error_classification(exc), "effective_action": "none"})
            events.append(event)
            raise
        after_runs = []
        if target:
            with contextlib.suppress(Exception):
                after_runs = store.list_runs(target, owner)
        event["run_count_delta"] = len(after_runs) - len(before_runs)
        event["linked_followup"] = bool(
            len(after_runs) == len(before_runs) + 1 and before_runs
            and after_runs[-1].get("previous_run_id") == before_runs[-1].get("run_id")
        )
        event["targets_latest_run"] = bool(
            after_runs and payload.get("run_id") == after_runs[-1].get("run_id")
        )
        events.append(event)
        return result

    parent._dispatch_worker_interface = dispatch
    return events, lambda: setattr(parent, "_dispatch_worker_interface", original)


def _worker_observations(store: Any, owner: str, routes: Iterable[RouteChoice]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {route.profile: route for route in routes}
    workers = []
    usage: dict[str, int | float] = {}
    known_costs: list[float] = []
    for worker in store.list_workers(owner):
        runs = store.list_runs(worker["worker_id"], owner)
        route = expected.get(worker.get("profile"))
        wire_matches = []
        tools_match = []
        retained_summary = ""
        for run in runs:
            result = run.get("result") or {}
            receipt = result.get("route") or {}
            expected_provider = (route.expected_provider or route.provider) if route else None
            resolved = (receipt.get("resolved_provider"), receipt.get("resolved_model"))
            transmitted = (receipt.get("transmitted_provider"), receipt.get("transmitted_model"))
            match = bool(
                route and receipt.get("requested_profile") == route.profile
                and resolved == (expected_provider, route.model)
                and transmitted == (expected_provider, route.model)
                and receipt.get("request_evidence_source")
                and receipt.get("response_evidence_source")
            )
            wire_matches.append(match)
            tools_match.append(result.get("effective_tools") == [])
            retained_summary = str(result.get("summary") or "").lower()
            for key, value in _safe_usage((result.get("usage") or {}).get("tokens")).items():
                usage[key] = usage.get(key, 0) + value
            cost = _number((result.get("cost") or {}).get("usd"))
            if cost is not None:
                known_costs.append(float(cost))
        workers.append({
            "profile": worker.get("profile"),
            "provider": (route.expected_provider or route.provider) if route else None,
            "run_count": len(runs),
            "all_terminal_acked": bool(runs) and all(
                run.get("status") in TERMINAL_STATES and bool(run.get("completion_ack")) for run in runs
            ),
            "wire_matches": bool(wire_matches) and all(wire_matches),
            "tools_match": bool(tools_match) and all(tools_match),
            "retained_cobalt_violet": "cobalt" in retained_summary and "violet" in retained_summary,
            "retained_sapphire_silver": "sapphire" in retained_summary and "silver" in retained_summary,
        })
    return workers, {
        "child_usage": usage,
        "cost": {"status": "known", "usd": round(sum(known_costs), 8)} if known_costs else {"status": UNKNOWN},
    }


def _setup_foreign(store: Any) -> tuple[str, str, dict[str, Any]]:
    owner = "qualification-foreign-owner"
    worker = store.create_worker(owner, profile="synthetic-foreign")
    queued = store.enqueue_run(worker["worker_id"], owner, goal="synthetic foreign retained state")
    active = store.claim_run(queued["run_id"], owner)
    store.finish_run(
        active["run_id"], owner, active["lease_token"], status="SUCCEEDED",
        result={"summary": "synthetic foreign result"},
        history=[{"role": "user", "content": "synthetic"}, {"role": "assistant", "content": "complete"}],
    )
    return owner, worker["worker_id"], _stable_foreign_snapshot(store, owner, worker["worker_id"])


def _setup_idle(parent: Any, store: Any, owner: str, route: RouteChoice, timeout: float) -> tuple[str, str]:
    from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(
        goal="Remember the synthetic label sapphire and report that you retained it.",
        profile=route.profile,
        role="leaf",
    ))
    terminal = service.wait(handle, timeout_seconds=timeout)
    if not terminal.completed or terminal.state.value != "SUCCEEDED":
        raise RuntimeError("Idle-scenario setup worker did not complete")
    store.ack_completion(handle.run_id, owner)
    return str(handle.worker_id), str(handle.run_id)


def _build_config(packet: Mapping[str, Any]) -> dict[str, Any]:
    routes = [RouteChoice(**item) for item in packet["routes"]]
    profiles = {
        route.profile: {
            "description": "Perform only the supplied synthetic qualification task.",
            "instructions": "Keep the answer concise and retain prior synthetic labels within this worker conversation.",
            "provider": route.provider,
            "model": route.model,
            "reasoning_effort": route.reasoning_effort,
            "tool_policy": {"allowed_tools": [], "allowed_mcp_tools": []},
            "workspace_context": {"mode": "inherit", "include_memory": False, "include_context_files": False},
            "execution_limits": {
                "max_iterations": packet["limits"]["max_child_iterations"],
                "timeout_seconds": packet["limits"]["max_child_seconds"],
                "max_spawn_depth": 0,
            },
        }
        for route in routes
    }
    return {
        "model": {"provider": packet["parent_provider"], "default": packet["parent_model"]},
        "orchestration": {"interface": packet["requested_interface"]},
        "delegation": {
            "profiles": profiles,
            "routing_mode": "profile_only",
            "max_concurrent_children": packet["limits"]["max_children"],
            "max_iterations": packet["limits"]["max_child_iterations"],
        },
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
    }


def _execute_child(packet: Mapping[str, Any]) -> dict[str, Any]:
    import yaml

    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        yaml.safe_dump(_build_config(packet)), encoding="utf-8"
    )
    credentials = packet["credentials"]
    from hermes_cli import runtime_provider
    original_resolver = runtime_provider.resolve_runtime_provider

    def resolve(*, requested: str | None = None, target_model: str | None = None, **_kwargs: Any) -> dict[str, Any]:
        if requested not in credentials:
            raise ValueError("Qualification child only enables explicitly selected providers")
        material = credentials[requested]
        return original_resolver(
            requested=requested,
            target_model=target_model,
            explicit_api_key=material.get("api_key"),
            explicit_base_url=material.get("base_url"),
        )

    runtime_provider.resolve_runtime_provider = resolve
    from agent.worker_store import WorkerStore
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.worker_interfaces import advertised_worker_tool_names

    parent_runtime = resolve(requested=packet["parent_provider"], target_model=packet["parent_model"])
    db = SessionDB()
    owner = f"interface-qualification-{packet['scenario']['id']}"
    db.ensure_session(owner, source="worker-orchestration-interface-qualification")
    parent = AIAgent(
        **{key: parent_runtime[key] for key in ("provider", "api_key", "base_url", "api_mode")},
        model=packet["parent_model"],
        session_id=owner,
        session_db=db,
        enabled_toolsets=["delegation"],
        max_iterations=packet["limits"]["max_parent_iterations"],
        max_tokens=packet["limits"]["max_parent_tokens"],
        reasoning_config={"effort": packet["parent_reasoning_effort"]},
        run_budget_seconds=packet["limits"]["max_seconds_per_scenario"],
        skip_memory=True,
        skip_context_files=True,
        skip_background_review=True,
        save_trajectories=False,
        quiet_mode=True,
    )
    store = WorkerStore(db)
    store.ensure_schema()
    routes = [RouteChoice(**item) for item in packet["routes"]]
    scenario = packet["scenario"]
    prompt = scenario.get("prompt")
    foreign = None
    prior_idle_run = None
    if scenario["id"] == "foreign_worker_denied":
        foreign_owner, foreign_id, foreign_before = _setup_foreign(store)
        foreign = (foreign_owner, foreign_id, foreign_before)
        prompt = scenario["prompt_template"].format(foreign_worker_id=foreign_id)
    elif scenario["id"] == "idle_guidance_then_followup":
        worker_id, prior_idle_run = _setup_idle(
            parent, store, owner, routes[0], packet["limits"]["max_child_seconds"]
        )
        prompt = scenario["prompt_template"].format(worker_id=worker_id)
    counts_before = _counts(store, owner)
    gate = threading.Event()
    events, restore_interface = _instrument_interface(
        parent, store, owner, accepted_running_guidance=gate.set
    )
    original_http = AIAgent._interruptible_api_call
    held: set[str] = set()
    held_lock = threading.Lock()

    def held_http(agent: Any, *args: Any, **kwargs: Any) -> Any:
        worker_id = getattr(agent, "_worker_id", None)
        if scenario["id"] == "two_provider_retained_workflow" and worker_id:
            with held_lock:
                should_hold = worker_id not in held
                held.add(worker_id)
            if should_hold and not gate.wait(min(60, packet["limits"]["max_child_seconds"])):
                raise TimeoutError("Controlled worker scheduling barrier expired")
        return original_http(agent, *args, **kwargs)

    AIAgent._interruptible_api_call = held_http
    started = time.monotonic()
    run_result: Any = None
    run_error: Exception | None = None
    try:
        run_result = parent.run_conversation(str(prompt))
    except Exception as exc:
        run_error = exc
    finally:
        gate.set()
        AIAgent._interruptible_api_call = original_http
        restore_interface()
    elapsed = time.monotonic() - started
    counts_after = _counts(store, owner)
    workers, run_evidence = _worker_observations(store, owner, routes)
    all_runs = [run for worker in store.list_workers(owner) for run in store.list_runs(worker["worker_id"], owner)]
    run_ids = [run["run_id"] for run in all_runs]
    parent_usage = _safe_usage({
        key: getattr(parent, f"session_{key}", None)
        for key in SAFE_USAGE_KEYS
    })
    combined_usage = dict(run_evidence["child_usage"])
    for key, value in parent_usage.items():
        combined_usage[key] = combined_usage.get(key, 0) + value
    selection = parent._worker_interface_selection
    advertised = sorted(advertised_worker_tool_names(selection) & set(parent.valid_tool_names))
    foreign_unchanged = None
    if foreign:
        foreign_unchanged = foreign[2] == _stable_foreign_snapshot(store, foreign[0], foreign[1])
    retained_answer_match = any(
        item["retained_cobalt_violet"] or item["retained_sapphire_silver"] for item in workers
    )
    latest_idle = None
    if prior_idle_run and workers:
        idle_worker = next((item for item in store.list_workers(owner) if item.get("profile") == routes[0].profile), None)
        if idle_worker:
            idle_runs = store.list_runs(idle_worker["worker_id"], owner)
            latest_idle = idle_runs[-1] if idle_runs else None
    wire_values = [item["wire_matches"] for item in workers]
    tools_values = [item["tools_match"] for item in workers]
    observation = {
        "parent_provider": parent.provider,
        "parent_model": parent.model,
        "selected_interface": selection.name,
        "interface_version": selection.version,
        "advertised_tool_names": advertised,
        "events": events,
        "workers": workers,
        "parent_completed": run_error is None and not (isinstance(run_result, Mapping) and run_result.get("failed")),
        "counts_before": counts_before,
        "counts_after": counts_after,
        "foreign_preimage_unchanged": foreign_unchanged,
        "retained_answer_match": retained_answer_match,
        "wire_route_agreement": all(wire_values) if wire_values else "not_applicable",
        "tool_receipts_match": all(tools_values) if tools_values else "not_applicable",
        "duplicate_owned_executions": len(run_ids) - len(set(run_ids)),
        "authorization_violations": int(bool(foreign and not foreign_unchanged)),
        "prior_run_replayed": (
            latest_idle is not None and latest_idle.get("run_id") == prior_idle_run
        ) if prior_idle_run else False,
        "elapsed_seconds": elapsed,
        "token_usage": combined_usage,
        "cost": run_evidence["cost"],
        "malformed_tool_calls": UNKNOWN,
        "corrective_turns": UNKNOWN,
        "error_type": type(run_error).__name__ if run_error else None,
    }
    parent.close()
    db.close()
    runtime_provider.resolve_runtime_provider = original_resolver
    return observation


def _source_identity(repo_root: Path) -> str:
    candidate = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=repo_root,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo_root,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Live qualification requires a committed, clean candidate")
    return candidate


def _resolve_credentials(routes: Iterable[tuple[str, str]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    credentials: dict[str, dict[str, Any]] = {}
    resolved_names: dict[str, str] = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for provider, model in routes:
            if provider in credentials:
                continue
            runtime = resolve_runtime_provider(requested=provider, target_model=model)
            credentials[provider] = {key: runtime.get(key) for key in SAFE_RUNTIME_KEYS}
            resolved_names[provider] = str(runtime.get("provider") or provider)
    return credentials, resolved_names


def _child_failure(packet: Mapping[str, Any], error_type: str) -> dict[str, Any]:
    return {
        "parent_provider": packet["parent_provider"],
        "parent_model": packet["parent_model"],
        "selected_interface": UNKNOWN,
        "interface_version": UNKNOWN,
        "advertised_tool_names": [],
        "events": [],
        "workers": [],
        "parent_completed": False,
        "elapsed_seconds": 0,
        "token_usage": {},
        "cost": {"status": UNKNOWN},
        "wire_route_agreement": UNKNOWN,
        "authorization_violations": 0,
        "duplicate_owned_executions": 0,
        "error_type": error_type,
    }


def _run_child_process(packet: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hermes-interface-qualification-") as home:
        env = dict(os.environ, HERMES_HOME=home)
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child"],
                input=json.dumps(packet), text=True, capture_output=True, env=env,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return _child_failure(packet, "ScenarioTimeout")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError):
        return _child_failure(packet, "UnstructuredChildOutput")
    return payload if isinstance(payload, dict) else _child_failure(packet, "InvalidChildOutput")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Allow the selected live provider calls.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--parent-provider", required="--child" not in sys.argv)
    parser.add_argument("--parent-model", required="--child" not in sys.argv)
    parser.add_argument("--parent-reasoning-effort", required="--child" not in sys.argv)
    parser.add_argument("--interface", choices=("hermes", "codex", "claude"), required="--child" not in sys.argv)
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
    fixture = load_fixture(args.fixture)
    policy = fixture["execution_policy"]
    routes = [
        RouteChoice(args.worker_a_profile, args.worker_a_provider, args.worker_a_model, args.worker_a_reasoning_effort),
        RouteChoice(args.worker_b_profile, args.worker_b_provider, args.worker_b_model, args.worker_b_reasoning_effort),
    ]
    if len({route.profile for route in routes}) != 2 or "missing-qualification-profile" in {route.profile for route in routes}:
        raise SystemExit("Worker profiles must be distinct and may not use the reserved missing-profile fixture name")
    if len({route.provider for route in routes}) != 2:
        raise SystemExit("The two worker routes must select different providers")
    candidate = _source_identity(REPO_ROOT)
    credentials, resolved_names = _resolve_credentials([
        (args.parent_provider, args.parent_model),
        *((route.provider, route.model) for route in routes),
    ])
    routes = [RouteChoice(**{**asdict(route), "expected_provider": resolved_names[route.provider]}) for route in routes]
    limits = {
        key: policy[key] for key in (
            "max_parent_iterations", "max_child_iterations", "max_children",
            "max_seconds_per_scenario", "max_child_seconds",
        )
    }
    limits["max_parent_tokens"] = args.max_parent_tokens
    common = {
        "candidate_sha": candidate,
        "parent_provider": resolved_names[args.parent_provider],
        "parent_model": args.parent_model,
        "requested_interface": args.interface,
        "required_receipt_fields": fixture["required_receipt_fields"],
        "limits": limits,
    }

    def execute(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
        packet = {
            "scenario": scenario,
            "credentials": credentials,
            "routes": [asdict(route) for route in routes],
            "parent_provider": args.parent_provider,
            "parent_model": args.parent_model,
            "parent_reasoning_effort": args.parent_reasoning_effort,
            "requested_interface": args.interface,
            "limits": limits,
        }
        return _run_child_process(packet, timeout=limits["max_seconds_per_scenario"] + 15)

    print(json.dumps(run_suite(fixture, execute, common), sort_keys=True))


def main() -> int:
    try:
        _main()
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({
            "qualification": "unqualified",
            "qualified": False,
            "error_type": type(exc).__name__,
            "proof_boundary": "Controller preflight failed before four-scenario qualification completed.",
        }, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
