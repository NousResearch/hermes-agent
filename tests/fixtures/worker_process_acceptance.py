#!/usr/bin/env python3
"""Hermetic subprocess driver for durable worker crash/restart acceptance tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OWNER = "process-acceptance-owner"
PROFILE = "process-acceptance"
TOOL_NAME = "process_acceptance_effect"
SYNTHETIC_MARKERS = (
    "enqueue-one",
    "enqueue-two",
    "message-one",
    "message-two",
    "checkpoint-resume",
    "post-checkpoint",
    "ambiguous-effect",
    "reconciled-resume",
    "group-one",
    "group-two",
)


def _append_json(path: str, value: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _touch_marker() -> None:
    marker = os.environ.get("HERMES_ACCEPTANCE_MARKER", "")
    if marker:
        Path(marker).touch()


def _message_roles(messages) -> list[str]:
    return [str(item.get("role") or "") for item in messages if isinstance(item, dict)]


def _latest_user_text(messages) -> str:
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "user":
            return str(item.get("content") or "")
    return ""


def _tool_result_categories(messages) -> list[str]:
    categories = []
    for item in messages:
        if not isinstance(item, dict) or item.get("role") != "tool":
            continue
        content = str(item.get("content") or "").lower()
        if "not permitted" in content:
            categories.append("policy_denied")
        elif "error executing tool" in content or '"error"' in content:
            categories.append("execution_error")
        elif '"ok": true' in content:
            categories.append("synthetic_effect_ok")
        else:
            categories.append("other")
    return categories


def _capture_request(kwargs: dict, request_index: int) -> None:
    messages = kwargs.get("messages") or []
    system = next(
        (str(item.get("content") or "") for item in messages
         if isinstance(item, dict) and item.get("role") == "system"),
        "",
    )
    latest_user = _latest_user_text(messages)
    _append_json(
        os.environ.get("HERMES_ACCEPTANCE_CAPTURE", ""),
        {
            "pid": os.getpid(),
            "request_index": request_index,
            "roles": _message_roles(messages),
            "system_hash": hashlib.sha256(system.encode()).hexdigest(),
            "latest_user_markers": [marker for marker in SYNTHETIC_MARKERS if marker in latest_user],
            "tool_names": sorted(
                item.get("function", {}).get("name", "")
                for item in (kwargs.get("tools") or []) if isinstance(item, dict)
            ),
            "tool_result_categories": _tool_result_categories(messages),
            "model": str(kwargs.get("model") or ""),
        },
    )


def _tool_response():
    call = SimpleNamespace(
        id="process-acceptance-call",
        type="function",
        function=SimpleNamespace(
            name=TOOL_NAME,
            arguments=json.dumps({"effect_id": "synthetic-effect"}),
        ),
    )
    message = SimpleNamespace(
        content=None,
        tool_calls=[call],
        reasoning=None,
        reasoning_content=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=None,
        model="fixture/model",
    )


def _final_response():
    message = SimpleNamespace(
        content="synthetic completion",
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
        model="fixture/model",
    )


class _FixtureCompletions:
    def __init__(self) -> None:
        self.request_index = 0

    def create(self, **kwargs):
        current = self.request_index
        self.request_index += 1
        _capture_request(kwargs, current)
        behavior = os.environ.get("HERMES_ACCEPTANCE_BEHAVIOR", "final")
        if behavior == "block-before-checkpoint" and current == 0:
            _touch_marker()
            while True:
                time.sleep(1)
        if behavior in {"checkpoint-block", "tool-crash"} and current == 0:
            return _tool_response()
        if behavior == "checkpoint-block" and current == 1:
            _touch_marker()
            while True:
                time.sleep(1)
        return _final_response()


class _FixtureOpenAI:
    def __init__(self, **kwargs) -> None:
        self.base_url = kwargs.get("base_url")
        self.chat = SimpleNamespace(completions=_FixtureCompletions())

    def close(self) -> None:
        pass


def _effect_handler(args: dict, **_metadata) -> str:
    effect_id = str(args.get("effect_id") or "")
    _append_json(
        os.environ.get("HERMES_ACCEPTANCE_EFFECT", ""),
        {"effect_id": effect_id, "pid": os.getpid()},
    )
    if os.environ.get("HERMES_ACCEPTANCE_BEHAVIOR") == "tool-crash":
        _touch_marker()
        os._exit(91)
    return json.dumps({"ok": True, "effect_id": effect_id})


def _install_boundaries() -> None:
    import agent.process_bootstrap as process_bootstrap
    from tools.registry import registry

    process_bootstrap.OpenAI = _FixtureOpenAI
    registry.register(
        TOOL_NAME,
        "file",
        {
            "name": TOOL_NAME,
            "description": "Record one synthetic local acceptance effect.",
            "parameters": {
                "type": "object",
                "properties": {"effect_id": {"type": "string"}},
                "required": ["effect_id"],
            },
        },
        _effect_handler,
    )


def _parent(home: Path):
    from hermes_state import SessionDB

    db = SessionDB(home / "state.db")
    # A real parent has a durable session before it delegates. Child session
    # rows reference it through the existing sessions foreign key.
    if db.get_session(OWNER) is None:
        db.create_session(OWNER, source="cli", model="fixture/parent")
    parent = SimpleNamespace(
        session_id=OWNER,
        _session_db=db,
        enabled_toolsets=["file"],
        disabled_toolsets=[],
        valid_tool_names={TOOL_NAME},
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        provider="openrouter",
        api_mode="chat_completions",
        model="fixture/parent",
        request_overrides={},
        platform="cli",
        capabilities={},
        prefill_messages=None,
        skip_context_files=True,
        skip_memory=True,
        _delegate_depth=0,
        _delegate_spawn_allowed=True,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
        session_estimated_cost_usd=0.0,
        session_cost_status="unknown",
        session_cost_source="none",
        _current_turn_id="process-acceptance-turn",
    )
    return parent, db


def _expire_prior_leases(store) -> None:
    import agent.worker_store as worker_store

    # Advance only the recovery read. New runs must receive real-time leases;
    # retaining the offset would make the next process's clock move backwards.
    prior_clock = worker_store.time
    worker_store.time = SimpleNamespace(time=lambda: time.time() + 120)
    try:
        store.recover_expired_runs(OWNER)
    finally:
        worker_store.time = prior_clock


def _emit(value: dict) -> None:
    print(json.dumps(value, sort_keys=True), flush=True)


def _seed(args) -> None:
    from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService

    _install_boundaries()
    parent, db = _parent(args.home)
    try:
        service = SubagentLifecycleService(lambda: parent)
        handle = service.launch(SubagentLaunchRequest(goal="seed-worker", profile=PROFILE))
        terminal = service.wait(handle, timeout_seconds=20)
        _emit({
            "worker_id": handle.worker_id,
            "run_id": handle.run_id,
            "status": terminal.state.value,
        })
    finally:
        db.close()


def _enqueue(args) -> None:
    from agent.worker_store import WorkerStore

    parent, db = _parent(args.home)
    try:
        run = WorkerStore(db).enqueue_run(
            args.worker_id,
            OWNER,
            goal=args.message,
            previous_run_id=args.run_id,
        )
        _emit({"worker_id": args.worker_id, "run_id": run["run_id"], "status": run["status"]})
    finally:
        db.close()


def _control(args) -> None:
    from tools.delegate_tool import delegate_task

    _install_boundaries()
    parent, db = _parent(args.home)
    try:
        if args.expire_leases:
            from agent.worker_store import WorkerStore
            _expire_prior_leases(WorkerStore(db))
        payload = json.loads(delegate_task(
            action=args.action,
            worker_id=args.worker_id,
            run_id=args.run_id,
            message=args.message,
            timeout_seconds=args.timeout,
            reconciliation_disposition=args.disposition,
            parent_agent=parent,
        ))
        if args.wait and payload.get("success") and payload.get("run_id"):
            payload["terminal"] = json.loads(delegate_task(
                action="wait",
                worker_id=payload["worker_id"],
                run_id=payload["run_id"],
                timeout_seconds=args.timeout,
                parent_agent=parent,
            ))
        terminal = payload.get("terminal") or payload
        if terminal.get("status") == "FAILED":
            inspected = json.loads(delegate_task(
                action="inspect", worker_id=terminal["worker_id"],
                run_id=terminal["run_id"], parent_agent=parent,
            ))
            failure = (inspected.get("run") or {}).get("result") or {}
            # This fixture owns only synthetic inputs and no live credentials.
            # Expose the existing diagnostic fields, never the conversation.
            payload["failure"] = {
                key: failure.get(key) for key in ("error_classification", "error_message", "termination")
            }
        _emit(payload)
    finally:
        db.close()


def _snapshot(args) -> None:
    from agent.worker_store import WorkerStore

    parent, db = _parent(args.home)
    try:
        store = WorkerStore(db)
        store.ensure_schema()
        if args.expire_leases:
            _expire_prior_leases(store)
        worker = store.get_worker(args.worker_id, OWNER)
        history = list(worker.get("history") or [])
        serialized = json.dumps(history, sort_keys=True)
        runs = store.list_runs(args.worker_id, OWNER)
        messages = store.list_messages(args.worker_id, OWNER)
        _emit({
            "runs": [
                {
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "uncertain_side_effect": run["uncertain_side_effect"],
                    "completion_ack": run["completion_ack"],
                }
                for run in runs
            ],
            "message_statuses": [item["status"] for item in messages],
            "history_roles": [item.get("role") for item in history],
            "history_has_provider_session_handle": any(
                token in serialized for token in ("provider_session", "resume_handle", "session_handle")
            ),
            "worker_uncertain_side_effect": worker["uncertain_side_effect"],
        })
    finally:
        db.close()


def _async_group(args) -> None:
    from tools.async_delegation import get_durable_delegation
    from tools.delegate_tool import delegate_task

    _install_boundaries()
    parent, db = _parent(args.home)
    try:
        dispatched = json.loads(delegate_task(
            tasks=[
                {"goal": "Complete synthetic group-one task", "group": "joined", "profile": PROFILE},
                {"goal": "Complete synthetic group-two task", "group": "joined", "profile": PROFILE},
            ],
            background=True,
            parent_agent=parent,
        ))
        if not dispatched.get("delegation_id"):
            error = str(dispatched.get("error") or "").lower()
            category = (
                "provider_unconfigured" if "no llm provider configured" in error
                else "profile_resolution" if "profile" in error
                else "dispatch_rejected"
            )
            raise RuntimeError(
                f"group dispatch returned no delegation_id "
                f"(status={dispatched.get('status')!r}, category={category})"
            )
        delegation_id = dispatched["delegation_id"]
        deadline = time.monotonic() + args.timeout
        durable = None
        while time.monotonic() < deadline:
            durable = get_durable_delegation(delegation_id)
            if durable and durable["state"] not in {"running", "finalizing"}:
                break
            time.sleep(0.02)
        if not durable or durable["state"] in {"running", "finalizing"}:
            raise TimeoutError("grouped completion did not become durable")
        _emit({
            "delegation_id": delegation_id,
            "dispatch_status": dispatched["status"],
            "durable_state": durable["state"],
            "delivery_state": durable["delivery_state"],
        })
    finally:
        db.close()


def _async_restore(args) -> None:
    from tools.async_delegation import (
        claim_event_delivery,
        complete_event_delivery,
        get_durable_delegation,
        restore_undelivered_completions,
    )

    events = queue.Queue()
    restored = restore_undelivered_completions(events)
    summaries = []
    while not events.empty():
        event = events.get_nowait()
        claim = claim_event_delivery(event, "process-acceptance") if args.ack else None
        if args.ack and claim is not None:
            complete_event_delivery(event, claim)
        durable = get_durable_delegation(str(event.get("delegation_id") or ""))
        summaries.append({
            "delegation_id": event.get("delegation_id"),
            "group": event.get("group"),
            "restored": event.get("restored"),
            "result_indexes": [item.get("task_index") for item in event.get("results") or []],
            "result_statuses": [item.get("status") for item in event.get("results") or []],
            "claimed": claim is not None if args.ack else None,
            "delivery_state": durable.get("delivery_state") if durable else None,
        })
    _emit({"restored": restored, "events": summaries})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seed", "enqueue", "control", "snapshot", "async-group", "async-restore"))
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--worker-id")
    parser.add_argument("--run-id")
    parser.add_argument("--action")
    parser.add_argument("--message", default="")
    parser.add_argument("--disposition")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--ack", action="store_true")
    parser.add_argument("--expire-leases", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if Path(os.environ["HERMES_HOME"]).resolve() != args.home.resolve():
        raise RuntimeError("subprocess HERMES_HOME does not match the isolated test profile")
    {
        "seed": _seed,
        "enqueue": _enqueue,
        "control": _control,
        "snapshot": _snapshot,
        "async-group": _async_group,
        "async-restore": _async_restore,
    }[args.command](args)


if __name__ == "__main__":
    main()
