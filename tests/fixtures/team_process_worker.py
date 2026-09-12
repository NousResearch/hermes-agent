#!/usr/bin/env python3
"""Hermetic subprocess driver for integrated team restart acceptance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OWNER = "team-process-owner"
PROFILE = "team-process"
EFFECT_TOOL = "team_process_effect"
TEAM_TOOLS = {
    "kanban_team", "delegate_task", "kanban_create", "kanban_heartbeat",
    "kanban_request_review", "kanban_request_changes", "kanban_complete", "kanban_block",
}
MARKERS = ("implementation-context", "review-one", "correction-context", "review-two")


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
    marker = os.environ.get("HERMES_TEAM_MARKER", "")
    if marker:
        Path(marker).touch()


def _capture_request(kwargs: dict) -> None:
    messages = [item for item in (kwargs.get("messages") or []) if isinstance(item, dict)]
    user_text = "\n".join(
        str(item.get("content") or "") for item in messages if item.get("role") == "user"
    )
    _append_json(
        os.environ.get("HERMES_TEAM_CAPTURE", ""),
        {
            "phase": os.environ.get("HERMES_TEAM_PHASE", ""),
            "pid": os.getpid(),
            "roles": [str(item.get("role") or "") for item in messages],
            "user_markers": [marker for marker in MARKERS if marker in user_text],
        },
    )


def _response(*, tool: bool = False):
    if tool:
        call = SimpleNamespace(
            id="team-process-effect-call",
            type="function",
            function=SimpleNamespace(
                name=EFFECT_TOOL,
                arguments=json.dumps({"effect_id": "review-effect"}),
            ),
        )
        message = SimpleNamespace(
            content=None, tool_calls=[call], reasoning=None, reasoning_content=None,
        )
        finish_reason = "tool_calls"
    else:
        message = SimpleNamespace(
            content="synthetic team completion", tool_calls=None,
            reasoning=None, reasoning_content=None,
        )
        finish_reason = "stop"
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=None,
        model="fixture/model",
    )


class _FixtureCompletions:
    def __init__(self) -> None:
        self.request_index = 0

    def create(self, **kwargs):
        current = self.request_index
        self.request_index += 1
        _capture_request(kwargs)
        if os.environ.get("HERMES_TEAM_BEHAVIOR") == "tool-crash" and current == 0:
            return _response(tool=True)
        return _response()


class _FixtureOpenAI:
    def __init__(self, **kwargs) -> None:
        self.base_url = kwargs.get("base_url")
        self.chat = SimpleNamespace(completions=_FixtureCompletions())

    def close(self) -> None:
        pass


def _effect_handler(args: dict, **_metadata) -> str:
    _append_json(
        os.environ.get("HERMES_TEAM_EFFECT", ""),
        {"effect_id": str(args.get("effect_id") or ""), "pid": os.getpid()},
    )
    _touch_marker()
    os._exit(91)


def _install_boundaries() -> None:
    import agent.process_bootstrap as process_bootstrap
    from tools.registry import registry

    process_bootstrap.OpenAI = _FixtureOpenAI
    registry.register(
        EFFECT_TOOL,
        "file",
        {
            "name": EFFECT_TOOL,
            "description": "Record one synthetic team acceptance effect.",
            "parameters": {
                "type": "object",
                "properties": {"effect_id": {"type": "string"}},
                "required": ["effect_id"],
            },
        },
        _effect_handler,
    )


def _service(home: Path):
    from agent.shared_discovery import build_local_discovery_scope
    from agent.team_orchestration import TeamOrchestrationService
    from hermes_state import SessionDB

    db = SessionDB(home / "state.db")
    if db.get_session(OWNER) is None:
        db.create_session(OWNER, source="cli", model="fixture/parent")
    parent = SimpleNamespace(
        session_id=OWNER,
        _session_db=db,
        _shared_discovery_scope=build_local_discovery_scope(),
        _worker_effective_tool_names=set(TEAM_TOOLS) | {EFFECT_TOOL},
        _executable_tool_names=set(TEAM_TOOLS) | {EFFECT_TOOL},
        enabled_toolsets=["file"],
        disabled_toolsets=[],
        valid_tool_names=set(TEAM_TOOLS) | {EFFECT_TOOL},
        tools=[],
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
        _current_turn_id="team-process-turn",
    )
    service = TeamOrchestrationService(parent)
    service._start_monitor = lambda *_args, **_kwargs: None
    return service, db


def _expire_leases(db) -> None:
    import agent.worker_store as worker_store
    from agent.worker_store import WorkerStore

    prior_clock = worker_store.time
    worker_store.time = SimpleNamespace(time=lambda: time.time() + 120)
    try:
        WorkerStore(db).recover_expired_runs(OWNER)
    finally:
        worker_store.time = prior_clock


def _hold_before_schedule(service) -> None:
    original = service.lifecycle.schedule_team_execution

    def hold(worker_id, run_id, *, admission):
        _append_json(
            os.environ.get("HERMES_TEAM_EVENTS", ""),
            {
                "phase": os.environ.get("HERMES_TEAM_PHASE", ""),
                "worker_ref": f"worker:{worker_id}",
                "run_ref": f"run:{run_id}",
                "task_ref": f"task:{admission['task_id']}",
            },
        )
        _touch_marker()
        while True:
            time.sleep(1)

    service.lifecycle.schedule_team_execution = hold
    service._original_schedule_team_execution = original


def _emit(value: dict) -> None:
    print(json.dumps(value, sort_keys=True), flush=True)


def _dispatch(args) -> None:
    _install_boundaries()
    service, db = _service(args.home)
    try:
        if args.expire_leases:
            _expire_leases(db)
        if os.environ.get("HERMES_TEAM_BEHAVIOR") == "block-before-schedule":
            _hold_before_schedule(service)
        result = dict(service.dispatch(json.loads(args.payload)))
        run_ref = str(result.get("run_ref") or "")
        worker_ref = str(result.get("worker_ref") or "")
        if args.wait and run_ref and worker_ref:
            result["terminal"] = service.lifecycle.control(
                "wait",
                worker_id=worker_ref.partition(":")[2],
                run_id=run_ref.partition(":")[2],
                timeout_seconds=args.timeout,
            )
        _emit(result)
    finally:
        db.close()


def _snapshot(args) -> None:
    from agent.worker_store import WorkerStore
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    _service_obj, db = _service(args.home)
    conn = kbc.connect()
    try:
        task_id = args.task_ref.partition(":")[2]
        task = kb.get_task(conn, task_id)
        store = WorkerStore(db)
        attached = []
        for event in kb.list_events(conn, task_id):
            if event.kind != "execution_attached":
                continue
            ref = event.payload
            worker_id = str(ref.get("worker_ref") or "").partition(":")[2]
            run_id = str(ref.get("run_ref") or "").partition(":")[2]
            run = store.get_run(run_id, OWNER)
            worker = store.get_worker(worker_id, OWNER)
            attached.append({
                "role": ref.get("role"),
                "worker_ref": ref.get("worker_ref"),
                "run_ref": ref.get("run_ref"),
                "status": run["status"],
                "previous_run_ref": f"run:{run['previous_run_id']}" if run.get("previous_run_id") else None,
                "uncertain_side_effect": bool(
                    run.get("uncertain_side_effect") or worker.get("uncertain_side_effect")
                ),
            })
        _emit({
            "task_ref": args.task_ref,
            "task_status": task.status,
            "current_run_id": task.current_run_id,
            "attachments": attached,
            "ownership_verified": True,
        })
    finally:
        conn.close()
        db.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dispatch", "snapshot"))
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--payload", default="{}")
    parser.add_argument("--task-ref", default="")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--expire-leases", action="store_true")
    parser.add_argument("--timeout", type=float, default=20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if Path(os.environ["HERMES_HOME"]).resolve() != args.home.resolve():
        raise RuntimeError("subprocess HERMES_HOME does not match the isolated test profile")
    {"dispatch": _dispatch, "snapshot": _snapshot}[args.command](args)


if __name__ == "__main__":
    main()
