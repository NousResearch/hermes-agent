"""``hermes workflow`` — list stored graphs and start / signal a run."""

from __future__ import annotations

import json
from typing import Callable


def build_workflow_parser(subparsers, *, cmd_workflow: Callable) -> None:
    parser = subparsers.add_parser(
        "workflow",
        help="Run stored agent workflows",
        description="List workflows on disk and start or signal a run.",
    )
    sub = parser.add_subparsers(dest="workflow_command")

    sub.add_parser("list", aliases=["ls"], help="List stored workflows")

    run = sub.add_parser("run", help="Start a workflow")
    run.add_argument("name", help="Workflow id or name")
    run.add_argument("--payload", default="", help="JSON payload handed to the first step")

    event = sub.add_parser("event", help="Resume waits / start event triggers")
    event.add_argument("name", help="Event name, e.g. github.pull_request.merged")
    event.add_argument("--payload", default="", help="JSON payload")

    status = sub.add_parser("status", help="Show live or recent runs")
    status.add_argument("name", nargs="?", help="Workflow id or name")

    parser.set_defaults(func=cmd_workflow)


def _parse_payload(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    return json.loads(text)


def workflow_command(args) -> None:
    action = getattr(args, "workflow_command", None)
    if action in {None, ""}:
        print("usage: hermes workflow {list,run,event,status}")
        return

    if action in {"list", "ls"}:
        from workflow.store import load_documents

        docs = load_documents()["docs"]
        if not docs:
            print("No workflows stored.")
            return
        for doc in docs:
            steps = (doc.get("scenario") or {}).get("steps") or []
            print(f"{doc['id']}\t{doc.get('name') or doc['id']}\t{len(steps)} steps")
        return

    if action == "run":
        from workflow.runner import start_run

        try:
            payload = _parse_payload(getattr(args, "payload", "") or "")
            state = start_run(args.name, payload=payload, source="cli")
        except json.JSONDecodeError as exc:
            print(f"Bad --payload: {exc}")
            return
        except ValueError as exc:
            print(exc)
            return
        print(f"{state['runId']}\t{state.get('status')}\t{state.get('workflowId')}")
        return

    if action == "event":
        from workflow.runner import start_matching

        try:
            payload = _parse_payload(getattr(args, "payload", "") or "")
            started = start_matching(event=args.name, payload=payload, source="cli")
        except json.JSONDecodeError as exc:
            print(f"Bad --payload: {exc}")
            return
        if not started:
            print("No matching workflow or parked wait.")
            return
        for state in started:
            print(f"{state['runId']}\t{state.get('status')}\t{state.get('workflowId')}")
        return

    if action == "status":
        from workflow.store import get_document, list_runs

        name = getattr(args, "name", None)
        if name:
            doc = get_document(name)
            if doc is None:
                print(f"No workflow '{name}'.")
                return
            runs = list_runs(doc["id"])
        else:
            runs = list_runs()
        live = {"running", "paused", "waiting_human", "waiting_world"}
        shown = [r for r in runs if r.get("status") in live] or runs[-5:]
        if not shown:
            print("No runs.")
            return
        for state in shown:
            print(f"{state['runId']}\t{state.get('status')}\t{state.get('workflowId')}")
        return
