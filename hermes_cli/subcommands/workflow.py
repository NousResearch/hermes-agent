"""``hermes workflow`` — list stored graphs, and publish the events that move their runs.

Nothing here runs a graph. ``run``, ``event``, ``approve`` and ``deny`` publish a workflow event;
the reactor in the desktop backend or the messaging gateway (whichever holds the lease) acts on it.
With neither up the event waits on disk and is picked up when one starts. ``/workflow`` in a chat
runs the same verbs (``run_slash``).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shlex
from typing import Callable

_LIVE = ("running", "paused", "waiting_human", "waiting_world")


def _add_verbs(sub) -> None:
    sub.add_parser("list", aliases=["ls"], help="List stored workflows")

    run = sub.add_parser("run", help="Start a workflow")
    run.add_argument("name", help="Workflow id or name")
    run.add_argument("--payload", default="", help="JSON payload handed to the first step")

    event = sub.add_parser("event", help="Publish an event: starts event triggers, resumes waits")
    event.add_argument("name", help="Event name, e.g. github.pull_request.merged")
    event.add_argument("--payload", default="", help="JSON payload")

    status = sub.add_parser("status", help="Show live or recent runs, and what they wait on")
    status.add_argument("name", nargs="?", help="Workflow id or name")

    for verb, text in (("approve", "Approve a step waiting on you"), ("deny", "Deny a step waiting on you")):
        answer = sub.add_parser(verb, help=text)
        answer.add_argument("code", help="The approval code from the request (see `hermes workflow status`)")


def build_workflow_parser(subparsers, *, cmd_workflow: Callable) -> None:
    parser = subparsers.add_parser(
        "workflow",
        help="Run stored agent workflows",
        description="List workflows and publish the events that start, signal and answer their runs.",
    )
    _add_verbs(parser.add_subparsers(dest="workflow_command"))
    parser.set_defaults(func=cmd_workflow)


def _parse_payload(raw: str):
    text = (raw or "").strip()
    return json.loads(text) if text else None


def _note_if_no_host() -> None:
    from workflow.reactor import host_running

    if not host_running():
        print("Queued. It runs when the desktop app or `hermes gateway` is up.")


def _print_runs(runs: list[dict]) -> None:
    for state in runs:
        print(f"{state['runId']}\t{state.get('status')}\t{state.get('workflowId')}")
        for park in (state.get("parks") or {}).values():
            if park.get("kind") == "human":
                print(f"  waiting on {park.get('who') or 'you'}: {park.get('prompt')}  (code {park.get('code')})")
            else:
                print(f"  waiting on {park.get('until')}: {park.get('event') or park.get('url') or ''}".rstrip())


def _list(args) -> None:
    from workflow.store import load_documents

    docs = load_documents()["docs"]
    if not docs:
        print("No workflows stored.")
    for doc in docs:
        steps = (doc.get("scenario") or {}).get("steps") or []
        print(f"{doc['id']}\t{doc.get('name') or doc['id']}\t{len(steps)} steps")


def _run(args) -> None:
    from workflow import events
    from workflow.store import get_document, new_run_id

    doc = get_document(args.name)
    if doc is None:
        print(f"No workflow called '{args.name}'.")
        return
    try:
        payload = _parse_payload(getattr(args, "payload", "") or "")
    except json.JSONDecodeError as exc:
        print(f"Bad --payload: {exc}")
        return
    run_id = new_run_id()
    events.publish(events.START, {"workflowId": doc["id"], "runId": run_id, "payload": payload, "source": "cli"},
                   source="cli")
    print(f"{run_id}\tqueued\t{doc['id']}")
    _note_if_no_host()


def _event(args) -> None:
    from workflow import events

    try:
        payload = _parse_payload(getattr(args, "payload", "") or "")
    except json.JSONDecodeError as exc:
        print(f"Bad --payload: {exc}")
        return
    queued = events.publish(args.name, payload, source="cli")
    print(f"published {queued['name']} ({queued['id']})")
    _note_if_no_host()


def _status(args) -> None:
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
    shown = [r for r in runs if r.get("status") in _LIVE] or runs[-5:]
    if not shown:
        print("No runs.")
        return
    _print_runs(shown)


def _answer(args) -> None:
    from workflow import events
    from workflow.waits import find_approval

    found = find_approval(args.code)
    if found is None:
        print(f"Nothing is waiting on approval code '{args.code}'.")
        return
    state, park = found
    decision = "approved" if args.workflow_command == "approve" else "denied"
    events.publish(events.ANSWER, {"runId": state["runId"], "nodeId": park["nodeId"], "decision": decision,
                                   "by": getattr(args, "by", None) or "cli"}, source="cli")
    print(f"{decision}: {park.get('prompt')} ({state.get('name') or state.get('workflowId')})")
    _note_if_no_host()


_VERBS: dict[str, Callable] = {
    "list": _list, "ls": _list, "run": _run, "event": _event, "status": _status, "approve": _answer, "deny": _answer,
}


def workflow_command(args) -> None:
    handler = _VERBS.get(getattr(args, "workflow_command", None) or "")
    if handler is None:
        print("usage: hermes workflow {list,run,event,status,approve,deny}")
        return
    handler(args)


def run_slash(text: str, *, by: str | None = None) -> str:
    """``/workflow <verb> ...`` from a chat: the same verbs, output captured as the reply."""
    parser = argparse.ArgumentParser(prog="/workflow", add_help=False, exit_on_error=False)
    _add_verbs(parser.add_subparsers(dest="workflow_command"))
    out = io.StringIO()
    try:
        args = parser.parse_args(shlex.split(text or ""))
    except (argparse.ArgumentError, SystemExit, ValueError) as exc:
        return f"{exc}\nusage: /workflow {{list,run,event,status,approve,deny}}"
    args.by = by
    with contextlib.redirect_stdout(out):
        workflow_command(args)
    return out.getvalue().strip()
