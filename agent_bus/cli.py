"""hermes-bus CLI — used by OpenClaw (via terminal tool) and humans."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make sibling packages importable when invoked as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_bus import core


def _default_agent() -> str:
    return os.environ.get("HERMES_BUS_AGENT", "openclaw")


def _print(obj, *, as_json: bool, human_fmt=None) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
        return
    if human_fmt is None:
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
        return
    print(human_fmt(obj))


def _fmt_task_row(task: dict) -> str:
    tid = task["task_id"]
    status = task["status"]
    goal = (task["goal"] or "").strip().replace("\n", " ")
    if len(goal) > 80:
        goal = goal[:77] + "..."
    return f"{tid}  [{status:<8}]  {task['from_agent']:>8} → {task['to_agent']:<8}  {goal}"


def _cmd_assign(args) -> None:
    task = core.assign_task(
        from_agent=args["as"],
        to_agent=args["to"],
        goal=args["goal"],
        success_criteria=args.get("success"),
        context=args.get("context"),
        priority=args.get("priority", "P2"),
        deadline_minutes=args.get("deadline_minutes"),
        parent_task_id=args.get("parent"),
    )
    _print(task, as_json=args.get("json"), human_fmt=lambda t: (
        f"Assigned {t['task_id']}: {args['as']} → {args['to']}\n"
        f"  goal: {t['goal']}\n"
        f"  slack_thread: {t.get('slack_thread_ts') or '(none)'}"
    ))


def _cmd_ack(args) -> None:
    task = core.ack_task(task_id=args["task_id"], agent=args["as"], note=args.get("note"))
    _print(task, as_json=args.get("json"), human_fmt=lambda t: f"ACK {t['task_id']} (by {args['as']})")


def _cmd_progress(args) -> None:
    task = core.progress_task(task_id=args["task_id"], agent=args["as"], note=args["note"])
    _print(task, as_json=args.get("json"), human_fmt=lambda t: f"PROGRESS {t['task_id']}")


def _cmd_done(args) -> None:
    task = core.complete_task(
        task_id=args["task_id"], agent=args["as"],
        result=args["result"], learning=args.get("learning"),
    )
    _print(task, as_json=args.get("json"), human_fmt=lambda t: f"DONE {t['task_id']} (by {args['as']})")


def _cmd_fail(args) -> None:
    task = core.fail_task(
        task_id=args["task_id"], agent=args["as"],
        reason=args["reason"], learning=args.get("learning"),
    )
    _print(task, as_json=args.get("json"), human_fmt=lambda t: f"FAIL {t['task_id']} (by {args['as']})")


def _cmd_inbox(args) -> None:
    tasks = core.get_outstanding(args["agent"])
    _print(tasks, as_json=args.get("json"), human_fmt=lambda rs: (
        f"Inbox for {args['agent']} ({len(rs)} outstanding):\n" +
        ("\n".join(_fmt_task_row(r) for r in rs) if rs else "  (empty)")
    ))


def _cmd_outbox(args) -> None:
    tasks = core.get_sent(args["agent"], include_terminal=args.get("all", False))
    _print(tasks, as_json=args.get("json"), human_fmt=lambda rs: (
        f"Outbox for {args['agent']} ({len(rs)} items):\n" +
        ("\n".join(_fmt_task_row(r) for r in rs) if rs else "  (empty)")
    ))


def _cmd_show(args) -> None:
    task = core.get_task(args["task_id"])
    if not task:
        print(f"Task not found: {args['task_id']}", file=sys.stderr)
        sys.exit(2)
    _print(task, as_json=args.get("json"))


def _cmd_recent(args) -> None:
    tasks = core.list_recent(limit=args.get("limit", 20))
    _print(tasks, as_json=args.get("json"), human_fmt=lambda rs: (
        f"{len(rs)} recent tasks:\n" + "\n".join(_fmt_task_row(r) for r in rs)
    ))


def _cmd_check_timeouts(args) -> None:
    timed = core.check_timeouts()
    _print(timed, as_json=args.get("json"), human_fmt=lambda rs: (
        f"Timed out: {len(rs)}\n" + "\n".join(_fmt_task_row(r) for r in rs)
    ))


def _cmd_check_exit(args) -> None:
    """Slice 5 — show open tasks for `agent`. Exit 1 if any, so shell wrappers
    can gate: `hermes-bus check-exit --as openclaw || handle_open_tasks`.
    """
    import sys as _sys
    from agent_bus import finalizer
    agent = args.get("as") or _default_agent()
    open_tasks = finalizer.check_session_before_exit(agent)
    if args.get("json"):
        print(json.dumps({"agent": agent, "open_tasks": open_tasks},
                         ensure_ascii=False, indent=2, default=str))
    else:
        if not open_tasks:
            print(f"✅ {agent}: no open tasks — safe to exit")
        else:
            print(f"⚠️  {agent}: {len(open_tasks)} open task(s) — decide done/fail/keep-alive:")
            for t in open_tasks:
                age_min = int(t["age_sec"] / 60)
                print(f"  {t['task_id']}  [{t['status']}] {age_min}m old — {t['goal'][:70]}")
    _sys.exit(1 if open_tasks else 0)


def _cmd_check_artifacts(args) -> None:
    """Slice 6 — watchdog: scan open tasks for wiki artifact matches.

    `--nudge` actually posts Slack nudges + increments watchdog_nudged_count.
    Without `--nudge`, advisory-only dry-run.
    """
    from agent_bus import core as _core
    from agent_bus import storage as _storage

    # Query open tasks
    open_rows = _storage.query_tasks(
        status_in=["pending", "ack", "progress", "keep-alive"], limit=100,
    )
    results = []
    for t in open_rows:
        age_sec = time.time() - (t.get("created_at") or time.time())
        if age_sec < 1800:  # <30 min, too early
            continue
        goal = t.get("goal") or ""
        if not goal.strip():
            continue
        hits = _core.wiki_query(goal[:80], limit=3)
        # Filter out bus-learning self-references
        relevant = [
            h for h in hits
            if not h.get("path", "").startswith("memory/") or "_agent-bus_" not in h.get("path", "")
        ]
        if not relevant:
            continue
        results.append({
            "task_id": t["task_id"],
            "status": t["status"],
            "age_min": int(age_sec / 60),
            "goal": goal[:80],
            "matched": relevant[0].get("path"),
        })

    if args.get("json"):
        print(json.dumps({"suspect_tasks": results},
                         ensure_ascii=False, indent=2, default=str))
    else:
        if not results:
            print("✓ No artifact-exists-but-bus-open suspects")
        else:
            print(f"⚠️  {len(results)} task(s) look like they have output but bus is open:")
            for r in results:
                print(f"  {r['task_id']}  [{r['status']}] {r['age_min']}m  → {r['matched']}")
                print(f"                 goal: {r['goal']}")
            if args.get("nudge"):
                print("(nudge mode — posting Slack reminders is NOT YET IMPLEMENTED "
                      "in this CLI path; use dashboard UI for one-click follow-up.)")


def _add_json_flag(parser):
    """Add --json to both top-level and each subparser so users can put it
    either before or after the subcommand (`hermes-bus --json show T-XXX`
    AND `hermes-bus show T-XXX --json` both work)."""
    parser.add_argument(
        "--json", action="store_true", default=None,
        help="Output JSON instead of human-readable",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hermes-bus", description="Agent-to-agent task bus")
    _add_json_flag(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    # assign
    a = sub.add_parser("assign", help="Assign a new task")
    _add_json_flag(a)
    a.add_argument("--as", dest="as_agent", default=_default_agent(),
                   help="Agent issuing the task (default: $HERMES_BUS_AGENT or 'openclaw')")
    a.add_argument("--to", required=True, help="Recipient agent (hermes|openclaw)")
    a.add_argument("--goal", required=True, help="What to do (one line)")
    a.add_argument("--success", help="Success criteria")
    a.add_argument("--context", help="Extra context (markdown)")
    a.add_argument("--priority", default="P2", choices=["P0", "P1", "P2", "P3"])
    a.add_argument("--deadline-minutes", type=int, help="Deadline in minutes from now")
    a.add_argument("--parent", help="Parent task_id for sub-tasks")
    a.set_defaults(func=lambda ns: _cmd_assign({
        "as": ns.as_agent, "to": ns.to, "goal": ns.goal,
        "success": ns.success, "context": ns.context,
        "priority": ns.priority, "deadline_minutes": ns.deadline_minutes,
        "parent": ns.parent, "json": ns.json,
    }))

    # state-changing (ack/progress/done/fail)
    for name, fn, extra_flags in [
        ("ack", _cmd_ack, [("--note", {"required": False})]),
        ("progress", _cmd_progress, [("--note", {"required": True})]),
        ("done", _cmd_done, [("--result", {"required": True}), ("--learning", {"required": False})]),
        ("fail", _cmd_fail, [("--reason", {"required": True}), ("--learning", {"required": False})]),
    ]:
        sp = sub.add_parser(name, help=f"{name.upper()} a task")
        _add_json_flag(sp)
        sp.add_argument("task_id")
        sp.add_argument("--as", dest="as_agent", default=_default_agent())
        for flag, kwargs in extra_flags:
            sp.add_argument(flag, **kwargs)
        sp.set_defaults(func=_wrap_state_cmd(fn))

    # inbox/outbox/show/recent
    ib = sub.add_parser("inbox", help="Tasks assigned to me (still open)")
    _add_json_flag(ib)
    ib.add_argument("--agent", default=_default_agent())
    ib.set_defaults(func=lambda ns: _cmd_inbox({"agent": ns.agent, "json": ns.json}))

    ob = sub.add_parser("outbox", help="Tasks I sent")
    _add_json_flag(ob)
    ob.add_argument("--agent", default=_default_agent())
    ob.add_argument("--all", action="store_true", help="Include completed tasks")
    ob.set_defaults(func=lambda ns: _cmd_outbox({"agent": ns.agent, "all": ns.all, "json": ns.json}))

    sh = sub.add_parser("show", help="Show a task + its event log")
    _add_json_flag(sh)
    sh.add_argument("task_id")
    sh.set_defaults(func=lambda ns: _cmd_show({"task_id": ns.task_id, "json": ns.json}))

    rc = sub.add_parser("recent", help="Recent tasks across both agents")
    _add_json_flag(rc)
    rc.add_argument("--limit", type=int, default=20)
    rc.set_defaults(func=lambda ns: _cmd_recent({"limit": ns.limit, "json": ns.json}))

    ct = sub.add_parser("check-timeouts", help="Mark past-deadline tasks as timed out")
    _add_json_flag(ct)
    ct.set_defaults(func=lambda ns: _cmd_check_timeouts({"json": ns.json}))

    # Slice 5: check-exit
    ce = sub.add_parser("check-exit", help="Show open tasks for `agent` (Slice 5)")
    _add_json_flag(ce)
    ce.add_argument("--as", dest="as_agent", default=_default_agent(),
                    help="Agent whose open tasks to check")
    ce.set_defaults(func=lambda ns: _cmd_check_exit({"as": ns.as_agent, "json": ns.json}))

    # Slice 6: check-artifacts
    ca = sub.add_parser(
        "check-artifacts",
        help="Scan open tasks for wiki artifact matches (Slice 6 watchdog)",
    )
    _add_json_flag(ca)
    ca.add_argument("--nudge", action="store_true",
                    help="Actually post nudges (default: advisory dry-run)")
    ca.set_defaults(func=lambda ns: _cmd_check_artifacts({
        "json": ns.json, "nudge": ns.nudge,
    }))

    return p


def _wrap_state_cmd(fn):
    def _wrapped(ns):
        args = {
            "task_id": ns.task_id,
            "as": ns.as_agent,
            "json": ns.json,
        }
        for attr in ("note", "result", "reason", "learning"):
            if hasattr(ns, attr):
                args[attr] = getattr(ns, attr)
        fn(args)
    return _wrapped


def main(argv=None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    # Argparse records the last occurrence of --json. Top-level value is
    # captured pre-subparser; subparser overrides. Coerce to bool.
    ns.json = bool(getattr(ns, "json", None))
    try:
        ns.func(ns)
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
