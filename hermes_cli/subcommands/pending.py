"""``hermes pending`` — the review surface for approval-required persistent changes.

One queue, one command. When ``agent.require_persistent_change_approval`` is on (#110429),
every gated control-file write — memory (MEMORY.md / USER.md), skills, config.yaml — is
*staged* under ``<HERMES_HOME>/pending/<subsystem>/`` instead of committed. This subcommand
lists the queue with a preview and applies (``approve``) or drops (``reject``) an entry,
replaying it through ``tools.write_approval.apply_pending`` — the same dispatcher the
``/memory`` and ``/skills`` slash commands use, so the surfaces cannot drift apart.

Approval is the ONLY path that writes a staged change; a rejected entry is deleted and the
control file is never touched.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from tools import write_approval as wa


def _fmt_preview(record: Dict[str, Any], *, full: bool = False) -> str:
    """One human-readable preview line (or block) for a staged write. Never prints a secret."""
    subsystem = record.get("subsystem") or ""
    payload = record.get("payload") or {}
    if subsystem == wa.CONFIG:
        key = payload.get("key", "?")
        if payload.get("unset"):
            return f"unset {key}"
        try:
            from hermes_cli.config import _masked_display_value
            shown = _masked_display_value(key, payload.get("value", ""))
        except Exception:
            shown = "********"
        return f"set {key} = {shown}"
    if subsystem == wa.MEMORY:
        action = payload.get("action", "?")
        if action == "batch":
            return f"batch: {len(payload.get('operations') or [])} op(s) on {payload.get('target', 'memory')}"
        return f"{action} on {payload.get('target', 'memory')}: {(payload.get('content') or payload.get('old_text') or '')[:120]}"
    if subsystem == wa.SKILLS:
        if full:
            return wa.skill_pending_diff(record)
        return record.get("summary", "")
    return record.get("summary", "")


def _fmt_list(records: List[Dict[str, Any]]) -> str:
    if not records:
        return ("No pending persistent changes.\n"
                f"Gate: agent.require_persistent_change_approval = "
                f"{'on' if wa.persistent_change_approval_required() else 'off'}")
    lines = [f"Pending persistent changes ({len(records)}):"]
    for r in records:
        tag = " [auto]" if r.get("origin") == "background_review" else ""
        lines.append(f"  {r['id']}  {r.get('subsystem', '?'):<7}{tag}  {r.get('summary', '')}")
    lines.append("")
    lines.append("Show:    hermes pending show <id>")
    lines.append("Apply:   hermes pending approve <id|all>")
    lines.append("Discard: hermes pending reject <id|all>")
    return "\n".join(lines)


def _memory_store():
    """Load the on-disk memory store for replaying an approved memory write (no live agent
    exists in a one-shot CLI process)."""
    from tools.memory_tool import load_on_disk_store

    return load_on_disk_store()


def _approve(args) -> int:
    target = (args.id or "").strip()
    if not target:
        print("Usage: hermes pending approve <id|all>", file=sys.stderr)
        return 2
    subsystem = getattr(args, "subsystem", None)
    records = ([wa.get_pending(subsystem, target)] if subsystem else [wa.find_pending(target)])
    if target.lower() == "all":
        records = wa.list_pending(subsystem) if subsystem else wa.all_pending()
    records = [r for r in records if r]
    if not records:
        print(f"No pending write with id '{target}'.")
        return 1

    memory_store: Optional[object] = None
    applied, failed = 0, []
    for rec in records:
        sub = rec.get("subsystem") or subsystem or ""
        if sub == wa.MEMORY and memory_store is None:
            memory_store = _memory_store()
        result = wa.apply_pending(sub, rec, memory_store=memory_store)
        if result.get("success"):
            wa.discard_pending(sub, rec["id"])
            applied += 1
            print(f"✓ applied {rec['id']}: {_fmt_preview(rec)}")
        else:
            failed.append(f"{rec['id']}: {result.get('error', 'unknown error')}")
    print(f"Approved {applied} change(s).")
    if failed:
        print("Failed:")
        for line in failed:
            print(f"  {line}")
        return 1
    return 0


def _reject(args) -> int:
    target = (args.id or "").strip()
    if not target:
        print("Usage: hermes pending reject <id|all>", file=sys.stderr)
        return 2
    subsystem = getattr(args, "subsystem", None)
    if target.lower() == "all":
        records = wa.list_pending(subsystem) if subsystem else wa.all_pending()
        n = sum(1 for rec in records
                if rec.get("id") and wa.discard_pending(rec.get("subsystem") or subsystem, rec["id"]))
        print(f"Rejected {n} change(s). Nothing was written.")
        return 0
    rec = wa.get_pending(subsystem, target) if subsystem else wa.find_pending(target)
    if rec is None or not wa.discard_pending(rec.get("subsystem") or subsystem, target):
        print(f"No pending write with id '{target}'.")
        return 1
    print(f"Rejected {target}. Nothing was written.")
    return 0


def _show(args) -> int:
    target = (args.id or "").strip()
    if not target:
        print("Usage: hermes pending show <id>", file=sys.stderr)
        return 2
    rec = wa.find_pending(target)
    if rec is None:
        print(f"No pending write with id '{target}'.")
        return 1
    print(f"# Pending {rec.get('subsystem')} change {rec['id']} ({rec.get('origin')})")
    print(f"  {rec.get('summary', '')}")
    print()
    print(_fmt_preview(rec, full=True))
    return 0


def _status(args) -> int:
    on = wa.persistent_change_approval_required()
    print(f"agent.{wa.GLOBAL_KEY} = {'on' if on else 'off'}")
    for sub in (wa.MEMORY, wa.SKILLS, wa.CONFIG):
        print(f"  {sub:<7} gate={'on' if wa.write_approval_enabled(sub) else 'off'}"
              f"  pending={wa.pending_count(sub)}")
    if not on:
        print("\nThe global gate is off; per-subsystem write_approval flags still apply.")
        print(f"Turn it on with: hermes config set {wa.GLOBAL_SWITCH} true")
    return 0


_ACTIONS = {"approve": _approve, "reject": _reject, "show": _show, "status": _status}

# Subparser aliases arrive verbatim in ``args.pending_action``; fold them back.
_ALIASES = {"ls": "list", "diff": "show", "apply": "approve", "deny": "reject", "drop": "reject"}


def cmd_pending(args) -> int:
    action = _ALIASES.get(getattr(args, "pending_action", None) or "list",
                          getattr(args, "pending_action", None) or "list")
    if action == "list":
        subsystem = getattr(args, "subsystem", None)
        print(_fmt_list(wa.list_pending(subsystem) if subsystem else wa.all_pending()))
        return 0
    handler = _ACTIONS.get(action)
    if handler is None:
        print("Unknown pending action. See: hermes pending --help", file=sys.stderr)
        return 2
    return handler(args)


def build_pending_parser(subparsers) -> None:
    """Attach the ``pending`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "pending",
        help="Review and approve staged control-file changes "
             "(memory / skills / config.yaml)",
        description=(
            "The review queue for approval-required persistent changes. With "
            f"'{wa.GLOBAL_SWITCH}: true' the agent proposes control-file "
            "changes instead of writing them (memory, skills, config.yaml); each proposal "
            "waits here until it is approved. Approving replays the change; rejecting "
            "deletes it and the control file is left untouched. Off by default — with the "
            "gate off the queue stays empty and nothing is ever staged."),
        epilog=(
            "Examples:\n"
            "  hermes pending                      # list everything waiting\n"
            "  hermes pending show 1a2b3c4d         # preview one change\n"
            "  hermes pending approve 1a2b3c4d      # apply it\n"
            "  hermes pending reject all            # drop the whole queue\n"
            "  hermes pending status                # show which gates are on\n"
            "\n"
            "Exit codes: 0 ok, 1 nothing applied, 2 usage error."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pending_sub = parser.add_subparsers(dest="pending_action")

    list_p = pending_sub.add_parser("list", aliases=["ls"], help="List staged changes (default)")
    list_p.add_argument("--subsystem", choices=[wa.MEMORY, wa.SKILLS, wa.CONFIG], default=None,
                        help="Only show one subsystem")

    show_p = pending_sub.add_parser("show", aliases=["diff"], help="Preview one staged change")
    show_p.add_argument("id", help="Pending change id (from 'hermes pending')")

    approve_p = pending_sub.add_parser("approve", aliases=["apply"], help="Apply a staged change")
    approve_p.add_argument("id", nargs="?", default="", help="Pending change id, or 'all'")
    approve_p.add_argument("--subsystem", choices=[wa.MEMORY, wa.SKILLS, wa.CONFIG], default=None,
                           help="Restrict to one subsystem (with 'all')")

    reject_p = pending_sub.add_parser("reject", aliases=["deny", "drop"], help="Drop a staged change")
    reject_p.add_argument("id", nargs="?", default="", help="Pending change id, or 'all'")
    reject_p.add_argument("--subsystem", choices=[wa.MEMORY, wa.SKILLS, wa.CONFIG], default=None,
                          help="Restrict to one subsystem (with 'all')")

    pending_sub.add_parser("status", help="Show the global gate and per-subsystem state")

    parser.set_defaults(func=cmd_pending)
