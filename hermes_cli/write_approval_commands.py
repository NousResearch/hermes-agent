#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands.

Both the interactive CLI (``cli.py``) and the gateway (``gateway/run.py``) call
into this module so the pending-review UX (list / approve / reject / diff /
mode) lives in one place. Each caller owns only its surface concerns:
formatting the returned text and, for the gateway, persisting config + evicting
the cached agent on a mode change.

Every public handler returns a plain text string suitable for both a terminal
and a chat message. Skill diffs are intentionally NOT inlined here — the
``diff`` handler returns the full diff for the CLI pager, but on a messaging
platform the gateway truncates it and points the user at the dashboard / file.
"""

from __future__ import annotations

import json
from typing import List, Optional

from tools import write_approval as wa


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    return f"{subsystem}.write_approval = {'on' if on else 'off'}"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _memory_action_menu(pending_id: str) -> str:
    return "\n".join([
        "Choose:",
        f"  A) Approve this memory write    /memory approve {pending_id}   (or /memory a {pending_id})",
        f"  B) Reject this memory write     /memory reject {pending_id}    (or /memory b {pending_id})",
        "  C) Show all pending memory writes /memory pending              (or /memory c)",
        "  D) Reject all pending memory writes /memory reject all          (or /memory d)",
        f"  E) Review one-by-one / edit      /memory review                (or /memory e)",
        f"     Edit before approving: /memory edit {pending_id} <new text>",
    ])


def _fmt_memory_record(record) -> str:
    payload = record.get("payload", {}) if isinstance(record, dict) else {}
    pending_id = record.get("id", "")
    action = payload.get("action") or record.get("action", "")
    target = payload.get("target", "memory")
    content = payload.get("content") or ""
    old_text = payload.get("old_text") or ""

    lines = [
        "MEMORY WRITE APPROVAL",
        f"Pending ID: {pending_id}",
        f"Action: {action} to {target.upper()}",
    ]
    if old_text:
        lines.extend(["Old content:", old_text])
    if content:
        lines.extend(["Content:", content])
    lines.extend(["", _memory_action_menu(pending_id)])
    return "\n".join(lines)


def _fmt_pending_list(subsystem: str) -> str:
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
    if subsystem == wa.MEMORY:
        lines = [
            f"MEMORY WRITE APPROVAL — pending writes ({len(records)}):",
        ]
        for r in records:
            payload = r.get("payload", {})
            target = str(payload.get("target", "memory")).upper()
            action = payload.get("action", r.get("action", ""))
            content = (payload.get("content") or payload.get("old_text") or r.get("summary", "")).replace("\n", " ")
            if len(content) > 120:
                content = content[:117] + "..."
            lines.append(f"  {r['id']}  {action} {target}: {content}")
        lines.append("")
        lines.append("Review one at a time: /memory review    (or /memory e)")
        lines.append("Approve: /memory approve <id>           (or /memory a <id>)")
        lines.append("Reject:  /memory reject <id>            (or /memory b <id>)")
        lines.append("Edit:    /memory edit <id> <new text>")
        lines.append("Reject all: /memory reject all          (or /memory d)")
        return "\n".join(lines)

    lines = [f"Pending {subsystem} writes ({len(records)}):"]
    for r in records:
        origin = r.get("origin", "foreground")
        tag = " [auto]" if origin == "background_review" else ""
        lines.append(f"  {r['id']}{tag}  {r.get('summary', '')}")
    where = "/{s} approve <id>".format(s=subsystem)
    lines.append("")
    lines.append(f"Apply: {where}   Reject: /{subsystem} reject <id>")
    if subsystem == wa.SKILLS:
        lines.append("Review full diff: /skills diff <id>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------

def handle_pending_subcommand(
    subsystem: str,
    args: List[str],
    *,
    memory_store=None,
    set_mode_fn=None,
) -> Optional[str]:
    """Dispatch a /memory or /skills subcommand.

    Args:
        subsystem: ``memory`` or ``skills``.
        args: tokens after the slash command (e.g. ``["approve", "a1b2"]``).
        memory_store: live MemoryStore for applying approved memory writes
            (CLI passes ``self.agent._memory_store``; gateway applies against a
            freshly loaded store).
        set_mode_fn: optional callable ``(enabled: bool) -> None`` that
            persists the new write_approval boolean to config (gateway provides
            this; CLI uses its own ``save_config_value`` and passes a closure).

    Returns a text string to show the user. Returns None when the args are not
    a write-approval subcommand (caller falls through to its other handling,
    e.g. /skills search).
    """
    if not args:
        # Bare /memory or /skills with no sub → show pending + gate state.
        return f"{_fmt_state(subsystem)}\n\n" + _fmt_pending_list(subsystem)

    sub = args[0].lower()
    rest = args[1:]

    if subsystem == wa.MEMORY:
        alias_map = {
            "a": "approve",
            "b": "reject",
            "c": "pending",
            "d": "reject_all",
            "e": "review",
        }
        sub = alias_map.get(sub, sub)

    if sub == "pending":
        return _fmt_pending_list(subsystem)

    if sub in {"review", "show"} and subsystem == wa.MEMORY:
        return _review_memory(rest)

    if sub == "edit" and subsystem == wa.MEMORY:
        return _edit_memory(rest)

    if sub == "reject_all" and subsystem == wa.MEMORY:
        return _reject(subsystem, ["all"])

    if sub in {"approve", "apply"}:
        return _approve(subsystem, rest, memory_store)

    if sub in {"reject", "deny", "drop"}:
        return _reject(subsystem, rest)

    if sub == "diff" and subsystem == wa.SKILLS:
        return _diff(rest)

    if sub in {"approval", "mode"}:  # 'mode' kept as a back-compat alias
        return _set_approval(subsystem, rest, set_mode_fn)

    return None  # not ours — caller handles


def _resolve_one(subsystem: str, rest: List[str]):
    if not rest:
        return None, f"Usage: /{subsystem} approve|reject <id>  (or 'all')"
    return rest[0], None


def _approve(subsystem: str, rest: List[str], memory_store) -> str:
    target, err = _resolve_one(subsystem, rest)
    if err or target is None:
        return err or f"Usage: /{subsystem} approve <id>"

    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."

    if target.lower() == "all":
        targets = list(records)
    else:
        rec = wa.get_pending(subsystem, target)
        if not rec:
            return f"No pending {subsystem} write with id '{target}'."
        targets = [rec]

    applied, failed = 0, []
    for rec in targets:
        ok, msg = _apply_one(subsystem, rec, memory_store)
        if ok:
            wa.discard_pending(subsystem, rec["id"])
            applied += 1
        else:
            failed.append(f"{rec['id']}: {msg}")

    out = [f"Approved {applied} {subsystem} write(s)."]
    if failed:
        out.append("Failed:")
        out.extend(f"  {f}" for f in failed)
    return "\n".join(out)


def _apply_one(subsystem: str, rec, memory_store):
    payload = rec.get("payload", {})
    try:
        if subsystem == wa.MEMORY:
            if memory_store is None:
                return False, "memory store unavailable"
            from tools.memory_tool import apply_memory_pending
            result = apply_memory_pending(payload, memory_store)
            return bool(result.get("success")), result.get("error", "")
        else:
            from tools.skill_manager_tool import apply_skill_pending
            result = json.loads(apply_skill_pending(payload))
            return bool(result.get("success")), result.get("error", "")
    except Exception as e:
        return False, str(e)


def _reject(subsystem: str, rest: List[str]) -> str:
    target, err = _resolve_one(subsystem, rest)
    if err or target is None:
        return err or f"Usage: /{subsystem} reject <id>"
    if target.lower() == "all":
        n = 0
        for rec in wa.list_pending(subsystem):
            if wa.discard_pending(subsystem, rec["id"]):
                n += 1
        return f"Rejected {n} pending {subsystem} write(s)."
    if wa.discard_pending(subsystem, target):
        return f"Rejected pending {subsystem} write '{target}'."
    return f"No pending {subsystem} write with id '{target}'."


def _review_memory(rest: List[str]) -> str:
    if rest:
        rec = wa.get_pending(wa.MEMORY, rest[0])
        if not rec:
            return f"No pending memory write with id '{rest[0]}'."
    else:
        records = wa.list_pending(wa.MEMORY)
        if not records:
            return "No pending memory writes."
        rec = records[0]
    return _fmt_memory_record(rec)


def _edit_memory(rest: List[str]) -> str:
    if len(rest) < 2:
        return "Usage: /memory edit <id> <new text>"
    pending_id = rest[0]
    rec = wa.get_pending(wa.MEMORY, pending_id)
    if not rec:
        return f"No pending memory write with id '{pending_id}'."
    payload = dict(rec.get("payload", {}))
    action = payload.get("action")
    if action not in {"add", "replace"}:
        return f"Pending memory write '{pending_id}' is action '{action}' and cannot be edited. Reject it instead."
    new_text = " ".join(rest[1:]).strip()
    if not new_text:
        return "Usage: /memory edit <id> <new text>"
    payload["content"] = new_text
    target = payload.get("target", "memory")
    summary = f"{action} to {target}: {new_text[:120]}"
    updated = wa.update_pending(wa.MEMORY, pending_id, payload, summary=summary)
    if updated is None:
        return f"Failed to edit pending memory write '{pending_id}'."
    return "Updated pending memory write.\n\n" + _fmt_memory_record(updated)


def _diff(rest: List[str]) -> str:
    if not rest:
        return "Usage: /skills diff <id>"
    rec = wa.get_pending(wa.SKILLS, rest[0])
    if not rec:
        return f"No pending skill write with id '{rest[0]}'."
    diff = wa.skill_pending_diff(rec)
    header = f"# Pending skill write {rec['id']}: {rec.get('summary', '')}\n"
    return header + "\n" + diff


def _set_approval(subsystem: str, rest: List[str], set_mode_fn) -> str:
    """Turn the approval gate on/off for a subsystem.

    ``set_mode_fn`` (when provided) persists the new boolean to config.
    """
    if not rest:
        return (f"{_fmt_state(subsystem)}\n"
                f"Set with: /{subsystem} approval <on|off>")
    arg = rest[0].strip().lower()
    truthy = {"on", "true", "yes", "1", "enable", "enabled"}
    falsey = {"off", "false", "no", "0", "disable", "disabled"}
    if arg in truthy:
        enabled = True
    elif arg in falsey:
        enabled = False
    else:
        return f"Invalid value '{arg}'. Use: on or off."
    if set_mode_fn is None:
        val = "true" if enabled else "false"
        return (f"To change the {subsystem} approval gate, run:\n"
                f"  hermes config set {subsystem}.write_approval {val}")
    try:
        set_mode_fn(enabled)
    except Exception as e:
        return f"Failed to set {subsystem}.write_approval: {e}"
    return f"{subsystem}.write_approval set to '{'on' if enabled else 'off'}'."
