#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands."""

from __future__ import annotations

import json
from typing import List, Optional

from tools import write_approval as wa


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    return f"{subsystem}.write_approval = {'on' if on else 'off'}"


def _fmt_memory_record(record) -> str:
    """Render one pending memory write for explicit review (#44963): exact payload
    content plus the A–E action menu, so a memory write never looks like a generic
    command approval. Batch payloads render every operation (sweeper finding on #44966)."""
    payload = record.get("payload", {}) if isinstance(record, dict) else {}
    pending_id = record.get("id", "")
    action = payload.get("action", record.get("action", ""))
    target = str(payload.get("target", "memory")).upper()
    lines = [f"MEMORY WRITE APPROVAL",
             f"Pending ID: {pending_id}",
             f"Action: {action} to {target}"]
    if action == "batch":
        from tools.memory_tool import format_batch_op_line
        lines.extend(format_batch_op_line(op) for op in payload.get("operations") or [])
    else:
        old_text = payload.get("old_text") or ""
        content = payload.get("content") or ""
        if old_text:
            lines.extend(["Old content:", old_text])
        if content:
            lines.extend(["Content:", content])
    lines.extend(["",
                  "Choose:",
                  f"  A) Approve this memory write    /memory approve {pending_id}   (or /memory a {pending_id})",
                  f"  B) Reject this memory write     /memory reject {pending_id}    (or /memory b {pending_id})",
                  "  C) Show all pending memory writes  /memory pending              (or /memory c)",
                  "  D) Reject all pending memory writes  /memory reject all         (or /memory d)",
                  "  E) Review one-by-one / edit      /memory review                (or /memory e)"])
    if action in {"add", "replace"}:
        lines.append(f"     Edit before approving: /memory edit {pending_id} <new text>")
    return "\n".join(lines)


def _fmt_pending_list(subsystem: str) -> str:
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
    if subsystem == wa.MEMORY:
        lines = [f"MEMORY WRITE APPROVAL — pending writes ({len(records)}):"]
        for r in records:
            payload = r.get("payload", {})
            target = str(payload.get("target", "memory")).upper()
            action = payload.get("action", r.get("action", ""))
            if action == "batch":
                detail = f"batch ({len(payload.get('operations') or [])} op(s))"
            else:
                detail = str(payload.get("content") or payload.get("old_text") or r.get("summary") or "")
                detail = detail.replace("\n", " ")
                if len(detail) > 120:
                    detail = detail[:117] + "..."
            tag = " [auto]" if r.get("origin") == "background_review" else ""
            lines.append(f"  {r['id']}{tag}  {action} to {target}: {detail}")
        lines.extend(["",
                      "Review one at a time: /memory review    (or /memory e)",
                      "Approve: /memory approve <id>           (or /memory a <id>)",
                      "Reject:  /memory reject <id>            (or /memory b <id>)",
                      "Edit:    /memory edit <id> <new text>",
                      "Reject all: /memory reject all          (or /memory d)"])
        return "\n".join(lines)
    lines = [f"Pending {subsystem} writes ({len(records)}):"]
    for r in records:
        origin = r.get("origin", "foreground")
        tag = " [auto]" if origin == "background_review" else ""
        lines.append(f"  {r['id']}{tag}  {r.get('summary', '')}")
    lines.append("")
    lines.append(f"Apply: /{subsystem} approve <id>   Reject: /{subsystem} reject <id>")
    if subsystem == wa.SKILLS:
        lines.append("Review full diff: /skills diff <id>")
    return "\n".join(lines)


def handle_pending_subcommand(
    subsystem: str, args: List[str], *, memory_store=None, set_mode_fn=None) -> Optional[str]:
    """Dispatch a /memory or /skills write-approval subcommand.

    ``memory_store`` applies approved memory writes (CLI passes its live store; gateway a freshly
    loaded one); ``set_mode_fn`` persists the write_approval boolean. Returns text for the user,
    or None when the args are not a write-approval subcommand so the caller falls through to its
    other handling (e.g. /skills search).
    """
    if not args:
        return f"{_fmt_state(subsystem)}\n\n" + _fmt_pending_list(subsystem)
    sub, rest = args[0].lower(), args[1:]
    if subsystem == wa.MEMORY:
        # #44963 A–E single-letter aliases (memory-only; dispatch-level so the
        # registry/palette surfaces keep showing full words).
        sub = {"a": "approve", "b": "reject", "c": "pending", "d": "reject_all",
               "e": "review"}.get(sub, sub)
    if sub == "pending":
        return _fmt_pending_list(subsystem)
    if sub == "review" and subsystem == wa.MEMORY:
        return _review_memory(rest)
    if sub == "edit" and subsystem == wa.MEMORY:
        return _edit_memory(rest)
    if sub == "reject_all" and subsystem == wa.MEMORY:
        if rest:
            return (f"'/memory d' rejects ALL pending writes. "
                    f"To reject a single write, use '/memory b {rest[0]}'.")
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


def _usage(subsystem: str) -> str:
    return f"Usage: /{subsystem} approve|reject <id>  (or 'all')"


def _approve(subsystem: str, rest: List[str], memory_store) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
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
        else:
            from tools.skill_manager_tool import apply_skill_pending
            result = json.loads(apply_skill_pending(payload))
        return bool(result.get("success")), result.get("error", "")
    except Exception as e:
        return False, str(e)


def _reject(subsystem: str, rest: List[str]) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
    if target.lower() == "all":
        n = sum(1 for rec in wa.list_pending(subsystem) if wa.discard_pending(subsystem, rec["id"]))
        return f"Rejected {n} pending {subsystem} write(s)."
    if wa.discard_pending(subsystem, target):
        return f"Rejected pending {subsystem} write '{target}'."
    return f"No pending {subsystem} write with id '{target}'."


def _review_memory(rest: List[str]) -> str:
    """Show one pending memory write at a time (#44963 E-loop): no id → the oldest
    pending record (list_pending order, stateless — no review cursor state)."""
    if rest:
        rec = wa.get_pending(wa.MEMORY, rest[0])
        if not rec:
            return f"No pending memory write with id '{rest[0]}'."
        return _fmt_memory_record(rec)
    records = wa.list_pending(wa.MEMORY)
    if not records:
        return "No pending memory writes."
    return _fmt_memory_record(records[0])


def _edit_memory(rest: List[str]) -> str:
    """Edit a pending memory write's content before approval (#44963). Single-op
    add/replace records only — batch/remove records are rejected-and-reissued, never
    half-edited (a batch edits as one atomic unit via /memory reject + re-stage)."""
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
    updated = wa.update_pending(wa.MEMORY, pending_id, payload,
                                summary=f"{action} to {target}: {new_text[:120]}")
    if updated is None:
        return f"Failed to edit pending memory write '{pending_id}'."
    return "Updated pending memory write.\n\n" + _fmt_memory_record(updated)


def _diff(rest: List[str]) -> str:
    if not rest:
        return "Usage: /skills diff <id>"
    rec = wa.get_pending(wa.SKILLS, rest[0])
    if not rec:
        return f"No pending skill write with id '{rest[0]}'."
    return f"# Pending skill write {rec['id']}: {rec.get('summary', '')}\n\n" + wa.skill_pending_diff(rec)


_APPROVAL_VALUES = {
    **dict.fromkeys(("on", "true", "yes", "1", "enable", "enabled"), True),
    **dict.fromkeys(("off", "false", "no", "0", "disable", "disabled"), False)}


def _set_approval(subsystem: str, rest: List[str], set_mode_fn) -> str:
    """Turn the approval gate on/off for a subsystem."""
    if not rest:
        return (f"{_fmt_state(subsystem)}\n"
                f"Set with: /{subsystem} approval <on|off>")
    arg = rest[0].strip().lower()
    enabled = _APPROVAL_VALUES.get(arg)
    if enabled is None:
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
