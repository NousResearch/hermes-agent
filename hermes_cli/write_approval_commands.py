#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands."""

from __future__ import annotations

import json
import re
from typing import List, Optional

from tools import write_approval as wa


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    return f"{subsystem}.write_approval = {'on' if on else 'off'}"


_DEFAULT_LIMIT = 20
_SUMMARY_CLIP = 80


def _fmt_pending_list(subsystem: str) -> str:
    return _fmt_pending_list_inner(subsystem, show_all=False)


def _fmt_all_pending_list(subsystem: str) -> str:
    return _fmt_pending_list_inner(subsystem, show_all=True)


def _fmt_pending_list_inner(subsystem: str, show_all: bool) -> str:
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
    total = len(records)
    visible = records if show_all else records[:_DEFAULT_LIMIT]
    lines = [f"Pending {subsystem} writes ({total}):"]
    if subsystem == wa.SKILLS:
        _fmt_skills_grouped(lines, visible)
    else:
        for record in visible:
            _fmt_one_record(lines, record)
            if subsystem == wa.MEMORY:
                lines.extend(f"      {line}" for line in _matched_entries(record["payload"]))
    if not show_all and total > _DEFAULT_LIMIT:
        lines.extend(["", f"… and {total - _DEFAULT_LIMIT} more — /{subsystem} pending --all"])
    lines.extend(["", f"Apply: /{subsystem} approve <id>   Reject: /{subsystem} reject <id>"])
    if subsystem == wa.SKILLS:
        lines.append("Review full diff: /skills diff <id>")
    return "\n".join(lines)


def _fmt_one_record(lines: list[str], record: dict) -> None:
    origin = record.get("origin", "foreground")
    tag = " [auto]" if origin == "background_review" else ""
    summary = (record.get("summary") or "").strip()
    if len(summary) > _SUMMARY_CLIP:
        summary = summary[:_SUMMARY_CLIP].rstrip() + "…"
    lines.append(f"  {record['id']}{tag}  {summary}")


def _fmt_skills_grouped(lines: list[str], records: list[dict]) -> None:
    by_name: dict[str, list[dict]] = {}
    names: list[str] = []
    for record in records:
        name = _extract_skill_name(record)
        if name not in by_name:
            by_name[name] = []
            names.append(name)
        by_name[name].append(record)
    for name in names:
        group = by_name[name]
        if len(group) == 1:
            _fmt_one_record(lines, group[0])
        else:
            actions = ", ".join(record.get("action", "?") for record in group)
            lines.append(f"  ⚠ {name}  ({len(group)} pending: {actions})")
            for record in group:
                _fmt_one_record(lines, record)


def _extract_skill_name(record: dict) -> str:
    payload = record.get("payload", {})
    name = payload.get("name", "")
    if name:
        return str(name).strip()[:60]
    summary = (record.get("summary") or "").strip()
    match = re.match(r"^(create|update|patch|delete)\s+(.+?)(?:\s*[—–-].*)?$", summary, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    fallback = payload.get("old_string", "").split("\n")[0][:40]
    return fallback or summary[:40]



def handle_pending_subcommand(
    subsystem: str, args: list[str], *, memory_store=None, set_mode_fn=None) -> Optional[str]:
    """Dispatch a /memory or /skills write-approval subcommand.

    ``memory_store`` applies approved memory writes (CLI passes its live store; gateway a freshly
    loaded one); ``set_mode_fn`` persists the write_approval boolean. Returns text for the user,
    or None when the args are not a write-approval subcommand so the caller falls through to its
    other handling (e.g. /skills search).
    """
    if not args:
        return f"{_fmt_state(subsystem)}\n\n" + _fmt_pending_list(subsystem)
    sub, rest = args[0].lower(), args[1:]
    if sub == "pending":
        show_all = len(rest) >= 1 and rest[0].lower() in {"--all", "-a", "all"}
        return (_fmt_all_pending_list(subsystem) if show_all else _fmt_pending_list(subsystem))
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


def _approve(subsystem: str, rest: list[str], memory_store) -> str:
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

    applied, failed, overwritten, patched, removed = 0, [], [], [], []
    for rec in targets:
        ok, msg, result = _apply_one(subsystem, rec, memory_store)
        if ok:
            wa.discard_pending(subsystem, rec["id"])
            applied += 1
            overwritten.extend(f"  {rec['id']}: {text}" for text in _changed_entries(result, "replaced"))
            patched.extend(f"  {rec['id']}: {text}" for text in _changed_entries(result, "patched"))
            removed.extend(f"  {rec['id']}: {text}" for text in _changed_entries(result, "removed"))
        else:
            failed.append(f"{rec['id']}: {msg}")

    out = [f"Approved {applied} {subsystem} write(s)."]
    if overwritten:
        # A memory 'replace' overwrites the WHOLE matched entry (#117952); the approver
        # is the last person who can notice a clause went missing, so show what was lost.
        out.append("Overwrote entire entry (re-add anything you still need):")
        out.extend(overwritten)
    if patched:
        # A patch keeps the rest of the entry, so show it as it was, not as lost text.
        out.append("Patched entry (as it was before the edit):")
        out.extend(patched)
    if removed:
        out.append("Removed entry (re-add anything you still need):")
        out.extend(removed)
    if failed:
        out.append("Failed:")
        out.extend(f"  {f}" for f in failed)
    return "\n".join(out)


def _changed_entries(result: dict, kind: str) -> list[str]:
    """Full text of every entry a memory write changed, single-op or batch shape."""
    single = result.get(f"{kind}_entry")
    if kind == "patched" and single is None:
        single = result.get("patched_entry_before")
    batch = result.get(f"{kind}_entries") or {}
    return ([single] if single else []) + [batch[k] for k in sorted(batch, key=int)]


def _matched_entries(payload) -> list[str]:
    """The full entry each staged memory replace/remove is pinned to: the summary shows only
    the old_text search string, and approval applies to this entry, not to that search."""
    from tools.memory_tool import destructive_ops
    return [f"{op['action']}s entry: {op['matched_entry']}" if op.get("matched_entry")
            else f"{op['action']}: unpinned legacy target \u2014 reject and recreate before approving"
            for op in destructive_ops(payload)]


def _apply_one(subsystem: str, rec, memory_store):
    """``(ok, error, result)`` — *result* is the applier's full payload (empty on exceptions)."""
    payload = rec.get("payload", {})
    try:
        if subsystem == wa.MEMORY:
            if memory_store is None:
                return False, "memory store unavailable", {}
            from tools.memory_tool import apply_memory_pending
            result = apply_memory_pending(payload, memory_store)
        else:
            from tools.skill_manager_tool import apply_skill_pending
            result = json.loads(apply_skill_pending(payload))
        return bool(result.get("success")), result.get("error", ""), result
    except Exception as e:
        return False, str(e), {}


def _reject(subsystem: str, rest: list[str]) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
    if target.lower() == "all":
        n = sum(1 for rec in wa.list_pending(subsystem) if wa.discard_pending(subsystem, rec["id"]))
        return f"Rejected {n} pending {subsystem} write(s)."
    if wa.discard_pending(subsystem, target):
        return f"Rejected pending {subsystem} write '{target}'."
    return f"No pending {subsystem} write with id '{target}'."


def _diff(rest: list[str]) -> str:
    if not rest:
        return "Usage: /skills diff <id>"
    rec = wa.get_pending(wa.SKILLS, rest[0])
    if not rec:
        return f"No pending skill write with id '{rest[0]}'."
    return f"# Pending skill write {rec['id']}: {rec.get('summary', '')}\n\n" + wa.skill_pending_diff(rec)


_APPROVAL_VALUES = {
    **dict.fromkeys(("on", "true", "yes", "1", "enable", "enabled"), True),
    **dict.fromkeys(("off", "false", "no", "0", "disable", "disabled"), False)}


def _set_approval(subsystem: str, rest: list[str], set_mode_fn) -> str:
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
