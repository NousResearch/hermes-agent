#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands."""

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import List, Optional

from tools import write_approval as wa


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    state = f"{subsystem}.write_approval = {'on' if on else 'off'}"
    if subsystem == wa.SKILLS:
        try:
            scope = wa.skill_write_approval_mode()
        except ValueError:
            scope = "invalid; select create or all"
        state += f" (scope: {scope})"
    return state


def _fmt_pending_list(subsystem: str) -> str:
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
    lines = [f"Pending {subsystem} writes ({len(records)}):"]
    for r in records:
        origin = r.get("origin", "foreground")
        tag = " [auto]" if origin == "background_review" else ""
        lines.append(f"  {r['id']}{tag}  {r.get('summary', '')}")
        if subsystem == wa.MEMORY:
            lines.extend(f"      {line}" for line in _matched_entries(r["payload"]))
    lines.append("")
    lines.append(f"Apply: /{subsystem} approve <id>   Reject: /{subsystem} reject <id>")
    if subsystem == wa.SKILLS:
        lines.append("Review full diff: /skills diff <id>")
    return "\n".join(lines)


def handle_pending_subcommand(
    subsystem: str, args: List[str], *, memory_store=None, set_mode_fn=None) -> Optional[str]:
    """Dispatch a /memory or /skills write-approval subcommand.

    ``memory_store`` applies approved memory writes (CLI passes its live store; gateway a freshly
    loaded one); ``set_mode_fn(enabled, scope=None)`` persists the gate plus optional skill scope.
    Legacy boolean-only setters still work for on/off. Returns text for the user,
    or None when the args are not a write-approval subcommand so the caller falls through to its
    other handling (e.g. /skills search).
    """
    if not args:
        return f"{_fmt_state(subsystem)}\n{_approval_help(subsystem)}\n\n" + _fmt_pending_list(subsystem)
    sub, rest = args[0].lower(), args[1:]
    if sub == "pending":
        return _fmt_pending_list(subsystem)
    if sub in {"approve", "apply", "reject", "deny", "drop"}:
        return _review_write(subsystem, sub, rest, memory_store)
    if sub == "diff" and subsystem == wa.SKILLS:
        return _diff(rest)
    if sub in {"approval", "mode"}:  # 'mode' kept as a back-compat alias
        return _set_approval(subsystem, rest, set_mode_fn)
    return None  # not ours — caller handles


def _usage(subsystem: str) -> str:
    return f"Usage: /{subsystem} approve|reject <id>  (or 'all')"


def _review_write(subsystem, sub, rest, memory_store):
    """Hold the skill fence through lookup, apply/reject and removal of the exact pending ID."""
    from tools.skill_write_approval import creation_transaction
    fence = creation_transaction() if subsystem == wa.SKILLS else nullcontext()
    try:
        with fence:
            if sub in {"approve", "apply"}:
                return _approve(subsystem, rest, memory_store)
            return _reject(subsystem, rest)
    except TimeoutError:
        return "Another skill writer is busy. Pending requests were not changed; retry the review command."
    except ValueError as exc:
        return str(exc)


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

    applied, failed, overwritten, removed = 0, [], [], []
    for rec in targets:
        ok, msg, result = _apply_one(subsystem, rec, memory_store)
        if ok:
            wa.discard_pending(subsystem, rec["id"])
            applied += 1
            overwritten.extend(f"  {rec['id']}: {text}" for text in _changed_entries(result, "replaced"))
            removed.extend(f"  {rec['id']}: {text}" for text in _changed_entries(result, "removed"))
        else:
            failed.append(f"{rec['id']}: {msg}")

    out = [f"Approved {applied} {subsystem} write(s)."]
    if overwritten:
        # A memory 'replace' overwrites the WHOLE matched entry (#117952); the approver
        # is the last person who can notice a clause went missing, so show what was lost.
        out.append("Overwrote entire entry (re-add anything you still need):")
        out.extend(overwritten)
    if removed:
        out.append("Removed entry (re-add anything you still need):")
        out.extend(removed)
    if failed:
        out.append("Failed:")
        out.extend(f"  {f}" for f in failed)
    return "\n".join(out)


def _changed_entries(result: dict, kind: str) -> List[str]:
    """Full text of every entry a memory replace overwrote (``kind="replaced"``) or remove
    deleted (``"removed"``), single-op or batch shape."""
    single = result.get(f"{kind}_entry")
    batch = result.get(f"{kind}_entries") or {}
    return ([single] if single else []) + [batch[k] for k in sorted(batch, key=int)]


def _matched_entries(payload) -> List[str]:
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


def persist_write_approval(subsystem: str, enabled: bool, scope=None, *, config_path=None) -> None:
    """Save the gate and optional skill scope in ONE profile-local round-trip write.

    Raise on invalid input or I/O failure; callers must not acknowledge a failed write.
    Omitted scope preserves the existing selection (including the legacy default: all).
    """
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from hermes_constants import get_hermes_home

    if subsystem not in {wa.SKILLS, wa.MEMORY}:
        raise ValueError("Unknown write-approval subsystem")
    if scope is not None and (subsystem != wa.SKILLS or scope not in {"create", "all"}):
        raise ValueError("Only skills support scope 'create' or 'all'")
    path = config_path if config_path is not None else get_hermes_home() / "config.yaml"
    config = read_user_config_raw(path)
    settings = config.setdefault(subsystem, {})
    settings["write_approval"] = bool(enabled)
    if scope is not None:
        settings["write_approval_mode"] = scope
    atomic_config_write(path, config)


def _approval_help(subsystem: str) -> str:
    if subsystem == wa.SKILLS:
        return ("Set with: /skills approval <on|off|create|all> (alias: /skills mode)\n"
                "create: enable approval for new skills only; existing skills improve automatically.\n"
                "all: enable approval for every skill mutation (default scope).\n"
                "on/off: enable/disable the gate without changing the selected scope.\n"
                "Changes affect this profile's next write; pending writes stay pending.\n"
                "Review: /skills pending, /skills diff <id>, /skills approve <id>, /skills reject <id>.")
    return f"Set with: /{subsystem} approval <on|off>"


def _set_approval(subsystem: str, rest: List[str], set_mode_fn) -> str:
    """Toggle the gate, or atomically enable and select a skill-only scope."""
    arg = rest[0].strip().lower() if rest else ""
    if not rest or (len(rest) == 1 and arg in {"status", "current", "help"}):
        return f"{_fmt_state(subsystem)}\n{_approval_help(subsystem)}"
    scope = arg if subsystem == wa.SKILLS and arg in {"create", "all"} else None
    enabled = True if scope else _APPROVAL_VALUES.get(arg)
    if len(rest) != 1 or enabled is None:
        values = "on, off, create or all" if subsystem == wa.SKILLS else "on or off"
        return f"Invalid value '{' '.join(rest)}'. Use: {values}."
    if set_mode_fn is None:
        if scope:
            return f"Use /skills approval {scope} in a session with settings persistence.\n{_approval_help(subsystem)}"
        val = "true" if enabled else "false"
        return (f"To change the {subsystem} approval gate, run:\n"
                f"  hermes config set {subsystem}.write_approval {val}")
    try:
        # Legacy boolean-only callbacks keep working for on/off; scopes opt into the new keyword.
        if scope is None:
            set_mode_fn(enabled)
        else:
            set_mode_fn(enabled, scope=scope)
    except Exception as e:
        return f"Failed to set {subsystem}.write_approval: {e}"
    out = f"{subsystem}.write_approval set to '{'on' if enabled else 'off'}'."
    if subsystem == wa.SKILLS:
        state = f"scope: {scope}" if scope else _fmt_state(subsystem)
        out += f"\n{state}\nPending writes stay pending; review with /skills pending."
    return out
