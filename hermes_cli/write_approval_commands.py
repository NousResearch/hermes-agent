#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands."""

from __future__ import annotations

from typing import List, Optional

from tools import write_approval as wa

_NOT_PENDING = "no longer pending (already approved or rejected)"


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    return f"{subsystem}.write_approval = {'on' if on else 'off'}"


def _fmt_pending_list(subsystem: str) -> str:
    text = _fmt_queue(subsystem)
    stuck = wa.list_needs_review(subsystem)
    if not stuck:
        return text
    lines = [text, "", f"Needs review ({len(stuck)}): an approval stopped mid-write, so these may be partly applied:"]
    for item in stuck:
        lines.append(f"  {item['id']}  {item['summary']}")
        lines.append(f"      {item['reason']}")
    lines.append("")
    lines.append(f"Check the files by hand, then clear one: /{subsystem} reject <id>  ('reject all' skips these)")
    return "\n".join(lines)


def _fmt_queue(subsystem: str) -> str:
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
        return _fmt_pending_list(subsystem)
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
    if target.lower() == "all":
        if not records:
            return f"No pending {subsystem} writes."
        targets = list(records)
    else:
        rec = wa.get_pending(subsystem, target)
        if not rec:
            if _needs_review(subsystem, target):
                return (f"Write '{target}' needs review, so it won't be applied again. "
                        f"Check the files, then clear it with /{subsystem} reject {target}.")
            return _not_pending(subsystem, target)
        targets = [rec]

    applied, failed, overwritten, patched, removed = 0, [], [], [], []
    for listed in targets:
        pid = listed["id"]
        # legacy_unchecked: a skill record staged without base revisions has nothing to check. The
        # slash commands (CLI, gateway, TUI) still apply it, so records queued by an older version stay
        # approvable. resolve_pending documents this exception; the plugin API always needs a preview.
        r = wa.resolve_pending(subsystem, pid, "approve", memory_store=memory_store, legacy_unchecked=True)
        if r["status"] != "applied":
            failed.append(f"{pid}: {_NOT_PENDING if r['status'] == 'already_resolved' else r['reason']}")
            continue
        applied += 1
        result = r["result"]
        overwritten.extend(f"  {pid}: {text}" for text in _changed_entries(result, "replaced"))
        patched.extend(f"  {pid}: {text}" for text in _changed_entries(result, "patched"))
        removed.extend(f"  {pid}: {text}" for text in _changed_entries(result, "removed"))

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


def _not_pending(subsystem: str, target: str) -> str:
    return f"No pending {subsystem} write with id '{target}'."


def _needs_review(subsystem: str, target: str) -> bool:
    return any(item["id"] == target for item in wa.list_needs_review(subsystem))


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


def _reject(subsystem: str, rest: list[str]) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
    if target.lower() == "all":  # the queue only: a write that needs review takes its explicit id
        n = sum(1 for rec in wa.list_pending(subsystem)
                if wa.resolve_pending(subsystem, rec["id"], "reject")["status"] == "rejected")
        return f"Rejected {n} pending {subsystem} write(s)."
    if wa.get_pending(subsystem, target) is None and not _needs_review(subsystem, target):
        return _not_pending(subsystem, target)
    r = wa.resolve_pending(subsystem, target, "reject", close_interrupted=True)
    if r["status"] == "rejected" and r.get("warning"):
        return f"Rejected {subsystem} write '{target}'. Warning: {r['warning']}"
    if r["status"] == "rejected":
        return f"Rejected pending {subsystem} write '{target}'."
    if r["status"] == "already_resolved":
        return _not_pending(subsystem, target)
    return f"Could not reject pending {subsystem} write '{target}': {r['reason']}"


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
