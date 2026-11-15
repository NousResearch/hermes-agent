#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands."""

from __future__ import annotations

import json
from typing import List, Optional

from tools import write_approval as wa


def _fmt_state(subsystem: str) -> str:
    if not wa.write_approval_enabled(subsystem):
        return f"{subsystem}.write_approval = off"
    return f"{subsystem}.write_approval = on ({_scope_label(wa.write_approval_scope(subsystem))})"


def _scope_label(scope: str) -> str:
    return "background writes only; your own turns apply directly" if scope == wa.SCOPE_BACKGROUND else "all writes"


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
    subsystem: str, args: list[str], *, memory_store=None, set_mode_fn=None) -> Optional[str]:
    """Dispatch a /memory or /skills write-approval subcommand.

    ``memory_store`` applies approved memory writes (CLI passes its live store; gateway a freshly
    loaded one); ``set_mode_fn(settings)`` persists exactly the ``{key: value}`` settings it is given
    (``write_approval`` and, when turning the gate on, ``write_approval_scope``). Returns text for the user,
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
            result = json.loads(apply_skill_pending(payload, origin=rec.get("origin", "foreground")))
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


# Settings per word, scope first: a setter that persists key by key and fails half way leaves the
# old gate with the new scope, never a gate narrower than both the old and the requested state.
_ON = {wa.SCOPE_KEY: wa.SCOPE_ALL, wa.CONFIG_KEY: True}
_BACKGROUND = {wa.SCOPE_KEY: wa.SCOPE_BACKGROUND, wa.CONFIG_KEY: True}
_OFF = {wa.CONFIG_KEY: False}  # the scope is kept for the next "on"
_APPROVAL_SETTINGS = {
    **dict.fromkeys(("on", "true", "yes", "1", "enable", "enabled"), _ON),
    **dict.fromkeys(wa.BACKGROUND_SCOPE_WORDS, _BACKGROUND),
    **dict.fromkeys(("off", "false", "no", "0", "disable", "disabled"), _OFF)}


def settings_saver(save_fn, subsystem: str):
    """A ``set_mode_fn`` over a per-key ``save_fn(dotted_key, value) -> bool`` (the classic CLI's
    config writer): saves the settings in the order given and raises on the first False, so the
    command reports the change as not saved instead of confirming it."""
    def _save_settings(settings: dict) -> None:
        for key, value in settings.items():
            if not save_fn(f"{subsystem}.{key}", value):
                raise RuntimeError(f"could not save {subsystem}.{key} to config.yaml")
    return _save_settings


def _set_approval(subsystem: str, rest: list[str], set_mode_fn) -> str:
    """Turn the approval gate on (every write), on for background writes only, or off."""
    if not rest:
        return (f"{_fmt_state(subsystem)}\n"
                f"Set with: /{subsystem} approval <on|off|background>")
    arg = rest[0].strip().lower()
    settings = _APPROVAL_SETTINGS.get(arg)
    if settings is None:
        return (f"Invalid value '{arg}'. Use: on, off, or background "
                "(gate only background writes; your own turns apply directly).")
    if set_mode_fn is None:
        commands = "\n".join(f"  hermes config set {subsystem}.{key} {str(value).lower()}"
                             for key, value in settings.items())
        return f"To change the {subsystem} approval gate, run:\n{commands}"
    try:
        set_mode_fn(dict(settings))
    except Exception as e:
        return f"Failed to set {subsystem}.write_approval: {e}"
    if not settings[wa.CONFIG_KEY]:
        return f"{subsystem}.write_approval set to 'off'."
    return f"{subsystem}.write_approval set to 'on' ({_scope_label(settings[wa.SCOPE_KEY])})."
