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
import logging
from typing import List, Optional

from tools import write_approval as wa


logger = logging.getLogger(__name__)


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    return f"{subsystem}.write_approval = {'on' if on else 'off'}"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pending_list(subsystem: str) -> str:
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
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


def _fmt_receipt_list(subsystem: str) -> str:
    """Format recent immutable decisions without exposing proposal content."""
    try:
        from agent.verification_evidence import list_approval_decision_receipts

        records = list_approval_decision_receipts(subsystem=subsystem, limit=20)
    except Exception:
        logger.exception("Failed to read %s approval decision receipts", subsystem)
        return f"{subsystem} approval receipt history is unavailable."

    if not records:
        return f"No terminal {subsystem} decision receipts."

    lines = [f"Recent {subsystem} terminal decision receipts ({len(records)}):"]
    for receipt in records:
        failure = receipt.get("failure_code")
        suffix = f" ({failure})" if failure else ""
        lines.append(
            "  {id}: {pending_id} {decision}/{outcome} [{origin}] {recorded_at}{suffix}".format(
                id=receipt["id"],
                pending_id=receipt["pending_id"],
                decision=receipt["decision"],
                outcome=receipt["terminal_outcome"],
                origin=receipt["proposal_origin"],
                recorded_at=receipt["recorded_at"],
                suffix=suffix,
            )
        )
    lines.append("")
    lines.append("Read-only audit history; terminal decisions are never replayed from this view.")
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

    if sub == "pending":
        return _fmt_pending_list(subsystem)

    if sub in {"receipt", "receipts", "history"}:
        return _fmt_receipt_list(subsystem)

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


def _record_terminal_receipt(rec, *, decision: str, terminal_outcome: str):
    """Persist one immutable audit receipt for an already-terminal claim.

    The receipt database is deliberately separate from outcome-learning
    candidates.  A failure is returned to the caller so it can retain the
    private claim for manual reconciliation instead of replaying a mutation.
    """
    try:
        from agent.verification_evidence import record_approval_decision_receipt

        return record_approval_decision_receipt(
            record=rec,
            decision=decision,
            terminal_outcome=terminal_outcome,
        )
    except Exception:
        logger.exception("Failed to record terminal %s receipt", rec.get("subsystem"))
        return None


def _held_claim_message(pending_id: str, claim_path) -> str:
    """Describe the non-replayable manual-recovery state after receipt failure."""
    return (
        f"{pending_id}: terminal decision is final and must not be reapplied; "
        f"approval receipt was not recorded. Held non-actionable claim: {claim_path}. "
        "Manual reconciliation is required."
    )


def _approve(subsystem: str, rest: List[str], memory_store) -> str:
    target, err = _resolve_one(subsystem, rest)
    if err or target is None:
        return err or f"Usage: /{subsystem} approve <id>"

    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."

    if target.lower() == "all":
        targets = [str(record.get("id", "")) for record in records]
    else:
        if not wa.valid_pending_id(target) or not wa.get_pending(subsystem, target):
            return f"No pending {subsystem} write with id '{target}'."
        targets = [target]

    applied, failed = 0, []
    for pending_id in targets:
        claimed = wa.claim_pending(subsystem, pending_id)
        if claimed is None:
            failed.append(f"{pending_id}: no longer pending or already being processed")
            continue
        rec, claim_path = claimed
        apply_result = _apply_one(subsystem, rec, memory_store)
        ok, msg = apply_result[:2]
        terminal = bool(apply_result[2]) if len(apply_result) > 2 else False
        if ok:
            applied += 1
            if _record_terminal_receipt(
                rec, decision="approved", terminal_outcome="applied"
            ) is None:
                failed.append(_held_claim_message(pending_id, claim_path))
                continue
            if not wa.complete_claim(claim_path):
                failed.append(f"{pending_id}: applied; cleanup is pending and will not be replayed")
        elif terminal:
            if _record_terminal_receipt(
                rec, decision="approved", terminal_outcome="terminal_noop"
            ) is None:
                failed.append(_held_claim_message(pending_id, claim_path))
                continue
            if wa.complete_claim(claim_path):
                failed.append(f"{pending_id}: {msg}")
            else:
                failed.append(
                    f"{pending_id}: {msg}; terminal cleanup is pending and requires manual recovery"
                )
        else:
            release_result = wa.release_claim(subsystem, pending_id, claim_path)
            if release_result is False:
                failed.append(f"{pending_id}: {msg}; retry is held for manual recovery")
            elif release_result is None:
                failed.append(
                    f"{pending_id}: {msg}; retry remains available, but stale claim cleanup needs manual recovery"
                )
            else:
                failed.append(f"{pending_id}: {msg}")

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
                return False, "memory store unavailable", False
            from tools.memory_tool import apply_memory_pending_record
            result = apply_memory_pending_record(rec, memory_store)
            return (
                bool(result.get("success")),
                result.get("error", ""),
                bool(result.get("terminal")),
            )
        else:
            from tools.skill_manager_tool import apply_skill_pending
            result = json.loads(apply_skill_pending(payload, origin=rec.get("origin")))
            return bool(result.get("success")), result.get("error", ""), False
    except Exception as e:
        return False, str(e), False


def _reject(subsystem: str, rest: List[str]) -> str:
    target, err = _resolve_one(subsystem, rest)
    if err or target is None:
        return err or f"Usage: /{subsystem} reject <id>"
    if target.lower() == "all":
        n, failed = 0, []
        for rec in wa.list_pending(subsystem):
            pending_id = str(rec.get("id", ""))
            claimed = wa.claim_pending(subsystem, pending_id)
            if claimed is None:
                continue
            rec, claim_path = claimed
            if _record_terminal_receipt(
                rec, decision="rejected", terminal_outcome="rejected"
            ) is None:
                failed.append(_held_claim_message(pending_id, claim_path))
            elif wa.complete_claim(claim_path):
                n += 1
            else:
                failed.append(f"{pending_id}: cleanup is pending and requires manual recovery")
        out = [f"Rejected {n} pending {subsystem} write(s)."]
        if failed:
            out.append("Failed:")
            out.extend(f"  {item}" for item in failed)
        return "\n".join(out)
    claimed = wa.claim_pending(subsystem, target)
    if claimed is not None:
        rec, claim_path = claimed
        if _record_terminal_receipt(
            rec, decision="rejected", terminal_outcome="rejected"
        ) is None:
            return _held_claim_message(target, claim_path)
        if wa.complete_claim(claim_path):
            return f"Rejected pending {subsystem} write '{target}'."
        return (
            f"Rejected pending {subsystem} write '{target}', but cleanup is pending "
            "and requires manual recovery."
        )
    return f"No pending {subsystem} write with id '{target}'."


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
