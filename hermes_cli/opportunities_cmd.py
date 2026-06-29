"""Shared `/opportunities` command logic for CLI, TUI, and gateway."""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OpportunityCommandResult:
    """Result of an opportunities command.

    `text` is shown immediately. When `agent_seed` is set, the calling surface
    should submit it as the next normal user turn.
    """

    text: str
    agent_seed: Optional[str] = None


def _save_proactive_enabled(enabled: bool) -> bool:
    try:
        from cli import save_config_value

        return bool(save_config_value("proactive.enabled", bool(enabled)))
    except Exception as exc:
        logger.debug("failed to persist proactive.enabled=%s: %s", enabled, exc)
        return False


def _fmt_status(store) -> str:
    state = "on" if store.is_enabled() else "off"
    meta = store.load_meta()
    last = meta.get("last_usage_scan_at") or "never"
    pending = len(store.list_pending())
    return (
        f"Proactive opportunities: {state}\n"
        f"Pending: {pending}\n"
        f"Last usage scan: {last}\n"
        "Controls: /opportunities enable | disable | scan | seed | accept N | dismiss N"
    )


def _fmt_pending(store) -> str:
    pending = store.list_pending()
    if not pending:
        hint = "Run /opportunities seed for starter ideas or /opportunities scan to inspect recent usage."
        if not store.is_enabled():
            hint += "\nProactive scanning is off. Enable it with /opportunities enable."
        return "No pending opportunities right now.\n" + hint

    lines = ["Opportunities - /opportunities accept N or dismiss N:\n"]
    for idx, record in enumerate(pending, 1):
        action = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
        action_type = action.get("type", "?")
        source = record.get("source", "?")
        lines.append(f"  {idx}. {record.get('title', '(untitled)')}  ({action_type}, {source})")
        desc = str(record.get("description") or "").strip()
        if desc:
            lines.append(f"     {desc}")
        evidence = [str(e).strip() for e in record.get("evidence") or [] if str(e).strip()]
        for item in evidence[:2]:
            lines.append(f"     - {item}")
    return "\n".join(lines)


def _usage() -> str:
    return (
        "Usage:\n"
        "  /opportunities                 list pending opportunities\n"
        "  /opportunities status          show mode and scan state\n"
        "  /opportunities enable|disable  toggle opt-in proactive scanning\n"
        "  /opportunities scan            scan recent chats now\n"
        "  /opportunities seed            add starter proactive ideas\n"
        "  /opportunities accept N        accept and run opportunity N\n"
        "  /opportunities dismiss N       dismiss opportunity N\n"
        "  /opportunities clear           prune accepted records"
    )


def handle_opportunities_command(args: str, *, surface: str = "cli") -> OpportunityCommandResult:
    """Dispatch `/opportunities`.

    The command manages a generic opportunity inbox. Accepting most
    opportunities returns an agent seed instead of doing the work inline, so
    the normal agent/tool/approval paths remain in charge.
    """
    try:
        from agent import opportunities as store
    except Exception as exc:  # pragma: no cover - import guard
        logger.debug("opportunities store import failed: %s", exc)
        return OpportunityCommandResult("Opportunities are unavailable in this build.")

    try:
        parts = shlex.split(args or "")
    except ValueError:
        parts = (args or "").split()

    sub = parts[0].lower() if parts else ""
    rest = " ".join(parts[1:]).strip()

    if not sub or sub in {"list", "ls"}:
        return OpportunityCommandResult(_fmt_pending(store))

    if sub == "status":
        return OpportunityCommandResult(_fmt_status(store))

    if sub in {"enable", "on"}:
        ok = _save_proactive_enabled(True)
        if ok:
            return OpportunityCommandResult(
                "Proactive opportunities enabled. Hermes will periodically scan recent chats "
                "for repeated work and add consent-first proposals to /opportunities."
            )
        return OpportunityCommandResult("Could not enable proactive opportunities.")

    if sub in {"disable", "off"}:
        ok = _save_proactive_enabled(False)
        if ok:
            return OpportunityCommandResult("Proactive opportunities disabled. Existing pending items are unchanged.")
        return OpportunityCommandResult("Could not disable proactive opportunities.")

    if sub in {"seed", "starter", "starters"}:
        try:
            created = store.seed_starter_opportunities()
        except Exception as exc:
            logger.debug("starter opportunity seed failed: %s", exc)
            return OpportunityCommandResult(f"Could not seed starter opportunities: {exc}")
        if not created:
            return OpportunityCommandResult(
                "No new starter opportunities to add; they were already offered or the list is full."
            )
        names = ", ".join(c.get("title", "?") for c in created)
        return OpportunityCommandResult(f"Added {len(created)} starter opportunity(s): {names}.\nRun /opportunities to review.")

    if sub in {"scan", "review"}:
        result = store.scan_recent_usage(force=True)
        created = result.get("created") or []
        if not result.get("scanned"):
            return OpportunityCommandResult(f"Scan skipped: {result.get('reason', 'unknown')}.")
        if not created:
            reason = result.get("reason", "ok")
            return OpportunityCommandResult(
                f"Scanned recent chats; no new opportunities found ({reason})."
            )
        names = ", ".join(c.get("title", "?") for c in created)
        return OpportunityCommandResult(f"Added {len(created)} opportunity(s): {names}.\nRun /opportunities to review.")

    if sub in {"accept", "run", "do"}:
        if not rest:
            return OpportunityCommandResult("Usage: /opportunities accept <number|id>")
        accepted = store.accept_opportunity(rest)
        if accepted is None:
            return OpportunityCommandResult(f"No pending opportunity matches '{rest}'. Run /opportunities to list them.")
        notice = accepted.get("notice") or "Accepted opportunity."
        seed = accepted.get("message")
        if seed:
            return OpportunityCommandResult(f"{notice}\nStarting it as a normal Hermes turn.", agent_seed=seed)
        return OpportunityCommandResult(notice)

    if sub in {"dismiss", "reject", "no"}:
        if not rest:
            return OpportunityCommandResult("Usage: /opportunities dismiss <number|id>")
        ok = store.dismiss_opportunity(rest)
        return OpportunityCommandResult(
            "Dismissed. I won't suggest that opportunity again."
            if ok
            else f"No pending opportunity matches '{rest}'."
        )

    if sub == "clear":
        removed = store.clear_accepted()
        return OpportunityCommandResult(f"Cleared {removed} accepted opportunity record(s).")

    return OpportunityCommandResult(_usage())
