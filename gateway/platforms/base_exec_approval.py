"""Shared wording for the exec-approval prompt every messaging surface renders.

The button card (``BasePlatformAdapter._format_exec_approval``) and the plain-text
``/approve`` fallback (``gateway.run._format_exec_approval_fallback``) must tell the user
the same three things: what Hermes wants to run, why it was flagged, and that silence means
the command does NOT run once ``approvals.timeout`` elapses. Keeping the text here means one
edit changes every platform; adapters only wrap these strings in their own markup.

The ``EA_*`` / ``APPROVAL_TIMED_OUT_NOTICE`` module constants are the English source of truth
(catalog keys ``gateway.exec_approval.*``); anything that renders the prompt calls the sibling
accessors (``ea_header_text()`` …) so the active ``display.language`` is honoured at render time.

No imports from ``gateway.platforms.base`` or ``gateway.run`` — both import this module.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.i18n import t

# Bare strings; adapters add their own bold/HTML around them.
EA_HEADER_TEXT = "Hermes wants to run a command that needs your OK"
EA_REASON_LABEL_TEXT = "Why it was flagged"

# Timeout notice posted when nobody answered the prompt (``{window}`` = "5 minutes").
APPROVAL_TIMED_OUT_NOTICE = (
    "⌛ Approval timed out after {window} — the command was NOT run. "
    "Ask me to try again if you still want it, or raise approvals.timeout in config.yaml.")


def ea_header_text() -> str:
    """Localized ``EA_HEADER_TEXT``."""
    return t("gateway.exec_approval.header")


def ea_reason_label_text() -> str:
    """Localized ``EA_REASON_LABEL_TEXT``."""
    return t("gateway.exec_approval.reason_label")


def ea_smart_deny_line_text() -> str:
    """Localized smart-deny sentence (bare; adapters add their own line breaks/markup)."""
    return t("gateway.exec_approval.smart_deny_line")


def ea_default_reason_text() -> str:
    """Localized default "why it was flagged" reason when the approval carries none."""
    return t("gateway.exec_approval.default_reason")


def ea_action_labels() -> dict[str, str]:
    """Localized ``choice -> button label`` for the four approval tiers."""
    return {
        "once": t("gateway.exec_approval.action_once"),
        "session": t("gateway.exec_approval.action_session"),
        "always": t("gateway.exec_approval.action_always"),
        "deny": t("gateway.exec_approval.action_deny"),
    }


def approval_timed_out_notice(window: str) -> str:
    """Localized ``APPROVAL_TIMED_OUT_NOTICE`` for an already-formatted ``window``."""
    return t("gateway.exec_approval.timed_out_notice", window=window)


def approval_timeout_seconds() -> int:
    """The configured ``approvals.timeout`` (default 300s); module attribute so tests can pin it."""
    from tools.approval_context import _get_approval_timeout
    return _get_approval_timeout()


def format_approval_window(seconds: int) -> str:
    """Human wording for a timeout (300 → "5 minutes"); one formatter shared with the CLI notice and
    the tool result's ``user_summary`` — see ``tools.approval_context.format_approval_window``."""
    from tools.approval_context import format_approval_window as _shared
    return _shared(seconds)


def format_approval_deadline_line(timeout_s: int) -> str:
    """The last line of every approval prompt: doing nothing is a safe no."""
    return t("gateway.exec_approval.deadline", window=format_approval_window(timeout_s))


def format_approval_timed_out_notice(timeout_s: int) -> str:
    return approval_timed_out_notice(format_approval_window(timeout_s))


def render_exec_approval(adapter: Any, command: str, description: str, smart_denied: bool,
                         full_command_shown: bool) -> tuple[str, bool]:
    """``(card text, truncated)`` for ``adapter``'s markup. A preview that cannot hold the whole command
    says so in the card: where the full command was sent just before it (``full_command_shown``), or
    that only Deny is offered because it could not be (#62359)."""
    if adapter._EA_REASON_BUDGET:
        description = adapter._ea_fit(str(description or ""), adapter._EA_REASON_BUDGET)
    budget = adapter._exec_approval_cmd_budget(description, smart_denied)
    preview = adapter._ea_fit(command, budget)
    truncated = preview != command
    notice = ""
    if truncated:
        key = "truncated_full_above" if full_command_shown else "truncated_deny_only"
        notice = "\n\n" + adapter._ea_escape(t(f"gateway.exec_approval.{key}", total=len(command)))
        # The notice shares the cap the budget was sized for; a tiny fixed budget keeps half its preview.
        preview = adapter._ea_fit(command, max(budget - adapter.message_len_fn(notice), budget // 2))
    text = (f"{adapter._EA_HEADER}"
            f"{adapter._EA_CODE_OPEN}{adapter._ea_escape(preview)}{adapter._EA_CODE_CLOSE}"
            f"{adapter._EA_REASON_LABEL}{adapter._ea_escape(description)}{notice}"
            f"{adapter._ea_deadline_line()}")
    return (text + adapter._EA_SMART_DENY_LINE if smart_denied else text), truncated


async def send_full_command(adapter: Any, chat_id: str, command: str, metadata: Optional[dict]) -> bool:
    """Send the whole command as a fenced block (the adapter's ``send`` splits it to fit) before a
    card whose preview is truncated. False when it cannot be shown verbatim and whole, or a chunk failed."""
    if "```" in command:  # it would close the fence early and the rest would render as markdown
        return False
    content = f"```\n{command}\n```"
    # ``send`` reports success even when a chunk cap (Discord's) swapped the tail for a notice.
    cap = getattr(adapter, "MAX_SPLIT_MESSAGES", 0)
    if cap and len(adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH)) > cap:
        return False
    # A mid-turn send: unmarked, a stream-is-the-message adapter seals the live response with it.
    result = await adapter.send(chat_id, content, metadata={**(metadata or {}), "_interim_send": True})
    raw = getattr(result, "raw_response", None)
    clipped = isinstance(raw, dict) and bool(raw.get("warnings") or raw.get("partial_overflow"))
    return bool(getattr(result, "success", False)) and not clipped
