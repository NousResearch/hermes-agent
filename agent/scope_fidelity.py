"""Requirement-to-evidence gate for completion claims.

Prompt-only "finish the job" guidance is not enough: a model can still
replace an explicit operational mandate with a narrower substitute and
describe that substitute as complete. This module is the machine half of
the fix.

It never calls a model. It:
- extracts an explicit or implicit requirement contract from the user turn
- parses an optional ``<delivery_receipt>`` in the assistant reply
- qualifies completion language when the receipt is missing or incomplete

The contract is stored on the agent so it survives context compaction
rebuilds in the same session.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


SCOPE_FIDELITY_GUIDANCE = (
    "# Scope fidelity\n"
    "When the user states an explicit operational mandate or numbered "
    "acceptance criteria, that list is the job. Do not silently replace it "
    "with a smaller, offline, read-only, routing-DISABLED, or demonstration "
    "substitute and then describe the substitute as finished.\n"
    "A hard boundary on personally submitting a live order must not block "
    "permitted work: adapters, live/fresh data, signal engines, authenticated "
    "reads, operator-controlled order paths, or reconciliation. Say the exact "
    "blocked action; continue everything else.\n"
    "Do not attribute a host, VM, desktop, or VNC endpoint to another session "
    "without independently verified identity.\n"
    "Before any completion claim (finished / complete / ready / successfully "
    "built), emit a receipt with every criterion marked PROVEN, MISSING, or "
    "BLOCKED plus concrete evidence. Completion language is forbidden while "
    "any criterion is MISSING or BLOCKED. Partial components, tests, or a GUI "
    "startup are not evidence for a broader deliverable.\n"
    "<delivery_receipt>\n"
    "1. <criterion> — PROVEN|MISSING|BLOCKED — <evidence>\n"
    "</delivery_receipt>"
)

SCOPE_FIDELITY_FOOTER = (
    "\n\n---\n"
    "**Scope-fidelity note (automatic):** Completion language was used, but a "
    "requirement-to-evidence receipt is missing or incomplete. Treat the work "
    "above as a **partial delivery**, not a finished operational system.\n"
    "{status_lines}"
    "Allowed statuses are PROVEN, MISSING, or BLOCKED with concrete evidence. "
    "A signal-only, offline, or routing-DISABLED component is not the requested "
    "operational path."
)

_COMPLETION_CLAIM_RE = re.compile(
    r"(?is)(?:"
    r"\b(?:successfully|now)\s+(?:created|implemented|built|completed)\b|"
    r"\b(?:setup|system|build|job|task|work|engine|path)\s+"
    r"(?:has been\s+)?(?:successfully\s+)?(?:created|completed|finished)\b|"
    r"\b(?:is|are)\s+(?:now\s+)?(?:complete|completed|finished|done)\b|"
    r"\bcompleted and ready\b|"
    r"\bready for use\b|"
    r"\bthe (?:operational\s+)?(?:system|build)\s+is\s+completed\b|"
    r"\b(?:all (?:acceptance )?criteria (?:are|were) (?:proven|met))\b"
    r")"
)

_RECEIPT_RE = re.compile(
    r"<delivery_receipt>(?P<body>.*?)</delivery_receipt>",
    re.IGNORECASE | re.DOTALL,
)

_STATUS_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*]|\d+[.)])\s+(?P<body>.+)$"
)

_STATUS_TOKEN_RE = re.compile(r"\b(?P<status>PROVEN|MISSING|BLOCKED)\b")

_CRITERION_LINE_RE = re.compile(
    r"(?m)^\s*(?:[-*]|\d+[.)])\s+(?P<text>\S.*)$"
)

_OPERATIONAL_MARKERS = (
    "operational",
    "live/fresh",
    "fresh market",
    "live market",
    "order path",
    "routing",
    "read-only",
    "inert",
    "offline",
    "do not silently replace",
    "do not replace",
    "not a shortcut",
    "acceptance criteria",
)

_STATUSES = frozenset({"PROVEN", "MISSING", "BLOCKED"})


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def latest_user_text(messages: Optional[Iterable[Any]]) -> str:
    if not messages:
        return ""
    for message in reversed(list(messages)):
        if isinstance(message, dict) and message.get("role") == "user":
            if message.get("_verification_stop_synthetic") or message.get(
                "_kanban_stop_synthetic"
            ):
                continue
            return _message_text(message).strip()
    return ""


def extract_criteria(user_text: str) -> list[str]:
    """Return explicit numbered/bulleted criteria, or implicit operational ones."""
    text = (user_text or "").strip()
    if not text:
        return []

    listed: list[str] = []
    for match in _CRITERION_LINE_RE.finditer(text):
        line = match.group("text").strip()
        if len(line) < 8:
            continue
        listed.append(line)
    if listed:
        return listed[:12]

    lowered = text.lower()
    hits = sum(1 for marker in _OPERATIONAL_MARKERS if marker in lowered)
    if hits >= 2:
        return [
            "Live/fresh market data (not an offline mockup)",
            "Strategy signal wired into the requested operational path",
            "Operator-controlled order/routing path present and not DISABLED",
            "No silent substitute presented as the requested deliverable",
        ]
    return []


def parse_receipt_statuses(assistant_text: str) -> list[tuple[str, str]]:
    """Return (criterion, status) pairs from a receipt or status-tagged lines."""
    text = assistant_text or ""
    receipt = _RECEIPT_RE.search(text)
    body = receipt.group("body") if receipt else text
    found: list[tuple[str, str]] = []
    for match in _STATUS_LINE_RE.finditer(body):
        line = match.group("body").strip()
        status_match = _STATUS_TOKEN_RE.search(line)
        if not status_match:
            continue
        status = status_match.group("status").upper()
        criterion = _STATUS_TOKEN_RE.sub("", line)
        criterion = re.sub(r"\s+[—-]\s+", " ", criterion)
        criterion = re.sub(r"\s+", " ", criterion).strip(" :.-")
        found.append((criterion or "criterion", status))
    return found


def has_completion_claim(assistant_text: str) -> bool:
    return bool(_COMPLETION_CLAIM_RE.search(assistant_text or ""))


def receipt_covers_contract(
    statuses: list[tuple[str, str]],
    criteria: list[str],
) -> bool:
    if not criteria:
        return True
    if len(statuses) < len(criteria):
        return False
    return all(status == "PROVEN" for _criterion, status in statuses[: len(criteria)])


def format_status_lines(
    criteria: list[str],
    statuses: list[tuple[str, str]],
) -> str:
    by_index = {i: statuses[i] for i in range(min(len(criteria), len(statuses)))}
    lines: list[str] = []
    for i, criterion in enumerate(criteria):
        if i in by_index:
            _label, status = by_index[i]
        else:
            status = "MISSING"
        short = criterion if len(criterion) <= 120 else criterion[:117] + "..."
        lines.append(f"- {short} — {status}\n")
    return "".join(lines)


def render_contract_prompt(criteria: list[str]) -> str:
    if not criteria:
        return ""
    numbered = "\n".join(f"{i}. {item}" for i, item in enumerate(criteria, start=1))
    return (
        "# Active requirement contract\n"
        "These acceptance criteria remain in force across compaction. "
        "Do not claim completion unless every item is PROVEN.\n"
        f"{numbered}"
    )


def persist_contract_journal(
    agent: Any,
    criteria: list[str],
    statuses: list[tuple[str, str]],
    *,
    qualified_partial: bool,
) -> Path | None:
    """Write a redacted local receipt next to the session (no secrets)."""
    session_id = getattr(agent, "session_id", None)
    if not agent or not session_id or not criteria:
        return None
    try:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "sessions" / str(session_id) / "scope_fidelity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": str(session_id),
            "criteria": list(criteria),
            "statuses": [
                {"criterion": label, "status": status} for label, status in statuses
            ],
            "qualified_partial": bool(qualified_partial),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path
    except Exception:
        logger.debug("scope-fidelity journal write failed", exc_info=True)
        return None


def apply_scope_fidelity(
    assistant_text: str,
    messages: Optional[Iterable[Any]] = None,
    *,
    agent: Any = None,
    enabled: Optional[bool] = None,
) -> str:
    """Qualify a completion claim that lacks a complete evidence receipt."""
    if enabled is None:
        enabled = True if agent is None else bool(getattr(agent, "_scope_fidelity", True))
    if not enabled:
        return assistant_text

    text = assistant_text or ""
    user_text = latest_user_text(messages)
    criteria = extract_criteria(user_text)
    if agent is not None:
        stored = getattr(agent, "_scope_fidelity_contract", None)
        if criteria:
            agent._scope_fidelity_contract = list(criteria)
        elif stored:
            criteria = list(stored)

    if not criteria or not has_completion_claim(text):
        return text

    statuses = parse_receipt_statuses(text)
    covered = receipt_covers_contract(statuses, criteria)
    persist_contract_journal(
        agent, criteria, statuses, qualified_partial=not covered
    )
    if covered:
        return text
    if SCOPE_FIDELITY_FOOTER.split("{status_lines}", 1)[0].strip() in text:
        return text

    footer = SCOPE_FIDELITY_FOOTER.format(
        status_lines=format_status_lines(criteria, statuses)
    )
    return text.rstrip() + footer


def looks_like_operational_mandate(user_text: str) -> bool:
    lowered = (user_text or "").lower()
    return sum(1 for marker in _OPERATIONAL_MARKERS if marker in lowered) >= 2


def parse_receipt(assistant_text: str) -> list[tuple[str, str]]:
    """Alias used by tests: (item, status)."""
    return parse_receipt_statuses(assistant_text)


def qualify_final_response(
    assistant_text: str,
    messages: Optional[Iterable[Any]] = None,
    *,
    enabled: bool = True,
    user_text: Optional[str] = None,
) -> str:
    """Test-facing wrapper around apply_scope_fidelity."""
    if user_text is not None:
        messages = list(messages or []) + [{"role": "user", "content": user_text}]
    return apply_scope_fidelity(assistant_text, messages, enabled=enabled)
