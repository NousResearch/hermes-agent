"""Outgoing-claim evidence gate for chat surfaces. Read-only over the finished
turn: when the final message asserts an outcome ("verified", "fixed", "tested",
"deployed", ...) but the turn produced no tool evidence (no tool results at all),
it annotates the message with a visible caveat instead of letting an unbacked
assertion read exactly like a proven one. Policy-pure like the verification
ledger (#124657): it never runs anything and never blocks delivery."""

from __future__ import annotations

import re
from typing import Any

_CLAIM_GATE_CAVEAT = (
    "\n\n> ⚠️ Claim gate: this message states an outcome, but this turn produced "
    "no tool output backing it — the claim is unverified."
)

# Outcome language: a finite list of verbs/adjectives asserting a completed,
# checkable result. Kept narrow deliberately — broad patterns ("works", "done")
# would annotate ordinary prose on nearly every turn. Word-bounded,
# case-insensitive.
_CLAIM_PATTERN = re.compile(
    r"\b("
    r"verified|unverified|confirmed|validated|tested|fixed|resolved|deployed|"
    r"rebuilt|passed|failing|passing"
    r")\b",
    re.IGNORECASE,
)

# A negation directly before the claim ("not verified", "never tested") reports
# the absence of an outcome and must not be annotated.
_NEGATION_PREFIX = re.compile(r"\b(not|never)\s+$", re.IGNORECASE)

_FALSY_MODES = frozenset({"off", "disabled", "none"})


def _contains_outcome_claim(text: str) -> bool:
    """True when the message asserts a checkable outcome in a non-negated way."""
    for match in _CLAIM_PATTERN.finditer(text):
        prefix = text[max(0, match.start() - 12):match.start()]
        if _NEGATION_PREFIX.search(prefix):
            continue
        return True
    return False


def _turn_has_tool_evidence(messages: list[dict[str, Any]] | None) -> bool:
    """True when the finished turn contains at least one tool result. Any tool
    output counts as evidence for annotation purposes — the caveat only fires
    when the turn is all prose and no tool calls ran at all."""
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return True
            if content not in (None, ""):
                return True
    return False


def claim_gate_mode(config: dict[str, Any] | None = None) -> str:
    """Resolved ``agent.claim_gate`` mode: ``off`` | ``annotate``.

    Default ``annotate`` — the point of #124657 is that on chat surfaces an
    unbacked outcome claim is indistinguishable from a proven one, so a visible
    caveat (never a block, never a rewrite) is the safe default. ``off``
    disables the gate entirely (programmatic callers, tests). A bool maps
    true→annotate, false→off. Unrecognized values fall back to ``annotate``.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    value = agent_cfg.get("claim_gate") if isinstance(agent_cfg, dict) else None
    if isinstance(value, bool):
        return "annotate" if value else "off"
    token = str(value).strip().lower() if value is not None else ""
    if token in _FALSY_MODES:
        return "off"
    if token == "annotate":
        return "annotate"
    return "annotate"


def apply_claim_gate(
    response: str,
    messages: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Annotate an outcome-asserting final message that the turn's tool results
    do not back. Returns the response unchanged when the gate is off, the
    message carries no outcome language, or any tool result exists in the turn
    (tool calls ran — the ledger/verify-on-stop layer owns their quality)."""
    if not response or claim_gate_mode(config) != "annotate":
        return response
    if not _contains_outcome_claim(response):
        return response
    if _turn_has_tool_evidence(messages):
        return response
    return response + _CLAIM_GATE_CAVEAT
