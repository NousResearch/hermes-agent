"""Agent protocol run state (https://github.com/phrac/fluxchat/blob/main/docs/agent-protocol.md).

Adapters that declare ``AGENT_STATUS = True`` (Matrix) receive a ``fluxchat_agent`` dict in the
``metadata`` of the turn's tool-progress bubble (``state: "working"`` with the latest step and the
step count) and of its final reply (``done`` / ``failed`` with model, context use and the session's
spend), and put it in the message content so clients can show the run without decoding reactions.
Everything here is best effort: a missing field is left out, never guessed.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Progress lines are ``{emoji} {text}``; the step is the text.
_LEADING_EMOJI = re.compile(r"^\W+\s+")
_DEDUP_SUFFIX = re.compile(r"\s*\(×\d+\)$")
_STEP_MAX_CHARS = 120


def supported(adapter: Any) -> bool:
    return bool(getattr(adapter, "AGENT_STATUS", False))


def step_text(line: Any) -> str:
    """The step a progress line describes: its text without the tool emoji or the repeat count."""
    first = str(line or "").strip().splitlines()[0] if str(line or "").strip() else ""
    text = _DEDUP_SUFFIX.sub("", _LEADING_EMOJI.sub("", first, count=1)).strip()
    return text[:_STEP_MAX_CHARS]


def working(run: Optional[str], step: Any, steps: int) -> dict:
    status: dict = {"v": 1, "state": "working"}
    if run:
        status["run"] = str(run)
    if text := step_text(step):
        status["step"] = text
    if steps > 0:
        status["steps"] = steps
    return status


def _result_line(response: Any) -> str:
    """The reply's first non-empty line, trimmed to one result line (protocol ``result``)."""
    text = str(response or "").strip()
    for line in text.splitlines():
        if line := line.strip():
            # Heading/list markers lead; bold markers wrap the lead-in ("**Done.**").
            line = line.lstrip("#-> ").lstrip()
            if line.startswith("**") and line.endswith("**"):
                line = line[2:-2].strip()
            return line[:240]
    return ""


def finished(run: Optional[str], agent_result: Any, failed: bool, response: Any = None) -> dict:
    """``done`` / ``failed`` plus what the turn's result knows: model, context use, session spend.

    ``response`` is the reply text; its first line (no more) becomes ``result``, the run's
    one-line outcome (protocol docs: written for readers of run lists, not the whole reply).
    """
    status: dict = {"v": 1, "state": "failed" if failed else "done"}
    if run:
        status["run"] = str(run)
    if line := _result_line(response):
        status["result"] = line
    result = agent_result if isinstance(agent_result, dict) else {}
    if model := str(result.get("model") or "").strip():
        status["model"] = model
    tokens, length = result.get("last_prompt_tokens") or 0, result.get("context_length") or 0
    if isinstance(tokens, (int, float)) and isinstance(length, (int, float)) and tokens > 0 and length > 0:
        status["context"] = round(min(1.0, tokens / length), 3)
    cost = result.get("estimated_cost_usd")
    if isinstance(cost, (int, float)) and cost >= 0 and result.get("cost_status") != "unknown":
        status["cost_usd"] = round(float(cost), 6)
    return status


def with_status(metadata: Optional[dict], status: Optional[dict]) -> Optional[dict]:
    """A copy of ``metadata`` carrying ``status``; ``metadata`` itself is left alone (it's shared)."""
    if not status:
        return metadata
    merged = dict(metadata or {})
    merged["fluxchat_agent"] = status
    return merged
