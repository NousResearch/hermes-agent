"""Shared LLM JSON parsing — single helper for all call sites.

Four modules previously had independent JSON parsers with near-identical
logic (fence-stripping, first-{ to last-}, direct parse).  This module
is the single source.  Also hosts ``_profile_author`` to break the
circular import between ``kanban.py`` ↔ ``kanban_specify.py`` /
``kanban_decompose.py``.

JSON-1 + I-3 (2026-06-12 kanban/orchestration hardening).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Thinking-mode traces: some providers (and local fallback models) wrap their
# answer in <｜end▁of▁thinking｜> blocks when reasoning output is enabled. Strip them
# BEFORE any JSON strategy so the real JSON payload is reachable.
_THINK_BLOCK_RE = re.compile(
    r"<\s*(?:think|thinking|reasoning|analysis)\s*>.*?<\s*/\s*(?:think|thinking|reasoning|analysis)\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_think_blocks(text: str) -> str:
    """Remove `` (and variants) reasoning traces from an LLM response."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # A dangling opening tag (truncated response) — drop everything before the
    # first closing tag if one exists, else leave as-is (other strategies will fail).
    if cleaned is not text:
        text = cleaned
    return text


def parse_llm_json(
    raw: str,
    *,
    label: str = "",
    raise_on_failure: bool = True,
) -> Optional[dict]:
    """Parse JSON from an LLM response, handling markdown code fences.

    Strategies, in order:
    1. Direct ``json.loads`` on the stripped input.
    2. Strip ```json fences, then try again.
    3. Find the first ``{`` to last ``}`` span and parse that.

    Thinking-mode traces (``, ``) are stripped before
    all strategies — providers with reasoning enabled wrap their payload.

    When ``raise_on_failure`` is True (default), raises ``ValueError``
    with a label-prefixed message on failure.  When False, returns None.
    This preserves each call site's existing contract without duplicating
    the extraction logic.

    Args:
        raw: The raw LLM response text.
        label: Optional context label for error messages (e.g. "Member 1").
        raise_on_failure: If True, raise ValueError; if False, return None.

    Returns:
        Parsed dict, or None when raise_on_failure=False and parsing fails.

    Raises:
        ValueError: When raise_on_failure=True and no strategy succeeds.
    """
    if not raw:
        if raise_on_failure:
            raise ValueError(f"{label}: empty input" if label else "empty input")
        return None

    text = _strip_think_blocks(raw).strip()

    # Strategy 1: direct parse
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip ```json fences
    stripped = _FENCE_RE.sub("", text)
    try:
        val = json.loads(stripped)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass

    # Strategy 3: first { to last }
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = stripped[first : last + 1]
        try:
            val = json.loads(candidate)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass

    if raise_on_failure:
        prefix = f"{label}: " if label else ""
        raise ValueError(f"{prefix}could not parse JSON from response: {raw[:500]}")
    return None


def _profile_author() -> str:
    """Best-effort author name for an interactive CLI call.

    Single source; previously duplicated in ``kanban.py``,
    ``kanban_specify.py``, and ``kanban_decompose.py`` to avoid a
    circular import.  Now lives here so all three can import it
    without cycles.
    """
    for env in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        v = os.environ.get(env)
        if v:
            return v
    return os.environ.get("USER", "unknown")
