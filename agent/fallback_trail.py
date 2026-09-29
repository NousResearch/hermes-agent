"""Per-walk record of what the fallback chain actually did, so a terminal failure can
name EVERY hop's outcome instead of only the last one's error.

Why this exists
---------------
Both terminal paths in :mod:`agent.turn_recovery` summarize a single ``api_error`` —
the one raised by whichever backend happened to be active when the turn gave up. Once
the fallback chain has been walked, that is the LAST hop, so the surfaced failure reads
like a terminal-hop problem even when the primary was the real cause and the last hop
only took over because the earlier ones were rate-limited, unentitled, or skipped
entirely (never attempted). Cron agent jobs raise ``RuntimeError(result["error"])``
(``cron/scheduler.py``), so the masked text is exactly what an operator sees.

The hop-by-hop facts already existed, but only as ``logger.*`` lines inside
``try_activate_fallback`` — invisible to any surface that reports the *result*. This
module keeps them on the agent and renders them beside the terminal error.

Shape
-----
``agent._fallback_trail`` is a list of dicts, appended in walk order::

    {"hop": 1, "role": "primary", "backend": "nous/deepseek-v4.1-flash",
     "outcome": "failed", "reason": "rate limit", "detail": "HTTP 429: ..."}

``outcome`` is one of ``failed`` / ``skipped`` / ``exhausted``. ``role`` is
``primary`` for the backend active before any hop was taken, ``fallback`` for a chain
entry, ``chain`` for the end-of-walk marker.

Scope is one *walk*, not one session: ``try_activate_fallback`` resets the trail when
it is entered with ``_fallback_index == 0`` (a fresh walk), matching how the retry loop
re-arms the chain.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_TRAIL_ATTR = "_fallback_trail"
# Bounded so a pathological chain cannot grow the result dict without limit.
_TRAIL_CAP = 12
_REASON_CAP = 180
_DETAIL_CAP = 200

_ROLE_LABEL = {"primary": "primary", "fallback": "fallback", "chain": "chain"}
_OUTCOME_VERB = {"failed": "failed", "skipped": "skipped", "exhausted": "stopped"}


def _trail(agent: Any) -> List[Dict[str, Any]]:
    """The agent's trail list, created on first use. ``[]`` if the agent refuses it."""
    trail = getattr(agent, _TRAIL_ATTR, None)
    if isinstance(trail, list):
        return trail
    trail = []
    try:
        setattr(agent, _TRAIL_ATTR, trail)
    except Exception:  # slots / frozen test doubles: degrade to no trail, never raise
        return []
    return trail


def reset_fallback_trail(agent: Any) -> None:
    """Start a fresh walk. Called once per fresh chain walk, not per attempt."""
    try:
        setattr(agent, _TRAIL_ATTR, [])
    except Exception:
        pass


def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _backend(provider: Any, model: Any) -> str:
    provider = str(provider or "").strip()
    model = str(model or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    return model or provider or "unknown"


def _append(agent: Any, entry: Dict[str, Any]) -> None:
    if not isinstance(getattr(agent, _TRAIL_ATTR, None), list):
        _trail(agent)  # materialize; a test double that refuses it records nothing
    trail = getattr(agent, _TRAIL_ATTR, None)
    if not isinstance(trail, list) or len(trail) >= _TRAIL_CAP:
        return
    entry = dict(entry)
    entry["hop"] = len(trail) + 1
    trail.append(entry)


def record_hop_failure(
    agent: Any, *, provider: Any, model: Any, role: str = "fallback",
    reason: Any = "", detail: Any = "",
) -> None:
    """Record a hop (or the primary) that was tried and failed.

    Deduped against the previous entry: the retry loop calls
    ``try_activate_fallback`` again for the same still-failing backend once the chain is
    exhausted, and one line per backend is what the reader needs.
    """
    trail = _trail(agent)
    backend = _backend(provider, model)
    if trail:
        last = trail[-1]
        if last.get("outcome") in {"failed", "exhausted"} and last.get("backend") == backend:
            return
    _append(agent, {
        "role": role if role in _ROLE_LABEL else "fallback",
        "backend": backend,
        "outcome": "failed",
        "reason": _clip(reason, _REASON_CAP),
        "detail": _clip(detail, _DETAIL_CAP),
    })


def record_hop_skip(agent: Any, *, provider: Any, model: Any, reason: Any) -> None:
    """Record a chain entry that was never sent to — with the reason it was passed over."""
    _append(agent, {
        "role": "fallback",
        "backend": _backend(provider, model),
        "outcome": "skipped",
        "reason": _clip(reason, _REASON_CAP),
        "detail": "",
    })


def record_chain_end(agent: Any, *, entries: int = 0, note: str = "") -> None:
    """Mark the end of the walk: no hop remained, so nothing further was attempted."""
    trail = _trail(agent)
    if not trail:
        return  # nothing was tried — an end-of-walk line would be the only entry
    if trail[-1].get("outcome") == "exhausted":
        return
    _append(agent, {
        "role": "chain",
        "backend": "—",
        "outcome": "exhausted",
        "reason": note or (f"end of chain ({entries} entr{'y' if entries == 1 else 'ies'})"),
        "detail": "",
    })


def fallback_trail(agent: Any) -> List[Dict[str, Any]]:
    """A copy of the recorded walk (empty before any hop work happens)."""
    return [dict(e) for e in _trail(agent)]


def format_fallback_trail(agent: Any) -> str:
    """Render the walk as operator-facing text; ``""`` when there is nothing to explain.

    Returns ``""`` for a single-entry trail: with no hop taken, the terminal error
    already names that backend and the block would be pure noise.
    """
    trail = _trail(agent)
    if len(trail) < 2:
        return ""
    failed = [e for e in trail if e.get("outcome") == "failed"]
    lines = [
        f"Fallback chain walk ({len(trail)} step(s) recorded — the error above is the "
        f"LAST hop's, not necessarily the cause):"
    ]
    for entry in trail:
        verb = _OUTCOME_VERB.get(str(entry.get("outcome")), str(entry.get("outcome")))
        role = _ROLE_LABEL.get(str(entry.get("role")), "fallback")
        line = f"  hop {entry.get('hop')} {role} {entry.get('backend')} — {verb}"
        reason = str(entry.get("reason") or "")
        if reason:
            line += f": {reason}"
        detail = str(entry.get("detail") or "")
        if detail and detail != reason:
            line += f" [{detail}]"
        lines.append(line)
    if len(failed) == 1:
        lines.append(
            "  (only one hop reported an error surface; the others were passed over — "
            "see the reasons above)")
    return "\n".join(lines)


def attach_fallback_trail(result: Dict[str, Any], agent: Any) -> Dict[str, Any]:
    """Add the walk to a terminal result: structured on ``fallback_trail``, rendered onto
    ``error`` (the machine-facing field every caller reads — cron raises it as
    ``RuntimeError``). Returns ``result`` unchanged when there is nothing to add."""
    if not isinstance(result, dict):
        return result
    text = format_fallback_trail(agent)
    if not text:
        return result
    result["fallback_trail"] = fallback_trail(agent)
    error = str(result.get("error") or "").strip()
    result["error"] = f"{error}\n\n{text}" if error else text
    return result
