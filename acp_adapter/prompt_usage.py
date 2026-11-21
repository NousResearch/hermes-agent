"""``PromptResponse.usage``: the tokens one prompt turn consumed.

Hermes keeps session-cumulative counters (``AIAgent.session_*_tokens``, and the gateway's
``session.usage`` / ``session.stats`` dict that mirrors them). A client that bills each prompt —
it stores the usage of every turn and sums them — must receive this turn's share, not the running
total, or a resumed session is charged for every earlier turn again. So the turn's usage is the
counters at turn end minus a snapshot taken when the turn started.

The buckets do not overlap, as in the reference ACP adapters (claude-agent-acp, codex-acp):
``inputTokens`` is uncached input only, cache reads and writes ride their own fields, and
``totalTokens`` is their sum plus output.
"""
from __future__ import annotations

from typing import Any

from acp.schema import Usage

COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def _int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def counters_of(source: Any) -> dict[str, int] | None:
    """The cumulative counters of an agent (``session_<name>`` attributes) or of a usage dict.

    None when the input/output counters are missing: nothing to report from.
    """
    if isinstance(source, dict):
        read = {name: _int(source.get(name)) for name in COUNTERS}
    elif source is not None:
        read = {name: _int(getattr(source, f"session_{name}", None)) for name in COUNTERS}
    else:
        return None
    if read["input_tokens"] is None or read["output_tokens"] is None:
        return None
    return {name: value or 0 for name, value in read.items()}


def turn_delta(before: dict[str, int] | None, after: dict[str, int] | None) -> dict[str, int] | None:
    """``after - before``. A counter that went down means the session's counters restarted during
    the turn (a new agent, ``/new``): what accumulated since the restart is then the turn's usage."""
    if after is None:
        return None
    if before is None or any(after[name] < before[name] for name in COUNTERS):
        return dict(after)
    return {name: after[name] - before[name] for name in COUNTERS}


def prompt_usage(before: dict[str, int] | None, after: dict[str, int] | None) -> Usage | None:
    delta = turn_delta(before, after)
    if delta is None:
        return None
    return Usage(
        input_tokens=delta["input_tokens"], output_tokens=delta["output_tokens"],
        total_tokens=(delta["input_tokens"] + delta["output_tokens"]
                      + delta["cache_read_tokens"] + delta["cache_write_tokens"]),
        thought_tokens=delta["reasoning_tokens"] or None,
        cached_read_tokens=delta["cache_read_tokens"] or None,
        cached_write_tokens=delta["cache_write_tokens"] or None,
    )
