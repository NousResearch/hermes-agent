"""``PromptResponse.usage``: the tokens one prompt turn consumed.

Hermes keeps session-cumulative counters (``AIAgent.session_*_tokens``, and the gateway's
``session.usage`` / ``session.stats`` dict that mirrors them). A client that bills each prompt —
it stores the usage of every turn and sums them — must receive this turn's share, not the running
total, or a resumed session is charged for every earlier turn again. So the turn's usage is the
counters at turn end minus the base the turn started from.

An agent's counters include its ``delegate_task`` subagents (``tools/delegate_tool_usage.py``);
the gateway adapter reads ``session.stats``, whose frozen field list carries the agent's own only.
A subagent can be folded in after the turn that spawned it has ended (an interrupted parallel batch
does not join its workers), so the base is the counters at the previous report, not a snapshot of
the counters when the turn started: tokens added between two prompts are reported by the next one.

The buckets do not overlap, as in the reference ACP adapters (claude-agent-acp, codex-acp):
``inputTokens`` is uncached input only, cache reads and writes ride their own fields, and
``totalTokens`` is their sum plus output.
"""
from __future__ import annotations

from typing import Any

from acp.schema import Usage

from tools.delegate_tool_usage import TOKEN_COUNTERS as COUNTERS, subagent_tokens

_REPORTED_ATTR = "_acp_reported_counters"


def _int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def counters_of(source: Any) -> dict[str, int] | None:
    """The cumulative counters of an agent (``session_<name>`` attributes plus its folded subagents)
    or of a usage dict.

    None when the input/output counters are missing: nothing to report from.
    """
    if isinstance(source, dict):
        read = {name: _int(source.get(name)) for name in COUNTERS}
        children = dict.fromkeys(COUNTERS, 0)
    elif source is not None:
        read = {name: _int(getattr(source, f"session_{name}", None)) for name in COUNTERS}
        children = subagent_tokens(source)
    else:
        return None
    if read["input_tokens"] is None or read["output_tokens"] is None:
        return None
    return {name: (value or 0) + children[name] for name, value in read.items()}


def turn_base(agent: Any) -> dict[str, int] | None:
    """The base of the agent's next report: its counters at the previous report, or its current
    counters if it has not reported yet. Taken when a prompt starts and kept by that prompt: a queued
    prompt drained inside it reports first, and the outer prompt must still count it."""
    reported = getattr(agent, _REPORTED_ATTR, None)
    if isinstance(reported, dict) and all(type(reported.get(name)) is int for name in COUNTERS):
        return dict(reported)
    return counters_of(agent)


def report_usage(agent: Any, base: dict[str, int] | None, result: Any) -> Usage | None:
    """The prompt's usage: the agent's counters now minus ``base``, recorded as the next base.

    Read off the agent rather than the turn result, which is taken before ``_finish_turn`` drains
    queued prompts and carries no subagents; a result is the fallback for an agent with no counters.
    """
    after = counters_of(agent)
    if after is None:
        return prompt_usage(base, counters_of(result))
    setattr(agent, _REPORTED_ATTR, after)
    return prompt_usage(base, after)


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
