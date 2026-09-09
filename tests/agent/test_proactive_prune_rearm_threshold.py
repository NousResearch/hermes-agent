"""Proactive-prune rearm must not lock out an over-threshold session (#101889).

``_proactive_prune_rearm_tokens`` is armed from a message-bodies-only estimate,
but the provider bills the system prompt and tool schemas too. On a schema-heavy
session the message-only estimate can sit permanently just below the rearm mark
while the real request rides *above* ``threshold_tokens`` — the prune declines
every iteration, full compression never gets there, and nothing is logged. The
session then grows until the provider rejects the request.

Pinned here as invariants (no frozen config literals): the gates are evaluated
against this compressor's own ``threshold_tokens`` / ``proactive_prune_tokens``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens

LARGE_WINDOW = 1_000_000


def _compressor(**kw: Any) -> ContextCompressor:
    defaults = dict(
        model="test",
        quiet_mode=True,
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=4,
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
    )
    defaults.update(kw)
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=LARGE_WINDOW,
    ):
        return ContextCompressor(**defaults)


def _history(n_pairs: int = 8, big: int = 9_000) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": cid,
            "content": chr(65 + i) * big if i < 3 else "ok",
        })
    return msgs


def _park_rearm_just_above_messages(
    compressor: ContextCompressor, messages: List[Dict[str, Any]]
) -> int:
    """Reproduce the reporter's state: message-only estimate stuck 913 tokens
    below the rearm mark (schema overhead makes up the rest of the request)."""
    before = sum(_estimate_msg_budget_tokens(m) for m in messages)
    compressor._proactive_prune_rearm_tokens = before + 913
    assert before < compressor._proactive_prune_rearm_tokens
    return before


def _over_threshold_warnings(caplog) -> list:
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "over the compression threshold" in r.getMessage()
    ]


def test_billed_basis_over_threshold_defeats_message_only_rearm_lockout() -> None:
    """Over ``threshold_tokens`` on the provider-billed basis, the rearm gate
    must not short-circuit the prune on the message-only estimate alone."""
    c = _compressor()
    msgs = _history()
    _park_rearm_just_above_messages(c, msgs)
    billed = c.threshold_tokens + 1  # provider says: over threshold, now

    scans: List[int] = []
    # Stand in for the real multi-pass scan: a NEW list whose old tool outputs
    # are reclaimed, so the (untouched) reclaim gate can commit it.
    reclaimed = [dict(m) for m in msgs]
    for m in reclaimed[:-2]:
        if m.get("role") == "tool":
            m["content"] = "[pruned]"

    def _scan(*args: Any, **kwargs: Any) -> tuple[List[Dict[str, Any]], int]:
        scans.append(1)
        return reclaimed, 3

    with patch.object(c, "_prune_old_tool_results", _scan):
        result, pruned = c.prune_tool_results_only(msgs, current_tokens=billed)

    assert scans, "rearm gate short-circuited on the message-only estimate"
    assert pruned == 3
    assert result is not msgs


def test_message_only_rearm_still_holds_below_threshold() -> None:
    """Prompt-cache hysteresis is intact while the real request is under the
    compression threshold — the rearm bypass is an overflow escape hatch only."""
    c = _compressor()
    msgs = _history()
    _park_rearm_just_above_messages(c, msgs)
    under = c.threshold_tokens - 1
    assert under >= c.proactive_prune_tokens  # above the prune trigger

    with patch.object(
        c,
        "_prune_old_tool_results",
        side_effect=AssertionError("scan must not run below threshold"),
    ):
        result, pruned = c.prune_tool_results_only(msgs, current_tokens=under)

    assert result is msgs
    assert pruned == 0


def test_no_op_below_the_prune_trigger() -> None:
    """Under ``proactive_prune_tokens`` nothing is reclaimed, rearm or not —
    the bypass must not turn into over-pruning of small sessions."""
    c = _compressor()
    msgs = _history()
    c.on_session_reset()  # fully rearmed; only the trigger gates

    with patch.object(
        c,
        "_prune_old_tool_results",
        side_effect=AssertionError("scan must not run below the trigger"),
    ):
        result, pruned = c.prune_tool_results_only(
            msgs, current_tokens=c.proactive_prune_tokens - 1
        )

    assert result is msgs
    assert pruned == 0


def test_over_threshold_reclamation_no_op_warns_once(caplog) -> None:
    """A session riding above the threshold with every reclamation path
    declining must be distinguishable in the log — and must not spam the same
    reason on every tool iteration."""
    # Reclaim floor above anything this transcript can free: the scan runs,
    # finds candidates, and the commit gate rejects it — a silent no-op today.
    c = _compressor(proactive_prune_min_reclaim_tokens=10_000_000)
    msgs = _history()
    billed = c.threshold_tokens + 5_000

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        result, pruned = c.prune_tool_results_only(msgs, current_tokens=billed)
    assert (result, pruned) == (msgs, 0)

    warnings = _over_threshold_warnings(caplog)
    assert warnings, "over-threshold reclamation no-op was silent"

    # Same state on the next tool iteration: deduped, not re-logged.
    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        c.prune_tool_results_only(msgs, current_tokens=billed)
    assert len(_over_threshold_warnings(caplog)) == len(warnings)


def test_under_threshold_no_op_is_not_warned(caplog) -> None:
    """Ordinary hysteresis below the threshold stays quiet."""
    c = _compressor(proactive_prune_min_reclaim_tokens=10_000_000)
    msgs = _history()

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        result, pruned = c.prune_tool_results_only(
            msgs, current_tokens=c.threshold_tokens - 1
        )

    assert (result, pruned) == (msgs, 0)
    assert not _over_threshold_warnings(caplog)


def test_lockout_warns_again_after_rearm_reset(caplog) -> None:
    """A full compaction (or session rebind / model recalibration) zeroes the
    rearm mark. An identical lockout afterwards must warn again — the dedup key
    must not outlive the reclamation that should have cleared it."""
    c = _compressor(proactive_prune_min_reclaim_tokens=10_000_000)
    msgs = _history()
    billed = c.threshold_tokens + 5_000

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        c.prune_tool_results_only(msgs, current_tokens=billed)
    assert len(_over_threshold_warnings(caplog)) == 1
    # Same state, same key (reason, rearm=0): deduped.
    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        c.prune_tool_results_only(msgs, current_tokens=billed)
    assert len(_over_threshold_warnings(caplog)) == 1

    # A public rearm boundary (same helper as compress(), on_session_end,
    # bind_session_state and update_model): pins the wiring, not just the body.
    c.on_session_reset()
    assert c._proactive_prune_rearm_tokens == 0

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        c.prune_tool_results_only(msgs, current_tokens=billed)
    assert len(_over_threshold_warnings(caplog)) == 2, (
        "lockout after a rearm reset was deduped against the stale key"
    )


def test_dropping_under_threshold_clears_dedup_key(caplog) -> None:
    """Back under threshold (e.g. compaction elsewhere shrank the request), the
    key is released so the next over-threshold lockout is reported."""
    c = _compressor(proactive_prune_min_reclaim_tokens=10_000_000)
    msgs = _history()
    billed = c.threshold_tokens + 5_000

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        c.prune_tool_results_only(msgs, current_tokens=billed)
        c.prune_tool_results_only(msgs, current_tokens=c.threshold_tokens - 1)
        c.prune_tool_results_only(msgs, current_tokens=billed)
    assert len(_over_threshold_warnings(caplog)) == 2
