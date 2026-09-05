"""Tests for proactive tool-result pruning.

``ContextCompressor.prune_tool_results_only`` runs the cheap, deterministic
Phase-1 prune (summarize old tool outputs, dedup repeats) on a cost-oriented
trigger that is INDEPENDENT of the full-compression threshold. On large-window
models ``should_compress()`` (~50% of the window) rarely fires, so without this
the old tool outputs ride in history and are re-sent verbatim every turn.

Mirrors the construction/patching conventions in test_context_compressor.py.
"""

from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _PRUNED_TOOL_PLACEHOLDER,
    _estimate_msg_budget_tokens,
)

LARGE_WINDOW = 1_000_000


def _compressor(**kw):
    defaults = dict(
        model="test",
        quiet_mode=True,
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=4,
    )
    defaults.update(kw)
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=LARGE_WINDOW,
    ):
        return ContextCompressor(**defaults)


def _assistant_call(cid, name="terminal", args='{"cmd":"ls"}'):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": args}}
        ],
    }


def _tool_msg(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _build(n_pairs, big_indices, big_chars=9000, small="ok"):
    """system + n_pairs of (assistant tool_call, tool result).

    Tool results whose pair index is in ``big_indices`` get a distinct payload
    of ``big_chars`` characters; the rest get a tiny payload.
    """
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append(_assistant_call(cid))
        if i in big_indices:
            msgs.append(_tool_msg(cid, chr(65 + (i % 26)) * big_chars))
        else:
            msgs.append(_tool_msg(cid, small))
    return msgs


def _tool_by_id(msgs, cid):
    return [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == cid][0]


def test_prunes_below_compression_threshold():
    """The whole point: prune fires at 120k tokens, far below the ~500k
    (50% of 1M) full-compression trigger that would otherwise never run."""
    c = _compressor(proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
    assert c.should_compress(prompt_tokens=120_000) is False  # compression would NOT run
    msgs = _build(8, big_indices={0, 1, 2})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert pruned >= 3
    assert len(result) == len(msgs)
    for cid in ("call_0", "call_1", "call_2"):
        m = _tool_by_id(result, cid)
        assert len(m["content"]) < 9000                       # summarized
        assert m["content"] != _PRUNED_TOOL_PLACEHOLDER       # informative, not a blank placeholder












def test_idempotent():
    c = _compressor(proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
    msgs = _build(8, big_indices={0, 1, 2})
    first, n1 = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert n1 >= 3
    # No usage reading bypasses the token gate and exercises prune idempotence.
    second, n2 = c.prune_tool_results_only(first, current_tokens=None)
    assert n2 == 0
    assert [m.get("content") for m in second] == [m.get("content") for m in first]


def test_rearms_only_after_reclaimed_token_runway():
    """A prune boundary must earn back its cache break before the next one."""
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
    )
    msgs = _build(8, big_indices={0, 1, 2, 6, 7})

    first, n1 = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert n1 >= 3
    rearm_tokens = sum(map(_estimate_msg_budget_tokens, first)) + 48_000

    # Age the two protected large results out of the tail.  They are now a
    # valid >=4K-token prune candidate, but the post-prune prompt has not yet
    # regrown the tokens reclaimed at the first cache-breaking boundary.
    grown = first + [
        _assistant_call("call_8"),
        _tool_msg("call_8", "ok"),
        _assistant_call("call_9"),
        _tool_msg("call_9", "ok"),
    ]
    assert sum(map(_estimate_msg_budget_tokens, grown)) < rearm_tokens
    blocked, n2 = c.prune_tool_results_only(grown, current_tokens=1_000_000)
    assert n2 == 0
    assert blocked is grown
    assert len(_tool_by_id(blocked, "call_6")["content"]) == 9000
    assert len(_tool_by_id(blocked, "call_7")["content"]) == 9000

    missing = rearm_tokens - sum(map(_estimate_msg_budget_tokens, grown))
    regrown = grown + [{"role": "user", "content": "x" * (missing * 4)}]
    assert sum(map(_estimate_msg_budget_tokens, regrown)) >= rearm_tokens
    rearmed, n3 = c.prune_tool_results_only(regrown, current_tokens=1_000_000)
    assert n3 >= 2
    assert rearmed is not regrown


def test_successful_full_compression_resets_proactive_runway():
    """A full compression establishes a fresh cache boundary and baseline."""
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
    )
    first, n1 = c.prune_tool_results_only(
        _build(8, big_indices={0, 1, 2}), current_tokens=120_000,
    )
    assert n1 >= 3

    history = [{"role": "system", "content": "sys"}]
    for i in range(12):
        history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"turn {i} " + ("x" * 1000),
        })
    c.tail_token_budget = 50
    with patch.object(c, "_generate_summary", return_value="summary"):
        compressed = c.compress(history, current_tokens=500_000, force=True)
    assert c._last_compression_made_progress is True
    assert len(compressed) < len(history)

    # The successful full boundary supersedes the old proactive-prune runway.
    fresh = _build(8, big_indices={0, 1, 2})
    result, pruned = c.prune_tool_results_only(fresh, current_tokens=48_000)
    assert pruned >= 3
    assert result is not fresh






# ---------------------------------------------------------------------------
# Salvage follow-ups: no-op caller contract, prompt-cache hysteresis gate,
# no-orphan pairing invariant, and the default-off behavior pin.
# ---------------------------------------------------------------------------








def test_min_reclaim_gate_default_and_clamp():
    """Default 4096; negative/None coerce to disabled (0)."""
    assert _compressor().proactive_prune_min_reclaim_tokens == 4096
    assert _compressor(proactive_prune_min_reclaim_tokens=0).proactive_prune_min_reclaim_tokens == 0
    assert _compressor(proactive_prune_min_reclaim_tokens=-5).proactive_prune_min_reclaim_tokens == 0
    assert _compressor(proactive_prune_min_reclaim_tokens=None).proactive_prune_min_reclaim_tokens == 0


def test_no_orphans_both_directions():
    """tool_call_id pairing survives the prune in BOTH directions: every
    surviving tool result has its assistant call, and every assistant tool_call
    has its result row (the #69830 test-pin rule — never assert exact surviving
    pair counts, only the pairing invariant)."""
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,
    )
    msgs = _build(10, big_indices={0, 1, 2, 3, 4})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert pruned >= 1
    call_ids = set()
    for m in result:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                call_ids.add(tc["id"] if isinstance(tc, dict) else tc.id)
    result_ids = {m["tool_call_id"] for m in result if m.get("role") == "tool"}
    assert result_ids <= call_ids, "orphan tool results without a matching call"
    assert call_ids <= result_ids, "orphan tool calls without a matching result"


def test_unset_config_zero_behavior_change():
    """Pin: with the config knobs unset, the compressor behaves byte-identically
    to pre-feature main — the prune path is dead code and the full-compression
    Phase-1 caller keeps its 200-char floor."""
    c = _compressor()  # nothing configured
    assert c.proactive_prune_tokens == 0
    msgs = _build(8, big_indices={0, 1, 2})
    import copy
    snapshot = copy.deepcopy(msgs)
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=10_000_000)
    assert pruned == 0
    assert result is msgs
    assert msgs == snapshot  # input never mutated
    # And the compression-path caller still prunes at the 200-char default floor
    # (min_prune_chars default unchanged).
    import inspect
    sig = inspect.signature(c._prune_old_tool_results)
    assert sig.parameters["min_prune_chars"].default == 200
