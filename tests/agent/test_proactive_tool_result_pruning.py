"""Tests for proactive tool-result pruning.

``ContextCompressor.prune_tool_results_only`` runs the cheap, deterministic
Phase-1 prune (summarize old tool outputs, dedup repeats) on a cost-oriented
trigger that is INDEPENDENT of the full-compression threshold. On large-window
models ``should_compress()`` (~50% of the window) rarely fires, so without this
the old tool outputs ride in history and are re-sent verbatim every turn.

Mirrors the construction/patching conventions in test_context_compressor.py.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _PRUNED_TOOL_PLACEHOLDER

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


def test_disabled_by_default_is_noop():
    c = _compressor()  # proactive_prune_tokens defaults to 0
    assert c.proactive_prune_tokens == 0
    msgs = _build(8, big_indices={0, 1, 2})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=500_000)
    assert pruned == 0
    assert [m.get("content") for m in result] == [m.get("content") for m in msgs]


def test_below_trigger_is_noop():
    c = _compressor(proactive_prune_tokens=48_000)
    msgs = _build(8, big_indices={0, 1, 2})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=10_000)
    assert pruned == 0


def test_recent_tail_is_protected():
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,  # gate off: this test pins tail semantics
    )
    # pair 0 tool is old (index 2); pair 7 tool is in the last-4 protected tail (index 16)
    msgs = _build(8, big_indices={0, 7})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert len(_tool_by_id(result, "call_7")["content"]) == 9000   # protected, untouched
    assert len(_tool_by_id(result, "call_0")["content"]) < 9000    # old, summarized


def test_size_floor_spares_small_results():
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,  # gate off: this test pins the size floor
    )
    msgs = _build(8, big_indices={1}, big_chars=9000)
    for m in msgs:                      # make pair 0's tool 5000 chars (< 8000 floor), still old
        if m.get("tool_call_id") == "call_0":
            m["content"] = "Z" * 5000
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert len(_tool_by_id(result, "call_0")["content"]) == 5000   # under floor -> untouched
    assert len(_tool_by_id(result, "call_1")["content"]) < 9000    # over floor -> summarized


def test_structure_preserved():
    c = _compressor(proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
    msgs = _build(8, big_indices={0, 1, 2})
    roles_before = [m["role"] for m in msgs]
    ids_before = [m.get("tool_call_id") for m in msgs]
    result, _ = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert len(result) == len(msgs)
    assert [m["role"] for m in result] == roles_before
    assert [m.get("tool_call_id") for m in result] == ids_before


def test_idempotent():
    c = _compressor(proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
    msgs = _build(8, big_indices={0, 1, 2})
    first, n1 = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert n1 >= 3
    second, n2 = c.prune_tool_results_only(first, current_tokens=120_000)
    assert n2 == 0
    assert [m.get("content") for m in second] == [m.get("content") for m in first]


def test_prune_old_tool_results_default_floor_unchanged():
    """Backward-compat: without min_prune_chars, _prune_old_tool_results still
    prunes >200-char results (the compression Phase-1 caller's behavior)."""
    c = _compressor()
    msgs = _build(8, big_indices=set())
    for m in msgs:                      # a 300-char old tool result
        if m.get("tool_call_id") == "call_0":
            m["content"] = "Q" * 300
    result, pruned = c._prune_old_tool_results(msgs, protect_tail_count=4)
    assert len(_tool_by_id(result, "call_0")["content"]) < 300
    assert pruned >= 1


def test_min_result_chars_floor_is_clamped():
    """Config-robustness: a floor below 200 (or negative) is clamped up to 200,
    while a configured 0 falls back to the 8000 default via ``or``. Without the
    clamp, a tiny floor lets Pass 2 re-summarize its own (short) summary every
    turn, and a negative floor strips every non-tail tool result."""
    assert _compressor(proactive_prune_min_result_chars=0).proactive_prune_min_result_chars == 8000
    assert _compressor(proactive_prune_min_result_chars=50).proactive_prune_min_result_chars == 200
    assert _compressor(proactive_prune_min_result_chars=-1).proactive_prune_min_result_chars == 200
    assert _compressor(proactive_prune_min_result_chars=8000).proactive_prune_min_result_chars == 8000


# ---------------------------------------------------------------------------
# Salvage follow-ups: no-op caller contract, prompt-cache hysteresis gate,
# no-orphan pairing invariant, and the default-off behavior pin.
# ---------------------------------------------------------------------------


def test_noop_paths_return_input_object():
    """Standard caller contract: every no-op path hands back the INPUT list
    object so callers can gate bookkeeping on ``result is not input``."""
    msgs = _build(8, big_indices={0, 1, 2})
    # Disabled (default)
    c = _compressor()
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=500_000)
    assert pruned == 0 and result is msgs
    # Below trigger
    c = _compressor(proactive_prune_tokens=48_000)
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=10_000)
    assert pruned == 0 and result is msgs
    # Above trigger but nothing prunable (all results tiny)
    c = _compressor(proactive_prune_tokens=48_000)
    tiny = _build(8, big_indices=set())
    result, pruned = c.prune_tool_results_only(tiny, current_tokens=120_000)
    assert pruned == 0 and result is tiny


def test_min_reclaim_gate_blocks_small_prunes():
    """Prompt-cache hysteresis: a prune that would reclaim less than
    ``proactive_prune_min_reclaim_tokens`` must NOT commit (returns the input
    object) — rewriting already-sent history for a trivial saving would break
    the provider's cached prefix every tool iteration."""
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=1_000_000,  # unreachably high
    )
    msgs = _build(8, big_indices={0, 1, 2})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert pruned == 0
    assert result is msgs  # input object — caller commits nothing


def test_min_reclaim_gate_allows_large_prunes():
    """A prune reclaiming more than the gate commits normally."""
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=1_000,  # 3×9000 chars ≈ 6.7K tokens reclaimed
    )
    msgs = _build(8, big_indices={0, 1, 2})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert pruned >= 3
    assert result is not msgs


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
