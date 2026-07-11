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
    c = _compressor(proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
    # pair 0 tool is old (index 2); pair 7 tool is in the last-4 protected tail (index 16)
    msgs = _build(8, big_indices={0, 7})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert len(_tool_by_id(result, "call_7")["content"]) == 9000   # protected, untouched
    assert len(_tool_by_id(result, "call_0")["content"]) < 9000    # old, summarized


def test_size_floor_spares_small_results():
    c = _compressor(proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
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
