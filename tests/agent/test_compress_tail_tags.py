"""Batch ``compress()`` tags its carried-forward tail dicts (#86366), so the caller can count them for the commit and the
store keeps their originals out of recall."""

from unittest.mock import patch

from agent.context_compressor import _COMPACTION_TAIL_MARKER, ContextCompressor


def test_compress_marks_carried_forward_tail_dicts():
    """compress() must tag its carried-forward tail dicts so the caller
    can pass an accurate tail_count to archive_and_compact (#86366);
    untagged, the tail originals are recalled as compacted history."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/model", protect_first_n=2, protect_last_n=2, quiet_mode=True,
        )
        _ = compressor.context_length

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"filler turn {i} " + "x" * 400}
        for i in range(12)
    ]
    with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
        out = compressor.compress(list(history))

    assert len(out) < len(history)
    tagged = [m for m in out if isinstance(m, dict) and m.get(_COMPACTION_TAIL_MARKER)]
    assert tagged, "carried-forward tail rows were not tagged"
    # The tag marks exactly the verbatim tail: the trailing run of the output,
    # each row a copy of an original from the end of the history.
    assert out[-len(tagged):] == tagged
    originals = {m["content"] for m in history[-len(tagged):]}
    assert {m["content"] for m in tagged} <= originals
    assert not any(m.get(_COMPACTION_TAIL_MARKER) for m in out[:2])
