"""Bedrock ConverseStream can carry several signed reasoning blocks under ONE contentBlockIndex.

Each signature signs only its own block, so the stream normalizer must keep them apart; a merged
``"first"+"second"`` / ``"SIG1"+"SIG2"`` block is rejected on replay with "Invalid signature".
"""

import base64

from agent.bedrock_adapter import _replay_ordered_blocks, normalize_converse_stream_events


def _delta(reasoning):
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": reasoning}}}


def _stream(*deltas):
    return {"stream": [
        {"messageStart": {"role": "assistant"}},
        *deltas,
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "answer"}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]}


def test_signed_blocks_sharing_an_index_replay_separately():
    msg = normalize_converse_stream_events(_stream(
        _delta({"text": "fir"}), _delta({"text": "st"}), _delta({"signature": "SIG"}), _delta({"signature": "1"}),
        _delta({"text": "second"}), _delta({"signature": "SIG2"}),
        _delta({"redactedContent": b"\x00ab"}), _delta({"redactedContent": b"cd\xff"}),
    )).choices[0].message

    assert _replay_ordered_blocks(msg.bedrock_content_blocks) == [
        {"reasoningContent": {"reasoningText": {"text": "first", "signature": "SIG1"}}},
        {"reasoningContent": {"reasoningText": {"text": "second", "signature": "SIG2"}}},
        # Binary chunks of one redacted block join as bytes, not as overwritten fragments.
        {"reasoningContent": {"redactedContent": b"\x00abcd\xff"}},
        {"text": "answer"},
    ]
    assert msg.reasoning_details == [
        {"type": "thinking", "thinking": "first", "signature": "SIG1"},
        {"type": "thinking", "thinking": "second", "signature": "SIG2"},
        {"type": "redacted_thinking", "data": base64.b64encode(b"\x00abcd\xff").decode()},
    ]


def test_signed_block_with_omitted_text_keeps_its_signature():
    """Omitted thinking streams only a signature; dropping the block loses the turn's reasoning."""
    msg = normalize_converse_stream_events(_stream(_delta({"signature": "SIG"}))).choices[0].message

    assert _replay_ordered_blocks(msg.bedrock_content_blocks)[0] == {
        "reasoningContent": {"reasoningText": {"text": "", "signature": "SIG"}}}
    assert msg.reasoning_details == [{"type": "thinking", "thinking": "", "signature": "SIG"}]
