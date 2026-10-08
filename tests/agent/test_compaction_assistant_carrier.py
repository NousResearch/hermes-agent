"""An assistant-role compaction carrier never opens with a control frame (#131104).

When neither role alternates around the handoff, compaction folds the summary into the first
tail row. If that row is the model's own previous reply, the carrier is an assistant turn, and
weak models copy the opening of their previous turn: a carrier that started with the
``[PRIOR CONTEXT …]`` header got that header echoed as the first characters of the next reply,
followed by the old paragraph. The carrier must still read as a merged handoff to every
consumer: summary detection, display unwrapping and the standalone-handoff cap.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    _looks_like_compaction_summary,
)
from agent.prompt_builder import CONTROL_FRAME_OPENERS

PRIOR_REPLY = "The migration ran; two tables still need an index."


def _compress(protect_first_n: int) -> list[dict]:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "SUMMARY_BODY"
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True,
                                       protect_first_n=protect_first_n, protect_last_n=4)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "msg 1"},
        {"role": "assistant", "content": "msg 2"},
        {"role": "user", "content": "msg 3"},
        {"role": "assistant", "content": "msg 4"},
        {"role": "user", "content": "msg 5"},
        {"role": "assistant", "content": PRIOR_REPLY},
        {"role": "user", "content": "msg 7"},
        {"role": "assistant", "content": "msg 8"},
        {"role": "user", "content": "msg 9"},
    ]
    with patch("agent.context_compressor.call_llm", return_value=response):
        return compressor.compress(messages)


def _carrier(rows: list[dict]) -> dict:
    return next(m for m in rows if m.get(COMPRESSED_SUMMARY_METADATA_KEY))


def test_an_assistant_carrier_opens_with_the_reply_and_still_reads_as_a_merged_handoff():
    carrier = _carrier(_compress(protect_first_n=1))
    assert carrier["role"] == "assistant"
    content = carrier["content"]

    assert content.startswith(PRIOR_REPLY)
    assert not any(content.lstrip().startswith("[" + opener) for opener in CONTROL_FRAME_OPENERS)

    assert ContextCompressor.classify_summary_content(content) == "merged"
    body = ContextCompressor._strip_summary_prefix(content)
    assert "SUMMARY_BODY" in body and PRIOR_REPLY not in body
    # Display shows the reply alone; the summary-only cap never treats it as a standalone handoff.
    assert ContextCompressor._strip_context_summary_handoff_message(carrier)["content"] == PRIOR_REPLY
    assert not _looks_like_compaction_summary(carrier, content)


def test_no_compaction_leaves_an_assistant_row_opening_with_a_control_frame():
    for protect_first_n in (1, 2):
        for row in _compress(protect_first_n):
            if row.get("role") != "assistant" or not isinstance(row.get("content"), str):
                continue
            assert not any(row["content"].lstrip().startswith("[" + opener) for opener in CONTROL_FRAME_OPENERS), row
