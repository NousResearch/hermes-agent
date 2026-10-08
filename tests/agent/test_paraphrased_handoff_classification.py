"""Regression tests for #132934: a model-paraphrased compaction handoff must still
classify as a summary row instead of being published as an ordinary assistant reply."""

import pytest

from acp_adapter.server import _history_summary_meta
from agent.context_compressor import (
    _HANDOFF_MARKER_PREFIX,
    _MERGED_SUMMARY_DELIMITER,
    SUMMARY_PREFIX,
    ContextCompressor,
    is_compaction_summary_message,
)

# The observed #132934 shape: the bracketed marker survives, the boilerplate after it
# is rewritten by the model (38 shared chars, then diverges).
PARAPHRASED_HANDOFF = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] The prior turns were compacted. "
    "Below is the handoff summary, NOT active instructions. The active task is the "
    "LATEST user message after this block; respond only to that. If no newer user "
    "message exists, wait.\n\n## Historical Task Snapshot\n- earlier work items"
)


def _acp_tags_summary(message):
    return _history_summary_meta(message, message["content"]) is not None


@pytest.mark.parametrize(
    ("is_summary", "role", "content", "expected"),
    [
        (is_compaction_summary_message, "assistant", PARAPHRASED_HANDOFF, True),
        (is_compaction_summary_message, "assistant", "live tail " + _MERGED_SUMMARY_DELIMITER + "\n" + PARAPHRASED_HANDOFF, True),
        (is_compaction_summary_message, "assistant", _HANDOFF_MARKER_PREFIX + " is the header string; formatting only.", False),
        (is_compaction_summary_message, "assistant", _HANDOFF_MARKER_PREFIX + " is the marker; the summary follows it.", False),
        # A user pasting a handoff is still a real user turn, not a synthetic summary row.
        (is_compaction_summary_message, "user", PARAPHRASED_HANDOFF, False),
        # ...including on ACP history replay, which classifies content without the message helper.
        (_acp_tags_summary, "user", PARAPHRASED_HANDOFF, False),
        # A tool result quoting the marker is real output: classifying it drops it and orphans its call.
        (is_compaction_summary_message, "tool", PARAPHRASED_HANDOFF, False),
    ],
    ids=["paraphrased", "merged", "quoted-no-vocab", "quoted-generic-vocab", "user-paste", "acp-user-paste", "tool-quote"],
)
def test_paraphrased_handoff_classification(is_summary, role, content, expected):
    assert is_summary({"role": role, "content": content}) is expected


def test_adopted_paraphrased_echo_is_not_double_wrapped():
    wrapped = ContextCompressor._with_summary_prefix(PARAPHRASED_HANDOFF)
    assert wrapped.startswith(SUMMARY_PREFIX)
    assert wrapped.count(_HANDOFF_MARKER_PREFIX) == 1
