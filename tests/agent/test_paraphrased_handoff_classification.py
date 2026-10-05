"""Regression tests for #132934: a model-paraphrased compaction handoff must still
classify as a summary row instead of being published as an ordinary assistant reply."""

import pytest

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


@pytest.mark.parametrize(
    ("role", "content", "expected"),
    [
        ("assistant", PARAPHRASED_HANDOFF, True),
        ("assistant", "live tail " + _MERGED_SUMMARY_DELIMITER + "\n" + PARAPHRASED_HANDOFF, True),
        ("assistant", _HANDOFF_MARKER_PREFIX + " is the header string; formatting only.", False),
        ("assistant", _HANDOFF_MARKER_PREFIX + " is the marker; the summary follows it.", False),
        # A user pasting a handoff is still a real user turn, not a synthetic summary row.
        ("user", PARAPHRASED_HANDOFF, False),
    ],
    ids=["paraphrased", "merged", "quoted-no-vocab", "quoted-generic-vocab", "user-paste"],
)
def test_paraphrased_handoff_classification(role, content, expected):
    assert is_compaction_summary_message({"role": role, "content": content}) is expected


def test_adopted_paraphrased_echo_is_not_double_wrapped():
    wrapped = ContextCompressor._with_summary_prefix(PARAPHRASED_HANDOFF)
    assert wrapped.startswith(SUMMARY_PREFIX)
    assert wrapped.count(_HANDOFF_MARKER_PREFIX) == 1
