"""``extract_reasoning`` must not display the same thinking twice.

``reasoning_details`` carries the per-block thinking that the joined ``reasoning`` /
``reasoning_content`` string already holds -- the two differ only in how the streamed deltas were
rejoined. Exact-match dedupe therefore let every block through a second time, which is what surfaced
when the Bedrock adapter started recording signed thinking for replay across ``--resume`` (#121293).
"""

from types import SimpleNamespace

from agent.agent_runtime_helpers import extract_reasoning

BLOCK_1 = "The user wants both the seed file and an echo; run them in parallel first."
BLOCK_2 = "I have both codewords and the echo, so I can answer now."


def _msg(**kwargs):
    fields = {"content": None, "reasoning": None, "reasoning_content": None, "reasoning_details": None}
    return SimpleNamespace(**{**fields, **kwargs})


def test_per_block_details_do_not_repeat_the_joined_reasoning_content():
    reasoning = extract_reasoning(None, _msg(
        reasoning_content=f"{BLOCK_1}\n\n{BLOCK_2}",
        reasoning_details=[{"type": "thinking", "thinking": BLOCK_1, "signature": "s1"},
                           {"type": "thinking", "thinking": BLOCK_2, "signature": "s2"}],
    ))

    assert reasoning == f"{BLOCK_1}\n\n{BLOCK_2}"


def test_whitespace_only_differences_still_count_as_the_same_thinking():
    """The joined string can carry blank lines the per-block copy does not (#98468), so the dedupe
    cannot compare the two literally."""
    shredded = BLOCK_1.replace("parallel", "par\n\nallel")

    assert extract_reasoning(None, _msg(
        reasoning_content=shredded,
        reasoning_details=[{"type": "thinking", "thinking": BLOCK_1, "signature": "s1"}],
    )) == shredded


def test_reasoning_and_reasoning_content_holding_the_same_text_collapse():
    assert extract_reasoning(None, _msg(reasoning=BLOCK_1, reasoning_content=BLOCK_1)) == BLOCK_1


def test_detail_text_absent_from_the_joined_string_is_still_appended():
    assert extract_reasoning(None, _msg(
        reasoning_content=BLOCK_1,
        reasoning_details=[{"type": "reasoning.summary", "summary": BLOCK_2}],
    )) == f"{BLOCK_1}\n\n{BLOCK_2}"


def test_no_reasoning_anywhere_is_none():
    assert extract_reasoning(None, _msg()) is None
