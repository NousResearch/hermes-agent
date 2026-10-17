"""A recalled bullet is stated once per context WINDOW, not once per turn.

The composed recall block is stamped into the user row it accompanied (``api_content`` sidecar,
or a durable text part on list content) and replayed verbatim while that row stays in context.
``_drop_repeated_recall_lines`` only dedupes inside one block, so a conversation that stays on a
topic re-paid the same facts every turn until compression dropped the rows (#136229).

``window_recall_bullets`` reads what the window still carries — the set is derived, never
stored, so rows removed by compression, a session switch or ``/new`` stop contributing with no
bookkeeping to invalidate.
"""
from __future__ import annotations

from agent.memory_manager import build_memory_context_block, window_recall_bullets


def _body(block: str) -> str:
    """The provider's own content, without the wrapper and the system note that precedes it."""
    return block.split("]\n\n", 1)[1].rsplit("\n</memory-context>", 1)[0]


def _stamped_row(user_text: str, recalled: str) -> dict:
    """A string user row as it sits in history after its turn: clean ``content`` for the
    transcript, the full sent bytes (user text + recall block) in the sidecar."""
    block = build_memory_context_block(recalled)
    return {"role": "user", "content": user_text, "api_content": f"{user_text}\n\n{block}"}


def test_window_recall_bullets_reads_sidecar_and_text_parts():
    rows = [
        _stamped_row("what do you know?", "- prefers draft PRs\n- works on hermes\n"),
        # Multimodal row: the recall block rides as a durable text part on the list.
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            {"type": "text", "text": build_memory_context_block("- reviews in English\n")},
        ]},
        # Clean rows (no injection) and foreign roles contribute nothing.
        {"role": "user", "content": "plain question"},
        {"role": "assistant", "content": "answer"},
        "not even a dict",
    ]

    assert window_recall_bullets(rows) == {
        "- prefers draft PRs", "- works on hermes", "- reviews in English",
    }
    assert window_recall_bullets([]) == set()
    assert window_recall_bullets(None) == set()


def test_a_bullet_the_window_already_replays_is_not_stamped_again():
    window = {"- prefers draft PRs", "- works on hermes"}

    block = build_memory_context_block(
        "- prefers draft PRs\n- ships on Fridays\n", already_in_window=window,
    )

    assert _body(block).splitlines() == ["- ships on Fridays"]


def test_a_bullet_with_continuation_lines_is_never_dropped_across_turns():
    # Same headline, different provenance underneath: dropping the new one would re-parent
    # its provenance under the earlier row's entry and invent a record no provider reported.
    raw = "- prefers draft PRs\n  (logged 3 Feb, source: builtin)\n"
    window = {"- prefers draft PRs"}

    assert _body(build_memory_context_block(raw, already_in_window=window)) == raw


def test_a_fully_replayed_block_yields_no_injection_at_all():
    window = {"- prefers draft PRs"}

    assert build_memory_context_block("- prefers draft PRs\n", already_in_window=window) == ""


def test_two_turns_in_sequence_pay_each_fact_once():
    # Turn 1 stamps its row; turn 2 derives the window from that row and composes fresh.
    turn1_row = _stamped_row("q1", "- alpha\n- beta\n")
    turn2_block = build_memory_context_block(
        "- alpha\n- beta\n- gamma\n", already_in_window=window_recall_bullets([turn1_row]),
    )

    assert _body(turn2_block).splitlines() == ["- gamma"]

    # Compression (or /new) removing the row frees the facts for a fresh stamp.
    assert build_memory_context_block("- alpha\n", already_in_window=window_recall_bullets([]))
