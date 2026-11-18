"""The turn-end persist override must not overwrite a head row a context engine compacted in place.

``finalize_turn`` rewrites the current-turn user row with the clean transcript just before the durable
snapshot. When an external context engine compacted the live list during the turn, the anchored index
points at the row that now carries the summary (prompt + summary), and the rewrite erased it from the
continuation history (#157402, #144581). The row is rewritten only while it still holds this turn's ask.
"""

from agent.message_metadata import MERGED_TURN_PREFIX
from agent.session_persistence import SessionPersistenceMixin, note_persist_user_anchor
from session_rules.turns import repair_message_sequence

SUMMARY_HEAD = "<summary>\n\nUser: fix the flaky test"


def _agent(idx, override):
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = idx
    agent._persist_user_message_override = override
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    return agent


def test_an_in_place_compacted_head_row_keeps_its_summary():
    """The engine wrote its head into the same row and set no Hermes summary marker, so the metadata guard
    cannot see it; the row no longer holds this turn's ask, and the summary stays."""
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the flaky test"},
        {"role": "assistant", "content": "running the suite"},
        {"role": "tool", "content": "FAILED test_x"},
    ]
    agent = _agent(1, "fix the flaky test")
    note_persist_user_anchor(agent, messages)

    messages[1]["content"] = SUMMARY_HEAD  # in-place compaction, no Hermes marker
    agent._apply_persist_user_message_override(messages)

    assert COMPRESSED_SUMMARY_METADATA_KEY not in messages[1]
    assert messages[1]["content"] == SUMMARY_HEAD


def test_a_normal_turn_still_gets_the_clean_override():
    """Negative control: the anchored row still holds the API variant, so the clean text replaces it."""
    live = "[gateway note] observed context\n\nactual question"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": live},
        {"role": "assistant", "content": "answer"},
    ]
    agent = _agent(1, "actual question")
    note_persist_user_anchor(agent, messages)

    agent._apply_persist_user_message_override(messages)

    assert messages[1]["content"] == "actual question"


def test_a_merged_row_keeps_the_prefix_and_takes_the_clean_tail():
    """A consecutive-user merge folds this turn's row into the earlier unanswered one: the survivor still
    holds this turn's ask past its recorded prefix, so the override joins like the merge does."""
    wire = "[2026-04-13T17:02:06+0200] can you also run the smoke tests"
    messages = [
        {"role": "assistant", "content": "previous response"},
        {"role": "user", "content": "please deploy build 42 to staging"},
        {"role": "user", "content": wire},
    ]
    agent = _agent(2, "can you also run the smoke tests")
    note_persist_user_anchor(agent, messages)

    assert repair_message_sequence(None, messages) == 1  # our row folds into the earlier one
    agent._persist_user_message_idx = 1  # what reanchor does for a rebuilt list
    agent._apply_persist_user_message_override(messages)

    assert messages[1][MERGED_TURN_PREFIX] == "please deploy build 42 to staging"
    assert messages[1]["content"] == "please deploy build 42 to staging\n\ncan you also run the smoke tests"


def test_a_multimodal_row_still_takes_its_list_override_after_appended_context():
    """``_append_multimodal_context`` appends parts after the ask; the row still holds it, and a list
    override is still the clean payload that replaces the parts."""
    ask = "what is this?"
    content = [{"type": "text", "text": ask}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": content},
    ]
    agent = _agent(1, [{"type": "text", "text": ask}])
    note_persist_user_anchor(agent, messages)

    messages[1]["content"] = [*content, {"type": "text", "text": "\n\nPLUGIN-CTX"}]
    agent._apply_persist_user_message_override(messages)

    assert messages[1]["content"] == [{"type": "text", "text": ask}]