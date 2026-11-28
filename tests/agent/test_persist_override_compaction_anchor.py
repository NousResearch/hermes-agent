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


def test_an_ask_prefixed_compaction_summary_is_preserved():
    messages = [{"role": "user", "content": "actual question"}]
    agent = _agent(0, "actual question")
    note_persist_user_anchor(agent, messages)
    compacted = (
        "actual question\n\n"
        "<summary>Tests already ran; preserve the deployment decision.</summary>"
    )
    messages[0]["content"] = compacted

    agent._apply_persist_user_message_override(messages)

    assert messages[0]["content"] == compacted


def test_a_real_todo_fold_starts_with_the_header_and_takes_the_clean_override():
    from agent.conversation_compression import _fold_todo_snapshot
    from tools.todo_tool import TODO_INJECTION_HEADER, TodoStore

    staged = "[API-only guidance]\n\nactual question"
    messages = [{"role": "user", "content": staged}]
    agent = _agent(0, "actual question")
    agent._todo_store = TodoStore()
    agent._todo_store.write([{"id": "1", "content": "run tests", "status": "pending"}])
    snapshot = agent._todo_store.format_for_injection()
    assert snapshot is not None
    assert snapshot.startswith(TODO_INJECTION_HEADER)
    note_persist_user_anchor(agent, messages)

    _fold_todo_snapshot(agent, messages)

    assert messages[0]["content"] == staged + "\n\n" + snapshot
    agent._apply_persist_user_message_override(messages)
    assert messages[0]["content"] == "actual question"


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


def test_an_absorbed_turn_transfers_ownership_to_the_survivor():
    """Sequence repair transfers ownership when it absorbs this turn's unchanged row."""
    wire = "[2026-04-13T17:02:06+0200] can you also run the smoke tests"
    messages = [
        {"role": "assistant", "content": "previous response"},
        {"role": "user", "content": "please deploy build 42 to staging"},
        {"role": "user", "content": wire},
    ]
    agent = _agent(2, "can you also run the smoke tests")
    note_persist_user_anchor(agent, messages)

    assert repair_message_sequence(agent, messages) == 1
    agent._persist_user_message_idx = 1  # the turn index follows the surviving merged row
    agent._apply_persist_user_message_override(messages)

    assert messages[1][MERGED_TURN_PREFIX] == "please deploy build 42 to staging"
    assert messages[1]["content"] == "please deploy build 42 to staging\n\ncan you also run the smoke tests"
    assert agent._persist_user_message_anchor_row is messages[1]


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

    from agent.turn_context import _append_multimodal_context

    _append_multimodal_context(agent, messages[1], "", "PLUGIN-CTX", preflight_compressed=False)
    assert len(messages[1]["content"]) > len(content) - 1
    agent._apply_persist_user_message_override(messages)

    assert messages[1]["content"] == [{"type": "text", "text": ask}]

def test_a_stale_merge_prefix_does_not_rescue_a_compacted_head():
    """The merge metadata outlives a row whose content the engine replaced: the recorded prefix is no
    longer in the content, so it must not certify that the row still holds this turn's ask."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the flaky test"},
        {"role": "assistant", "content": "running the suite"},
    ]
    agent = _agent(1, "fix the flaky test")
    note_persist_user_anchor(agent, messages)

    # A todo snapshot was absorbed onto this ask (empty tail), then the engine compacted the row in place
    # and kept the row dict — the merge metadata survives the content swap.
    messages[1][MERGED_TURN_PREFIX] = "fix the flaky test"
    messages[1]["content"] = SUMMARY_HEAD

    agent._apply_persist_user_message_override(messages)

    assert messages[1]["content"] == SUMMARY_HEAD


def test_a_folded_todo_snapshot_does_not_disarm_the_anchor():
    """The compression boundary appends its todo snapshot to the trailing real user row. That built-in
    mutation keeps this turn's ask at the head, but the frozen anchor is the bare ask, so the finalizer
    must still rewrite the row — otherwise the API-only text stays in the returned history and in
    trajectory exports while only the DB projection is clean."""
    from agent.conversation_compression import _fold_todo_snapshot
    from tools.todo_tool import TODO_INJECTION_HEADER, TodoStore

    agent = _agent(1, "actual question")
    agent._todo_store = TodoStore()
    agent._todo_store.write([{"id": "1", "content": "run tests", "status": "pending"}])
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "[API-only guidance] actual question"},
    ]
    note_persist_user_anchor(agent, messages)
    _fold_todo_snapshot(agent, messages)
    assert TODO_INJECTION_HEADER in messages[1]["content"]  # the boundary really folded the snapshot in

    agent._apply_persist_user_message_override(messages)

    assert messages[1]["content"] == "actual question"


def test_stale_prefix_cannot_certify_an_ask_prefixed_summary():
    messages = [{"role": "user", "content": "actual question"}]
    agent = _agent(0, "actual question")
    note_persist_user_anchor(agent, messages)
    messages[0][MERGED_TURN_PREFIX] = "actual question"
    compacted = "actual question\n\n<summary>keep the decision</summary>"
    messages[0]["content"] = compacted
    agent._apply_persist_user_message_override(messages)
    assert messages[0]["content"] == compacted


def test_nested_multimodal_mutation_does_not_mutate_the_frozen_anchor():
    messages = [{"role": "user", "content": [{"type": "text", "text": "ask"}]}]
    agent = _agent(0, [{"type": "text", "text": "clean"}])
    note_persist_user_anchor(agent, messages)
    messages[0]["content"][0]["text"] = "summary"
    agent._apply_persist_user_message_override(messages)
    assert messages[0]["content"][0]["text"] == "summary"


def test_missing_staged_row_and_index_drift_fail_closed():
    for initial_idx in (0, 99):
        messages = [{"role": "user", "content": "API ask"}]
        agent = _agent(initial_idx, "clean")
        note_persist_user_anchor(agent, messages)
        agent._persist_user_message_idx = 0
        messages[0] = {"role": "user", "content": "unrelated summary"}
        agent._apply_persist_user_message_override(messages)
        assert messages[0]["content"] == "unrelated summary"


def test_hermes_todo_fold_cannot_recertify_an_engine_replacement():
    from agent.conversation_compression import _fold_todo_snapshot
    from tools.todo_tool import TodoStore

    messages = [{"role": "user", "content": "API ask"}]
    agent = _agent(0, "clean")
    agent._todo_store = TodoStore()
    agent._todo_store.write([{"id": "1", "content": "run tests", "status": "pending"}])
    note_persist_user_anchor(agent, messages)
    messages[0]["content"] = "API ask\n\n<summary>keep</summary>"
    _fold_todo_snapshot(agent, messages)
    folded = messages[0]["content"]
    agent._apply_persist_user_message_override(messages)
    assert messages[0]["content"] == folded


def test_owned_survivor_does_not_certify_absorbed_context():
    messages = [{"role": "user", "content": "API ask"}, {"role": "user", "content": "context"}]
    agent = _agent(0, "clean")
    note_persist_user_anchor(agent, messages)
    assert repair_message_sequence(agent, messages) == 1
    agent._apply_persist_user_message_override(messages)
    assert messages[0]["content"] == "API ask\n\ncontext"


def test_a_following_summary_stays_separate_and_survives_the_override():
    from session_rules.turns import _merge_consecutive_users

    for ask in ("API ask", ""):
        messages = [
            {"role": "user", "content": ask},
            {"role": "user", "content": "<summary>KEEP</summary>", "_compressed_summary": True},
        ]
        agent = _agent(0, "clean")
        note_persist_user_anchor(agent, messages)
        merged, repairs = _merge_consecutive_users(messages, agent)
        agent._apply_persist_user_message_override(merged)
        assert any("KEEP" in row["content"] for row in merged)
        assert repairs == 0
        assert merged[0]["content"] == "clean"
        assert merged[1] is messages[1]


def test_an_equal_historical_row_cannot_refresh_or_adopt_ownership():
    from copy import deepcopy

    from agent.session_persistence import adopt_persist_user_anchor, refresh_persist_user_anchor

    for payload in ("ask", [{"type": "text", "text": "ask"}]):
        owner = {"role": "user", "content": deepcopy(payload)}
        historical = deepcopy(owner)
        agent = _agent(0, "clean")
        note_persist_user_anchor(agent, [owner])
        anchor = agent._persist_user_message_anchor
        historical["content"] = "historical context"
        refresh_persist_user_anchor(agent, historical, payload)
        assert agent._persist_user_message_anchor is anchor
        assert agent._persist_user_message_anchor_row is owner
        historical["content"] = deepcopy(payload)
        target = {"role": "user", "content": "merged history"}
        adopt_persist_user_anchor(agent, target, historical)
        assert agent._persist_user_message_anchor is anchor
        assert agent._persist_user_message_anchor_row is owner


def test_a_copied_unmarked_summary_keeps_its_bytes():
    from copy import deepcopy

    content = "API ask\n\n<summary>KEEP</summary>"
    messages = [{"role": "user", "content": content}]
    agent = _agent(0, "clean")
    note_persist_user_anchor(agent, messages)
    messages[0] = deepcopy(messages[0])
    messages[0]["content"] = content

    agent._apply_persist_user_message_override(messages)

    assert "KEEP" in messages[0]["content"]
    assert messages[0]["content"] == content


def test_a_copied_row_keeps_its_bytes():
    messages = [{"role": "user", "content": "API ask"}]
    agent = _agent(0, "clean")
    note_persist_user_anchor(agent, messages)
    messages[0] = dict(messages[0])
    agent._apply_persist_user_message_override(messages)
    assert messages[0]["content"] == "API ask"
