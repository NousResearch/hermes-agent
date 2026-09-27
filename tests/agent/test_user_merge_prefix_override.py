"""Regression tests for #124731: the persist user-message override must not
drop an earlier unanswered user message that alternation repair merged into
the current turn's user row.

A crash or restart between turn start and the reply leaves a user row R
durable in state.db with no assistant answer. When the next user message U
arrives, ``_merge_consecutive_users`` folds U into R's dict and re-anchoring
falls back to that merged row, so finalize's persist override rewrites the
merged row's content to U alone and R's bytes leave the live list (which ACP
hosts keep across turns, and which context-engine plugins compare at
``post_llm_call``).

The fix stamps the pre-merge content onto the merge survivor
(``USER_MERGE_PREFIX_KEY``) and re-attaches it when the override rewrites a
merged row. The DB row path keeps the plain override so the store stays in
its reload-idempotent two-row shape (``R`` + ``U``).
"""

from __future__ import annotations

from run_agent import AIAgent
from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import USER_MERGE_PREFIX_KEY, _DB_PERSISTED_MARKER
from agent.session_persistence import _post_override_content
from agent.turn_context import reanchor_current_turn_user_idx
from hermes_state import SessionDB


def _crash_orphan_history():
    """Rows as a restart loads them: the unanswered user row R, born durable."""
    return [
        {
            "role": "user",
            "content": "old task",
            _DB_PERSISTED_MARKER: True,
            "_row_id": 1,
        },
    ]


def _next_turn_user_content():
    """The live variant of U: gateway/ACP send the API-only bytes to the wire."""
    return "[timestamp 2026-09-27 03:12] deploy service"


def test_merge_pass_stamps_pre_merge_prefix():
    agent = AIAgent.__new__(AIAgent)
    r = _crash_orphan_history()[0]
    u = {"role": "user", "content": _next_turn_user_content()}
    messages = [r, u]

    repairs = repair_message_sequence(agent, messages)

    assert repairs == 1
    assert len(messages) == 1
    assert messages[0]["content"] == "old task\n\n" + _next_turn_user_content()
    assert messages[0][USER_MERGE_PREFIX_KEY] == "old task"
    assert _DB_PERSISTED_MARKER not in messages[0]


def test_chained_merge_prefix_is_full_pre_merge_content():
    agent = AIAgent.__new__(AIAgent)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
    ]

    repairs = repair_message_sequence(agent, messages)

    assert repairs == 2
    assert messages[0]["content"] == "first\n\nsecond\n\nthird"
    assert messages[0][USER_MERGE_PREFIX_KEY] == "first\n\nsecond"


def test_no_op_merge_leaves_no_prefix():
    """An empty incoming turn reproduces the survivor's bytes; nothing was merged
    away, so there is nothing for the override to re-attach."""
    agent = AIAgent.__new__(AIAgent)
    messages = [
        {"role": "user", "content": "kept"},
        {"role": "user", "content": ""},
    ]

    repair_message_sequence(agent, messages)

    assert messages[0]["content"] == "kept"
    assert USER_MERGE_PREFIX_KEY not in messages[0]


def test_plain_row_override_is_unaffected():
    agent = AIAgent.__new__(AIAgent)
    messages = [{"role": "user", "content": "[gateway note] observed\n\nactual question"}]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "actual question"
    agent._persist_user_message_timestamp = None

    agent._apply_persist_user_message_override(messages)

    assert messages[0]["content"] == "actual question"
    assert USER_MERGE_PREFIX_KEY not in messages[0]


def test_merged_row_override_reattaches_prefix():
    """The #124731 live-path regression: finalize's override on a repair-merged row
    must keep the pre-merge bytes in the live list."""
    agent = AIAgent.__new__(AIAgent)
    r = _crash_orphan_history()[0]
    u = {"role": "user", "content": _next_turn_user_content()}
    messages = [r, u]

    repair_message_sequence(agent, messages)
    # Re-anchor the way turn-iteration prep does after a repair shrank the list.
    agent._persist_user_message_idx = reanchor_current_turn_user_idx(
        messages, u["content"]
    )
    agent._persist_user_message_override = "deploy service"
    agent._persist_user_message_timestamp = 1730000000

    agent._apply_persist_user_message_override(messages)

    assert agent._persist_user_message_idx == 0
    assert messages[0]["content"] == "old task\n\ndeploy service"
    assert messages[0]["timestamp"] == 1730000000


def test_multimodal_override_on_merged_row_keeps_plain_replacement():
    """A list override is the clean multimodal payload; the text prefix is
    re-attachable only for str overrides, and a list turn never merges in the
    first place (the merge pass requires str content on both sides)."""
    agent = AIAgent.__new__(AIAgent)
    merged = {
        "role": "user",
        "content": "old task\n\nactual question",
        USER_MERGE_PREFIX_KEY: "old task",
    }
    clean_payload = [
        {"type": "text", "text": "Describe this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = clean_payload
    agent._persist_user_message_timestamp = None

    agent._apply_persist_user_message_override([merged])

    assert merged["content"] == clean_payload


def test_post_override_content_unit_behaviors():
    assert _post_override_content({USER_MERGE_PREFIX_KEY: "old"}, "clean") == "old\n\nclean"
    assert _post_override_content({}, "clean") == "clean"
    # Empty prefix: nothing was merged away.
    assert _post_override_content({USER_MERGE_PREFIX_KEY: ""}, "clean") == "clean"
    # No str-override re-attachment for list payloads.
    assert _post_override_content({USER_MERGE_PREFIX_KEY: "old"}, ["block"]) == ["block"]
    # A None override is the no-override case.
    assert _post_override_content({USER_MERGE_PREFIX_KEY: "old"}, None) is None


def test_crash_recovery_round_trip_live_and_reload_agree(tmp_path):
    """End to end: crash-orphan row R, next turn U. The turn-start flush writes
    U's row (plain override); repair then merges U into R's live dict and the
    finalize override must keep R's bytes in the live list; the reload-time
    repair of the two store rows must reproduce that live slot, and the store
    keeps its idempotent two-user-row shape (never a merged third row)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-crash-recover"
    db.create_session(session_id=sid, source="cli")
    try:
        # Pre-crash: R was flushed at its own turn start and never answered.
        db.append_message(sid, "user", content="old task")

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True

        # Restart: the gateway/ACP host reloads the history for the next turn.
        history = db.get_messages_as_conversation(
            sid, repair_alternation=False, include_row_ids=True
        )
        assert [m["content"] for m in history] == ["old task"]

        # Turn prologue: append U (API-only live bytes) and anchor the persist override.
        u_live = _next_turn_user_content()
        messages = list(history) + [{"role": "user", "content": u_live}]
        agent._persist_user_message_idx = len(messages) - 1
        agent._persist_user_message_override = "deploy service"
        agent._persist_user_message_timestamp = None

        # The turn-start flush writes U's row with the plain override content;
        # the wire bytes ride the api_content sidecar.
        agent._flush_messages_to_session_db(messages, history)
        rows = db.get_messages_as_conversation(sid, repair_alternation=False)
        assert [m["content"] for m in rows if m.get("role") == "user"] == [
            "old task",
            "deploy service",
        ]
        assert rows[1].get("api_content") == u_live

        # Iteration prep: alternation repair merges U into R's dict and re-anchors.
        assert repair_message_sequence(agent, messages) == 1
        agent._persist_user_message_idx = reanchor_current_turn_user_idx(
            messages, u_live
        )
        assert agent._persist_user_message_idx == 0

        # The model answers and the reply is appended before finalize.
        messages.append({"role": "assistant", "content": "service deployed"})

        # finalize_turn's override step, then the persist flush (history passed so
        # the loaded rows stay skipped, exactly like the host's finalize call).
        agent._apply_persist_user_message_override(messages)
        agent._flush_messages_to_session_db(messages, history)

        live = messages[0]["content"]
        assert live == "old task\n\ndeploy service", (
            "R's bytes left the live list; ACP hosts keep this list across turns"
        )

        # Store shape: R's row and U's row stay separate active rows — the merged
        # row was history-skipped, so no merged third row can duplicate R later.
        active = db.get_messages_as_conversation(sid, repair_alternation=False)
        assert [m["content"] for m in active if m.get("role") == "user"] == [
            "old task",
            "deploy service",
        ]

        # Reload: the load-time repair reproduces the live slot byte-for-byte.
        reloaded = db.get_messages_as_conversation(
            sid, repair_alternation=True, include_row_ids=True
        )
        user_rows = [m for m in reloaded if m.get("role") == "user"]
        assert user_rows[0]["content"] == live

        # Second reload: still idempotent.
        reloaded_again = db.get_messages_as_conversation(
            sid, repair_alternation=True, include_row_ids=True
        )
        user_rows_again = [m for m in reloaded_again if m.get("role") == "user"]
        assert user_rows_again[0]["content"] == live
    finally:
        db.close()
