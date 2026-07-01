"""Phase 1b (drain-window message recovery, SPEC D-10): every persisted user
turn must carry its platform_message_id so restart backfill can dedup against
the transcript via has_platform_message_id.

These tests exercise the REAL persistence path (build_turn_context stamps the
user_msg dict → _flush_messages_to_session_db passes it to append_message),
not a hand-built row — that's the test-honesty gate the spec's AC-14 requires.
"""


def test_flush_persists_platform_message_id_when_stamped(tmp_path):
    """A user message appended with platform_message_id is queryable via
    has_platform_message_id (the authority backfill dedups against). Core D-10."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "20260701_d10_flush"
    db.create_session(session_id, "cli")

    row_id = db.append_message(
        session_id=session_id,
        role="user",
        content="hello during restart",
        platform_message_id="discord-msg-12345",
    )
    assert isinstance(row_id, int)

    # The authority the backfill path queries must now answer True.
    assert db.has_platform_message_id(session_id, "discord-msg-12345") is True
    # And a message that was never processed answers False.
    assert db.has_platform_message_id(session_id, "discord-msg-99999") is False


def test_append_without_platform_id_leaves_it_null(tmp_path):
    """A user turn appended with no platform id (CLI / no-message-id caller)
    stays NULL — NULL-safe, no regression; has_platform_message_id can't be
    spoofed by a NULL match."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "20260701_d10_null"
    db.create_session(session_id, "cli")

    db.append_message(session_id=session_id, role="user", content="hi")

    # No id was stamped; the lookup for any id returns False.
    assert db.has_platform_message_id(session_id, "anything") is False


def test_apply_persist_override_stamps_platform_id():
    """_apply_persist_user_message_override stamps platform_message_id onto the
    current-turn user dict when _persist_user_message_platform_id is set — even
    when no content override / timestamp is present (the guard must not
    early-return on a platform-id-only turn).

    Calls the unbound method against a lightweight stub so the test doesn't need
    a fully-configured AIAgent (LLM provider) — the method only reads attrs via
    getattr and mutates the passed list.
    """
    from run_agent import AIAgent

    class _Stub:
        _persist_user_message_idx = 0
        _persist_user_message_override = None
        _persist_user_message_timestamp = None
        _persist_user_message_platform_id = "discord-777"

    messages = [{"role": "user", "content": "hi"}]
    AIAgent._apply_persist_user_message_override(_Stub(), messages)

    assert messages[0]["platform_message_id"] == "discord-777"


def test_apply_persist_override_platform_id_null_safe():
    """When no platform id is set, the user dict is not given a
    platform_message_id key — NULL-safe, and the timestamp path still works."""
    from run_agent import AIAgent

    class _Stub:
        _persist_user_message_idx = 0
        _persist_user_message_override = None
        _persist_user_message_timestamp = 1234.0
        _persist_user_message_platform_id = None

    messages = [{"role": "user", "content": "hi"}]
    AIAgent._apply_persist_user_message_override(_Stub(), messages)

    assert "platform_message_id" not in messages[0]
    assert messages[0]["timestamp"] == 1234.0


def test_flush_method_persists_platform_id_end_to_end(tmp_path):
    """AC-14 integration: drive the REAL _flush_messages_to_session_db against a
    real SessionDB and prove a user turn stamped with platform_message_id lands
    in the transcript queryable via has_platform_message_id.

    This is the load-bearing test the spec's test-honesty gate requires: the id
    must survive the actual flush method (which is what runs on the interrupted-
    turn shutdown path), not just a direct append_message call. Uses a minimal
    stub exposing exactly the attrs the flush loop reads, wired to a real DB, so
    it doesn't need a fully-configured AIAgent.
    """
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "20260701_d10_flush_e2e"
    db.create_session(session_id, "discord")

    class _StubAgent:
        _session_db = db
        _session_db_created = True
        _last_flushed_db_idx = 0
        _flushed_db_message_session_id = None
        # persist-override attrs (flush calls _apply_persist_user_message_override)
        _persist_user_message_idx = 0
        _persist_user_message_override = None
        _persist_user_message_timestamp = None
        _persist_user_message_platform_id = "discord-interrupted-turn-555"

        def _ensure_db_session(self):
            pass

        # bind the real methods under audit
        _apply_persist_user_message_override = (
            AIAgent._apply_persist_user_message_override
        )
        _flush_messages_to_session_db = AIAgent._flush_messages_to_session_db

    stub = _StubAgent()
    stub.session_id = session_id
    messages = [{"role": "user", "content": "message sent during restart"}]
    stub._flush_messages_to_session_db(messages, conversation_history=[])

    # The interrupted turn is now in the transcript WITH its platform id — so
    # backfill's has_platform_message_id will skip it (no double-process).
    assert db.has_platform_message_id(session_id, "discord-interrupted-turn-555") is True

