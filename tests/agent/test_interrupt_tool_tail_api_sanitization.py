"""Interrupted tool-tail closure stays API-only."""

from agent.agent_runtime_helpers import sanitize_api_messages


def _tool_tail(*, interrupted=False):
    tool_result = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "ok edited",
    }
    if interrupted:
        tool_result["_interrupted_tool_tail"] = True
    return [
        {"role": "user", "content": "edit the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "patch", "arguments": "{}"}}
            ],
        },
        tool_result,
    ]


def test_user_after_interrupted_tool_tail_is_closed_only_in_api_copy():
    canonical = _tool_tail(interrupted=True) + [
        {"role": "user", "content": "do something else"}
    ]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert [message["role"] for message in canonical[-2:]] == ["tool", "user"]
    assert canonical[-2]["_interrupted_tool_tail"] is True
    assert [message["role"] for message in wire[-3:]] == [
        "tool",
        "assistant",
        "user",
    ]
    assert wire[-2]["content"] == "Operation interrupted."
    assert all("_interrupted_tool_tail" not in message for message in wire)


def test_normal_user_redirect_reaches_api_copy_unchanged():
    canonical = _tool_tail() + [{"role": "user", "content": "do something else"}]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert wire == canonical


def test_real_partial_assistant_text_remains_the_closure():
    canonical = _tool_tail() + [
        {"role": "assistant", "content": "Partial answer so far"},
        {"role": "user", "content": "do something else"},
    ]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert wire == canonical
    assert wire[-2]["content"] == "Partial answer so far"


def test_legacy_synthetic_interrupt_sentinel_remains_compatible():
    canonical = _tool_tail() + [
        {"role": "assistant", "content": "Operation interrupted."},
        {"role": "user", "content": "do something else"},
    ]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert wire == canonical


def test_tool_tail_without_followup_is_unchanged():
    canonical = _tool_tail()

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert wire == canonical


# ---------------------------------------------------------------------------
# Durable provenance round-trip (quad-review gap: the DB layer had zero
# coverage — a mutant binding 0 / skipping replay passed the whole suite).
# ---------------------------------------------------------------------------
def test_interrupted_tool_tail_survives_db_roundtrip(tmp_path):
    """Flush → persist → reload must carry interrupted_tool_tail=True: the
    provenance is DURABLE canonical state, not an in-memory-only flag."""
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="tui")
    db.append_message(
        "s1", role="tool", content="partial result", tool_call_id="call-1",
    )
    assert db.mark_tool_tail_interrupted("s1", "call-1") is True

    # Zero-row match returns False (unknown id / not-yet-flushed row) —
    # the flush-side guard relies on this to stay eligible for back-stamp.
    assert db.mark_tool_tail_interrupted("s1", "missing-call") is False

    reloaded = db.get_messages_as_conversation("s1", include_ancestors=False)
    assert reloaded and reloaded[0].get("_interrupted_tool_tail") is True


def test_interrupted_tool_tail_column_upgrades_legacy_db(tmp_path):
    """A state.db written BEFORE this PR (no interrupted_tool_tail column)
    reconciles on open: the column is added and legacy rows default to 0."""
    from hermes_state import SessionDB

    path = tmp_path / "legacy.db"
    db = SessionDB(path)
    db.create_session("s1", source="tui")
    db.append_message("s1", role="user", content="hi")
    db.close()

    # Simulate the pre-PR on-disk shape: drop the column the DDL just added
    # (a plain DROP COLUMN keeps triggers/indexes intact, unlike a rebuild).
    con = __import__("sqlite3").connect(path)
    con.execute("PRAGMA writable_schema = OFF")
    con.execute("ALTER TABLE messages DROP COLUMN interrupted_tool_tail")
    con.commit()
    con.close()

    reopened = SessionDB(path)  # _reconcile_columns must re-ADD the column
    cols = {row[1] for row in reopened._conn.execute("PRAGMA table_info(messages)")}
    assert "interrupted_tool_tail" in cols
    reloaded = reopened.get_messages("s1")
    assert reloaded and reloaded[0].get("_interrupted_tool_tail") in (None, False)
