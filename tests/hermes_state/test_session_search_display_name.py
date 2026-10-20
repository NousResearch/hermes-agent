"""Session search matches the gateway channel/thread name, not just the title.

``display_name`` is the gateway's presentation string for a messaging origin
("Server / #channel / thread"). Users remember a gateway conversation by the
channel it lives in far more reliably than by whatever title it ended up with,
so ``list_sessions_rich(search_query=...)`` has to match it.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "search_state.db")
    yield session_db
    session_db.close()


def make_gateway_session(
    db,
    session_id,
    *,
    source="discord",
    title=None,
    display_name=None,
    text="some ordinary conversation text",
):
    """Create a session the way the gateway does: row, message, origin, title."""
    db.create_session(session_id, source)
    db.append_message(session_id, "user", text)
    if display_name is not None:
        db.record_gateway_session_peer(
            session_id,
            source=source,
            session_key=f"key:{session_id}",
            display_name=display_name,
        )
    if title is not None:
        db.set_session_title(session_id, title)
    return session_id


def search(db, needle, **kwargs):
    rows = db.list_sessions_rich(order_by_last_active=True, search_query=needle, **kwargs)
    return [row["id"] for row in rows]


# ---------------------------------------------------------------------------
# The gap: channel/thread names were unsearchable
# ---------------------------------------------------------------------------


def test_a_session_is_findable_by_its_channel_name(db):
    """The regression this fixes.

    A conversation in "Acme Guild / #finance" whose title says nothing about
    finance was previously unreachable from search: the title lane missed it,
    and message FTS only covers message text.
    """
    make_gateway_session(
        db,
        "s_channel",
        title="Quarterly Budget Review",
        display_name="Acme Guild / #finance",
        text="numbers and projections",
    )

    assert search(db, "finance") == ["s_channel"]
    assert search(db, "Acme Guild") == ["s_channel"]


def test_channel_matching_is_case_insensitive(db):
    make_gateway_session(db, "s_case", display_name="Acme Guild / #Voice-Assistant")

    assert search(db, "VOICE") == ["s_case"]
    assert search(db, "acme guild") == ["s_case"]


def test_a_thread_name_is_searchable_the_same_way(db):
    """display_name carries server / channel / thread; all three must match."""
    make_gateway_session(
        db, "s_thread", display_name="Acme Guild / #general / deploy-postmortem"
    )

    assert search(db, "postmortem") == ["s_thread"]


def test_the_punctuation_stripped_variant_applies_to_channel_names_too(db):
    """`an94` already finds `AN-94` in a title; the same must hold for a channel.

    The compact variant is what makes hyphenated channel names findable the way
    users type them.
    """
    make_gateway_session(db, "s_compact", display_name="Acme Guild / #an-94-ops")

    assert search(db, "an94") == ["s_compact"]


# ---------------------------------------------------------------------------
# The existing lanes must not regress
# ---------------------------------------------------------------------------


def test_title_matching_still_works(db):
    make_gateway_session(db, "s_title", title="Quarterly Budget Review")

    assert search(db, "Quarterly") == ["s_title"]
    assert search(db, "budget") == ["s_title"]


def test_id_matching_still_works(db):
    make_gateway_session(db, "s_distinctive_id", title="Untitled")

    assert search(db, "distinctive") == ["s_distinctive_id"]


def test_a_session_with_no_display_name_is_unaffected(db):
    """A CLI session has display_name NULL; COALESCE must not make it match all."""
    make_gateway_session(db, "s_cli", source="cli", title="Local Work")

    assert search(db, "Local") == ["s_cli"]
    assert search(db, "anything-else-entirely") == []


def test_search_still_narrows_rather_than_returning_everything(db):
    """A widened clause must not degenerate into matching every row."""
    make_gateway_session(db, "s_a", title="Alpha", display_name="Server / #alpha-room")
    make_gateway_session(db, "s_b", title="Beta", display_name="Server / #beta-room")
    make_gateway_session(db, "s_c", title="Gamma", display_name="Server / #gamma-room")

    assert search(db, "beta") == ["s_b"]
    assert search(db, "zzz-matches-nothing") == []
    assert sorted(search(db, "room")) == ["s_a", "s_b", "s_c"]


def test_like_wildcards_in_the_needle_are_escaped_not_interpreted(db):
    """A user typing `%` must match a literal percent sign, not every session.

    An unescaped `%` in the widened display_name clause would turn the whole
    predicate into a tautology and return the entire session table.
    """
    make_gateway_session(db, "alpha", title="Plain Title", display_name="Server / #room")
    make_gateway_session(db, "beta", display_name="Server / #100%-uptime")

    assert search(db, "%") == ["beta"]
    assert search(db, "_") == []


def test_the_needle_matches_display_name_across_a_compression_chain(db):
    """Search admits a root whose forward chain matches — channel lane included.

    The clause walks the same ``chain`` CTE the title and id lanes use, so a
    conversation that compressed onto a continuation still surfaces under its
    channel name from the root row.
    """
    root = make_gateway_session(db, "s_root", title="Old Root")
    db.end_session(root, end_reason="compression")
    db.create_session("s_tip", "discord", parent_session_id=root)
    db.append_message("s_tip", "user", "continued")
    db.record_gateway_session_peer(
        "s_tip",
        source="discord",
        session_key="key:s_tip",
        display_name="Acme Guild / #late-named-channel",
    )

    # The root is what the query admits; the default tip projection then
    # surfaces it as the live continuation, exactly as the title lane does.
    (row,) = db.list_sessions_rich(
        order_by_last_active=True, search_query="late-named-channel"
    )

    assert row["id"] == "s_tip"
    assert row["_lineage_root_id"] == "s_root"


def test_display_name_survives_to_the_returned_row(db):
    """Callers ranking a channel hit need the value that matched."""
    make_gateway_session(db, "s_payload", display_name="Acme Guild / #finance")

    (row,) = db.list_sessions_rich(order_by_last_active=True, search_query="finance")

    assert row["display_name"] == "Acme Guild / #finance"
