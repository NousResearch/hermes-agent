"""The canonical Bot Chat's title is its identity: renames must be refused.

Bot Mode resolves a bot's forever-chat by exact-title lookup on
(profile, "Bot Chat") every time it opens; there is no session-id pointer.
A user rename therefore orphans the whole conversation: resolution misses,
the next click mints an empty replacement, and UNIQUE(title) then blocks
renaming the original back (#92473).

The guard lives in the session repository's title write, the single write path
every rename surface funnels through (gateway session.title RPC, /title,
CLI rename, REST), and keys on hidden + exact canonical title so ordinary
sessions a user happens to call "Bot Chat" stay freely renameable.
"""
import pytest

from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE, SessionQuery, SessionWindow, TitleSource
from persistence.housekeeping.maintenance import PruneCriteria
from persistence_assembly import open_persistence, profile_scope
from session_retention import archive_sessions
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


@pytest.fixture
def db(tmp_path):
    store = open_persistence(profile_scope(home=tmp_path))
    yield store
    store.close()


def _write_sql(db, sql, params=()):
    raw_holder(db)._execute_write(lambda conn: conn.execute(sql, params))


def _by_title(db, title):
    record = db.sessions.get_by_title(title)
    return None if record is None else record.as_session_row()


def _make_canonical(db, session_id="forever"):
    seed_session(db, session_id, "desktop")
    assert db.sessions.set_title(session_id, CANONICAL_BOT_CHAT_TITLE)
    assert db.sessions.set_hidden(session_id, True)
    return session_id


def test_user_rename_of_canonical_bot_chat_is_refused(db):
    sid = _make_canonical(db)
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        db.sessions.set_title(sid, "My cool chat")
    # Identity intact: exact-title lookup still finds the forever chat.
    row = _by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_clearing_the_canonical_title_is_refused(db):
    sid = _make_canonical(db)
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        db.sessions.set_title(sid, "")
    row = _by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_rewriting_the_same_canonical_title_is_a_noop_not_an_error(db):
    # The plugin's eager session.title write re-asserts the canonical title
    # on creation paths; that must never start failing.
    sid = _make_canonical(db)
    assert db.sessions.set_title(sid, CANONICAL_BOT_CHAT_TITLE)


def test_visible_session_titled_bot_chat_stays_renameable(db):
    # hidden discriminates the registry row: a normal visible session the
    # user happened to name "Bot Chat" is not canonical and renames freely.
    seed_session(db, "ordinary", "cli")
    assert db.sessions.set_title("ordinary", CANONICAL_BOT_CHAT_TITLE)
    assert db.sessions.set_title("ordinary", "renamed away")
    assert session_row(db, "ordinary")["title"] == "renamed away"


def test_deliberately_archived_canonical_chat_releases_name_for_replacement(db):
    """Retiring a Bot Chat must not leave its registry name permanently locked.

    Title is NULLed inside ``set_archived`` itself: Desktop never calls
    ``set_title`` on a replacement while lookup still hits the retired row.
    """
    retired = _make_canonical(db, "retired")
    assert db.sessions.set_archived(retired, True).changed

    assert session_row(db, retired)["archived"]
    assert session_row(db, retired)["title"] is None
    assert db.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE) is None
    assert db.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) is None

    seed_session(db, "replacement", "desktop")
    assert db.sessions.set_title("replacement", CANONICAL_BOT_CHAT_TITLE)
    assert db.sessions.set_hidden("replacement", True)

    # The retired row stays archived, while exact-title resolution reaches the
    # replacement that Bot Mode will subsequently open.
    assert session_row(db, retired)["archived"]
    assert session_row(db, retired)["title"] is None
    row = _by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == "replacement"


def _archive_stale_sessions(db, idle_days):
    """The idle sweep: archive every idle lineage tip the maintenance repository names."""
    ids = db.maintenance.list_idle_archive_candidates(idle_days)
    for session_id in ids:
        db.maintenance.auto_archive_lineage(session_id)
    return len(ids)


def test_auto_archive_sweep_skips_the_canonical_chat(db):
    """Only a deliberate archive may retire a Bot Chat; the idle sweep must not
    (it would strand an unrecoverable, soon-to-be-untitled row)."""
    import time

    forever = _make_canonical(db)
    seed_session(db, "ordinary", "desktop")
    stale = time.time() - 10 * 86400
    _write_sql(db, "UPDATE sessions SET started_at = ?, last_activity_at = ?", (stale, stale))

    assert _archive_stale_sessions(db, 3) == 1
    assert session_row(db, "ordinary")["archived"]
    assert not session_row(db, forever)["archived"]
    assert session_row(db, forever)["title"] == CANONICAL_BOT_CHAT_TITLE


def test_auto_titler_still_cannot_touch_the_canonical_row(db):
    # Pre-existing provenance contract, re-pinned here: user-authority title
    # outranks derived/llm, so the turn-start auto-titler can never displace
    # the registry name.
    sid = _make_canonical(db)
    assert not db.sessions.set_auto_title(sid, "Chat about groceries", source=TitleSource.LLM)
    row = _by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_auto_titler_cannot_rename_derived_canonical_bot_chat(db):
    # #99517: the guard must be provenance-blind. A derived (rank 0) canonical
    # title loses to an llm (rank 1) auto-title on precedence alone, so the
    # identity check, not precedence, has to stop the write.
    seed_session(db, "derived", "desktop")
    assert db.sessions.set_auto_title(
        "derived",
        CANONICAL_BOT_CHAT_TITLE,
        source=TitleSource.DERIVED,
    )
    assert db.sessions.set_hidden("derived", True)

    assert not db.sessions.set_auto_title(
        "derived",
        "Renamed by titler",
        source=TitleSource.LLM,
    )
    row = session_row(db, "derived")
    assert row["title"] == CANONICAL_BOT_CHAT_TITLE
    assert row["title_source"] == TitleSource.DERIVED.value


def test_auto_titler_can_rename_visible_derived_bot_chat(db):
    # Control: hidden is still the discriminator: a visible session that
    # merely carries the text "Bot Chat" upgrades derived -> llm as usual.
    seed_session(db, "visible", "desktop")
    assert db.sessions.set_auto_title(
        "visible",
        CANONICAL_BOT_CHAT_TITLE,
        source=TitleSource.DERIVED,
    )

    assert db.sessions.set_auto_title(
        "visible",
        "Renamed by titler",
        source=TitleSource.LLM,
    )
    assert session_row(db, "visible")["title"] == "Renamed by titler"


def test_numbered_branch_never_shadows_the_canonical_chat(db):
    """Title resolution must land on the canonical Bot Chat, never on a "#N" sibling.

    Every DM transport resolves the target bot by that exact name
    (``hermes -p <bot> chat --in ~ -c "Bot Chat"``: message_agent, bot_relay, cron
    delivery). A numbered sibling (a Desktop branch, which is visible and NOT
    Bot-Mode-managed) would otherwise swallow each teammate's message into a
    session whose bot has no message_agent and cannot answer.
    """
    canonical = _make_canonical(db, "forever")
    seed_session(db, "branch", "desktop", parent_session_id=canonical)
    assert db.sessions.set_auto_title(
        "branch",
        f"{CANONICAL_BOT_CHAT_TITLE} #2",
        source=TitleSource.DERIVED,
    )
    assert db.sessions.set_hidden("branch", False)

    assert db.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) == canonical


def test_numbered_continuation_still_wins_for_ordinary_titles(db):
    # Control: the "#N continuation" preference is the point of the lookup: it
    # stays intact for every non-Bot-Chat title.
    seed_session(db, "plain", "cli")
    assert db.sessions.set_title("plain", "My session")
    seed_session(db, "plain2", "cli")
    assert db.sessions.set_title("plain2", "My session #2")

    assert db.sessions.resolve_by_title("My session") == "plain2"


def _stamp_legacy_archive(db, session_id, *, end_reason=None):
    """An archive written before the name release: ``archived=1`` while the row still holds its title."""
    if end_reason is None:
        _write_sql(
            db,
            "UPDATE sessions SET archived = 1, auto_archived = 0 WHERE id = ?",
            (session_id,),
        )
        return
    _write_sql(
        db,
        "UPDATE sessions SET archived = 1, auto_archived = 0, end_reason = ? WHERE id = ?",
        (end_reason, session_id),
    )


def test_title_lookup_skips_legacy_deliberately_archived_hidden_bot_chat(db):
    """Live evidence: retired rows still hold title='Bot Chat'. Lookup must skip them."""
    retired = _make_canonical(db, "retired")
    _stamp_legacy_archive(db, retired)
    assert session_row(db, retired)["title"] == CANONICAL_BOT_CHAT_TITLE
    assert session_row(db, retired)["archived"]
    assert db.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE) is None
    assert db.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) is None


def test_archived_canonical_does_not_fall_through_to_numbered_sibling(db):
    canonical = _make_canonical(db, "forever")
    seed_session(db, "branch", "desktop", parent_session_id=canonical)
    assert db.sessions.set_auto_title(
        "branch",
        f"{CANONICAL_BOT_CHAT_TITLE} #2",
        source=TitleSource.DERIVED,
    )
    assert db.sessions.set_hidden("branch", False)
    _stamp_legacy_archive(db, canonical)
    assert db.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) is None


@pytest.mark.parametrize("reason", ["ws_orphan_reap", "agent_close"])
def test_recoverable_auto_archive_keeps_title_and_resurrects(db, reason):
    """#92687: orphan-reaper archives must stay findable and unarchive_recoverable."""
    sid = _make_canonical(db, "reaped")
    db.sessions.end(sid, reason)
    _stamp_legacy_archive(db, sid, end_reason=reason)
    assert session_row(db, sid)["title"] == CANONICAL_BOT_CHAT_TITLE
    row = _by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid
    assert db.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) == sid
    assert db.sessions.unarchive_recoverable(sid)
    fresh = session_row(db, sid)
    assert not fresh["archived"]
    assert _by_title(db, CANONICAL_BOT_CHAT_TITLE)["id"] == sid


def test_legacy_recoverable_archive_still_found_by_title(db):
    """REST resurrection uses the title lookup to find the archived row."""
    sid = _make_canonical(db, "reaped")
    _stamp_legacy_archive(db, sid, end_reason="ws_orphan_reap")
    row = _by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid and row["archived"]
    assert db.sessions.unarchive_recoverable(sid)
    assert not session_row(db, sid)["archived"]


def test_title_lookup_finds_recoverable_compressed_bot_chat(db):
    """Title lives on the compression root; recoverability is on the tip (#92687)."""
    seed_session(db, "root1", "desktop")
    assert db.sessions.set_title("root1", CANONICAL_BOT_CHAT_TITLE)
    assert db.sessions.set_hidden("root1", True)
    db.sessions.end("root1", "compression")
    seed_session(db, "tip1", "desktop", parent_session_id="root1")
    db.sessions.end("tip1", "ws_orphan_reap")
    _write_sql(db, "UPDATE sessions SET archived = 1 WHERE id IN ('root1', 'tip1')")
    row = _by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == "root1"
    assert db.sessions.unarchive_recoverable(row["id"])
    assert not session_row(db, "root1")["archived"]
    assert not session_row(db, "tip1")["archived"]


@pytest.mark.parametrize("already_archived", [False, True])
def test_deliberate_archive_of_reaped_bot_chat_retires_the_name(db, already_archived):
    """#139779: the reaper's recoverable stamp keeps "Bot Chat" alive for resurrection, but the
    user's Archive means "done with this chat" even then: the name is released and nothing
    resurrects the old row, so the next bot click mints a fresh "Bot Chat"."""
    sid = _make_canonical(db, "reaped")
    db.sessions.end(sid, "ws_orphan_reap")
    if already_archived:
        _stamp_legacy_archive(db, sid, end_reason="ws_orphan_reap")
    assert db.sessions.set_archived(sid, True).changed
    row = session_row(db, sid)
    assert row["archived"] and row["title"] is None and row["end_reason"] is None
    assert db.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE) is None
    assert not db.sessions.unarchive_recoverable(sid)


def test_archived_ordinary_title_still_resolves(db):
    """Only the canonical "Bot Chat" retires on archive: ``-c "<title>"``, ``/resume <title>`` and
    ``--create-if-missing`` keep reopening any other user-archived session (main's behaviour)."""
    seed_session(db, "proj", "cli")
    assert db.sessions.set_title("proj", "My project")
    assert db.sessions.set_archived("proj", True).changed
    assert _by_title(db, "My project")["id"] == "proj"
    assert db.sessions.resolve_by_title("My project") == "proj"
    assert session_row(db, "proj")["title"] == "My project"


def test_bulk_archive_spares_bot_chat_and_unarchived_retired_chat_is_listed(db):
    """``hermes sessions archive`` is housekeeping like the idle sweep: it never retires the
    canonical chat. A per-chat Archive does, and unarchiving the retired chat returns it as an
    ordinary visible conversation; it never takes "Bot Chat" back."""
    sid = _make_canonical(db, "forever")
    db.sessions.end(sid, "cli_close")
    assert archive_sessions(db, db.scope.profile_dir, criteria=PruneCriteria(include_open=True)) == 0
    assert session_row(db, sid)["title"] == CANONICAL_BOT_CHAT_TITLE
    assert db.sessions.set_archived(sid, True).changed
    assert db.sessions.set_archived(sid, False).changed
    row = session_row(db, sid)
    assert (row["title"], row["hidden"], row["archived"]) == (None, 0, 0)
    assert sid in [entry.session.session_id for entry in db.sessions.list_sessions(SessionQuery(), SessionWindow(limit=50))]
