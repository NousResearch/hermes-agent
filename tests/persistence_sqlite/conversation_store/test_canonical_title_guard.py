"""The canonical Bot Chat's title is its identity: renames must be refused.

Bot Mode resolves a bot's forever-chat by exact-title lookup on
(profile, "Bot Chat") every time it opens; there is no session-id pointer.
A user rename therefore orphans the whole conversation: resolution misses,
the next click mints an empty replacement, and UNIQUE(title) then blocks
renaming the original back (#92473).

The guard lives in the backend's ``_set_session_title``, the single write path
every rename surface funnels through (gateway session.title RPC, /title,
CLI rename, REST), and keys on hidden + exact canonical title so ordinary
sessions a user happens to call "Bot Chat" stay freely renameable.
"""
import pytest

from persistence.sqlite._state.common import TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM
from persistence.sqlite.conversation._store.core import CANONICAL_BOT_CHAT_TITLE
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    get_session,
    set_session_archived,
    set_session_hidden,
)
from persistence.sqlite.conversation._store.titles import (
    _set_session_title,
    get_session_by_title,
    resolve_session_by_title,
    set_auto_title,
    set_session_title,
)


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    return store.holder()


def _make_canonical(db, session_id="forever"):
    create_session(db, session_id, source="desktop")
    assert set_session_title(db, session_id, CANONICAL_BOT_CHAT_TITLE)
    assert set_session_hidden(db, session_id, True)
    return session_id


def test_user_rename_of_canonical_bot_chat_is_refused(db):
    sid = _make_canonical(db)
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        set_session_title(db, sid, "My cool chat")
    # Identity intact: exact-title lookup still finds the forever chat.
    row = get_session_by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_clearing_the_canonical_title_is_refused(db):
    sid = _make_canonical(db)
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        set_session_title(db, sid, "")
    row = get_session_by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_rewriting_the_same_canonical_title_is_a_noop_not_an_error(db):
    # The plugin's eager session.title write re-asserts the canonical title
    # on creation paths; that must never start failing.
    sid = _make_canonical(db)
    assert set_session_title(db, sid, CANONICAL_BOT_CHAT_TITLE)


def test_visible_session_titled_bot_chat_stays_renameable(db):
    # hidden discriminates the registry row: a normal visible session the
    # user happened to name "Bot Chat" is not canonical and renames freely.
    create_session(db, "ordinary", source="cli")
    assert set_session_title(db, "ordinary", CANONICAL_BOT_CHAT_TITLE)
    assert set_session_title(db, "ordinary", "renamed away")
    assert get_session(db, "ordinary")["title"] == "renamed away"


def test_deliberately_archived_canonical_chat_releases_name_for_replacement(db):
    """Retiring a Bot Chat must not leave its registry name permanently locked."""
    retired = _make_canonical(db, "retired")
    assert set_session_archived(db, retired, True)[0]

    create_session(db, "replacement", source="desktop")
    assert set_session_title(db, "replacement", CANONICAL_BOT_CHAT_TITLE)
    assert set_session_hidden(db, "replacement", True)

    # The retired row stays archived, while exact-title resolution reaches the
    # replacement that Bot Mode will subsequently open.
    assert get_session(db, retired)["archived"]
    assert get_session(db, retired)["title"] is None
    row = get_session_by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == "replacement"


def _archive_stale_sessions(store, idle_days):
    """The idle sweep: archive every idle lineage tip the maintenance repository names."""
    ids = store.maintenance.list_idle_archive_candidates(idle_days)
    for session_id in ids:
        store.maintenance.auto_archive_lineage(session_id)
    return len(ids)


def test_auto_archive_sweep_skips_the_canonical_chat(store, db):
    """Only a deliberate archive may retire a Bot Chat; the idle sweep must not
    (it would strand an unrecoverable, soon-to-be-untitled row)."""
    import time

    forever = _make_canonical(db)
    create_session(db, "ordinary", source="desktop")
    stale = time.time() - 10 * 86400
    db._write_sql("UPDATE sessions SET started_at = ?, last_activity_at = ?", (stale, stale))

    assert _archive_stale_sessions(store, 3) == 1
    assert get_session(db, "ordinary")["archived"]
    assert not get_session(db, forever)["archived"]
    assert get_session(db, forever)["title"] == CANONICAL_BOT_CHAT_TITLE


def test_auto_titler_still_cannot_touch_the_canonical_row(db):
    # Pre-existing provenance contract, re-pinned here: user-authority title
    # outranks derived/llm, so the turn-start auto-titler can never displace
    # the registry name.
    sid = _make_canonical(db)
    assert not set_auto_title(db, sid, "Chat about groceries", source=TITLE_SOURCE_LLM)
    row = get_session_by_title(db, CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_auto_titler_cannot_rename_derived_canonical_bot_chat(db):
    # #99517: the guard must be provenance-blind. A derived (rank 0) canonical
    # title loses to an llm (rank 1) auto-title on precedence alone, so the
    # identity check, not precedence, has to stop the write.
    create_session(db, "derived", source="desktop")
    assert _set_session_title(
        db,
        "derived",
        CANONICAL_BOT_CHAT_TITLE,
        source=TITLE_SOURCE_DERIVED,
    )
    assert set_session_hidden(db, "derived", True)

    assert not set_auto_title(
        db,
        "derived",
        "Renamed by titler",
        source=TITLE_SOURCE_LLM,
    )
    row = get_session(db, "derived")
    assert row["title"] == CANONICAL_BOT_CHAT_TITLE
    assert row["title_source"] == TITLE_SOURCE_DERIVED


def test_auto_titler_can_rename_visible_derived_bot_chat(db):
    # Control: hidden is still the discriminator: a visible session that
    # merely carries the text "Bot Chat" upgrades derived -> llm as usual.
    create_session(db, "visible", source="desktop")
    assert _set_session_title(
        db,
        "visible",
        CANONICAL_BOT_CHAT_TITLE,
        source=TITLE_SOURCE_DERIVED,
    )

    assert set_auto_title(
        db,
        "visible",
        "Renamed by titler",
        source=TITLE_SOURCE_LLM,
    )
    assert get_session(db, "visible")["title"] == "Renamed by titler"


def test_numbered_branch_never_shadows_the_canonical_chat(db):
    """Title resolution must land on the canonical Bot Chat, never on a "#N" sibling.

    Every DM transport resolves the target bot by that exact name
    (``hermes -p <bot> chat --in ~ -c "Bot Chat"``: message_agent, bot_relay, cron
    delivery). A numbered sibling (a Desktop branch, which is visible and NOT
    Bot-Mode-managed) would otherwise swallow each teammate's message into a
    session whose bot has no message_agent and cannot answer.
    """
    canonical = _make_canonical(db, "forever")
    create_session(db, "branch", source="desktop", parent_session_id=canonical)
    assert _set_session_title(
        db,
        "branch",
        f"{CANONICAL_BOT_CHAT_TITLE} #2",
        source=TITLE_SOURCE_DERIVED,
    )
    assert set_session_hidden(db, "branch", False)

    assert resolve_session_by_title(db, CANONICAL_BOT_CHAT_TITLE) == canonical


def test_numbered_continuation_still_wins_for_ordinary_titles(db):
    # Control: the "#N continuation" preference is the point of the lookup: it
    # stays intact for every non-Bot-Chat title.
    create_session(db, "plain", source="cli")
    assert set_session_title(db, "plain", "My session")
    create_session(db, "plain2", source="cli")
    assert set_session_title(db, "plain2", "My session #2")

    assert resolve_session_by_title(db, "My session") == "plain2"
