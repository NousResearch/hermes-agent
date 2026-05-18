"""Unit tests for the ``/new all`` bulk-reset helpers on
:class:`gateway.run.GatewayRunner` (#24362).

The bulk path has three small but security-relevant pure helpers:

  * :meth:`GatewayRunner._is_bulk_reset_token` — same matcher used by
    the dispatch site (for confirmation copy) and the handler.  A drift
    here is the worst possible bug — users get the "single session"
    confirm dialog and then a bulk wipe.
  * :meth:`GatewayRunner._chat_type_is_group_like` — the DM/private
    short-circuit that prevents ``/new all`` from ever wiping siblings
    in a private chat.
  * :meth:`GatewayRunner._collect_topic_session_keys_for_bulk_reset` —
    the eligibility filter.  Everything else hangs off this list.

These tests pin the helpers in isolation; the end-to-end test module
(``test_new_all_bulk_reset.py``) drives the full handler.
"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source(
    *,
    chat_type: str = "supergroup",
    chat_id: str = "group-1",
    user_id: str = "user-A",
    thread_id: str | None = None,
    platform: Platform = Platform.TELEGRAM,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        chat_type=chat_type,
        thread_id=thread_id,
    )


def _make_entry(
    *,
    key: str,
    chat_id: str = "group-1",
    chat_type: str = "supergroup",
    user_id: str | None = None,
    platform: Platform = Platform.TELEGRAM,
    thread_id: str | None = None,
) -> SessionEntry:
    return SessionEntry(
        session_key=key,
        session_id=f"sid-{key}",
        created_at=datetime(2026, 5, 12),
        updated_at=datetime(2026, 5, 12),
        origin=SessionSource(
            platform=platform,
            chat_id=chat_id,
            user_id=user_id,
            chat_type=chat_type,
            thread_id=thread_id,
        ),
        platform=platform,
        chat_type=chat_type,
    )


def _make_runner_with_entries(entries: dict[str, SessionEntry]) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = MagicMock()
    runner.session_store._entries = entries
    return runner


# ---------------------------------------------------------------------------
# _is_bulk_reset_token  (12 cases — covers casing, whitespace, aliases,
# adjacent-but-not-equal tokens, and explicitly-non-token strings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("all", True),
        ("ALL", True),
        ("All", True),
        ("aLL", True),
        ("  all  ", True),
        ("\tall\n", True),
        ("*", True),
        (" * ", True),
        ("", False),
        ("   ", False),
        ("all-hands", False),
        ("allnighter", False),
        ("everything", False),
        ("/all", False),
        ("**", False),
        ("a", False),
        ("ALL TOPICS", False),
        ("'all'", False),
        ("all all", False),
    ],
)
def test_is_bulk_reset_token(raw: str, expected: bool) -> None:
    assert GatewayRunner._is_bulk_reset_token(raw) is expected


# ---------------------------------------------------------------------------
# _chat_type_is_group_like
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chat_type, expected",
    [
        ("dm", False),
        ("DM", False),
        ("Dm", False),
        ("private", False),
        ("PRIVATE", False),
        ("", False),
        (None, False),
        ("group", True),
        ("GROUP", True),
        ("supergroup", True),
        ("channel", True),
        ("forum", True),
        ("thread", True),
        # Unknown chat types default to group-like (safer for new
        # platforms — DMs are the only thing we MUST exclude to protect
        # private conversations).
        ("mystery", True),
    ],
)
def test_chat_type_is_group_like(chat_type, expected: bool) -> None:
    assert GatewayRunner._chat_type_is_group_like(chat_type) is expected


# ---------------------------------------------------------------------------
# _collect_topic_session_keys_for_bulk_reset
# ---------------------------------------------------------------------------


def test_collect_returns_all_topic_siblings_in_same_group() -> None:
    """Three topics in the same group → all three are eligible."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
        "agent:main:telegram:supergroup:group-1:topic-B": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-B",
            thread_id="topic-B",
        ),
        "agent:main:telegram:supergroup:group-1:topic-C": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-C",
            thread_id="topic-C",
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source()

    got = runner._collect_topic_session_keys_for_bulk_reset(source)
    assert set(got) == set(entries.keys())


def test_collect_skips_dm_sessions_even_with_same_chat_id() -> None:
    """A DM session that happens to share a chat_id (very rare but possible
    when a user DMs from inside a group manager bot) must NOT be touched."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            chat_type="supergroup",
            thread_id="topic-A",
        ),
        "agent:main:telegram:dm:group-1": _make_entry(
            key="agent:main:telegram:dm:group-1",
            chat_type="dm",
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source()

    got = runner._collect_topic_session_keys_for_bulk_reset(source)
    assert got == ["agent:main:telegram:supergroup:group-1:topic-A"]


def test_collect_returns_empty_when_source_is_dm() -> None:
    """DM-issued ``/new all`` collects nothing — the caller turns this
    into a friendly notice instead of bulk-resetting the user's DMs."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source(chat_type="dm")

    assert runner._collect_topic_session_keys_for_bulk_reset(source) == []


def test_collect_returns_empty_when_source_has_no_chat_id() -> None:
    """Source without chat_id → can't identify siblings, so return
    nothing rather than over-matching on platform alone."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source(chat_id="")

    assert runner._collect_topic_session_keys_for_bulk_reset(source) == []


def test_collect_filters_by_platform() -> None:
    """A Slack session with the same chat_id must not be touched by a
    Telegram ``/new all`` (cross-platform reset is a security bug)."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
        "agent:main:slack:channel:group-1:topic-A": _make_entry(
            key="agent:main:slack:channel:group-1:topic-A",
            chat_type="channel",
            thread_id="topic-A",
            platform=Platform.SLACK,
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source(platform=Platform.TELEGRAM)

    got = runner._collect_topic_session_keys_for_bulk_reset(source)
    assert got == ["agent:main:telegram:supergroup:group-1:topic-A"]


def test_collect_filters_by_chat_id() -> None:
    """Different group → no overlap."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            chat_id="group-1",
            thread_id="topic-A",
        ),
        "agent:main:telegram:supergroup:group-2:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-2:topic-A",
            chat_id="group-2",
            thread_id="topic-A",
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source(chat_id="group-1")

    got = runner._collect_topic_session_keys_for_bulk_reset(source)
    assert got == ["agent:main:telegram:supergroup:group-1:topic-A"]


def test_collect_per_user_thread_mode_only_resets_callers_own_threads() -> None:
    """When thread_sessions_per_user=true, each (chat, thread, user)
    has its own session.  /new all from user A must NEVER reset user
    B's per-user threads -- otherwise a Telegram admin could wipe the
    whole group's conversations with one command (cross-user blast)."""
    entries = {
        # User A's threads (eligible)
        "agent:main:telegram:supergroup:group-1:topic-A:user-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A:user-A",
            thread_id="topic-A",
            user_id="user-A",
        ),
        "agent:main:telegram:supergroup:group-1:topic-B:user-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-B:user-A",
            thread_id="topic-B",
            user_id="user-A",
        ),
        # User B's thread (must skip)
        "agent:main:telegram:supergroup:group-1:topic-A:user-B": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A:user-B",
            thread_id="topic-A",
            user_id="user-B",
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source(user_id="user-A")

    got = runner._collect_topic_session_keys_for_bulk_reset(source)
    assert set(got) == {
        "agent:main:telegram:supergroup:group-1:topic-A:user-A",
        "agent:main:telegram:supergroup:group-1:topic-B:user-A",
    }


def test_collect_shared_thread_entries_have_no_user_filter() -> None:
    """In shared-thread mode (default) the session_key has no user_id
    and entry.origin.user_id is whatever user happened to create the
    session.  We MUST still include those entries -- a thread is
    shared by definition, so any participant's /new all touches it."""
    # Shared thread: origin.user_id is set (creator) but the session is
    # not per-user (no user suffix on the key).  Caller is a *different*
    # user; we still expect to reset it because the thread is shared.
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
            user_id=None,  # shared
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source(user_id="user-B")

    got = runner._collect_topic_session_keys_for_bulk_reset(source)
    assert got == ["agent:main:telegram:supergroup:group-1:topic-A"]


def test_collect_handles_entry_without_origin() -> None:
    """Legacy entries loaded from sessions.json before origin tracking
    landed have ``origin=None``.  They must be silently skipped --
    never error out -- so a single legacy row can't break bulk reset
    for the whole group."""
    legacy_entry = _make_entry(
        key="agent:main:telegram:supergroup:group-1:topic-X",
        thread_id="topic-X",
    )
    legacy_entry.origin = None  # simulate legacy row
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-X": legacy_entry,
        "agent:main:telegram:supergroup:group-1:topic-Y": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-Y",
            thread_id="topic-Y",
        ),
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source()

    got = runner._collect_topic_session_keys_for_bulk_reset(source)
    assert got == ["agent:main:telegram:supergroup:group-1:topic-Y"]


def test_collect_handles_missing_session_store_gracefully() -> None:
    """If session_store is None (a runner being torn down) we must
    return [] rather than crashing -- bulk reset is convenience, not
    critical path."""
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    source = _make_source()
    assert runner._collect_topic_session_keys_for_bulk_reset(source) == []


def test_collect_handles_session_store_without_entries() -> None:
    """Same guarantee when the store has no ``_entries`` attribute
    (e.g. a test double that only stubs the public surface)."""
    runner = object.__new__(GatewayRunner)
    runner.session_store = MagicMock(spec=[])  # no _entries attr
    source = _make_source()
    assert runner._collect_topic_session_keys_for_bulk_reset(source) == []


def test_collect_returns_unique_keys() -> None:
    """Sanity: keys are unique (the underlying dict already enforces
    this, but anchor the contract so a future refactor that switches
    to a list of (key, entry) tuples can't accidentally duplicate)."""
    entries = {
        f"agent:main:telegram:supergroup:group-1:topic-{i}": _make_entry(
            key=f"agent:main:telegram:supergroup:group-1:topic-{i}",
            thread_id=f"topic-{i}",
        )
        for i in range(5)
    }
    runner = _make_runner_with_entries(entries)
    source = _make_source()

    got = runner._collect_topic_session_keys_for_bulk_reset(source)
    assert len(got) == len(set(got)) == 5
