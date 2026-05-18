"""End-to-end tests for the ``/new all`` bulk-reset flow (#24362).

These tests drive :meth:`gateway.run.GatewayRunner._handle_reset_command`
directly with bulk-reset arguments and assert on the observable side
effects across every sibling topic session: session_id rotation, cached
agent eviction, per-session override cleanup, hook emission counts,
and the user-facing summary copy.

A regression anchor (``test_bug_24362_repro_*``) intentionally fails if
someone later removes the bulk-token detection from
:meth:`_handle_reset_command` — that detection is the heart of the
feature, and the title-setting code path right below it would silently
swallow ``/new all`` as "create new session titled 'all'" otherwise.
"""
import threading
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_source(
    *,
    chat_type: str = "supergroup",
    chat_id: str = "group-1",
    user_id: str = "user-A",
    thread_id: str | None = "topic-A",
    platform: Platform = Platform.TELEGRAM,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        chat_type=chat_type,
        thread_id=thread_id,
    )


def _make_event(text: str, source: SessionSource | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=source if source is not None else _make_source(),
        message_id="m1",
    )


def _make_entry(
    *,
    key: str,
    chat_id: str = "group-1",
    chat_type: str = "supergroup",
    user_id: str | None = None,
    thread_id: str | None = None,
    platform: Platform = Platform.TELEGRAM,
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


def _make_runner_with_topic_sessions(
    entries: Dict[str, SessionEntry],
) -> GatewayRunner:
    """Build a minimally-stubbed GatewayRunner ready to drive
    :meth:`_handle_reset_command` end-to-end.

    The mock ``session_store.reset_session`` rotates the entry in place
    and returns the new SessionEntry — mirroring the real
    :class:`SessionStore.reset_session` contract just enough to satisfy
    the bulk-reset loop.  Anything that touches SQLite, files, or the
    network is stubbed out.
    """
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._agent_cache = {}
    runner._queued_events = {}
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    store = MagicMock()
    store._entries = dict(entries)

    def _reset_session(key: str) -> SessionEntry | None:
        existing = store._entries.get(key)
        if existing is None:
            return None
        new = SessionEntry(
            session_key=existing.session_key,
            session_id=f"{existing.session_id}-reset",
            created_at=existing.created_at,
            updated_at=datetime(2026, 5, 13),
            origin=existing.origin,
            platform=existing.platform,
            chat_type=existing.chat_type,
            is_fresh_reset=True,
        )
        store._entries[key] = new
        return new

    store.reset_session.side_effect = _reset_session
    store.get_or_create_session.return_value = next(iter(entries.values()))
    store._generate_session_key.return_value = next(iter(entries.keys()))
    runner.session_store = store
    return runner


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_all_resets_every_sibling_topic_session() -> None:
    """``/new all`` rotates session_id for every sibling topic in the
    same group chat and emits hooks per session."""
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
    runner = _make_runner_with_topic_sessions(entries)
    pre_ids = {k: e.session_id for k, e in entries.items()}

    reply = await runner._handle_reset_command(_make_event("/new all"))

    # User-facing summary mentions the count (singular vs plural is in
    # locales/en.yaml; "Reset 3 topic sessions" is the rendered plural).
    assert isinstance(reply, EphemeralReply)
    assert "3 topic sessions" in reply.text

    # session_id rotated for every sibling.
    for key, pre_sid in pre_ids.items():
        post = runner.session_store._entries[key]
        assert post.session_id != pre_sid, f"sibling {key} kept its old session_id"
        assert post.session_id.endswith("-reset")
        assert post.is_fresh_reset is True

    # session:end + session:reset hooks emitted per sibling (3 + 3).
    emit_calls = runner.hooks.emit.call_args_list
    hook_names = [c.args[0] for c in emit_calls]
    assert hook_names.count("session:end") == 3
    assert hook_names.count("session:reset") == 3


@pytest.mark.asyncio
async def test_new_all_reset_alias_works_the_same() -> None:
    """The feature request mentioned ``/reset all`` as an alias; this is
    handled by the same shared bypass that maps ``/reset`` to ``/new``.

    The test exercises the handler directly with the canonical command
    name and verifies the bulk-token detection doesn't depend on which
    spelling routed to it (single matcher, two slash spellings)."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
        "agent:main:telegram:supergroup:group-1:topic-B": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-B",
            thread_id="topic-B",
        ),
    }
    runner = _make_runner_with_topic_sessions(entries)

    reply = await runner._handle_reset_command(_make_event("/reset all"))

    assert isinstance(reply, EphemeralReply)
    assert "2 topic sessions" in reply.text


@pytest.mark.asyncio
async def test_new_all_supports_star_alias() -> None:
    """``/new *`` is documented as a power-user alias."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
    }
    runner = _make_runner_with_topic_sessions(entries)

    reply = await runner._handle_reset_command(_make_event("/new *"))

    assert isinstance(reply, EphemeralReply)
    assert "1 topic session" in reply.text  # singular phrasing


@pytest.mark.asyncio
async def test_new_all_case_insensitive() -> None:
    """``/new ALL`` and ``/new All`` must both trigger bulk mode."""
    for variant in ("/new ALL", "/new All", "/new aLL"):
        entries = {
            "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
                key="agent:main:telegram:supergroup:group-1:topic-A",
                thread_id="topic-A",
            ),
        }
        runner = _make_runner_with_topic_sessions(entries)
        reply = await runner._handle_reset_command(_make_event(variant))
        assert isinstance(reply, EphemeralReply)
        assert "topic session" in reply.text, f"variant {variant!r} did not bulk-reset"


# ---------------------------------------------------------------------------
# Scoping rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_all_in_dm_falls_through_to_single_reset() -> None:
    """``/new all`` in a DM must NOT iterate siblings — it falls
    through to the existing per-session reset so the literal title
    'all' lands on the user's solo session, exactly like ``/new foo``
    today.  This protects users who type ``/new all`` in a DM by
    accident from accidentally wiping nothing (no siblings exist) or
    everything (cross-DM blast)."""
    dm_source = _make_source(chat_type="dm", thread_id=None)
    dm_key = "agent:main:telegram:dm:group-1"
    entries = {
        dm_key: _make_entry(
            key=dm_key,
            chat_type="dm",
            thread_id=None,
        ),
    }
    runner = _make_runner_with_topic_sessions(entries)
    # The single-session path needs session_store.get_or_create_session
    # and _session_key_for_source to line up.
    runner.session_store._generate_session_key.return_value = dm_key
    runner.session_store.get_or_create_session.return_value = entries[dm_key]

    # /new all in a DM should NOT take the bulk path.  The single-
    # session path sets a session title ("all"), so we assert that the
    # reply is NOT the bulk summary copy.
    reply = await runner._handle_reset_command(_make_event("/new all", dm_source))

    assert isinstance(reply, EphemeralReply)
    assert "topic sessions in this group" not in reply.text
    assert "topic session in this group" not in reply.text


@pytest.mark.asyncio
async def test_new_all_does_not_touch_other_groups() -> None:
    """Sibling sessions in a *different* group must stay intact."""
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
    runner = _make_runner_with_topic_sessions(entries)
    other_pre = entries["agent:main:telegram:supergroup:group-2:topic-A"].session_id

    await runner._handle_reset_command(_make_event("/new all"))

    other_post = runner.session_store._entries[
        "agent:main:telegram:supergroup:group-2:topic-A"
    ]
    assert other_post.session_id == other_pre, "other group's session was wiped"


@pytest.mark.asyncio
async def test_new_all_does_not_cross_platforms() -> None:
    """A Slack channel sharing the same chat_id must NOT be touched
    by a Telegram ``/new all`` — different platform, different scope."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
        "agent:main:slack:channel:group-1:thread-A": _make_entry(
            key="agent:main:slack:channel:group-1:thread-A",
            chat_type="channel",
            thread_id="thread-A",
            platform=Platform.SLACK,
        ),
    }
    runner = _make_runner_with_topic_sessions(entries)
    slack_pre = entries["agent:main:slack:channel:group-1:thread-A"].session_id

    await runner._handle_reset_command(_make_event("/new all"))

    slack_post = runner.session_store._entries[
        "agent:main:slack:channel:group-1:thread-A"
    ]
    assert slack_post.session_id == slack_pre


@pytest.mark.asyncio
async def test_new_all_per_user_thread_mode_respects_caller_user() -> None:
    """In per-user-thread mode (thread_sessions_per_user=true) the
    caller's ``/new all`` must reset only their own per-user sessions
    — never another user's.  This is the cross-user blast guard."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A:user-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A:user-A",
            user_id="user-A",
            thread_id="topic-A",
        ),
        "agent:main:telegram:supergroup:group-1:topic-A:user-B": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A:user-B",
            user_id="user-B",
            thread_id="topic-A",
        ),
    }
    runner = _make_runner_with_topic_sessions(entries)
    user_b_pre = entries[
        "agent:main:telegram:supergroup:group-1:topic-A:user-B"
    ].session_id

    await runner._handle_reset_command(_make_event("/new all"))

    user_b_post = runner.session_store._entries[
        "agent:main:telegram:supergroup:group-1:topic-A:user-B"
    ]
    assert user_b_post.session_id == user_b_pre, "cross-user blast happened"

    user_a_post = runner.session_store._entries[
        "agent:main:telegram:supergroup:group-1:topic-A:user-A"
    ]
    assert user_a_post.session_id.endswith("-reset"), "caller's own thread not reset"


# ---------------------------------------------------------------------------
# Side-effect cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_all_clears_per_session_overrides() -> None:
    """Each sibling's model/reasoning override and pending-model note
    must be cleared — otherwise the new agent on each topic would
    silently inherit the previous conversation's overrides."""
    keys = [
        "agent:main:telegram:supergroup:group-1:topic-A",
        "agent:main:telegram:supergroup:group-1:topic-B",
    ]
    entries = {
        k: _make_entry(key=k, thread_id=k.split(":")[-1]) for k in keys
    }
    runner = _make_runner_with_topic_sessions(entries)
    for k in keys:
        runner._session_model_overrides[k] = {
            "model": "x", "provider": "y", "api_key": "z", "base_url": "", "api_mode": "openai",
        }
        runner._session_reasoning_overrides[k] = {"enabled": True, "effort": "high"}
        runner._pending_model_notes[k] = "[switched]"

    await runner._handle_reset_command(_make_event("/new all"))

    for k in keys:
        assert k not in runner._session_model_overrides
        assert k not in runner._session_reasoning_overrides
        assert k not in runner._pending_model_notes


@pytest.mark.asyncio
async def test_new_all_evicts_cached_agents_for_every_sibling() -> None:
    """The cached agent for each sibling must be evicted so the next
    message in that topic spawns a fresh agent with the rotated
    session_id.

    Note: ``_evict_cached_agent`` short-circuits when
    ``_agent_cache_lock`` is None (a defensive guard for test fixtures
    that skip __init__), so this test installs a real lock to exercise
    the actual eviction path that production code follows.
    """
    keys = [
        "agent:main:telegram:supergroup:group-1:topic-A",
        "agent:main:telegram:supergroup:group-1:topic-B",
    ]
    entries = {k: _make_entry(key=k, thread_id=k.split(":")[-1]) for k in keys}
    runner = _make_runner_with_topic_sessions(entries)
    runner._agent_cache = {k: ("cached-agent", "cfg") for k in keys}
    runner._agent_cache_lock = threading.RLock()

    await runner._handle_reset_command(_make_event("/new all"))

    for k in keys:
        assert k not in runner._agent_cache


@pytest.mark.asyncio
async def test_new_all_drops_queued_events_for_every_sibling() -> None:
    """/queue overflow from the prior conversation must not leak into
    the freshly-rotated sibling sessions."""
    keys = [
        "agent:main:telegram:supergroup:group-1:topic-A",
        "agent:main:telegram:supergroup:group-1:topic-B",
    ]
    entries = {k: _make_entry(key=k, thread_id=k.split(":")[-1]) for k in keys}
    runner = _make_runner_with_topic_sessions(entries)
    runner._queued_events = {k: [SimpleNamespace(text="queued")] for k in keys}

    await runner._handle_reset_command(_make_event("/new all"))

    for k in keys:
        assert k not in runner._queued_events


# ---------------------------------------------------------------------------
# Empty / error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_all_with_no_sibling_sessions_returns_friendly_notice() -> None:
    """If no eligible siblings exist (fresh group, no recorded
    sessions yet), the user gets a notice -- never a silent no-op."""
    entries: Dict[str, SessionEntry] = {}
    runner = _make_runner_with_topic_sessions({
        # Dummy entry just so the runner has a session_store layout;
        # we then strip it before invoking the handler.
        "placeholder": _make_entry(
            key="placeholder",
            chat_type="dm",
            chat_id="other-group",
        ),
    })
    runner.session_store._entries.clear()

    reply = await runner._handle_reset_command(_make_event("/new all"))

    assert isinstance(reply, EphemeralReply)
    assert "No topic sessions to reset" in reply.text


@pytest.mark.asyncio
async def test_new_all_summary_singular_for_one_session() -> None:
    """Singular phrasing kicks in when exactly one sibling resets."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
    }
    runner = _make_runner_with_topic_sessions(entries)

    reply = await runner._handle_reset_command(_make_event("/new all"))

    assert isinstance(reply, EphemeralReply)
    assert "Reset 1 topic session" in reply.text
    assert "topic sessions" not in reply.text  # plural phrasing must NOT fire


@pytest.mark.asyncio
async def test_new_all_records_partial_failure_in_summary() -> None:
    """If some siblings fail to reset (reset_session returns None) the
    summary surfaces the failure count -- never silently swallowed."""
    entries = {
        "agent:main:telegram:supergroup:group-1:topic-A": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-A",
            thread_id="topic-A",
        ),
        "agent:main:telegram:supergroup:group-1:topic-B": _make_entry(
            key="agent:main:telegram:supergroup:group-1:topic-B",
            thread_id="topic-B",
        ),
    }
    runner = _make_runner_with_topic_sessions(entries)

    # Force topic-B to fail.
    original = runner.session_store.reset_session.side_effect
    def _flaky_reset(key: str):
        if key.endswith("topic-B"):
            return None
        return original(key)
    runner.session_store.reset_session.side_effect = _flaky_reset

    reply = await runner._handle_reset_command(_make_event("/new all"))

    assert isinstance(reply, EphemeralReply)
    assert "Reset 1 topic session" in reply.text
    assert "1 topic(s) could not be reset" in reply.text


# ---------------------------------------------------------------------------
# Regression anchor — fails if the bulk-token dispatch is removed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bug_24362_repro_new_all_must_reset_more_than_one_session() -> None:
    """Anchor for #24362.

    Pre-fix behaviour: ``/new all`` was treated as ``/new <title='all'>``,
    so only the current session got reset and the user had to manually
    visit every topic.

    This test asserts the post-fix invariant: when the same caller types
    ``/new all`` once, AT LEAST two sibling topic sessions get their
    session_id rotated.  If a future refactor removes the bulk-token
    detection from :meth:`_handle_reset_command`, this test fails
    because at most one session would be reset (the caller's own).
    """
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
    runner = _make_runner_with_topic_sessions(entries)
    pre = {k: e.session_id for k, e in entries.items()}

    await runner._handle_reset_command(_make_event("/new all"))

    rotated: List[str] = [
        k for k, pre_sid in pre.items()
        if runner.session_store._entries[k].session_id != pre_sid
    ]
    assert len(rotated) >= 2, (
        f"#24362 regression: /new all only rotated {len(rotated)} session(s) "
        f"({rotated}); the feature requires every sibling topic to be reset."
    )
