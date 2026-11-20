"""Contract cases for the full-fidelity listing row: the :class:`SessionRecord` that ``SessionRepository.list_sessions`` and ``list_cron_job_runs`` carry, and the ``LIMIT``/``OFFSET`` window with its pinned back-fill.

Backend cases seed through public protocol operations only and check listing rows against other public reads; the rules are the ``persistence/conversation/sessions.py`` docstrings. Negative window bounds are validated without a backend.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from persistence import SessionId
from persistence.authority.coordination import HandoffStatus
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionOrder, SessionQuery, SessionWindow

T0 = datetime(2026, 9, 23, 12, 0, 0, tzinfo=UTC)


def _msg(content: str, i: int) -> NewMessage:
    return NewMessage(role="user", content=content, timestamp=(T0 + timedelta(seconds=i)).timestamp())


def _rows(backend, scope, query: SessionQuery = SessionQuery(), limit: int = 5):
    store = backend.open(scope)
    return store.sessions.list_sessions(query, SessionWindow(limit=limit))


def test_columns_read_what_each_owning_family_wrote(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("det-cols")
    store_a.sessions.create(
        NewSession(session_id=sid, source="telegram", system_prompt="be brief", display_name="Ada", origin={"platform": "telegram"}),
    )
    [fresh] = _rows(backend, scope_a)
    c = fresh.session
    # A never-touched row: counters 0, flags False, no activity labels, no handoff/cooldown/route.
    assert (c.runtime_revision, c.runtime_generation, c.git_metadata_generation, c.rewind_count) == (0, 0, 0, 0)
    assert c.expiry_finalized is False and fresh.unread is False and fresh.lineage_ids == ()
    assert (c.last_activity_at, c.last_activity_description, c.last_activity_provenance) == (None, None, None)
    # Creation provenance is the source the create wrote; no ingress stamped a transport profile; no overload strikes.
    assert (c.created_source, c.transport_profile, c.compression_overload_streak) == ("telegram", None, 0)
    assert (c.handoff_state, c.handoff_platform, c.handoff_error) == (None, None, None)
    assert (c.compression_failure_cooldown_until, c.compression_failure_error) == (None, None)
    assert (c.display_name, dict(c.origin or {}), c.system_prompt) == ("Ada", {"platform": "telegram"}, "be brief")

    generation = store_a.sessions.update_cwd(sid, "/work/a")
    pin = {"version": "v1", "tools": [{"type": "function", "function": {"name": "terminal"}}]}
    store_a.sessions.update_tool_pin(sid, pin)
    store_a.sessions.touch_activity(sid, T0.timestamp(), description="thinking", provenance="unknown")
    store_a.coordination.request_handoff(session_id=sid, platform="discord")
    until = (T0 + timedelta(hours=1)).timestamp()
    store_a.messages.record_compression_cooldown(sid, until, "summary failed")
    store_a.lineage.set_compression_fallback_streak(sid, 2)
    store_a.lineage.set_compression_ineffective_count(sid, 3)
    store_a.lineage.set_compression_overload_streak(sid, 4)
    store_a.lineage.set_compression_recovery_deadline(sid, until)
    store_a.usage.set_billing_route(sid, provider="openrouter", base_url="https://or.example", billing_mode="api")
    [row] = _rows(backend, scope_a)
    c = row.session
    assert c.git_metadata_generation == int(generation)
    # The listing read does not join the pin to the prompt store: the row carries the hash, so tool_pin is None; get resolves it.
    assert c.tool_pin is None and store_a.sessions.get(sid).tool_pin == pin
    assert c.last_activity_at == T0.timestamp() and c.last_activity_description == "thinking"
    # Handoff states are the stored strings.
    assert (c.handoff_state, c.handoff_platform, c.handoff_error) == (HandoffStatus.PENDING.value, "discord", None)
    assert (c.compression_failure_cooldown_until, c.compression_failure_error) == (until, "summary failed")
    assert (c.compression_fallback_streak, c.compression_ineffective_count, c.compression_recovery_deadline) == (2, 3, until)
    assert c.compression_overload_streak == 4 == store_a.lineage.compression_overload_streak(sid)
    assert c.created_source == "telegram"
    assert (c.billing_provider, c.billing_base_url, c.billing_mode) == ("openrouter", "https://or.example", "api")
    # Typed, not stored integers or JSON text.
    assert isinstance(c.pinned, bool) and isinstance(c.archived, bool)
    assert isinstance(c.expiry_finalized, bool)


@pytest.mark.parametrize("pin", [["terminal", "terminal"], {"version": "v1", "tools": []}, [], None])
def test_listing_and_cron_rows_carry_the_unresolved_pin(backend, scope_a, pin):
    sid = SessionId("cron_pin_20260923_120000")
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=sid, source="cron"))
    sessions.update_tool_pin(sid, pin)
    [listed] = _rows(backend, scope_a)
    [run] = sessions.list_cron_job_runs("pin", SessionWindow(limit=5))
    # CONTRACT: the listing and cron reads leave tool_names as its content hash (no join), so their records carry no pin; get resolves it.
    assert listed.session.tool_pin is None and run.entry.session.tool_pin is None
    assert sessions.get(sid).tool_pin == pin
    sessions.update_tool_pin(sid, None)
    [cleared] = _rows(backend, scope_a)
    assert cleared.session.tool_pin is None


def test_projected_root_carries_its_chain_and_the_root_columns(backend, scope_a):
    store_a = backend.open(scope_a)
    s = backend.open(scope_a).sessions
    root, tip = SessionId("det-root"), SessionId("det-tip")
    s.create(NewSession(session_id=root, source="cli", system_prompt="root prompt", display_name="Root"))
    store_a.messages.append(root, _msg("start", 1))
    store_a.lineage.set_compression_fallback_streak(root, 4)
    s.end(root, "compression")
    s.create(NewSession(session_id=tip, source="cli", parent_session_id=root, system_prompt="tip prompt"))
    store_a.messages.append(tip, _msg("go on", 2))
    [row] = _rows(backend, scope_a)
    assert row.session.session_id == tip and row.lineage_root_id == root
    assert row.lineage_ids == (root, tip)
    # The ROOT's columns, except the tip's prompt snapshot.
    assert row.session.compression_fallback_streak == 4 and row.session.display_name == "Root"
    assert row.session.system_prompt == "tip prompt"


def test_cron_run_entries_carry_the_run_columns(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("cron_job9_20260923_120000")
    store_a.sessions.create(NewSession(session_id=sid, source="cron", system_prompt="cron prompt"))
    store_a.coordination.request_handoff(session_id=sid, platform="telegram")
    [entry] = store_a.sessions.list_cron_job_runs("job9", SessionWindow(limit=5))
    assert (entry.cron_job_id, entry.cron_execution_id) == ("job9", None)
    assert entry.entry.session.system_prompt == "cron prompt"
    assert entry.entry.session.handoff_state == HandoffStatus.PENDING.value and entry.entry.session.handoff_platform == "telegram"


def _every_row(backend, scope, query):
    """The whole listing for ``query`` (no pin back-fill), in its order."""
    store = backend.open(scope)
    return list(store.sessions.list_sessions(query, SessionWindow(limit=1000)))


def _seed_window(backend, scope):
    store = backend.open(scope)
    s = backend.open(scope).sessions
    for i, sid in enumerate(["win-1", "win-2", "win-3", "win-4", "win-5", "win-6"]):
        s.create(NewSession(session_id=SessionId(sid), source="cli"))
        store.messages.append(SessionId(sid), _msg(f"m {sid}", 10 - i))
    for sid in ("win-2", "win-5", "win-4", "win-6"):
        s.set_pinned(SessionId(sid), True)
    s.set_hidden(SessionId("win-4"), True)
    s.set_archived(SessionId("win-6"), True)
    s.set_archived(SessionId("win-3"), True)


@pytest.mark.parametrize("limit,offset", [(0, 0), (1, 0), (2, 1), (3, 2), (50, 0), (2, 400)])
@pytest.mark.parametrize("arm", ["default", "recent", "archived_only"])
def test_window_is_the_offset_slice_then_the_missed_pins(backend, scope_a, arm, limit, offset):
    store_a = backend.open(scope_a)
    _seed_window(backend, scope_a)
    query = {
        "default": SessionQuery(include_pinned=True),
        "recent": SessionQuery(include_pinned=True, order=SessionOrder.LAST_ACTIVE),
        "archived_only": SessionQuery(include_pinned=True, archived_only=True),
    }[arm]
    served = store_a.sessions.list_sessions(query, SessionWindow(limit=limit, offset=offset))
    # The window is the listing's own order sliced by offset, pinned rows in place.
    window = _every_row(backend, scope_a, replace(query, include_pinned=False))[offset : offset + limit]
    listed = {d.session.session_id for d in window}
    # Then every other pin, archived or not (hidden only under archived_only here), newest started first whatever the order.
    pin_admission = SessionQuery(include_archived=True, include_hidden=query.archived_only)
    missed = [d for d in _every_row(backend, scope_a, pin_admission) if d.session.pinned and d.session.session_id not in listed]
    assert list(served) == window + missed
    ids = [d.session.session_id for d in served]
    assert "win-6" in ids  # the archived pin rides on every window, even an empty or exhausted one
    assert ("win-4" in ids) is query.archived_only
    assert len(ids) == len(set(ids))


def test_window_without_pins_is_the_plain_slice(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_window(backend, scope_a)
    listed = _every_row(backend, scope_a, SessionQuery())
    served = store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=2, offset=1))
    assert list(served) == listed[1:3]
    assert store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=0)) == ()


def test_archived_view_admits_hidden_rows(backend, scope_a):
    s = backend.open(scope_a).sessions
    for sid in ("arc-1", "arc-2", "arc-3"):
        s.create(NewSession(session_id=SessionId(sid), source="cli"))
    for sid in ("arc-1", "arc-2"):
        s.set_archived(SessionId(sid), True)
    s.set_hidden(SessionId("arc-1"), True)
    # The route's query shape: an integer message floor is always present.
    query = SessionQuery(archived_only=True, min_message_count=0)
    served = s.list_sessions(query, SessionWindow(limit=50))
    assert [d.session.session_id for d in served] == ["arc-2", "arc-1"]


def test_window_ranks_a_projected_tip_by_its_root(backend, scope_a):
    s = backend.open(scope_a).sessions
    root, mid, tip = SessionId("cw-1-root"), SessionId("cw-2-mid"), SessionId("cw-3-tip")
    s.create(NewSession(session_id=root, source="cli"))
    s.create(NewSession(session_id=mid, source="cli"))
    s.end(root, "compression")
    s.create(NewSession(session_id=tip, source="cli", parent_session_id=root))
    query = SessionQuery(include_pinned=True, min_message_count=0)

    def ids(limit):
        return [d.session.session_id for d in s.list_sessions(query, SessionWindow(limit=limit))]

    # The chain ranks where its root was started, behind the newer unrelated session, though its tip is the newest row.
    assert ids(1) == [mid]
    assert ids(2) == [mid, tip]
    s.set_pinned(root, True)
    # The pinned chain is back-filled once as its projected tip, and not repeated when the window already holds it.
    assert ids(1) == [mid, tip]
    assert ids(2) == [mid, tip]


@pytest.mark.parametrize("limit,offset", [(-1, 0), (1, -1)])
def test_session_window_rejects_negative_bounds(limit, offset):
    with pytest.raises(ValueError):
        SessionWindow(limit=limit, offset=offset)
