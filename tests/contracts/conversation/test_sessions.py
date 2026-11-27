"""Contract suite for session flags and facets on ``SessionRepository``.

Runs against every registered backend through public protocol operations only; mutation cases assert pre-state, result, and post-state, and refused or no-op operations prove non-mutation. Race cases run two contenders in two threads, each with its own event loop, so a backend must hold its own lock across each read-then-write verb (the GIL does not make multi-statement verbs atomic). The activity labels, the prompt snapshot and the YOLO flag are read off the :class:`SessionRecord` ``get`` returns.

The session facets section covers the facet verbs and the gateway listing's :class:`SessionRecencyEntry`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import pytest

from persistence import AuthorityEpoch, AuthorityRefusal, SessionId, UnknownExecution
from persistence.authority.admissions import AdmissionOutcome
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import (
    ACTIVITY_DESCRIPTION_MAX,
    ACTIVITY_PROVENANCE_UNKNOWN,
    ACTIVITY_PROVENANCES,
    AUTOMATIC_END_REASONS,
    BOUNDARY_END_REASONS,
    END_REASON_COMPRESSION,
    MAX_TITLE_LENGTH,
    RUNTIME_LOCK_KEY,
    SKILL_SCAFFOLD_PREFIX,
    YOLO_KEY,
    NewSession,
    RecentBrowseQuery,
    RecentBrowseTimeout,
    RuntimeLock,
    SessionCwdUsage,
    SessionLifecycleStatus,
    SessionLineageKind,
    SessionQuery,
    SessionWindow,
    is_explicit_close_reason,
    sanitize_title,
    session_yolo_enabled,
)
from persistence.conversation.usage import TokenCounts, UsageDelta
from tests.contracts._concurrency import run_concurrently
from tests.contracts.conftest import require_capability
from tests.contracts.conversation._session_probes import is_retired
from tests.contracts._scopes import MEDIA_ROOT

T0 = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
T0S = T0.timestamp()


def _new(session_id: str, source: str = "cli", **kwargs: Any) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source=source, **kwargs)


def _msg(role: str, content: str = "x", *, tool_calls: tuple = (), at: datetime = T0) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=at.timestamp(), tool_calls=tool_calls)


def _race(*calls: Callable[[], Any]) -> list:
    """Run each call on its own thread (real contention on one handle); results in call order."""
    return run_concurrently(*calls)


def _begin_epoch(backend, scope) -> AuthorityEpoch:
    return AuthorityEpoch(backend.begin_epoch(scope, "instance-contract"))


def _activity(backend, scope, session_id: str):
    """``(last_activity_at, description, provenance)`` of the row, ``None`` without one."""
    store = backend.open(scope)
    row = store.sessions.get(SessionId(session_id))
    return None if row is None else (row.last_activity_at, row.last_activity_description, row.last_activity_provenance)


def _prompt(backend, scope, session_id: str):
    """``(system_prompt, tool_pin)`` of the row, ``None`` without one."""
    store = backend.open(scope)
    row = store.sessions.get(SessionId(session_id))
    return None if row is None else (row.system_prompt, row.tool_pin)


def _yolo(backend, scope, session_id: str) -> bool:
    store = backend.open(scope)
    row = store.sessions.get(SessionId(session_id))
    return session_yolo_enabled(row.model_config if row is not None else None)


# ── activity heartbeat ────────────────────────────────────────────────────────


def test_activity_defaults_and_unknown_id(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-act0"))
    # CONTRACT: a never-touched row carries no activity stamp and no labels.
    assert _activity(backend, scope_a, "sf-act0") == (None, None, None)
    # CONTRACT: a touch or a clear on an unknown id creates nothing.
    store_a.sessions.touch_activity(SessionId("sf-ghost"), T0S, description="boo")
    assert store_a.sessions.get(SessionId("sf-ghost")) is None
    store_a.sessions.clear_activity_labels(SessionId("sf-ghost"))
    assert store_a.sessions.get(SessionId("sf-ghost")) is None


def test_touch_stamps_and_labels_then_reads_back(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-act1"))
    store_a.sessions.touch_activity(SessionId("sf-act1"), T0S, description="  executing tool  ", provenance="agent.compression")
    # CONTRACT: the stamp is the one passed, the description stripped, the provenance kept when recognised.
    assert _activity(backend, scope_a, "sf-act1") == (T0S, "executing tool", "agent.compression")
    # CONTRACT: no transcript row is written by a heartbeat.
    assert store_a.messages.read_transcript(SessionId("sf-act1"), TranscriptRead(view=TranscriptView.ALL)) == ()


def test_touch_stores_the_stamp_as_sent(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-act-exact"))
    stamp = 1_700_000_000.123_456_7
    store_a.sessions.touch_activity(SessionId("sf-act-exact"), stamp, description="exact")
    # CONTRACT: the stamp reads back bit for bit: the float it is sent is stored as it is.
    assert _activity(backend, scope_a, "sf-act-exact")[0] == stamp

def test_touch_defaults_the_stamp_to_now(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-act-now"))
    before = datetime.now(UTC).timestamp()
    store_a.sessions.touch_activity(SessionId("sf-act-now"), description="now")
    stamp, description, provenance = _activity(backend, scope_a, "sf-act-now")
    # CONTRACT: without ``at`` the stamp is the current time.
    assert stamp >= before and (description, provenance) == ("now", ACTIVITY_PROVENANCE_UNKNOWN)


def test_touch_is_monotonic_older_and_equal_stamps_are_no_ops(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-act2"))
    store_a.sessions.touch_activity(SessionId("sf-act2"), T0S, description="newer")
    # CONTRACT: an older stamp changes nothing, labels included.
    store_a.sessions.touch_activity(SessionId("sf-act2"), T0S - 1, description="older")
    assert _activity(backend, scope_a, "sf-act2") == (T0S, "newer", ACTIVITY_PROVENANCE_UNKNOWN)
    # CONTRACT: an equal stamp changes nothing either (strictly advancing only).
    store_a.sessions.touch_activity(SessionId("sf-act2"), T0S, description="same")
    assert _activity(backend, scope_a, "sf-act2") == (T0S, "newer", ACTIVITY_PROVENANCE_UNKNOWN)
    # CONTRACT: a strictly newer stamp lands with its own labels.
    store_a.sessions.touch_activity(SessionId("sf-act2"), T0S + 1, description="newest")
    assert _activity(backend, scope_a, "sf-act2") == (T0S + 1, "newest", ACTIVITY_PROVENANCE_UNKNOWN)


def test_touch_two_touchers_newer_wins_regardless_of_arrival(backend, scope_a):
    store_a = backend.open(scope_a)
    older, newer = T0S, T0S + 5
    # Both sequential orders...
    store_a.sessions.create(_new("sf-act3a"))
    store_a.sessions.touch_activity(SessionId("sf-act3a"), newer, description="new")
    store_a.sessions.touch_activity(SessionId("sf-act3a"), older, description="old")
    assert _activity(backend, scope_a, "sf-act3a") == (newer, "new", ACTIVITY_PROVENANCE_UNKNOWN)
    store_a.sessions.create(_new("sf-act3b"))
    store_a.sessions.touch_activity(SessionId("sf-act3b"), older, description="old")
    store_a.sessions.touch_activity(SessionId("sf-act3b"), newer, description="new")
    assert _activity(backend, scope_a, "sf-act3b") == (newer, "new", ACTIVITY_PROVENANCE_UNKNOWN)
    # ...and two concurrent contenders: HEARTBEAT OUTCOME, the newer stamp and ITS labels win.
    store_a.sessions.create(_new("sf-act3c"))
    _race(
        lambda: store_a.sessions.touch_activity(SessionId("sf-act3c"), older, description="old"),
        lambda: store_a.sessions.touch_activity(SessionId("sf-act3c"), newer, description="new"),
    )
    assert _activity(backend, scope_a, "sf-act3c") == (newer, "new", ACTIVITY_PROVENANCE_UNKNOWN)


def test_touch_bounds_description_and_normalizes_provenance(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-act4"))
    long_text = "x" * (ACTIVITY_DESCRIPTION_MAX + 40)
    store_a.sessions.touch_activity(SessionId("sf-act4"), T0S, description=long_text, provenance="not.a.provenance")
    _, description, provenance = _activity(backend, scope_a, "sf-act4")
    # CONTRACT: clamped to the budget with a trailing ellipsis; unrecognised provenance stored as unknown.
    assert len(description) == ACTIVITY_DESCRIPTION_MAX and description.endswith("…")
    assert description[:-1] == "x" * (ACTIVITY_DESCRIPTION_MAX - 1)
    assert provenance == ACTIVITY_PROVENANCE_UNKNOWN
    # CONTRACT: every member of the closed vocabulary is stored verbatim (padded input stripped).
    for index, name in enumerate(sorted(ACTIVITY_PROVENANCES), start=1):
        store_a.sessions.touch_activity(SessionId("sf-act4"), T0S + index, provenance=f" {name} ")
        assert _activity(backend, scope_a, "sf-act4")[2] == name


def test_clear_labels_keeps_the_stamp(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-act6"))
    store_a.sessions.touch_activity(SessionId("sf-act6"), T0S, description="compressing", provenance="agent.compression")
    store_a.sessions.clear_activity_labels(SessionId("sf-act6"))
    # CONTRACT: labels emptied, last_activity_at untouched.
    assert _activity(backend, scope_a, "sf-act6") == (T0S, "", ACTIVITY_PROVENANCE_UNKNOWN)
    # CONTRACT: a no-op clear is a no-op (idempotent), and a later touch still needs to advance the stamp.
    store_a.sessions.clear_activity_labels(SessionId("sf-act6"))
    store_a.sessions.touch_activity(SessionId("sf-act6"), T0S, description="stale")
    assert _activity(backend, scope_a, "sf-act6") == (T0S, "", ACTIVITY_PROVENANCE_UNKNOWN)


def test_touch_moves_the_activity_watermark_read_by_listings(backend, scope_a):
    # The roster and the gateway listing read the activity watermark where present; touch_activity is how it comes to be present.
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-wm-old", chat_id="local-1", session_key="k-old"))
    store_a.sessions.create(_new("sf-wm-new", chat_id="local-2", session_key="k-new"))
    latest = store_a.sessions.roster_sessions(owner_user_id=None, owner=None).last_session
    assert latest is not None and latest.id == SessionId("sf-wm-new")
    far_future = datetime(2030, 1, 1, tzinfo=UTC).timestamp()
    store_a.sessions.touch_activity(SessionId("sf-wm-old"), far_future)
    latest = store_a.sessions.roster_sessions(owner_user_id=None, owner=None).last_session
    assert latest is not None and latest.id == SessionId("sf-wm-old") and latest.last_active == far_future
    entries = store_a.sessions.list_gateway_sessions()
    by_id = {entry.session.session_id: entry.last_active for entry in entries}
    assert by_id[SessionId("sf-wm-old")] == far_future
    assert entries[0].session.session_id == SessionId("sf-wm-old")


# ── prompt-cache prefix ───────────────────────────────────────────────────────


def test_system_prompt_is_byte_stable_and_none_clears(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-pr1"))
    assert _prompt(backend, scope_a, "sf-pr1") == (None, None)
    prompt = "You are Hermes.  \r\n\tRome: Ro\u0301me vs R\u00f3me \u200b\n\n   "
    store_a.sessions.update_system_prompt(SessionId("sf-pr1"), prompt)
    seen, _ = _prompt(backend, scope_a, "sf-pr1")
    # CONTRACT: exactly the bytes passed (trailing whitespace, CRLF, NFD vs NFC, zero-width all survive).
    assert seen == prompt and seen.encode() == prompt.encode()
    store_a.sessions.update_system_prompt(SessionId("sf-pr1"), "")
    assert _prompt(backend, scope_a, "sf-pr1") == ("", None)
    store_a.sessions.update_system_prompt(SessionId("sf-pr1"), None)
    assert _prompt(backend, scope_a, "sf-pr1") == (None, None)
    # CONTRACT: unknown ids are a silent no-op that creates nothing.
    store_a.sessions.update_system_prompt(SessionId("sf-ghost"), "p")
    assert store_a.sessions.get(SessionId("sf-ghost")) is None


def test_tool_pin_is_stored_and_returned_whole(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-pr2"))
    store_a.sessions.create(_new("sf-pr2b"))
    pin = {"version": "v1", "tools": [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
                                      {"type": "function", "function": {"name": "read_file"}}]}
    store_a.sessions.update_tool_pin(SessionId("sf-pr2"), pin)
    # The same pin on a second session (a backend may content-address it) reads back whole on both.
    store_a.sessions.update_tool_pin(SessionId("sf-pr2b"), pin)
    # CONTRACT: the versioned document round-trips whole; storage does not interpret the tool definitions.
    assert _prompt(backend, scope_a, "sf-pr2")[1] == pin
    assert _prompt(backend, scope_a, "sf-pr2b")[1] == pin
    # A legacy bare name list keeps verbatim order, duplicates and case; an empty list is a stored empty list.
    store_a.sessions.update_tool_pin(SessionId("sf-pr2"), ["terminal", "read_file", "terminal", "Read_File"])
    assert _prompt(backend, scope_a, "sf-pr2")[1] == ["terminal", "read_file", "terminal", "Read_File"]
    store_a.sessions.update_tool_pin(SessionId("sf-pr2"), [])
    assert _prompt(backend, scope_a, "sf-pr2")[1] == []
    store_a.sessions.update_tool_pin(SessionId("sf-pr2"), None)
    assert _prompt(backend, scope_a, "sf-pr2")[1] is None
    # Clearing one session leaves the other's copy of the shared pin intact.
    assert _prompt(backend, scope_a, "sf-pr2b")[1] == pin
    store_a.sessions.update_tool_pin(SessionId("sf-ghost"), ["x"])
    assert store_a.sessions.get(SessionId("sf-ghost")) is None


def test_prompt_and_tool_pin_are_independent(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-pr3"))
    store_a.sessions.update_system_prompt(SessionId("sf-pr3"), "P")
    store_a.sessions.update_tool_pin(SessionId("sf-pr3"), {"version": "v1", "tools": []})
    assert _prompt(backend, scope_a, "sf-pr3") == ("P", {"version": "v1", "tools": []})
    store_a.sessions.update_system_prompt(SessionId("sf-pr3"), None)
    assert _prompt(backend, scope_a, "sf-pr3") == (None, {"version": "v1", "tools": []})


# ── runtime lock and YOLO ─────────────────────────────────────────────────────


def test_runtime_lock_merges_coalesces_model_and_keeps_the_prompt(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-lk1", model="m-before", model_config={"temperature": 0.2}))
    store_a.sessions.update_system_prompt(SessionId("sf-lk1"), "footer says m-before")
    store_a.sessions.update_tool_pin(SessionId("sf-lk1"), ["t1"])
    before = datetime.now(UTC).timestamp() - 1
    store_a.sessions.set_runtime_lock(SessionId("sf-lk1"), RuntimeLock(provider="openrouter", confirmed=True))
    stored = store_a.sessions.get_model_config_value(SessionId("sf-lk1"), RUNTIME_LOCK_KEY)
    assert isinstance(stored, dict)
    first_stamp = stored.pop("updated_at")
    # CONTRACT: the source's empty defaults for unstated fields, a float epoch updated_at.
    assert stored == {"provider": "openrouter", "model": "", "model_options": {}, "route_source": "", "confirmed": True}
    assert isinstance(first_stamp, float) and first_stamp >= before
    row = store_a.sessions.get(SessionId("sf-lk1"))
    assert row is not None
    # CONTRACT: model kept (COALESCE with an empty lock model); other config keys survive; the prompt-cache prefix is untouched.
    assert row.model == "m-before" and row.model_config["temperature"] == 0.2
    assert _prompt(backend, scope_a, "sf-lk1") == ("footer says m-before", ["t1"])
    store_a.sessions.set_runtime_lock(
        SessionId("sf-lk1"), RuntimeLock(model="m-after", provider="p", model_options={"reasoning": "high"}, route_source="browser")
    )
    stored = store_a.sessions.get_model_config_value(SessionId("sf-lk1"), RUNTIME_LOCK_KEY)
    assert isinstance(stored, dict) and stored.pop("updated_at") >= first_stamp
    assert stored == {"provider": "p", "model": "m-after", "model_options": {"reasoning": "high"}, "route_source": "browser", "confirmed": False}
    row = store_a.sessions.get(SessionId("sf-lk1"))
    assert row is not None and row.model == "m-after"
    # CONTRACT: unknown ids create nothing.
    store_a.sessions.set_runtime_lock(SessionId("sf-ghost"), RuntimeLock(model="m"))
    assert store_a.sessions.get(SessionId("sf-ghost")) is None


def test_runtime_lock_and_yolo_keep_typed_lineage(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-lk-par"))
    store_a.sessions.create(_new("sf-lk-kid", parent_session_id=SessionId("sf-lk-par"), lineage_kind=SessionLineageKind.DELEGATE))
    store_a.sessions.set_runtime_lock(SessionId("sf-lk-kid"), RuntimeLock(model="m"))
    store_a.sessions.set_yolo(SessionId("sf-lk-kid"), True)
    kid = store_a.sessions.get(SessionId("sf-lk-kid"))
    # CONTRACT: the merge never disturbs typed lineage; the delete cascade still classifies the child as a delegate.
    assert kid is not None and kid.lineage_kind is SessionLineageKind.DELEGATE
    assert store_a.sessions.delete_targets(SessionId("sf-lk-par")) == (SessionId("sf-lk-par"), SessionId("sf-lk-kid"))


def test_yolo_round_trip_and_false_on_absence(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-y1", model_config={"keep": 1}))
    assert _yolo(backend, scope_a, "sf-y1") is False
    store_a.sessions.set_yolo(SessionId("sf-y1"), True)
    assert _yolo(backend, scope_a, "sf-y1") is True
    row = store_a.sessions.get(SessionId("sf-y1"))
    # CONTRACT: rides model_config under YOLO_KEY; other keys survive.
    assert row is not None and row.model_config[YOLO_KEY] is True and row.model_config["keep"] == 1
    store_a.sessions.set_yolo(SessionId("sf-y1"), False)
    assert _yolo(backend, scope_a, "sf-y1") is False
    # CONTRACT: Python truth over whatever a raw patch stored; never raises.
    store_a.sessions.patch_model_config(SessionId("sf-y1"), {YOLO_KEY: "yes"})
    assert _yolo(backend, scope_a, "sf-y1") is True
    store_a.sessions.patch_model_config(SessionId("sf-y1"), {YOLO_KEY: 0})
    assert _yolo(backend, scope_a, "sf-y1") is False
    # CONTRACT: unknown ids read False and a set creates nothing.
    assert _yolo(backend, scope_a, "sf-ghost") is False
    store_a.sessions.set_yolo(SessionId("sf-ghost"), True)
    assert store_a.sessions.get(SessionId("sf-ghost")) is None
    assert _yolo(backend, scope_a, "sf-ghost") is False


# ── lifecycle statuses ────────────────────────────────────────────────────────


def test_lifecycle_statuses_classify_the_last_row(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    for sid in ("sf-ls-empty", "sf-ls-complete", "sf-ls-user", "sf-ls-tool", "sf-ls-calls"):
        store_a.sessions.create(_new(sid))
    store_a.messages.append_batch(SessionId("sf-ls-complete"), [_msg("user", "q"), _msg("assistant", "a")])
    store_a.messages.append_batch(SessionId("sf-ls-user"), [_msg("assistant", "a"), _msg("user", "q")])
    call = {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
    store_a.messages.append_batch(
        SessionId("sf-ls-tool"),
        [_msg("assistant", "a", tool_calls=(call,)), NewMessage(role="tool", content="r", timestamp=T0S, tool_call_id="c1", tool_name="t")],
    )
    store_a.messages.append_batch(
        SessionId("sf-ls-calls"),
        [_msg("user", "q"), _msg("assistant", "", tool_calls=({"id": "c2", "type": "function", "function": {"name": "t", "arguments": "{}"}},))],
    )
    # A same-id row in the other scope must not leak into this scope's classification.
    store_b.sessions.create(_new("sf-ls-empty"))
    store_b.messages.append(SessionId("sf-ls-empty"), _msg("user", "other profile"))
    ids = ["sf-ls-empty", "sf-ls-complete", "sf-ls-user", "sf-ls-tool", "sf-ls-calls", "sf-ghost", "sf-ls-complete"]
    statuses = store_a.sessions.lifecycle_statuses([SessionId(sid) for sid in ids])
    assert dict(statuses) == {
        SessionId("sf-ls-empty"): SessionLifecycleStatus.EMPTY,
        SessionId("sf-ls-complete"): SessionLifecycleStatus.COMPLETE,
        SessionId("sf-ls-user"): SessionLifecycleStatus.INTERRUPTED,
        SessionId("sf-ls-tool"): SessionLifecycleStatus.INTERRUPTED,
        SessionId("sf-ls-calls"): SessionLifecycleStatus.INTERRUPTED,
        SessionId("sf-ghost"): SessionLifecycleStatus.EMPTY,
    }
    assert store_a.sessions.lifecycle_statuses([]) == {}


def test_lifecycle_statuses_error_finish_wins(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-ls-err"))
    store_a.messages.append_batch(SessionId("sf-ls-err"), [_msg("user", "q"), NewMessage(role="assistant", content="", timestamp=T0S, finish_reason="error")])
    # CONTRACT: a last row finishing with an error reason classifies ERROR.
    assert store_a.sessions.lifecycle_statuses([SessionId("sf-ls-err")]) == {SessionId("sf-ls-err"): SessionLifecycleStatus.ERROR}


def test_lifecycle_statuses_follow_the_last_row_as_it_changes(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-ls2"))
    store_a.messages.append(SessionId("sf-ls2"), _msg("user", "q"))
    assert store_a.sessions.lifecycle_statuses([SessionId("sf-ls2")])[SessionId("sf-ls2")] is SessionLifecycleStatus.INTERRUPTED
    store_a.messages.append(SessionId("sf-ls2"), _msg("assistant", "a", at=T0 + timedelta(seconds=1)))
    assert store_a.sessions.lifecycle_statuses([SessionId("sf-ls2")])[SessionId("sf-ls2")] is SessionLifecycleStatus.COMPLETE


# ── reopen_if_explicitly_closed ───────────────────────────────────────────────


def _reopen(backend, scope, session_id, **kwargs):
    store = backend.open(scope)
    return store.sessions.reopen_if_explicitly_closed(SessionId(session_id), provenance="contract", **kwargs)


@pytest.mark.parametrize("reason", ["tui_close", "cli_close", "webhook_complete", "custom_reason"])
def test_reopen_clears_an_explicit_close_and_returns_it(backend, scope_a, reason):
    store_a = backend.open(scope_a)
    assert is_explicit_close_reason(reason)
    store_a.sessions.create(_new("sf-ro1"))
    store_a.sessions.end(SessionId("sf-ro1"), reason)
    assert _reopen(backend, scope_a, "sf-ro1") == reason
    row = store_a.sessions.get(SessionId("sf-ro1"))
    assert row is not None and row.ended_at is None and row.end_reason is None
    # CONTRACT: an open row reads None (nothing to clear); a later close lands on the reopened row.
    assert _reopen(backend, scope_a, "sf-ro1") is None
    store_a.sessions.end(SessionId("sf-ro1"), "cli_close")
    row = store_a.sessions.get(SessionId("sf-ro1"))
    assert row is not None and row.end_reason == "cli_close"


@pytest.mark.parametrize("reason", sorted(AUTOMATIC_END_REASONS) + sorted(BOUNDARY_END_REASONS) + [END_REASON_COMPRESSION])
def test_reopen_never_touches_automatic_boundary_or_compression_stamps(backend, scope_a, reason):
    store_a = backend.open(scope_a)
    assert not is_explicit_close_reason(reason)
    store_a.sessions.create(_new("sf-ro2"))
    store_a.sessions.end(SessionId("sf-ro2"), reason)
    before = store_a.sessions.get(SessionId("sf-ro2"))
    assert _reopen(backend, scope_a, "sf-ro2") is None
    assert store_a.sessions.get(SessionId("sf-ro2")) == before


def test_reopen_unknown_id_reads_none(backend, scope_a):
    store_a = backend.open(scope_a)
    assert _reopen(backend, scope_a, "sf-ghost") is None
    assert store_a.sessions.get(SessionId("sf-ghost")) is None


def test_reopen_blocked_by_a_continuation_child_only(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-ro3"))
    store_a.sessions.end(SessionId("sf-ro3"), "tui_close")
    # Delegate, branch, and tool children do not block.
    store_a.sessions.create(_new("sf-ro3-d", parent_session_id=SessionId("sf-ro3"), lineage_kind=SessionLineageKind.DELEGATE))
    store_a.sessions.create(_new("sf-ro3-b", parent_session_id=SessionId("sf-ro3"), lineage_kind=SessionLineageKind.BRANCH))
    store_a.sessions.create(_new("sf-ro3-t", source="tool", parent_session_id=SessionId("sf-ro3")))
    assert _reopen(backend, scope_a, "sf-ro3") == "tui_close"
    # A plain continuation child blocks.
    store_a.sessions.end(SessionId("sf-ro3"), "tui_close")
    store_a.sessions.create(_new("sf-ro3-c", parent_session_id=SessionId("sf-ro3")))
    before = store_a.sessions.get(SessionId("sf-ro3"))
    assert _reopen(backend, scope_a, "sf-ro3") is None
    assert store_a.sessions.get(SessionId("sf-ro3")) == before


def test_reopen_two_reopeners_exactly_one_wins(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-ro4"))
    store_a.sessions.end(SessionId("sf-ro4"), "tui_close")
    results = _race(lambda: _reopen(backend, scope_a, "sf-ro4"), lambda: _reopen(backend, scope_a, "sf-ro4"))
    # CAS OUTCOME: one reopener gets the reason, the other None; the row is open once.
    assert sorted(results, key=str) == [None, "tui_close"]
    row = store_a.sessions.get(SessionId("sf-ro4"))
    assert row is not None and row.ended_at is None


@pytest.mark.parametrize("patience_s", [0.0, 0.5, None])
def test_reopen_patience_admits_an_uncontended_store(backend, scope_a, patience_s):
    """A bounded patience changes only how long admission may wait; uncontended, every budget (zero included) reopens."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-ro6"))
    store_a.sessions.end(SessionId("sf-ro6"), "tui_close")
    assert _reopen(backend, scope_a, "sf-ro6", patience_s=patience_s) == "tui_close"
    row = store_a.sessions.get(SessionId("sf-ro6"))
    assert row is not None and row.ended_at is None


def test_reopen_negative_patience_still_makes_one_attempt(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-ro7"))
    store_a.sessions.end(SessionId("sf-ro7"), "tui_close")
    # CONTRACT: a negative patience is a deadline already passed; the reopen still makes its first attempt, so an uncontended reopen lands.
    assert _reopen(backend, scope_a, "sf-ro7", patience_s=-1.0) == "tui_close"
    row = store_a.sessions.get(SessionId("sf-ro7"))
    assert row is not None and row.ended_at is None and row.end_reason is None


# ── discard_unadmitted ────────────────────────────────────────────────────────


def test_discard_unadmitted_removes_row_and_transcript_without_retiring(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-dc1"))
    store_a.messages.append(SessionId("sf-dc1"), _msg("user", "seed copy"))
    # CONTRACT: the removed ids: the row itself when it existed.
    assert store_a.sessions.discard_unadmitted(SessionId("sf-dc1"), media_root=MEDIA_ROOT) == (SessionId("sf-dc1"),)
    assert store_a.sessions.get(SessionId("sf-dc1")) is None
    assert store_a.messages.read_transcript(SessionId("sf-dc1"), TranscriptRead(view=TranscriptView.ALL)) == ()
    # CONTRACT: no retirement fence, so the id is creatable again; a second discard removes nothing.
    assert store_a.sessions.discard_unadmitted(SessionId("sf-dc1"), media_root=MEDIA_ROOT) == ()
    assert store_a.sessions.create(_new("sf-dc1")) == SessionId("sf-dc1")
    assert store_a.sessions.discard_unadmitted(SessionId("sf-ghost"), media_root=MEDIA_ROOT) == ()


def test_discard_two_discarders_exactly_one_true(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-dc2"))
    results = _race(
        lambda: store_a.sessions.discard_unadmitted(SessionId("sf-dc2"), media_root=MEDIA_ROOT),
        lambda: store_a.sessions.discard_unadmitted(SessionId("sf-dc2"), media_root=MEDIA_ROOT),
    )
    assert sorted(results) == [(), (SessionId("sf-dc2"),)]
    assert store_a.sessions.get(SessionId("sf-dc2")) is None
    assert not is_retired(backend, scope_a, SessionId("sf-dc2"))


def test_discard_admitted_row_takes_the_fenced_delete_path(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-dc3", model="keep-me"))
    store_a.sessions.seed_ledger_admission(SessionId("sf-dc3"), status="queued")
    with pytest.raises(AuthorityRefusal) as refused:
        store_a.sessions.discard_unadmitted(SessionId("sf-dc3"), media_root=MEDIA_ROOT)
    assert refused.value.reason == "session_busy"
    row = store_a.sessions.get(SessionId("sf-dc3"))
    assert row is not None and row.model == "keep-me"
    store_a.sessions.create(_new("sf-dc4"))
    store_a.sessions.seed_ledger_admission(SessionId("sf-dc4"), status="unknown")
    with pytest.raises(UnknownExecution):
        store_a.sessions.discard_unadmitted(SessionId("sf-dc4"), media_root=MEDIA_ROOT)
    assert store_a.sessions.get(SessionId("sf-dc4")) is not None


def test_discard_with_terminal_admission_deletes_and_retires(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "durable_admissions")
    store_a.sessions.create(_new("sf-dc5"))
    store_a.sessions.create(_new("sf-dc5-sub", parent_session_id=SessionId("sf-dc5"), lineage_kind=SessionLineageKind.DELEGATE))
    epoch = _begin_epoch(backend, scope_a)
    record = store_a.admissions.admit(
        epoch=epoch, principal_id="principal-m2", session_id=SessionId("sf-dc5"), request_id="req-1", payload={"input": "hi"}
    )[0]
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SessionId("sf-dc5"))
    assert claimed is not None
    store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id, generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED
    )
    # CONTRACT: a real (once-admitted) session is deleted through the fence and RETIRED, unlike the unadmitted arm; the verb returns delete's removed ids, its delegate child included.
    assert store_a.sessions.discard_unadmitted(SessionId("sf-dc5"), media_root=MEDIA_ROOT) == (SessionId("sf-dc5"), SessionId("sf-dc5-sub"))
    assert store_a.sessions.get(SessionId("sf-dc5")) is None
    assert is_retired(backend, scope_a, SessionId("sf-dc5")) and is_retired(backend, scope_a, SessionId("sf-dc5-sub"))


# ── parity cases ───────


def test_runtime_lock_empty_model_overwrites_like_coalesce(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-rl3", model="m-before"))
    store_a.sessions.set_runtime_lock(SessionId("sf-rl3"), RuntimeLock(provider="p"))
    row = store_a.sessions.get(SessionId("sf-rl3"))
    assert row is not None and row.model == "m-before"
    # CONTRACT: None keeps the column; ANY string, the empty one included, overwrites (source COALESCE(?, model)).
    store_a.sessions.set_runtime_lock(SessionId("sf-rl3"), RuntimeLock(model=""))
    row = store_a.sessions.get(SessionId("sf-rl3"))
    assert row is not None and row.model == ""


def test_reopen_treats_an_empty_reason_as_explicit(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-ro5"))
    store_a.sessions.end(SessionId("sf-ro5"), "")
    # CONTRACT: the source excludes automatic, compression and boundary reasons only; "" reopens and is returned.
    assert _reopen(backend, scope_a, "sf-ro5") == ""
    row = store_a.sessions.get(SessionId("sf-ro5"))
    assert row is not None and row.ended_at is None and row.end_reason is None


def test_reopen_two_reopeners_repeated_rounds_exactly_one_wins(backend, scope_a):
    # Ten rounds of two threaded contenders: the CAS must never hand the reason to both.
    store_a = backend.open(scope_a)
    for round_no in range(10):
        sid = f"sf-ro6-{round_no}"
        store_a.sessions.create(_new(sid))
        store_a.sessions.end(SessionId(sid), "cli_close")
        results = _race(lambda: _reopen(backend, scope_a, sid), lambda: _reopen(backend, scope_a, sid))
        assert sorted(results, key=str) == [None, "cli_close"], round_no


def test_touch_two_touchers_repeated_rounds_newer_always_wins(backend, scope_a):
    store_a = backend.open(scope_a)
    for round_no in range(10):
        sid = f"sf-act9-{round_no}"
        store_a.sessions.create(_new(sid))
        _race(
            lambda: store_a.sessions.touch_activity(SessionId(sid), T0S + 1, description="old"),
            lambda: store_a.sessions.touch_activity(SessionId(sid), T0S + 2, description="new"),
        )
        assert _activity(backend, scope_a, sid)[:2] == (T0S + 2, "new"), round_no


def _import_payload(session_id: str, role: str | None = "user") -> dict:
    messages = [{"role": role, "content": "imported", "timestamp": T0S}] if role else []
    return {"id": session_id, "source": "cli", "started_at": T0S, "model": "m", "messages": messages}


def test_lifecycle_statuses_see_imported_transcripts(backend, scope_a):
    store_a = backend.open(scope_a)
    result = store_a.maintenance.import_sessions([_import_payload("sf-imp1", "user"), _import_payload("sf-imp2", "assistant")])
    assert result.imported == 2
    statuses = store_a.sessions.lifecycle_statuses([SessionId("sf-imp1"), SessionId("sf-imp2")])
    # CONTRACT: imported rows are transcript rows; a last imported user row is INTERRUPTED, an assistant row COMPLETE.
    assert statuses == {SessionId("sf-imp1"): SessionLifecycleStatus.INTERRUPTED, SessionId("sf-imp2"): SessionLifecycleStatus.COMPLETE}


def test_discard_removes_imported_transcripts_too(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.maintenance.import_sessions([_import_payload("sf-imp3", "user")])
    assert store_a.sessions.discard_unadmitted(SessionId("sf-imp3"), media_root=MEDIA_ROOT) == (SessionId("sf-imp3"),)
    store_a.sessions.create(_new("sf-imp3"))
    exported = store_a.maintenance.export_session(SessionId("sf-imp3"))
    # CONTRACT: the discarded id's old transcript is gone; the recreated session exports nothing.
    assert exported.messages == ()


def test_touch_labels_survive_into_the_export_header(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sf-exp1"))
    store_a.sessions.touch_activity(SessionId("sf-exp1"), T0S, description="active", provenance="agent.compression")
    header = store_a.maintenance.export_session(SessionId("sf-exp1")).session
    # CONTRACT: the durable activity trio the heartbeat writes is the trio export carries.
    assert (header.last_activity_at, header.last_activity_description, header.last_activity_provenance) == (T0S, "active", "agent.compression")


# ── session facets ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        ("   \t\n  ", None),
        ("  Hello   world ", "Hello world"),
        ("tab\tsep\nline", "tab sep line"),
        ("nul\x00bell\x07x", "nulbellx"),
        ("zero\u200bwidth\u200f bidi\u202e obj\ufffc feff\ufeff", "zerowidth bidi obj feff"),
        ("\ud800lone", "\ufffdlone"),
        ("Ünïcode 🚀 keeps CASE", "Ünïcode 🚀 keeps CASE"),
        ("\u200b" * 50, None),
    ],
)
def test_sanitize_title_table(raw, expected):
    # CONTRACT: pure and total; surrogates replaced, controls and invisibles stripped, whitespace collapsed, empty reads None.
    assert sanitize_title(raw) == expected


def test_sanitize_title_length_is_measured_after_cleaning():
    # CONTRACT: invisible padding cannot push a title past the cap; a whitespace run cannot save one.
    assert sanitize_title("a" * MAX_TITLE_LENGTH + "\u200b" * 10) == "a" * MAX_TITLE_LENGTH
    with pytest.raises(ValueError):
        sanitize_title("a" * (MAX_TITLE_LENGTH + 1))
    with pytest.raises(ValueError):
        sanitize_title(" ".join(["ab"] * MAX_TITLE_LENGTH))


def test_set_title_applies_the_contract_sanitizer(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sfc-title"))
    assert store_a.sessions.set_title(SessionId("sfc-title"), " zero\u200bwidth  \x00title ")
    assert store_a.sessions.get(SessionId("sfc-title")).title == "zerowidth title"
    with pytest.raises(ValueError):
        store_a.sessions.set_title(SessionId("sfc-title"), "b" * (MAX_TITLE_LENGTH + 1))
    # Refused: the previous title stands.
    assert store_a.sessions.get(SessionId("sfc-title")).title == "zerowidth title"


# ── repo discovery ────────────────────────────────────────────────────────────


def test_distinct_cwds_groups_counts_and_orders(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("sfc-cwd-a1", cwd="/w/a"))
    store_a.sessions.create(_new("sfc-cwd-a2", cwd="/w/a", parent_session_id=SessionId("sfc-cwd-a1")))
    store_a.sessions.set_hidden(SessionId("sfc-cwd-a2"), True)
    store_a.sessions.create(_new("sfc-cwd-b1", cwd="/w/b"))
    store_a.sessions.create(_new("sfc-cwd-blank", cwd="   "))
    store_a.sessions.create(_new("sfc-cwd-tab", cwd="\t"))
    store_a.sessions.create(_new("sfc-cwd-none"))
    store_a.sessions.create(_new("sfc-cwd-arch", cwd="/w/archived"))
    store_a.sessions.set_archived(SessionId("sfc-cwd-arch"), True)
    store_b.sessions.create(_new("sfc-cwd-other", cwd="/w/other-scope"))
    store_a.sessions.end(SessionId("sfc-cwd-a1"), "cli_close")
    usage = store_a.sessions.distinct_session_cwds()
    # CONTRACT: space-only and null cwds are dropped (a tab is not a space: admitted verbatim), archived rows only with the flag, hidden children count, cwd ascending, other scopes never count.
    assert [u.cwd for u in usage] == ["\t", "/w/a", "/w/b"]
    assert [u.sessions for u in usage] == [1, 2, 1]
    usage = usage[1:]
    a1 = store_a.sessions.get(SessionId("sfc-cwd-a1"))
    a2 = store_a.sessions.get(SessionId("sfc-cwd-a2"))
    # CONTRACT: last_active is the newest ended_at-else-started_at over the group.
    assert usage[0].last_active == max(a1.ended_at, a2.started_at)
    b1 = store_a.sessions.get(SessionId("sfc-cwd-b1"))
    assert usage[1] == SessionCwdUsage(cwd="/w/b", sessions=1, last_active=b1.started_at)
    with_archived = store_a.sessions.distinct_session_cwds(include_archived=True)
    assert [u.cwd for u in with_archived] == ["\t", "/w/a", "/w/archived", "/w/b"]
    other = store_b.sessions.get(SessionId("sfc-cwd-other"))
    assert store_b.sessions.distinct_session_cwds() == (SessionCwdUsage(cwd="/w/other-scope", sessions=1, last_active=other.started_at),)


def test_backfill_repo_roots_fills_empty_only(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("sfc-root-empty", cwd="/w/x"))
    store_a.sessions.create(_new("sfc-root-set", cwd="/w/x", git_repo_root="/w/x-known"))
    store_a.sessions.create(_new("sfc-root-other", cwd="/w/y"))
    store_a.sessions.create(_new("sfc-root-case", cwd="/W/X"))
    store_b.sessions.create(_new("sfc-root-b", cwd="/w/x"))
    store_a.sessions.backfill_repo_roots({"/w/x": "/w/x-root", "": "/ignored", "/w/y": ""})

    def root(scope, sid):
        return backend.open(scope).sessions.get(SessionId(sid)).git_repo_root

    # CONTRACT: empty roots filled, recorded roots kept, empty pairs ignored, verbatim match, other scopes untouched.
    assert root(scope_a, "sfc-root-empty") == "/w/x-root"
    assert root(scope_a, "sfc-root-set") == "/w/x-known"
    assert root(scope_a, "sfc-root-other") is None
    assert root(scope_a, "sfc-root-case") is None
    assert root(scope_b, "sfc-root-b") is None
    # CONTRACT: idempotent, and a second mapping never clobbers the first.
    store_a.sessions.backfill_repo_roots({"/w/x": "/w/x-DIFFERENT"})
    assert root(scope_a, "sfc-root-empty") == "/w/x-root"
    store_a.sessions.backfill_repo_roots({})
    assert root(scope_a, "sfc-root-empty") == "/w/x-root"


# ── owner repair ──────────────────────────────────────────────────────────────


def test_backfill_session_owners_is_idempotent_and_scope_local(backend, scope_a, scope_b):
    # A created row is owned by the scope's profile; an imported one carries no owner until the repair adopts it.
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("sfc-own-created"))
    store_a.maintenance.import_sessions([_import_payload(sid, None) for sid in ("sfc-own-a1", "sfc-own-a2")])
    store_b.maintenance.import_sessions([_import_payload(sid, None) for sid in ("sfc-own-b1", "sfc-own-b2", "sfc-own-b3")])
    assert store_a.sessions.get(SessionId("sfc-own-created")).profile_name == scope_a.name
    assert store_a.sessions.get(SessionId("sfc-own-a1")).profile_name is None
    # CONTRACT: the repair stamps the scope's profile on exactly the scope's unowned rows, then is idempotent.
    assert store_a.sessions.backfill_session_owners() == 2
    assert store_a.sessions.backfill_session_owners() == 0
    assert store_a.sessions.get(SessionId("sfc-own-a1")).profile_name == scope_a.name
    # CONTRACT: scope A's repair adopted nothing in scope B: B's own first repair still sees every one of B's unowned rows, then nothing.
    assert store_b.sessions.backfill_session_owners() == 3
    assert store_b.sessions.backfill_session_owners() == 0
    listed = store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))
    assert {str(e.session.session_id) for e in listed} == {"sfc-own-created", "sfc-own-a1", "sfc-own-a2"}


# ── cron runs ─────────────────────────────────────────────────────────────────


def test_list_cron_job_runs_legacy_ids_and_windows(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    ids = [f"cron_j1_20260921_12000{n}" for n in range(1, 6)]
    for sid in ids:
        store_a.sessions.create(_new(sid, source="cron"))
    store_a.sessions.set_archived(SessionId(ids[1]), True)
    store_a.sessions.end(SessionId(ids[2]), "cron_complete")
    store_a.sessions.create(_new("cron_j10_20260921_120001", source="cron"))
    store_a.sessions.create(_new("cron_j1_x_20260921_120001", source="cron"))
    store_a.sessions.create(_new("cron_j1_1", source="cron"))
    store_a.sessions.create(_new("cron_j1_20260921_120009", source="cli"))
    store_b.sessions.create(_new("cron_j1_20260921_120008", source="cron"))
    seen = [e for offset in range(0, 6, 2) for e in store_a.sessions.list_cron_job_runs("j1", SessionWindow(limit=2, offset=offset))]
    # CONTRACT: only source=cron rows named `cron_<job>_` plus exactly a YYYYmmdd_HHMMSS stamp, every visibility, newest first, no loss or duplication across windows, other scopes excluded.
    assert [str(e.entry.session.session_id) for e in seen] == list(reversed(ids))
    assert all((e.cron_job_id, e.cron_execution_id) == ("j1", None) for e in seen)
    ended = next(e for e in seen if e.entry.session.session_id == ids[2])
    assert ended.entry.session.ended_at is not None and ended.entry.last_active >= ended.entry.session.started_at
    assert all(e.entry.unread is False for e in seen)
    assert store_a.sessions.list_cron_job_runs("nope", SessionWindow(limit=5)) == ()
    # CONTRACT: an empty job id selects the `cron__` prefix like any other.
    store_a.sessions.create(_new("cron__20260921_120000", source="cron"))
    assert [str(e.entry.session.session_id) for e in store_a.sessions.list_cron_job_runs("", SessionWindow(limit=5))] == ["cron__20260921_120000"]


def test_cron_run_identity_reads_the_legacy_id(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("cron_nightly_20260921_120000", source="cron"))
    store_a.sessions.create(_new("plain-run", source="cron"))
    # CONTRACT: a pre-unified run id names its job with no execution; any other id without a scheduler admission reads None.
    identity = store_a.sessions.cron_run_identity(SessionId("cron_nightly_20260921_120000"))
    assert (identity.cron_job_id, identity.cron_execution_id) == ("nightly", None)
    assert store_a.sessions.cron_run_identity(SessionId("plain-run")) is None
    assert store_a.sessions.cron_run_identity(SessionId("sf-ghost")) is None


# ── the bounded recent browse ─────────────────────────────────────────────────


def _build_recent_fixture(backend, scope) -> None:
    store = backend.open(scope)
    s = backend.open(scope).sessions
    # Lineage: root R (titled) --compression--> C1 --compression--> tip C2 (untitled).
    s.create(_new("sfc-rb-root", cwd="/w/r"))
    s.set_title(SessionId("sfc-rb-root"), "Root title")
    s.end(SessionId("sfc-rb-root"), END_REASON_COMPRESSION)
    s.create(_new("sfc-rb-c1", parent_session_id=SessionId("sfc-rb-root")))
    s.end(SessionId("sfc-rb-c1"), END_REASON_COMPRESSION)
    s.create(_new("sfc-rb-c2", parent_session_id=SessionId("sfc-rb-c1")))
    store.messages.append_batch(SessionId("sfc-rb-c2"), [_msg("user", "q"), _msg("assistant", "a")])
    # Plain root P, the freshest by activity.
    s.create(_new("sfc-rb-plain"))
    s.touch_activity(SessionId("sfc-rb-plain"), (T0 + timedelta(days=3650)).timestamp())
    # Excluded rows: hidden, archived, delegate child, excluded source.
    s.create(_new("sfc-rb-hidden"))
    s.set_hidden(SessionId("sfc-rb-hidden"), True)
    s.create(_new("sfc-rb-archived"))
    s.set_archived(SessionId("sfc-rb-archived"), True)
    s.create(_new("sfc-rb-delegate", parent_session_id=SessionId("sfc-rb-plain"), lineage_kind=SessionLineageKind.DELEGATE))
    s.create(_new("sfc-rb-webhook", source="webhook"))


def test_list_recent_sessions_projects_lineage_and_orders(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    _build_recent_fixture(backend, scope_a)
    store_b.sessions.create(_new("sfc-rb-other-scope"))
    entries = store_a.sessions.list_recent_sessions(RecentBrowseQuery(limit=10, exclude_sources=("webhook",)))
    # CONTRACT: hidden, archived, delegate and excluded-source rows are out; a compression lineage is one conversation projected to its tip; other scopes never appear.
    assert [str(e.session_id) for e in entries] == ["sfc-rb-plain", "sfc-rb-c2"]
    plain, tip = entries
    assert plain.lineage_root_id is None and plain.title is None
    root = store_a.sessions.get(SessionId("sfc-rb-root"))
    # CONTRACT: root started_at, title falls back to the root's, lineage_root_id names the root, counters are the tip's.
    assert tip.lineage_root_id == "sfc-rb-root"
    assert tip.started_at == root.started_at
    assert tip.title == "Root title"
    assert tip.message_count == 2 and tip.tool_call_count == 0 and tip.preview == "q"
    assert tip.source == "cli" and tip.ended_at is None
    assert plain.last_active >= tip.last_active
    # CONTRACT: the browse reads no read watermark, so no entry is unread.
    assert not any(e.unread for e in entries)
    # CONTRACT: the excluded source is listed when not excluded; limit bounds the result.
    unfiltered = store_a.sessions.list_recent_sessions(RecentBrowseQuery(limit=10))
    assert {str(e.session_id) for e in unfiltered} == {"sfc-rb-plain", "sfc-rb-c2", "sfc-rb-webhook"}
    assert [str(e.session_id) for e in store_a.sessions.list_recent_sessions(RecentBrowseQuery(limit=1))] == ["sfc-rb-plain"]


def test_list_recent_sessions_edges_follow_the_current_parent_row(backend, scope_a):
    s = backend.open(scope_a).sessions
    # The child exists BEFORE the parent ends by compression: the edge is a join on current rows, not a creation-time stamp.
    s.create(_new("sfc-rl-root"))
    s.create(_new("sfc-rl-child", parent_session_id=SessionId("sfc-rl-root")))
    before = s.list_recent_sessions(RecentBrowseQuery(limit=10))
    # CONTRACT: a plain child of a live parent is not a listable root, and the parent is its own conversation.
    assert [str(e.session_id) for e in before] == ["sfc-rl-root"]
    s.end(SessionId("sfc-rl-root"), END_REASON_COMPRESSION)
    after = s.list_recent_sessions(RecentBrowseQuery(limit=10))
    # CONTRACT: once the parent has ended by compression, the child is the lineage tip.
    assert [(str(e.session_id), e.lineage_root_id) for e in after] == [("sfc-rl-child", "sfc-rl-root")]


def test_list_recent_sessions_lineage_ceiling_counts_seeds_per_direction(backend, scope_a):
    s = backend.open(scope_a).sessions
    s.create(_new("sfc-rc-r"))
    s.end(SessionId("sfc-rc-r"), END_REASON_COMPRESSION)
    s.create(_new("sfc-rc-c", parent_session_id=SessionId("sfc-rc-r")))
    s.end(SessionId("sfc-rc-c"), END_REASON_COMPRESSION)
    s.create(_new("sfc-rc-t", parent_session_id=SessionId("sfc-rc-c")))
    # Three rows per direction (seed + two steps) fit a ceiling of 3 in each direction independently.
    fits = s.list_recent_sessions(RecentBrowseQuery(limit=1, candidate_limit=1, lineage_limit=3))
    assert [str(e.session_id) for e in fits] == ["sfc-rc-t"]
    # CONTRACT: a ceiling that cuts the walk before a terminal row omits the lineage instead of expanding it.
    cut = s.list_recent_sessions(RecentBrowseQuery(limit=1, candidate_limit=1, lineage_limit=2))
    assert cut == ()


def test_list_recent_sessions_zero_deadline_raises_without_fallback(backend, scope_a):
    store_a = backend.open(scope_a)
    _build_recent_fixture(backend, scope_a)
    # CONTRACT: an expired deadline raises the typed transient error, a TimeoutError carrying the store's text, and yields no rows.
    with pytest.raises(RecentBrowseTimeout) as timed_out:
        store_a.sessions.list_recent_sessions(RecentBrowseQuery(limit=10, timeout_seconds=0))
    assert isinstance(timed_out.value, TimeoutError) and str(timed_out.value) == "recent-session browse exceeded 0s deadline"


def test_recent_browse_query_normalizes_bounds():
    q = RecentBrowseQuery(limit=0, timeout_seconds=-1, candidate_limit=1, lineage_limit=1)
    # CONTRACT: the source's clamps, applied at construction.
    assert (q.limit, q.timeout_seconds, q.candidate_limit, q.lineage_limit) == (1, 0.0, 1, 1)
    d = RecentBrowseQuery(limit=20)
    assert (d.timeout_seconds, d.candidate_limit, d.lineage_limit) == (3.0, 160, 1280)
    big = RecentBrowseQuery(limit=100, candidate_limit=10**6, lineage_limit=10**7)
    assert (big.candidate_limit, big.lineage_limit) == (2048, 8192)
    # The lower clamp wins over the ceiling: a limit above the candidate cap is still honoured as the candidate floor (the source's max(limit, min(...)) order).
    assert RecentBrowseQuery(limit=5000).candidate_limit == 5000


# ── skill-scaffolded sessions ─────────────────────────────────────────────────


def test_list_skill_scaffolded_sessions_first_user_row_rule(backend, scope_a, scope_b):
    store_b = backend.open(scope_b)
    s, m = backend.open(scope_a).sessions, backend.open(scope_a).messages
    scaffold = SKILL_SCAFFOLD_PREFIX + "'deploy' skill.]\n\nship it"
    s.create(_new("sfc-sk-hit"))
    s.set_title(SessionId("sfc-sk-hit"), "deploy run")
    m.append_batch(SessionId("sfc-sk-hit"), [_msg("user", scaffold), _msg("assistant", "ok")])
    s.create(_new("sfc-sk-second"))
    s.set_title(SessionId("sfc-sk-second"), "second turn")
    m.append_batch(SessionId("sfc-sk-second"), [_msg("user", "plain"), _msg("user", scaffold, at=T0 + timedelta(seconds=1))])
    s.create(_new("sfc-sk-untitled"))
    m.append(SessionId("sfc-sk-untitled"), _msg("user", scaffold))
    s.create(_new("sfc-sk-after-assistant"))
    s.set_title(SessionId("sfc-sk-after-assistant"), "assistant first")
    m.append_batch(SessionId("sfc-sk-after-assistant"), [_msg("assistant", "hello"), _msg("user", scaffold, at=T0 + timedelta(seconds=1))])
    s.create(_new("sfc-sk-tree"))
    s.set_title(SessionId("sfc-sk-tree"), "tree content")
    m.append(SessionId("sfc-sk-tree"), NewMessage(role="user", content=[{"type": "text", "text": scaffold}], timestamp=T0S))
    store_b.sessions.create(_new("sfc-sk-other"))
    store_b.sessions.set_title(SessionId("sfc-sk-other"), "other scope")
    store_b.messages.append(SessionId("sfc-sk-other"), _msg("user", scaffold))
    rows = s.list_skill_scaffolded_sessions()
    # CONTRACT: only titled sessions whose FIRST user row is text starting with the prefix; newest first; content is that row's text; other scopes excluded.
    assert [(str(r.session_id), r.title, r.content) for r in rows] == [
        ("sfc-sk-after-assistant", "assistant first", scaffold),
        ("sfc-sk-hit", "deploy run", scaffold),
    ]
    assert [str(r.session_id) for r in s.list_skill_scaffolded_sessions(limit=1)] == ["sfc-sk-after-assistant"]
    # CONTRACT: the limit is the store's LIMIT: zero lists nothing.
    assert s.list_skill_scaffolded_sessions(limit=0) == ()


def test_list_skill_scaffolded_sessions_admits_the_auto_load_prefix(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence.conversation.sessions import AUTO_LOAD_SCAFFOLD_PREFIX

    auto = AUTO_LOAD_SCAFFOLD_PREFIX + "deploy\" skill was auto-loaded.]"
    store_a.sessions.create(_new("sfc-sk-auto"))
    store_a.sessions.set_title(SessionId("sfc-sk-auto"), "auto loaded")
    store_a.messages.append(SessionId("sfc-sk-auto"), _msg("user", auto))
    # CONTRACT: a gateway skill auto-load scaffold counts like a /skill expansion.
    assert [(str(r.session_id), r.content) for r in store_a.sessions.list_skill_scaffolded_sessions()] == [("sfc-sk-auto", auto)]


# ── unlinked Telegram sessions ────────────────────────────────────────────────


def test_list_unlinked_telegram_sessions_admission_and_order(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    s = backend.open(scope_a).sessions
    for sid in ("sfc-tg-1", "sfc-tg-2", "sfc-tg-3"):
        s.create(_new(sid, source="telegram", user_id="u1", chat_id="c1", session_key=f"telegram:{sid}"))
    s.create(_new("sfc-tg-u2", source="telegram", user_id="u2"))
    s.create(_new("sfc-tg-cli", source="cli", user_id="u1"))
    s.create(_new("sfc-tg-foreign", source="telegram", user_id="u1", profile_name="someone-else"))
    store_b.sessions.create(_new("sfc-tg-b", source="telegram", user_id="u1"))
    s.set_archived(SessionId("sfc-tg-3"), True)
    s.touch_activity(SessionId("sfc-tg-1"), (T0 + timedelta(days=3650)).timestamp())
    # CONTRACT: before the topic tables exist the listing is unscoped: this user's telegram rows of any owner.
    assert {str(e.session.session_id) for e in s.list_unlinked_telegram_sessions(user_id="u1")} == {"sfc-tg-1", "sfc-tg-2", "sfc-tg-3", "sfc-tg-foreign"}
    store_a.routing.prepare_topic_storage()
    listed = s.list_unlinked_telegram_sessions(user_id="u1")
    # CONTRACT: telegram rows of this user owned by the scope's profile, every visibility, freshest activity first, other users/sources/owners/scopes excluded.
    assert str(listed[0].session.session_id) == "sfc-tg-1"
    assert {str(e.session.session_id) for e in listed} == {"sfc-tg-1", "sfc-tg-2", "sfc-tg-3"}
    assert listed[0].last_active == (T0 + timedelta(days=3650)).timestamp()
    store_a.routing.bind_topic(chat_id="c1", thread_id="t1", user_id="u1", session_key="telegram:sfc-tg-2", session_id="sfc-tg-2")
    # CONTRACT: a topic-bound session leaves the listing.
    assert {str(e.session.session_id) for e in s.list_unlinked_telegram_sessions(user_id="u1")} == {"sfc-tg-1", "sfc-tg-3"}
    assert [str(e.session.session_id) for e in s.list_unlinked_telegram_sessions(user_id="u1", limit=1)] == ["sfc-tg-1"]
    assert s.list_unlinked_telegram_sessions(user_id="nobody") == ()
    # CONTRACT: an empty user id is a plain value, not an error (no stored row carries it here); a zero limit lists nothing.
    assert s.list_unlinked_telegram_sessions(user_id="") == ()
    assert s.list_unlinked_telegram_sessions(user_id="u1", limit=0) == ()


def test_list_unlinked_telegram_sessions_lists_the_named_profile(backend, scope_a):
    s = backend.open(scope_a).sessions
    owners = {"sfc-tgp-own": None, "sfc-tgp-unset": None, "sfc-tgp-blank": None, "sfc-tgp-default": "default", "sfc-tgp-work": "work", "sfc-tgp-padded": "work"}
    for sid, owner in owners.items():
        s.create(_new(sid, source="telegram", user_id="u1", profile_name=owner))
    rows = backend.scope_store(scope_a)["sessions"]
    rows["sfc-tgp-unset"]["profile_name"] = None
    rows["sfc-tgp-blank"]["profile_name"] = "  "
    rows["sfc-tgp-padded"]["profile_name"] = " work "

    def listed(**kwargs):
        return {str(e.session.session_id) for e in s.list_unlinked_telegram_sessions(user_id="u1", **kwargs)}

    # CONTRACT: before the topic tables exist a named profile changes nothing: the unscoped query lists every owner.
    assert listed(profile_name="work") == set(owners)
    backend.open(scope_a).routing.prepare_topic_storage()
    # CONTRACT: a named profile lists that profile's rows of the same store, the stored owner compared with spaces trimmed and the name stripped.
    assert listed(profile_name="work") == listed(profile_name=" work ") == {"sfc-tgp-work", "sfc-tgp-padded"}
    # CONTRACT: ``default`` and a blank name list the rows stamped ``default`` and those with an unset or blank owner.
    assert listed(profile_name="default") == listed(profile_name="") == {"sfc-tgp-unset", "sfc-tgp-blank", "sfc-tgp-default"}
    # CONTRACT: no name lists the store's own profile.
    assert listed() == {"sfc-tgp-own"}
    assert listed(profile_name="nobody") == set()


# ── gateway listing entries ───────────────────────────────────────────────────


def test_gateway_entry_carries_usage_and_peer_columns(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sfc-gw", source="telegram", session_key="telegram:sfc-gw", chat_id="c9"))

    def entry():
        return next(e for e in store_a.sessions.list_gateway_sessions() if e.session.session_id == "sfc-gw")

    e = entry()
    # CONTRACT: no peer, no usage: the columns read None / 0.
    assert (e.session.display_name, e.session.origin, e.session.input_tokens, e.session.output_tokens) == (None, None, 0, 0)
    store_a.usage.queue_usage(SessionId("sfc-gw"), UsageDelta(tokens=TokenCounts(input_tokens=5, output_tokens=7), api_call_count=1))
    e = entry()
    # CONTRACT: token counters follow UsageRepository accounting.
    assert (e.session.input_tokens, e.session.output_tokens) == (5, 7)
    # A later create fills the peer columns the first writer left unset.
    store_a.sessions.create(_new("sfc-gw", source="telegram", display_name="Ada", origin={"platform": "telegram", "chat_id": "c9"}))
    e = entry()
    # CONTRACT: the recorded peer columns read back as the display name and the parsed origin object.
    assert e.session.display_name == "Ada"
    assert dict(e.session.origin) == {"platform": "telegram", "chat_id": "c9"}
