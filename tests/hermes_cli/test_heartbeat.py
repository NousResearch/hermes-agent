"""Tests for /heartbeat (hermes_cli/heartbeat.py)."""

import time

import pytest

from hermes_cli.heartbeat import (
    PROFILE_SCOPE_KEY,
    HeartbeatManager,
    HeartbeatState,
    MIN_INTERVAL_SECONDS,
    format_interval,
    load_heartbeat,
    migrate_heartbeat_to_session,
    parse_interval,
    promote_session_heartbeat_to_profile,
    save_heartbeat,
)


# ──────────────────────────────────────────────────────────────────────
# interval parsing
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("10m", 600),
        ("every 10m", 600),
        ("2h", 7200),
        ("every 2 hours", 7200),
        ("1d", 86400),
        ("90 minutes", 5400),
        ("600s", 600),
    ],
)
def test_parse_interval_valid(text, expected):
    assert parse_interval(text) == expected


@pytest.mark.parametrize("text", ["", "banana", "check CI", "every", "m10"])
def test_parse_interval_not_an_interval(text):
    assert parse_interval(text) is None


def test_parse_interval_too_small_is_rejected():
    assert parse_interval("5s") == -1
    assert parse_interval("30s") == -1
    # Exactly the floor is allowed.
    assert parse_interval(f"{MIN_INTERVAL_SECONDS}s") == MIN_INTERVAL_SECONDS


def test_format_interval():
    assert format_interval(600) == "10m"
    assert format_interval(7200) == "2h"
    assert format_interval(86400) == "1d"
    assert format_interval(90) == "90s"


# ──────────────────────────────────────────────────────────────────────
# state + due logic
# ──────────────────────────────────────────────────────────────────────


def test_state_roundtrip():
    s = HeartbeatState(prompt="check CI", interval_seconds=600, created_at=time.time())
    loaded = HeartbeatState.from_json(s.to_json())
    assert loaded.prompt == "check CI"
    assert loaded.interval_seconds == 600
    assert loaded.status == "active"


def test_is_due_anchors_on_created_then_last_fired():
    now = time.time()
    s = HeartbeatState(prompt="p", interval_seconds=600, created_at=now)
    assert s.is_due(now + 1) is False
    assert s.is_due(now + 601) is True
    s.last_fired_at = now + 601
    assert s.is_due(now + 700) is False
    assert s.is_due(now + 1300) is True


def test_paused_never_due():
    now = time.time()
    s = HeartbeatState(prompt="p", interval_seconds=60, created_at=now - 3600, status="paused")
    assert s.is_due(now) is False


def test_render_prompt_contains_instruction_and_interval():
    s = HeartbeatState(prompt="check the deploy", interval_seconds=600)
    rendered = s.render_prompt()
    assert "check the deploy" in rendered
    assert "10m" in rendered


# ──────────────────────────────────────────────────────────────────────
# manager
# ──────────────────────────────────────────────────────────────────────


def test_manager_set_pause_resume_clear():
    mgr = HeartbeatManager(session_id="hb-lifecycle-sid")
    state = mgr.set("watch CI", 600)
    assert state.status == "active"
    assert mgr.is_active()

    mgr.pause()
    assert not mgr.is_active()
    assert mgr.has_heartbeat()

    mgr.resume()
    assert mgr.is_active()

    assert mgr.clear() is True
    assert not mgr.has_heartbeat()
    # Cleared rows don't resurrect on reload.
    assert load_heartbeat("hb-lifecycle-sid") is None


def test_manager_rejects_bad_input():
    mgr = HeartbeatManager(session_id="hb-bad-sid")
    with pytest.raises(ValueError):
        mgr.set("", 600)
    with pytest.raises(ValueError):
        mgr.set("ok", 5)


def test_manager_persists_across_instances():
    mgr = HeartbeatManager(session_id="hb-persist-sid")
    mgr.set("persisted prompt", 600)
    again = HeartbeatManager(session_id="hb-persist-sid")
    assert again.has_heartbeat()
    assert again.state.prompt == "persisted prompt"


def test_due_prompt_fires_once_and_reanchors():
    mgr = HeartbeatManager(session_id="hb-due-sid")
    mgr.set("tick", 600)
    # Not due immediately after set.
    assert mgr.due_prompt() is None
    # Force due by rewinding the anchor.
    mgr.state.created_at = time.time() - 700
    prompt = mgr.due_prompt()
    assert prompt is not None and "tick" in prompt
    assert mgr.state.fire_count == 1
    # Immediately after firing it re-anchors — not due again.
    assert mgr.due_prompt() is None


def test_missed_ticks_coalesce():
    mgr = HeartbeatManager(session_id="hb-coalesce-sid")
    mgr.set("tick", 600)
    # Simulate 5 missed intervals: exactly ONE fire results.
    mgr.state.created_at = time.time() - 600 * 5 - 10
    assert mgr.due_prompt() is not None
    assert mgr.due_prompt() is None
    assert mgr.state.fire_count == 1


def test_resume_reanchors_instead_of_instant_fire():
    mgr = HeartbeatManager(session_id="hb-resume-sid")
    mgr.set("tick", 600)
    mgr.state.created_at = time.time() - 3600
    mgr.pause()
    mgr.resume()
    assert mgr.due_prompt() is None


# ──────────────────────────────────────────────────────────────────────
# compression migration
# ──────────────────────────────────────────────────────────────────────


def test_migrate_heartbeat_to_session():
    save_heartbeat(
        "hb-parent-sid",
        HeartbeatState(prompt="carry me", interval_seconds=600, created_at=time.time()),
    )
    assert migrate_heartbeat_to_session("hb-parent-sid", "hb-child-sid") is True
    child = load_heartbeat("hb-child-sid")
    assert child is not None and child.prompt == "carry me"
    assert load_heartbeat("hb-parent-sid") is None


def test_migrate_noop_without_source():
    assert migrate_heartbeat_to_session("hb-none-a", "hb-none-b") is False
    assert migrate_heartbeat_to_session("same", "same") is False


def test_migrate_never_moves_the_profile_row():
    """A compression rotation must not carry the profile-wide heartbeat into a session, or archive it."""
    mgr = HeartbeatManager(session_id=PROFILE_SCOPE_KEY, scope="profile")
    mgr.set("profile-wide", 600)
    assert migrate_heartbeat_to_session(PROFILE_SCOPE_KEY, "hb-rotate-child") is False
    assert load_heartbeat(PROFILE_SCOPE_KEY) is not None


# ──────────────────────────────────────────────────────────────────────
# profile scope: one standing instruction that outlives the session
# ──────────────────────────────────────────────────────────────────────


def _set_profile(prompt: str = "profile-wide", interval: int = 600) -> HeartbeatState:
    return HeartbeatManager(session_id=PROFILE_SCOPE_KEY, scope="profile").set(prompt, interval)


def test_profile_heartbeat_is_inherited_by_a_fresh_session():
    _set_profile()
    mgr = HeartbeatManager(session_id="hb-inherit-fresh")
    assert mgr.has_heartbeat()
    assert mgr.is_inherited is True
    assert mgr.state.prompt == "profile-wide"


def test_inherited_heartbeat_does_not_fire_a_backlog_on_adoption():
    """A session joining late waits a full interval: the profile clock is not replayed into it."""
    _set_profile(interval=600)
    mgr = HeartbeatManager(session_id="hb-inherit-late")
    assert mgr.due_prompt() is None
    assert mgr.due_prompt(now=time.time() + 601) is not None


def test_inherited_heartbeat_materialises_into_the_session_on_first_fire():
    _set_profile()
    mgr = HeartbeatManager(session_id="hb-inherit-materialise")
    mgr.due_prompt(now=time.time() + 601)

    persisted = load_heartbeat("hb-inherit-materialise")
    assert persisted is not None, "the fired tick must survive the conversation, under the session's own key"
    assert persisted.fire_count == 1
    assert mgr.is_inherited is False

    # A later session picks up the profile default again, unaffected by the materialised copy.
    assert HeartbeatManager(session_id="hb-inherit-materialise-2").is_inherited is True


def test_two_sessions_do_not_both_fire_the_same_tick():
    """The profile row is the cadence: the first idle session to claim a tick takes it, the rest stay quiet."""
    _set_profile(interval=600)
    first = HeartbeatManager(session_id="hb-shared-a")
    second = HeartbeatManager(session_id="hb-shared-b")
    fire_at = time.time() + 601

    assert first.due_prompt(now=fire_at) is not None
    assert second.due_prompt(now=fire_at) is None, "the second session must not re-fire the tick just taken"
    assert load_heartbeat("hb-shared-a").fire_count == 1
    assert load_heartbeat("hb-shared-b") is None, "the losing session keeps no row until it wins a tick"


def test_adopted_anchor_survives_a_fresh_manager_each_poll():
    """Gateway and TUI pollers rebuild the manager every few seconds; a per-instance anchor would
    reset forever and an inherited heartbeat would never come due."""
    _set_profile(interval=600)
    first_poll = time.time()
    assert HeartbeatManager(session_id="hb-rebuilt").due_prompt(now=first_poll) is None
    # A brand-new manager, as the poller builds on every tick, still knows the profile's real schedule.
    later = first_poll + 601
    assert HeartbeatManager(session_id="hb-rebuilt").due_prompt(now=later) is not None


def test_a_session_joining_late_does_not_replay_a_backlog():
    _set_profile(interval=600)
    claimed_at = time.time()
    assert HeartbeatManager(session_id="hb-early").due_prompt(now=claimed_at + 601) is not None
    # A session created long after the profile fired must wait for the next tick, not fire immediately.
    assert HeartbeatManager(session_id="hb-late").due_prompt(now=claimed_at + 601) is None
    assert HeartbeatManager(session_id="hb-late").due_prompt(now=claimed_at + 1202) is not None


def test_inherited_tick_is_refunded_to_the_profile_when_the_turn_never_starts():
    _set_profile(interval=600)
    mgr = HeartbeatManager(session_id="hb-inherit-refund")
    fire_at = time.time() + 601
    assert mgr.due_prompt(now=fire_at) is not None
    assert mgr.abandon_fire() is True
    assert load_heartbeat("hb-inherit-refund").fire_count == 0, "tick not consumed — still due next poll"
    # The shared tick came back too, so another session can still take it.
    assert HeartbeatManager(session_id="hb-inherit-refund-other").due_prompt(now=fire_at) is not None


def test_inherited_heartbeat_follows_a_replaced_profile_prompt():
    """Re-setting the profile heartbeat must reach sessions already firing the old one."""
    _set_profile("first prompt", interval=600)
    mgr = HeartbeatManager(session_id="hb-replaced")
    assert mgr.due_prompt(now=time.time() + 601) is not None  # materialised on the first prompt

    _set_profile("second prompt", interval=600)
    later = time.time() + 1202
    prompt = mgr.due_prompt(now=later)
    assert prompt is not None and "second prompt" in prompt
    assert "first prompt" not in prompt, "must not keep firing an instruction the profile replaced"
    assert load_heartbeat("hb-replaced").prompt == "second prompt"


def test_clearing_the_profile_detaches_a_session_but_keeps_its_copy():
    """/heartbeat profile clear stops the standing instruction without silently killing a live session.

    The detach happens on the session's next fire attempt rather than on a read, so a status or snapshot
    read never writes.
    """
    _set_profile(interval=600)
    mgr = HeartbeatManager(session_id="hb-detach")
    assert mgr.due_prompt(now=time.time() + 601) is not None
    assert load_heartbeat("hb-detach").from_profile is True

    HeartbeatManager(session_id=PROFILE_SCOPE_KEY, scope="profile").clear()
    assert load_heartbeat("hb-detach") is not None, "the live session keeps its copy"
    # The next tick detaches it and it carries on from its own clock.
    assert mgr.due_prompt(now=time.time() + 1202) is not None
    assert load_heartbeat("hb-detach").from_profile is False


def test_session_heartbeat_wins_over_the_profile_default():
    _set_profile("profile default")
    own = HeartbeatManager(session_id="hb-own-wins")
    own.set("session specific", 900)
    assert own.is_inherited is False
    assert own.state.prompt == "session specific"


def test_paused_profile_heartbeat_is_not_inherited():
    mgr = HeartbeatManager(session_id=PROFILE_SCOPE_KEY, scope="profile")
    mgr.set("quiet please", 600)
    mgr.pause()
    assert HeartbeatManager(session_id="hb-inherit-paused").has_heartbeat() is False


def test_cleared_profile_heartbeat_stops_new_sessions_only():
    """/heartbeat profile clear disarms future sessions; the one already firing keeps its own copy."""
    _set_profile(interval=600)
    live = HeartbeatManager(session_id="hb-clear-live")
    live.due_prompt(now=time.time() + 601)

    HeartbeatManager(session_id=PROFILE_SCOPE_KEY, scope="profile").clear()
    assert HeartbeatManager(session_id="hb-clear-fresh").has_heartbeat() is False
    assert load_heartbeat("hb-clear-live") is not None, "clearing the profile must not disarm a live session"


def test_clearing_a_session_heartbeat_leaves_the_profile_one():
    _set_profile()
    mgr = HeartbeatManager(session_id="hb-clear-own")
    mgr.clear()
    assert HeartbeatManager(session_id=PROFILE_SCOPE_KEY, scope="profile").has_heartbeat() is True


def test_promote_copies_the_session_heartbeat_to_the_profile():
    mgr = HeartbeatManager(session_id="hb-promote-src")
    mgr.set("watch the deploy", 900)
    promoted = promote_session_heartbeat_to_profile("hb-promote-src")
    assert promoted is not None and promoted.prompt == "watch the deploy"

    profile = load_heartbeat(PROFILE_SCOPE_KEY)
    assert profile is not None and profile.interval_seconds == 900
    # The promoted row starts its own clock so a later session waits a full interval.
    fresh = HeartbeatManager(session_id="hb-promote-dst")
    assert fresh.is_inherited is True
    assert fresh.due_prompt() is None


def test_promote_without_a_session_heartbeat_is_a_noop():
    assert promote_session_heartbeat_to_profile("hb-promote-empty") is None


def test_session_scope_refuses_the_profile_sentinel():
    """Handing the sentinel to a session manager would alias both scopes onto one key."""
    with pytest.raises(ValueError):
        HeartbeatManager(session_id=PROFILE_SCOPE_KEY)
    save_heartbeat(PROFILE_SCOPE_KEY, HeartbeatState(prompt="clobber", interval_seconds=600))
    assert load_heartbeat(PROFILE_SCOPE_KEY) is None
