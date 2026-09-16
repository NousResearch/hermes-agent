"""Tests for /heartbeat (hermes_cli/heartbeat.py)."""

import time

import pytest

from hermes_cli.heartbeat import (
    HeartbeatManager,
    HeartbeatState,
    MIN_INTERVAL_SECONDS,
    format_interval,
    load_heartbeat,
    migrate_heartbeat_to_session,
    parse_interval,
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


@pytest.mark.parametrize("text", ["", "banana", "check CI", "every", "m10", "9" * 400 + "s"])
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
    assert "Heartbeat" in rendered


# ──────────────────────────────────────────────────────────────────────
# manager
# ──────────────────────────────────────────────────────────────────────


def _age(key: str, **fields) -> None:
    """Persist an aged heartbeat state — managers read the row, never an in-memory copy."""
    state = load_heartbeat(key)
    assert state is not None
    for name, value in fields.items():
        setattr(state, name, value)
    save_heartbeat(key, state)


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


def test_manager_update_edits_message_and_interval_in_place():
    mgr = HeartbeatManager(session_id="hb-update-sid")
    mgr.set("watch CI", 600)
    created_at = mgr.state.created_at
    _age("hb-update-sid", fire_count=3)

    state = mgr.update(prompt="watch the staging deploy", interval_seconds=1800)
    assert (state.prompt, state.interval_seconds, state.status) == ("watch the staging deploy", 1800, "active")
    # An edit is not a reset: identity and history survive.
    assert state.created_at == created_at and state.fire_count == 3
    again = HeartbeatManager(session_id="hb-update-sid")
    assert (again.state.prompt, again.state.interval_seconds, again.state.fire_count) == (
        "watch the staging deploy", 1800, 3)


def test_update_allows_partial_edits():
    mgr = HeartbeatManager(session_id="hb-update-partial-sid")
    mgr.set("tick", 600)

    assert mgr.update(prompt="tick v2").interval_seconds == 600
    assert mgr.update(interval_seconds=120).prompt == "tick v2"


def test_update_interval_change_reanchors_while_a_message_edit_keeps_the_schedule():
    mgr = HeartbeatManager(session_id="hb-update-anchor-sid")
    mgr.set("tick", 600)
    aged = time.time() - 590
    _age("hb-update-anchor-sid", last_fired_at=aged)

    # Message-only edit: the existing anchor stands, so a nearly-due tick stays due.
    mgr.update(prompt="tick v2")
    assert mgr.state.last_fired_at == aged

    # Interval change: re-anchored to NOW, so a shorter interval never fires instantly.
    _age("hb-update-anchor-sid", last_fired_at=aged)
    mgr.update(interval_seconds=120)
    assert mgr.state.last_fired_at > aged
    assert mgr.due_prompt() is None


def test_update_rejects_bad_input_and_a_missing_heartbeat():
    mgr = HeartbeatManager(session_id="hb-update-bad-sid")
    with pytest.raises(ValueError):
        mgr.update(prompt="ghost")

    mgr.set("tick", 600)
    with pytest.raises(ValueError):
        mgr.update()
    with pytest.raises(ValueError):
        mgr.update(prompt="   ")
    with pytest.raises(ValueError):
        mgr.update(interval_seconds=5)
    with pytest.raises(ValueError):
        mgr.update(interval_seconds=0)
    # A rejected edit changed nothing.
    assert (mgr.state.prompt, mgr.state.interval_seconds) == ("tick", 600)

    mgr.clear()
    with pytest.raises(ValueError):
        mgr.update(prompt="ghost")


def test_update_keeps_a_paused_heartbeat_paused():
    mgr = HeartbeatManager(session_id="hb-update-paused-sid")
    mgr.set("tick", 600)
    mgr.pause()

    state = mgr.update(prompt="tick v2", interval_seconds=1800)
    assert state.status == "paused"
    assert mgr.has_heartbeat() and not mgr.is_active()


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
    # Force due by rewinding the anchor (persisted — the manager reads the row).
    _age("hb-due-sid", created_at=time.time() - 700)
    prompt = mgr.due_prompt()
    assert prompt is not None and "tick" in prompt
    assert mgr.state.fire_count == 1
    # Immediately after firing it re-anchors — not due again.
    assert mgr.due_prompt() is None


def test_abandon_fire_preserves_a_message_edited_after_the_claim():
    mgr = HeartbeatManager(session_id="hb-edit-claim-sid")
    mgr.set("original", 600)
    _age("hb-edit-claim-sid", created_at=time.time() - 700)
    assert mgr.due_prompt() is not None

    HeartbeatManager(mgr.session_id).update(prompt="edited")
    assert mgr.abandon_fire()

    persisted = load_heartbeat(mgr.session_id)
    assert persisted is not None
    assert persisted.prompt == "edited"
    assert persisted.fire_count == 0
    assert persisted.is_due()
    prompt = mgr.due_prompt()
    assert prompt is not None and "edited" in prompt


def test_missed_ticks_coalesce():
    mgr = HeartbeatManager(session_id="hb-coalesce-sid")
    mgr.set("tick", 600)
    # Simulate 5 missed intervals: exactly ONE fire results.
    _age("hb-coalesce-sid", created_at=time.time() - 600 * 5 - 10)
    assert mgr.due_prompt() is not None
    assert mgr.due_prompt() is None
    assert mgr.state.fire_count == 1


def test_resume_reanchors_instead_of_instant_fire():
    mgr = HeartbeatManager(session_id="hb-resume-sid")
    mgr.set("tick", 600)
    _age("hb-resume-sid", created_at=time.time() - 3600)
    mgr.pause()
    mgr.resume()
    assert mgr.due_prompt() is None


def test_a_stale_cached_instance_never_overwrites_a_newer_edit():
    """The card edits through the backend process while the CLI / slash worker holds its own cached
    instance: that instance's next command must persist the edit, not its superseded copy."""
    worker = HeartbeatManager(session_id="hb-stale-sid")
    worker.set("original prompt", 60)
    _age("hb-stale-sid", created_at=time.time() - 700)

    HeartbeatManager("hb-stale-sid").update(prompt="edited prompt", interval_seconds=600)

    worker.pause()
    persisted = load_heartbeat("hb-stale-sid")
    assert (persisted.prompt, persisted.interval_seconds, persisted.status) == ("edited prompt", 600, "paused")


def test_a_stale_cached_instance_never_fires_the_superseded_instruction():
    """The worker's watchdog holds a pre-edit copy that believes the OLD interval is due: it must not
    enqueue the old instruction, and must not rewrite the row with the stale values."""
    HeartbeatManager("hb-stale-fire-sid").set("original prompt", 60)
    _age("hb-stale-fire-sid", created_at=time.time() - 700)
    watchdog = HeartbeatManager("hb-stale-fire-sid")  # cached view: due, old prompt/interval

    HeartbeatManager("hb-stale-fire-sid").update(prompt="edited prompt", interval_seconds=600)

    assert watchdog.due_prompt() is None
    persisted = load_heartbeat("hb-stale-fire-sid")
    assert (persisted.prompt, persisted.interval_seconds, persisted.fire_count) == ("edited prompt", 600, 0)


def test_a_stale_cached_instance_reports_and_resumes_the_newer_state():
    HeartbeatManager("hb-stale-view-sid").set("original prompt", 60)
    worker = HeartbeatManager("hb-stale-view-sid")

    HeartbeatManager("hb-stale-view-sid").update(prompt="edited prompt", interval_seconds=600)
    assert "edited prompt" in worker.status_line() and "every 10m" in worker.status_line()

    stale = HeartbeatManager("hb-stale-view-sid")  # cached view: still active
    HeartbeatManager("hb-stale-view-sid").pause()
    assert not stale.is_active()
    HeartbeatManager("hb-stale-view-sid").resume()
    assert stale.is_active()


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
