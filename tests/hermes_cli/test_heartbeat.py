"""Tests for /heartbeat (hermes_cli/heartbeat.py)."""

import json
import time

import pytest

from hermes_cli.heartbeat import (
    HeartbeatManager,
    HeartbeatState,
    MIN_INTERVAL_SECONDS,
    format_interval,
    list_active_heartbeats,
    load_heartbeat,
    migrate_heartbeat_to_session,
    parse_interval,
    save_heartbeat,
)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


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
    assert "Heartbeat" in rendered


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


# ──────────────────────────────────────────────────────────────────────
# restart recovery — list_active_heartbeats
# ──────────────────────────────────────────────────────────────────────


def test_state_roundtrip_has_no_route_field():
    """Routing lives in ``SessionEntry.origin`` (gateway/run.py), not on
    ``HeartbeatState`` — this used to be a heartbeat-local snapshot that
    dropped profile/scope qualifiers and left pre-existing rows unable to
    recover after upgrading. See gateway/run.py::_gateway_session_origin_for_id.
    """
    s = HeartbeatState(prompt="check CI", interval_seconds=600)
    assert not hasattr(s, "route")
    loaded = HeartbeatState.from_json(s.to_json())
    assert not hasattr(loaded, "route")


def test_state_roundtrip_ignores_legacy_route_key():
    """A row persisted by a prior build of this fix carried a ``route`` key
    in its JSON; loading it must not choke, and the field is simply dropped
    since routing is now resolved from SessionEntry.origin instead.
    """
    raw = json.dumps(
        {
            "prompt": "check CI",
            "interval_seconds": 600,
            "status": "active",
            "route": {"platform": "telegram", "chat_id": "999"},
        }
    )
    loaded = HeartbeatState.from_json(raw)
    assert loaded.prompt == "check CI"
    assert not hasattr(loaded, "route")


def test_list_active_heartbeats(hermes_home):
    HeartbeatManager(session_id="hb-a").set("task a", 600)
    mgr_b = HeartbeatManager(session_id="hb-b")
    mgr_b.set("task b", 600)
    mgr_b.pause()

    active = dict(list_active_heartbeats())
    assert "hb-a" in active
    assert "hb-b" not in active


def test_list_active_heartbeats_reports_a_legacy_row_with_no_route(hermes_home):
    """A heartbeat set before this fix landed (or with no gateway routing at
    all) must still show up as active — the gateway resolves its routing
    separately, from SessionEntry.origin keyed by session_id, so there is
    nothing route-shaped this list needs to carry.
    """
    HeartbeatManager(session_id="hb-legacy-sid").set("watch deploy", 600)

    active = dict(list_active_heartbeats())
    assert "hb-legacy-sid" in active
