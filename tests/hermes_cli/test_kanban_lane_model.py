"""Lane-model overrides: time-boxed, board-level model routing at dispatch.

Covers the store (TTL boundary, lane precedence, atomic clear/expire), the
dispatcher (card pin > assignee lane > board-wide lane > profile default on
BOTH the ready and review lanes, expiry reported exactly once, never
persisted onto the card) and the CLI surface (set/show/clear, stats).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_lanes as kbl


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    for prof in ("alpha", "beta", "default"):
        (home / "profiles" / prof).mkdir(parents=True)
        (home / "profiles" / prof / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class _Spawns:
    """spawn_fn stub recording the route each spawned Task carried."""

    def __init__(self):
        self.routes: dict[str, tuple] = {}

    def __call__(self, task, workspace, board=None):
        self.routes[task.id] = (task.provider_override, task.model_override, task.reasoning_effort)
        return None


def _tick(conn, **kw):
    spawns = _Spawns()
    res = kbd.dispatch_once(conn, spawn_fn=spawns, **kw)
    return res, spawns


# --- store -----------------------------------------------------------------


def test_ttl_boundary_belongs_to_profile_default(kanban_home):
    with kbc.connect_closing() as conn:
        kbl.set_lane_model_override(conn, provider="p", model="m", expires_at=1100, now=1000)
        assert kbl.get_lane_model_override(conn, now=1099) is not None
        # Exclusive expiry: at expires_at the lane is already gone, even
        # though the row is still in the table until the sweep deletes it.
        assert kbl.get_lane_model_override(conn, now=1100) is None
        assert conn.execute("SELECT COUNT(*) FROM lane_model_overrides").fetchone()[0] == 1


def test_assignee_lane_beats_board_wide_lane(kanban_home):
    now = int(time.time())
    with kbc.connect_closing() as conn:
        kbl.set_lane_model_override(conn, provider="wide", model="w", expires_at=now + 60)
        kbl.set_lane_model_override(conn, provider="own", model="o", expires_at=now + 60, assignee="alpha")
        assert kbl.get_lane_model_override(conn, assignee="alpha").provider == "own"
        assert kbl.get_lane_model_override(conn, assignee="beta").provider == "wide"


def test_reset_is_an_upsert_not_a_stack(kanban_home):
    now = int(time.time())
    with kbc.connect_closing() as conn:
        kbl.set_lane_model_override(conn, provider="p", model="a", expires_at=now + 60)
        kbl.set_lane_model_override(conn, provider="p", model="b", expires_at=now + 120)
        rows = kbl.list_lane_model_overrides(conn)
    assert [(r.model, r.expires_at) for r in rows] == [("b", now + 120)]


@pytest.mark.parametrize("provider,model,expires_delta", [("", "m", 60), ("p", "", 60), ("p", "m", 0)])
def test_set_refuses_incomplete_or_already_expired_rows(kanban_home, provider, model, expires_delta):
    with kbc.connect_closing() as conn, pytest.raises(ValueError):
        kbl.set_lane_model_override(
            conn, provider=provider, model=model, expires_at=1000 + expires_delta, now=1000,
        )


def test_expire_deletes_and_reports_each_row_once(kanban_home):
    with kbc.connect_closing() as conn:
        kbl.set_lane_model_override(conn, provider="p", model="gone", expires_at=1100, now=1000)
        kbl.set_lane_model_override(conn, provider="p", model="live", expires_at=5000, now=1000, assignee="alpha")
        first = kbl.expire_lane_model_overrides(conn, now=1200)
        second = kbl.expire_lane_model_overrides(conn, now=1200)
        remaining = kbl.list_lane_model_overrides(conn, now=1200)
    assert [r.model for r in first] == ["gone"]
    assert second == []
    assert [r.model for r in remaining] == ["live"]


def test_successor_label_names_the_surviving_lane(kanban_home):
    assert kbl.lane_successor_label(None) == "profile default"
    wide = kbl.LaneModelOverride(assignee=None, provider="p", model="m")
    assert kbl.lane_successor_label(wide) == "board-wide lane p/m"


# --- dispatcher ------------------------------------------------------------


def test_lane_routes_unpinned_cards_and_card_pin_wins(kanban_home):
    now = int(time.time())
    with kbc.connect_closing() as conn:
        free = kb.create_task(conn, title="free", assignee="alpha")
        pinned = kb.create_task(conn, title="pinned", assignee="alpha")
        kb.set_model_override(conn, pinned, "card-model", provider="card-prov")
        kbl.set_lane_model_override(
            conn, provider="lane-prov", model="lane-model", expires_at=now + 600,
            reasoning_effort="low",
        )
        res, spawns = _tick(conn)
        stored = kb.get_task(conn, free)
    assert spawns.routes[free] == ("lane-prov", "lane-model", "low")
    assert spawns.routes[pinned] == ("card-prov", "card-model", None)
    assert res.spawn_route_sources[free].startswith("lane-override(")
    assert res.spawn_route_sources[pinned] == "card-override"
    assert res.spawn_routes[free] == "lane-prov/lane-model"
    # The lane route is applied in memory only: persisting it would turn a
    # time-boxed window into a permanent pin.
    assert stored.model_override is None and stored.provider_override is None


def test_card_effort_beats_lane_effort(kanban_home):
    now = int(time.time())
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="alpha")
        kb.set_reasoning_effort(conn, tid, "high")
        kbl.set_lane_model_override(
            conn, provider="p", model="m", expires_at=now + 600, reasoning_effort="low",
        )
        _, spawns = _tick(conn)
    assert spawns.routes[tid] == ("p", "m", "high")


def test_assignee_lane_only_routes_its_own_profile(kanban_home):
    now = int(time.time())
    with kbc.connect_closing() as conn:
        a = kb.create_task(conn, title="a", assignee="alpha")
        b = kb.create_task(conn, title="b", assignee="beta")
        kbl.set_lane_model_override(conn, provider="p", model="m", expires_at=now + 600, assignee="alpha")
        res, spawns = _tick(conn)
    assert spawns.routes[a][:2] == ("p", "m")
    assert spawns.routes[b][:2] == (None, None)
    assert res.spawn_route_sources[b] == "profile-default"
    assert b not in res.spawn_routes


def test_review_lane_spawns_are_routed_too(kanban_home):
    """A window that re-routes workers but strands reviewers on the old
    provider would block the lane that unblocks everything else."""
    now = int(time.time())
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="to review", assignee="alpha")
        assert kb.request_review(conn, tid, reviewer="beta")
        assert kb.get_task(conn, tid).status == "review"
        kbl.set_lane_model_override(conn, provider="p", model="m", expires_at=now + 600)
        res, spawns = _tick(conn)
    assert spawns.routes[tid][:2] == ("p", "m")
    assert res.spawn_route_sources[tid].startswith("lane-override(")


def test_expired_lane_is_retired_once_and_names_its_successor(kanban_home):
    now = int(time.time())
    with kbc.connect_closing() as conn:
        kbl.set_lane_model_override(conn, provider="old", model="m", expires_at=now + 600, assignee="alpha")
        kbl.set_lane_model_override(conn, provider="wide", model="w", expires_at=now + 600)
        conn.execute("UPDATE lane_model_overrides SET expires_at = ? WHERE assignee = 'alpha'", (now - 1,))
        conn.commit()
        tid = kb.create_task(conn, title="t", assignee="alpha")
        first, spawns = _tick(conn)
        second, _ = _tick(conn)
    assert first.expired_lane_models == [("alpha", "old/m")]
    assert first.expired_lane_successors == {"alpha": "board-wide lane wide/w"}
    assert second.expired_lane_models == []
    # The card fell through to the still-active board-wide lane.
    assert spawns.routes[tid][:2] == ("wide", "w")


def test_dry_run_never_retires_a_lane(kanban_home):
    now = int(time.time())
    with kbc.connect_closing() as conn:
        kbl.set_lane_model_override(conn, provider="p", model="m", expires_at=now + 600)
        conn.execute("UPDATE lane_model_overrides SET expires_at = ?", (now - 1,))
        conn.commit()
        res = kbd.dispatch_once(conn, dry_run=True)
        left = conn.execute("SELECT COUNT(*) FROM lane_model_overrides").fetchone()[0]
    assert res.expired_lane_models == []
    assert left == 1


# --- CLI -------------------------------------------------------------------


def test_cli_set_show_clear_round_trip(kanban_home):
    out = kc.run_slash('lane-model set openrouter/anthropic/claude-x --ttl 2h --reason "quota" --assignee alpha')
    assert "route=openrouter/anthropic/claude-x" in out and "lane=alpha" in out and "ttl=2h" in out
    shown = json.loads(kc.run_slash("lane-model show --json"))
    assert [(r["assignee"], r["provider"], r["model"], r["reason"]) for r in shown] == [
        ("alpha", "openrouter", "anthropic/claude-x", "quota"),
    ]
    assert 0 < shown[0]["ttl_remaining_seconds"] <= 7200
    stats = kc.run_slash("stats")
    assert "Active lane-model overrides:" in stats and "alpha: route=openrouter/anthropic/claude-x" in stats
    cleared = kc.run_slash("lane-model clear --assignee alpha")
    assert "now routes via profile default" in cleared
    assert json.loads(kc.run_slash("lane-model show --json")) == []


@pytest.mark.parametrize("line", [
    'lane-model set just-a-model --ttl 1h --reason r',
    'lane-model set p/m --ttl 0 --reason r',
    'lane-model set p/m --ttl soon --reason r',
    'lane-model set p/m --ttl 1h --reason "  "',
])
def test_cli_set_refuses_bad_input_without_writing(kanban_home, line):
    out = kc.run_slash(line)
    assert "kanban:" in out
    with kbc.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM lane_model_overrides").fetchone()[0] == 0


def test_cli_ttl_is_required(kanban_home):
    out = kc.run_slash("lane-model set p/m --reason r")
    assert "--ttl" in out
    with kbc.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM lane_model_overrides").fetchone()[0] == 0


def test_cli_clear_assignee_lane_names_board_wide_successor(kanban_home):
    kc.run_slash("lane-model set wide/w --ttl 1h --reason r")
    kc.run_slash("lane-model set own/o --ttl 1h --reason r --assignee alpha")
    out = kc.run_slash("lane-model clear --assignee alpha")
    assert "now routes via board-wide lane wide/w" in out


def test_cli_clear_all(kanban_home):
    kc.run_slash("lane-model set wide/w --ttl 1h --reason r")
    kc.run_slash("lane-model set own/o --ttl 1h --reason r --assignee alpha")
    out = kc.run_slash("lane-model clear --all")
    assert out.count("Cleared lane-model override") == 2
    assert "no lane-model override" in kc.run_slash("lane-model clear --all")
