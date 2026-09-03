"""Tests for the Linear -> Kanban bridge.

Everything runs against injected issue fixtures and a temp kanban home — no
network, no live board. The load-bearing assertions:

  * reference-label classification is explicit and fail-loud (no label skips;
    unknown/conflicting agent labels are UNROUTABLE every tick);
  * dry-run dedup on Linear issue id across ticks without writing cards;
  * dry_run=false creates Kanban cards with DB-level Linear UUID idempotency;
  * dry-run seen entries do not suppress a later live create;
  * live creation is capped to a safe default of one and invalid caps fail closed;
  * explicit issue-id allowlists are UUID-only filters and cannot bypass the cap;
  * routing labels can be restricted by an explicit allowed profile list;
  * duplicate DB idempotency hits are reported separately from new creations;
  * key resolution: process env, then ~/.hermes/.env, else fail loud.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gateway import linear_bridge as lb
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "ghost").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


BCFG = {
    "enabled": True,
    "dry_run": True,
    "poll_interval_seconds": 300,
    "team_keys": ["BUI"],
    "status_types": ["unstarted"],
    "api_key_env": "LINEAR_API_KEY",
    "routing_label_prefix": "agent:",
}


def _uuid_for_ident(ident):
    numeric = int("".join(ch for ch in str(ident) if ch.isdigit()) or "0")
    return f"00000000-0000-4000-8000-{numeric:012d}"


def _issue(ident, title, labels=None, state_type="unstarted", iid=None):
    return {
        "id": iid or _uuid_for_ident(ident),
        "identifier": ident,
        "title": title,
        "url": f"https://linear.app/x/issue/{ident}",
        "priority": 2,
        "state": {"name": "Todo", "type": state_type},
        "labels": {"nodes": [{"name": name} for name in (labels or [])]},
        "team": {"key": "BUI"},
    }


def test_classify_routing_labels(kanban_home):
    assert lb.classify_linear_labels(["agent:ghost"], BCFG) == (
        "mapped", "ghost", ["agent:ghost"]
    )
    # Matching is case-insensitive; non-routing labels are ignored.
    assert lb.classify_linear_labels(
        ["Bug", "Agent:Ghost"], BCFG
    ) == ("mapped", "ghost", ["Agent:Ghost"])
    # No routing reference means skip, not error.
    assert lb.classify_linear_labels([], BCFG) == ("unlabeled", None, [])
    assert lb.classify_linear_labels(["Bug"], BCFG) == ("unlabeled", None, [])
    # Unknown profile and conflicting references stay loudly unroutable.
    assert lb.classify_linear_labels(
        ["agent:no-such-agent"], BCFG
    )[0] == "unroutable"
    # A known pull lane is not a Hermes profile and is invalid for agent labels.
    assert lb.classify_linear_labels(["agent:fable"], BCFG)[0] == "unroutable"
    assert lb.classify_linear_labels(["agent:"], BCFG)[0] == "unroutable"
    assert lb.classify_linear_labels(["agentish:ghost"], BCFG) == (
        "unlabeled", None, []
    )
    assert lb.classify_linear_labels(
        ["agent:ghost", "agent:patch"], BCFG
    )[0] == "unroutable"


def test_classify_routing_labels_enforces_allowed_profiles(kanban_home):
    (kanban_home / "profiles" / "patch").mkdir(parents=True)
    cfg = dict(BCFG, allowed_profiles=["patch"])

    assert lb.classify_linear_labels(["agent:patch"], cfg) == (
        "mapped", "patch", ["agent:patch"]
    )
    assert lb.classify_linear_labels(["agent:ghost"], cfg)[0] == "unroutable"


def test_tick_buckets_and_dedup(kanban_home):
    issues = [
        _issue("BUI-1", "bridge me", ["agent:ghost"]),
        _issue("BUI-2", "no routing label"),
        _issue("BUI-3", "ordinary label only", ["Bug"]),
        _issue("BUI-4", "unknown agent", ["agent:no-such-agent"]),
        _issue(
            "BUI-5", "already started", ["agent:ghost"],
            state_type="started",
        ),
        _issue("BUI-6", "conflicting routes", ["agent:ghost", "agent:patch"]),
        _issue("BUI-7", "empty route", ["agent:"]),
    ]
    r1 = lb.run_bridge_tick(BCFG, issues=issues, now=int(time.time()))
    assert r1["ok"] is True
    assert [c["identifier"] for c in r1["would_create"]] == ["BUI-1"]
    assert r1["would_create"][0]["hermes_assignee"] == "ghost"
    assert r1["would_create"][0]["routing_label"] == "agent:ghost"
    assert r1["would_create"][0]["planned_idempotency_key"] == (
        "linear:00000000-0000-4000-8000-000000000001"
    )
    assert [u["identifier"] for u in r1["unroutable"]] == ["BUI-4", "BUI-6", "BUI-7"]
    assert r1["skipped_unlabeled"] == 2
    assert r1["skipped_status"] == 1
    assert r1["already_seen"] == 0

    # Second tick with the same issues: BUI-1 deduped via the seen-store.
    r2 = lb.run_bridge_tick(BCFG, issues=issues, now=int(time.time()))
    assert r2["would_create"] == []
    assert r2["already_seen"] == 1
    # Unroutable stays loud every tick until fixed — it must not "dedup away".
    assert [u["identifier"] for u in r2["unroutable"]] == ["BUI-4", "BUI-6", "BUI-7"]


def test_linear_query_reads_labels_not_assignee():
    assert "labels { nodes { name } }" in lb._ISSUES_QUERY
    assert "assignee" not in lb._ISSUES_QUERY.casefold()


def test_dedup_uses_linear_issue_id(kanban_home):
    stable_uuid = "11111111-1111-4111-8111-111111111111"
    first = _issue(
        "BUI-10", "first title", ["agent:ghost"], iid=stable_uuid
    )
    renamed = _issue(
        "BUI-999", "renamed issue", ["agent:ghost"], iid=stable_uuid
    )
    r1 = lb.run_bridge_tick(BCFG, issues=[first], now=1)
    assert [card["identifier"] for card in r1["would_create"]] == ["BUI-10"]
    r2 = lb.run_bridge_tick(BCFG, issues=[renamed], now=2)
    assert r2["would_create"] == []
    assert r2["already_seen"] == 1


def _all_tasks():
    conn = kb.connect(board="default")
    try:
        return conn.execute(
            "SELECT id, title, body, assignee, status, idempotency_key "
            "FROM tasks ORDER BY created_at, id"
        ).fetchall()
    finally:
        conn.close()


def test_dry_run_creates_no_kanban_cards(kanban_home):
    before = len(_all_tasks())

    report = lb.run_bridge_tick(
        BCFG,
        issues=[_issue("BUI-9", "would-be card", ["agent:ghost"])],
    )
    assert report["would_create"][0]["hermes_assignee"] == "ghost"

    after = len(_all_tasks())
    assert before == after == 0, "dry-run must not create kanban cards"


def test_disabled_bridge_tick_does_nothing(kanban_home):
    cfg = dict(BCFG, enabled=False, dry_run=False)

    report = lb.run_bridge_tick(
        cfg,
        issues=[_issue("BUI-8", "disabled must not create", ["agent:ghost"])],
    )

    assert report["ok"] is True
    assert report["disabled"] is True
    assert report["created"] == []
    assert report["would_create"] == []
    assert len(_all_tasks()) == 0


def test_key_resolution_order(kanban_home, monkeypatch, tmp_path):
    # 1) process env wins
    monkeypatch.setenv("LINEAR_API_KEY", "from-process-env")
    assert lb.resolve_linear_api_key(BCFG) == ("from-process-env", "env")
    # 2) falls back to the shared ~/.hermes/.env (any lock-winning gateway)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_env", lambda: {"LINEAR_API_KEY": "from-hermes-env"}
    )
    assert lb.resolve_linear_api_key(BCFG) == ("from-hermes-env", "hermes-env")
    # 3) missing everywhere -> fail loud at the tick
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: {})
    assert lb.resolve_linear_api_key(BCFG) == (None, "missing")
    report = lb.run_bridge_tick(BCFG)  # no injected issues -> needs a key
    assert report["ok"] is False
    assert "key missing" in report["error"]


def test_non_dry_run_creates_kanban_card_once(kanban_home):
    cfg = dict(BCFG, dry_run=False)
    issue = _issue("BUI-1", "ship live bridge", ["agent:ghost"])

    report = lb.run_bridge_tick(cfg, issues=[issue], now=11)

    assert report["ok"] is True
    assert report["would_create"] == []
    assert report["duplicates"] == []
    assert report["created"] == [
        {
            "identifier": "BUI-1",
            "linear_issue_id": "00000000-0000-4000-8000-000000000001",
            "kanban_task_id": report["created"][0]["kanban_task_id"],
            "title": "ship live bridge",
            "hermes_assignee": "ghost",
            "routing_label": "agent:ghost",
            "idempotency_key": "linear:00000000-0000-4000-8000-000000000001",
        }
    ]

    tasks = _all_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task["title"] == "ship live bridge"
    assert task["assignee"] == "ghost"
    assert task["status"] == "ready"
    assert task["idempotency_key"] == (
        "linear:00000000-0000-4000-8000-000000000001"
    )
    assert "Linear: BUI-1" in task["body"]
    assert "Linear UUID: 00000000-0000-4000-8000-000000000001" in task["body"]
    assert "https://linear.app/x/issue/BUI-1" in task["body"]

    second = lb.run_bridge_tick(cfg, issues=[issue], now=12)
    assert second["created"] == []
    assert second["duplicates"] == []
    assert second["already_seen"] == 1
    assert len(_all_tasks()) == 1

    # Even if the JSON seen-store is lost, the Kanban DB idempotency key is
    # the final duplicate-creation guard, and the report must not inflate the
    # new-created count.
    lb.save_seen({})
    third = lb.run_bridge_tick(cfg, issues=[issue], now=13)
    assert third["created"] == []
    assert third["duplicates"] == [
        {
            "identifier": "BUI-1",
            "linear_issue_id": "00000000-0000-4000-8000-000000000001",
            "kanban_task_id": task["id"],
            "title": "ship live bridge",
            "hermes_assignee": "ghost",
            "routing_label": "agent:ghost",
            "idempotency_key": "linear:00000000-0000-4000-8000-000000000001",
        }
    ]
    assert len(_all_tasks()) == 1


def test_renamed_identifier_same_uuid_lost_seen_reports_duplicate(kanban_home):
    stable_uuid = "22222222-2222-4222-8222-222222222222"
    cfg = dict(BCFG, dry_run=False)
    first = _issue("BUI-10", "first title", ["agent:ghost"], iid=stable_uuid)
    renamed = _issue("BUI-999", "renamed issue", ["agent:ghost"], iid=stable_uuid)

    created = lb.run_bridge_tick(cfg, issues=[first], now=14)
    task_id = created["created"][0]["kanban_task_id"]
    lb.save_seen({})
    duplicate = lb.run_bridge_tick(cfg, issues=[renamed], now=15)

    assert [card["identifier"] for card in created["created"]] == ["BUI-10"]
    assert duplicate["created"] == []
    assert duplicate["duplicates"] == [
        {
            "identifier": "BUI-999",
            "linear_issue_id": stable_uuid,
            "kanban_task_id": task_id,
            "title": "renamed issue",
            "hermes_assignee": "ghost",
            "routing_label": "agent:ghost",
            "idempotency_key": f"linear:{stable_uuid}",
        }
    ]
    assert len(_all_tasks()) == 1


def test_missing_linear_uuid_no_card_no_seen_and_repeats(kanban_home):
    cfg = dict(BCFG, dry_run=False)
    issue = _issue("BUI-11", "missing uuid", ["agent:ghost"])
    issue["id"] = ""

    first = lb.run_bridge_tick(cfg, issues=[issue], now=16)
    second = lb.run_bridge_tick(cfg, issues=[issue], now=17)

    for report in (first, second):
        assert report["ok"] is False
        assert report["created"] == []
        assert report["duplicates"] == []
        assert report["invalid_issue_ids"][0]["identifier"] == "BUI-11"
        assert "Linear issue UUID" in report["error"]
    assert lb.load_seen() == {}
    assert len(_all_tasks()) == 0


def test_live_create_accepts_numeric_integer_string_cap(kanban_home):
    cfg = dict(BCFG, dry_run=False, max_creates_per_tick="2")
    issues = [
        _issue("BUI-12", "first", ["agent:ghost"]),
        _issue("BUI-13", "second", ["agent:ghost"]),
        _issue("BUI-14", "third", ["agent:ghost"]),
    ]

    report = lb.run_bridge_tick(cfg, issues=issues, now=18)

    assert report["ok"] is True
    assert [card["identifier"] for card in report["created"]] == ["BUI-12", "BUI-13"]
    assert report["skipped_cap"] == 1
    assert len(_all_tasks()) == 2


def test_invalid_issue_id_allowlist_fails_closed(kanban_home):
    cfg = dict(BCFG, dry_run=False, issue_id_allowlist=["BUI-12"])

    report = lb.run_bridge_tick(
        cfg,
        issues=[_issue("BUI-12", "must not create", ["agent:ghost"])],
        now=19,
    )

    assert report["ok"] is False
    assert "issue_id_allowlist" in report["error"]
    assert report["created"] == []
    assert len(_all_tasks()) == 0


def test_live_bridge_card_dispatch_once_and_safe_fake_completion(kanban_home):
    cfg = dict(BCFG, dry_run=False)
    report = lb.run_bridge_tick(
        cfg,
        issues=[_issue("BUI-15", "dispatch me", ["agent:ghost"])],
        now=20,
    )
    task_id = report["created"][0]["kanban_task_id"]

    spawned = []

    def fake_spawn(task, workspace):
        spawned.append((task.id, task.assignee, workspace))
        return 12345

    conn = kb.connect(board="default")
    try:
        dispatch = kb.dispatch_once(conn, spawn_fn=fake_spawn, board="default")
        assert dispatch.spawned == [(task_id, "ghost", spawned[0][2])]
        assert spawned[0][0] == task_id

        active_run = conn.execute(
            "SELECT id, profile, status, worker_pid, ended_at "
            "FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert active_run["profile"] == "ghost"
        assert active_run["status"] == "running"
        assert active_run["worker_pid"] == 12345
        assert active_run["ended_at"] is None
        run_id = active_run["id"]

        event_kinds = [event.kind for event in kb.list_events(conn, task_id)]
        assert event_kinds[0] == "created"
        assert "claimed" in event_kinds
        assert "spawned" in event_kinds
        assert event_kinds.index("claimed") < event_kinds.index("spawned")

        assert kb.complete_task(conn, task_id, result="safe fake completion")
        assert kb.get_task(conn, task_id).status == "done"

        completed_run = conn.execute(
            "SELECT status, outcome, summary, ended_at "
            "FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert completed_run["status"] == "done"
        assert completed_run["outcome"] == "completed"
        assert completed_run["summary"] == "safe fake completion"
        assert completed_run["ended_at"] is not None

        events = kb.list_events(conn, task_id)
        completed_event_kinds = [event.kind for event in events]
        assert completed_event_kinds[0] == "created"
        assert completed_event_kinds[-1] == "completed"
        assert completed_event_kinds.index("claimed") < completed_event_kinds.index("spawned")
        assert completed_event_kinds.index("spawned") < completed_event_kinds.index("completed")
        run_events = [
            event for event in events
            if event.kind in {"claimed", "spawned", "completed"}
        ]
        assert all(event.run_id == run_id for event in run_events)
    finally:
        conn.close()



def test_dry_run_seen_entry_does_not_suppress_later_live_create(kanban_home):
    issue = _issue("BUI-2", "dry then live", ["agent:ghost"])
    dry = lb.run_bridge_tick(BCFG, issues=[issue], now=21)
    assert [card["identifier"] for card in dry["would_create"]] == ["BUI-2"]
    assert len(_all_tasks()) == 0

    live = lb.run_bridge_tick(dict(BCFG, dry_run=False), issues=[issue], now=22)
    assert [card["identifier"] for card in live["created"]] == ["BUI-2"]
    assert live["already_seen"] == 0
    assert len(_all_tasks()) == 1


def test_live_create_respects_max_create_cap(kanban_home):
    cfg = dict(BCFG, dry_run=False, max_creates_per_tick=2)
    issues = [
        _issue("BUI-20", "first", ["agent:ghost"]),
        _issue("BUI-21", "second", ["agent:ghost"]),
        _issue("BUI-22", "third", ["agent:ghost"]),
    ]

    report = lb.run_bridge_tick(cfg, issues=issues, now=31)

    assert report["ok"] is True
    assert [card["identifier"] for card in report["created"]] == ["BUI-20", "BUI-21"]
    assert report["skipped_cap"] == 1
    assert len(_all_tasks()) == 2


def test_live_create_defaults_to_one_when_cap_missing(kanban_home):
    cfg = dict(BCFG, dry_run=False)
    issues = [
        _issue("BUI-40", "first", ["agent:ghost"]),
        _issue("BUI-41", "second", ["agent:ghost"]),
    ]

    report = lb.run_bridge_tick(cfg, issues=issues, now=51)

    assert report["ok"] is True
    assert [card["identifier"] for card in report["created"]] == ["BUI-40"]
    assert report["skipped_cap"] == 1
    assert len(_all_tasks()) == 1


def test_live_create_explicit_cap_one_limits_many_eligible_issues(kanban_home):
    cfg = dict(BCFG, dry_run=False, max_creates_per_tick=1)
    issues = [
        _issue("BUI-44", "first", ["agent:ghost"]),
        _issue("BUI-45", "second", ["agent:ghost"]),
        _issue("BUI-46", "third", ["agent:ghost"]),
    ]

    report = lb.run_bridge_tick(cfg, issues=issues, now=55)

    assert report["ok"] is True
    assert [card["identifier"] for card in report["created"]] == ["BUI-44"]
    assert report["skipped_cap"] == 2
    assert len(_all_tasks()) == 1


def test_live_create_issue_uuid_allowlist_filters_without_identifier_alias(kanban_home):
    cfg = dict(
        BCFG,
        dry_run=False,
        max_creates_per_tick=2,
        issue_id_allowlist=[_uuid_for_ident("BUI-51")],
    )
    issues = [
        _issue("BUI-50", "identifier alias must not match", ["agent:ghost"]),
        _issue("BUI-51", "allowed by uuid", ["agent:ghost"]),
    ]

    report = lb.run_bridge_tick(cfg, issues=issues, now=61)

    assert report["ok"] is True
    assert [card["identifier"] for card in report["created"]] == ["BUI-51"]
    assert report["skipped_allowlist"] == 1
    assert report["skipped_cap"] == 0
    assert len(_all_tasks()) == 1


def test_issue_uuid_allowlist_cannot_bypass_cap(kanban_home):
    cfg = dict(
        BCFG,
        dry_run=False,
        max_creates_per_tick=1,
        issue_id_allowlist=[
            _uuid_for_ident("BUI-70"),
            _uuid_for_ident("BUI-71"),
            _uuid_for_ident("BUI-72"),
        ],
    )
    issues = [
        _issue("BUI-70", "first", ["agent:ghost"]),
        _issue("BUI-71", "second", ["agent:ghost"]),
        _issue("BUI-72", "third", ["agent:ghost"]),
    ]

    report = lb.run_bridge_tick(cfg, issues=issues, now=71)

    assert report["ok"] is True
    assert [card["identifier"] for card in report["created"]] == ["BUI-70"]
    assert report["skipped_cap"] == 2
    assert len(_all_tasks()) == 1



@pytest.mark.parametrize(
    "bad_cap",
    [0, -1, True, False, 1.5, "not-an-int", "1.0", None],
)
def test_live_create_invalid_max_create_cap_fails_closed(kanban_home, bad_cap):
    cfg = dict(BCFG, dry_run=False, max_creates_per_tick=bad_cap)

    report = lb.run_bridge_tick(
        cfg,
        issues=[_issue("BUI-30", "must not create", ["agent:ghost"])],
        now=41,
    )

    assert report["ok"] is False
    assert report["created"] == []
    assert report["duplicates"] == []
    assert "max_creates_per_tick" in report["error"]
    assert len(_all_tasks()) == 0
