"""Ready-queue admission — the V-suite of the mechanism spec.

The board under test is a real ``kanban.db`` inside an isolated ``HERMES_HOME``,
and the mechanism is switched on the way an operator does it: by writing the
``kanban:`` config keys. Nothing here monkeypatches the accessor, so the tests
exercise the config path the fleet actually runs.

Spec: ``yaan-platform/docs/READY-QUEUE-ADMISSION-MECHANISM.md`` (§3 mechanism,
§4 A4 surfaces, §5.2 read surface, §7V1-V8/V10).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import pytest
import yaml

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_admission as adm
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_output
from hermes_cli.kanban_parser import build_parser

# The keys §5.2 freezes for `queue-state`/the A7 control.
QUEUE_STATE_KEYS = (
    "depth", "budget", "budget_source", "drain_per_hour", "window_hours", "lanes",
    "deferred_count", "admitted_count", "bypass_count", "pre_mechanism_backlog",
    "oldest_ready_seconds", "ageing_warn_count", "ageing_escalate_count", "ageing_oldest",
)

# §5.1's contract table, copied as a LITERAL from
# yaan-platform/docs/READY-QUEUE-ADMISSION-MECHANISM.md §5.1 (lines 323-337 at
# md5 a9949ea2406aecef2e662ef47ff5461d). It is a literal on purpose and is NOT
# read from the doc at test time — the doc is a different repo. This is the
# second place the contract lives so that a contract edit is a visible two-place
# change: the branch's old parity test compared the module's DEFAULTS against the
# code's own defaults (code agreeing with code), which is how an 11-key drift
# passed 12 tests.
SPEC_5_1_KANBAN_DEFAULTS = {
    "admission_enabled_at": None,      # unset = off; also the bypass-audit baseline
    "admission_budget": None,          # integer pin, overrides the derivation
    "admission_window_hours": 24,      # trailing drain window
    "admission_budget_floor": 5,       # floor for a quiet lane
    "admission_p0_priority": 90,       # priority >= this is a P0 blocking fault
    "admission_lane_budgets": None,    # optional {lane: int} overrides
    "ageing_warn_hours": 24,           # routine tier
    "ageing_escalate_hours": 72,       # escalate tier
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with the kanban env pins stripped."""
    root = tmp_path / ".hermes"
    root.mkdir()
    for var in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_TASK", "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return root


@pytest.fixture
def conn(home):
    with kbc.connect() as c:
        yield c


def enable_admission(home: Path, **kanban) -> None:
    """Write the ``kanban:`` config block an operator would write."""
    cfg = {"admission_enabled_at": int(time.time())}
    cfg.update(kanban)
    (home / "config.yaml").write_text(yaml.safe_dump({"kanban": cfg}), encoding="utf-8")


def _seed_completions(conn, n: int, *, lane: str = "lane-a", age_seconds: int = 3600,
                      tag: str = "a") -> None:
    """Raw-SQL cohort seed: ``n`` completed cards inside the window."""
    ts = int(time.time()) - age_seconds
    for i in range(n):
        tid = f"t_seed{tag}{i:04d}"
        conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, priority, created_by, "
            "created_at, completed_at) VALUES (?, ?, ?, 'done', 0, 'seed', ?, ?)",
            (tid, f"seed {i}", lane, ts, ts),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'completed', NULL, ?)", (tid, ts),
        )
    conn.commit()


def _cli(*argv: str) -> argparse.Namespace:
    top = argparse.ArgumentParser()
    parser = build_parser(top.add_subparsers(dest="_top"))
    return parser.parse_args(list(argv))


def _json_out(capsys) -> dict:
    """Parse the JSON a CLI handler printed (it is pretty-printed)."""
    out = capsys.readouterr().out
    start = out.find("{")
    assert start >= 0, f"no JSON on stdout: {out!r}"
    return json.loads(out[start:])


def _admit_payload(conn, task_id: str) -> dict:
    for event in reversed(kb.list_events(conn, task_id)):
        payload = event.payload or {}
        if isinstance(payload, dict) and "admit" in payload:
            return payload["admit"]
    raise AssertionError(f"no admit snapshot on {task_id}'s events")


# --- V1: over budget, an ordinary create lands todo with a recorded reason ---


def test_v1_over_budget_create_parks_with_the_reason_on_the_event(home, conn):
    enable_admission(home, admission_budget=1)
    first = kb.create_task(conn, title="first", assignee="lane-a")
    assert kb.get_task(conn, first).status == "ready"

    second = kb.create_task(conn, title="second", assignee="lane-a")
    parked = kb.get_task(conn, second)
    assert parked.status == "todo"
    assert parked.admit_state == adm.DEFERRED
    assert parked.ready_since is None

    snapshot = _admit_payload(conn, second)
    assert snapshot["state"] == "deferred"
    assert snapshot["reason"] == adm.REASON_OVER_BUDGET
    assert snapshot["depth"] == 1 and snapshot["budget"] == 1

    # The parked card is visible on the read surface, not just in an event.
    state = adm.queue_state(conn)
    assert state["deferred_count"] == 1 and state["depth"] == 1


# --- V2: the lane budget binds before the global depth check ------------------


def test_v2_lane_budget_binds_before_the_global_depth(home, conn):
    # A5/§5.1: lane-a's bound is its OWN derivation (6 completions in the
    # window), and the board is pinned far higher — so only the LANE bound can
    # refuse this filing. The old surface gave every lane a flat % of the board.
    _seed_completions(conn, 6, lane="lane-a")
    enable_admission(home, admission_budget=100)
    ids = [kb.create_task(conn, title=f"a{i}", assignee="lane-a") for i in range(6)]
    assert all(kb.get_task(conn, i).status == "ready" for i in ids)

    verdict = adm.decide(conn, lane="lane-a")
    assert verdict.admitted is False and verdict.reason == adm.REASON_OVER_BUDGET
    assert verdict.budget == 100 and verdict.depth == 6          # board room
    assert verdict.lane_depth == 6 and verdict.lane_budget == 6  # lane is full

    seventh = kb.create_task(conn, title="a6", assignee="lane-a")
    assert kb.get_task(conn, seventh).status == "todo"
    # A different lane derives its OWN (quiet-board) budget and still has room.
    other = kb.create_task(conn, title="b0", assignee="lane-b")
    assert kb.get_task(conn, other).status == "ready"


# --- V3: the budget comes from the closing cohort, then the pin --------------


def test_v3_budget_derives_from_the_window_then_the_pin(home, conn):
    _seed_completions(conn, 196, age_seconds=3 * 3600)
    budget, source = adm.ready_queue_budget(conn)
    assert budget == 196 and source == adm.SOURCE_COHORT  # §5.1: the cohort, x1

    enable_admission(home, admission_budget=108)
    budget, source = adm.ready_queue_budget(conn)
    assert budget == 108 and source == adm.SOURCE_PIN
    # The pin moves the BOARD budget only; the lane bound stays the lane's own
    # derivation (196 here, not a share of the 108 pin).
    assert adm.lane_budget(conn, "lane-a") == 196

    # A quiet board is not a stopped one: the floor holds the budget up.
    enable_admission(home, admission_window_hours=1)
    _seed_completions(conn, 3, age_seconds=60, tag="b")
    budget, source = adm.ready_queue_budget(conn)
    assert budget == 5 and source == adm.SOURCE_FLOOR


# --- V4: re-entry is exempt (its demand was already admitted) ----------------


def test_v4_re_entry_is_exempt_and_restamps_the_wait_clock(home, conn):
    enable_admission(home, admission_budget=1)
    held = kb.create_task(conn, title="held", assignee="lane-a")     # fills the budget
    blocked = kb.create_task(conn, title="blocked", assignee="lane-a")
    assert kb.get_task(conn, blocked).status == "todo"

    conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (blocked,))
    conn.commit()
    before = int(time.time())
    assert kb.unblock_task(conn, blocked) is True
    relanded = kb.get_task(conn, blocked)
    assert relanded.status == "ready"
    assert relanded.admit_state == adm.ADMITTED
    assert relanded.ready_since is not None and relanded.ready_since >= before
    assert _admit_payload(conn, blocked)["reason"] == adm.REASON_RE_ENTRY
    assert kb.get_task(conn, held).status == "ready"


# --- V5: the recompute guard — a deferral outlives one dispatcher tick -------


def test_v5_deferred_child_waits_for_headroom_and_is_not_re_evented(home, conn):
    # Built while the mechanism is OFF, then switched on with the board full:
    # this is the deployment shape the mechanism was designed around.
    parent = kb.create_task(conn, title="parent", assignee="lane-a")
    filler_a = kb.create_task(conn, title="fa", assignee="lane-a")
    filler_b = kb.create_task(conn, title="fb", assignee="lane-a")
    child = kb.create_task(conn, title="child", parents=[parent], assignee="lane-a")
    assert kb.get_task(conn, child).status == "todo"
    enable_admission(home, admission_budget=1)

    report: dict = {}
    kb.complete_task(conn, parent, summary="parent done", report=report)
    # §3.4: the completing parent is told the child is deferred, not running.
    assert report["promoted"] == 0 and report["deferred"] == 1
    assert report["deferred_ids"] == [child]

    parked = kb.get_task(conn, child)
    assert parked.status == "todo" and parked.admit_state == adm.DEFERRED

    # One event per refusal transition, not one per tick: recompute runs every
    # tick and must not become an event firehose.
    refusals = [e for e in kb.list_events(conn, child) if e.kind == "promoted"]
    assert len(refusals) == 1
    kb.recompute_ready(conn)
    kb.recompute_ready(conn)
    assert len([e for e in kb.list_events(conn, child) if e.kind == "promoted"]) == 1
    assert kb.get_task(conn, child).status == "todo"

    # Drain the board: headroom appears and the same card lands ready.
    kb.complete_task(conn, filler_a, summary="x")
    kb.complete_task(conn, filler_b, summary="y")
    kb.recompute_ready(conn)
    landed = kb.get_task(conn, child)
    assert landed.status == "ready"
    assert landed.admit_state == adm.ADMITTED and landed.ready_since is not None


# --- V6: ageing is a column read, and the clock stops at the claim ------------


def test_v6_ageing_warns_escalates_and_clears_on_claim(home, conn):
    # §5.1: the tiers are HOURS spent in ready (24 / 72), not seconds.
    enable_admission(home, admission_budget=10,
                     ageing_warn_hours=24, ageing_escalate_hours=72)
    warned = kb.create_task(conn, title="slow", assignee="lane-a")
    escalated = kb.create_task(conn, title="slower", assignee="lane-a")
    now = int(time.time())
    conn.execute("UPDATE tasks SET ready_since = ? WHERE id = ?",
                 (now - 25 * 3600, warned))
    conn.execute("UPDATE tasks SET ready_since = ? WHERE id = ?",
                 (now - 80 * 3600, escalated))
    conn.commit()

    state = adm.queue_state(conn)
    assert state["oldest_ready_seconds"] >= 80 * 3600
    assert state["ageing_warn_count"] == 2       # 25 h and 80 h are both past 24 h
    assert state["ageing_escalate_count"] == 1   # only the 80 h card is past 72 h

    assert kb.claim_task(conn, warned, claimer="tester") is not None
    after = adm.queue_state(conn)
    assert after["ageing_warn_count"] == 1 and after["ageing_escalate_count"] == 1
    # The clock stops at the claim: a running card is no longer waiting.
    assert kb.claim_task(conn, escalated, claimer="tester") is not None
    drained = adm.queue_state(conn)
    assert drained["ageing_warn_count"] == 0 and drained["ageing_escalate_count"] == 0


# --- V7: with the mechanism off, nothing changes -----------------------------


def test_v7_off_means_the_previous_behaviour(home, conn):
    # No config.yaml at all: the keys are unset, so admission is off.
    assert adm.admission_enabled_at() == 0
    ids = [kb.create_task(conn, title=f"t{i}", assignee="one-lane") for i in range(6)]
    assert all(kb.get_task(conn, i).status == "ready" for i in ids)
    # ...and the event stream is byte-for-byte what it was before the mechanism.
    for tid in ids:
        created = [e for e in kb.list_events(conn, tid) if e.kind == "created"][0]
        assert "admit" not in (created.payload or {})

    verdict = adm.decide(conn, lane="one-lane")
    assert verdict.admitted is True and verdict.reason == adm.REASON_DISABLED

    # The §5.1 keys exist in the shipped defaults, so `hermes config get kanban.…`
    # answers and an operator can see what they are switching on. The full parity
    # check lives in test_5_1_parity_… below: a loop here over adm.DEFAULTS would
    # only compare the code with itself, which is how an 11-key drift passed.
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["kanban"]["admission_enabled_at"] is None


# --- V8: the pre-mechanism backlog is reported, never mass-addressed ----------


def test_v8_pre_mechanism_backlog_is_reported_not_touched(home, conn):
    legacy = [kb.create_task(conn, title=f"legacy{i}", assignee="lane-a") for i in range(3)]
    # A card that predates the mechanism carries no admission state at all.
    conn.execute("UPDATE tasks SET admit_state = NULL, ready_since = NULL")
    conn.commit()
    enable_admission(home, admission_budget=1)

    state = adm.queue_state(conn)
    assert state["pre_mechanism_backlog"] == 3
    assert state["bypass_count"] == 0
    assert state["deferred_count"] == 0

    kb.recompute_ready(conn)
    for tid in legacy:
        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.admit_state is None
        assert [e.kind for e in kb.list_events(conn, tid)] == ["created"]


# --- V10: no filing is dropped silently (parked or handed back) --------------


def test_v10_no_filing_is_dropped_silently(home, conn, capsys):
    # A full board built before the switch (the deployment's real shape), and an
    # origin card that has already finished -- the follow-up's real origin.
    origin = kb.create_task(conn, title="origin", assignee="lane-a")
    kb.create_task(conn, title="filler", assignee="lane-a")
    kb.complete_task(conn, origin, summary="filed the follow-up from here",
                     fire_lifecycle_hook=False)
    enable_admission(home, admission_budget=1)
    assert adm.queue_state(conn)["depth"] == 1  # at the budget: over it now

    # (a) no origin: the filing parks on its own card and is counted.
    orphan = kb.create_task(conn, title="orphan", assignee="lane-a")
    assert kb.get_task(conn, orphan).status == "todo"
    state = adm.queue_state(conn)
    assert state["deferred_count"] == 1 and state["fallback_count"] == 1

    # (b) an origin is named: the refusal is a comment on that card, and no
    #     orphan card is created at all.
    report: dict = {}
    target = kb.create_task(
        conn, title="follow-up", assignee="lane-a", parents=[origin],
        admit_origin=origin, report=report,
    )
    assert report["disposition"] == adm.DISPOSITION_COMMENT_ON_ORIGIN
    assert report["target"] == origin and target == origin
    bodies = [c.body for c in kb.list_comments(conn, origin)]
    assert any("refused" in b and str(report["admit"]["budget"]) in b for b in bodies)
    assert len(kb.list_comments(conn, origin)) == 1  # exactly one refusal, no orphan card

    # The CLI surface says the same thing, out loud, in --json.
    assert kanban_cli._cmd_create(_cli(
        "create", "second follow-up", "--assignee", "lane-a",
        "--parent", origin, "--json",
    )) == 0
    payload = _json_out(capsys)
    assert payload["deferred"] is True
    assert payload["disposition"] == adm.DISPOSITION_COMMENT_ON_ORIGIN
    assert payload["target"] == origin
    assert "task_id" not in payload

    # And the parked filing is reported as parked, not as created-and-running.
    assert kanban_cli._cmd_create(_cli(
        "create", "parked", "--assignee", "lane-a", "--json",
    )) == 0
    parked = _json_out(capsys)
    assert parked["status"] == "todo"
    assert parked["disposition"] == adm.DISPOSITION_PARKED
    assert parked["id"] and parked["id"] != origin


# --- A4: the dedup key is honest and readable --------------------------------


def test_a4_dedupe_is_reported_by_both_surfaces_and_the_key_is_readable(home, conn, capsys):
    from tools import kanban_tools

    args = _cli("create", "dedup me", "--assignee", "lane-a",
                "--idempotency-key", "spec-3.1/A4", "--json")
    assert kanban_cli._cmd_create(args) == 0
    first = _json_out(capsys)
    assert first["deduped"] is False
    assert first["idempotency_key"] == "spec-3.1/A4"

    assert kanban_cli._cmd_create(args) == 0
    second = _json_out(capsys)
    assert second["deduped"] is True and second["id"] == first["id"]
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = 'spec-3.1/A4'"
    ).fetchone()["n"] == 1

    # The tool surface answers the same question the same way.
    tool_first = json.loads(kanban_tools._handle_create(
        {"title": "dedup me too", "assignee": "lane-a", "idempotency_key": "spec-3.1/A4b"}))
    tool_second = json.loads(kanban_tools._handle_create(
        {"title": "dedup me too", "assignee": "lane-a", "idempotency_key": "spec-3.1/A4b"}))
    assert tool_first["deduped"] is False and tool_second["deduped"] is True
    assert tool_first["task_id"] == tool_second["task_id"]

    # A key you cannot see is not a key you can trust.
    detail = kanban_output._task_to_dict(kb.get_task(conn, first["id"]))
    assert detail["idempotency_key"] == "spec-3.1/A4"
    assert "admit_state" in detail and "ready_since" in detail


# --- The read surface the A7 control consumes --------------------------------


def test_queue_state_exposes_the_frozen_keys(home, conn, capsys):
    enable_admission(home, admission_budget=3)
    kb.create_task(conn, title="r", assignee="lane-a")
    assert kanban_cli._cmd_queue_state(_cli("queue-state", "--json")) == 0
    state = _json_out(capsys)
    for key in QUEUE_STATE_KEYS:
        assert key in state, key
    assert state["budget"] == 3 and state["budget_source"] == adm.SOURCE_PIN
    assert state["lanes"]["lane-a"]["depth"] == 1
    assert state["window_hours"] == 24


# --- §5.1: the shipped config surface IS the contract ------------------------


def test_5_1_parity_the_shipped_kanban_defaults_are_the_contract_table():
    """§5.1 is the contract; the code is conformed to it.

    Compares the SHIPPED defaults (and the kernel's own copy) against the §5.1
    table above — equality, not containment, in both directions: a key the
    contract does not carry (the retired ``admission_cohort_multiplier``, the
    ``admission_lane_budget_pct`` family), a renamed key, a moved value, or a
    missing key all fail here.
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    shipped = {
        key: value
        for key, value in DEFAULT_CONFIG["kanban"].items()
        if key.startswith("admission_") or key.startswith("ageing_")
    }
    assert shipped == SPEC_5_1_KANBAN_DEFAULTS
    # The kernel keeps its own copy so a board whose config.yaml predates the
    # keys is still governed by §5.1 — and it is the same table.
    assert adm.DEFAULTS == SPEC_5_1_KANBAN_DEFAULTS

    # And the readers actually answer with those values on a bare home.
    assert adm.admission_enabled_at() == 0          # unset = OFF
    assert adm.lane_budgets() == {}                 # unset = no per-lane override
    assert adm.is_p0_fault(89) is False and adm.is_p0_fault(90) is True


# --- §5.1: the lane override map and the unassigned case ----------------------


def test_lane_overrides_and_the_unassigned_demand_has_no_lane_bound(home, conn):
    """A5's lane bound is the lane's own derivation; the map is the ONE override."""
    _seed_completions(conn, 6, lane="lane-a")
    enable_admission(home, admission_budget=50,
                     admission_lane_budgets={"lane-a": 2, "lane-b": "3", "lane-d": -1})

    # An explicit entry wins, and is coerced rather than trusted as YAML handed it
    # over; a negative entry clamps to 0, which reads as "no lane bound".
    assert adm.lane_budgets() == {"lane-a": 2, "lane-b": 3, "lane-d": 0}
    assert adm.lane_budget(conn, "lane-a") == 2
    assert adm.lane_budget(conn, "lane-b") == 3
    assert adm.lane_budget(conn, "lane-d") == 0
    # A lane with no entry keeps its OWN derivation (floor on a quiet lane).
    assert adm.lane_budget(conn, "lane-c") == 5

    # An unassigned demand has no lane bound at all: the board bound alone applies.
    assert adm.lane_budget(conn, None) == 0
    assert adm.lane_budget(conn, "  ") == 0
    verdict = adm.decide(conn, lane=None)
    assert verdict.lane_budget == 0 and verdict.budget == 50 and verdict.admitted is True

    unassigned = kb.create_task(conn, title="nobody's", assignee=None)
    assert kb.get_task(conn, unassigned).status == "ready"

    # The override really binds: lane-a refuses at 2 ready cards, unmoved by the
    # 50-slot board budget.
    first = kb.create_task(conn, title="o1", assignee="lane-a")
    second = kb.create_task(conn, title="o2", assignee="lane-a")
    assert kb.get_task(conn, first).status == "ready"
    assert kb.get_task(conn, second).status == "ready"
    third = kb.create_task(conn, title="o3", assignee="lane-a")
    parked = kb.get_task(conn, third)
    assert parked.status == "todo" and parked.admit_state == adm.DEFERRED
    assert adm.decide(conn, lane="lane-a").lane_budget == 2


# --- Every ready-writer is accounted for -------------------------------------


def test_every_ready_status_writer_routes_through_the_admission_module():
    """A writer that bypasses the mechanism would be invisible to the audit.

    Source-level guard: any module that writes ``SET status = 'ready'`` must
    either BE the admission module or import it (so its landing is stamped).
    """
    import re

    package = Path(kb.__file__).resolve().parent.parent
    pattern = re.compile(r"SET\s+status\s*=\s*['\"]ready['\"]", re.IGNORECASE)
    offenders = []
    for path in sorted(package.rglob("*.py")):
        if "tests" in path.parts or path.name == Path(adm.__file__).name:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not pattern.search(source):
            continue
        if "kanban_db_admission" not in source:
            offenders.append(str(path.relative_to(package)))
    assert offenders == [], (
        "these modules write status='ready' without routing through "
        f"kanban_db_admission: {offenders}"
    )
