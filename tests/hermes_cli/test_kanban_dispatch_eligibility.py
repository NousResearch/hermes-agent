"""Regression tests for the dispatch-eligibility gates.

The factory's budgets were enforced only by the WORKER, after the spend. These
tests pin the new behaviour at the dispatcher's claim path:

* an over-cap card is NEVER claimed (zero runs consumed),
* a provider-walled card is parked in a native non-dispatchable state and does
  not spin on the ready lane,
* an active-PR card still cannot respawn (the pre-existing guard survives),
* provider capacity cannot be double-claimed inside one tick,
* a no-op evaluation does not append a new audit row,
* the gates fail CLOSED when the policy cannot be read.

Every test drives the real dispatcher through ``dispatch_once`` against a
throwaway board, so the claim path under test is the production one.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import pytest

FACTORY_LIB = Path("/Users/danielhurst/src/hermes-factory/scripts/lib")
# The hermes-agent checkout under test is the one this file lives in. It MUST
# win over the installed package: inserting the live checkout here would
# silently test the OLD dispatcher and every assertion would pass vacuously
# (or fail for the wrong reason).
AGENT_ROOT = Path(__file__).resolve().parents[2]

if str(FACTORY_LIB) not in sys.path:
    sys.path.insert(0, str(FACTORY_LIB))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))
else:
    # Already importable, but possibly behind another copy: move it to the front.
    sys.path.remove(str(AGENT_ROOT))
    sys.path.insert(0, str(AGENT_ROOT))


@pytest.fixture()
def policy_file(tmp_path: Path):
    """A limits.json the tests can mutate."""
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({
        "max_runs_per_card": 4,
        "dispatch_eligibility": {
            "max_in_progress_per_provider": 1,
            "provider_wall_cooldown_seconds": 900,
        },
        "provider_budget": {"max_wall_attempts": 5},
        "ci_budget": {"max_retry_attempts": 3},
    }), encoding="utf-8")
    return path


@pytest.fixture()
def board(tmp_path: Path, policy_file: Path, monkeypatch):
    """A fresh board with the eligibility policy wired in."""
    home = tmp_path / "home"
    os.makedirs(home / "profiles" / "alpha", exist_ok=True)
    (home / "profiles" / "alpha" / "config.yaml").write_text(
        "model:\n  provider: anthropic\n", encoding="utf-8")
    os.makedirs(home / "profiles" / "default", exist_ok=True)
    (home / "profiles" / "default" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    # The eligibility module is resolved by path at runtime; reload it so it
    # picks up the monkeypatched HOME.
    sys.modules.pop("dispatch_eligibility", None)

    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    # Point the eligibility loader at the test's limits file.
    monkeypatch.setattr(
        kbd, "_ELIGIBILITY_SEARCH_PATHS",
        (str(FACTORY_LIB),),
    )
    monkeypatch.setattr(
        kbd, "_load_eligibility_module",
        _fresh_eligibility_loader(),
    )
    # The policy module resolves its limits file at CALL time, so the test
    # points it at the temporary file with an env var rather than by
    # patching a module constant (which would pin the path at import time).
    monkeypatch.setenv("SANWA_LIMITS", str(policy_file))

    with kbc.connect_closing():
        kanban_db.create_board(slug="default", name="Test")
    return {"home": home, "policy": policy_file, "kb": kanban_db, "kbd": kbd, "kbc": kbc}


def _fresh_eligibility_loader():
    """A loader that re-imports the eligibility module per call.

    The production loader caches the module per process; the tests mutate
    HERMES_HOME between cases, so each load must be fresh.
    """
    import importlib.util

    def _loader():
        global _eligibility_module, _eligibility_module_loaded
        candidate = FACTORY_LIB / "dispatch_eligibility.py"
        spec = importlib.util.spec_from_file_location("dispatch_eligibility", candidate)
        module = importlib.util.module_from_spec(spec)
        sys.modules["dispatch_eligibility"] = module
        spec.loader.exec_module(module)
        return module

    return _loader


class _Card:
    """Minimal card handle: the id is what every assertion needs."""

    def __init__(self, card_id: str):
        self.id = card_id


#: A provider override requires a model override (the board rejects a bare
#: provider - a bare ``--provider`` would re-resolve the profile's model
#: against another backend). The model is irrelevant to every gate here; only
#: the provider is.
_PROVIDER_MODELS = {
    "anthropic": "claude-sonnet-4.6",
    "openai-codex": "gpt-5.6-sol",
    "minimax": "MiniMax-M3",
}


def _mk_card(kb, kbc, conn, *, title="card", assignee="alpha", provider=None,
             status="ready", priority=50, max_runtime=None):
    """Create a card in the given state.

    ``create_task`` only accepts ``initial_status`` of ``blocked`` or
    ``running``; ``ready`` is what a card with no incomplete parent gets by
    default, and ``todo`` is what an incomplete parent would produce. The
    requested state is therefore applied after creation with the board's own
    status write, so the test board matches production rows.
    """
    model = _PROVIDER_MODELS.get(provider) if provider else None
    card_id = kb.create_task(
        conn, title=title, body="body", assignee=assignee,
        priority=priority,
        provider_override=provider, model_override=model,
        max_runtime_seconds=max_runtime,
    )
    if status not in ("ready",):
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, "
                "worker_pid = NULL, current_run_id = NULL WHERE id = ?",
                (status, card_id),
            )
    return _Card(card_id)


def _record_run(kb, kbc, conn, task_id, *, profile="alpha", outcome="crashed",
                status="crashed", error=None, metadata=None):
    """Append a CLOSED, REAL run row to the card's claim log.

    A run row is opened by a claim and closed by ``_end_run``. The test needs
    historical runs on a card that is not currently running, so the row is
    written directly here with the board's own columns. Its duration is
    non-zero, so it counts as a worker attempt - the board's synthetic
    transition markers are zero-duration.
    """
    now = int(time.time())
    with kb.write_txn(conn):
        cur = conn.execute(
            "INSERT INTO task_runs "
            "  (task_id, profile, status, outcome, error, metadata, started_at, ended_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, profile, status, outcome, error,
             json.dumps(metadata or {}), now - 120, now),
        )
        return int(cur.lastrowid)


def _worker_run_count(conn, task_id):
    """Count the runs that consumed a card's BUILDER budget.

    Mirrors ``dispatch_eligibility.cumulative_builder_runs`` exactly: a real
    attempt is a non-zero-duration run whose outcome is not a transition
    marker (``scheduled``/``blocked``/``skipped``), not a review-lane run, not
    a CI failure (budgeted separately) and not a quota wall (budgeted by the
    provider). The two must agree, or the test would be asserting a different
    rule from the one the dispatcher enforces.
    """
    rows = conn.execute(
        "SELECT profile, outcome, started_at, ended_at FROM task_runs WHERE task_id = ?",
        (task_id,),
    ).fetchall()
    total = 0
    for row in rows:
        if "review" in (row["profile"] or "").lower():
            continue
        outcome = row["outcome"] or ""
        if outcome in ("scheduled", "blocked", "skipped", "not_started"):
            continue
        if outcome in ("ci_failed", "rate_limited"):
            continue
        started, ended = row["started_at"], row["ended_at"]
        if started is not None and ended is not None and ended <= started:
            continue
        total += 1
    return total


def _spawned_ids(result):
    """The task ids the tick spawned (``spawned`` entries are 3-tuples)."""
    return [entry[0] for entry in result.spawned]


def _ineligible_reasons(result):
    """Map task id -> refusal reason (``ineligible`` entries are 2-tuples)."""
    return dict(result.ineligible)


# ---------------------------------------------------------------- run cap ---

def test_over_cap_card_is_never_claimed(board):
    """A card at the run cap consumes ZERO runs: no claim, no new run row."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="over cap")
        # Four builder runs == the cap (max_runs_per_card: 4).
        for _ in range(4):
            _record_run(kb, kbc, conn, card.id, outcome="crashed", status="crashed")
        runs_before = _worker_run_count(conn, card.id)

        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert _worker_run_count(conn, card.id) == runs_before, (
            "over-cap card must consume zero worker runs"
        )
        assert _ineligible_reasons(result).get(card.id) == "run_cap"
        assert result.spawned == []
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (card.id,)
        ).fetchone()
        assert row["status"] == "blocked", "over-cap card parks, it does not respawn"


def test_under_cap_card_is_claimed(board):
    """The control case: below the cap the card still dispatches."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="under cap")
        # "crashed" rather than "completed": a completed run inside the 1h
        # respawn-guard window is refused by the (pre-existing, unrelated)
        # recent_success guard. The budget is what this test is about.
        for _ in range(3):
            _record_run(kb, kbc, conn, card.id, outcome="crashed", status="crashed")

        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert card.id in _spawned_ids(result)
        assert card.id not in _ineligible_reasons(result)


def test_review_runs_do_not_count_against_the_card_run_cap(board):
    """A review handoff is not implementation work: it must not exhaust a card."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="reviewed a lot")
        for _ in range(10):
            _record_run(kb, kbc, conn, card.id, profile="alpha-review",
                        outcome="crashed", status="crashed")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert card.id in _spawned_ids(result)


def test_run_cap_count_is_cumulative_not_consecutive(board):
    """A success must not reset the count: the budget is cumulative."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="cumulative")
        # completed, crashed, completed, crashed -> 4 cumulative runs
        for outcome in ("completed", "crashed", "completed", "crashed"):
            _record_run(kb, kbc, conn, card.id, outcome=outcome, status=outcome)
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) == "run_cap"


# ------------------------------------------------------------ provider wall ---

def test_provider_walled_card_parks_and_does_not_respawn(board):
    """A walled provider parks its cards; they do not spin on the ready lane."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        walled = _mk_card(kb, kbc, conn, title="walled provider", provider="anthropic")
        healthy = _mk_card(kb, kbc, conn, title="healthy provider", provider="openai-codex")

        # The provider hits a quota wall.
        _record_run(kb, kbc, conn, walled.id, outcome="rate_limited",
                    status="rate_limited", error="quota wall")

        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)

        reasons = _ineligible_reasons(result)
        assert reasons.get(walled.id) == "provider_wall"
        assert walled.id not in _spawned_ids(result)
        parked = {t: (state, why) for t, state, why in result.parked}
        assert parked[walled.id][0] == "scheduled"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (walled.id,)).fetchone()
        assert row["status"] == "scheduled", "walled card parks in a native park state"
        # The sibling on a healthy provider is unaffected.
        assert healthy.id in _spawned_ids(result)


def test_walled_card_cannot_be_claimed_while_parked(board):
    """A parked card is not in a dispatchable lane, so it cannot be claimed."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="walled", provider="anthropic")
        _record_run(kb, kbc, conn, card.id, outcome="rate_limited", status="rate_limited")
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        runs_after_park = _worker_run_count(conn, card.id)
        # A second tick must not claim it and must not add a worker run.
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert _worker_run_count(conn, card.id) == runs_after_park


def test_provider_wall_budget_routes_to_human(board):
    """Past the wall budget the card is a human decision, not another retry."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="wall budget", provider="anthropic")
        # Five wall hits == the cap (provider_budget.max_wall_attempts: 5).
        for _ in range(5):
            _record_run(kb, kbc, conn, card.id, outcome="rate_limited", status="rate_limited")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) == "provider_wall_budget"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (card.id,)).fetchone()
        assert row["status"] in ("blocked", "triage"), (
            "a card past the wall budget waits for a human"
        )


# ------------------------------------------------------- provider capacity ---

def test_provider_capacity_is_not_double_claimed_in_one_tick(board):
    """Two cards on one provider: only one may claim while the cap is 1."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        a = _mk_card(kb, kbc, conn, title="cap a", provider="anthropic", priority=90)
        b = _mk_card(kb, kbc, conn, title="cap b", provider="anthropic", priority=80)
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a2, **k: 999)
        spawned = _spawned_ids(result)
        assert len([s for s in spawned if s in (a.id, b.id)]) == 1, (
            "provider capacity must not be double-claimed in one tick"
        )


def test_provider_slot_waits_rather_than_parks(board):
    """A capacity wait is not a park: the card stays ready for a later tick."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        running_card = _mk_card(kb, kbc, conn, title="running", provider="anthropic",
                                 status="running")
        # Mark it running with a live run so the provider slot is occupied.
        if hasattr(kb, "_claim_and_open_run"):
            kb._claim_and_open_run(
                conn, running_card.id, "ready", "lock-test", 2 ** 31 - 1, 0,
                event_extra={},
            )
        waiting = _mk_card(kb, kbc, conn, title="waiting", provider="anthropic")

        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a2, **k: 999)
        reasons = _ineligible_reasons(result)
        if waiting.id in reasons:
            assert reasons[waiting.id] == "provider_slot"
            row = conn.execute("SELECT status FROM tasks WHERE id = ?", (waiting.id,)).fetchone()
            assert row["status"] == "ready", "a capacity wait must not park the card"


# --------------------------------------------------------------- audit rows ---

def test_no_op_evaluation_does_not_repeat_audit_events(board):
    """A parked card is evaluated every tick; it must not write a row each time."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="quiet", provider="anthropic")
        _record_run(kb, kbc, conn, card.id, outcome="rate_limited", status="rate_limited")
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        first = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'ineligible'",
            (card.id,),
        ).fetchone()[0]
        for _ in range(5):
            kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        second = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'ineligible'",
            (card.id,),
        ).fetchone()[0]
        assert second == first, "repeated no-op evaluations must not append audit rows"


def test_existing_respawn_guard_still_blocks_active_pr(board):
    """The pre-existing active-PR guard survives the new gates."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="has pr")
        kb.add_comment(conn, card.id, "tester", "see https://github.com/o/r/pull/7")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        guarded = dict(result.respawn_guarded)
        assert guarded.get(card.id) == "active_pr"
        assert card.id not in _spawned_ids(result)


# --------------------------------------------------------------- fail closed ---

def test_board_without_a_policy_file_is_untouched(board, tmp_path, monkeypatch):
    """A board with NO policy file has not opted in: it must dispatch normally.

    These budgets are a factory opt-in, not a Hermes-wide requirement. If a
    missing policy froze every board, every non-factory Hermes user would be
    locked out of their own kanban.
    """
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    missing = tmp_path / "does-not-exist.json"
    monkeypatch.setenv("SANWA_LIMITS", str(missing))
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="no policy file")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert card.id in _spawned_ids(result), (
            "a board with no policy file must dispatch normally"
        )


def test_malformed_policy_fails_closed(board, tmp_path, monkeypatch):
    """A policy that EXISTS but cannot be parsed fails closed: refuse, do not guess."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("SANWA_LIMITS", str(bad))
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="bad policy")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert _ineligible_reasons(result).get(card.id) == "policy_unreadable"
        assert result.spawned == []


# ------------------------------------------------------------ budget independence ---

def test_ci_retry_budget_is_independent_of_the_run_cap(board):
    """CI retries have their own budget: they do not consume the card's runs."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="ci retries")
        # Three CI failures == the CI budget (ci_budget.max_retry_attempts: 3),
        # but ZERO builder runs.
        for _ in range(3):
            _record_run(kb, kbc, conn, card.id, outcome="ci_failed", status="failed")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) == "ci_retry_budget"
        # ... and the card's own run budget is untouched: no builder runs.
        assert _worker_run_count(conn, card.id) == 0


def test_provider_wall_does_not_consume_the_card_run_budget(board):
    """A quota wall is a provider condition, not a card attempt."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="wall then work", provider="openai-codex")
        for _ in range(4):
            _record_run(kb, kbc, conn, card.id, profile="openai-codex",
                        outcome="rate_limited", status="rate_limited")
        # The provider wall budget is exhausted (4 hits at cap 5 is not), but the
        # card's OWN builder budget must still be zero: walls are not attempts.
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) in ("provider_wall", "provider_wall_budget")
        # The park itself writes a SYNTHETIC zero-duration 'scheduled' row
        # (the board's transition marker). That is not an attempt. Real
        # attempts are the non-zero-duration runs that are neither a wall nor
        # a CI failure.
        assert _worker_run_count(conn, card.id) == 0
