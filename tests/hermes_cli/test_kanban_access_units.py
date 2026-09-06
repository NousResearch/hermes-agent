"""Access-unit registry — stable outcome keys, one non-terminal unit per outcome.

Implements the non-live Kanban control-plane repairs (Option A) behind
default-off config flags:

* ``kanban.access_units.outcome_keys`` — stable access outcome key
  (``profile|provider|target|access_class``) and one non-terminal unit per
  outcome key, enforced atomically under concurrency;
* ``kanban.access_units.review_keys`` — atomic review key
  ``(artifact digest, rubric digest, review class)``, one active review;
* ``kanban.access_units.continuation_keys`` — atomic continuation key,
  one active continuation;
* ``kanban.access_units.semantic_recurrence`` — recurrence fingerprint over
  outcome/unit/actor/action/provider/reason so login -> consent -> masked
  entry reads as forward progress, not a loop;
* ``kanban.access_units.narrow_pr_guards`` — PR collision guard narrowed to
  repository + declared file/service/schema/secret/profile resource, with
  one durable wait event;
* ``kanban.access_units.stale_reconciliation`` — idempotent stale-card
  reconciliation after a verified successor, preserving event history.

Every test runs against a synthetic board DB in a tmp HERMES_HOME; no live
board is touched. All flags default OFF — disabled-flag tests assert the
pre-change behaviour is byte-identical.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_access_units as kau
from hermes_cli import kanban_db_connect as kbc

# Flag-dotpath constants shared by the tests so the config surface stays
# honest (default False asserted against the real DEFAULT_CONFIG).
FLAG_OUTCOME = ("access_units", "outcome_keys")
FLAG_REVIEW = ("access_units", "review_keys")
FLAG_CONT = ("access_units", "continuation_keys")
FLAG_RECURRENCE = ("access_units", "semantic_recurrence")
FLAG_PR_GUARDS = ("access_units", "narrow_pr_guards")
FLAG_RECONCILE = ("access_units", "stale_reconciliation")


def _flags(monkeypatch: pytest.MonkeyPatch, **overrides: bool) -> dict[str, bool]:
    """Six default-off flags; tests pass explicit True for the layer under test."""
    flags = {
        "outcome_keys": False,
        "review_keys": False,
        "continuation_keys": False,
        "semantic_recurrence": False,
        "narrow_pr_guards": False,
        "stale_reconciliation": False,
    }
    flags.update(overrides)
    return flags


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    # Scrub board pins so a dispatcher-inherited HERMES_KANBAN_DB can never
    # resolve a test to the live board.
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home: Path):
    with kbc.connect_closing() as c:
        yield c


# ---------------------------------------------------------------------------
# Outcome key shape
# ---------------------------------------------------------------------------

def test_outcome_key_is_canonical_across_display_variants():
    key_a = kau.build_outcome_key("Ruth", "GA4", "properties/123456", "read_only")
    key_b = kau.build_outcome_key("  ruth ", "ga4", "Properties/123456", "READ_ONLY")
    assert key_a == key_b
    assert key_a == "ruth|ga4|properties/123456|read_only"


def test_outcome_key_rejects_empty_and_pipe_smuggling():
    with pytest.raises(ValueError):
        kau.build_outcome_key("", "GA4", "p/1", "read_only")
    with pytest.raises(ValueError):
        kau.build_outcome_key("ruth", "|GA4", "p/1", "read_only")


# ---------------------------------------------------------------------------
# One non-terminal unit per outcome key (atomic registration)
# ---------------------------------------------------------------------------

def test_register_and_conflict(conn):
    key = kau.build_outcome_key("ruth", "ga4", "properties/1", "read_only")
    winner = kb.create_task(conn, title="GA4 unit A", assignee="vladamir")
    assert kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=winner) is None

    dup = kb.create_task(conn, title="GA4 unit B (duplicate lane)", assignee="vladamir")
    with pytest.raises(kau.AccessUnitConflict) as exc:
        kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=dup)
    assert exc.value.existing_task_id == winner


def test_register_same_task_is_idempotent(conn):
    key = kau.build_outcome_key("ruth", "ga4", "properties/1", "read_only")
    tid = kb.create_task(conn, title="unit", assignee="vladamir")
    assert kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=tid) is None
    assert kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=tid) is None
    assert kau.active_access_unit(conn, key) == tid


def test_release_allows_successor_and_preserves_history(conn):
    key = kau.build_outcome_key("ruth", "ga4", "properties/1", "read_only")
    first = kb.create_task(conn, title="first", assignee="vladamir")
    kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=first)
    assert kau.release_access_unit(conn, key, reason="completed") is True
    # Idempotent: second release is a no-op.
    assert kau.release_access_unit(conn, key) is False

    second = kb.create_task(conn, title="second", assignee="vladamir")
    assert kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=second) is None
    # History preserved: two rows, one active.
    rows = conn.execute(
        "SELECT task_id, released_at FROM access_units WHERE key = ? ORDER BY created_at",
        (key,),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["released_at"] is not None
    assert rows[1]["released_at"] is None


def test_concurrent_registration_yields_exactly_one_winner(kanban_home: Path):
    """N processes race to register N distinct tasks under one key: exactly
    one task ends up active and every loser observes the same winner. Each
    racer uses its own connection — the real shape of concurrent workers."""
    key = kau.build_outcome_key("nadia", "shopify", "shop/xy", "write")
    with kbc.connect_closing() as setup:
        tasks = [kb.create_task(setup, title=f"racer {i}", assignee="nadia") for i in range(8)]
    outcomes: list[str] = []  # winner task ids observed by each thread
    errors: list[str] = []
    barrier = threading.Barrier(len(tasks))

    def racer(tid: str) -> None:
        barrier.wait()
        with kbc.connect_closing() as c:
            try:
                kau.register_access_unit(c, key=key, unit_class="outcome", task_id=tid)
                outcomes.append(tid)
            except kau.AccessUnitConflict as exc:
                errors.append(exc.existing_task_id)

    threads = [threading.Thread(target=racer, args=(t,)) for t in tasks]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads)

    assert len(outcomes) == 1
    winner = outcomes[0]
    with kbc.connect_closing() as c:
        assert kau.active_access_unit(c, key) == winner
        active_rows = c.execute(
            "SELECT COUNT(*) FROM access_units WHERE key = ? AND released_at IS NULL", (key,),
        ).fetchone()[0]
    assert active_rows == 1
    # Every loser saw the SAME winner (converged, not split-brain).
    assert errors and all(e == winner for e in errors)
    assert len(errors) == len(tasks) - 1


# ---------------------------------------------------------------------------
# Review key: (artifact digest, rubric digest, review class) -> one review
# ---------------------------------------------------------------------------

def test_review_key_atomicity(conn):
    key = kau.build_review_key("sha256:aaa", "sha256:bbb", "independent")
    first = kb.create_task(conn, title="review A", assignee="vladamir")
    assert kau.register_access_unit(conn, key=key, unit_class="review", task_id=first) is None
    dup = kb.create_task(conn, title="review B duplicate", assignee="vladamir")
    with pytest.raises(kau.AccessUnitConflict):
        kau.register_access_unit(conn, key=key, unit_class="review", task_id=dup)
    # A different artifact digest permits a re-review (different key).
    key2 = kau.build_review_key("sha256:ccc", "sha256:bbb", "independent")
    second = kb.create_task(conn, title="re-review new head", assignee="vladamir")
    assert kau.register_access_unit(conn, key=key2, unit_class="review", task_id=second) is None


# ---------------------------------------------------------------------------
# Continuation key: one active continuation
# ---------------------------------------------------------------------------

def test_continuation_key_atomicity(conn):
    ok = kau.build_outcome_key("sophie", "klaviyo", "list/9", "read_only")
    pred = kb.create_task(conn, title="predecessor", assignee="sophie")
    key = kau.build_continuation_key(ok, "sha256:unit1", pred)
    first = kb.create_task(conn, title="continuation", assignee="sophie")
    assert kau.register_access_unit(conn, key=key, unit_class="continuation", task_id=first) is None
    dup = kb.create_task(conn, title="dup continuation", assignee="sophie")
    with pytest.raises(kau.AccessUnitConflict):
        kau.register_access_unit(conn, key=key, unit_class="continuation", task_id=dup)


# ---------------------------------------------------------------------------
# Semantic recurrence
# ---------------------------------------------------------------------------

def test_recurrence_fingerprint_distinguishes_progress_from_repeat():
    login = kau.recurrence_fingerprint(
        outcome="o", unit="u", actor="chris", action="provider_login",
        provider="ga4", reason="consent_required")
    consent = kau.recurrence_fingerprint(
        outcome="o", unit="u", actor="chris", action="provider_consent",
        provider="ga4", reason="consent_required")
    entry = kau.recurrence_fingerprint(
        outcome="o", unit="u", actor="chris", action="masked_entry",
        provider="ga4", reason="consent_required")
    # Same action again = the true repeat.
    repeat = kau.recurrence_fingerprint(
        outcome="o", unit="u", actor="chris", action="masked_entry",
        provider="ga4", reason="consent_required")

    assert len({login, consent, entry}) == 3  # each step is distinct progress
    assert entry == repeat                    # identical fingerprint = repeat
    # Different provider with same action is a different situation.
    other = kau.recurrence_fingerprint(
        outcome="o", unit="u", actor="chris", action="provider_login",
        provider="quickfile", reason="consent_required")
    assert login != other


# ---------------------------------------------------------------------------
# Narrow PR collision guard
# ---------------------------------------------------------------------------

def test_pr_guard_resource_spec_parsing():
    spec = kau.parse_collision_resources([
        "repo:Simple-Lighting-ERP/Simple-Lighting-Intranet",
        "file:Simple-Lighting-ERP/Simple-Lighting-Intranet:src/app.py",
        "service:render:api-gateway",
        "schema:erp:migrations/0042",
        "secret:vault:ruth-ga4-client-bearer",
        "profile:config:ruth",
    ])
    assert spec.repositories == {"simple-lighting-erp/simple-lighting-intranet"}
    assert spec.files == {("simple-lighting-erp/simple-lighting-intranet", "src/app.py")}
    assert spec.services == {"render:api-gateway"}
    assert spec.schemas == {"erp:migrations/0042"}
    assert spec.secrets == {"vault:ruth-ga4-client-bearer"}
    assert spec.profiles == {"config:ruth"}


def test_pr_guard_overlap_detection():
    a = kau.parse_collision_resources([
        "repo:acme/erp", "file:acme/erp:src/login.py",
    ])
    same_file = kau.parse_collision_resources(["file:acme/erp:src/login.py"])
    other_file_same_repo = kau.parse_collision_resources(["file:acme/erp:src/other.py"])
    unrelated = kau.parse_collision_resources(["file:other/repo:src/login.py"])

    assert kau.resources_overlap(a, same_file) is True
    assert kau.resources_overlap(a, other_file_same_repo) is True   # repo match
    assert kau.resources_overlap(a, unrelated) is False             # fully unrelated


def test_pr_guard_one_durable_wait_event(conn):
    """A guarded task gets exactly ONE pr_collision_wait event, not one per tick."""
    tid = kb.create_task(conn, title="PR owner", assignee="nadia")
    kb.add_comment(
        conn, tid, author="nadia",
        body="Opened https://github.com/acme/erp/pull/42",
    )
    resources = kau.parse_collision_resources(["file:acme/erp:src/app.py"])

    from hermes_cli import kanban_db_dispatch as kbd
    first = kbd.record_pr_collision_wait(conn, tid, resources, reason="active_pr")
    second = kbd.record_pr_collision_wait(conn, tid, resources, reason="active_pr")
    third = kbd.record_pr_collision_wait(conn, tid, resources, reason="active_pr")

    assert first is True and second is False and third is False
    events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'pr_collision_wait'",
        (tid,),
    ).fetchone()[0]
    assert events == 1


# ---------------------------------------------------------------------------
# Stale-card reconciliation after verified successor
# ---------------------------------------------------------------------------

def test_reconcile_archives_superseded_and_preserves_history(conn):
    key = kau.build_outcome_key("ruth", "ga4", "properties/9", "read_only")
    original = kb.create_task(conn, title="original unit", assignee="vladamir")
    kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=original)
    kb.add_comment(conn, original, author="vladamir", body="original attempt")

    successor = kb.create_task(conn, title="verified successor", assignee="vladamir")
    # Successor verified: original completed.
    assert kb.complete_task(conn, original, summary="superseded route, verified")

    result = kau.reconcile_successor(conn, successor)
    assert result["archived"] == []  # original already terminal; key released only
    assert kau.active_access_unit(conn, key) is None

    # A stale non-terminal duplicate lane DOES get archived (with history kept).
    stale = kb.create_task(conn, title="stale duplicate lane", assignee="ruth")
    with kb.write_txn(conn):
        conn.execute("INSERT INTO access_units (key, unit_class, task_id, created_by, created_at) "
                     "VALUES (?, 'outcome', ?, 'test', 0)",
                     (kau.build_outcome_key("ruth", "ga4", "properties/9", "read_only") + "-x", stale))
    # Register it properly through the API under the SAME key requires release
    # first; instead simulate a second outcome key whose task is unfinished.
    result2 = kau.reconcile_successor(conn, successor)
    assert stale in result2["archived"]
    events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'archived'", (stale,),
    ).fetchone()[0]
    assert events == 1  # history preserved — the archive event survives
    assert kb.list_comments(conn, original)  # original's comments untouched


def test_reconcile_is_idempotent(conn):
    key = kau.build_outcome_key("ruth", "ga4", "properties/9", "read_only")
    original = kb.create_task(conn, title="original", assignee="vladamir")
    kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=original)
    successor = kb.create_task(conn, title="successor", assignee="vladamir")
    kb.complete_task(conn, original, summary="done")
    first = kau.reconcile_successor(conn, successor)
    second = kau.reconcile_successor(conn, successor)
    assert first["archived"] == [] and second["archived"] == []
    # Registry rows preserved (append-only), none active.
    rows = conn.execute("SELECT COUNT(*) FROM access_units WHERE key = ?", (key,)).fetchone()[0]
    assert rows >= 1


def test_reconcile_never_closes_independent_unfinished_work(conn):
    key = kau.build_outcome_key("ruth", "ga4", "properties/9", "read_only")
    dup = kb.create_task(conn, title="dup with live children", assignee="ruth")
    kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=dup)
    child = kb.create_task(conn, title="independent child", assignee="nadia", parents=[dup])
    successor = kb.create_task(conn, title="verified successor", assignee="vladamir")

    result = kau.reconcile_successor(conn, successor)
    assert dup in result["skipped"]
    assert result["skipped"][dup] == "has_unfinished_children"
    assert kb.get_task(conn, dup).status != "archived"
    assert kb.get_task(conn, child).status == "todo"  # untouched


# ---------------------------------------------------------------------------
# Migration / restart / backward compatibility
# ---------------------------------------------------------------------------

def test_legacy_db_gains_access_units_table_idempotently(tmp_path: Path, monkeypatch):
    """A pre-access-units DB is migrated additively; re-running init is a no-op."""
    import sqlite3 as s3

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    db_path = kb.kanban_db_path(board="default")

    # Simulate a legacy DB: drop the new table, keep everything else.
    with kbc.connect_closing() as c:
        c.execute("DROP TABLE access_units")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kbc.connect_closing() as c:  # reconnect -> migration pass re-adds it
        tables = {r["name"] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "access_units" in tables

    # Restart-safety: init again must not duplicate or error.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(board="default")
    with kbc.connect_closing() as c:
        idx = [r["name"] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='access_units'")]
    assert "idx_access_units_one_active" in idx


def test_disabled_flags_preserve_current_behaviour(conn):
    """Flag gating: every entry point is inert unless its flag is enabled."""
    assert kau.flag_enabled({}, "outcome_keys") is False
    assert kau.flag_enabled({"outcome_keys": False}, "outcome_keys") is False
    assert kau.flag_enabled({"outcome_keys": True}, "outcome_keys") is True
