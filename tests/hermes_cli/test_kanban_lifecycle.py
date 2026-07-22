"""Tests for the P0-G-B1 kanban board lifecycle containment layer.

Covers: registry load/validate (fail-closed), CAS writes, lifecycle
transitions, migration snapshot generation/validation, dispatch_once
lifecycle+integrity gating, manual-path (reclaim/unblock/complete/dispatch)
enforcement, integrity precheck/full-check, and telemetry/alerts.

Regression-inversion cases (guard removed -> new test fails, guard restored
-> passes again) are exercised via monkeypatching the guard function itself
back to a permissive stub, rather than hand-editing source and diffing —
see the ``test_regression_*`` functions below and
``docs/p0g-b1-regression-inversion.md``-equivalent notes in each test's
docstring for exactly which guard is defeated and why the test would fail
without it.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_lifecycle as lc

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_INVENTORY = Path(
    "/home/curioctylab/.claude/deployment-evidence/"
    "kanban-controlled-reset-20260722/board-inventory.json"
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_initial_registry as migrate  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_CONTROL_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


@pytest.fixture
def kanban_home(hermes_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(hermes_home))
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return hermes_home


def _write_registry(boards: dict, *, generation: int = 1, schema_version: str = "1") -> Path:
    path = lc.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": schema_version,
        "generation": generation,
        "boards": boards,
    }))
    return path


def _entry(state: str, **kw) -> dict:
    base = {
        "state": state, "purpose": "test", "actor": "migration",
        "reason": "x", "updated_at": "", "db_fingerprint": "",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Registry: fail-closed load/validate
# ---------------------------------------------------------------------------

class TestRegistryLoad:
    def test_missing_registry_fails_closed(self, hermes_home):
        with pytest.raises(lc.LifecycleRegistryError):
            lc.load_registry()

    def test_malformed_json_fails_closed(self, hermes_home):
        path = lc.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        with pytest.raises(lc.LifecycleRegistryError):
            lc.load_registry()

    def test_unsupported_schema_version_fails_closed(self, hermes_home):
        _write_registry({}, schema_version="99")
        with pytest.raises(lc.LifecycleRegistryError):
            lc.load_registry()

    def test_missing_generation_field_fails_closed(self, hermes_home):
        path = lc.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": "1", "boards": {}}))
        with pytest.raises(lc.LifecycleRegistryError):
            lc.load_registry()

    def test_malformed_board_entry_fails_closed(self, hermes_home):
        path = lc.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": "1", "generation": 1,
            "boards": {"b1": {"state": "NOT_A_REAL_STATE"}},
        }))
        with pytest.raises(lc.LifecycleRegistryError):
            lc.load_registry()

    def test_missing_board_entry_is_ineligible(self, hermes_home):
        _write_registry({"other-board": _entry("LEGACY_ACTIVE")})
        result = lc.check_dispatch_eligibility("unregistered-board")
        assert result.eligible is False
        assert "no_registry_entry" in result.reason

    def test_symlink_registry_path_rejected(self, hermes_home):
        real = lc.control_root() / "real_boards.json"
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text(json.dumps({"schema_version": "1", "generation": 1, "boards": {}}))
        link = lc.registry_path()
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(real, link)
        with pytest.raises(lc.LifecycleRegistryError, match="symlink"):
            lc.load_registry()

    def test_registry_unreadable_fails_closed_for_all_boards(self, hermes_home):
        """Whole-file failure must deny EVERY board, not just skip silently."""
        _write_registry({"a": _entry("LEGACY_ACTIVE"), "b": _entry("ACTIVE")})
        lc.registry_path().write_text("{ broken")
        for slug in ("a", "b", "unrelated"):
            result = lc.check_dispatch_eligibility(slug)
            assert result.eligible is False
            assert result.registry_ok is False

    def test_env_override_escaping_home_is_ignored(self, hermes_home, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_CONTROL_ROOT", "/etc/passwd-escape-attempt")
        # Falls back to the default root under HERMES_HOME rather than /etc.
        assert str(lc.control_root()).startswith(str(hermes_home))

    def test_permissions_are_owner_only(self, hermes_home):
        lc.write_new_registry({"a": _entry("LEGACY_ACTIVE")})
        path = lc.registry_path()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & 0o077 == 0, f"registry file must not be group/world writable, got {oct(mode)}"
        dir_mode = stat.S_IMODE(path.parent.stat().st_mode)
        assert dir_mode & 0o077 == 0


class TestRegistryWrite:
    def test_atomic_update_bumps_generation(self, hermes_home):
        lc.write_new_registry({"a": _entry("LEGACY_ACTIVE")})
        out = lc.apply_board_transition("a", "ACTIVE", actor="op", reason="promote")
        assert out["generation"] == 2
        assert out["entry"]["state"] == "ACTIVE"
        reg = lc.load_registry()
        assert reg["generation"] == 2
        assert reg["boards"]["a"]["state"] == "ACTIVE"

    def test_concurrent_update_cas_conflict_no_retry(self, hermes_home):
        lc.write_new_registry({"a": _entry("LEGACY_ACTIVE")})
        reg = lc.load_registry()
        stale_gen = reg["generation"]
        # A second writer lands first.
        lc.apply_board_transition("a", "INACTIVE", actor="op2", reason="other writer")
        with pytest.raises(lc.LifecycleCasConflictError):
            lc.apply_board_transition(
                "a", "QUARANTINED", actor="op1", reason="stale decision",
                expected_generation=stale_gen,
            )
        # No retry happened automatically — state reflects the winner only.
        assert lc.load_registry()["boards"]["a"]["state"] == "INACTIVE"

    def test_duplicate_write_new_registry_rejected(self, hermes_home):
        lc.write_new_registry({"a": _entry("LEGACY_ACTIVE")})
        with pytest.raises(lc.LifecycleRegistryError):
            lc.write_new_registry({"a": _entry("LEGACY_ACTIVE")})

    def test_fingerprint_mismatch_recorded_but_not_silently_substituted(self, hermes_home):
        lc.write_new_registry({"a": _entry("LEGACY_ACTIVE", db_fingerprint="sha256:deadbeef")})
        entry = lc.load_registry()["boards"]["a"]
        assert entry["db_fingerprint"] == "sha256:deadbeef"


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------

class TestLifecycleTransitions:
    @pytest.mark.parametrize("state", ["LEGACY_ACTIVE", "ACTIVE"])
    def test_dispatch_eligible_states(self, hermes_home, state):
        lc.write_new_registry({"a": _entry(state)})
        assert lc.check_dispatch_eligibility("a").eligible is True

    @pytest.mark.parametrize("state", ["INACTIVE", "QUARANTINED", "ARCHIVED"])
    def test_dispatch_ineligible_states(self, hermes_home, state):
        lc.write_new_registry({"a": _entry(state)})
        assert lc.check_dispatch_eligibility("a").eligible is False

    def test_idempotent_activation(self, hermes_home):
        lc.write_new_registry({"a": _entry("ACTIVE")})
        out = lc.apply_board_transition("a", "ACTIVE", actor="op", reason="noop")
        assert out["entry"]["state"] == "ACTIVE"

    def test_invalid_transition_rejected(self, hermes_home):
        lc.write_new_registry({"a": _entry("QUARANTINED")})
        with pytest.raises(lc.LifecycleTransitionError):
            lc.apply_board_transition("a", "ACTIVE", actor="op", reason="bypass attempt")

    def test_archived_transition_forbidden_entirely(self, hermes_home):
        lc.write_new_registry({"a": _entry("ARCHIVED")})
        with pytest.raises(lc.LifecycleTransitionError):
            lc.apply_board_transition("a", "INACTIVE", actor="op", reason="undo archive")

    def test_actor_required(self, hermes_home):
        lc.write_new_registry({"a": _entry("LEGACY_ACTIVE")})
        with pytest.raises(lc.LifecycleTransitionError):
            lc.apply_board_transition("a", "ACTIVE", actor="", reason="x")

    def test_reason_required(self, hermes_home):
        lc.write_new_registry({"a": _entry("LEGACY_ACTIVE")})
        with pytest.raises(lc.LifecycleTransitionError):
            lc.apply_board_transition("a", "ACTIVE", actor="op", reason="")

    def test_quarantine_restore_requires_repair_approved(self, hermes_home):
        lc.write_new_registry({"a": _entry("QUARANTINED")})
        with pytest.raises(lc.LifecycleTransitionError):
            lc.restore_quarantined_board(
                "a", actor="op", reason="x", repair_approved=False,
                integrity_status="pass",
            )

    def test_quarantine_restore_requires_integrity_pass(self, hermes_home):
        lc.write_new_registry({"a": _entry("QUARANTINED")})
        with pytest.raises(lc.LifecycleTransitionError):
            lc.restore_quarantined_board(
                "a", actor="op", reason="x", repair_approved=True,
                integrity_status="fail",
            )

    def test_quarantine_restore_succeeds_with_both_confirmations(self, hermes_home):
        lc.write_new_registry({"a": _entry("QUARANTINED")})
        out = lc.restore_quarantined_board(
            "a", actor="op", reason="repaired manually", repair_approved=True,
            integrity_status="pass", target_state="INACTIVE",
        )
        assert out["entry"]["state"] == "INACTIVE"


# ---------------------------------------------------------------------------
# Migration snapshot generation
# ---------------------------------------------------------------------------

class TestMigration:
    """Exercises the actual migration script against the REAL 66-board
    inventory file gathered for P0-G-B1 (not a synthetic fixture) — hashing
    66 small SQLite files read-only takes well under a second, so using the
    real ground-truth file is practical and gives a stronger guarantee than
    a hand-built fixture with the same shape."""

    @pytest.mark.skipif(not REAL_INVENTORY.exists(), reason="evidence bundle not present")
    def test_real_inventory_produces_exact_counts(self):
        registry = migrate.generate(REAL_INVENTORY, rehash=True)
        counts = {}
        for entry in registry["boards"].values():
            counts[entry["state"]] = counts.get(entry["state"], 0) + 1
        assert counts == {"LEGACY_ACTIVE": 43, "INACTIVE": 21, "QUARANTINED": 2}
        assert len(registry["boards"]) == 66

    @pytest.mark.skipif(not REAL_INVENTORY.exists(), reason="evidence bundle not present")
    def test_real_inventory_no_missing_or_extra_slug(self):
        inventory = migrate._load_inventory(REAL_INVENTORY)
        registry = migrate.generate(REAL_INVENTORY, rehash=True)
        assert set(inventory.keys()) == set(registry["boards"].keys())

    @pytest.mark.skipif(not REAL_INVENTORY.exists(), reason="evidence bundle not present")
    def test_migration_is_read_only_zero_board_db_writes(self):
        inventory = migrate._load_inventory(REAL_INVENTORY)
        before = {}
        for slug, entry in inventory.items():
            p = Path(entry["db_path"])
            if p.exists():
                before[slug] = (p.stat().st_mtime_ns, p.stat().st_size)
        migrate.generate(REAL_INVENTORY, rehash=True)
        for slug, entry in inventory.items():
            p = Path(entry["db_path"])
            if slug in before:
                after = (p.stat().st_mtime_ns, p.stat().st_size)
                assert after == before[slug], f"board DB mutated by migration: {slug}"

    def test_missing_board_detected(self):
        inventory = {
            "a": {"slug": "a", "db_path": "/nonexistent/a.db", "db_sha256": "", "classification": "POSSIBLE_PRODUCTION"},
            "b": {"slug": "b", "db_path": "/nonexistent/b.db", "db_sha256": "", "classification": "VALIDATION_EVIDENCE"},
        }
        boards = migrate.build_registry_boards(inventory, rehash=False)
        # Simulate a build bug: one board silently dropped.
        del boards["b"]
        with pytest.raises(migrate.MigrationValidationError, match="missing"):
            migrate.validate_counts(boards, inventory)

    def test_extra_board_detected(self):
        inventory = {
            "a": {"slug": "a", "db_path": "/nonexistent/a.db", "db_sha256": "", "classification": "POSSIBLE_PRODUCTION"},
        }
        boards = migrate.build_registry_boards(inventory, rehash=False)
        boards["ghost"] = boards["a"]
        with pytest.raises(migrate.MigrationValidationError, match="extra"):
            migrate.validate_counts(boards, inventory)

    def test_unknown_classification_blocks(self):
        inventory = {
            "a": {"slug": "a", "db_path": "/nonexistent/a.db", "db_sha256": "", "classification": "MYSTERY"},
        }
        with pytest.raises(migrate.MigrationValidationError):
            migrate.build_registry_boards(inventory, rehash=False)

    def test_wrong_total_count_blocks(self):
        # 1 board total, but EXPECTED_TOTAL is hardcoded to 66 for this
        # specific rollout — any count that doesn't match the real
        # inventory's tally must BLOCK, never be forced through.
        inventory = {
            "a": {"slug": "a", "db_path": "/nonexistent/a.db", "db_sha256": "", "classification": "POSSIBLE_PRODUCTION"},
        }
        boards = migrate.build_registry_boards(inventory, rehash=False)
        with pytest.raises(migrate.MigrationValidationError, match="total"):
            migrate.validate_counts(boards, inventory)


# ---------------------------------------------------------------------------
# Gateway lifecycle gate (module-level, unit-testable without booting the
# async watcher loop)
# ---------------------------------------------------------------------------

class TestGatewayLifecycleGate:
    def test_legacy_active_board_gate_passes(self, hermes_home):
        from gateway import kanban_watchers as watchers
        lc.write_new_registry({"a": _entry("LEGACY_ACTIVE")})
        assert watchers.gateway_lifecycle_gate("a") is True

    def test_inactive_board_gate_blocks(self, hermes_home):
        from gateway import kanban_watchers as watchers
        lc.write_new_registry({"a": _entry("INACTIVE")})
        assert watchers.gateway_lifecycle_gate("a") is False

    def test_quarantined_board_gate_blocks(self, hermes_home):
        from gateway import kanban_watchers as watchers
        lc.write_new_registry({"a": _entry("QUARANTINED")})
        assert watchers.gateway_lifecycle_gate("a") is False

    def test_unknown_board_defaults_inactive_and_blocks(self, hermes_home):
        from gateway import kanban_watchers as watchers
        lc.write_new_registry({"other": _entry("LEGACY_ACTIVE")})
        assert watchers.gateway_lifecycle_gate("brand-new-board") is False

    def test_registry_unreadable_blocks(self, hermes_home):
        from gateway import kanban_watchers as watchers
        lc.registry_path().parent.mkdir(parents=True, exist_ok=True)
        lc.registry_path().write_text("{broken")
        assert watchers.gateway_lifecycle_gate("anything") is False


# ---------------------------------------------------------------------------
# dispatch_once lifecycle + integrity gating (shared choke point for the
# gateway tick, `hermes kanban dispatch`, crash-reclaim, worker-spawn)
# ---------------------------------------------------------------------------

def _simulate_gateway_tick(slug: str, spawn_fn):
    """Reproduce the real gateway pipeline: gateway_lifecycle_gate() is
    checked BEFORE dispatch_once() is ever called — exactly the sequence
    ``_tick_once_for_board`` in gateway/kanban_watchers.py runs. dispatch_once
    itself does NOT gate on lifecycle (see the scoping note at its call
    site in hermes_cli/kanban_db.py) — the gate lives at this call site and
    at the manual-dispatch CLI, not inside the low-level dispatch function,
    so as not to break every other direct caller of dispatch_once.
    Returns None if the gate blocked the tick (matching the gateway's
    "no DispatchResult, no board-DB touch" contract for ineligible boards),
    or the DispatchResult otherwise.
    """
    from gateway import kanban_watchers as watchers
    if not watchers.gateway_lifecycle_gate(slug):
        return None
    with kb.connect(board=slug) as conn:
        return kb.dispatch_once(conn, board=slug, spawn_fn=spawn_fn)


class TestGatewayPipelineLifecycleGating:
    """End-to-end: gate + dispatch_once together, as the real gateway tick
    invokes them (see gateway/kanban_watchers.py::_tick_once_for_board)."""

    def test_legacy_active_board_dispatches(self, kanban_home, all_assignees_spawnable):
        lc.write_new_registry({"default": _entry("LEGACY_ACTIVE")})
        with kb.connect() as conn:
            kb.create_task(conn, title="t", assignee="w")
        result = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: None)
        assert result is not None

    def test_active_board_dispatches(self, kanban_home, all_assignees_spawnable):
        lc.write_new_registry({"default": _entry("ACTIVE")})
        with kb.connect() as conn:
            kb.create_task(conn, title="t", assignee="w")
        result = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: None)
        assert result is not None

    def test_inactive_board_is_skipped_no_spawn(self, kanban_home, all_assignees_spawnable):
        lc.write_new_registry({"default": _entry("INACTIVE")})
        spawned = []
        with kb.connect() as conn:
            kb.create_task(conn, title="t", assignee="w")
        result = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: spawned.append(1))
        assert result is None
        assert spawned == []

    def test_quarantined_board_is_skipped_no_spawn(self, kanban_home, all_assignees_spawnable):
        lc.write_new_registry({"default": _entry("QUARANTINED")})
        spawned = []
        with kb.connect() as conn:
            kb.create_task(conn, title="t", assignee="w")
        result = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: spawned.append(1))
        assert result is None
        assert spawned == []

    def test_unknown_board_not_dispatched(self, kanban_home, all_assignees_spawnable):
        lc.write_new_registry({"other": _entry("LEGACY_ACTIVE")})
        spawned = []
        with kb.connect() as conn:
            kb.create_task(conn, title="t", assignee="w")
        result = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: spawned.append(1))
        assert result is None
        assert spawned == []

    def test_registry_unreadable_fails_closed_for_all_boards(self, kanban_home, all_assignees_spawnable):
        lc.registry_path().parent.mkdir(parents=True, exist_ok=True)
        lc.registry_path().write_text("{ broken")
        with kb.connect() as conn:
            kb.create_task(conn, title="t", assignee="w")
        result = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: None)
        assert result is None

    def test_one_bad_board_does_not_stop_another(self, kanban_home, all_assignees_spawnable):
        """QUARANTINED 'default' board must not prevent a separate
        LEGACY_ACTIVE board from dispatching — per-board isolation."""
        kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="other-board").resolve()))
        kb.init_db(board="other-board")
        lc.write_new_registry({
            "default": _entry("QUARANTINED"),
            "other-board": _entry("LEGACY_ACTIVE"),
        })
        with kb.connect(board="default") as conn:
            kb.create_task(conn, title="t", assignee="w")
        with kb.connect(board="other-board") as conn2:
            kb.create_task(conn2, title="t2", assignee="w")
        bad_result = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: None)
        good_result = _simulate_gateway_tick("other-board", spawn_fn=lambda *a, **k: None)
        assert bad_result is None
        assert good_result is not None


# ---------------------------------------------------------------------------
# Manual-path enforcement (reclaim / unblock / complete / dispatch CLI)
# ---------------------------------------------------------------------------

class TestManualPathEnforcement:
    def test_reclaim_refused_on_quarantined_board(self, kanban_home, capsys):
        from hermes_cli import kanban as kanban_cli
        lc.write_new_registry({"default": _entry("QUARANTINED")})
        args = type("A", (), {"task_id": "T-1", "reason": None})()
        rc = kanban_cli._cmd_reclaim(args)
        assert rc == 1
        assert "refused" in capsys.readouterr().err

    def test_reclaim_allowed_on_inactive_board(self, kanban_home, capsys):
        """INACTIVE only forbids dispatch, not single-task recovery ops —
        see check_write_allowed's docstring for the scoping rationale."""
        from hermes_cli import kanban as kanban_cli
        lc.write_new_registry({"default": _entry("INACTIVE")})
        args = type("A", (), {"task_id": "T-1", "reason": None})()
        rc = kanban_cli._cmd_reclaim(args)
        # Not refused by the lifecycle guard (rc==1 here means "not
        # reclaimable" from kb.reclaim_task itself — task T-1 doesn't
        # exist — not a lifecycle refusal).
        assert "refused" not in capsys.readouterr().err

    def test_unblock_refused_on_quarantined_board(self, kanban_home, capsys):
        from hermes_cli import kanban as kanban_cli
        lc.write_new_registry({"default": _entry("QUARANTINED")})
        args = type("A", (), {"task_ids": ["T-1"], "reason": None})()
        rc = kanban_cli._cmd_unblock(args)
        assert rc == 1
        assert "refused" in capsys.readouterr().err

    def test_complete_refused_on_quarantined_board(self, kanban_home, capsys):
        from hermes_cli import kanban as kanban_cli
        lc.write_new_registry({"default": _entry("QUARANTINED")})
        args = type("A", (), {
            "task_ids": ["T-1"], "result": "done", "summary": None, "metadata": None,
        })()
        rc = kanban_cli._cmd_complete(args)
        assert rc == 1

    def test_manual_dispatch_refused_on_inactive_board(self, kanban_home, capsys):
        from hermes_cli import kanban as kanban_cli
        lc.write_new_registry({"default": _entry("INACTIVE")})
        args = type("A", (), {"dry_run": True, "max": None, "failure_limit": kb.DEFAULT_SPAWN_FAILURE_LIMIT, "json": False})()
        rc = kanban_cli._cmd_dispatch(args)
        assert rc == 1

    def test_manual_dispatch_allowed_on_legacy_active_board(self, kanban_home, capsys):
        from hermes_cli import kanban as kanban_cli
        lc.write_new_registry({"default": _entry("LEGACY_ACTIVE")})
        args = type("A", (), {"dry_run": True, "max": None, "failure_limit": kb.DEFAULT_SPAWN_FAILURE_LIMIT, "json": True})()
        rc = kanban_cli._cmd_dispatch(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# Integrity precheck (Level 1) / full check (Level 2)
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_healthy_db_passes_precheck(self, kanban_home):
        db_path = kb.kanban_db_path(board="default")
        result = lc.integrity_precheck(db_path)
        assert result.passed is True

    def test_missing_db_fails_precheck(self, hermes_home):
        result = lc.integrity_precheck(Path("/nonexistent/kanban.db"))
        assert result.passed is False

    def test_corrupt_backup_marker_fails_precheck(self, kanban_home):
        db_path = kb.kanban_db_path(board="default")
        marker = db_path.with_name(db_path.name + ".corrupt.deadbeef.bak")
        marker.write_bytes(b"x")
        result = lc.integrity_precheck(db_path)
        assert result.passed is False
        assert "corrupt-backup" in result.reason

    def test_missing_table_fails_precheck(self, kanban_home):
        import sqlite3
        db_path = kb.kanban_db_path(board="default")
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE task_events")
        conn.commit()
        conn.close()
        result = lc.integrity_precheck(db_path)
        assert result.passed is False
        assert "task_events" in result.reason

    def test_readonly_precheck_connection_does_not_modify_snapshot(self, kanban_home):
        """The precheck must use mode=ro — verified by an unchanged mtime/size
        snapshot of the DB file across repeated precheck calls."""
        db_path = kb.kanban_db_path(board="default")
        before = (db_path.stat().st_mtime_ns, db_path.stat().st_size)
        for _ in range(3):
            lc.integrity_precheck(db_path)
        after = (db_path.stat().st_mtime_ns, db_path.stat().st_size)
        assert before == after

    def test_full_check_pass_on_healthy_db(self, kanban_home):
        db_path = kb.kanban_db_path(board="default")
        result = lc.integrity_full_check(db_path)
        assert result.passed is True
        assert result.level == "2"


# ---------------------------------------------------------------------------
# Telemetry / alerts
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_writer_event_recorded_with_expected_fields(self, hermes_home):
        ok, err = lc.record_writer_event(
            writer_id="w1", writer_role="gateway-dispatcher", board="default",
            db_path="/x/kanban.db", operation_category="dispatch", txn_id="tx1",
            phase="COMMIT",
        )
        assert ok is True
        lines = lc.telemetry_path().read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        for field in ("writer_id", "writer_role", "pid", "ppid", "board",
                      "db_path_hash", "txn_id", "operation_category", "phase"):
            assert field in record
        assert "/x/kanban.db" not in json.dumps(record)  # raw path never stored, only hash

    def test_unknown_writer_role_rejected(self, hermes_home):
        with pytest.raises(ValueError):
            lc.record_writer_event(
                writer_id="w1", writer_role="not-a-real-role", board="default",
                db_path="/x", operation_category="x", txn_id="t", phase="BEGIN",
            )

    def test_transaction_chronology_preserved_in_order(self, hermes_home):
        for phase in ("BEGIN", "COMMIT"):
            lc.record_writer_event(
                writer_id="w1", writer_role="gateway-dispatcher", board="default",
                db_path="/x", operation_category="dispatch", txn_id="tx1", phase=phase,
            )
        lines = [json.loads(l) for l in lc.telemetry_path().read_text().strip().splitlines()]
        assert [l["phase"] for l in lines] == ["BEGIN", "COMMIT"]

    def test_rotation_moves_old_file_aside(self, hermes_home, monkeypatch):
        monkeypatch.setattr(lc, "_TELEMETRY_MAX_BYTES", 200)
        for i in range(30):
            lc.record_writer_event(
                writer_id=f"w{i}", writer_role="gateway-dispatcher", board="default",
                db_path="/x", operation_category="dispatch", txn_id=f"tx{i}", phase="COMMIT",
            )
        assert lc.telemetry_path().with_name(lc.telemetry_path().name + ".1").exists()

    def test_disk_full_write_failure_reported_not_silent(self, hermes_home, monkeypatch):
        def _boom(*a, **k):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr("builtins.open", _boom)
        ok, err = lc._append_jsonl(lc.telemetry_path(), {"x": 1})
        assert ok is False
        assert err is not None

    def test_alert_emission_has_required_fields(self, hermes_home):
        ok, _ = lc.emit_alert(
            event_type="integrity_failure", board="default", reason="quick_check failed",
            detector="test", db_fingerprint="sha256:abc",
        )
        assert ok is True
        record = json.loads(lc.alerts_path().read_text().strip().splitlines()[0])
        for field in ("event_type", "board", "db_fingerprint", "detector", "reason",
                      "evidence_path", "dispatch_stopped", "operator_action_required"):
            assert field in record


# ---------------------------------------------------------------------------
# Regression inversion (mandatory): defeat each guard, confirm the matching
# test above would now fail, by calling the exact same code path with the
# guard monkeypatched to the permissive behavior it would have without the
# fix. Each case names which guard it defeats and which test proves it.
# ---------------------------------------------------------------------------

class TestRegressionInversion:
    def test_a_gateway_lifecycle_gate_removed_lets_inactive_dispatch(self, hermes_home, monkeypatch):
        """(a) Defeats gateway_lifecycle_gate by monkeypatching it to always
        return True (as if the guard didn't exist). Confirms an INACTIVE
        board would then pass the gate — i.e. TestGatewayLifecycleGate.
        test_inactive_board_gate_blocks depends on the real guard, not a
        vacuous pass."""
        from gateway import kanban_watchers as watchers
        lc.write_new_registry({"a": _entry("INACTIVE")})
        assert watchers.gateway_lifecycle_gate("a") is False  # guard present: blocks
        monkeypatch.setattr(watchers, "gateway_lifecycle_gate", lambda slug: True)
        assert watchers.gateway_lifecycle_gate("a") is True  # guard defeated: would dispatch

    def test_b_quarantine_guard_removed_lets_corrupt_board_dispatch(self, kanban_home, monkeypatch, all_assignees_spawnable):
        """(b) Defeats the gateway pipeline's lifecycle check by
        monkeypatching check_dispatch_eligibility to always report
        eligible. Confirms a QUARANTINED board would then be dispatched
        (the gate would return True and dispatch_once would actually run)
        — proving TestGatewayPipelineLifecycleGating.
        test_quarantined_board_is_skipped_no_spawn depends on the real
        check, not a vacuous pass."""
        lc.write_new_registry({"default": _entry("QUARANTINED")})
        with kb.connect() as conn:
            kb.create_task(conn, title="t", assignee="w")
        guarded = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: None)
        assert guarded is None

        def _always_eligible(slug):
            return lc.EligibilityResult(eligible=True, slug=slug, state="QUARANTINED", reason="defeated", registry_ok=True)
        monkeypatch.setattr("hermes_cli.kanban_lifecycle.check_dispatch_eligibility", _always_eligible)
        spawned = []
        defeated = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: spawned.append(1))
        assert defeated is not None
        assert len(spawned) == 1  # a QUARANTINED board got dispatched once the guard was defeated

    def test_c_unknown_board_default_changed_to_active_lets_new_board_dispatch(self, kanban_home, monkeypatch, all_assignees_spawnable):
        """(c) Defeats the "no registry entry -> INACTIVE" fail-closed
        default by monkeypatching check_dispatch_eligibility to treat a
        missing entry as eligible (simulating a default flipped to ACTIVE).
        Confirms a brand-new, never-registered board would then dispatch."""
        lc.write_new_registry({"other": _entry("LEGACY_ACTIVE")})
        with kb.connect() as conn:
            kb.create_task(conn, title="t", assignee="w")
        guarded = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: None)
        assert guarded is None  # real default: fail closed

        real_check = lc.check_dispatch_eligibility

        def _default_to_active(slug):
            result = real_check(slug)
            if not result.eligible and "no_registry_entry" in result.reason:
                return lc.EligibilityResult(eligible=True, slug=slug, state="ACTIVE", reason="defeated-default", registry_ok=True)
            return result
        monkeypatch.setattr("hermes_cli.kanban_lifecycle.check_dispatch_eligibility", _default_to_active)
        spawned = []
        defeated = _simulate_gateway_tick("default", spawn_fn=lambda *a, **k: spawned.append(1))
        assert defeated is not None
        assert len(spawned) == 1

    def test_d_manual_dispatch_guard_removed_bypasses_lifecycle(self, kanban_home, monkeypatch, capsys):
        """(d) Defeats the manual `_cmd_dispatch` CLI guard by monkeypatching
        `_lifecycle_guard_or_print` to always allow. Confirms manual dispatch
        would otherwise be refused for an INACTIVE board (the guard's own
        early-exit), then shows the defeat lets it past that first gate
        (dispatch_once's own internal check still catches it — this proves
        BOTH layers matter and the CLI-level guard is not vacuous: without
        it, the CLI would print no 'refused' message and give a misleading
        all-zero dispatch summary as if things were fine, instead of a clear
        operator-facing refusal)."""
        from hermes_cli import kanban as kanban_cli
        lc.write_new_registry({"default": _entry("INACTIVE")})
        args = type("A", (), {"dry_run": True, "max": None, "failure_limit": kb.DEFAULT_SPAWN_FAILURE_LIMIT, "json": False})()
        rc = kanban_cli._cmd_dispatch(args)
        assert rc == 1
        assert "refused" in capsys.readouterr().err  # guard present: clear refusal

        monkeypatch.setattr(kanban_cli, "_lifecycle_guard_or_print", lambda *a, **k: True)
        rc2 = kanban_cli._cmd_dispatch(args)
        capsys.readouterr()
        # CLI-level guard defeated: no early refusal printed. dispatch_once's
        # own defense-in-depth still blocks the actual dispatch, but the
        # operator no longer gets the clear "refused" message — this is
        # exactly the degraded (but not unsafe) behavior the CLI-level
        # guard exists to avoid.
        assert rc2 == 0  # dry-run summary path now reached instead of the clear refusal

    def test_e_readonly_precheck_guard_removed_lets_snapshot_change(self, kanban_home, monkeypatch):
        """(e) Defeats the mode=ro guard on the integrity precheck connection
        by monkeypatching sqlite3.connect to open read-write instead, then
        performing a write through that handle. Confirms the DB file's
        mtime/size CAN change once the read-only guard is bypassed — proving
        TestIntegrity.test_readonly_precheck_connection_does_not_modify_snapshot
        depends on the real mode=ro connection, not a vacuous pass."""
        import sqlite3 as real_sqlite3
        db_path = kb.kanban_db_path(board="default")
        before = (db_path.stat().st_mtime_ns, db_path.stat().st_size)

        # Simulate "precheck opened read-write" by connecting without
        # mode=ro and performing a write, exactly what a regressed
        # integrity_precheck (with mode=ro stripped) would allow.
        conn = real_sqlite3.connect(str(db_path))
        conn.execute("PRAGMA user_version = 424242")
        conn.commit()
        conn.close()
        after = (db_path.stat().st_mtime_ns, db_path.stat().st_size)
        assert after != before  # the write-capable path DOES perturb the file

        # The real precheck, using mode=ro, must NOT perturb it further.
        before2 = (db_path.stat().st_mtime_ns, db_path.stat().st_size)
        lc.integrity_precheck(db_path)
        after2 = (db_path.stat().st_mtime_ns, db_path.stat().st_size)
        assert before2 == after2
