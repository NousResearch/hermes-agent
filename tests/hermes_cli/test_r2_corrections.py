"""R2 adversarial RED witnesses — Round 2 corrections.

R2-1  shadow exactly-once atomic under concurrent writers
R2-2  schema install must not commit the caller's transaction
R2-3  fail-closed legacy duplicate migration
R2-4  gate proves the full review→requirement→evidence chain
R2-5  pilot provenance not self-mintable
R2-6  emit_trigger honours the supplied home
R2-7  human provenance via validated registry + genuinely human seam
R2-8  dashboard raw-list validation + safe invalid root
R2-9  strict force_mode semantics
R2-10 identity/state finding recurrence
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import hermaguard_events as he
from hermes_cli import shadow_classifier as sc
from hermes_cli import pilot_evidence as pe


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: True)
    conn = kb.connect()
    yield conn, home
    conn.close()


def _task(conn, tier="full", kind="task", created_by="dashboard"):
    return kb.create_task(
        conn, title="t", assignee="octacon",
        body="## Problem\nx\n## Success Criteria\ny", triage=True,
        tier=tier, task_kind=kind, created_by=created_by,
    )


def _review(conn, tid):
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    conn.commit()
    kb.request_review(conn, tid, force=True)
    conn.commit()
    row = conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _report(home, tid, content=b"hermaguard report"):
    artifact = home / "feature-artifacts" / tid
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "hermaguard-report.md").write_bytes(content)
    return artifact


# ── R2-1 ─────────────────────────────────────────────────────────────────────

class TestR21ShadowAtomicity:
    def test_two_connections_no_duplicate_suggestion(self, env):
        """RED: two racing writers produce duplicate same-version suggestions."""
        conn, home = env
        tid = _task(conn)
        conn.commit()
        db_path = str(conn.execute("PRAGMA database_list").fetchone()[2])
        barrier = threading.Barrier(2)

        def worker():
            con2 = sqlite3.connect(db_path, timeout=10)
            try:
                barrier.wait()
                s = sc.suggest(con2, tid)
                if s is not None:
                    sc.insert_shadow_event(con2, tid, s)
                    con2.commit()
            finally:
                con2.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rows = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, sc.EVENT_KIND),
        ).fetchone()[0]
        assert rows == 1


# ── R2-2 ─────────────────────────────────────────────────────────────────────

class TestR22CallerTransaction:
    def test_ensure_does_not_commit_caller_txn(self, env):
        """RED: executescript() commits the caller's pending write, so a
        later rollback cannot undo it."""
        conn, home = env
        tid = _task(conn)
        # kanban_db uses isolation_level=None (autocommit); emulate a caller
        # with an explicit open transaction the way write_txn callers do.
        conn.execute("BEGIN")
        conn.execute("UPDATE tasks SET title = 'canary-title' WHERE id = ?", (tid,))
        # caller has an UNCOMMITTED write inside an explicit transaction
        he.ensure_uniqueness_invariants(conn)
        conn.rollback()
        title = conn.execute("SELECT title FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        assert title != "canary-title"  # rollback must undo the canary


# ── R3-7 ─────────────────────────────────────────────────────────────────────

class TestR37TransactionLifecycle:
    """R3-7: pin the transaction/index rollback lifecycle so a future
    "fix" cannot reintroduce an implicit commit.

    Contract pinned here:
      1. a caller rollback removes caller writes, Hermaguard writes and
         any index first installed inside that rolled-back transaction;
      2. the next committed ensure/append installs and persists exactly
         ONE index;
      3. concurrent installers converge to exactly one index;
      4. duplicate residue remains fail-closed and untouched.
    """

    @staticmethod
    def _index_count(conn):
        return conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
            "AND name='ux_hermaguard_task_kind_review'").fetchone()[0]

    def test_rollback_removes_caller_writes_and_installed_index(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        assert self._index_count(conn) == 0
        # Caller opens a transaction, does its own writes, then installs
        # the Hermaguard invariant (which uses execute(), never
        # executescript — so it stays inside the caller's transaction).
        conn.execute("BEGIN")
        conn.execute("UPDATE tasks SET title = 'r37-canary' WHERE id = ?", (tid,))
        he.ensure_uniqueness_invariants(conn)
        # Inside the transaction the index is visible to the caller...
        assert self._index_count(conn) == 1
        # ...but a caller rollback removes EVERYTHING: caller writes,
        # Hermaguard writes, AND the index installed in that transaction.
        conn.rollback()
        title = conn.execute("SELECT title FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        assert title != "r37-canary"  # caller write rolled back
        assert self._index_count(conn) == 0  # index rolled back with it

    def test_next_committed_ensure_persists_exactly_one_index(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        # First attempt: installed inside a rolled-back transaction (gone).
        conn.execute("BEGIN")
        he.ensure_uniqueness_invariants(conn)
        conn.rollback()
        assert self._index_count(conn) == 0
        # Next committed ensure persists EXACTLY ONE index, and it
        # survives a fresh connection (genuinely persisted).
        he.ensure_uniqueness_invariants(conn)
        conn.commit()
        assert self._index_count(conn) == 1
        fresh = sqlite3.connect(str(conn.execute("PRAGMA database_list").fetchone()[2]))
        assert self._index_count(fresh) == 1
        fresh.close()
        # A second ensure on the persisted index is a no-op (no duplicate).
        he.ensure_uniqueness_invariants(conn)
        assert self._index_count(conn) == 1

    def test_concurrent_installers_converge_to_one_index(self, env):
        conn, home = env
        _task(conn)
        db_path = str(conn.execute("PRAGMA database_list").fetchone()[2])
        barrier = threading.Barrier(4)

        def installer():
            con = sqlite3.connect(db_path, timeout=15)
            try:
                barrier.wait()
                con.execute("BEGIN IMMEDIATE")
                he.ensure_uniqueness_invariants(con)
                con.commit()
            except sqlite3.IntegrityError:
                con.rollback()  # raced installers converge via replay
            finally:
                con.close()

        threads = [threading.Thread(target=installer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All concurrent installers converge to exactly one index.
        assert self._index_count(conn) == 1

    def test_duplicate_residue_fail_closed_and_untouched(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        # Legacy duplicate residue (two requirement rows, same cycle).
        payload = json.dumps({"review_event_id": rid, "policy_version": "old"})
        for _ in range(2):
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, ?, strftime('%s','now'))",
                (tid, he.KIND_REQUIRED, payload))
        conn.commit()
        # Fail closed: no index is created over duplicate residue...
        result = he.ensure_uniqueness_invariants_safe(conn)
        assert result["installed"] is False
        assert result["duplicate_groups"] >= 1
        assert self._index_count(conn) == 0
        # ...and the residue is never deleted or rewritten (append-only).
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, he.KIND_REQUIRED)).fetchone()[0] == 2
        # And it stays fail-closed on every subsequent call.
        result2 = he.ensure_uniqueness_invariants_safe(conn)
        assert result2["installed"] is False
        assert self._index_count(conn) == 0


# ── R2-3 ─────────────────────────────────────────────────────────────────────

class TestR23LegacyDuplicates:
    def test_index_install_fails_closed_on_duplicates(self, env):
        """RED: legacy DB with duplicate requirement rows must fail with a
        bounded diagnostic, not crash every later call."""
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        # Inject duplicate requirement rows directly (legacy residue)
        payload = json.dumps({"review_event_id": rid, "policy_version": "old"})
        for _ in range(2):
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, ?, strftime('%s','now'))",
                (tid, he.KIND_REQUIRED, payload),
            )
        conn.commit()
        result = he.ensure_uniqueness_invariants_safe(conn)
        assert result["installed"] is False
        assert result["duplicate_groups"] >= 1
        # retry must be deterministic and non-destructive
        result2 = he.ensure_uniqueness_invariants_safe(conn)
        assert result2["installed"] is False
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, he.KIND_REQUIRED)).fetchone()[0] == 2  # never deduped silently

    def test_migration_plan_reports_safe_ids_only(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        payload = json.dumps({"review_event_id": rid, "policy_version": "old"})
        for _ in range(2):
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, ?, strftime('%s','now'))",
                (tid, he.KIND_REQUIRED, payload),
            )
        conn.commit()
        plan = he.migration_plan(conn)
        assert plan["duplicate_groups"] >= 1
        assert plan["affected_task_ids"] == [tid]  # safe IDs, no payloads


# ── R2-4 ─────────────────────────────────────────────────────────────────────

class TestR24GateChain:
    def test_forged_evidence_without_requirement_blocked(self, env):
        """RED: structurally valid evidence row with zero requirement rows
        must NOT open the gate."""
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        artifact = home / "feature-artifacts" / tid
        artifact.mkdir(parents=True, exist_ok=True)
        report = artifact / "hermaguard-report.md"
        report.write_bytes(b"forged evidence")
        import hashlib
        digest = hashlib.sha256(b"forged evidence").hexdigest()
        # Directly insert a valid-looking evidence row — no requirement exists
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, ?, strftime('%s','now'))",
            (tid, he.KIND_EVIDENCE,
             json.dumps({"review_event_id": rid, "report": "hermaguard-report.md",
                         "report_sha256": digest, "status": "pass",
                         "version": "1.0.0"})),
        )
        conn.commit()
        ok, reason = he.gate_review_transition(
            home / "kanban.db", conn, tid, review_event_id=rid,
            artifact_dir=artifact,
            kanban_cfg={"hermaguard_event_mode": True},
        )
        assert ok is False  # forged evidence must not open the gate


# ── R2-5 ─────────────────────────────────────────────────────────────────────

class TestR25ProvenanceForgery:
    def test_public_mint_does_not_confer_authority(self, env):
        """RED: the public mint helper must not confer authority — keyless
        minting raises, and attacker-keyed provenance fails the gate."""
        conn, home = env
        tid = _task(conn)
        record = {
            "task_id": tid,
            "pipeline_contract_version": "c1",
            "final_status": "done",
            "reviewer_verdict": "pass",
            "label": None,
            "ever_simulated": False,
        }
        # Keyless mint is refused outright
        with pytest.raises(ValueError):
            pe.mint_live_pilot_provenance(
                tid, review_event_id=1, evidence_event_id=2,
                pipeline_contract_version="c1", authorised_by="attacker",
            )
        # Attacker minting with their OWN key still fails the gate (the gate
        # verifies with the TRUSTED key, not the attacker's)
        provenance = pe.mint_live_pilot_provenance(
            tid, review_event_id=1, evidence_event_id=2,
            pipeline_contract_version="c1", authorised_by="attacker",
            signing_key=b"attacker-key",
        )
        record["live_provenance"] = provenance
        assert pe.real_pilot_gate_satisfied(record, signing_key=b"trusted-key") is False

    def test_valid_trusted_provenance_passes(self):
        record = {
            "task_id": "t1", "pipeline_contract_version": "c1",
            "final_status": "done", "reviewer_verdict": "pass",
            "label": None, "ever_simulated": False,
        }
        prov = pe.mint_live_pilot_provenance(
            "t1", review_event_id=1, evidence_event_id=2,
            pipeline_contract_version="c1", authorised_by="sahil",
            signing_key=b"trusted-key",
        )
        record["live_provenance"] = prov
        assert pe.real_pilot_gate_satisfied(record, signing_key=b"trusted-key") is True

    def test_wrong_key_rejected(self, env):
        record = {
            "task_id": "t1", "pipeline_contract_version": "c1",
            "final_status": "done", "reviewer_verdict": "pass",
            "label": None, "ever_simulated": False,
        }
        # signed with key A, verified with key B
        from hermes_cli import pilot_evidence as pe
        prov = pe.mint_live_pilot_provenance(
            "t1", review_event_id=1, evidence_event_id=2,
            pipeline_contract_version="c1", authorised_by="sahil",
            signing_key=b"key-a",
        )
        record["live_provenance"] = prov
        assert pe.real_pilot_gate_satisfied(record, signing_key=b"key-b") is False


# ── R2-6 ─────────────────────────────────────────────────────────────────────

class TestR26ExplicitHome:
    def test_emit_trigger_writes_to_supplied_home(self, env, tmp_path, monkeypatch):
        """RED: emit_trigger(..., hermes_home=B) must write/verify against B,
        not the process HERMES_HOME."""
        import importlib.util
        import sys
        home_a = tmp_path / "home-a"
        (home_a / "profiles" / "octacon").mkdir(parents=True)
        (home_a / "governance").mkdir()
        (home_a / "profiles" / "octacon" / "config.yaml").write_text("model:\n  default: t\n")
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        home_b = tmp_path / "home-b"
        (home_b / "governance").mkdir(parents=True)

        # R3-5: derive core checkout from this test file's own location
        REPO = str(Path(__file__).resolve().parents[2])
        spec = importlib.util.spec_from_file_location(
            "set_eval_r26", f"{REPO}/scripts/denji-self-eval-trigger.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import time
        now = int(time.time())
        from hermes_cli.profile_activity_ledger import append_event
        for i in range(4):
            append_event(source="t", event_type="kanban.crashed",
                         event_id=f"r26-{i}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="x",
                         occurred_at=now - 100 - i)
        d = mod.decide_trigger("octacon", since=now - 86400,
                               hermes_home=home_a, now=now)
        assert d["trigger"] is True
        eid = mod.emit_trigger(d, hermes_home=home_b)
        # Event must land in home B, not A
        con_b = sqlite3.connect(f"file:{home_b}/governance/profile-activity-ledger.sqlite?mode=ro", uri=True)
        rows_b = con_b.execute(
            "SELECT COUNT(*) FROM activity_events WHERE event_type='profile.self_eval.trigger'").fetchone()[0]
        con_b.close()
        assert rows_b == 1


# ── R3-1 ─────────────────────────────────────────────────────────────────────

def _jsonl_event_count(home, event_id):
    """Count JSONL mirror lines for event_id under home (B or A)."""
    mirror_dir = Path(home) / "governance" / "logboard" / "profile-activity-ledger"
    if not mirror_dir.exists():
        return 0
    count = 0
    for f in mirror_dir.glob("*.jsonl"):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    if json.loads(line).get("event_id") == event_id:
                        count += 1
                except json.JSONDecodeError:
                    continue
    return count


def _db_event_count(home, event_id):
    db = Path(home) / "governance" / "profile-activity-ledger.sqlite"
    if not db.exists():
        return 0
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n = con.execute("SELECT COUNT(*) FROM activity_events WHERE event_id = ?",
                    (event_id,)).fetchone()[0]
    con.close()
    return n


class TestR31ExplicitHomeIsolation:
    """R3-1: explicit_home=B must keep SQLite AND JSONL in B, leave process
    home A byte/row/path unchanged, and emit_trigger must return the exact
    event id after a confirmed B read-back."""

    @staticmethod
    def _load_trigger():
        import importlib.util
        REPO = str(Path(__file__).resolve().parents[2])
        spec = importlib.util.spec_from_file_location(
            "set_eval_r31", f"{REPO}/scripts/denji-self-eval-trigger.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_full_ab_isolation_and_exact_return(self, tmp_path, monkeypatch):
        import importlib.util
        from hermes_cli.profile_activity_ledger import append_event, query_events
        import time

        home_a = tmp_path / "home-a"
        (home_a / "profiles" / "octacon").mkdir(parents=True)
        (home_a / "governance").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        home_b = tmp_path / "home-b"
        (home_b / "governance").mkdir(parents=True)

        now = int(time.time())
        # Process home A already holds one event (the pre-existing baseline)
        append_event(source="t", event_type="kanban.crashed",
                     event_id="r31-a-baseline", actor_profile="octacon",
                     target_profile="octacon", occurred_at=now - 500)
        a_db_before = _db_event_count(home_a, "r31-a-baseline")
        assert a_db_before == 1
        # Snapshot A's JSONL state
        a_jsonl_before = _jsonl_event_count(home_a, "r31-a-baseline")
        a_db_all_before = sqlite3.connect(
            f"file:{home_a}/governance/profile-activity-ledger.sqlite?mode=ro", uri=True)
        a_total_before = a_db_all_before.execute("SELECT COUNT(*) FROM activity_events").fetchone()[0]
        a_db_all_before.close()

        # ── append with explicit_home=B ──
        eid = "r31-b-event"
        append_event(source="t", event_type="profile.self_eval.trigger",
                     event_id=eid, actor_profile="denji", target_profile="octacon",
                     object_type="profile.self_eval", occurred_at=now,
                     explicit_home=home_b)

        # 1) B SQLite contains the event exactly once
        assert _db_event_count(home_b, eid) == 1
        # 2) B JSONL contains the event exactly once
        assert _jsonl_event_count(home_b, eid) == 1
        # 3) query_events(explicit_home=B) returns B's event, never A's
        b_rows = query_events(event_types=["profile.self_eval.trigger"], explicit_home=home_b)
        assert any(r["event_id"] == eid for r in b_rows)
        assert all(r["event_id"] != "r31-a-baseline" for r in b_rows)
        # 4) query_events(explicit_home=A) never returns B's event
        a_rows = query_events(explicit_home=home_a)
        assert all(r["event_id"] != eid for r in a_rows)
        # 5) A DB unchanged (same row count as before the B append)
        a_db_after = sqlite3.connect(
            f"file:{home_a}/governance/profile-activity-ledger.sqlite?mode=ro", uri=True)
        a_total_after = a_db_after.execute("SELECT COUNT(*) FROM activity_events").fetchone()[0]
        a_db_after.close()
        assert a_total_after == a_total_before
        # 6) A JSONL unchanged
        assert _jsonl_event_count(home_a, "r31-a-baseline") == a_jsonl_before
        # 7) A DB baseline row still present and unchanged
        assert _db_event_count(home_a, "r31-a-baseline") == 1

        # 8) emit_trigger returns the EXACT event id after confirmed B read-back
        mod = self._load_trigger()
        for i in range(4):
            append_event(source="t", event_type="kanban.crashed",
                         event_id=f"r31-fail-{i}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="x",
                         occurred_at=now - 100 - i, explicit_home=home_b)
        d = mod.decide_trigger("octacon", since=now - 86400, hermes_home=home_b, now=now)
        assert d["trigger"] is True
        returned = mod.emit_trigger(d, hermes_home=home_b)
        assert returned is not None, "emit_trigger must return the event id"
        assert returned.startswith("selfeval-trigger-octacon-")
        # 9) the returned id is actually present in B (DB + JSONL)
        assert _db_event_count(home_b, returned) == 1
        assert _jsonl_event_count(home_b, returned) == 1

    def test_idempotent_replay_one_row_one_mirror(self, tmp_path, monkeypatch):
        from hermes_cli.profile_activity_ledger import append_event
        import time
        home_a = tmp_path / "home-a"
        (home_a / "governance").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        home_b = tmp_path / "home-b"
        (home_b / "governance").mkdir(parents=True)
        now = int(time.time())
        eid = "r31-replay"
        for _ in range(3):
            append_event(source="t", event_type="profile.self_eval.trigger",
                         event_id=eid, actor_profile="denji", target_profile="octacon",
                         occurred_at=now, explicit_home=home_b)
        assert _db_event_count(home_b, eid) == 1
        assert _jsonl_event_count(home_b, eid) == 1


# ── R2-7 ─────────────────────────────────────────────────────────────────────

class TestR27HumanProvenance:
    def test_malformed_registry_cannot_authorise_spoofed_name(self, env, tmp_path):
        """RED: malformed registry (invalid entry) must yield no eligibility."""
        conn, home = env
        gov = home / "governance"
        gov.mkdir(parents=True, exist_ok=True)
        # Malformed registry: kensei-review listed as profile with bad enum
        (gov / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: webhook-7831, kind: emperor, parent: KENSEI, lifecycle: zombie, domains: [], gateway_unit: null}\n"
        )
        tid = _task(conn, created_by="webhook-7831")
        assert sc.suggest(conn, tid) is None

    def test_registry_membership_alone_not_human(self, env, tmp_path):
        """RED: a registry-verified worker profile's stamp is not proof of
        interactive human creation."""
        conn, home = env
        gov = home / "governance"
        gov.mkdir(parents=True, exist_ok=True)
        import yaml
        (gov / "profile-registry.yaml").write_text(yaml.safe_dump({
            "schema_version": 1,
            "root": {"name": "KENSEI", "description": "r"},
            "profiles": [
                {"name": "octacon-backend", "kind": "worker", "parent": "KENSEI",
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
            ],
        }))
        tid = _task(conn, created_by="octacon-backend")
        assert sc.suggest(conn, tid) is None  # membership ≠ human

    def test_interactive_marker_enables_profile_author(self, env, tmp_path, monkeypatch):
        conn, home = env
        gov = home / "governance"
        gov.mkdir(parents=True, exist_ok=True)
        import yaml
        (gov / "profile-registry.yaml").write_text(yaml.safe_dump({
            "schema_version": 1,
            "root": {"name": "KENSEI", "description": "r"},
            "profiles": [
                {"name": "misa-misa", "kind": "lead", "parent": "KENSEI",
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
            ],
        }))
        # Task created by registry-verified author WITH an interactive marker
        tid = _task(conn, created_by="misa-misa")
        monkeypatch.setattr(sc, "_task_has_interactive_marker", lambda c, t: True)
        assert sc.suggest(conn, tid) is not None
        # Without the marker: not human
        monkeypatch.setattr(sc, "_task_has_interactive_marker", lambda c, t: False)
        assert sc.suggest(conn, tid) is None


# ── R2-8 ─────────────────────────────────────────────────────────────────────

class TestR28DashboardValidation:
    @pytest.fixture
    def dash(self, tmp_path, monkeypatch):
        # R3-5: the dashboard checkout is supplied EXPLICITLY via the
        # EVIDENCE_SPINE_DASHBOARD env var (bound by the selector).  The core
        # checkout is derived from this file; the dashboard is cross-repo and
        # must never be assumed at a builder path.
        dash_root = os.environ.get("EVIDENCE_SPINE_DASHBOARD")
        if not dash_root or not Path(dash_root).is_dir():
            pytest.skip("EVIDENCE_SPINE_DASHBOARD not supplied (cross-repo test)")
        assert isinstance(dash_root, str)  # narrow for static checkers
        home = tmp_path / "hermes-home"
        (home / "profiles").mkdir(parents=True)
        (home / "governance").mkdir()
        import sys
        sys.path.insert(0, dash_root)
        # R2-8 isolation: 'backend' may already be cached from another
        # selector file (module-name collision) — force a fresh import of
        # the r2 dashboard's backend for this fixture.
        import importlib
        saved_backend = sys.modules.pop("backend", None)
        from backend import profile_docs
        monkeypatch.setattr(profile_docs, "HERMES_HOME", home)
        monkeypatch.setattr(profile_docs, "PROFILES_DIR", home / "profiles")
        monkeypatch.setattr(profile_docs, "GATEWAY_PROFILES", ["octacon"])
        yield home, profile_docs
        sys.modules.pop("backend", None)
        if saved_backend is not None:
            sys.modules["backend"] = saved_backend
        sys.path.remove(dash_root)

    def test_duplicate_entries_rejected(self, dash):
        home, profile_docs = dash
        (home / "profiles" / "octacon").mkdir()
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: lead, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
            "- {name: octacon, kind: lead, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
        )
        h = profile_docs.profile_hierarchy()
        node = next(n for n in h["nodes"] if n["name"] == "octacon")
        assert node.get("registry_state") == "invalid_registry"

    def test_missing_required_fields_rejected(self, dash):
        home, profile_docs = dash
        (home / "profiles" / "octacon").mkdir()
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: lead}\n"  # missing parent/lifecycle/domains/gateway_unit
        )
        h = profile_docs.profile_hierarchy()
        node = next(n for n in h["nodes"] if n["name"] == "octacon")
        assert node.get("registry_state") == "invalid_registry"

    def test_invalid_registry_safe_root(self, dash):
        """RED: invalid registry must NOT control the returned root."""
        home, profile_docs = dash
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: EVIL_ROOT, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: emperor, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
        )
        h = profile_docs.profile_hierarchy()
        assert h["root"] == "KENSEI"  # safe constant

    def test_empty_profiles_rejected(self, dash):
        home, profile_docs = dash
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\nroot: {name: KENSEI, description: r}\nprofiles: []\n"
        )
        h = profile_docs.profile_hierarchy()
        assert h["root"] == "KENSEI"


# ── R3-4 ─────────────────────────────────────────────────────────────────────

class TestR34GlobalInvalidRegistry:
    """R3-4: a registry that EXISTS but fails validation is a GLOBAL invalid
    state — every filesystem profile is ``invalid_registry``, including ones
    the malformed list does not mention.  ``unregistered`` applies only when
    a VALID registry omits a profile (or the registry is absent).
    """

    @pytest.fixture
    def dash(self, tmp_path, monkeypatch):
        dash_root = os.environ.get("EVIDENCE_SPINE_DASHBOARD")
        if not dash_root or not Path(dash_root).is_dir():
            pytest.skip("EVIDENCE_SPINE_DASHBOARD not supplied (cross-repo test)")
        assert isinstance(dash_root, str)
        home = tmp_path / "hermes-home"
        (home / "profiles").mkdir(parents=True)
        (home / "governance").mkdir()
        import sys
        sys.path.insert(0, dash_root)
        import importlib
        saved_backend = sys.modules.pop("backend", None)
        from backend import profile_docs
        monkeypatch.setattr(profile_docs, "HERMES_HOME", home)
        monkeypatch.setattr(profile_docs, "PROFILES_DIR", home / "profiles")
        monkeypatch.setattr(profile_docs, "GATEWAY_PROFILES", ["octacon"])
        yield home, profile_docs
        sys.modules.pop("backend", None)
        if saved_backend is not None:
            sys.modules["backend"] = saved_backend
        sys.path.remove(dash_root)

    def _mk_profiles(self, home):
        for name in ("octacon", "wesker"):
            (home / "profiles" / name).mkdir(parents=True)

    def test_malformed_registry_all_profiles_invalid(self, dash):
        """RED: duplicate entry makes the registry malformed → BOTH fs
        profiles (listed and unlisted) must be invalid_registry."""
        home, profile_docs = dash
        self._mk_profiles(home)
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: lead, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
            "- {name: octacon, kind: lead, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
        )
        h = profile_docs.profile_hierarchy()
        states = {n["name"]: n["registry_state"] for n in h["nodes"] if n["type"] != "root"}
        assert states["octacon"] == "invalid_registry"
        # the unlisted profile must ALSO be globally invalid, not unregistered
        assert states["wesker"] == "invalid_registry"

    def test_valid_registry_omitted_profile_unregistered(self, dash):
        home, profile_docs = dash
        self._mk_profiles(home)
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: lead, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
        )
        h = profile_docs.profile_hierarchy()
        states = {n["name"]: n["registry_state"] for n in h["nodes"] if n["type"] != "root"}
        assert states["octacon"] == "registered"
        assert states["wesker"] == "unregistered"

    def test_absent_registry_unregistered(self, dash):
        home, profile_docs = dash
        self._mk_profiles(home)
        # No registry file at all — NOT a malformed registry
        h = profile_docs.profile_hierarchy()
        states = {n["name"]: n["registry_state"] for n in h["nodes"] if n["type"] != "root"}
        assert states["octacon"] == "unregistered"
        assert states["wesker"] == "unregistered"


# ── R2-9 ─────────────────────────────────────────────────────────────────────


    # ── R4-1: all eight existing-invalid shapes yield global invalid_registry ──

    _R41_SHAPES = {
        "malformed_yaml": "{{{ not yaml",
        "yaml_scalar": "just-a-string",
        "wrong_schema_version": (
            "schema_version: 2\n"
            "root: {name: KENSEI}\n"
            "profiles:\n- {name: octacon, kind: worker, parent: KENSEI, "
            "lifecycle: active, domains: [x], gateway_unit: null}\n"
        ),
        "missing_schema_version": (
            "root: {name: KENSEI}\n"
            "profiles:\n- {name: octacon, kind: worker, parent: KENSEI, "
            "lifecycle: active, domains: [x], gateway_unit: null}\n"
        ),
        "empty_profiles": (
            "schema_version: 1\nroot: {name: KENSEI}\nprofiles: []\n"
        ),
        "missing_profiles": "schema_version: 1\nroot: {name: KENSEI}\n",
        "invalid_root_structure": (
            "schema_version: 1\nroot: KENSEI\n"
            "profiles:\n- {name: octacon, kind: worker, parent: KENSEI, "
            "lifecycle: active, domains: [x], gateway_unit: null}\n"
        ),
        "root_lacks_valid_name": (
            "schema_version: 1\nroot: {description: no name}\n"
            "profiles:\n- {name: octacon, kind: worker, parent: KENSEI, "
            "lifecycle: active, domains: [x], gateway_unit: null}\n"
        ),
    }

    @pytest.mark.parametrize("shape_name", sorted(_R41_SHAPES))
    def test_r41_existing_invalid_shape_global_invalid(self, dash, shape_name):
        """R4-1: every existing-invalid registry shape → global
        invalid_registry for BOTH a listed and an omitted fs profile."""
        home, profile_docs = dash
        self._mk_profiles(home)
        (home / "governance" / "profile-registry.yaml").write_text(
            self._R41_SHAPES[shape_name], encoding="utf-8"
        )
        h = profile_docs.profile_hierarchy()
        st = {n["name"]: n.get("registry_state") for n in h["nodes"] if n["type"] != "root"}
        assert h["root"] == "KENSEI"
        assert st == {"octacon": "invalid_registry", "wesker": "invalid_registry"}

    def test_r41_unreadable_registry_global_invalid(self, dash):
        """R4-1: OSError on read → existing-invalid, not absent."""
        home, profile_docs = dash
        self._mk_profiles(home)
        reg = home / "governance" / "profile-registry.yaml"
        reg.write_text("schema_version: 1\nroot: {name: KENSEI}\n", encoding="utf-8")
        os.chmod(reg, 0o000)
        try:
            h = profile_docs.profile_hierarchy()
        finally:
            os.chmod(reg, 0o644)
        st = {n["name"]: n["registry_state"] for n in h["nodes"] if n["type"] != "root"}
        assert st == {"octacon": "invalid_registry", "wesker": "invalid_registry"}

    def test_r41_absent_and_valid_omit_controls(self, dash):
        """R4-1 controls: absent → unregistered; valid-omit → unregistered."""
        home, profile_docs = dash
        self._mk_profiles(home)
        # absent
        h = profile_docs.profile_hierarchy()
        st = {n["name"]: n["registry_state"] for n in h["nodes"] if n["type"] != "root"}
        assert st == {"octacon": "unregistered", "wesker": "unregistered"}
        # valid registry omitting wesker
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: lead, parent: KENSEI, lifecycle: active, "
            "domains: [coding], gateway_unit: hermes-gateway-octacon}\n",
            encoding="utf-8",
        )
        h = profile_docs.profile_hierarchy()
        st = {n["name"]: n["registry_state"] for n in h["nodes"] if n["type"] != "root"}
        assert st["octacon"] == "registered"
        assert st["wesker"] == "unregistered"

class TestR29ForceMode:
    def test_non_boolean_force_mode_rejected(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        assert he.emit_requirement_on_review(conn, tid, rid, force_mode="yes") is None
        assert he.emit_requirement_on_review(conn, tid, rid, force_mode=1) is None

    def test_hook_honours_explicit_true_with_config_off(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: False)
        tid = _task(conn)
        rid = _review(conn, tid)
        he.hook_request_review(conn, tid, rid, force_mode=True)
        events = [e for e in kb.list_events(conn, tid) if e.kind == he.KIND_REQUIRED]
        assert len(events) == 1  # explicit True honoured despite config off

    def test_hook_explicit_false_overrides_config_on(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: True)
        tid = _task(conn)
        rid = _review(conn, tid)
        he.hook_request_review(conn, tid, rid, force_mode=False)
        events = [e for e in kb.list_events(conn, tid) if e.kind == he.KIND_REQUIRED]
        assert events == []


# ── R2-10 ────────────────────────────────────────────────────────────────────

class TestR210FindingRecurrence:
    def test_many_updates_one_finding_is_one_open(self, fake_home_r2):
        """RED: one finding updated 3 times must count as ONE open finding."""
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        for i in range(3):
            append_event(source="t", event_type="governance.finding.opened" if i == 0
                         else "governance.finding.updated",
                         event_id=f"r210a-{i}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="octacon",
                         object_id="finding-1", occurred_at=now - 100 + i)
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] == 1

    def test_opened_before_window_resolved_inside(self, fake_home_r2):
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        # opened 10 days ago, resolved today
        append_event(source="t", event_type="governance.finding.opened",
                     event_id=f"r210b-open-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 10 * 86400)
        append_event(source="t", event_type="governance.finding.resolved",
                     event_id=f"r210b-res-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 7 * 86400)
        # finding's current state is resolved → not open
        assert dim["evidence"]["governance_findings_open"] == 0

    def test_resolved_then_reopened_is_recurring(self, fake_home_r2):
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        append_event(source="t", event_type="governance.finding.opened",
                     event_id=f"r210c-o1-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 300)
        append_event(source="t", event_type="governance.finding.resolved",
                     event_id=f"r210c-r1-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 200)
        append_event(source="t", event_type="governance.finding.opened",
                     event_id=f"r210c-o2-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 86400)
        # one identity currently open, but resolved+reopened → recurrence
        assert dim["evidence"]["governance_findings_open"] == 1
        assert dim["evidence"]["recurring_findings"] is True
        assert dim["verdict"] == "ATTENTION"

    def test_two_distinct_open_findings(self, fake_home_r2):
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        for fid in ("finding-1", "finding-2"):
            append_event(source="t", event_type="governance.finding.opened",
                         event_id=f"r210d-{fid}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="octacon",
                         object_id=fid, occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] == 2
        assert sorted(dim["evidence"]["governance_findings_open_ids"]) == ["finding-1", "finding-2"]

    def test_duplicate_replay_idempotent(self, fake_home_r2):
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        # same event_id replayed (idempotent ledger) — must not inflate counts
        for _ in range(2):
            append_event(source="t", event_type="governance.finding.opened",
                         event_id="r210e-dup", actor_profile="octacon",
                         target_profile="octacon", object_id="finding-1",
                         occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] == 1

    @staticmethod
    def _load():
        import importlib.util
        # R3-5: derive the core checkout from this test file's own location
        # (the worktree root is two directories up from tests/hermes_cli/).
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "drc_r210", str(root / "scripts" / "denji-review-cycle.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


@pytest.fixture
def fake_home_r2(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "profiles" / "octacon").mkdir(parents=True)
    (h / "governance").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


# ── R3-2: recurrence + equal-timestamp ordering (RED witnesses) ──────────────

class TestR32RecurrenceAndTies:
    @staticmethod
    def _load():
        import importlib.util
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "drc_r32", str(root / "scripts" / "denji-review-cycle.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_two_distinct_open_findings_are_not_recurring(self, fake_home_r2):
        """R3-2: two UNRELATED open findings must NOT count as recurrence.

        Recurrence means a single finding was resolved/dismissed and later
        reopened — not "more than one open finding".  Two freshly-opened,
        never-closed findings → open=2, recurring=False → WATCH (not the
        recurrence-driven ATTENTION).
        """
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        for fid in ("finding-1", "finding-2"):
            append_event(source="t", event_type="governance.finding.opened",
                         event_id=f"r32a-{fid}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="octacon",
                         object_id=fid, occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 86400)
        ev = dim["evidence"]
        assert ev["governance_findings_open"] == 2
        assert ev["recurring_findings"] is False
        # Recurrence is what escalates to ATTENTION; two fresh findings stay WATCH.
        assert dim["verdict"] == "WATCH"

    def test_same_second_opened_resolved_is_closed(self, fake_home_r2):
        """R3-2: opened then resolved at the SAME second → latest (higher row id)
        is resolved → the finding is closed (not open)."""
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        now = 1_000_000  # fixed same-second timestamp for both events
        append_event(source="t", event_type="governance.finding.opened",
                     event_id="r32b-open", actor_profile="octacon",
                     target_profile="octacon", object_id="finding-1",
                     occurred_at=now)
        append_event(source="t", event_type="governance.finding.resolved",
                     event_id="r32b-res", actor_profile="octacon",
                     target_profile="octacon", object_id="finding-1",
                     occurred_at=now)
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] == 0
        assert dim["evidence"]["recurring_findings"] is False

    def test_same_second_resolved_then_opened_is_recurring(self, fake_home_r2):
        """R3-2: resolved then re-opened at the SAME second → the later
        (higher row id) event is the open, and the identity has a closed
        history → open AND recurring."""
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        now = 1_000_000
        append_event(source="t", event_type="governance.finding.resolved",
                     event_id="r32c-res", actor_profile="octacon",
                     target_profile="octacon", object_id="finding-1",
                     occurred_at=now)
        append_event(source="t", event_type="governance.finding.opened",
                     event_id="r32c-open", actor_profile="octacon",
                     target_profile="octacon", object_id="finding-1",
                     occurred_at=now)
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] == 1
        assert dim["evidence"]["recurring_findings"] is True
        assert dim["verdict"] == "ATTENTION"