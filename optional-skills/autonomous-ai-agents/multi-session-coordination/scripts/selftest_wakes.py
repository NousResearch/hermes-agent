#!/usr/bin/env python3
"""End-to-end CLI regression tests for durable wake episodes."""

import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CLI = HERE / "session_coord.py"
EVENT_KEYS = {
    "event_id",
    "attempt",
    "episode_id",
    "board_session_id",
    "native_session_id",
    "profile_home",
    "transport",
    "target",
    "checkpoint_path",
    "reason",
    "verify_required",
    "prompt",
    "episode_created_at",
    "created_at",
}


class WakeCLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "board.db"
        self.profile = self.root / "profile"
        self.profile.mkdir()
        self.env = os.environ.copy()
        self.env.update(
            {
                "HERMES_COORD_DB": str(self.db_path),
                "HERMES_COORD_DISABLED_FILE": str(self.root / "disabled"),
                "HERMES_COORD_DISABLED": "0",
                "HERMES_COORD_CRON_MANIFEST": str(self.root / "cron.json"),
                "HERMES_COORD_CRON_JOBS": str(self.root / "jobs.json"),
                "HERMES_COORD_PROFILES_DIR": str(self.root / "profiles"),
            }
        )
        self.env.pop("HERMES_SESSION_ID", None)
        self.env.pop("HERMES_HOME", None)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args, rc=0, json_output=True) -> Any:
        argv = ["python3", str(CLI), *args]
        if json_output and "--json" not in args:
            argv.append("--json")
        proc = subprocess.run(argv, env=self.env, text=True, capture_output=True)
        self.assertEqual(
            proc.returncode,
            rc,
            msg=f"command: {' '.join(argv)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        if not json_output:
            return proc
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, msg=f"no JSON output\nstderr:\n{proc.stderr}")
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            self.fail(f"bad JSON: {exc}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    def register(self, sid, *, rank=None, parent=None, target_kind="desktop",
                 native=None, target_session=None):
        native = native or f"native-{sid}"
        target_session = target_session or native
        target = {
            "profile_home": str(self.profile),
            "kind": target_kind,
            "session_id": target_session,
        }
        if parent:
            target["parent_session_id"] = target_session
        args = [
            "register",
            "--id",
            sid,
            "--task",
            f"task {sid}",
            "--surface",
            "subagent" if parent else target_kind,
            "--native-session-id",
            native,
            "--wake-transport",
            "hermes",
            "--wake-target-json",
            json.dumps(target, sort_keys=True),
        ]
        if rank:
            args += ["--rank", rank]
        if parent:
            args += ["--parent", parent, "--slot", "a"]
        return self.run_cli(*args)

    def checkpoint(self, name):
        path = self.root / name
        path.write_text(f"checkpoint {name}\n", encoding="utf-8")
        return path

    def receipt(self, event):
        return {
            "event_id": event["event_id"],
            "native_session_id": event["native_session_id"],
            "profile_home": event["profile_home"],
            "admitted": True,
            "durable": True,
        }

    def disposition_receipt(self, event, outcome):
        receipt = {
            "event_id": event["event_id"],
            "native_session_id": event["native_session_id"],
            "profile_home": event["profile_home"],
            "status": outcome,
            "payload_hash": "selftest-durable-readback",
            "payload_sha256": "selftest-durable-readback",
        }
        if outcome == "not_sent":
            receipt.update(not_sent=True, submitted=False)
        return receipt

    def yield_claim(self, sid, resources, checkpoint=None, rc=75):
        checkpoint = checkpoint or self.checkpoint(f"{sid}.json")
        args = ["claim", "--id", sid]
        for resource in resources:
            args += ["--res", resource]
        args += ["--yield", "--checkpoint", str(checkpoint)]
        return self.run_cli(*args, rc=rc), checkpoint

    def lease(self, sid, *, kind="desktop", worker="worker-1"):
        return self.run_cli(
            "wake-lease",
            "--worker",
            worker,
            "--transport",
            "hermes",
            "--profile-home",
            str(self.profile),
            "--native-session-id",
            f"native-{sid}",
            "--kind",
            kind,
        )["event"]

    def deliver(self, event):
        return self.run_cli(
            "wake-result",
            "--event",
            event["event_id"],
            "--attempt",
            str(event["attempt"]),
            "--outcome",
            "delivered",
            "--receipt-json",
            json.dumps(self.receipt(event), sort_keys=True),
        )

    def pending_events(self):
        payload = self.run_cli("wake-pending", "--profile-home", str(self.profile))
        return [item["event"] for item in payload["events"]]

    def test_release_enqueues_exact_target_and_continue_claims_full_request(self):
        self.register("holder")
        self.register("waiter")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        checkpoint = self.checkpoint("waiter.json")

        yielded = self.run_cli(
            "claim",
            "--id",
            "waiter",
            "--res",
            "res:A",
            "--yield",
            "--checkpoint",
            str(checkpoint),
            rc=75,
        )
        self.assertTrue(yielded["yielded"])
        self.assertIn("STOP", yielded["instruction"])
        self.assertEqual(yielded["checkpoint"], str(checkpoint))
        pending = self.run_cli("wake-pending", "--profile-home", str(self.profile))
        self.assertEqual(pending["events"], [])

        self.run_cli("release", "--id", "holder", "--res", "res:A")
        pending = self.run_cli("wake-pending", "--profile-home", str(self.profile))
        self.assertEqual(len(pending["events"]), 1)

        leased = self.run_cli(
            "wake-lease",
            "--worker",
            "worker-1",
            "--transport",
            "hermes",
            "--profile-home",
            str(self.profile),
            "--native-session-id",
            "native-waiter",
            "--kind",
            "desktop",
        )["event"]
        self.assertEqual(set(leased), EVENT_KEYS)
        self.assertEqual(leased["target"]["session_id"], "native-waiter")
        self.assertIn(leased["event_id"], leased["prompt"])
        self.assertIn(str(checkpoint), leased["prompt"])
        self.assertIn("continue --id waiter --event", leased["prompt"])

        result = self.run_cli(
            "wake-result",
            "--event",
            leased["event_id"],
            "--attempt",
            str(leased["attempt"]),
            "--outcome",
            "delivered",
            "--receipt-json",
            json.dumps(self.receipt(leased), sort_keys=True),
        )
        self.assertEqual(result["status"], "delivered")

        continued = self.run_cli(
            "continue",
            "--id",
            "waiter",
            "--event",
            leased["event_id"],
        )
        self.assertTrue(continued["acquired"])
        self.assertEqual(continued["claimed"], ["res:A"])
        self.assertEqual(continued["checkpoint"], str(checkpoint))

        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            held = conn.execute(
                "SELECT resource FROM claims WHERE session_id='waiter' AND status='held'"
            ).fetchall()
        self.assertEqual(held, [("res:A",)])

    def test_partial_release_then_partial_ttl_expiry_wakes_only_full_set(self):
        self.register("hold-a")
        self.register("hold-b")
        self.register("wait-ab")
        self.run_cli("claim", "--id", "hold-a", "--res", "res:A")
        self.run_cli("claim", "--id", "hold-b", "--res", "res:B")
        yielded, checkpoint = self.yield_claim("wait-ab", ["res:A", "res:B"])

        self.run_cli("release", "--id", "hold-a", "--res", "res:A")
        pending = self.run_cli("wake-pending", "--profile-home", str(self.profile))
        self.assertEqual(pending["events"], [])
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            partial = conn.execute(
                "SELECT resource FROM claims WHERE session_id='wait-ab' AND status='held'"
            ).fetchall()
            conn.execute(
                "UPDATE claims SET claimed_at=0 WHERE session_id='hold-b' AND status='held'"
            )
        self.assertEqual(partial, [])

        reconciled = self.run_cli("wake-reconcile")
        self.assertEqual(reconciled["enqueued"], 1)
        event = self.lease("wait-ab")
        self.assertEqual(event["episode_id"], yielded["episode_id"])
        self.assertEqual(event["reason"], "ttl_expired")
        self.assertTrue(event["verify_required"])
        self.assertIn("TTL does not prove", event["prompt"])
        self.deliver(event)
        continued = self.run_cli(
            "continue", "--id", "wait-ab", "--event", event["event_id"]
        )
        self.assertEqual(continued["claimed"], ["res:A", "res:B"])
        self.assertEqual(continued["checkpoint"], str(checkpoint))
        self.assertTrue(continued["verify_required"])
        self.assertEqual(
            {item["resource"] for item in continued["verification"]}, {"res:B"}
        )

    def test_rank_then_fifo_fair_wake_and_durable_fencing(self):
        self.register("holder")
        self.register("low-first", rank="2")
        self.register("high-late", rank="1")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        self.yield_claim("low-first", ["res:A"])
        self.yield_claim("high-late", ["res:A"])
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("UPDATE waiters SET last_poll=0 WHERE episode_id IS NOT NULL")
        self.run_cli("release", "--id", "holder", "--res", "res:A")

        self.assertIsNone(self.lease("low-first"))
        high = self.lease("high-late")
        self.assertIsNotNone(high)
        self.deliver(high)
        self.run_cli("continue", "--id", "high-late", "--event", high["event_id"])
        self.run_cli("done", "--id", "high-late")
        low = self.lease("low-first")
        self.assertIsNotNone(low)

        # Equal ranks use durable waiter creation time as FIFO.
        self.register("holder2")
        self.register("fifo-one")
        self.register("fifo-two")
        self.run_cli("claim", "--id", "holder2", "--res", "res:B")
        self.yield_claim("fifo-one", ["res:B"])
        self.yield_claim("fifo-two", ["res:B"])
        self.run_cli("release", "--id", "holder2", "--res", "res:B")
        self.assertIsNone(self.lease("fifo-two"))
        self.assertIsNotNone(self.lease("fifo-one"))

    def test_changed_request_supersedes_and_explicit_cancel_tombstones_event(self):
        self.register("holder")
        self.register("waiter")
        self.run_cli("claim", "--id", "holder", "--res", "res:A", "--res", "res:B")
        checkpoint = self.checkpoint("changed.json")
        first, _ = self.yield_claim("waiter", ["res:A"], checkpoint)
        same, _ = self.yield_claim("waiter", ["res:A"], checkpoint)
        self.assertEqual(same["episode_id"], first["episode_id"])
        self.assertTrue(same["reused"])
        changed, _ = self.yield_claim("waiter", ["res:A", "res:B"], checkpoint)
        self.assertNotEqual(changed["episode_id"], first["episode_id"])
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            old_status = conn.execute(
                "SELECT status FROM wait_episodes WHERE episode_id=?",
                (first["episode_id"],),
            ).fetchone()[0]
        self.assertEqual(old_status, "superseded")

        self.run_cli("release", "--id", "holder", "--all")
        event = self.pending_events()[0]
        canceled = self.run_cli(
            "cancel-wait",
            "--id",
            "waiter",
            "--episode",
            changed["episode_id"],
            "--reason",
            "user pressed Stop",
        )
        self.assertEqual(canceled["canceled"], [changed["episode_id"]])
        self.assertIsNone(self.lease("waiter"))
        rejected = self.run_cli(
            "continue", "--id", "waiter", "--event", event["event_id"], rc=75
        )
        self.assertTrue(rejected["stale"])

    def test_missing_identity_and_bad_checkpoint_are_honest_blockers(self):
        self.run_cli("register", "--id", "holder", "--task", "holder")
        self.run_cli("register", "--id", "legacy", "--task", "legacy")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        checkpoint = self.checkpoint("legacy.json")
        blocked, _ = self.yield_claim("legacy", ["res:A"], checkpoint)
        self.assertFalse(blocked["yielded"])
        self.assertIn("unavailable", blocked["error"])
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM wait_episodes").fetchone()[0], 0)
        bad = self.run_cli(
            "claim", "--id", "legacy", "--res", "res:A", "--yield",
            "--checkpoint", "relative.json", rc=1, json_output=False
        )
        self.assertIn("absolute", bad.stderr)

    def test_concurrent_reconcile_repairs_one_missing_event(self):
        self.register("holder")
        self.register("waiter")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        self.yield_claim("waiter", ["res:A"])
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DELETE FROM wake_outbox")

        def reconcile_once(_):
            proc = subprocess.run(
                ["python3", str(CLI), "wake-reconcile", "--json"],
                env=self.env,
                text=True,
                capture_output=True,
            )
            return proc.returncode, json.loads(proc.stdout)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(reconcile_once, range(8)))
        self.assertTrue(all(rc == 0 for rc, _ in results), results)
        self.assertEqual(sum(item["enqueued"] for _, item in results), 1)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            rows = conn.execute(
                "SELECT episode_id,wake_seq,status FROM wake_outbox"
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1:], (2, "pending"))

    def test_lost_ack_unknown_receipt_and_stale_attempt_do_not_double_send(self):
        self.register("holder")
        self.register("waiter")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        self.yield_claim("waiter", ["res:A"])
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        first = self.lease("waiter")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "UPDATE wake_outbox SET lease_expires_at=0 WHERE event_id=?",
                (first["event_id"],),
            )
        reconciled = self.run_cli("wake-reconcile")
        self.assertEqual(reconciled["unknown_leases"], 1)
        self.assertIsNone(self.lease("waiter"))
        recovered = self.deliver(first)
        self.assertEqual(recovered["status"], "delivered")

        # A proven not-sent attempt can be leased again; the old attempt is fenced.
        self.run_cli("cancel-wait", "--id", "waiter", "--reason", "new attempt fixture")
        self.register("wait2")
        self.run_cli("claim", "--id", "holder", "--res", "res:B")
        self.yield_claim("wait2", ["res:B"])
        self.run_cli("release", "--id", "holder", "--res", "res:B")
        attempt1 = self.lease("wait2")
        not_sent = self.run_cli(
            "wake-result", "--event", attempt1["event_id"], "--attempt", "1",
            "--outcome", "not_sent", "--receipt-json",
            json.dumps(self.disposition_receipt(attempt1, "not_sent"), sort_keys=True)
        )
        self.assertEqual(not_sent["status"], "pending")
        attempt2 = self.lease("wait2", worker="worker-2")
        self.assertEqual(attempt2["attempt"], 2)
        stale = self.run_cli(
            "wake-result", "--event", attempt1["event_id"], "--attempt", "1",
            "--outcome", "unknown", "--receipt-json", "{}", rc=75
        )
        self.assertTrue(stale["stale"])
        unknown = self.run_cli(
            "wake-result", "--event", attempt2["event_id"], "--attempt", "2",
            "--outcome", "unknown", "--receipt-json", "{}"
        )
        self.assertEqual(unknown["status"], "unknown")
        self.assertIsNone(self.lease("wait2", worker="worker-3"))
        self.assertEqual(self.deliver(attempt2)["status"], "delivered")

    def test_priority_change_before_lease_and_after_admission_refences(self):
        self.register("holder")
        self.register("was-first", rank="1")
        self.register("now-first", rank="2")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        self.yield_claim("was-first", ["res:A"])
        self.yield_claim("now-first", ["res:A"])
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        self.run_cli("prioritize", "--session", "now-first", "--rank", "0")
        self.assertIsNone(self.lease("was-first"))
        new_head = self.lease("now-first")
        self.assertIsNotNone(new_head)

        # Put was-first back at the head, admit it, then move now-first ahead.
        self.run_cli("wake-result", "--event", new_head["event_id"], "--attempt", "1",
                     "--outcome", "not_sent", "--receipt-json",
                     json.dumps(self.disposition_receipt(new_head, "not_sent"), sort_keys=True))
        self.run_cli("prioritize", "--session", "was-first", "--rank", "0")
        admitted = self.lease("was-first")
        self.deliver(admitted)
        self.run_cli("prioritize", "--session", "now-first", "--rank", "0")
        self.run_cli("prioritize", "--session", "was-first", "--rank", "2")
        rewait = self.run_cli(
            "continue", "--id", "was-first", "--event", admitted["event_id"], rc=75
        )
        self.assertTrue(rewait["yielded"])
        self.assertIsNotNone(self.lease("now-first"))

    def test_off_freezes_delivery_then_reenable_revalidates(self):
        self.register("holder")
        self.register("waiter")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        self.yield_claim("waiter", ["res:A"])
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        self.env["HERMES_COORD_DISABLED"] = "1"
        self.assertEqual(
            self.run_cli("wake-pending", "--profile-home", str(self.profile))["events"], []
        )
        self.assertIsNone(self.lease("waiter"))
        frozen = self.run_cli("wake-reconcile")
        self.assertTrue(frozen["disabled"])
        self.env["HERMES_COORD_DISABLED"] = "0"
        self.assertIsNotNone(self.lease("waiter"))

    def test_v1_migration_does_not_backfill_legacy_waiter(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE sessions(id TEXT PRIMARY KEY,task TEXT,surface TEXT,
                  started_at REAL,last_seen REAL,status TEXT DEFAULT 'active');
                CREATE TABLE claims(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,
                  resource TEXT,mode TEXT,task TEXT,claimed_at REAL,ttl_min REAL,
                  released_at REAL,status TEXT DEFAULT 'held');
                CREATE TABLE waiters(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,
                  resource TEXT,since REAL,note TEXT,active INTEGER DEFAULT 1);
                CREATE TABLE notifications(id INTEGER PRIMARY KEY AUTOINCREMENT,
                  to_session TEXT,from_session TEXT,resource TEXT,kind TEXT,body TEXT,
                  created_at REAL,read_at REAL);
                CREATE TABLE cron_events(id INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT,
                  job_name TEXT,event TEXT,session_id TEXT,reason TEXT,created_at REAL);
                INSERT INTO sessions VALUES('legacy','old wait','cli',1,1,'active');
                INSERT INTO waiters(session_id,resource,since,note,active)
                  VALUES('legacy','res:A',1,'legacy polling waiter',1);
                """
            )
        self.run_cli("wake-reconcile")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            waiter = conn.execute(
                "SELECT episode_id FROM waiters WHERE session_id='legacy'"
            ).fetchone()
            episodes = conn.execute("SELECT COUNT(*) FROM wait_episodes").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM wake_outbox").fetchone()[0]
        self.assertIsNone(waiter[0])
        self.assertEqual((episodes, events), (0, 0))

    def test_pause_yield_snapshots_full_parked_set(self):
        self.register("pauser")
        self.run_cli("claim", "--id", "pauser", "--res", "res:A", "--res", "res:B",
                     "--ttl", "17")
        checkpoint = self.checkpoint("pause.json")
        paused = self.run_cli(
            "pause", "--id", "pauser", "--yield", "--checkpoint", str(checkpoint), rc=75
        )
        self.assertEqual(paused["paused"], ["res:A", "res:B"])
        event = self.lease("pauser")
        self.deliver(event)
        continued = self.run_cli(
            "continue", "--id", "pauser", "--event", event["event_id"]
        )
        self.assertEqual(continued["claimed"], ["res:A", "res:B"])
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            ttl_values = conn.execute(
                "SELECT DISTINCT ttl_min FROM claims WHERE session_id='pauser' "
                "AND status='held'"
            ).fetchall()
        self.assertEqual(ttl_values, [(17.0,)])

    def test_child_event_targets_parent_immutably_with_redispatch_metadata(self):
        self.register("parent")
        self.register("holder")
        old_target = {
            "profile_home": str(self.profile),
            "kind": "subagent_parent",
            "session_id": "native-parent",
            "parent_session_id": "native-parent",
            "child_checkpoint_identity": "child-slot-a",
        }
        self.run_cli(
            "register", "--id", "child", "--task", "child task", "--surface", "subagent",
            "--parent", "parent", "--slot", "a", "--native-session-id", "native-child",
            "--wake-transport", "hermes", "--wake-target-json",
            json.dumps(old_target, sort_keys=True)
        )
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        yielded, _checkpoint = self.yield_claim("child", ["res:A"])
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            frozen_target = json.loads(conn.execute(
                "SELECT target_json FROM wait_episodes WHERE episode_id=?",
                (yielded["episode_id"],),
            ).fetchone()[0])
        changed_target = dict(old_target)
        changed_target["session_id"] = "native-other-parent"
        changed_target["parent_session_id"] = "native-other-parent"
        self.run_cli(
            "register", "--id", "child", "--task", "child task", "--surface", "subagent",
            "--parent", "parent", "--slot", "a", "--native-session-id", "native-child",
            "--wake-transport", "hermes", "--wake-target-json",
            json.dumps(changed_target, sort_keys=True)
        )
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        event = self.run_cli(
            "wake-lease", "--worker", "parent-worker", "--transport", "hermes",
            "--profile-home", str(self.profile), "--native-session-id", "native-parent",
            "--kind", "subagent_parent"
        )["event"]
        self.assertIsNotNone(event)
        self.assertEqual(event["native_session_id"], "native-parent")
        self.assertEqual(event["target"], frozen_target)
        self.assertEqual(event["target"]["compression_lineage"], [old_target["session_id"]])
        self.assertIn("redispatch", event["prompt"])
        self.assertIn("child", event["prompt"])

    def test_native_child_stop_cancels_parent_targeted_episode(self):
        self.register("parent")
        self.register("holder")
        target = {
            "profile_home": str(self.profile),
            "kind": "subagent_parent",
            "session_id": "native-parent",
            "parent_session_id": "native-parent",
            "child_checkpoint_identity": "child-slot-a",
        }
        self.run_cli(
            "register", "--id", "child", "--task", "child task", "--surface", "subagent",
            "--parent", "parent", "--slot", "a", "--native-session-id", "native-child",
            "--wake-transport", "hermes", "--wake-target-json", json.dumps(target)
        )
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        yielded, _ = self.yield_claim("child", ["res:A"])
        canceled = self.run_cli(
            "cancel-wait", "--native-session-id", "native-child",
            "--profile-home", str(self.profile), "--reason", "native Stop lifecycle"
        )
        self.assertEqual(canceled["canceled"], [yielded["episode_id"]])
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        self.assertEqual(self.pending_events(), [])

    def test_manual_resume_of_yielded_pause_cancels_undelivered_wake(self):
        self.register("pauser")
        self.run_cli("claim", "--id", "pauser", "--res", "res:A")
        checkpoint = self.checkpoint("manual-resume.json")
        self.run_cli(
            "pause", "--id", "pauser", "--yield", "--checkpoint", str(checkpoint), rc=75
        )
        resumed = self.run_cli("resume", "--id", "pauser")
        self.assertEqual(resumed["claimed"], ["res:A"])
        self.assertIsNone(self.lease("pauser"))
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            status = conn.execute(
                "SELECT status FROM wait_episodes WHERE session_id='pauser'"
            ).fetchone()[0]
        self.assertEqual(status, "canceled")

    def test_receiver_canceled_outcome_tombstones_episode(self):
        self.register("holder")
        self.register("waiter")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        self.yield_claim("waiter", ["res:A"])
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        event = self.lease("waiter")
        canceled = self.run_cli(
            "wake-result", "--event", event["event_id"], "--attempt", "1",
            "--outcome", "canceled", "--receipt-json",
            json.dumps(self.disposition_receipt(event, "canceled"), sort_keys=True)
        )
        self.assertEqual(canceled["status"], "canceled")
        self.run_cli("wake-reconcile")
        self.assertIsNone(self.lease("waiter"))
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            status = conn.execute(
                "SELECT status FROM wait_episodes WHERE session_id='waiter'"
            ).fetchone()[0]
        self.assertEqual(status, "canceled")

    def test_changed_yield_request_that_claims_immediately_supersedes_old_episode(self):
        self.register("holder")
        self.register("waiter")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        checkpoint = self.checkpoint("changed-immediate.json")
        first, _ = self.yield_claim("waiter", ["res:A"], checkpoint)
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        acquired, _ = self.yield_claim("waiter", ["res:B"], checkpoint, rc=0)
        self.assertEqual(acquired["claimed"], ["res:B"])
        self.assertIsNone(self.lease("waiter"))
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            status = conn.execute(
                "SELECT status FROM wait_episodes WHERE episode_id=?",
                (first["episode_id"],),
            ).fetchone()[0]
        self.assertEqual(status, "superseded")

    def test_register_auto_captures_trustworthy_hermes_environment(self):
        self.register("holder")
        self.env["HERMES_SESSION_ID"] = "native-auto"
        self.env["HERMES_HOME"] = str(self.profile)
        registered = self.run_cli(
            "register", "--id", "auto-wait", "--task", "auto", "--surface", "desktop"
        )
        self.assertEqual(registered["native_session_id"], "native-auto")
        self.run_cli("claim", "--id", "holder", "--res", "res:A")
        self.yield_claim("auto-wait", ["res:A"])
        self.run_cli("release", "--id", "holder", "--res", "res:A")
        event = self.run_cli(
            "wake-lease", "--worker", "auto-worker", "--transport", "hermes",
            "--profile-home", str(self.profile), "--native-session-id", "native-auto",
            "--kind", "desktop"
        )["event"]
        self.assertIsNotNone(event)

    def test_v2_migration_does_not_backfill_legacy_waiter(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE sessions(id TEXT PRIMARY KEY,task TEXT,surface TEXT,
                  started_at REAL,last_seen REAL,status TEXT DEFAULT 'active',priority TEXT,
                  parent_id TEXT,slot TEXT,paused INTEGER DEFAULT 0,checkpoint_note TEXT,
                  rank_set_at REAL);
                CREATE TABLE claims(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,
                  resource TEXT,mode TEXT,task TEXT,claimed_at REAL,ttl_min REAL,
                  released_at REAL,status TEXT DEFAULT 'held',paused_at REAL);
                CREATE TABLE waiters(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,
                  resource TEXT,since REAL,note TEXT,active INTEGER DEFAULT 1,mode TEXT,
                  last_poll REAL);
                CREATE TABLE notifications(id INTEGER PRIMARY KEY AUTOINCREMENT,
                  to_session TEXT,from_session TEXT,resource TEXT,kind TEXT,body TEXT,
                  created_at REAL,read_at REAL);
                CREATE TABLE cron_events(id INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT,
                  job_name TEXT,event TEXT,session_id TEXT,reason TEXT,created_at REAL);
                INSERT INTO sessions(id,task,surface,started_at,last_seen,status)
                  VALUES('legacy','v2 wait','cli',1,1,'active');
                INSERT INTO waiters(session_id,resource,since,note,active,mode,last_poll)
                  VALUES('legacy','res:A',1,'legacy v2 waiter',1,'exclusive',1);
                """
            )
        self.run_cli("wake-reconcile")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            waiter = conn.execute(
                "SELECT episode_id FROM waiters WHERE session_id='legacy'"
            ).fetchone()
            counts = (
                conn.execute("SELECT COUNT(*) FROM wait_episodes").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM wake_outbox").fetchone()[0],
            )
        self.assertIsNone(waiter[0])
        self.assertEqual(counts, (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
