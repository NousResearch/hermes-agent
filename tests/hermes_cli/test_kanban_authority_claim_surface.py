"""C1/C2/C3 regressions for the authority claim path.

Added by Claude Code Windows session 82b5e4d1-0800-4482-baf7-160b11f66249 on
2026-09-11, under Siba's "Ok finish" correction authorisation, after the Mac
review NO-GO on 1cbddef5. Written RED-first: every test here must FAIL on
1cbddef5 and pass only on the correction.

C1  The minted bearer must not reach ANY public surface. The enumeration is
    mechanical, not remembered: the test seeds a distinctive capability, drives
    the real claim paths, then sweeps every stored event payload and the
    dashboard-shaped projections for the token. A previous round asserted
    "absent from every public field" after checking two places it thought of,
    and missed the event payload entirely, which is what this sweep exists to
    prevent recurring.
C2  An unpredictable token must still satisfy the hostname-prefix locality
    contract, or a LIVE worker whose TTL lapsed is reclaimed out from under
    itself instead of extended.
C3  Review claims must use the same secure mint-and-bind path as normal claims.
"""
import json
import os
import sqlite3
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_dispatch import _set_worker_pid as _kbd_set_worker_pid


class AuthorityClaimSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "k.db"
        kb._INITIALIZED_PATHS.discard(str(self.db.resolve()))
        self.conn = kb.connect(db_path=self.db, board="c123")
        self.addCleanup(self.conn.close)

    def _enrolled_task(self, title="t"):
        task_id = kb.create_task(self.conn, title=title, board="c123")
        if kb.authority_history_capability(self.conn) is None:
            kb.enroll_authority_history(self.conn)
        kb.bind_authority_task(self.conn, task_id, repo_id="r", project_id="p")
        return task_id

    def _all_event_payloads(self):
        return [row[0] for row in self.conn.execute(
            "SELECT payload FROM task_events WHERE payload IS NOT NULL")]

    # ---- C1 -------------------------------------------------------------
    def test_c1_minted_bearer_never_appears_in_any_stored_event_payload(self):
        task_id = self._enrolled_task("c1")
        task = kb.claim_task(self.conn, task_id, ttl_seconds=60)
        self.assertIsNotNone(task, "fixture: the claim must succeed")
        bearer = task.claim_lock
        self.assertTrue(bearer, "fixture: a capability must have been issued")
        leaked = [p for p in self._all_event_payloads() if bearer in p]
        self.assertEqual(
            leaked, [],
            "the claim capability is published in an event payload, so anyone who "
            "can read the board holds it: " + repr(leaked))

    def test_c1_public_run_projection_carries_no_bearer(self):
        """The dashboard publishes a run dict over HTTP; it must not carry the bearer.

        Deliberately NOT asserting that tasks.claim_lock differs from the bearer:
        that column IS the fencing token, and 'fixing' it there would break
        heartbeat and reclaim. The defect is PUBLICATION, so the test targets the
        projection the dashboard actually returns.
        """
        task_id = self._enrolled_task("c1b")
        task = kb.claim_task(self.conn, task_id, ttl_seconds=60)
        bearer = task.claim_lock
        run = self.conn.execute(
            "SELECT * FROM task_runs WHERE task_id=?", (task_id,)).fetchone()
        published = kb.public_run_fields(run)
        leaked = [k for k, v in published.items() if isinstance(v, str) and bearer in v]
        self.assertEqual(
            leaked, [],
            "the run projection the dashboard returns exposes the capability in "
            + repr(leaked))

    # ---- C2 -------------------------------------------------------------
    def test_c2_live_local_worker_with_lapsed_ttl_is_extended_not_reclaimed(self):
        enrolled = self._enrolled_task("c2-enrolled")
        plain = kb.create_task(self.conn, title="c2-plain", board="c123")
        got_e = kb.claim_task(self.conn, enrolled, ttl_seconds=1)
        got_p = kb.claim_task(self.conn, plain, ttl_seconds=1)
        self.assertIsNotNone(got_e)
        self.assertIsNotNone(got_p)
        # Let the TTL lapse naturally rather than writing ownership fields by
        # hand: a raw UPDATE of those columns inside write_txn is an unjournaled
        # transition and the audit rightly refuses it. Attach a live pid through
        # the supported path so the worker is genuinely local and alive.
        _kbd_set_worker_pid(self.conn, enrolled, os.getpid())
        _kbd_set_worker_pid(self.conn, plain, os.getpid())
        time.sleep(2.2)
        kb.release_stale_claims(self.conn)
        after_e = kb.get_task(self.conn, enrolled)
        after_p = kb.get_task(self.conn, plain)
        self.assertEqual(
            after_p.status, "running",
            "control: a live non-enrolled worker must keep its claim")
        self.assertEqual(
            after_e.status, "running",
            "a LIVE enrolled worker was reclaimed because the minted token carries "
            "no host prefix, so the locality gate could not recognise it as local")

    # ---- C3 -------------------------------------------------------------
    def _to_review(self, task_id):
        """Move a bound task into 'review', the state claim_review_task requires."""
        with kb.write_txn(self.conn):
            self.conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
            kb._append_event(self.conn, task_id, 'status', {'status': 'review'})

    def test_c3_review_claim_works_on_an_enrolled_bound_task(self):
        task_id = self._enrolled_task("c3")
        self._to_review(task_id)
        try:
            result = kb.claim_review_task(self.conn, task_id, ttl_seconds=60)
        except Exception as exc:  # noqa: BLE001 - the defect is the raise itself
            self.fail("review claim on an enrolled bound task raised "
                      f"{type(exc).__name__}: {exc}")
        self.assertIsNotNone(result, "review claim returned nothing")

    def test_c3_review_claim_does_not_publish_its_bearer_either(self):
        task_id = self._enrolled_task("c3b")
        self._to_review(task_id)
        try:
            kb.claim_review_task(self.conn, task_id, ttl_seconds=60)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"review claim raised {type(exc).__name__}: {exc}")
        row = self.conn.execute(
            "SELECT claim_lock FROM tasks WHERE id=?", (task_id,)).fetchone()
        bearer = row["claim_lock"]
        if not bearer:
            self.skipTest("no lock recorded; covered by the previous test")
        leaked = [p for p in self._all_event_payloads() if bearer in p]
        self.assertEqual(leaked, [], "review claim publishes its bearer: " + repr(leaked))


if __name__ == "__main__":
    unittest.main()

class BearerPublicationSweepTests(unittest.TestCase):
    """F1/F2 from the Mac review of 620e4fd5.

    F1 exists because fixing C2 RE-ARMED a path: restoring the host prefix made
    lock.startswith(host_prefix) true again, which made the release_stale_claims
    extension branch reachable again, which republished the bearer at a line that
    had been dead for minted tokens. The earlier C1 test read the `claimed`
    payload and the earlier C2 test read the task ROW, so the defect sat exactly
    in the seam between them. This one therefore drives claim, extension AND
    reclaim, then enumerates EVERY column of EVERY table with no filtering.

    The absence of a filter is deliberate. Three of my misses on this work came
    from a heuristic applied AFTER the enumeration; this asserts over the raw set.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "k.db"
        kb._INITIALIZED_PATHS.discard(str(self.db.resolve()))
        self.conn = kb.connect(db_path=self.db, board="f12")
        self.addCleanup(self.conn.close)

    def _bound(self, title):
        tid = kb.create_task(self.conn, title=title, board="f12")
        if kb.authority_history_capability(self.conn) is None:
            kb.enroll_authority_history(self.conn)
        kb.bind_authority_task(self.conn, tid, repo_id="r", project_id="p")
        return tid

    def _where_does_it_appear(self, needle):
        """Every column of every table. No filter, no keyword guess, no head."""
        found = set()
        for (table,) in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"):
            try:
                cols = [c[1] for c in self.conn.execute(
                    'PRAGMA table_info("%s")' % table)]
                for row in self.conn.execute('SELECT * FROM "%s"' % table):
                    for col, val in zip(cols, row):
                        if isinstance(val, str) and needle in val:
                            found.add("%s.%s" % (table, col))
            except sqlite3.Error:
                continue
        return found

    LEGITIMATE = {"tasks.claim_lock", "task_runs.claim_lock"}

    def _live_worker_pid(self):
        """A genuinely alive process that is NOT the test runner.

        reclaim_task SIGTERMs the recorded worker pid (kanban_db kill site at the
        _pid_alive/terminate path). An earlier version of this test recorded
        os.getpid(), so calling reclaim killed the test process itself and pytest
        exited 15 with no output. Use a real child instead: honest liveness,
        and the signal lands somewhere harmless.
        """
        import subprocess
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self._reap, child)
        return child.pid

    @staticmethod
    def _reap(child):
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)

    def test_f1_bearer_absent_after_claim_extension_and_reclaim(self):
        task_id = self._bound("f1")
        task = kb.claim_task(self.conn, task_id, ttl_seconds=1)
        self.assertIsNotNone(task)
        bearer = task.claim_lock
        _kbd_set_worker_pid(self.conn, task_id, self._live_worker_pid())
        time.sleep(2.2)
        kb.release_stale_claims(self.conn)          # the EXTENSION path (F1)
        kb.reclaim_task(self.conn, task_id, reason="sweep test")
        leaked = self._where_does_it_appear(bearer) - self.LEGITIMATE
        self.assertEqual(
            leaked, set(),
            "the bearer is published outside the two legitimate lock columns "
            "after extension/reclaim: " + repr(sorted(leaked)))

    def test_f2_colon_free_explicit_secret_is_never_published(self):
        secret = "private-bearer-token"          # no colon, as the repo's suites use
        self.assertNotEqual(
            kb._public_label(secret), secret,
            "public_claim_label returns a colon-free capability unchanged, so the "
            "redaction is a no-op for explicit claimers")
        task_id = self._bound("f2")
        # An explicit claimer on a BOUND task currently fails closed: no owner
        # binding exists for it, so capture() refuses and the claim rolls back.
        # Either outcome is acceptable for F2 -- what must never happen is the
        # secret reaching a published record -- so the test accepts both and
        # sweeps regardless. Reported to the reviewers rather than widened into
        # a behaviour change, which would be outside a C1-only round.
        try:
            kb.claim_task(self.conn, task_id, ttl_seconds=60, claimer=secret)
        except Exception:
            pass
        leaked = self._where_does_it_appear(secret) - self.LEGITIMATE
        self.assertEqual(
            leaked, set(),
            "an explicit colon-free secret is published in " + repr(sorted(leaked)))
