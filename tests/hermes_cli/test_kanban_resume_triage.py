"""Target-only operator continuation; real native SQLite/lifecycle fixtures."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as notify


class ResumeTriageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="resume-triage-"))
        self.env = patch.dict(os.environ, {
            "HOME": str(self.root), "HERMES_HOME": str(self.root / ".hermes"),
            "HERMES_KANBAN_HOME": str(self.root / ".hermes"),
            "HERMES_KANBAN_BOARD": "default", "HERMES_PROFILE": "nyra",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.db = self.root / "fixture.db"
        self.c = kbc.connect(db_path=self.db)
        self.addCleanup(self.c.close)
        self.addCleanup(self.snapshot)
        print("FIXTURE_ROOT " + str(self.root), flush=True)
        self.assertEqual(Path(self.c.execute("PRAGMA database_list").fetchone()["file"]), self.db)
        self.tid = kb.create_task(
            self.c, title="Approved candidate", body="exact body\nall HOLDS\n",
            assignee="nyra", triage=True, workspace_kind="dir",
            workspace_path=str(self.root / "candidate"), model_override="frozen-model",
            provider_override="frozen-provider", completion_contract="local-only")
        notify.add_notify_sub(self.c, task_id=self.tid, platform="desktop",
                              chat_id="original-parent", notifier_profile="nyra",
                              delivery_mode="notify+wake")
        self.digest = hashlib.sha256(self.row()["body"].encode()).hexdigest()

    def row(self):
        return dict(self.c.execute("SELECT * FROM tasks WHERE id=?", (self.tid,)).fetchone())

    def snapshot(self):
        data = {table: [tuple(r) for r in self.c.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                for table in ("tasks", "task_runs", "task_events", "task_links", "task_comments",
                              "kanban_notify_subs")}
        with (self.root / "invariants.jsonl").open("a") as output:
            output.write(json.dumps(data) + "\n")
        return data

    def resume(self, **kw):
        self.assertTrue(hasattr(kb, "resume_triage_task"), "new operator API is not shipped")
        return kb.resume_triage_task(self.c, self.tid, actor="nyra", expected_owner="nyra",
                                     expected_body_sha256=kw.pop("digest", self.digest),
                                     reason="actual prerequisite changed", **kw)

    def test_target_only_sibling_invariance(self):
        import importlib.util
        path = Path(kb.__file__).parents[1] / "plugins/kanban/dashboard/plugin_api.py"
        spec = importlib.util.spec_from_file_location("resume_dashboard_fixture", path)
        assert spec is not None and spec.loader is not None
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        sibling = kb.create_task(self.c, title="sibling")
        self.assertTrue(api._set_status_direct(self.c, sibling, "todo"))
        before = dict(self.c.execute("SELECT * FROM tasks WHERE id=?", (sibling,)).fetchone())
        self.assertEqual(before["status"], "todo")
        if hasattr(kb, "resume_triage_task"):
            self.assertIsNone(self.resume()[1])
        else:
            # Explicit unchanged-base counterfactual: existing deterministic route.
            self.assertTrue(kb.specify_triage_task(self.c, self.tid))
        after = dict(self.c.execute("SELECT * FROM tasks WHERE id=?", (sibling,)).fetchone())
        self.assertEqual(after, before, "continuation must not sweep unrelated eligible cards")

    def test_preserves_exact_bindings_and_guard(self):
        kb.specify_triage_task(self.c, self.tid)
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 1):
            kb.block_task(self.c, self.tid, kind="needs_input", reason="original guard reason")
            if self.row()["status"] == "triage":
                break
            kb.unblock_task(self.c, self.tid)
        before = self.row()
        self.assertEqual(before["block_recurrences"], kb.BLOCK_RECURRENCE_LIMIT)
        history = self.snapshot()
        subs = notify.list_notify_subs(self.c, self.tid)
        task, error = self.resume()
        self.assertIsNone(error)
        assert task is not None
        self.assertEqual(task.status, "ready")
        after = self.row()
        self.assertEqual({k: v for k, v in before.items() if k != "status"},
                         {k: v for k, v in after.items() if k != "status"})
        self.assertEqual(notify.list_notify_subs(self.c, self.tid), subs)
        for table in ("task_runs", "task_comments", "task_links"):
            self.assertEqual(self.snapshot()[table], history[table])
        self.assertEqual(self.snapshot()["task_events"][:-1], history["task_events"])

    def test_open_parent_and_wrong_bindings_refuse_without_mutation(self):
        for kwargs in ({"digest": "0" * 64}, {"digest": "bad"}):
            before = self.snapshot()
            self.assertIsNotNone(self.resume(**kwargs)[1])
            self.assertEqual(self.snapshot(), before)
        for actor, owner in (("other", "nyra"), ("nyra", "other")):
            before = self.snapshot()
            task, error = kb.resume_triage_task(
                self.c, self.tid, actor=actor, expected_owner=owner,
                expected_body_sha256=self.digest, reason="changed")
            self.assertIsNone(task)
            self.assertIsNotNone(error)
            self.assertEqual(self.snapshot(), before)
        parent = kb.create_task(self.c, title="unresolved", triage=True)
        kb.link_tasks(self.c, parent, self.tid)
        before = self.snapshot()
        self.assertIsNotNone(self.resume(claim=True)[1])
        self.assertEqual(self.snapshot(), before)

    def test_active_claim_and_null_pid_open_run_refuse(self):
        # Existing lifecycle can retain a direct claim with no worker PID.
        kb.specify_triage_task(self.c, self.tid)
        claimed = kb.claim_task(self.c, self.tid, claimer="incumbent")
        self.assertIsNotNone(claimed)
        self.assertIsNone(self.row()["worker_pid"])
        before = self.snapshot()
        _, error = self.resume(claim=True)
        assert error is not None
        self.assertIn("open run", error)
        self.assertEqual(self.snapshot(), before)
        # Existing native close retains task claim fields: independently exercise
        # the claim guard without handwritten status/counter/run SQL.
        with kb.write_txn(self.c):
            kb._end_run(self.c, self.tid, outcome="blocked", status="blocked")
        before = self.snapshot()
        _, error = self.resume(claim=True)
        assert error is not None
        self.assertIn("active claim", error)
        self.assertEqual(self.snapshot(), before)

    def test_atomic_contenders_open_exactly_one_run(self):
        original = self.row()
        history = self.snapshot()
        subs = notify.list_notify_subs(self.c, self.tid)
        barrier = threading.Barrier(2)
        results = []
        def compete():
            with contextlib.closing(kbc.connect(db_path=self.db)) as conn:
                barrier.wait(timeout=10)
                results.append(kb.resume_triage_task(
                    conn, self.tid, actor="nyra", expected_owner="nyra",
                    expected_body_sha256=self.digest, reason="changed", claim=True))
        threads = [threading.Thread(target=compete) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 2)
        self.assertEqual(sum(task is not None for task, _ in results), 1)
        self.assertEqual(self.row()["status"], "running")
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0], 1)
        self.assertIsNone(self.c.execute("SELECT outcome FROM task_runs").fetchone()[0])
        changed = {"status", "claim_lock", "claim_expires", "started_at", "current_run_id"}
        self.assertEqual({k: v for k, v in original.items() if k not in changed},
                         {k: v for k, v in self.row().items() if k not in changed})
        self.assertEqual(notify.list_notify_subs(self.c, self.tid), subs)
        self.assertEqual(self.snapshot()["task_events"][:-2], history["task_events"])
        winner = self.snapshot()
        self.assertIsNotNone(self.resume(claim=True)[1])
        self.assertEqual(self.snapshot(), winner)

    def test_cli_binding_and_delegated_denial(self):
        parser = argparse.ArgumentParser()
        cli.build_parser(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["kanban", "resume-triage", self.tid,
                                  "--expected-owner", "nyra", "--expected-body-sha256", self.digest,
                                  "--reason", "changed", "--claim", "--json"])
        with patch.dict(os.environ, {"HERMES_DELEGATED_CHILD_CONTEXT": "1"}):
            self.assertTrue(cli._is_delegated_child_cli_mutation(args))
            before = self.snapshot()
            with self.assertRaises(PermissionError):
                self.resume(claim=True)
            self.assertEqual(self.snapshot(), before)
        with patch.object(kbc, "connect_closing", lambda: kbc_context(self.db)):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli._cmd_resume_triage(args), 0)
            self.assertIn('"status": "running"', output.getvalue())
            self.assertIn('"run_id":', output.getvalue())


@contextlib.contextmanager
def kbc_context(path):
    with contextlib.closing(kbc.connect(db_path=path)) as conn:
        yield conn
