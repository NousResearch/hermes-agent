"""Tests for the continuous knowledge-sync worker."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from packages.knowledge import KnowledgeService, KnowledgeSyncWorker  # noqa: E402
from packages.knowledge.worker import start_health_server  # noqa: E402
from tests.knowledge.test_knowledge import cfg  # noqa: E402


class WorkerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vault = os.path.join(self.tmp, "vault")
        os.makedirs(os.path.join(self.vault, "architecture"))
        self.svc = KnowledgeService(config=cfg(self.tmp))
        self.worker = KnowledgeSyncWorker(
            [self.vault], service=self.svc, debounce=0.0,
            full_scan_on_start=False, reconcile_interval=0,
            state_path=os.path.join(self.tmp, "state.json"),
        )

    def tearDown(self):
        self.worker.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rel, text):
        p = os.path.join(self.vault, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
        return p


class TestIncrementalIndexing(WorkerTestBase):
    def test_new_file_is_indexed(self):
        p = self.write("architecture/kafka.md", "# Kafka\nPartitioned commit log.")
        self.worker.notify(p)
        self.assertEqual(self.worker.drain(force=True), 1)
        self.assertEqual(self.worker.state.indexed, 1)
        self.assertTrue(self.svc.search("partitioned commit log").chunks)

    def test_modified_file_is_updated_not_duplicated(self):
        p = self.write("architecture/kafka.md", "# Kafka\nOriginal text.")
        self.worker.notify(p); self.worker.drain(force=True)
        before = len(self.svc.list_documents())

        self.write("architecture/kafka.md", "# Kafka\nRewritten with tiered storage.")
        self.worker.notify(p); self.worker.drain(force=True)

        self.assertEqual(self.worker.state.updated, 1)
        self.assertEqual(len(self.svc.list_documents()), before,
                         "update must not create a second document")
        self.assertTrue(self.svc.search("tiered storage").chunks)
        self.assertFalse([c for c in self.svc.search("Original text").chunks
                          if "Original text" in c.text])

    def test_unchanged_file_is_skipped(self):
        p = self.write("architecture/kafka.md", "# Kafka\nStable.")
        self.worker.notify(p); self.worker.drain(force=True)
        self.worker.notify(p); self.worker.drain(force=True)
        self.assertEqual(self.worker.state.indexed, 1)
        self.assertEqual(self.worker.state.updated, 0)
        self.assertEqual(self.worker.state.skipped, 1)

    def test_deleted_file_removes_vectors(self):
        p = self.write("architecture/kafka.md", "# Kafka\nEphemeral note.")
        self.worker.notify(p); self.worker.drain(force=True)
        self.assertTrue(self.svc.search("ephemeral note").chunks)

        os.remove(p)
        self.worker.notify(p); self.worker.drain(force=True)
        self.assertEqual(self.worker.state.deleted, 1)
        self.assertFalse([c for c in self.svc.search("ephemeral note").chunks
                          if "Ephemeral" in c.text])

    def test_metadata_tracking_shape(self):
        p = self.write("architecture/kafka.md", "# Kafka\nHashed.")
        self.worker.notify(p); self.worker.drain(force=True)
        entries = list(self.worker.syncer.manifest.data.values())
        self.assertEqual(len(entries), 1)
        e = entries[0]
        for key in ("checksum", "path", "source_key", "synced_at"):
            self.assertIn(key, e)
        self.assertEqual(e["path"], p)
        self.assertEqual(len(e["checksum"]), 64)

    def test_manifest_persists_across_restart(self):
        p = self.write("architecture/kafka.md", "# Kafka\nDurable.")
        self.worker.notify(p); self.worker.drain(force=True)

        fresh = KnowledgeSyncWorker([self.vault], service=self.svc, debounce=0.0,
                                    full_scan_on_start=False,
                                    state_path=os.path.join(self.tmp, "s2.json"))
        fresh.notify(p); fresh.drain(force=True)
        self.assertEqual(fresh.state.skipped, 1, "restart must reuse the hash manifest")
        self.assertEqual(fresh.state.indexed, 0)


class TestFiltering(WorkerTestBase):
    def test_ignores_obsidian_and_git_dirs(self):
        for d in (".obsidian", ".git", "node_modules", "dist", "build"):
            p = self.write(f"{d}/thing.md", "junk")
            self.worker.notify(p)
        self.assertEqual(self.worker.drain(force=True), 0)

    def test_watches_only_configured_extensions(self):
        self.worker.notify(self.write("notes/a.md", "# A\nmarkdown"))
        self.worker.notify(self.write("notes/b.txt", "plain text note"))
        self.worker.notify(self.write("notes/c.png", "binaryish"))
        self.worker.notify(self.write("notes/d.js", "console.log(1)"))
        self.assertEqual(self.worker.drain(force=True), 2)

    def test_paths_outside_roots_rejected(self):
        outside = os.path.join(self.tmp, "elsewhere.md")
        with open(outside, "w") as fh:
            fh.write("# Nope")
        self.worker.notify(outside)
        self.assertEqual(self.worker.drain(force=True), 0)


class TestDebounceAndReconcile(WorkerTestBase):
    def test_debounce_coalesces_rapid_saves(self):
        w = KnowledgeSyncWorker([self.vault], service=self.svc, debounce=5.0,
                                full_scan_on_start=False,
                                state_path=os.path.join(self.tmp, "s3.json"))
        p = self.write("notes/x.md", "# X\nv1")
        for _ in range(5):
            w.notify(p)
        self.assertEqual(len(w._pending), 1, "five events must coalesce to one path")
        self.assertEqual(w.drain(), 0, "quiet window not elapsed yet")
        self.assertEqual(w.drain(force=True), 1)

    def test_reconcile_catches_changes_made_while_down(self):
        self.write("notes/one.md", "# One\nalpha")
        self.write("notes/two.md", "# Two\nbeta")
        reports = self.worker.reconcile()
        counts = reports[self.vault]["counts"]
        self.assertEqual(counts["added"], 2)
        self.assertTrue(self.svc.search("alpha").chunks)

    def test_polling_scan_detects_changes_without_inotify(self):
        self.write("notes/poll.md", "# Poll\ndetected by mtime sweep")
        self.worker._poll_scan()
        self.assertEqual(self.worker.drain(force=True), 1)
        self.assertTrue(self.svc.search("mtime sweep").chunks)


class TestManifestKeyAgreement(WorkerTestBase):
    """Worker and synchronizer must mint identical manifest source_keys.

    If they diverge, the periodic reconcile() treats worker-indexed documents
    as belonging to another source, finds them "stale", and deletes them --
    silently emptying the index on a live system.
    """

    def test_worker_and_reconcile_agree_on_source_key(self):
        p = self.write("notes/agree.md", "# Agree\nkey parity matters")
        self.worker.notify(p)
        self.worker.drain(force=True)
        worker_keys = {v["source_key"] for v in self.worker.syncer.manifest.data.values()}

        self.worker.reconcile()
        after = {v["source_key"] for v in self.worker.syncer.manifest.data.values()}
        self.assertEqual(worker_keys, after)
        self.assertEqual(self.worker.state.deleted, 0,
                         "reconcile must not prune worker-indexed documents")
        self.assertTrue(self.svc.search("key parity matters").chunks)

    def test_reconcile_survives_symlinked_root(self):
        link = os.path.join(self.tmp, "vault-link")
        os.symlink(self.vault, link)
        w = KnowledgeSyncWorker([link], service=self.svc, debounce=0.0,
                                full_scan_on_start=False,
                                state_path=os.path.join(self.tmp, "s7.json"))
        p = self.write("notes/sym.md", "# Sym\nreached through a symlink")
        w.notify(p)
        self.assertEqual(w.drain(force=True), 1, "symlinked root must resolve")
        w.reconcile()
        self.assertEqual(w.state.deleted, 0)
        self.assertTrue(self.svc.search("reached through a symlink").chunks)


class TestObservability(WorkerTestBase):
    def test_health_payload(self):
        self.worker.notify(self.write("notes/h.md", "# H\nhealthy"))
        self.worker.drain(force=True)
        h = self.worker.health()
        self.assertTrue(h["healthy"])
        self.assertIn("counts", h)
        self.assertIn("provider", h)
        self.assertEqual(h["backend"], "local")
        self.assertEqual(h["counts"]["indexed"], 1)

    def test_state_file_written(self):
        self.worker.notify(self.write("notes/s.md", "# S\nstate"))
        self.worker.drain(force=True)
        with open(self.worker.state_path) as fh:
            data = json.load(fh)
        self.assertEqual(data["counts"]["indexed"], 1)

    def test_health_http_endpoint(self):
        port = 8791
        srv = start_health_server(self.worker, port=port)
        self.assertIsNotNone(srv)
        time.sleep(0.2)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            payload = json.loads(r.read())
        self.assertIn("healthy", payload)
        self.assertIn("backend", payload)

    def test_provider_error_is_recorded_not_raised(self):
        def boom(*a, **kw):
            raise RuntimeError("index backend down")
        self.svc.index = boom  # type: ignore[method-assign]
        self.worker.notify(self.write("notes/e.md", "# E\nerror path"))
        self.worker.drain(force=True)
        self.assertEqual(self.worker.state.errors, 1)
        self.assertIn("index backend down", self.worker.state.last_error)


class TestEnrichmentHook(WorkerTestBase):
    def test_enrichment_called_for_new_notes_only(self):
        seen = []
        w = KnowledgeSyncWorker([self.vault], service=self.svc, debounce=0.0,
                                full_scan_on_start=False,
                                state_path=os.path.join(self.tmp, "s4.json"),
                                enrich=lambda d: seen.append(d.title))
        p = self.write("notes/n.md", "# Enrich Me\nbody")
        w.notify(p); w.drain(force=True)
        self.assertEqual(seen, ["Enrich Me"])

        self.write("notes/n.md", "# Enrich Me\nbody v2")
        w.notify(p); w.drain(force=True)
        self.assertEqual(seen, ["Enrich Me"], "updates must not re-trigger enrichment")

    def test_enrichment_failure_does_not_break_indexing(self):
        def bad(_d):
            raise RuntimeError("enricher exploded")
        w = KnowledgeSyncWorker([self.vault], service=self.svc, debounce=0.0,
                                full_scan_on_start=False,
                                state_path=os.path.join(self.tmp, "s5.json"),
                                enrich=bad)
        w.notify(self.write("notes/g.md", "# G\nstill indexed"))
        w.drain(force=True)
        self.assertEqual(w.state.indexed, 1)
        self.assertEqual(w.state.errors, 0)
        self.assertTrue(self.svc.search("still indexed").chunks)


class TestRunLoop(WorkerTestBase):
    def test_run_loop_picks_up_live_filesystem_change(self):
        w = KnowledgeSyncWorker([self.vault], service=self.svc, debounce=0.1,
                                poll_interval=0.3, full_scan_on_start=True,
                                reconcile_interval=0,
                                state_path=os.path.join(self.tmp, "s6.json"))
        t = threading.Thread(target=w.run, daemon=True)
        t.start()
        try:
            time.sleep(0.5)
            self.write("notes/live.md", "# Live\ncreated while the worker was running")
            deadline = time.time() + 15
            while time.time() < deadline:
                if self.svc.search("created while the worker").chunks:
                    break
                time.sleep(0.3)
            self.assertTrue(self.svc.search("created while the worker").chunks,
                            f"worker ({w.state.mode}) did not index the new file")
        finally:
            w.stop()
            t.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
