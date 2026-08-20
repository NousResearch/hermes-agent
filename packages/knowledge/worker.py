"""knowledge-sync-worker — continuous Obsidian → RAG synchronization.

    Obsidian ──LiveSync/Git──▶ vault on disk ──▶ this worker ──▶ KnowledgeProvider
                                                                  (AnythingLLM,
                                                                   local, …)

Design points:

* **Filesystem events, not full scans.** watchdog inotify drives the loop; a
  polling fallback keeps the worker functional when watchdog isn't installed.
* **Debounced.** Editors write a file several times per save; events are
  coalesced per path over a quiet window before touching the index.
* **Incremental.** Every path carries a SHA-256 in the metadata DB. Unchanged
  content is skipped; changed content is `update()`d; removed files are
  `delete()`d. The vector DB is never rebuilt wholesale.
* **Provider-agnostic.** The worker only ever calls KnowledgeService, which
  only ever calls KnowledgeProvider. Swapping AnythingLLM for Qdrant changes
  nothing here.
* **Observable.** Health/state is written to a JSON file and served over an
  optional HTTP health endpoint for systemd/monitoring.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .service import KnowledgeService, get_knowledge_service
from .sync import DocumentSynchronizer, SKIP_DIRS, read_document
from .types import Document

logger = logging.getLogger("hermes.knowledge.worker")


def _as_text(p: "str | bytes") -> str:
    """watchdog emits bytes paths when the watch root was given as bytes."""
    return p.decode("utf-8", "replace") if isinstance(p, bytes) else p

DEFAULT_EXTS = (".md", ".markdown", ".txt", ".pdf")
DEFAULT_IGNORE = set(SKIP_DIRS) | {".obsidian", ".trash", ".stfolder"}


# --------------------------------------------------------------------- state
@dataclass
class WorkerState:
    started_at: float = field(default_factory=time.time)
    last_event_at: float = 0.0
    last_sync_at: float = 0.0
    events_seen: int = 0
    indexed: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    errors: int = 0
    last_error: str = ""
    backend: str = ""
    watching: List[str] = field(default_factory=list)
    mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "healthy": self.errors == 0 or (now - self.last_sync_at) < 3600,
            "mode": self.mode,
            "backend": self.backend,
            "watching": self.watching,
            "uptime_seconds": round(now - self.started_at, 1),
            "staleness_seconds": (round(now - self.last_sync_at, 1)
                                  if self.last_sync_at else None),
            "events_seen": self.events_seen,
            "counts": {"indexed": self.indexed, "updated": self.updated,
                       "deleted": self.deleted, "skipped": self.skipped,
                       "errors": self.errors},
            "last_event_at": self.last_event_at or None,
            "last_sync_at": self.last_sync_at or None,
            "last_error": self.last_error,
        }


# -------------------------------------------------------------------- worker
class KnowledgeSyncWorker:
    """Continuously mirrors one or more vaults into the knowledge index."""

    def __init__(
        self,
        roots: List[str],
        service: Optional[KnowledgeService] = None,
        workspace: Optional[str] = None,
        source: str = "obsidian",
        exts: Optional[List[str]] = None,
        debounce: float = 2.0,
        poll_interval: float = 15.0,
        state_path: Optional[str] = None,
        full_scan_on_start: bool = True,
        reconcile_interval: float = 900.0,
        enrich: Optional[Callable[[Document], None]] = None,
    ):
        # realpath, not abspath: /opt/secondbrain is typically a symlink to the
        # actual vault, and inotify watches + os.walk must agree on one identity
        # or every event path would fail the _relevant() root check.
        self.roots = [os.path.realpath(os.path.expanduser(r)) for r in roots]
        self.service = service or get_knowledge_service()
        self.workspace = workspace or self.service.config.workspace
        self.source = source
        self.exts = tuple(e.lower() for e in (exts or DEFAULT_EXTS))
        self.debounce = debounce
        self.poll_interval = poll_interval
        self.reconcile_interval = reconcile_interval
        self.full_scan_on_start = full_scan_on_start
        self.enrich = enrich

        base = os.path.dirname(self.service.config.db_path)
        self.state_path = state_path or os.path.join(base, "sync_worker_state.json")
        self.syncer = DocumentSynchronizer(
            self.service, manifest_path=os.path.join(base, "sync_manifest.json"))

        self.state = WorkerState(backend=self.service.primary.name,
                                 watching=list(self.roots))
        self._pending: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._observer = None

    # -- helpers -------------------------------------------------------
    def _relevant(self, path: str) -> bool:
        if os.path.splitext(path)[1].lower() not in self.exts:
            return False
        parts = set(os.path.normpath(path).split(os.sep))
        if parts & DEFAULT_IGNORE:
            return False
        return any(path.startswith(r + os.sep) or path == r for r in self.roots)

    def _root_for(self, path: str) -> str:
        for r in self.roots:
            if path.startswith(r + os.sep):
                return r
        return self.roots[0] if self.roots else ""

    def _source_key(self, root: str) -> str:
        return f"{self.source}:{root}"

    def _write_state(self) -> None:
        try:
            tmp = f"{self.state_path}.tmp"
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(tmp, "w") as fh:
                json.dump(self.state.to_dict(), fh, indent=1)
            os.replace(tmp, self.state_path)
        except Exception as exc:  # pragma: no cover
            logger.debug("worker: state write failed: %s", exc)

    # -- event intake --------------------------------------------------
    def notify(self, path: str) -> None:
        """Queue a filesystem path for (debounced) reindexing."""
        path = os.path.realpath(path) if os.path.exists(path) else os.path.abspath(path)
        if not self._relevant(path):
            return
        with self._lock:
            self._pending[path] = time.time()
            self.state.events_seen += 1
            self.state.last_event_at = time.time()

    # -- indexing ------------------------------------------------------
    def _process(self, path: str) -> None:
        root = self._root_for(path)
        key = self._source_key(root)
        try:
            if not os.path.exists(path):
                self._handle_delete(path, key)
                return
            doc = read_document(path, self.source, self.workspace, root)
            if doc is None:
                return
            prev = self.syncer.manifest.data.get(doc.id)
            if prev and prev.get("checksum") == doc.checksum:
                self.state.skipped += 1
                return
            res = (self.service.update(doc) if prev else self.service.index(doc))
            if not res.ok:
                raise RuntimeError(res.detail)
            self.syncer.manifest.data[doc.id] = {
                "checksum": doc.checksum, "path": doc.path,
                "source_key": key, "synced_at": time.time(),
                "provider_id": res.detail or "",
            }
            self.syncer.manifest.save()
            if prev:
                self.state.updated += 1
                logger.info("worker: updated %s", path)
            else:
                self.state.indexed += 1
                logger.info("worker: indexed %s", path)
            self.state.last_sync_at = time.time()
            if self.enrich and not prev:
                try:
                    self.enrich(doc)
                except Exception as exc:
                    logger.warning("worker: enrichment failed for %s: %s", path, exc)
        except Exception as exc:
            self.state.errors += 1
            self.state.last_error = f"{path}: {exc}"
            logger.error("worker: failed on %s: %s", path, exc)

    def _handle_delete(self, path: str, source_key: str) -> None:
        victims = [k for k, v in self.syncer.manifest.data.items()
                   if v.get("path") == path]
        for doc_id in victims:
            res = self.service.delete(doc_id, workspace=self.workspace)
            if res.ok:
                self.syncer.manifest.data.pop(doc_id, None)
                self.state.deleted += 1
                self.state.last_sync_at = time.time()
                logger.info("worker: deleted %s", path)
            else:
                self.state.errors += 1
                self.state.last_error = f"delete {path}: {res.detail}"
        if victims:
            self.syncer.manifest.save()

    def drain(self, force: bool = False) -> int:
        """Index every debounced path whose quiet window has elapsed."""
        now = time.time()
        with self._lock:
            ready = [p for p, t in self._pending.items()
                     if force or (now - t) >= self.debounce]
            for p in ready:
                self._pending.pop(p, None)
        for p in ready:
            self._process(p)
        if ready:
            self._write_state()
        return len(ready)

    def reconcile(self) -> Dict[str, Any]:
        """Full manifest reconciliation — catches events missed while down."""
        reports = {}
        for root in self.roots:
            rep = self.syncer.sync_path(root, source=self.source,
                                        workspace=self.workspace,
                                        exts=list(self.exts))
            reports[root] = rep.to_dict()
            self.state.indexed += len(rep.added)
            self.state.updated += len(rep.updated)
            self.state.deleted += len(rep.deleted)
            self.state.skipped += len(rep.unchanged)
            self.state.errors += len(rep.failed)
            if rep.failed:
                self.state.last_error = rep.failed[-1]
        self.state.last_sync_at = time.time()
        self._write_state()
        return reports

    # -- lifecycle -----------------------------------------------------
    def _start_observer(self) -> bool:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("worker: watchdog not installed — falling back to polling")
            return False

        worker = self

        class _Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return
                worker.notify(_as_text(event.src_path))
                dest = getattr(event, "dest_path", None)
                if dest:
                    worker.notify(_as_text(dest))

        obs = Observer()
        for root in self.roots:
            if os.path.isdir(root):
                obs.schedule(_Handler(), root, recursive=True)
            else:
                logger.warning("worker: watch root missing: %s", root)
        obs.daemon = True
        obs.start()
        self._observer = obs
        return True

    def _poll_scan(self) -> None:
        """Fallback change detection: mtime sweep against the manifest."""
        for root in self.roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORE]
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in self.exts:
                        self.notify(os.path.join(dirpath, fn))
        # deletions
        for entry in list(self.syncer.manifest.data.values()):
            p = entry.get("path", "")
            if p and not p.startswith("session://") and not os.path.exists(p):
                self.notify(p)

    def run(self) -> None:
        """Blocking main loop. Runs until stop() or SIGTERM/SIGINT."""
        event_mode = self._start_observer()
        self.state.mode = "inotify" if event_mode else "polling"
        logger.info("worker: started in %s mode; roots=%s backend=%s",
                    self.state.mode, self.roots, self.state.backend)
        self._write_state()

        if self.full_scan_on_start:
            logger.info("worker: initial reconciliation…")
            self.reconcile()

        last_reconcile = time.time()
        last_poll = 0.0
        try:
            while not self._stop.is_set():
                if not event_mode and (time.time() - last_poll) >= self.poll_interval:
                    self._poll_scan()
                    last_poll = time.time()
                self.drain()
                if self.reconcile_interval and \
                        (time.time() - last_reconcile) >= self.reconcile_interval:
                    logger.info("worker: periodic reconciliation")
                    self.reconcile()
                    last_reconcile = time.time()
                self._stop.wait(0.5)
        finally:
            self.drain(force=True)
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:
                pass
            self._observer = None
        self._write_state()
        logger.info("worker: stopped")

    def health(self) -> Dict[str, Any]:
        out = self.state.to_dict()
        out["provider"] = self.service.health()
        with self._lock:
            out["pending"] = len(self._pending)
        return out


# ------------------------------------------------------------ health server
def start_health_server(worker: KnowledgeSyncWorker, port: int = 8787,
                        host: str = "127.0.0.1") -> Optional[threading.Thread]:
    """Expose GET /health and GET /metrics for systemd/monitoring."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - silence access log
            pass

        def do_GET(self):
            if self.path.rstrip("/") not in ("/health", "/metrics", ""):
                self.send_error(404)
                return
            payload = worker.health()
            body = json.dumps(payload, indent=1).encode()
            self.send_response(200 if payload.get("healthy") else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        srv = ThreadingHTTPServer((host, port), _H)
    except OSError as exc:
        logger.warning("worker: health server disabled (%s)", exc)
        return None
    t = threading.Thread(target=srv.serve_forever, daemon=True,
                         name="knowledge-health")
    t.start()
    logger.info("worker: health endpoint on http://%s:%d/health", host, port)
    return t


# --------------------------------------------------------------------- main
def build_worker_from_config(roots: Optional[List[str]] = None,
                             **kw: Any) -> KnowledgeSyncWorker:
    svc = get_knowledge_service()
    if not roots:
        roots = [s["path"] for s in (svc.config.sync_sources or [])
                 if s.get("path") and s.get("type") in ("obsidian", "markdown", "git", "mkdocs")]
    if not roots:
        raise SystemExit(
            "knowledge-sync-worker: no roots. Pass paths as arguments or set "
            "knowledge.sync_sources in config.yaml."
        )
    return KnowledgeSyncWorker(roots, service=svc, **kw)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="knowledge-sync-worker",
                                 description="Continuous vault → RAG sync")
    ap.add_argument("roots", nargs="*", help="Directories to watch")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--source", default="obsidian")
    ap.add_argument("--debounce", type=float, default=2.0)
    ap.add_argument("--poll-interval", type=float, default=15.0)
    ap.add_argument("--reconcile-interval", type=float, default=900.0)
    ap.add_argument("--no-initial-scan", action="store_true")
    ap.add_argument("--health-port", type=int, default=8787,
                    help="0 disables the health endpoint")
    ap.add_argument("--once", action="store_true",
                    help="Reconcile once and exit (cron mode)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    worker = build_worker_from_config(
        args.roots or None,
        workspace=args.workspace,
        source=args.source,
        debounce=args.debounce,
        poll_interval=args.poll_interval,
        reconcile_interval=args.reconcile_interval,
        full_scan_on_start=not args.no_initial_scan,
    )

    if args.once:
        print(json.dumps(worker.reconcile(), indent=1))
        return 0

    if args.health_port:
        start_health_server(worker, args.health_port)

    def _sig(_signum, _frame):
        worker.stop()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
