"""Automatic, profile-owned connected-source scans (no jobs or conversation turns)."""
from __future__ import annotations

import asyncio
from contextlib import closing
import json
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from gateway.connected_assistance_store import AssistanceStore
from gateway.session_authorities import owner_scope, serving_authorities

logger = logging.getLogger(__name__)
SCAN_INTERVAL_SECONDS = 15 * 60
SCAN_TIMEOUT_SECONDS = 60
POLL_SECONDS = 30


def _bounded_prior(records):
    retained, used = [], 0
    for record in records:
        size = len(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        if used + size <= 16_384:
            retained.append(record)
            used += size
    return retained


def enabled() -> bool:
    from hermes_cli.config_effective import load_user_config_effective
    config = load_user_config_effective()
    return (config.get("gateway") or {}).get("connected_assistance", True) is True


def _claim_scan(store, now):
    # Persist before spending requests: restarts and a failed scan respect the same cadence.
    with closing(store._connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT value FROM settings WHERE key='next_scan'").fetchone()
        if row and float(row[0]) > now:
            return False
        db.execute("INSERT INTO settings VALUES ('next_scan', ?) ON CONFLICT(key) "
                   "DO UPDATE SET value=excluded.value", (str(now + SCAN_INTERVAL_SECONDS),))
    return True


class _CurrentSources:
    def __init__(self, check):
        from tools.connectors.assistance import ConnectedSources
        self.sources = ConnectedSources()
        self.check = check

    def discover(self, **kwargs):
        self.check()
        return self.sources.discover(**kwargs)

    def read(self, *args, **kwargs):
        self.check()
        return self.sources.read(*args, check_current=self.check, **kwargs)

    def revalidate(self, *args, **kwargs):
        self.check()
        return self.sources.revalidate(*args, **kwargs)


class _WriterHandle:
    """One root-shutdown registration follows this serial runtime's physical thread."""
    def __init__(self, runtime):
        self.runtime = runtime

    def start(self):
        self.runtime.thread.start()

    def is_alive(self):
        return self.runtime.thread is not None and self.runtime.thread.is_alive()


class ConnectedAssistance:
    """One serial worker, retained until its blocking requests settle even on cancellation."""
    def __init__(self, runner):
        self.runner = runner
        self.worker = None
        self.thread = None
        self.writer_handle = None
        self.owner = None
        self.stopped = threading.Event()
        self.retired = {}
        self.tick_lock = asyncio.Lock()

    def current(self, authority):
        return (not self.stopped.is_set() and self.retired.get(authority.profile_id) is not authority
                and self.runner._running and not self.runner._draining
                and not getattr(authority, "retiring", False)
                and not getattr(authority, "retired", False)
                and any(a is authority for a in serving_authorities(self.runner)))

    def _scan(self, authority, cancelled):
        from agent.connected_assistance import scan_connected_sources
        from agent.auxiliary_client import call_llm
        with owner_scope(authority, hydrate_secrets=True):
            if not self.current(authority) or not enabled():
                return []
            store = AssistanceStore(Path(authority.profile_id))

            def check():
                if cancelled.is_set() or not self.current(authority) or store.is_paused() or not enabled():
                    raise PermissionError("assistance_paused")

            def model(**kwargs):
                check()
                return call_llm(**kwargs)

            if store.is_paused():
                return []
            deadline = time.monotonic() + SCAN_TIMEOUT_SECONDS
            sources = _CurrentSources(check)
            if not _claim_scan(store, time.time()):
                return self._ready(store, sources, check, deadline)
            result = scan_connected_sources(_bounded_prior(store.prior_evidence()), sources=sources, llm=model,
                                            deadline=deadline, timeout_seconds=SCAN_TIMEOUT_SECONDS, check_current=check)
            check()
            if result.get("connected") is not True or result.get("errors"):
                return []
            if result["evidence"] and not sources.revalidate(result["evidence"], deadline=deadline):
                return []
            check()
            store.ingest_scan(result, check_current=check)
            return self._ready(store, sources, check, deadline)

    @staticmethod
    def _ready(store, sources, check, deadline):
        rows = store.list_findings(status="open", limit=5, pending_notification=True)
        evidence = [e for row in rows for e in row["evidence"]]
        if not rows or not sources.revalidate(evidence, deadline=deadline):
            return []
        check()
        return rows

    async def tick(self):
        async with self.tick_lock:
            # A cancelled waiter does not free the worker slot or begin another scan.
            if self.worker is not None:
                if _worker_live(self):
                    return
                self._settle_worker()
            for authority in serving_authorities(self.runner):
                if not self.current(authority):
                    continue
                self.owner = authority
                cancelled = threading.Event()
                self.worker = self._start_worker(authority, cancelled)
                try:
                    rows = await asyncio.shield(self.worker)
                    if self.current(authority):
                        from gateway.connected_assistance_delivery import publish_private_findings
                        publish_private_findings(authority, rows, current=lambda: self.current(authority))
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                except Exception:  # health: allow BLE001 -- redacted provider/storage boundary; next cadence retries
                    logger.warning("Connected assistance scan unavailable")
                finally:
                    if self.worker.done():
                        self._settle_worker()

    def _start_worker(self, authority, cancelled):
        from agent.memory_provider import spawn_context_thread
        from gateway.runtime_ownership import process_ownership
        from gateway.session_runtime_workers import track_turn_worker

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        handle = SimpleNamespace(worker_done=threading.Event())

        def complete(result, error):
            if not future.done():
                if error is None:
                    future.set_result(result)
                else:
                    future.set_exception(error)

        def run():
            result, error = None, None
            try:
                result = self._scan(authority, cancelled)
            except BaseException as exc:  # health: allow BLE001 -- transfer the worker failure to its retained future; caller logs redacted provider errors
                error = exc
            finally:
                handle.worker_done.set()
                try:
                    loop.call_soon_threadsafe(complete, result, error)
                except RuntimeError:
                    logger.debug("Connected assistance worker settled after its loop closed")

        with owner_scope(authority):
            track_turn_worker(self.runner, handle, None)
            self.thread = spawn_context_thread(run, name="connected-assistance")
        try:
            if self.writer_handle is None:
                self.writer_handle = _WriterHandle(self)
                process_ownership.start_writer(self.writer_handle)
            else:
                self.thread.start()
        except RuntimeError:
            handle.worker_done.set()
            raise
        return future

    def _settle_worker(self):
        if not self.worker.cancelled():
            self.worker.exception()
        self.worker = self.owner = None

    async def stop(self, *, home=None, timeout: float = SCAN_TIMEOUT_SECONDS + 5):
        if home is None:
            self.stopped.set()
        else:
            from gateway.session_authorities import authority_for_home
            authority = authority_for_home(self.runner, home)
            if authority is not None:
                self.retired[authority.profile_id] = authority
        worker = self.worker
        owner = self.owner
        home_key = str(await asyncio.to_thread(Path(home).resolve)) if home is not None else None
        if worker is None or (home_key is not None and owner.profile_id != home_key):
            return True
        done, _ = await asyncio.wait({worker}, timeout=timeout)
        return bool(done) and not _worker_live(self)

    async def watch(self):
        while self.runner._running and not self.stopped.is_set():
            await self.tick()
            await asyncio.sleep(POLL_SECONDS)


def start_connected_assistance(runner):
    runtime = getattr(runner, "_connected_assistance", None)
    if runtime is None:
        runtime = runner._connected_assistance = ConnectedAssistance(runner)
        runner._spawn_supervised(runtime.watch, "connected_assistance")
    return runtime


async def stop_connected_assistance(runner, *, home=None, timeout: float = SCAN_TIMEOUT_SECONDS + 5):
    runtime = getattr(runner, "_connected_assistance", None)
    return True if runtime is None else await runtime.stop(home=home, timeout=timeout)


def assistance_worker_live(runner):
    runtime = getattr(runner, "_connected_assistance", None)
    return runtime is not None and _worker_live(runtime)


def _worker_live(runtime):
    return ((runtime.worker is not None and not runtime.worker.done())
            or (getattr(runtime, "thread", None) is not None and runtime.thread.is_alive()))
