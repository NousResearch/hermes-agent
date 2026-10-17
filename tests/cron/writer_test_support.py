"""Private proposal harness for real synchronous Cron file-writer custody.

This file is proposal-only and is never imported by the author.  A future test commit may place
it beside the three canonical tests.  One fixed ``spawn`` child imports the real producer modules
once, then serves actual writer commands over queues.  The parent owns the real native exclusive
lease before dispatching every blocking case; ``entered`` is only IPC sequencing.
"""
from __future__ import annotations

from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
import json
import os
from typing import Any, Iterator

from hermes_cron_export_custody import cron_capture_custody


_CHILD_BOOT_TIMEOUT_SECONDS = 45.0
_CHILD_RELEASE_TIMEOUT_SECONDS = 30.0
_NO_COMPLETION_TIMEOUT_SECONDS = 2.25


@contextmanager
def profile_env(home: Path) -> Iterator[None]:
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(home)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous


@contextmanager
def _store_scope(store_home: str | None) -> Iterator[None]:
    from cron import jobs

    if store_home is None:
        yield
        return
    with jobs.use_cron_store(Path(store_home)):
        yield


@contextmanager
def profile_store(home: Path, *, store_home: Path | None = None) -> Iterator[None]:
    with profile_env(home), _store_scope(str(store_home) if store_home else None):
        yield


def _dispatch(operation: str, home: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call finite, actual source producer entrypoints with provider-free values."""
    store_home = args.get("store_home")
    with _store_scope(str(store_home) if store_home else None):
        if operation == "seed_job":
            from cron import jobs

            job = jobs.create_job(
                prompt="opaque writer setup prompt",
                schedule="every 1h",
                name=str(args.get("name", "opaque writer setup")),
                deliver="local",
            )
            return {"id": job["id"], "name": job["name"]}

        if operation == "ticker_heartbeat":
            from cron import jobs

            jobs.record_ticker_heartbeat(success=True)
            return {"written": True}
        if operation == "ticker_error":
            from cron import jobs

            jobs.record_ticker_error("opaque ticker failure")
            return {"written": True}
        if operation == "ticker_clear":
            from cron import jobs

            jobs.clear_ticker_error()
            return {"cleared": True}
        if operation == "telemetry_recovery":
            from cron import jobs

            jobs._record_persisted_error_recovery(
                {"id": "opaque-recovery-job", "name": "opaque recovery job"},
                "opaque previous next run",
            )
            return {"written": True}
        if operation == "telemetry_timezone":
            from datetime import UTC, datetime
            from cron import jobs

            jobs._record_timezone_migration_catchup(
                {
                    "id": "opaque-timezone-job",
                    "name": "opaque timezone job",
                    "schedule": {"expr": "every 1h"},
                },
                datetime(2026, 10, 17, 5, tzinfo=UTC),
                datetime(2026, 10, 17, 6, tzinfo=UTC),
            )
            return {"written": True}
        if operation == "occurrence":
            from cron import occurrences

            occurrences.record_catch_up_occurrence()
            return {"written": True}

        if operation == "suggestion_add":
            from cron import suggestions

            record = suggestions.add_suggestion(
                title=str(args.get("title", "opaque writer suggestion")),
                description="opaque writer description",
                source="catalog",
                job_spec={
                    "prompt": "opaque suggested prompt",
                    "schedule": "every 1h",
                    "deliver": "local",
                },
                dedup_key=str(args.get("dedup_key", "opaque-writer-suggestion")),
            )
            if record is None:
                raise AssertionError("actual suggestion writer rejected the unique proposal")
            return {"id": record["id"], "status": record["status"]}
        if operation == "suggestion_set_status":
            from cron import suggestions

            return {
                "updated": suggestions._set_status(
                    str(args["suggestion_id"]), str(args["status"])
                )
            }
        if operation == "suggestion_dismiss":
            from cron import suggestions

            return {"updated": suggestions.dismiss_suggestion(str(args["suggestion_id"]))}
        if operation == "suggestion_clear":
            from cron import suggestions

            return {"removed": suggestions.clear_resolved()}

        if operation == "scheduler_forced_release":
            from cron import scheduler

            scheduler._record_forced_release(
                "opaque-forced-job", "opaque forced job", 7.5, 60.0
            )
            return {"written": True}
        if operation == "scheduler_usage_audit":
            from cron import scheduler

            scheduler._write_usage_audit(
                {
                    "event": "opaque usage audit",
                    "job_id": "opaque-audit-job",
                    "result": {"status": "opaque"},
                }
            )
            return {"written": True}

        if operation == "authority_save":
            from cron import scheduler_authority

            job_id = str(args["job_id"])
            request_id = str(args["request_id"])
            scheduler_authority._save_admission_journal(
                job_id,
                request_id,
                {
                    "params": {"job_id": job_id, "request_id": request_id},
                    "receipt": None,
                    "opaque": "opaque authority value",
                },
            )
            return {"written": True}
        if operation == "authority_remove":
            from cron import scheduler_authority

            scheduler_authority._remove_admission_journal(
                str(args["job_id"]), str(args["request_id"])
            )
            return {"removed": True}
        if operation == "bot_defer":
            from cron import bot_chat_legacy

            target_home = Path(str(args["target_home"]))
            bot_chat_legacy.defer(
                "opaque-bot-pending",
                {"id": "opaque-bot-job", "name": "opaque bot job"},
                "opaque bot pending content",
                target_home,
                for_failure=False,
            )
            return {"deferred": True}
        if operation == "seed_bot_malformed":
            from cron import bot_chat_legacy

            bot_chat_legacy._persist_pending_record(
                "opaque-malformed.json", {"status": "queued", "job": {}}
            )
            return {"seeded": True}
        if operation == "bot_drain_malformed":
            from cron import bot_chat_legacy

            bot_chat_legacy.drain_legacy_pending()
            return {"drained": True}

        return _dispatch_monitor(operation, args)


def _dispatch_monitor(operation: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch the one provider-free monitor mutation seam."""
    if operation != "monitor_state":
        raise AssertionError(f"unknown proposal operation: {operation!r}")
    from cron import monitor

    monitor._persist_monitor_state(
        str(args["job_id"]), "opaque-monitor-hash", "opaque monitor output\n"
    )
    return {"written": True}


def _writer_child(home: str, requests: Any, responses: Any) -> None:
    """Spawn target: import the producer family once, then accept real commands."""
    os.environ["HERMES_HOME"] = home
    from cron import (
        bot_chat_legacy,
        jobs,
        monitor,
        occurrences,
        scheduler,
        scheduler_authority,
        suggestions,
    )

    responses.put({"phase": "boot", "ok": True})
    while True:
        request = requests.get()
        if request == {"operation": "stop"}:
            responses.put({"phase": "stopped", "ok": True})
            return
        operation = str(request["operation"])
        args = dict(request.get("args") or {})
        responses.put({"phase": "entered", "operation": operation})
        try:
            result = _dispatch(operation, home, args)
        except BaseException as exc:
            responses.put(
                {
                    "phase": "done",
                    "operation": operation,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            responses.put(
                {"phase": "done", "operation": operation, "ok": True, "result": result}
            )


class WriterChild:
    """One exact owned child per test; commands reuse its already-booted imports."""

    def __init__(self, home: Path) -> None:
        self.home = home
        context = get_context("spawn")
        self._requests = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(
            target=_writer_child,
            args=(str(home), self._requests, self._responses),
            daemon=True,
        )

    def __enter__(self) -> WriterChild:
        try:
            self._process.start()
            boot = self._responses.get(timeout=_CHILD_BOOT_TIMEOUT_SECONDS)
            if boot != {"phase": "boot", "ok": True}:
                raise AssertionError(f"unexpected child boot response: {boot!r}")
        except BaseException:
            self._shutdown_owned_process()
            raise
        return self

    def _shutdown_owned_process(self) -> BaseException | None:
        """Stop this child within bounded waits, retaining the first cleanup failure."""
        cleanup_error: BaseException | None = None

        def keep_first(error: BaseException) -> None:
            nonlocal cleanup_error
            if cleanup_error is None:
                cleanup_error = error

        try:
            alive = self._process.is_alive()
        except BaseException as exc:
            alive = False
            keep_first(exc)
        if alive:
            try:
                self._requests.put({"operation": "stop"})
            except BaseException as exc:
                keep_first(exc)
            try:
                stopped = self._responses.get(timeout=_CHILD_RELEASE_TIMEOUT_SECONDS)
                if stopped != {"phase": "stopped", "ok": True}:
                    raise AssertionError(f"unexpected child stop response: {stopped!r}")
            except BaseException as exc:
                keep_first(exc)
            try:
                self._process.join(timeout=_CHILD_RELEASE_TIMEOUT_SECONDS)
            except BaseException as exc:
                keep_first(exc)
        try:
            alive = self._process.is_alive()
        except BaseException as exc:
            alive = False
            keep_first(exc)
        if alive:
            try:
                self._process.terminate()
            except BaseException as exc:
                keep_first(exc)
            try:
                self._process.join(timeout=_CHILD_RELEASE_TIMEOUT_SECONDS)
            except BaseException as exc:
                keep_first(exc)
        try:
            if self._process.is_alive() and cleanup_error is None:
                cleanup_error = AssertionError("owned writer child survived termination")
        except BaseException as exc:
            keep_first(exc)

        for queue in (self._requests, self._responses):
            try:
                queue.close()
            except BaseException as exc:
                keep_first(exc)
            try:
                queue.join_thread()
            except BaseException as exc:
                keep_first(exc)
        return cleanup_error

    def __exit__(self, exc_type, exc, tb) -> bool:
        cleanup_error = self._shutdown_owned_process()
        if exc_type is None and cleanup_error is not None:
            raise cleanup_error
        return False

    def run_unfenced(
        self,
        operation: str,
        args: dict[str, Any] | None = None,
        *,
        expect_ok: bool = True,
    ) -> dict[str, Any]:
        self._requests.put({"operation": operation, "args": args or {}})
        entered = self._responses.get(timeout=_CHILD_BOOT_TIMEOUT_SECONDS)
        assert entered == {"phase": "entered", "operation": operation}
        done = self._responses.get(timeout=_CHILD_RELEASE_TIMEOUT_SECONDS)
        assert done["phase"] == "done" and done["operation"] == operation
        assert bool(done.get("ok")) is expect_ok, json.dumps(done, sort_keys=True)
        return done if not expect_ok else dict(done.get("result") or {})

    def run_under_exclusive(
        self,
        operation: str,
        args: dict[str, Any] | None = None,
        *,
        lease_home: Path,
    ) -> dict[str, Any]:
        """Hold a real EX lease before dispatch; require completion only after release."""
        with profile_env(lease_home):
            with cron_capture_custody(timeout_seconds=2.0):
                self._requests.put({"operation": operation, "args": args or {}})
                entered = self._responses.get(timeout=_CHILD_BOOT_TIMEOUT_SECONDS)
                assert entered == {"phase": "entered", "operation": operation}
                try:
                    early = self._responses.get(timeout=_NO_COMPLETION_TIMEOUT_SECONDS)
                except Empty:
                    early = None
                if early is not None:
                    raise AssertionError(
                        f"{operation} completed while its real EX lease was held: "
                        f"{early}"
                    )
        done = self._responses.get(timeout=_CHILD_RELEASE_TIMEOUT_SECONDS)
        assert done["phase"] == "done" and done["operation"] == operation
        assert done.get("ok"), json.dumps(done, sort_keys=True)
        return dict(done.get("result") or {})


def journal_bytes(home: Path, job_id: str, request_id: str) -> bytes:
    from cron import scheduler_authority

    with profile_env(home):
        return scheduler_authority.journal_path(job_id, request_id).read_bytes()
