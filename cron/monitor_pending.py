"""Durable, bounded monitor attempts; uncertainty never authorizes a replay.

Only safe_retry monitors use this private profile-local store. A running attempt
surviving its caller is deliberately unresolved, even when its process died.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import os
import stat
import sqlite3
import uuid

from hermes_constants import get_hermes_home

MAX_ATTEMPTS = 2
MAX_PAYLOAD_BYTES = 262_144


class RecoveryRequired(RuntimeError):
    """An observation is retained and requires outcome reconciliation."""


def _binding(job):
    keys = ("monitor_script", "monitor_url", "script", "prompt", "provider", "model",
            "provider_snapshot", "model_snapshot", "base_url", "workdir", "deliver", "origin",
            "skills", "skill", "enabled_toolsets", "context_from", "no_agent")
    data = json.dumps({k: job.get(k) for k in keys}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


@contextmanager
def _store():
    from cron.jobs import _ensure_cron_dir
    directory = get_hermes_home() / "cron"
    _ensure_cron_dir(directory)
    path = directory / "monitor_pending.db"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    metadata = path.lstat()
    owner_uid = os.getuid() if hasattr(os, "getuid") else None
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or (os.name != "nt" and (metadata.st_mode & 0o077 or metadata.st_uid != owner_uid))):
        raise RecoveryRequired("monitor recovery store is not a private regular file")
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS pending (job TEXT PRIMARY KEY, binding TEXT NOT NULL, "
                   "hash TEXT NOT NULL, output TEXT NOT NULL, state TEXT NOT NULL, "
                   "attempts INTEGER NOT NULL, token TEXT NOT NULL, response TEXT, reason TEXT)")
        db.execute("BEGIN IMMEDIATE")
        yield db


def _row(db, job):
    row = db.execute("SELECT * FROM pending WHERE job=?", (job["id"],)).fetchone()
    if row is not None:
        try:
            valid = (
                row["state"] in {"running", "retry", "ready", "unknown", "acked"}
                and type(row["attempts"]) is int and 1 <= row["attempts"] <= MAX_ATTEMPTS
                and len(row["token"]) == 32 and all(c in "0123456789abcdef" for c in row["token"])
                and isinstance(row["output"], str) and len(row["output"].encode()) <= MAX_PAYLOAD_BYTES
                and hashlib.sha256(row["output"].encode()).hexdigest() == row["hash"]
                and (row["response"] is None or (
                    isinstance(row["response"], str) and len(row["response"].encode()) <= MAX_PAYLOAD_BYTES))
            )
        except (TypeError, UnicodeError):
            valid = False
        if not valid:
            raise RecoveryRequired("monitor recovery record is invalid; pending data retained")
    if row is not None and row["state"] != "acked" and row["binding"] != _binding(job):
        raise RecoveryRequired("monitor recovery configuration changed; pending observation retained")
    return dict(row) if row is not None else None


def _claim(db, row):
    if row["state"] != "retry" or row["attempts"] >= MAX_ATTEMPTS:
        raise RecoveryRequired("monitor recovery required: " + row["state"])
    row.update(state="running", attempts=row["attempts"] + 1, token=uuid.uuid4().hex)
    db.execute("UPDATE pending SET state=?, attempts=?, token=? WHERE job=?",
               (row["state"], row["attempts"], row["token"], row["job"]))
    return row


def resume(job):
    """Claim a positively retryable observation before consulting a newer source."""
    with _store() as db:
        row = _row(db, job)
        if row is None or row["state"] == "acked":
            return None
        return _claim(db, row)


def begin(job, output):
    """Persist an observation and claim its first attempt in one transaction."""
    if len(output.encode()) > MAX_PAYLOAD_BYTES:
        raise RecoveryRequired("monitor observation exceeds durable payload limit")
    digest = hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()
    with _store() as db:
        old = _row(db, job)
        if old and old["state"] != "acked":
            raise RecoveryRequired("monitor observation already pending")
        if old and old["binding"] == _binding(job) and old["hash"] == digest:
            return None
        row = {"job": job["id"], "binding": _binding(job), "hash": digest, "output": output,
               "state": "running", "attempts": 1, "token": uuid.uuid4().hex,
               "response": None, "reason": None}
        db.execute("INSERT OR REPLACE INTO pending VALUES (:job,:binding,:hash,:output,:state,"
                   ":attempts,:token,:response,:reason)", row)
        return row


def settle(job, token, *, response=None, safe_failure=False, completed=False):
    """Settle generation, never infer safety from an error string or elapsed time."""
    if response is not None and len(response.encode()) > MAX_PAYLOAD_BYTES:
        raise RecoveryRequired("monitor response exceeds durable payload limit")
    with _store() as db:
        row = _row(db, job)
        if not row or row["token"] != token or row["state"] != "running":
            raise RecoveryRequired("monitor attempt ownership changed")
        if response is not None:
            state, reason = ("ready", None) if completed else ("unknown", "incomplete_generation")
        elif safe_failure and row["attempts"] < MAX_ATTEMPTS:
            state, reason = "retry", "confirmed_pre_effect_failure"
        else:
            state, reason = "unknown", "attempt_limit" if safe_failure else "effects_not_excluded"
        db.execute("UPDATE pending SET state=?, response=?, reason=? WHERE job=? AND token=?",
                   (state, response, reason, job["id"], token))


def acknowledge(job, token):
    """Called only after successful delivery and the owner-fenced monitor commit."""
    with _store() as db:
        row = _row(db, job)
        if not row or row["token"] != token or row["state"] != "ready":
            raise RecoveryRequired("monitor acknowledgement ownership changed")
        db.execute("UPDATE pending SET state='acked', reason=NULL WHERE job=? AND token=?",
                   (job["id"], token))


def inspect(job):
    """Internal read-back; payloads must not be projected onto public job summaries."""
    with _store() as db:
        return _row(db, job)


class Attempt:
    """Observe this built-in agent invocation without treating an error as safety proof."""

    def __init__(self, job):
        import threading
        pending = job.get("_monitor_pending_commit") or {}
        self.token = pending.get("token") if job.get("monitor_commit_policy") == "safe_retry" else None
        self.job = job
        self.tools_started = threading.Event()
        self.started = False
        self.result = None
        self.response = None

    def attach(self, agent):
        if not self.token:
            return
        # An external runtime can execute tools before its event stream reports
        # them. Missing callbacks never prove that its failed turn had no effect.
        if (getattr(agent, "api_mode", None) == "codex_app_server"
                or str(getattr(agent, "base_url", "")).lower().startswith(("acp://", "acp+tcp://"))):
            self.tools_started.set()
        execute_tools = getattr(agent, "_execute_tool_calls", None)
        if callable(execute_tools):
            def guarded_tools(*args, **kwargs):
                self.tools_started.set()
                return execute_tools(*args, **kwargs)
            agent._execute_tool_calls = guarded_tools
        original = getattr(agent, "tool_start_callback", None)

        def on_start(*args, **kwargs):
            self.tools_started.set()
            if original is not None:
                original(*args, **kwargs)

        agent.tool_start_callback = on_start

    def finish(self, worker):
        if not self.token:
            return
        future = worker.get("future")
        finished = future is None or future.done()
        result = self.result or {}
        requested_retry = (self.response or "").strip() == "[MONITOR_RETRY]"
        safe_failure = (
            not self.job.get("script") and not self.tools_started.is_set() and finished
            and (requested_retry or (
                result.get("failed") is True
                and result.get("turn_exit_reason") != "interrupted"))
        )
        # A still-running watchdog worker may execute a tool later. Its absence
        # from a partial transcript is never evidence that retry is safe.
        settle(self.job, self.token, response=self.response if finished and not requested_retry else None,
               safe_failure=safe_failure, completed=result.get("completed") is True and result.get("failed") is not True)
