"""Durable, authenticated execution registry for inbound webhooks.

Execution status/cancel use a scoped capability token returned once in the 202
accept response. The plaintext token is never persisted and cannot be recovered
from the ledger; losing it intentionally means losing remote execution-control
authority. Public projections omit the backing session key and provider delivery
identifier. Authorization attempts are rate-limited at this shared registry
boundary so both status and cancel inherit the same brute-force ceiling.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "interrupted"})
ACTIVE_STATES = frozenset({"accepted", "running", "cancelling"})
_AUTH_WINDOW_SECONDS = 60.0
_AUTH_MAX_ATTEMPTS = 60
logger = logging.getLogger(__name__)


def _safe_error(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)[:4096]
    try:
        from agent.redact import redact_sensitive_text
        text = redact_sensitive_text(text, force=True)
    except Exception:
        pass
    return text[:1024]


class WebhookExecutionRegistry:
    """Persist observable execution state and bind it to the real agent task."""

    def __init__(
        self,
        path: Path,
        ttl_seconds: int = 3600,
        max_records: int = 4096,
        *,
        reconcile_restart: bool = True,
        read_only: bool = False,
    ):
        self.path = Path(path)
        self.ttl_seconds = int(ttl_seconds)
        self.max_records = int(max_records)
        self._reconcile_restart = bool(reconcile_restart)
        self._read_only = bool(read_only)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._auth_attempts: dict[tuple[str, str, str], deque[float]] = {}
        self._load()

    def _quarantine_corrupt(self) -> None:
        if self._read_only or not self.path.exists():
            return
        quarantine = self.path.with_name(
            f"{self.path.name}.corrupt-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
        )
        try:
            os.replace(self.path, quarantine)
        except OSError:
            pass

    def _ensure_writable(self) -> None:
        if self._read_only:
            raise RuntimeError("execution registry is read-only")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("webhook execution ledger is corrupt: %s", self.path)
            self._quarantine_corrupt()
            return
        if not isinstance(raw, dict):
            logger.warning("webhook execution ledger is not an object: %s", self.path)
            self._quarantine_corrupt()
            return
        now = time.time()
        changed = False
        for execution_id, record in raw.items():
            if not isinstance(record, dict):
                changed = True
                continue
            item = dict(record)
            if self._reconcile_restart and item.get("state") in ACTIVE_STATES:
                item["state"] = "interrupted"
                item["finished_at"] = now
                item["error"] = "gateway restarted before terminal state"
                changed = True
            self._records[str(execution_id)] = item
        changed = self.prune(now, persist=False) or changed
        if changed and not self._read_only:
            self._persist()

    def _persist(self) -> None:
        self._ensure_writable()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._records, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp, 0o600)
            os.replace(temp, self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        except BaseException:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        # session_key embeds the webhook chat/session identity and delivery_id is
        # a provider-controlled retry identifier. Neither is required to expose
        # execution state to the capability holder.
        hidden = {"token_hash", "session_key", "delivery_id"}
        return {key: value for key, value in dict(record).items() if key not in hidden}

    def _record_auth_attempt(
        self, execution_id: str, *, profile: str, route: str, now: float | None = None
    ) -> bool:
        """Rate-limit status/cancel authority checks at their shared seam."""
        now = time.time() if now is None else now
        key = (profile, route, execution_id)
        with self._lock:
            window = self._auth_attempts.setdefault(key, deque())
            cutoff = now - _AUTH_WINDOW_SECONDS
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= _AUTH_MAX_ATTEMPTS:
                return False
            window.append(now)
            if not window:
                self._auth_attempts.pop(key, None)
            return True

    def accept(
        self,
        *,
        profile: str,
        route: str,
        provider: str,
        delivery_id: str,
        session_key: str,
    ) -> dict[str, Any]:
        self._ensure_writable()
        now = time.time()
        token = secrets.token_urlsafe(32)
        identity = (
            f"{profile}\0{route}\0{provider}\0{delivery_id}\0{now}\0{secrets.token_hex(8)}"
        )
        execution_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        record = {
            "execution_id": execution_id,
            "profile": profile,
            "route": route,
            "provider": provider,
            "delivery_id": delivery_id,
            "session_key": session_key,
            "state": "accepted",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "token_hash": self._token_hash(token),
        }
        with self._lock:
            self._records[execution_id] = record
            self.prune(now)
            self._persist()
        return {**self.public(execution_id), "access_token": token}

    def authorize(self, execution_id: str, token: str) -> bool:
        if not token:
            return False
        with self._lock:
            record = self._records.get(execution_id)
            if not record:
                return False
            profile = str(record.get("profile") or "")
            route = str(record.get("route") or "")
        return self.authorize_scoped(
            execution_id, token, profile=profile, route=route
        )

    def authorize_scoped(
        self, execution_id: str, token: str, *, profile: str, route: str
    ) -> bool:
        # Both GET status and POST cancel call this exact authority seam. Apply
        # the ceiling before token comparison so guessed/invalid tokens cannot
        # brute-force an execution capability without consuming budget.
        if not self._record_auth_attempt(execution_id, profile=profile, route=route):
            logger.warning(
                "webhook execution authorization rate limit exceeded: profile=%r route=%r",
                profile,
                route,
            )
            return False
        if not token:
            return False
        with self._lock:
            record = self._records.get(execution_id)
            if not record:
                return False
            if str(record.get("profile")) != profile or str(record.get("route")) != route:
                return False
            expected = str(record.get("token_hash", ""))
        return hmac.compare_digest(self._token_hash(token), expected)

    def bind(self, execution_id: str, task: asyncio.Task) -> bool:
        """Bind one accepted execution to its exact agent-processing task.

        Binding is deliberately fail-soft after the real task exists. A ledger
        persistence failure must never propagate out of the lifecycle hook and
        orphan that task from the caller's normal tracking/shutdown path.
        """
        self._ensure_writable()
        cancel_immediately = False
        with self._lock:
            record = self._records.get(execution_id)
            if record is None or record.get("state") in TERMINAL_STATES:
                return False
            existing = self._tasks.get(execution_id)
            if existing is not None:
                return existing is task
            self._tasks[execution_id] = task
            cancel_immediately = record.get("state") == "cancelling"
            if not cancel_immediately:
                record["state"] = "running"
            record["started_at"] = record.get("started_at") or time.time()
            try:
                self._persist()
            except Exception:
                logger.exception(
                    "webhook execution ledger persist failed while binding %s; "
                    "real task remains tracked in memory",
                    execution_id,
                )

        def _done(real_task: asyncio.Task) -> None:
            try:
                if real_task.cancelled():
                    self.finish(execution_id, "cancelled")
                else:
                    error = real_task.exception()
                    self.finish(
                        execution_id,
                        "failed" if error else "completed",
                        error=_safe_error(error),
                    )
            except asyncio.CancelledError:
                self.finish(execution_id, "cancelled")
            except Exception as exc:
                logger.exception(
                    "webhook execution terminal persistence failed for %s",
                    execution_id,
                )
                with self._lock:
                    record = self._records.get(execution_id)
                    if record is not None and record.get("state") not in TERMINAL_STATES:
                        record["state"] = "failed"
                        record["error"] = _safe_error(exc)
                        record["finished_at"] = time.time()
                    self._tasks.pop(execution_id, None)

        task.add_done_callback(_done)
        if cancel_immediately and not task.done():
            task.cancel()
        return True

    def is_bound(self, execution_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(execution_id)
            return task is not None

    def finish_if_unbound(
        self, execution_id: str, state: str, error: str | None = None
    ) -> bool:
        """Finish only when the dispatcher failed before creating the real task."""
        self._ensure_writable()
        if state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal state: {state}")
        with self._lock:
            record = self._records.get(execution_id)
            if (
                record is None
                or record.get("state") in TERMINAL_STATES
                or execution_id in self._tasks
            ):
                return False
            record["state"] = state
            record["error"] = _safe_error(error)
            record["finished_at"] = time.time()
            self._persist()
            return True

    def finish(self, execution_id: str, state: str, error: str | None = None) -> bool:
        self._ensure_writable()
        if state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal state: {state}")
        with self._lock:
            record = self._records.get(execution_id)
            if record is None or record.get("state") in TERMINAL_STATES:
                return False
            record["state"] = state
            record["error"] = _safe_error(error)
            record["finished_at"] = time.time()
            self._tasks.pop(execution_id, None)
            self._persist()
            return True

    def request_cancel(self, execution_id: str) -> str:
        self._ensure_writable()
        with self._lock:
            record = self._records.get(execution_id)
            if record is None:
                return "unknown"
            state = str(record.get("state"))
            if state in TERMINAL_STATES:
                return state
            task = self._tasks.get(execution_id)
            record["state"] = "cancelling"
            self._persist()
            if task is not None and not task.done():
                task.cancel()
            return "cancelling"

    def public(self, execution_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(execution_id)
            if record is None:
                raise KeyError(execution_id)
            return self._public_record(record)

    def prune(
        self, now: float | None = None, *, persist: bool = True
    ) -> bool:
        now = time.time() if now is None else now
        changed = False
        with self._lock:
            cutoff = now - self.ttl_seconds

            # A genuinely live task is never aged out. But an ACTIVE ledger row
            # with no live task past the TTL is an orphan and must not pin the
            # registry forever. Convert it to truthful interrupted state first;
            # it then participates in the normal terminal retention/overflow
            # policy without pretending a cancellation occurred.
            for execution_id, record in list(self._records.items()):
                if record.get("state") not in ACTIVE_STATES:
                    continue
                created = float(record.get("started_at") or record.get("created_at") or 0)
                if created >= cutoff:
                    continue
                task = self._tasks.get(execution_id)
                if task is not None and not task.done():
                    continue
                record["state"] = "interrupted"
                record["finished_at"] = now
                record["error"] = "execution record exceeded active TTL without a live task"
                self._tasks.pop(execution_id, None)
                changed = True

            terminal = [
                (execution_id, record)
                for execution_id, record in self._records.items()
                if record.get("state") in TERMINAL_STATES
            ]
            for execution_id, record in terminal:
                finished = float(
                    record.get("finished_at") or record.get("created_at") or 0
                )
                if finished < cutoff:
                    self._records.pop(execution_id, None)
                    self._auth_attempts = {
                        key: value
                        for key, value in self._auth_attempts.items()
                        if key[2] != execution_id
                    }
                    changed = True

            overflow = len(self._records) - self.max_records
            if overflow > 0:
                candidates = sorted(
                    (
                        (execution_id, record)
                        for execution_id, record in self._records.items()
                        if record.get("state") in TERMINAL_STATES
                    ),
                    key=lambda item: float(
                        item[1].get("finished_at")
                        or item[1].get("created_at")
                        or 0
                    ),
                )
                for execution_id, _ in candidates[:overflow]:
                    self._records.pop(execution_id, None)
                    self._auth_attempts = {
                        key: value
                        for key, value in self._auth_attempts.items()
                        if key[2] != execution_id
                    }
                    changed = True

            if changed and persist and not self._read_only:
                self._persist()
        return changed

    def list_public(
        self, *, profile: str | None = None, route: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            records = [
                self._public_record(record)
                for record in self._records.values()
                if (profile is None or record.get("profile") == profile)
                and (route is None or record.get("route") == route)
            ]
        records.sort(
            key=lambda item: float(item.get("created_at") or 0), reverse=True
        )
        return records
