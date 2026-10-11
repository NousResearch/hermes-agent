"""Process-local Live admission ledger; bounded, owner-bound, never persist credentials.

Terminal tombstones last five minutes; active owners renew a ten-minute lease.
In-flight tasks are retained through provider creation AND closure, never TTL-evicted.
A server restart loses this ledger: this is not durable or multi-process admission.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request


def authenticated_owner(request: Request) -> tuple:
    """Use ONLY identities verified by the existing auth middleware/checker."""
    principal = getattr(request.state, "token_principal", None)
    if getattr(request.state, "token_authenticated", False) and principal is not None:
        return ("token", principal.provider, principal.principal)
    session = getattr(request.state, "session", None)
    if session is not None:
        return ("session", session.provider, session.org_id, session.user_id)
    from hermes_cli import web_server
    if not getattr(request.app.state, "auth_required", False) and web_server._has_valid_session_token(request):
        # The loopback API has one shared authenticated dashboard identity, not per-user auth.
        return ("dashboard-token",)
    raise HTTPException(status_code=401, detail="Unauthorized")


@dataclass
class _Entry:
    key: tuple
    fingerprint: str | None = None
    state: str = "creating"
    cancelled: bool = False
    task: Any = None
    result: Any = None
    error: Any = None
    provider_id: str | None = None
    closer: Any = None
    timer: Any = None


class AdmissionLedger:
    def __init__(self, max_entries=128, max_sessions=8, tombstone_seconds: float = 300, active_seconds: float = 600):
        self.entries: dict[tuple, _Entry] = {}
        self.max_entries = max_entries
        self.max_sessions = max_sessions
        self.tombstone_seconds = tombstone_seconds
        self.active_seconds = active_seconds

    def _new(self, key, *, admission=False):
        if len(self.entries) >= self.max_entries:
            raise HTTPException(status_code=429, detail="Live admission ledger is full")
        if admission and sum(e.state in {"creating", "active", "cancel_requested", "closing"}
                             for e in self.entries.values()) >= self.max_sessions:
            raise HTTPException(status_code=429, detail="Too many Live sessions")
        entry = _Entry(key)
        self.entries[key] = entry
        return entry

    def _retire(self, entry):
        if entry.timer is not None:
            entry.timer.cancel()
        def forget():
            if self.entries.get(entry.key) is entry and (entry.task is None or entry.task.done()):
                self.entries.pop(entry.key)
        entry.timer = asyncio.get_running_loop().call_later(self.tombstone_seconds, forget)

    @staticmethod
    def _view(entry):
        if entry.cancelled:
            return {"ok": True, "cancelled": True, "state": entry.state,
                    "finalization_confirmed": entry.state == "closed"}
        if entry.error is not None:
            raise HTTPException(status_code=entry.error[0], detail=entry.error[1])
        return {"ok": True, **entry.result, "state": entry.state, "cancelled": False}

    def keepalive(self, key):
        """Renew only an existing active owner; never admit or resurrect an operation."""
        entry = self.entries.get(key)
        if entry is None:
            raise HTTPException(status_code=404, detail="Live operation not found")
        if entry.cancelled or entry.state != "active":
            raise HTTPException(status_code=409, detail="Live operation is not active")
        self._renew(entry)
        return {"ok": True, "state": "active", "renewed": True}

    def _renew(self, entry):
        if entry.timer is not None:
            entry.timer.cancel()
        entry.timer = asyncio.get_running_loop().call_later(self.active_seconds, self.cancel, entry.key)

    async def create(self, key, sdp, history, create_fn):
        raw = json.dumps([sdp, history], sort_keys=True, ensure_ascii=False).encode()
        if len(raw) > 1024 * 1024:
            raise HTTPException(status_code=413, detail="Live offer/history is too large")
        fingerprint = hashlib.sha256(raw).hexdigest()
        entry = self.entries.get(key)
        if entry is None:
            entry = self._new(key, admission=True)
            entry.fingerprint = fingerprint
            entry.task = asyncio.create_task(self._run(entry, create_fn))
        elif entry.fingerprint is not None and entry.fingerprint != fingerprint:
            raise HTTPException(status_code=409, detail="Live operation payload conflicts")
        if entry.cancelled:
            return self._view(entry)
        try:
            await asyncio.shield(entry.task)
        except asyncio.CancelledError:
            # Cancelling the HTTP await cannot cancel the blocking upstream POST.
            self.cancel(key)
            raise
        return self._view(entry)

    async def _run(self, entry, create_fn):
        if entry.cancelled:
            entry.state = "closed"
            self._retire(entry)
            return
        loop = asyncio.get_running_loop()
        def capture(provider_id, closer):
            # The executor callback only queues a mutation back on the owning event loop.
            loop.call_soon_threadsafe(self._capture, entry, provider_id, closer)
        try:
            result = await create_fn(capture)
            # Flush the capture callback queued by the completing provider worker.
            await asyncio.sleep(0)
            if entry.cancelled:
                await self._close(entry)
            else:
                entry.result, entry.state = result, "active"
                self._renew(entry)
        except Exception as exc:  # health: allow BLE001 -- provider boundary; tracebacks may disclose credentials/SDP
            # A provider can fail after queuing cleanup ownership; flush that handoff too.
            await asyncio.sleep(0)
            # No upstream exception text or body enters HTTP details or ledger state.
            entry.error = (503 if isinstance(exc, ValueError) else 502, "GPT-Live session creation failed")
            if entry.closer is not None:
                entry.cancelled = True
                await self._close(entry)
            else:
                entry.state = "finalization_unconfirmed"
                self._retire(entry)

    @staticmethod
    def _capture(entry, provider_id, closer):
        entry.provider_id, entry.closer = provider_id, closer

    async def _close(self, entry):
        entry.result = None
        entry.state = "closing"
        confirmed = False
        try:
            if entry.closer is not None:
                confirmed = await asyncio.to_thread(entry.closer)
        except Exception:  # health: allow BLE001 -- cleanup boundary; raw upstream tracebacks are privacy-sensitive
            confirmed = False
        finally:
            entry.closer = None  # release the frozen creating credentials from RAM
        entry.state = "closed" if confirmed is True else "finalization_unconfirmed"
        self._retire(entry)

    def cancel(self, key):
        entry = self.entries.get(key)
        if entry is None:
            entry = self._new(key)
            entry.cancelled, entry.state = True, "closed"  # no provider was ever admitted
            self._retire(entry)
        elif not entry.cancelled:
            entry.cancelled = True
            entry.result = None
            if entry.state == "active":
                if entry.timer is not None:
                    entry.timer.cancel()
                entry.state = "closing"
                entry.task = asyncio.create_task(self._close(entry))
            elif entry.state == "creating":
                entry.state = "cancel_requested"
        return self._view(entry)
