"""Volatile approval receipts and their adapter-lifetime generation (#89754)."""

import logging
import threading
import time
import uuid
from dataclasses import dataclass

from aiohttp import web

from gateway.platforms.api_server_room_grants import _json_error


logger = logging.getLogger("gateway.platforms.api_server")


@dataclass
class _Receipt:
    fingerprint: str
    response: tuple[int, bytes, str] | None = None
    completed_at: float | None = None


class ApprovalAdmissions:
    """Reserve before mutation; unresolved/ambiguous receipts never expire or get evicted."""

    TTL_SECONDS = 3600
    MAX_RECEIPTS = 1024

    def __init__(self):
        self.generation = str(uuid.uuid4())
        self._receipts: dict[tuple[str, str], _Receipt] = {}
        self._lock = threading.Lock()

    def reserve(self, scope, key, fingerprint):
        with self._lock:
            now = time.monotonic()
            self._receipts = {identity: receipt for identity, receipt in self._receipts.items()
                              if receipt.completed_at is None or now - receipt.completed_at < self.TTL_SECONDS}
            identity = (scope, key)
            receipt = self._receipts.get(identity)
            if receipt is not None:
                return ("replay" if receipt.fingerprint == fingerprint else "conflict"), receipt
            if len(self._receipts) >= self.MAX_RECEIPTS:
                return "full", None
            receipt = self._receipts[identity] = _Receipt(fingerprint)
            return "reserved", receipt

    def complete(self, scope, key, receipt, response, *, retryable=False):
        with self._lock:
            if 200 <= response.status < 300:
                receipt.response = (response.status, response.body, response.headers["Content-Type"])
                receipt.completed_at = time.monotonic()
            elif retryable and 400 <= response.status < 500:
                del self._receipts[(scope, key)]
            # Exceptions and ambiguous errors retain their reservation, even for a 4xx.


def check_generation(self, request, *, _openai_error):
    expected = request.headers.get("X-Hermes-Expected-Generation")
    if expected is None:
        return None
    try:
        if not expected.strip() or len(expected) > 64:
            raise ValueError
        parsed = str(uuid.UUID(expected.strip()))
    except ValueError:
        return _json_error(_openai_error, "X-Hermes-Expected-Generation is invalid", status=400)
    if parsed != self._run_admissions.generation:
        return _json_error(_openai_error, "Hermes server generation mismatch",
                           code="server_generation_mismatch", status=409)
    return None


def run_creation_headers(self, request, *, _openai_error):
    error = check_generation(self, request, _openai_error=_openai_error)
    if error is not None:
        return None, error
    return self._parse_session_key_header(request)


async def runs_meta(self, request):
    auth_error = self._check_auth(request)
    if auth_error is not None:
        return auth_error
    return web.json_response({"server_generation": self._run_admissions.generation,
                              "idempotency": "v1",
                              "idempotency_ttl_seconds": self._run_admissions.TTL_SECONDS})


def parse_idempotency_key(request, *, _openai_error, allow_empty=False):
    key = request.headers.get("Idempotency-Key", "").strip()
    if ((not allow_empty and "Idempotency-Key" in request.headers and not key)
            or len(key) > 255 or any(ord(ch) < 33 or ord(ch) > 126 for ch in key)):
        return None, _json_error(_openai_error,
            "Idempotency-Key must be 1-255 visible ASCII characters",
            code="invalid_idempotency_key", status=400)
    return key, None


def _approval_idempotency_identity(self, request, run_id, body, key):
    """Bind the approval URL and full input using the canonical scope/digest owner."""
    from gateway.platforms.api_server_runs import _run_idempotency_identity
    return _run_idempotency_identity(
        self, request, {"run_id": run_id, "approval": body}, None, key)


async def approval_response(self, request, run_id, body, *, _api_server):
    key, error = parse_idempotency_key(request, _openai_error=_api_server._openai_error)
    if error is not None:
        return error
    if not key:
        response, _ = await _resolve_approval(self, request, run_id, body, _api_server=_api_server)
        return response
    scope, fingerprint = _approval_idempotency_identity(self, request, run_id, body, key)
    outcome, receipt = self._run_admissions.reserve(scope, key, fingerprint)
    if outcome == "conflict":
        return _json_error(_api_server._openai_error, "Idempotency-Key payload conflict",
                           code="idempotency_key_conflict", status=409)
    if outcome == "full":
        return _json_error(_api_server._openai_error, "Approval receipt capacity reached",
                           code="approval_receipts_full", status=503)
    if outcome == "replay":
        if receipt.response is None:
            return _json_error(_api_server._openai_error, "Approval resolution is pending or ambiguous",
                               code="approval_resolution_pending", status=409)
        status, body_bytes, content_type = receipt.response
        return web.Response(status=status, body=body_bytes,
                            headers={"Content-Type": content_type, "Idempotency-Replayed": "true"})
    # Cancellation and exceptions intentionally retain the reservation: never repeat an ambiguous write.
    response, retryable = await _resolve_approval(self, request, run_id, body, _api_server=_api_server)
    self._run_admissions.complete(scope, key, receipt, response, retryable=retryable)
    return response


_APPROVAL_CHOICE_ALIASES = {"approve": "once", "approved": "once", "allow": "once"}


async def _resolve_approval(self, request, run_id, body, *, _api_server):
    """Return the response and whether a proven non-mutating refusal may release its key."""
    from gateway.platforms.api_server_runs import _mark_run_event, _respond_authority_run

    _openai_error = _api_server._openai_error
    if getattr(self.gateway_runner, 'session_authority', None) is not None:
        if self._room_grant_token(request) and body.get('choice') not in {'once', 'deny'}:
            return _json_error(_openai_error, 'Room approvals require once or deny', status=400), True
        dispatched = False

        def on_dispatch():
            nonlocal dispatched
            dispatched = True

        response = await _respond_authority_run(
            self, run_id, body, kind='approval', _api_server=_api_server, on_dispatch=on_dispatch)
        # The same 4xx can follow a successful mutation and a failed detach.
        return response, not dispatched and 400 <= response.status < 500
    raw_choice = str(body.get("choice", "")).strip().lower()
    choice = _APPROVAL_CHOICE_ALIASES.get(raw_choice, raw_choice)
    room_scoped = bool(self._room_grant_token(request))
    raw_request_id = body.get("request_id")
    request_id = raw_request_id.strip() if isinstance(raw_request_id, str) else ""
    # Room grants may resolve exactly one request and never widen to session/always.
    allowed = {"once", "deny"} if room_scoped else {"once", "session", "always", "deny"}
    resolve_all = any(_api_server._coerce_request_bool(body.get(k), default=False) for k in ("all", "resolve_all"))
    approval_session_key = self._run_approval_sessions.get(run_id)
    for failed, message, code, status in (
        (raw_request_id is not None and (not request_id or len(request_id) > 256),
         "Approval request_id is invalid.", "invalid_approval_request", 400),
        (choice not in allowed,
         "Invalid approval choice; expected one of: " + ", ".join(sorted(allowed)),
         "invalid_approval_choice", 400),
        (room_scoped and resolve_all,
         "Room approvals can resolve only one exact request", "invalid_approval_scope", 400),
        (room_scoped and not request_id,
         "Room approvals require the exact request_id.", "approval_request_required", 400),
        (not approval_session_key,
         f"Run has no active approval session: {run_id}", "approval_not_active", 409)):
        if failed:
            return _json_error(_openai_error, message, code=code, status=status), True
    try:
        from tools.approval import resolve_gateway_approval
        resolved = resolve_gateway_approval(
            approval_session_key, choice, resolve_all=resolve_all, request_id=request_id or None)
    except Exception as exc:
        logger.exception("[api_server] approval resolution failed for run %s", run_id)
        return _json_error(_openai_error, str(exc), status=500), False
    if resolved <= 0:
        return _json_error(
            _openai_error, f"Run has no pending approval: {run_id}", code="approval_not_pending", status=409), True
    request_id_field = {"request_id": request_id} if request_id else {}
    _mark_run_event(self, run_id, "approval.responded", choice=choice, **request_id_field, resolved=resolved)
    return web.json_response({
        "object": "hermes.run.approval_response", "run_id": run_id, "choice": choice, **request_id_field,
        "resolved": resolved}), False
