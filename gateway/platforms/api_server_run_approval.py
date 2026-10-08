"""Live approval decisions, durable receipts, and recovery admission for API runs."""

import logging
import uuid

from gateway.platforms.api_server_room_grants import _json_error
from gateway.platforms.api_server_runs import (
    _load_owned_run,
    _mark_run_event,
    _publish_run_event,
    _reconcile_session_tool_receipt,
    _run_event,
    _schedule_recovery_run,
)

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


logger = logging.getLogger("gateway.platforms.api_server")
_APPROVAL_CHOICE_ALIASES = {"approve": "once", "approved": "once", "allow": "once"}


async def _handle_run_approval(self, request: "web.Request", *, _api_server) -> "web.Response":
    """POST /v1/runs/{run_id}/approval — resolve a pending run approval."""
    _openai_error = _api_server._openai_error
    run_id, _, _, _, err = _load_owned_run(
        self, request, _api_server=_api_server, permission="approve", active_fallback=False)
    if err is not None:
        return err
    try:
        body = await request.json()
    except Exception:
        return _json_error(_openai_error, "Invalid JSON", status=400)
    raw_choice = str(body.get("choice", "")).strip().lower()
    choice = _APPROVAL_CHOICE_ALIASES.get(raw_choice, raw_choice)
    room_scoped = bool(self._room_grant_token(request))
    raw_request_id = body.get("request_id")
    request_id = raw_request_id.strip() if isinstance(raw_request_id, str) else ""
    # Room grants may resolve exactly one request and never widen to session/always.
    allowed = {"once", "deny"} if room_scoped else {"once", "session", "always", "deny"}
    resolve_all = any(_api_server._coerce_request_bool(body.get(k), default=False) for k in ("all", "resolve_all"))
    approval_session_key = self._run_approval_sessions.get(run_id)
    scope = (
        self._run_idempotency_scope(request)
        if run_id in self._run_idempotency_ids
        else ""
    )
    durable_approval = self._run_idempotency_store.approval_for_run(
        scope, run_id, request_id
    ) if run_id in self._run_idempotency_ids else None
    durable_pending = durable_approval if (durable_approval or {}).get("state") == "pending" else None
    durable_request_id = str((durable_approval or {}).get("request_id") or "")
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
        (not approval_session_key and durable_approval is None,
         f"Run has no active approval session: {run_id}", "approval_not_active", 409)):
        if failed:
            return _json_error(_openai_error, message, code=code, status=status)
    receipt_request_id = request_id or durable_request_id
    if run_id in self._run_idempotency_ids and receipt_request_id:
        # The durable choice is the authority fence. It must commit before a live
        # waiter can wake and before a replacement can reserve continuation work.
        outcome, receipt = self._run_idempotency_store.resolve_approval(
            scope,
            run_id,
            receipt_request_id,
            choice,
            applied=False,
            resolved=0,
        )
        if outcome == "conflict":
            return _json_error(
                _openai_error,
                "Approval request already has a different decision.",
                code="approval_decision_conflict",
                status=409,
            )
        if outcome == "missing" or receipt is None:
            return _json_error(
                _openai_error, f"Run has no pending approval: {run_id}", code="approval_not_pending", status=409)
        resolved = 0
        should_deliver = bool(
            approval_session_key
            and self._run_idempotency_store.approval_dispatch_state(
                scope, run_id, receipt_request_id
            ) == "pending"
        )
        if should_deliver:
            dispatch_fenced = False
            try:
                if choice != "deny":
                    dispatch_fenced = self._run_idempotency_store.mark_tool_dispatching(
                        run_id, receipt_request_id
                    )
                    if not dispatch_fenced:
                        return _json_error(
                            _openai_error,
                            "Approval dispatch fence could not be acquired.",
                            code="approval_dispatch_fence_failed",
                            status=409,
                        )
                from tools.approval import resolve_gateway_approval
                resolved = resolve_gateway_approval(
                    approval_session_key,
                    choice,
                    resolve_all=resolve_all,
                    request_id=receipt_request_id,
                )
            except Exception as exc:
                logger.exception("[api_server] approval resolution failed for run %s", run_id)
                return _json_error(_openai_error, str(exc), status=500)
            if resolved > 0:
                self._run_idempotency_store.mark_approval_applied(
                    run_id, receipt_request_id, resolved=resolved
                )
                if choice == "deny":
                    self._run_idempotency_store.mark_tool_skipped(
                        run_id,
                        receipt_request_id,
                        {
                            "output": "[Tool execution denied by the user.]",
                            "denied": True,
                        },
                    )
                receipt = {**receipt, "applied": True, "resolved": resolved}
            elif durable_pending is not None:
                if dispatch_fenced:
                    self._run_idempotency_store.reset_tool_dispatching(
                        run_id, receipt_request_id
                    )
                return _json_error(
                    _openai_error,
                    f"Run has no pending approval: {run_id}",
                    code="approval_not_pending",
                    status=409,
                )
        if outcome == "created":
            fields = {
                "choice": choice,
                "request_id": receipt_request_id,
                "resolved": resolved,
                "applied": resolved > 0,
            }
            _publish_run_event(
                self, run_id, _run_event(run_id, "approval.responded", **fields)
            )
        recovery = None
        if not approval_session_key:
            try:
                await _reconcile_session_tool_receipt(self, scope, run_id)
                recovery = self._run_idempotency_store.reserve_recovery_successor(
                    scope,
                    run_id,
                    successor_run_id=f"run_{uuid.uuid4().hex}",
                    owner_pid=self._run_owner_pid,
                    owner_started=self._run_owner_started,
                )
            except ValueError:
                # Compatibility for receipts written before frozen tool recovery
                # shipped: retain conflict-safe replay without inventing a tool.
                recovery = None
            if recovery is not None and recovery.get("state") != "unrecoverable":
                parent_record = self._run_idempotency_store.status_for_run(scope, run_id)
                if parent_record is not None:
                    self._run_statuses[run_id] = dict(parent_record["status"])
                _schedule_recovery_run(self, recovery, _api_server=_api_server)
        response = {
            "object": "hermes.run.approval_response",
            **receipt,
            "replayed": outcome == "replayed",
        }
        if recovery is not None:
            response.update(
                successor_run_id=recovery.get("successor_run_id"),
                recovery_status=recovery.get("state"),
            )
        return web.json_response(response)
    resolved = 0
    if approval_session_key:
        try:
            from tools.approval import resolve_gateway_approval
            resolved = resolve_gateway_approval(
                approval_session_key, choice, resolve_all=resolve_all, request_id=request_id or None)
        except Exception as exc:
            logger.exception("[api_server] approval resolution failed for run %s", run_id)
            return _json_error(_openai_error, str(exc), status=500)
        if resolved <= 0:
            return _json_error(
                _openai_error, f"Run has no pending approval: {run_id}", code="approval_not_pending", status=409)
    request_id_field = {"request_id": request_id} if request_id else {}
    _mark_run_event(self, run_id, "approval.responded", choice=choice, **request_id_field, resolved=resolved)
    return web.json_response({
        "object": "hermes.run.approval_response", "run_id": run_id, "choice": choice, **request_id_field,
        "resolved": resolved})
