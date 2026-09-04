"""A2A task routing, lifecycle, and RPC protocol handlers.

Extracted from ``adapter.py`` to keep the main adapter module focused on
HTTP transport, server infrastructure, and adapter lifecycle.

Uses a mixin pattern: ``TaskRPCHandler`` is a base class whose methods
are mixed into ``A2AAdapter`` via multiple inheritance.  This preserves
the original ``self.*`` attribute access without adapter-passing overhead.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Optional

from gateway.platforms.base import MessageEvent, ProcessingOutcome
from . import protocol, security

logger = logging.getLogger(__name__)

# Keepalive and patience constants matching adapter.py
_SSE_KEEPALIVE = 15
_PATIENCE_MARGIN = 30


def _reply_timeout() -> float:
    """Seconds to wait for the agent to answer an inbound task."""
    import os
    try:
        return max(1.0, float(os.getenv("A2A_REPLY_TIMEOUT", "300")))
    except (ValueError, TypeError):
        return 300.0


class TaskRPCHandler:
    """Mixin providing A2A task lifecycle and JSON-RPC protocol handling.

    Must be mixed into a class that provides (at minimum):
        self.tasks, self._scope_for_agent(), self._agents,
        self._pending, self._pending_lock, self._pending_order,
        self._loop, self._gateway_ready, self._turns,
        self._context_peers, self._context_peers_lock,
        self._context_sessions, self._security_context,
        self._pop_pending(), self._add_pending(), self._resolve_task(),
        self._register_context_peer(), self._refine_peer_identity(),
        self._is_duplicate_inbound(), self._find_existing_nonterminal_task(),
        self._register_inline_push(), self._forward_to_profile(),
        self._wake_origin_session(), self._send_push_notification(),
        self.build_source(), self.handle_message().
    """

    # ── Inbound task handling ─────────────────────────────────────────────

    def _prepare_task_rpc(self, params: dict, peer: str, agent=None):
        """Validate, register, and dispatch an inbound message.

        Returns ``(terminal_task, None)`` when the task ends immediately,
        else ``(None, pending)``.
        """
        # Delegate to the adapter's existing _prepare_task which handles
        # session vars, context peer persistence, profile forwarding, etc.
        return self._prepare_task(params, peer, agent=agent)

    # ── JSON-RPC: message/send ───────────────────────────────────────────

    def _rpc_message_send(self, req_id, params: dict, peer: str,
                          agent=None, v1_response: bool = False,
                          client_alive=None):
        """Handle one blocking message/send JSON-RPC request."""
        terminal, pending = self._prepare_task(params, peer, agent=agent)
        if terminal is not None:
            result = protocol.send_message_response(terminal) if v1_response else terminal
            return protocol.jsonrpc_result(req_id, result)
        assert pending is not None
        pat = self._patience_for(params, pending["peer"])
        if client_alive is not None:
            def _probe():
                if not client_alive():
                    raise ConnectionResetError("A2A client disconnected while awaiting reply")
            state, reply, out_of_band_only, defer_finalization = self._await_reply(
                pending, keepalive=_probe, patience=pat)
        else:
            state, reply, out_of_band_only, defer_finalization = self._await_reply(
                pending, patience=pat)
        if defer_finalization:
            logger.info(
                "A2A: client disconnected for task %s; deferring finalization", pending["task_id"],
            )
            return protocol.jsonrpc_result(
                req_id, {"error": {"code": -32000, "message": "client disconnected, task pending"}}
            )
        try:
            state, reply = self._finalize_task(pending, state, reply)
        except protocol.DurablePublishError as dpe:
            logger.error("A2A: durable publish failed for task %s: %s", dpe.task_id, dpe)
            return protocol.durable_persistence_error(req_id, dpe.task_id, dpe.context_id, dpe.attempted_state, dpe.durable_state, dpe.dispatched)
        if out_of_band_only:
            if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED) and reply:
                _push_res = self._try_push_reply(pending, state, reply)
                # Strictly typed PushOutcome — no bool compatibility (Amendment B)
                if isinstance(_push_res, protocol.PushOutcome) and _push_res.success:
                    return None
                # Failed push: log but do not hide failure; return task response to client
                # so client does not assume out-of-band delivery succeeded.
                # No success side effect has been emitted for this failure.
        task = protocol.build_task(
            pending["task_id"], pending["context_id"], state, reply,
            created_at=pending["created_iso"],
        )
        result = protocol.send_message_response(task) if v1_response else task
        return protocol.jsonrpc_result(req_id, result)

    # ── JSON-RPC: message/stream (SSE) ──────────────────────────────────

    @staticmethod
    def _sse_headers(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.close_connection = True

    @staticmethod
    def _sse_write(handler, chunk: str):
        handler.wfile.write(chunk.encode("utf-8"))
        handler.wfile.flush()

    def _emit_terminal(self, handler, task_id: str, context_id: str,
                       state: str, reply: str, req_id=None):
        if reply and state == protocol.STATE_COMPLETED:
            self._sse_write(handler, protocol.sse_data(
                protocol.artifact_update(task_id, context_id, reply), req_id))
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, state), req_id))
        else:
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, state, reply), req_id))
        self._sse_write(handler, protocol.sse_done())

    def _rpc_message_stream(self, handler, req_id, params: dict,
                            peer: str, agent=None):
        protocol.metrics.streams_started += 1
        self._sse_headers(handler)
        try:
            terminal, pending = self._prepare_task(params, peer, agent=agent)
            if terminal is not None:
                self._emit_terminal(
                    handler, terminal["id"], terminal["contextId"],
                    terminal["status"]["state"],
                    protocol.extract_text(terminal.get("status", {}).get("message", {}) or {}),
                    req_id=req_id,
                )
                return
            assert pending is not None
            task_id, context_id = pending["task_id"], pending["context_id"]
            self._sse_write(handler, protocol.sse_data(protocol.stream_task(
                protocol.build_task(task_id, context_id, protocol.STATE_SUBMITTED, created_at=pending["created_iso"])),
                req_id))
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, protocol.STATE_WORKING), req_id))

            state, reply, _, defer_finalization = self._await_reply(
                pending, keepalive=lambda: self._sse_write(handler, ": keepalive\n\n"),
                patience=None,
            )
            if defer_finalization:
                self._emit_terminal(
                    handler, task_id, context_id,
                    protocol.STATE_WORKING, "[client disconnected, task pending]",
                    req_id=req_id,
                )
                return
            state, reply = self._finalize_task(pending, state, reply)
            self._emit_terminal(handler, task_id, context_id, state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: stream client disconnected")

    # ── JSON-RPC: tasks/subscribe ────────────────────────────────────────

    def _rpc_tasks_subscribe(self, handler, req_id, params: dict, agent=None):
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            handler._json(200, protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}"))
            return
        self._sse_headers(handler)
        try:
            fut = self.tasks.watch(task_id, *self._scope_for_agent(agent))
            if fut is None:
                self._sse_write(handler, protocol.sse_done())
                return
            deadline = time.time() + _reply_timeout()
            while True:
                try:
                    state, reply = fut.result(timeout=_SSE_KEEPALIVE)
                    break
                except FuturesTimeout:
                    if time.time() >= deadline:
                        state, reply = rec["state"], rec.get("reply", "")
                        break
                    self._sse_write(handler, ": keepalive\n\n")
            self._emit_terminal(handler, task_id, rec["context_id"], state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: subscribe client disconnected")

    # ── JSON-RPC: tasks/get ──────────────────────────────────────────────

    def _rpc_tasks_get(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        history_len = params.get("historyLength")
        try:
            history_len = int(history_len) if history_len is not None else None
        except (TypeError, ValueError):
            history_len = None
        return protocol.jsonrpc_result(req_id, protocol.TaskStore.to_task(rec, history_length=history_len))

    # ── JSON-RPC: tasks/list ─────────────────────────────────────────────

    def _rpc_tasks_list(self, req_id, params: dict, agent=None) -> dict:
        try:
            offset = int(params.get("pageToken") or 0)
        except (ValueError, TypeError):
            offset = 0
        try:
            page_size = int(params.get("pageSize") or 50)
        except (ValueError, TypeError):
            page_size = 50
        agent_slug, tenant = self._scope_for_agent(agent)
        recs, next_offset, total = self.tasks.list(
            context_id=str(params.get("contextId") or ""),
            state=str(params.get("status") or params.get("state") or ""),
            page_size=page_size,
            offset=max(0, offset),
            agent_slug=agent_slug,
            tenant=tenant,
            with_total=True,
        )
        include_artifacts = bool(params.get("includeArtifacts", False))
        history_len = params.get("historyLength")
        try:
            history_len = int(history_len) if history_len is not None else None
        except (TypeError, ValueError):
            history_len = None
        return protocol.jsonrpc_result(req_id, {
            "tasks": [protocol.TaskStore.to_task(r, history_length=history_len, include_artifacts=include_artifacts) for r in recs],
            "nextPageToken": str(next_offset) if next_offset else "",
            "pageSize": max(1, min(page_size, 100)),
            "totalSize": total,
        })

    # ── JSON-RPC: tasks/cancel ───────────────────────────────────────────

    def _rpc_tasks_cancel(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        if rec["state"] in protocol.TERMINAL_STATES:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_CANCELABLE,
                f"task {task_id} already {rec['state']}")
        _candidate = dict(rec)
        _candidate["state"] = protocol.STATE_CANCELED
        _candidate["reply"] = ""
        _candidate["completed_at"] = __import__("time").time()
        try:
            from .a2a_persistence import _task_ledger_path
            _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
        except Exception:
            logger.error("A2A: failed to persist task ledger at CANCELED for task %s", task_id, exc_info=True)
            _outcome = None
        if _outcome is None or not _outcome.published:
            logger.error("A2A: failed to persist CANCELED state for task %s — returning persistence failure", task_id)
            self._turns.reset(rec["context_id"])
            _durable = _outcome.durable_state if _outcome else rec["state"]
            return protocol.durable_persistence_error(req_id, task_id, rec["context_id"], protocol.STATE_CANCELED, _durable, True)
        self._turns.reset(rec["context_id"])
        if _outcome.newly_published:
            self._resolve_task(task_id, protocol.STATE_CANCELED, "")
            try:
                self._send_push_notification(task_id, rec["context_id"], "", protocol.STATE_CANCELED)
            except Exception:
                pass
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent)) or rec
        return protocol.jsonrpc_result(req_id, protocol.TaskStore.to_task(rec))

    # ── JSON-RPC: push notification config ───────────────────────────────

    def _register_inline_push(self, task_id: str, params: dict, agent=None):
        # Deprecated shim — inline push is now handled atomically in
        # _prepare_task via _inline_push_fields.  Kept for monkeypatch
        # compatibility; no longer called on the accepted path.
        return

    def _inline_push_fields(self, task_id: str, params: dict, *, localhost_mode: bool) -> tuple[str, str]:
        """Pure helper: validate inline push config and return (push_url, push_config_id).

        Inspects only ``params.configuration.taskPushNotificationConfig``.
        Accepts direct ``.url`` and compatibility-nested ``.pushNotificationConfig.url``.
        Requires dict containers, nonblank string URL, agreement when both present,
        and ``security.is_safe_callback_url`` acceptance for *localhost_mode*.
        On success returns (url, ``cfg-`` + 12 hex chars); otherwise ("", "").
        Never calls TaskStore or mutates state.  At most one bounded warning
        names a reason category without raw URL/configuration.
        """
        try:
            cfg_container = params.get("configuration") if isinstance(params, dict) else None
            if not isinstance(cfg_container, dict):
                return "", ""
            tpc = cfg_container.get("taskPushNotificationConfig")
            if tpc is None:
                return "", ""
            if not isinstance(tpc, dict):
                logger.warning("A2A: inline push config: task %s: non-dict taskPushNotificationConfig", task_id[:32] if isinstance(task_id, str) else "")
                return "", ""
            direct_url = tpc.get("url")
            nested_obj = tpc.get("pushNotificationConfig")
            nested_url = None
            if nested_obj is not None:
                if not isinstance(nested_obj, dict):
                    logger.warning("A2A: inline push config: task %s: non-dict pushNotificationConfig", task_id[:32] if isinstance(task_id, str) else "")
                    return "", ""
                nested_url = nested_obj.get("url")
            has_direct = direct_url is not None and direct_url != ""
            has_nested = nested_url is not None and nested_url != ""
            if not has_direct and not has_nested:
                return "", ""
            if has_direct and not isinstance(direct_url, str):
                logger.warning("A2A: inline push config: task %s: non-string url", task_id[:32] if isinstance(task_id, str) else "")
                return "", ""
            if has_nested and not isinstance(nested_url, str):
                logger.warning("A2A: inline push config: task %s: non-string nested url", task_id[:32] if isinstance(task_id, str) else "")
                return "", ""
            direct_str = str(direct_url).strip() if has_direct else ""
            nested_str = str(nested_url).strip() if has_nested else ""
            if has_direct and not direct_str:
                logger.warning("A2A: inline push config: task %s: blank url", task_id[:32] if isinstance(task_id, str) else "")
                return "", ""
            if has_nested and not nested_str:
                logger.warning("A2A: inline push config: task %s: blank nested url", task_id[:32] if isinstance(task_id, str) else "")
                return "", ""
            if has_direct and has_nested and direct_str != nested_str:
                logger.warning("A2A: inline push config: task %s: conflicting urls", task_id[:32] if isinstance(task_id, str) else "")
                return "", ""
            url = direct_str if has_direct else nested_str
            if not url:
                return "", ""
            try:
                if not security.is_safe_callback_url(url, localhost_mode=localhost_mode):
                    logger.warning("A2A: inline push config: task %s: unsafe url", task_id[:32] if isinstance(task_id, str) else "")
                    return "", ""
            except Exception:
                logger.warning("A2A: inline push config: task %s: url validation error", task_id[:32] if isinstance(task_id, str) else "")
                return "", ""
            import secrets as _secrets
            cfg_id = "cfg-" + _secrets.token_hex(6)
            return url, cfg_id
        except Exception:
            logger.warning("A2A: inline push config: task %s: parse error", task_id[:32] if isinstance(task_id, str) else "")
            return "", ""

    def _rpc_push_config_create(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or "")
        cfg = params.get("pushNotificationConfig") or params.get("config") or {}
        url = str((cfg or {}).get("url") or "")
        if not task_id or not url:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS,
                "taskId and pushNotificationConfig.url required")
        stored = self.tasks.set_push_config(task_id, url, *self._scope_for_agent(agent))
        if stored is None:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        return protocol.jsonrpc_result(req_id, stored)

    def _rpc_push_config_get(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or "")
        config_id = str(params.get("id") or params.get("configId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        cfg = self.tasks.get_push_config(task_id, config_id, *self._scope_for_agent(agent))
        if cfg is None:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND,
                f"push config not found for task: {task_id}")
        return protocol.jsonrpc_result(req_id, cfg)

    def _rpc_push_config_list(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        configs = self.tasks.list_push_configs(task_id, *self._scope_for_agent(agent))
        return protocol.jsonrpc_result(req_id, {"configs": configs, "nextPageToken": ""})

    def _rpc_push_config_delete(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or "")
        config_id = str(params.get("id") or params.get("configId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        deleted = self.tasks.delete_push_config(task_id, config_id, *self._scope_for_agent(agent))
        if not deleted:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND,
                f"push config not found for task: {task_id}")
        return protocol.jsonrpc_result(req_id, {"deleted": True})


    def _finalize_task(self, pending: dict, state: str, reply: str,
                       audit_direction: str = "outbound") -> tuple[str, str]:
        """Record the outcome of a dispatched task. Returns (state, reply) after
        redaction and input-required detection.

        ``audit_direction`` overrides the security audit direction (default
        ``"outbound"``).  Fire-and-forget loopback pushes use ``"push"``.
        """
        task_id = pending["task_id"]
        context_id = pending["context_id"]
        peer = pending["peer"]
        self._pop_pending(task_id)

        # B2: derive full safe redacted semantic reply before any surface
        try:
            _rrt = getattr(self, "_redacted_reply_text", None)
            if callable(_rrt):
                display_reply = _rrt(reply or "")
            else:
                v = reply or ""
                if not isinstance(v, str):
                    display_reply = "[redacted]"
                else:
                    try:
                        r = security.redact_outbound(v)
                        display_reply = r if isinstance(r, str) else "[redacted]"
                    except:
                        display_reply = "[redacted]"
        except:
            display_reply = "[redacted]"
        reply = display_reply

        # The agent flags clarification requests with a leading marker; map
        # them to the A2A input-required state so the peer knows to answer.
        if state == protocol.STATE_COMPLETED:
            stripped = reply.lstrip()
            if stripped.upper().startswith(protocol.INPUT_REQUIRED_MARKER):
                state = protocol.STATE_INPUT_REQUIRED
                reply = stripped[len(protocol.INPUT_REQUIRED_MARKER):].strip()
                display_reply = reply

        # Retrieve existing durable record to build candidate
        _existing = self.tasks.get(task_id)
        if _existing is None:
            bounded_tid = self._bounded_redacted_detail(task_id, 128) if hasattr(self, '_bounded_redacted_detail') else str(task_id)[:128]
            logger.error("A2A: _finalize_task called for unknown task %s", bounded_tid)
            return protocol.STATE_FAILED, "[unknown task]"
        _candidate = dict(_existing)
        _candidate["state"] = state
        _candidate["reply"] = reply
        _candidate["completed_at"] = time.time()
        from .a2a_persistence import _task_ledger_path
        try:
            _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
        except Exception as exc:
            # Convert publication exception to DurablePublishError before commit
            bounded = self._bounded_redacted_detail(exc, 300) if hasattr(self, '_bounded_redacted_detail') else str(exc)[:300]
            logger.error("A2A: publish_durable exception for task %s: %s", self._bounded_redacted_detail(task_id, 128) if hasattr(self, '_bounded_redacted_detail') else task_id, bounded)
            raise protocol.DurablePublishError(task_id, context_id, state, getattr(_existing, 'get', lambda k, d=None: None)('state') if isinstance(_existing, dict) else "WORKING", True)
        if not _outcome.published:
            bounded_err = self._bounded_redacted_detail(_outcome.error, 300) if hasattr(self, '_bounded_redacted_detail') else str(_outcome.error)[:300]
            logger.error("A2A: failed to durably publish terminal %s for task %s: %s", self._bounded_redacted_detail(state, 64) if hasattr(self, '_bounded_redacted_detail') else state, self._bounded_redacted_detail(task_id, 128) if hasattr(self, '_bounded_redacted_detail') else task_id, bounded_err)
            # Terminal-write failure leaves last durable state (normally WORKING) visible, no success side effects
            # Do not invent INDETERMINATE; keep WORKING visible
            raise protocol.DurablePublishError(task_id, context_id, state, _outcome.durable_state, True)
        # Post-commit ordering: only on newly_published do audit/metrics/push
        # These are best-effort append-only side effects; failure cannot downgrade committed success.
        # Enclose entire post-commit tail so no exception escapes a published success.
        if _outcome.newly_published:
            try:
                try:
                    protocol.persist_message(context_id, "agent", display_reply, task_id)
                except Exception as exc:
                    bounded = self._bounded_redacted_detail(exc, 300) if hasattr(self, '_bounded_redacted_detail') else str(exc)[:300]
                    logger.warning("A2A: _finalize_task persist failed for task %s (best-effort): %s", self._bounded_redacted_detail(task_id, 128) if hasattr(self, '_bounded_redacted_detail') else task_id, bounded)
                try:
                    audit_reply = self._bounded_redacted_detail(display_reply, 300) if hasattr(self, '_bounded_redacted_detail') else display_reply[:300]
                    audit_fn = getattr(self, "_audit_safe", None)
                    if callable(audit_fn):
                        audit_fn(audit_direction, peer, task_id, audit_reply, context_id=context_id)
                    else:
                        security.audit(audit_direction, peer, task_id, audit_reply, context_id=context_id)
                except Exception as exc:
                    bounded = self._bounded_redacted_detail(exc, 300) if hasattr(self, '_bounded_redacted_detail') else str(exc)[:300]
                    logger.warning("A2A: _finalize_task audit failed for task %s (best-effort): %s", self._bounded_redacted_detail(task_id, 128) if hasattr(self, '_bounded_redacted_detail') else task_id, bounded)
                # Metrics and push notification are also best-effort
                try:
                    if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED):
                        protocol.metrics.outbound_total += 1
                        protocol.metrics.tasks_completed += 1
                        protocol.metrics.record_latency(time.time() - pending["started"])
                    else:
                        protocol.metrics.tasks_failed += 1
                except Exception as exc:
                    bounded = self._bounded_redacted_detail(exc, 300) if hasattr(self, '_bounded_redacted_detail') else str(exc)[:300]
                    logger.warning("A2A: _finalize_task metrics failed for task %s (best-effort): %s", self._bounded_redacted_detail(task_id, 128) if hasattr(self, '_bounded_redacted_detail') else task_id, bounded)
                try:
                    self._send_push_notification(task_id, context_id, display_reply, state)
                except Exception as exc:
                    bounded = self._bounded_redacted_detail(exc, 300) if hasattr(self, '_bounded_redacted_detail') else str(exc)[:300]
                    logger.warning("A2A: _finalize_task push notification failed for task %s (best-effort): %s", self._bounded_redacted_detail(task_id, 128) if hasattr(self, '_bounded_redacted_detail') else task_id, bounded)
            except Exception as exc:
                # Post-commit tail must never escape; log bounded and preserve success
                try:
                    bounded = self._bounded_redacted_detail(exc, 300) if hasattr(self, '_bounded_redacted_detail') else str(exc)[:300]
                    logger.warning("A2A: _finalize_task post-commit tail failed for task %s (best-effort): %s", self._bounded_redacted_detail(task_id, 128) if hasattr(self, '_bounded_redacted_detail') else task_id, bounded)
                except Exception:
                    pass
        return state, display_reply


    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Resolve the task future when processing ends without a reply send.

        The success path resolves via send(); this hook catches failures,
        cancellations, and empty runs so the HTTP thread returns promptly
        instead of waiting out the reply timeout. For deferred disconnects
        (HTTP waiter already gone), failure/cancellation must immediately
        terminalize the TaskStore record and durably persist — otherwise
        GetTask/SubscribeToTask remain stale WORKING. Success keeps the
        deferred WORKING semantics so a late send() can still finalize the
        original task.
        """
        task_id = str(getattr(event, "message_id", "") or "")
        if not task_id:
            return
        if outcome == ProcessingOutcome.FAILURE:
            # Deferred failure: durable-publish FAILED before resolving watchers (section 5.7)
            rec = self.tasks.get(task_id)
            if rec and rec["state"] not in protocol.TERMINAL_STATES:
                _candidate = dict(rec)
                _candidate["state"] = protocol.STATE_FAILED
                _candidate["reply"] = "[agent processing failed]"
                _candidate["completed_at"] = __import__("time").time()
                from .a2a_persistence import _task_ledger_path
                _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
                if not _outcome.published:
                    logger.error("A2A: failed to durably publish FAILED for task %s: %s", task_id, _outcome.error)
                    # Keep last durable state (WORKING), no success watcher, no push
                    return
                if _outcome.newly_published:
                    self._resolve_task(task_id, protocol.STATE_FAILED, "[agent processing failed]")
                    protocol.metrics.tasks_failed += 1
                    try:
                        self._send_push_notification(task_id, rec["context_id"], "[agent processing failed]", protocol.STATE_FAILED)
                    except Exception:
                        pass
            else:
                self._resolve_task(task_id, protocol.STATE_FAILED, "[agent processing failed]")
        elif outcome == ProcessingOutcome.CANCELLED:
            rec = self.tasks.get(task_id)
            if rec and rec["state"] not in protocol.TERMINAL_STATES:
                _candidate = dict(rec)
                _candidate["state"] = protocol.STATE_CANCELED
                _candidate["reply"] = ""
                _candidate["completed_at"] = __import__("time").time()
                from .a2a_persistence import _task_ledger_path
                _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
                if not _outcome.published:
                    logger.error("A2A: failed to durably publish CANCELED for task %s: %s", task_id, _outcome.error)
                    return
                if _outcome.newly_published:
                    self._resolve_task(task_id, protocol.STATE_CANCELED, "")
                    protocol.metrics.tasks_failed += 1
                    try:
                        self._send_push_notification(task_id, rec["context_id"], "", protocol.STATE_CANCELED)
                    except Exception:
                        pass
            else:
                self._resolve_task(task_id, protocol.STATE_CANCELED, "")
        else:
            self._resolve_task(task_id, protocol.STATE_COMPLETED, "")

    # ── Push notification delivery ────────────────────────────────────────

    def _send_push_notification(self, task_id: str, context_id: str, reply: str, state: str):
        callback_url = self.tasks.pop_push_url(task_id)
        if not callback_url:
            return
        if not security.is_safe_callback_url(
            callback_url,
            localhost_mode=self._security_context.localhost_only(),
        ):
            logger.warning("A2A: push notification for task %s blocked — unsafe callback URL: %s",
                           task_id, callback_url)
            protocol.metrics.push_failed += 1
            return
        payload = protocol.status_update(task_id, context_id, state, (reply or "")[:2000])
        signature = self._security_context.sign_push_payload(payload)
        headers = {"Content-Type": "application/json"}
        if signature:
            headers["X-A2A-Signature"] = signature
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(callback_url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    protocol.metrics.push_sent += 1
                    logger.debug("A2A: push notification sent for task %s", task_id)
                else:
                    protocol.metrics.push_failed += 1
                    logger.warning("A2A: push notification for task %s got HTTP %d", task_id, resp.status)
        except Exception as e:
            protocol.metrics.push_failed += 1
            logger.warning("A2A: push notification for task %s failed: %s", task_id, e)
