"""Structured clarify cards and their exact run/registry bindings."""

from __future__ import annotations

import asyncio
import json
import uuid

from gateway.platforms import api_server_runs as runs
from gateway.platforms.base import SendResult


def interactive_run_toolsets(config):
    """Enable clarify for interactive defaults without overriding user restrictions."""
    from hermes_cli.tools_config import _get_platform_tools

    enabled = _get_platform_tools(config, "api_server")
    platforms = config.get("platform_toolsets") or {}
    if "api_server" not in platforms:
        config = {**config, "platform_toolsets": {
            **platforms, "api_server": sorted(enabled | {"clarify"})}}
        return _get_platform_tools(config, "api_server")
    return enabled


class APIClarifyMixin:
    async def send_clarify(self, chat_id, question, choices, clarify_id, session_key, metadata=None):
        from tools import clarify_gateway as clarify

        # Conversation ids can be shared by concurrent runs. Only the isolated
        # control-session binding can identify the run that owns this registry entry.
        candidates = [rid for rid, key in self._run_approval_sessions.items()
                      if key == session_key and rid in self._run_owners
                      and self._run_statuses.get(rid, {}).get("status") in {"running", "waiting_for_approval"}
                      and rid in self._run_streams]
        with clarify._lock:
            entry = clarify._entries.get(clarify_id)
            if (len(candidates) == 1 and entry is not None
                    and entry.session_key == session_key and not entry.event.is_set()):
                run_id = candidates[0]
                self._run_clarify_cards[clarify_id] = (run_id, entry)
                self._emit_clarify_event(run_id, "clarify.request", clarify_id=clarify_id,
                                         question=entry.question, choices=entry.choices,
                                         multi_select=entry.multi_select)
                metadata = {**(metadata or {}), "_api_clarify_run_id": run_id}
        # Always preserve the base prompt and its mark_awaiting_text side effect.
        return await super().send_clarify(
            chat_id, question, choices, clarify_id, session_key, metadata=metadata)

    def _emit_clarify_event(self, run_id, name, **fields):
        queue = self._run_streams.get(run_id)
        if queue is not None:
            queue.put_nowait(runs._run_event(run_id, name, **fields))

    def _send_clarify_text(self, content, metadata):
        run_id = (metadata or {}).get("_api_clarify_run_id")
        if run_id not in self._run_owners or run_id not in self._run_streams:
            return SendResult(success=False, error="Clarify run transport is no longer active")
        self._emit_clarify_event(run_id, "message.delta", delta=content)
        return SendResult(success=True)

    async def retire_clarify_card(self, clarify_id, notice):
        binding = self._run_clarify_cards.pop(clarify_id, None)
        if binding is None:
            return
        run_id, _ = binding
        self._emit_clarify_event(run_id, "clarify.retired", clarify_id=clarify_id, notice=notice)
        self._send_clarify_text(notice, {"_api_clarify_run_id": run_id})

    async def _handle_run_clarify(self, request):
        from aiohttp import web
        from gateway.platforms import api_server
        from tools import clarify_gateway as clarify

        run_id, _, _, _, error = runs._load_owned_run(
            self, request, _api_server=api_server, permission="approve", active_fallback=False)
        if error is not None:
            return error
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            body = None
        if (not isinstance(body, dict) or not isinstance(body.get("clarify_id"), str)
                or not body["clarify_id"].strip() or len(body["clarify_id"]) > 256
                or "response" not in body
                or (body["response"] is not None and not isinstance(body["response"], str))
                or (isinstance(body["response"], str) and len(body["response"]) > 65536)):
            return runs._json_error(
                api_server._openai_error, "Expected clarify_id and response (string or null).",
                code="invalid_clarify_response", status=400)
        clarify_id, response = body["clarify_id"], body["response"]
        # Body parsing yields: recheck liveness now. Hold the registry's RLock
        # across validation and resolution so timeout/text replies cannot race it.
        with clarify._lock:
            binding = self._run_clarify_cards.get(clarify_id)
            entry = clarify._entries.get(clarify_id)
            active = (self._run_statuses.get(run_id, {}).get("status") in {"running", "waiting_for_approval"}
                      and binding is not None and binding[0] == run_id
                      and entry is binding[1] and entry is not None
                      and entry.session_key == self._run_approval_sessions.get(run_id)
                      and not entry.event.is_set())
            if not active:
                return runs._json_error(
                    api_server._openai_error, "Run has no matching pending clarify request.",
                    code="clarify_not_pending", status=409)
            if response is None:
                clarify.mark_awaiting_text(clarify_id)
            else:
                clarify.resolve_gateway_clarify(clarify_id, response)
        if response is not None:
            await self.retire_clarify_card(clarify_id, f"✅ answered: {response}")
        return web.json_response({
            "object": "hermes.run.clarify_response", "run_id": run_id, "clarify_id": clarify_id,
            "resolved": response is not None, "awaiting_text": response is None})


def make_clarify_callback(adapter, run, loop):
    """Bridge the executor's blocking tool callback to the run's event loop."""
    from tools import clarify_gateway as clarify

    def ask(question, choices, multi_select):
        from gateway.run_turn_runner import _CLARIFY_EXPIRED_NOTICE
        from gateway.run_turn_runner_clarify_delivery import _clarify_send_then_wait

        clarify_id = uuid.uuid4().hex
        clarify.register(clarify_id, run.approval_session_key, question, choices, multi_select)
        future = asyncio.run_coroutine_threadsafe(adapter.send_clarify(
            run.session_id, question, choices, clarify_id, run.approval_session_key), loop)
        response, answered = _clarify_send_then_wait(
            future, clarify_id=clarify_id, session_key=run.approval_session_key, clarify_mod=clarify)
        notice = f"✅ answered: {response}" if answered else _CLARIFY_EXPIRED_NOTICE
        # Drain retirement before the executor returns and the run closes its stream.
        asyncio.run_coroutine_threadsafe(adapter.retire_clarify_card(clarify_id, notice), loop).result()
        return response, answered

    def callback(question, choices, multi_select=False, questions=None):
        if not questions:
            return ask(question, choices, multi_select)[0]
        answers = {}
        payload = {"answers": answers, "timed_out": False}
        for index, item in enumerate(questions):
            response, answered = ask(item.get("question", ""), item.get("choices"),
                                     bool(item.get("multi_select")))
            if not answered:
                payload.update(timed_out=True, notice=response)
                break
            answers[item.get("qid") or f"q{index}"] = response
        return json.dumps(payload, ensure_ascii=False)

    return callback


def clear_run_clarify(adapter, run_id):
    """Wake a blocked tool on stop, shutdown or task cancellation."""
    from tools.clarify_gateway import clear_session

    session_key = adapter._run_approval_sessions.get(run_id)
    if session_key:
        clear_session(session_key)


async def retire_run_clarify(adapter, run_id):
    clear_run_clarify(adapter, run_id)
    for clarify_id, (owner, _) in list(adapter._run_clarify_cards.items()):
        if owner == run_id:
            from gateway.run_turn_runner import _CLARIFY_EXPIRED_NOTICE
            await adapter.retire_clarify_card(clarify_id, _CLARIFY_EXPIRED_NOTICE)
