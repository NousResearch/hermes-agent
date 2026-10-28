"""Chat Completions approval bridge for the OpenAI-compatible API server.

`/v1/chat/completions` has no approval channel: the OpenAI stream schema has no frame for a
dangerous-command decision, and strict clients (Vercel AI SDK / opencode's
``openai-compatible-chat`` decoder) ignore SSE ``event:`` names and validate every ``data:`` line
as a ``chat.completion.chunk``, so an ``approval.request`` payload whose ``choices`` is a list of
strings aborts the whole turn.

This mixin owns the alternative contract: default to rendering the approval as assistant text and
ENDING that response (a chat UI shows no send box while a response is open), keep the run parked,
and treat the client's next message as the answer — a bare approval word resolves the pending
approval through ``POST /v1/runs/{run_id}/approval`` and the resumed work streams on that reply's
own response. ``platforms.api_server.approval_events: true`` restores the structured frame.

Kept beside ``api_server_openai_routes.py`` rather than inside it: that module is at its size cap.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


class ChatApprovalMixin:
    """The Chat Completions approval bridge (mixed into ``OpenAICompatRoutesMixin``)."""

    # Bare replies that answer a pending approval in chat -> the choice posted to
    # ``POST /v1/runs/{run_id}/approval``. The whole message must be one of these: a turn that
    # merely mentions "no" is a normal turn.
    _CHAT_APPROVAL_REPLY_CHOICES = {
        "approve": "once", "approved": "once", "allow": "once", "yes": "once", "y": "once",
        "ok": "once", "run it": "once", "go ahead": "once", "do it": "once", "once": "once",
        "session": "session", "this session": "session",
        "always": "always", "always allow": "always", "allow always": "always",
        "deny": "deny", "denied": "deny", "no": "deny", "n": "deny", "nope": "deny",
        "reject": "deny", "cancel": "deny", "stop": "deny",
    }
    # How long a surfaced approval stays answerable in chat (the agent's own wait is shorter).
    _CHAT_APPROVAL_MAX_AGE = 3600.0
    # Ceiling on following a resumed run to its end on the answering turn, so one parked turn
    # cannot hold a chat response open forever.
    _CHAT_APPROVAL_CONTINUE_TIMEOUT = 900.0

    def _chat_approval_books(self) -> dict[str, dict[str, Any]]:
        """Per-completion bookkeeping for the chat-approval bridge, created on first use.

        ``pending`` maps a completion id to the approval its stream surfaced, so the client's next
        chat turn can answer it; ``parked`` maps it to ``{"stream_q", "agent_task"}``, so that turn
        can stream the continuation. This state belongs to the Chat Completions surface, and its
        adapter file is at the size cap, so it lives here rather than in the adapter's ``__init__``.
        """
        books = self.__dict__.get("_chat_approval_books_state")
        if books is None:
            books = self.__dict__["_chat_approval_books_state"] = {"pending": {}, "parked": {}}
        return books

    def _chat_approval_events_enabled(self) -> bool:
        """``platforms.api_server.approval_events``: keep the structured frame on Chat Completions.

        Default OFF: see ``_chat_approval_frame`` — strict OpenAI clients validate every ``data:``
        frame as a chat chunk and abort the turn on the ``choices`` payload of ``approval.request``.
        """
        from gateway.platforms.api_server import _coerce_request_bool

        override = self.__dict__.get("_approval_events")  # explicit override (tests/manual wiring)
        if override is not None:
            return bool(override)
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        return _coerce_request_bool(extra.get("approval_events"), default=False)

    async def _chat_turn_admission(self, request: web.Request, user_message: Any, stream: bool,
                                   body: dict[str, Any]) -> tuple:
        """Session key for this request, plus any response that ends it before a turn starts.

        Two refusals ride here: a bad ``X-Hermes-Session-Key`` header, and a bare approval word
        answering a pending approval — this surface has no approval channel, so the prompt was
        rendered as text and the client's next message is the answer.
        """
        key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key, key_err
        reply = await self._chat_approval_reply(
            request, user_message, key, stream, body.get("model", self._model_name))
        return key, reply

    async def _chat_approval_frame(self, response: web.StreamResponse, event: dict[str, Any],
                                   completion_id: str, chunk) -> bool:
        """Deliver one ``approval.request`` on a Chat Completions stream -> ``True`` if it closed.

        ``approval.request`` is not part of the OpenAI stream schema, and strict clients ignore
        ``event:`` names — they validate every ``data:`` frame as a ``chat.completion.chunk``, so the
        frame's ``choices`` list aborts the turn (opencode's ``openai-compatible-chat`` decoder raises
        ``SchemaError``). Default: render it as assistant text the client can show and reply to, then
        CLOSE this response, because a chat UI replaces its send box with a stop button while a
        response is open and a client whose only answer channel is the next message must be able to
        type. The agent stays parked (its task is not interrupted); the reply turn streams the
        continuation (``_chat_approval_reply``). The opt-in flag keeps the structured frame for
        clients that asked for it.
        """
        from gateway.platforms.api_server import _chat_usage_payload, _sse_frame

        if self._chat_approval_events_enabled():
            await response.write(_sse_frame(event, event="approval.request"))
            return False
        await response.write(_sse_frame(chunk(
            {"content": self._approval_chat_text(event, completion_id)})))
        await response.write(_sse_frame(chunk(
            {}, "stop", usage=_chat_usage_payload(
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}))))
        await response.write(b"data: [DONE]\n\n")
        logger.info("[api_server] run %s parked on an approval; closed the stream so the client can "
                    "answer with its next message", completion_id)
        return True

    def _approval_chat_text(self, event: dict[str, Any], completion_id: str) -> str:
        """Assistant-text rendering of one ``approval.request`` payload.

        The structured frame cannot ride this surface (see ``_chat_approval_frame``), so the text
        has to carry what the client needs: what is being approved, the run id the HTTP answer
        addresses, and the words a chat reply may use.
        """
        event = event or {}
        command = str(event.get("command") or "").strip()
        description = str(event.get("description") or "").strip()
        choices = [str(c) for c in (event.get("choices") or []) if c]
        parts = ["\n\nApproval needed before this can continue.\n"]
        if command:
            parts.append(f"\nCommand: {command}\n")
        if description:
            parts.append(f"\nWhy: {description}\n")
        parts.append(
            f"\nReply in this chat with {' / '.join(choices) or 'once / session / always / deny'} to "
            f'answer, or POST {{"choice": "once"}} to /v1/runs/{completion_id}/approval.\n')
        return "".join(parts)

    async def _chat_approval_reply(
        self, request: web.Request, user_message: Any, gateway_session_key: Optional[str],
        stream: bool, model_name: str,
    ) -> Optional[web.StreamResponse]:
        """Answer a pending approval from the client's next chat turn -> response, else ``None``.

        This surface has no approval channel: the request is rendered as assistant text, so the only
        thing an OpenAI-compatible client can send back is another message. A turn whose whole
        content is one approval word resolves the pending approval through the same
        ``POST /v1/runs/{run_id}/approval`` door a native client uses, instead of starting a new turn
        that leaves the parked run waiting and the answer ignored.
        """
        pending = self._chat_approval_books()["pending"]
        if not pending or not isinstance(user_message, str):
            return None
        reply = user_message.strip().lower().rstrip(".!?,").strip()
        resolve_all = reply.endswith(" all")
        if resolve_all:
            reply = reply[:-4].strip()
        choice = self._CHAT_APPROVAL_REPLY_CHOICES.get(reply)
        if not choice:
            return None
        now = time.time()
        candidates = [p for p in pending.values()
                      if now - float(p.get("created") or 0) <= self._CHAT_APPROVAL_MAX_AGE]
        if not candidates:
            return None
        # Prefer the run whose session the client named, else the newest prompt: the answer is
        # unambiguous only because it was typed after the prompt that asked for it.
        session_hint = (request.headers.get("X-Hermes-Session-Id", "") or "").strip()
        matched = [p for p in candidates if session_hint and p.get("session_id") == session_hint]
        target = max(matched or candidates, key=lambda p: float(p.get("created") or 0))
        run_id = str(target.get("run_id") or "")
        if not run_id:
            return None
        body: dict[str, Any] = {"choice": choice}
        if resolve_all:
            body["resolve_all"] = True
        if target.get("request_id"):
            body["request_id"] = target["request_id"]
        if isinstance(target.get("execution_generation"), int):
            body["execution_generation"] = target["execution_generation"]
        # Answer through the public door rather than re-implementing it: it owns ownership scope,
        # room grants and the session-authority generation fence. Same listener, same principal.
        prefix = request.path.split("/v1/", 1)[0] if "/v1/" in request.path else ""
        host = self._host if self._host not in ("", "0.0.0.0", "::") else "127.0.0.1"
        url = f"http://{host}:{self._port}{prefix}/v1/runs/{run_id}/approval"
        headers = {"Content-Type": "application/json"}
        supplied_auth = request.headers.get("Authorization")
        if supplied_auth:
            headers["Authorization"] = supplied_auth  # the caller's own principal, never this seat's
        import aiohttp

        try:
            # trust_env stays OFF: this posts to this adapter's own listener on loopback, so the
            # environment proxy settings do not apply, and aiohttp's trust_env path also loads
            # ~/.netrc for basic auth -- a credential from the operator's home must never be
            # presented here. The caller's own Authorization header is the only principal.
            async with aiohttp.ClientSession(trust_env=False) as http:
                async with http.post(url, json=body, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    status = resp.status
                    payload = await resp.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            logger.warning("[api_server] chat approval reply for run %s failed: %s", run_id, exc)
            return None
        if status == 200:
            logger.info("[api_server] chat reply %r resolved the approval on run %s", choice, run_id)
            pending.pop(run_id, None)
            # The turn that asked for it was closed at the prompt, so its continuation belongs on
            # THIS response: the client is watching this one, not the parked run's.
            return await self._chat_approval_continue(
                request, choice, run_id, stream, model_name, gateway_session_key,
                command=str(target.get("command") or ""), resolve_all=resolve_all)

        if status == 409:  # nobody is waiting any more: the run ended or another surface answered
            pending.pop(run_id, None)
            logger.info("[api_server] chat approval reply found no pending approval on run %s", run_id)
            return None
        logger.warning("[api_server] chat approval reply rejected for run %s (%s): %s",
                       run_id, status, payload)
        return None

    async def _chat_approval_continue(
        self, request: web.Request, choice: str, run_id: str, stream: bool,
        model_name: str, gateway_session_key: Optional[str], *,
        command: str = "", resolve_all: bool = False,
    ):
        """Follow the parked run's continuation on the turn that answered its approval.

        The prompt closed the original response (see ``_chat_approval_frame``), so the only response
        the client is watching is this one: the acknowledgement leads, then whatever the resumed turn
        produces follows on the same stream.
        """
        parked = self._chat_approval_books()["parked"].get(run_id) or {}
        stream_q, agent_task = parked.get("stream_q"), parked.get("agent_task")
        lead = self._chat_approval_lead(choice, run_id, command=command, resolve_all=resolve_all)
        if stream_q is None or agent_task is None:
            # Nothing to follow: the run settled (or this process never parked it), and no other
            # stream belongs to this answer.
            return await self._chat_approval_ack(
                request, lead + "That run is no longer waiting on anything here.",
                stream, model_name, gateway_session_key)
        if stream:
            return await self._chat_approval_follow_sse(
                request, stream_q, agent_task, lead, run_id, model_name, gateway_session_key)
        return await self._chat_approval_follow_json(stream_q, agent_task, lead, run_id, model_name)

    @staticmethod
    def _chat_approval_lead(choice: str, run_id: str, *, command: str = "",
                            resolve_all: bool = False) -> str:
        """The acknowledgement that opens the answering turn."""
        lead = (f"Approval recorded: {choice}{' (all pending)' if resolve_all else ''} "
                f"for run {run_id}.")
        if command:
            lead += f"\nCommand: {command}"
        return lead + "\n\n"

    @staticmethod
    def _chat_chunk(completion_id: str, created: int, model_name: str, delta: dict[str, Any],
                    finish_reason=None, **extra) -> dict[str, Any]:
        """One ``chat.completion.chunk`` envelope."""
        return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra}

    async def _chat_stream_disconnect(self, agent_ref, agent_task, completion_id: str) -> None:
        """A client dropped its stream: keep a run parked on an answerable approval, else abandon it.

        The prompt is rendered as text and the answer arrives on a *later* turn, so interrupting a
        run that is waiting on one would kill the work the approval was asked for.
        """
        from gateway.platforms.api_server import _abandon_agent_task

        if completion_id in self._chat_approval_books()["pending"]:
            logger.info("SSE client disconnected while run %s awaits approval; keeping the task",
                        completion_id)
            return
        await _abandon_agent_task(agent_ref, agent_task, "SSE client disconnected")
        logger.info("SSE client disconnected; interrupted agent task %s", completion_id)

    async def _chat_approval_final_text(self, agent_task, run_id: str) -> str:
        """The resumed turn's own final text, for a turn that streamed no content delta."""
        from gateway.platforms.api_server import _resolve_media_to_data_urls
        try:
            result = (await agent_task)[0]
        except Exception:
            # The agent task owns its own failure mode; this reply must still go out.
            logger.warning("[api_server] resumed approval turn for run %s produced no result",
                           run_id, exc_info=True)
            return ""
        if not isinstance(result, dict):
            return ""
        return _resolve_media_to_data_urls(result.get("final_response") or "")

    async def _chat_approval_follow_json(self, stream_q, agent_task, lead: str, run_id: str,
                                         model_name: str) -> web.Response:
        """Non-streaming answer: one message carrying the acknowledgement and the continuation."""
        from gateway.platforms.api_server import _chat_usage_payload

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created, parts, resumed = int(time.time()), [lead], False
        try:
            async with asyncio.timeout(self._CHAT_APPROVAL_CONTINUE_TIMEOUT):
                while True:
                    item = await stream_q.get()
                    if item is None:
                        break
                    if isinstance(item, str):
                        resumed = True
                        parts.append(item)
        except TimeoutError:
            parts.append(f"(still running after {int(self._CHAT_APPROVAL_CONTINUE_TIMEOUT)}s)")
        if not resumed:
            tail = await self._chat_approval_final_text(agent_task, run_id)
            if tail:
                parts.append(tail)
        return web.json_response({
            "id": completion_id, "object": "chat.completion", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                         "finish_reason": "stop"}],
            "usage": _chat_usage_payload(
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})})

    async def _chat_approval_follow_sse(self, request: web.Request, stream_q, agent_task,
                                        lead: str, run_id: str, model_name: str,
                                        gateway_session_key: Optional[str]):
        """Streaming answer: the acknowledgement, then the resumed turn's deltas on this response."""
        from gateway.platforms.api_server import _chat_usage_payload, _sse_frame
        from gateway.platforms.api_server_openai_routes import _iter_stream_items

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())

        def _chunk(delta: dict[str, Any], finish_reason=None, **extra) -> dict[str, Any]:
            return self._chat_chunk(completion_id, created, model_name, delta, finish_reason, **extra)

        response = await self._prepare_sse_response(request, None, gateway_session_key)
        await response.write(_sse_frame(_chunk({"role": "assistant"})))
        await response.write(_sse_frame(_chunk({"content": lead})))
        closed_on_approval = False
        continued = False
        try:
            async with asyncio.timeout(self._CHAT_APPROVAL_CONTINUE_TIMEOUT):
                async for item in _iter_stream_items(stream_q, agent_task, response):
                    if item is None:
                        break
                    if isinstance(item, tuple) and len(item) == 2:
                        kind, payload = item
                        if kind == "__reasoning__":
                            await response.write(_sse_frame(_chunk({"reasoning_content": payload})))
                        elif kind == "__approval__":
                            # A second flagged action before this turn finished: same one-way
                            # contract -- ask in text, close, let the next message answer it.
                            await response.write(_sse_frame(_chunk(
                                {"content": self._approval_chat_text(payload, run_id)})))
                            closed_on_approval = True
                            break
                        continue  # tool progress / status frames carry no chat text
                    if isinstance(item, str) and item:
                        continued = True
                        await response.write(_sse_frame(_chunk({"content": item})))
            if not continued and not closed_on_approval:
                # A turn that ends on its final text without firing a content delta would otherwise
                # leave this answer as the bare acknowledgement.
                tail = await self._chat_approval_final_text(agent_task, run_id)
                if tail:
                    await response.write(_sse_frame(_chunk({"content": tail})))
        except TimeoutError:
            await response.write(_sse_frame(_chunk(
                {"content": f"\n\n(Still working after {int(self._CHAT_APPROVAL_CONTINUE_TIMEOUT)}s — "
                            "ask me for the rest.)"})))
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            if not closed_on_approval:
                logger.info("SSE client disconnected while following resumed run %s", run_id)
        await response.write(_sse_frame(_chunk(
            {}, "stop", usage=_chat_usage_payload(
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}))))
        await response.write(b"data: [DONE]\n\n")
        return response

    async def _chat_approval_ack(self, request: web.Request, text: str, stream: bool,
                                 model_name: str, gateway_session_key: Optional[str]):
        """A normal (empty-agent) completion carrying the approval acknowledgement."""
        from gateway.platforms.api_server import _chat_usage_payload, _sse_frame

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())
        if not stream:
            return web.json_response({
                "id": completion_id, "object": "chat.completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": _chat_usage_payload(
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})})
        response = await self._prepare_sse_response(request, None, gateway_session_key)

        def _chunk(delta: dict[str, Any], finish_reason=None) -> dict[str, Any]:
            return self._chat_chunk(completion_id, created, model_name, delta, finish_reason)

        await response.write(_sse_frame(_chunk({"role": "assistant"})))
        await response.write(_sse_frame(_chunk({"content": text})))
        await response.write(_sse_frame(_chunk({}, "stop")))
        await response.write(b"data: [DONE]\n\n")
        return response
