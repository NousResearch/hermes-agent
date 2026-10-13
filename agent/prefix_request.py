"""One same-prefix request for context compaction.

The host keeps the last completed Chat Completions request of the session. A
compaction attempt can send that request once more with the history rows that
came after it (the reply, tool results, a new user message) and one appended
user message. The system prompt, tools, tool choice, reasoning settings, and
extra body stay the same, so a server with prefix caching can reuse the cached
prefix instead of reading the whole conversation again. This works for a manual
/compress and for automatic compaction before or inside a turn.

The request is not streamed, dispatches no tools, and saves nothing to the
session. It does not prove that the server reuses its cache; the reply usage
reports cached tokens only when the server sends them.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import re
import logging
import math
import threading
import time

from agent.message_metadata import without_persistence_fields

logger = logging.getLogger(__name__)

_WIRE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")
_UNSUPPORTED_SETTINGS = ("response_format", "grammar", "functions", "function_call", "modalities", "audio")
_DEFAULT_OUTPUT_RESERVE = 4096
# The reply limit of the handoff. The instruction asks for at most 600 words; the upper limit leaves room for a
# thinking model that counts its reasoning in the same limit.
_HANDOFF_MIN_TOKENS, _HANDOFF_MAX_TOKENS = 2048, 8192
_MAX_INSTRUCTION_BYTES = 65536
_MAX_TIMEOUT_S = 600
_CURRENT_CAPTURE = object()


class PrefixRequestError(RuntimeError):
    """The same-prefix request is not available for this attempt. The message is a fixed reason code."""


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     default=str).encode()).hexdigest()


def _source(messages):
    return [{k: v for k, v in without_persistence_fields(m).items() if not k.startswith("_")} for m in messages]


def _key(row):
    """The row fields that the provider sees, with the ``api_content`` sidecar that replaces the content."""
    return {**{k: row.get(k) for k in _WIRE_KEYS}, "api_content": row.get("api_content")}


def _sent(row):
    """The row as the main loop sends it: the ``api_content`` sidecar replaces ``content``."""
    from agent.turn_context import substitute_api_content  # Lazy: an import cycle through conversation_compression.

    row = dict(row)
    substitute_api_content(row)
    return row


def _wire(row, copy_reasoning=None):
    """The Chat Completions form of a history row that came after the captured request. ``copy_reasoning`` is
    the agent's ``_copy_reasoning_content_for_api``: reasoning_content as a main request sends it.
    ``reasoning_details`` stays here; the transport keeps it only on a route that replays it."""
    source, row = row, _sent(row)
    out = {k: row[k] for k in (*_WIRE_KEYS, "reasoning_details") if row.get(k) is not None}
    if copy_reasoning is not None:
        copy_reasoning(source, out)
    if row.get("role") == "assistant":
        out.setdefault("content", None if out.get("tool_calls") else "")
    return out


def _shape(row):
    """The structure that the wire copy of a history row keeps: role, answered call id, call ids and names.
    The content can differ: the host replays the bytes it sent, adds request-time context, and normalizes
    tool calls."""
    calls = row.get("tool_calls") or []
    names = []
    for call in calls if isinstance(calls, list) else [calls]:
        function = call.get("function") if isinstance(call, dict) else None
        names.append((call.get("id") if isinstance(call, dict) else None,
                      function.get("name") if isinstance(function, dict) else None))
    return row.get("role"), row.get("tool_call_id"), tuple(names)


def _text(content):
    """The text of a content value. White space stays: it can change code, tables, or commands."""
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _arguments(value):
    """Canonical tool-call arguments: spacing and key order are not a change, a JSON type is (``true`` and ``1``
    differ, which Python ``==`` does not see). Text that is not JSON stays as it is."""
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return "text", value
    return "json", json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _same_row(wire, row):
    """The sent row carries the stored row: the same shape, the same name (the transport drops ``name`` from
    tool rows only), the stored text (``api_content`` when the row has it) inside the sent text (the host adds
    request-time context), and the same tool-call arguments. Non-text parts are refused before this check
    (``messages_unsupported``). A row that a hook or middleware rewrote would make the handoff summarize text
    that is not in the history it replaces."""
    if _shape(wire) != _shape(row):
        return False
    if wire.get("name") != row.get("name") and not (wire.get("name") is None and row.get("role") == "tool"):
        return False
    # The host stores the text that it sends (api_content): other text is a rewrite, on its own line too.
    if _text(_sent(row).get("content")) != _text(wire.get("content")):
        return False
    calls = lambda r: [_arguments((c.get("function") or {}).get("arguments")) for c in r.get("tool_calls") or []  # noqa: E731
                       if isinstance(c, dict)]
    return calls(wire) == calls(row)


# Fields that the main loop adds to a sent row apart from the transport conversion: the prompt caching marker.
_TRANSPORT_FIELDS = frozenset({"cache_control"})


def _no_extra_fields(wire, expected):
    """The sent row and its tool calls have no field that the transport does not send for the stored row (a
    provider control that a middleware added, for example): the model reads it."""
    if set(wire) - set(expected) - _TRANSPORT_FIELDS:
        return False
    for call, want in zip(wire.get("tool_calls") or [], expected.get("tool_calls") or []):
        if isinstance(call, dict) and isinstance(want, dict) and (set(call) - set(want) or (
                isinstance(call.get("function"), dict) and set(call["function"]) - set(want.get("function") or {}))):
            return False
        # The same keys with another type value: the provider reads another kind of call.
        if isinstance(call, dict) and isinstance(want, dict) and "type" in call and call["type"] != want.get("type"):
            return False
    return True


def _same_replay_fields(wire, expected):
    """The sent row has the reasoning fields and thought signatures that the main loop replays for the stored
    row on this route. The provider reads them: a middleware that changed them made a prefix that the history
    does not have."""
    if any(wire.get(key) != expected.get(key) for key in ("reasoning_content", "reasoning_details")):
        return False
    def signatures(row):
        return [call.get("extra_content") for call in row.get("tool_calls") or [] if isinstance(call, dict)]
    return signatures(wire) == signatures(expected)


def _dump(response):
    """A comparable copy of an SDK response."""
    dump = getattr(response, "model_dump", None)
    return dump() if callable(dump) else repr(response)


def _ends_with_instruction(row, instruction):
    """The row is a user row whose last block is the host instruction (it can join the last user row)."""
    if not isinstance(row, dict) or row.get("role") != "user":
        return False
    content = row.get("content")
    if isinstance(content, str):
        return content == instruction or content.endswith("\n\n" + instruction)
    return (isinstance(content, list) and bool(content) and isinstance(content[-1], dict)
            and content[-1].get("type") == "text" and content[-1].get("text") == instruction)


def _join_user_rows(rows, instruction):
    """Join adjacent user rows of the same author as the main loop joins them (``_merge_user_content``), then
    join the host instruction to the last user row (it keeps its name: the ordinary request also ended with that
    row). The ordinary request has no adjacent user rows, and strict chat templates refuse them. The instruction
    is the last block; it says that it comes from the host."""
    from agent.agent_runtime_helpers import _UNMERGEABLE, _merge_user_content

    out = []
    for row in rows:
        last = out[-1] if out else None
        if (last is not None and last.get("role") == row.get("role") == "user" and last.get("name") == row.get("name")
                and set(last) <= {"role", "content", "name"} and set(row) <= {"role", "content", "name"}):
            joined = _merge_user_content(last.get("content"), row.get("content"))
            if joined is not _UNMERGEABLE:
                out[-1] = {**last, "content": joined}
                continue
        out.append(row)
    last = out[-1] if out else None
    if last is not None and last.get("role") == "user" and set(last) <= {"role", "content", "name"}:
        joined = _merge_user_content(last.get("content"), instruction)
        if joined is not _UNMERGEABLE:
            out[-1] = {**last, "content": joined}
            return out
    return [*out, {"role": "user", "content": instruction}]


def _same_request(base, request, count, instruction):
    """The sent request keeps what the cache and the summary depend on: every setting (model, tools, sampling),
    the first ``count`` messages (the captured request), and the host instruction as the last block of the last
    message. Only the rows after the capture can differ, for example after a redaction."""
    return (isinstance(request, dict) and isinstance(request.get("messages"), list)
            and len(request["messages"]) > count
            and {k: v for k, v in request.items() if k != "messages"}
            == {k: v for k, v in base.items() if k != "messages"}
            and request["messages"][:count] == base["messages"][:count]
            and _ends_with_instruction(request["messages"][-1], instruction))


def _route(agent):
    client_settings = getattr(agent, "_client_kwargs", {}) or {}
    client_binding = {key: str(client_settings.get(key, "")) for key in
                      ("base_url", "api_key", "organization", "project", "default_headers", "default_query")}
    return (_digest({key: getattr(agent, key, None) for key in
                     ("session_id", "model", "provider", "base_url", "api_mode",
                      "tools", "reasoning_config", "_cached_system_prompt")}),
            _digest(client_binding), id(getattr(agent, "client", None)), id(agent.context_compressor))


def enabled(agent):
    """Capture only when the context engine asks for the same-prefix request."""
    return getattr(getattr(agent, "context_compressor", None), "wants_prefix_request", False) is True


def _merged(kwargs):
    """The SDK request body with extra_body merged. Only Relay's traceparent header is supported;
    it stays outside the JSON body. Request-local queries are not supported."""
    if kwargs.get("extra_query"):
        raise PrefixRequestError("request_options_unsupported")
    _trace_headers(kwargs)
    extra = kwargs.get("extra_body") or {}
    if type(extra) is not dict:
        raise PrefixRequestError("request_options_unsupported")
    body = {k: v for k, v in kwargs.items() if k not in {"extra_body", "extra_headers", "extra_query", "timeout"}}
    body.update(extra)
    return body


def _trace_headers(kwargs):
    """Permit Relay's current traceparent header only; it is not part of the JSON prefix."""
    headers = kwargs.get("extra_headers") or {}
    if type(headers) is not dict or any(str(key).lower() != "traceparent" or type(value) is not str
                                        for key, value in headers.items()):
        raise PrefixRequestError("request_options_unsupported")
    return headers


def final_body(kwargs):
    """A JSON copy of the request body that the SDK sends."""
    return _copy(_merged(kwargs))


def _row_digests(messages):
    return [_digest(_key(row)) for row in _source(messages)]


def begin_capture(agent, kwargs):
    """Record the final body of one physical ordinary call before it is sent."""
    if not enabled(agent):
        return
    agent._prefix_capsule = None
    agent._prefix_capture = None
    source = getattr(agent, "_prefix_source_messages", None)
    if type(source) is not list:
        return
    try:
        body = final_body(kwargs)
        agent._prefix_capture = {"source": _row_digests(source), "route": _route(agent), "body": body,
                                 "body_digest": _digest(body)}
        return agent._prefix_capture
    except (PrefixRequestError, TypeError, ValueError):
        # An unsupported capture must never stop an ordinary request.
        return


def capture_response(agent, kwargs, response, *, capture=_CURRENT_CAPTURE):
    """Bind the capture to the response that this exact body produced. Streamed and plain responses."""
    if not enabled(agent):
        return response
    if capture is _CURRENT_CAPTURE:
        capture = getattr(agent, "_prefix_capture", None)
    try:
        if capture is None or _digest(_merged(kwargs)) != capture["body_digest"]:
            return response
        object.__setattr__(response, "_hermes_prefix_capture", capture)
    except (PrefixRequestError, TypeError, ValueError, AttributeError):
        pass
    return response


def _single_reply(response):
    """Return (finish_reason, message) of a one-choice response, else (None, None). The finish reason is the
    lowercase contract value (``STOP`` is ``stop``), as the transport gives it for a main request."""
    from agent.message_sanitization import normalize_finish_reason

    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        return None, None
    return normalize_finish_reason(getattr(choices[0], "finish_reason", None)), getattr(choices[0], "message", None)


def publish_response(agent, response):
    """Keep the capture for a complete reply (text or tool calls) that passed the host redirect check.
    The reply itself is not kept: the history rows after the captured request carry it."""
    if not enabled(agent):
        return
    capture = getattr(response, "_hermes_prefix_capture", None)
    if capture is None or capture is not getattr(agent, "_prefix_capture", None) or capture["route"] != _route(agent):
        return
    finish, message = _single_reply(response)
    if (finish not in ("stop", "tool_calls") or getattr(message, "role", None) != "assistant"
            or getattr(message, "function_call", None) or getattr(message, "refusal", None)):
        return
    agent._prefix_capsule = {**capture, "usage": _usage(response), "completed_at": time.monotonic()}


def _usage(response):
    usage = getattr(response, "usage", None)
    raw = usage.model_dump(exclude_none=True) if hasattr(usage, "model_dump") else usage
    raw = raw if isinstance(raw, dict) else {}
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    count = lambda value: value if type(value) is int and value >= 0 else None  # noqa: E731
    # A missing cached-token count is unknown, not zero.
    return {"prompt_tokens": count(raw.get("prompt_tokens")), "completion_tokens": count(raw.get("completion_tokens")),
            "cache_read_tokens": count(cached)}


class PrefixRequest:
    """One same-prefix request for one compaction attempt. The engine gets no client or credential."""

    def __init__(self, agent, messages, commit_fence=None):
        self._agent, self._fence = agent, commit_fence
        self._messages = messages
        self._source = _copy(_source(messages))
        self._capsule = getattr(agent, "_prefix_capsule", None)
        self._route = _route(agent)
        self._lock = threading.Lock()
        self._used = False
        self._deadline = None
        live = getattr(agent, "_session_messages", None)
        self._live = live if type(live) is list else messages
        self._live_digest = _digest(_source(self._live))

    @property
    def capture_age_s(self):
        """Seconds since the captured request completed. None without a capture."""
        completed = (self._capsule or {}).get("completed_at")
        return None if completed is None else max(0.0, time.monotonic() - completed)

    @property
    def cache_read_tokens(self):
        """Cached prompt tokens that the server reported for the captured request. None when unknown."""
        usage = (self._capsule or {}).get("usage") or {}
        return usage.get("cache_read_tokens")

    def _check(self):
        agent = self._agent
        if self._fence is not None and self._fence.is_cancelled:
            raise PrefixRequestError("cancelled")
        event = getattr(agent, "_hard_interrupt_requested", None)
        if event is not None and event.is_set():
            raise PrefixRequestError("cancelled")
        if self._route != _route(agent):
            raise PrefixRequestError("route_changed")
        if _source(self._messages) != self._source or _digest(_source(self._live)) != self._live_digest:
            raise PrefixRequestError("history_changed")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise PrefixRequestError("deadline")

    @staticmethod
    def _text_messages(messages):
        if type(messages) is not list or not messages:
            raise PrefixRequestError("messages_unsupported")
        pending = set()
        for row in messages:
            if type(row) is not dict or row.get("role") not in {"system", "user", "assistant", "tool"}:
                raise PrefixRequestError("messages_unsupported")
            if row.get("content") is not None and type(row["content"]) is not str:
                raise PrefixRequestError("messages_unsupported")
            if row["role"] == "tool":
                if row.get("tool_call_id") not in pending:
                    raise PrefixRequestError("messages_unsupported")
                pending.remove(row["tool_call_id"])
            else:
                if pending:
                    raise PrefixRequestError("messages_unsupported")
                for call in row.get("tool_calls") or []:
                    if type(call) is not dict or not call.get("id") or call["id"] in pending:
                        raise PrefixRequestError("messages_unsupported")
                    pending.add(call["id"])
        if pending:
            raise PrefixRequestError("messages_unsupported")

    def _request(self, instruction):
        agent, capsule = self._agent, self._capsule
        if agent.api_mode != "chat_completions" or agent.provider == "moa":
            raise PrefixRequestError("api_mode_unsupported")
        if capsule is None:
            raise PrefixRequestError("no_capture")
        if capsule["route"] != self._route:
            raise PrefixRequestError("route_changed")
        # The session must start with exactly the captured history (one digest for each row). The rows after it
        # (reply, tool results, a new user message) go after the captured body, as the next ordinary request
        # would send them.
        count = len(capsule["source"])
        if len(self._source) < count or _row_digests(self._source[:count]) != capsule["source"]:
            raise PrefixRequestError("history_changed")
        source = self._source[:count]
        copy_reasoning = getattr(agent, "_copy_reasoning_content_for_api", None)
        suffix = [_wire(row, copy_reasoning) for row in self._source[len(source):]]
        # The main loop's transport rule: no call_id or response_item_id, extra_content only for Gemini, no name
        # on tool rows.
        from agent.transports.chat_completions import ChatCompletionsTransport
        # The provider profile of the main request: it can replay a native reasoning carrier on any route.
        profile = None
        with contextlib.suppress(Exception):
            from providers import get_provider_profile
            profile = get_provider_profile(agent.provider)
        suffix = ChatCompletionsTransport().convert_messages(suffix, model=agent.model, base_url=agent.base_url,
                                                             provider_profile=profile)
        body = _copy(capsule["body"])
        if (body.get("tool_choice") not in (None, "auto", "none") or body.get("n", 1) != 1
                or any(key in body for key in _UNSUPPORTED_SETTINGS)):
            raise PrefixRequestError("settings_unsupported")
        self._text_messages(body.get("messages"))
        offset = len(body["messages"]) - len(source)
        if offset < 0 or any(row.get("role") != "system" for row in body["messages"][:offset]):
            raise PrefixRequestError("source_transform_unsupported")
        # The body must render exactly these rows (no selection, merge, or extra row).
        rows = body["messages"][offset:]
        if len(rows) != len(source) or not all(_same_row(wire, row) for wire, row in zip(rows, source)):
            raise PrefixRequestError("source_transform_unsupported")
        expected = ChatCompletionsTransport().convert_messages([_wire(row, copy_reasoning) for row in source],
                                                               model=agent.model, base_url=agent.base_url,
                                                               provider_profile=profile)
        if not all(_same_replay_fields(wire, want) and _no_extra_fields(wire, want) for wire, want in zip(rows, expected)):
            raise PrefixRequestError("source_transform_unsupported")
        from agent.model_metadata import estimate_request_tokens_rough
        # The server count of the captured request is exact; only the new rows are estimated.
        reported = ((capsule.get("usage") or {}).get("prompt_tokens"))
        self._prefix_tokens = reported or estimate_request_tokens_rough(body["messages"], tools=body.get("tools"))
        added = _join_user_rows(suffix, instruction)
        body["messages"] = [*body["messages"], *added]
        self._text_messages(body["messages"])
        body["stream"] = False
        body.pop("stream_options", None)
        # A stop sequence of the main request could cut the handoff after its headings.
        body.pop("stop", None)
        # A web search costs a search and can bring text that is not in the conversation into the handoff.
        body.pop("web_search_options", None)
        # The reply limit of the main request is for another task: a small one cuts the handoff, a large one
        # reserves space that the handoff does not need. Keep the field that the route uses.
        limits = ("max_tokens", "max_completion_tokens")
        for key in limits:
            value = body.get(key)
            if type(value) is int and value > 0:
                body[key] = min(max(value, _HANDOFF_MIN_TOKENS), _HANDOFF_MAX_TOKENS)
        # Without a limit the server default applies: a small one cuts the handoff, a large one can take more space
        # than the capacity check reserves. The handoff gets its own limit in the field of the route.
        if not any(type(body.get(key)) is int and body.get(key) > 0 for key in limits):
            body.update(self._agent._max_tokens_param(_HANDOFF_MAX_TOKENS))
        self._check_capacity(body, len(capsule["body"]["messages"]))
        return body

    def _check_capacity(self, body, count):
        """The measured (or estimated) captured prefix, the estimated rows after it, and the reply reserve must
        fit in the context window."""
        from agent.model_metadata import estimate_messages_tokens_rough
        limit = int(getattr(self._agent.context_compressor, "context_length", 0) or 0)
        # A server can honor either reply limit: reserve the larger one.
        limits = [body.get(key) for key in ("max_tokens", "max_completion_tokens")]
        reserve = max([value for value in limits if type(value) is int and value > 0] or [_DEFAULT_OUTPUT_RESERVE])
        if limit <= 0 or self._prefix_tokens + estimate_messages_tokens_rough(body["messages"][count:]) + reserve > limit:
            raise PrefixRequestError("capacity")

    def __call__(self, instruction, *, timeout_s=120.0):
        with self._lock:
            if self._used:
                raise PrefixRequestError("already_used")
            self._used = True
        if type(instruction) is not str or not instruction.strip() or len(instruction.encode()) > _MAX_INSTRUCTION_BYTES:
            raise PrefixRequestError("instruction_invalid")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= _MAX_TIMEOUT_S:
            raise PrefixRequestError("timeout_invalid")
        self._deadline = time.monotonic() + timeout_s
        if self._fence is not None and getattr(self._fence, "deadline_monotonic", None) is not None:
            self._deadline = min(self._deadline, self._fence.deadline_monotonic)
        self._check()
        body = self._request(instruction)
        agent = self._agent
        from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
        context = {"purpose": "context_prefix_request", "api_request_id": None, "session_id": agent.session_id or "",
                   "model": agent.model, "provider": agent.provider, "base_url": agent.base_url,
                   "api_mode": agent.api_mode}
        # Like a main request, the rows after the capture go through llm_request middleware (redaction, policy).
        try:
            changed = apply_llm_request_middleware(body, **context).payload
        except Exception as error:
            raise PrefixRequestError("middleware_refused") from error
        if not isinstance(changed, dict) or not isinstance(changed.get("messages"), list):
            raise PrefixRequestError("middleware_refused")
        # The captured request went through llm_request middleware already: a change to it (for example, an added
        # system row) would apply twice and change the cached prefix. The rows after it can change.
        count = len(self._capsule["body"]["messages"])
        if not _same_request(body, changed, count, instruction):
            raise PrefixRequestError("middleware_rewrite")
        body = changed
        # A request middleware can add text to the rows after the capture: check the size again.
        self._check_capacity(body, count)
        client = agent._create_request_openai_client(reason="context_prefix_request", api_kwargs=body)
        try:
            from openai import OpenAI
            if not isinstance(client, OpenAI) or client.max_retries != 0:
                raise PrefixRequestError("plain_sdk_required")
            self._check()
            started = time.monotonic()

            sent = []
            # A copy that no middleware can change in place: the request identity check below compares with it.
            base = copy.deepcopy(body)
            send_lock = threading.RLock()
            send_active = True
            physical_used = False

            def _send_physical(request):
                nonlocal physical_used
                # Client release must wait for a callback that has started its provider call.
                with send_lock:
                    if not send_active:
                        raise PrefixRequestError("attempt_finished")
                    if physical_used:
                        raise PrefixRequestError("already_used")
                    # The captured part, settings, and instruction must stay unchanged.
                    final = _merged(request)
                    if not _same_request(base, final, count, instruction):
                        raise PrefixRequestError("middleware_rewrite")
                    # An execution middleware can add text after the captured prefix.
                    self._check_capacity(final, count)
                    # Check cancellation, route, history, and deadline at the provider boundary.
                    self._check()
                    physical_used = True
                    # The SDK merges extra_body after its typed fields.
                    response = client.chat.completions.create(
                        model=final["model"], messages=[], extra_body=final, extra_headers=_trace_headers(request),
                        timeout=self._deadline - time.monotonic())
                    sent.append((response, _dump(response)))
                    return response

            def _send(request):
                # Relay can run the terminal callback on another thread. Do not hold the lock across Relay.
                with send_lock:
                    if not send_active:
                        raise PrefixRequestError("attempt_finished")
                from agent import relay_llm

                return relay_llm.execute(
                    request, _send_physical, session_id=str(agent.session_id or ""),
                    name=str(agent.provider or "provider"), model_name=str(agent.model or ""),
                    metadata={"api_mode": agent.api_mode, "api_request_id": None,
                              "call_role": "auxiliary:compression", "auxiliary_task": "compression", "retry_count": 0})
            try:
                # Like a main request, this request goes through llm_execution middleware (audit, policy).
                try:
                    response = run_llm_execution_middleware(
                        body, _send, original_request=copy.deepcopy(base), **context)
                finally:
                    # A stored callback must not use this client after middleware returns or raises.
                    with send_lock:
                        send_active = False
            except PrefixRequestError:
                raise
            except Exception as error:
                raise PrefixRequestError("provider_error") from error
            # Only the server's own reply: a middleware that skipped the request or changed its reply would make
            # a text that the model did not write replace the history. A block (None) is incomplete_response.
            if response is not None and (not sent or response is not sent[0][0] or _dump(response) != sent[0][1]):
                raise PrefixRequestError("middleware_changed_reply")
            elapsed = time.monotonic() - started
            self._check()
        finally:
            agent._close_request_openai_client(client, reason="context_prefix_request")
        finish, message = _single_reply(response)
        if message is None:
            raise PrefixRequestError("incomplete_response")
        content = getattr(message, "content", None)
        return {"content": content if type(content) is str else None, "finish_reason": finish,
                "tool_calls": bool(getattr(message, "tool_calls", None) or getattr(message, "function_call", None)),
                "refusal": bool(getattr(message, "refusal", None)), "usage": _usage(response),
                "elapsed_s": round(elapsed, 3)}


def build_prefix_request(agent, messages, commit_fence=None):
    """The one-shot same-prefix request for an engine that asks for it (``wants_prefix_request``), else None.
    It must never stop the compaction: a failure here only means no warm request for this attempt."""
    if not enabled(agent):
        return None
    try:
        return PrefixRequest(agent, messages, commit_fence)
    except Exception as exc:  # health: allow BLE001 -- an optional fast path must never stop a compaction
        logger.info("same-prefix request not available for this compaction (%s)", type(exc).__name__)
        return None
