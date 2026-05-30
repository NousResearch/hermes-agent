"""Codex API runtime — App Server and Responses-API streaming paths.

Extracted from :class:`AIAgent` to keep the agent loop file focused.
Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  AIAgent keeps thin forwarder methods for backward
compatibility.

* ``run_codex_app_server_turn`` — drives one turn through the
  ``codex_app_server`` subprocess client (used when a Codex CLI install
  is the active provider).
* ``run_codex_stream`` — streams a Codex Responses API call (the
  ``codex_responses`` api_mode).
* ``run_codex_create_stream_fallback`` — recovery path when the
  Responses ``stream=True`` initial create fails.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _is_none_iterable_typeerror(exc: BaseException) -> bool:
    err_text = str(exc)
    return "NoneType" in err_text and "not iterable" in err_text


def _create_stream_or_non_stream_fallback(
    agent,
    api_kwargs: dict,
    active_client: Any,
    *,
    reason: str,
):
    try:
        return agent._run_codex_create_stream_fallback(api_kwargs, client=active_client)
    except TypeError as exc:
        if not _is_none_iterable_typeerror(exc):
            raise
        logger.debug(
            "Responses create(stream=True) fallback raised NoneType iterable "
            "TypeError after %s; falling back to raw SSE. %s err=%s",
            reason,
            agent._client_log_context(),
            exc,
        )
        return agent._run_codex_raw_sse_fallback(api_kwargs, client=active_client)


def run_codex_app_server_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Codex app-server runtime path. Hands the entire turn to a `codex
    app-server` subprocess and projects its events back into Hermes'
    messages list so memory/skill review keep working.

    Called from run_conversation() when agent.api_mode == "codex_app_server".
    Returns the same dict shape as the chat_completions path.
    """
    from agent.transports.codex_app_server_session import CodexAppServerSession

    # Lazy session: one CodexAppServerSession per AIAgent instance.
    # Spawned on first turn, reused across turns, closed at AIAgent
    # shutdown (see _cleanup hook).
    if not hasattr(agent, "_codex_session") or agent._codex_session is None:
        cwd = getattr(agent, "session_cwd", None) or os.getcwd()
        # Approval callback: defer to Hermes' standard prompt flow if a
        # CLI thread has installed one. Gateway / cron contexts get the
        # codex-side fail-closed default.
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None
        agent._codex_session = CodexAppServerSession(
            cwd=cwd,
            approval_callback=approval_callback,
        )

    # NOTE: the user message is ALREADY appended to messages by the
    # standard run_conversation() flow (line ~11823) before the early
    # return reaches us. Do NOT append again — that would duplicate.

    try:
        turn = agent._codex_session.run_turn(user_input=user_message)
    except Exception as exc:
        logger.exception("codex app-server turn failed")
        # Crash → unconditionally drop the session so the next turn
        # respawns from scratch instead of reusing a dead client.
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None
        return {
            "final_response": (
                f"Codex app-server turn failed: {exc}. "
                f"Fall back to default runtime with `/codex-runtime auto`."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    # If the turn signalled the underlying client is wedged (deadline
    # blown, post-tool watchdog tripped, OAuth refresh died, subprocess
    # exited), retire the session so the next turn respawns codex
    # rather than riding the broken process. Mirrors openclaw beta.8's
    # "retire timed-out app-server clients" fix.
    if getattr(turn, "should_retire", False):
        logger.warning(
            "codex app-server session retired (turn error: %s)",
            turn.error,
        )
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    # Splice projected messages into the conversation. The projector emits
    # standard {role, content, tool_calls, tool_call_id} entries, which
    # is exactly what curator.py / sessions DB expect.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)

    # Counter ticks for the agent-improvement loop.
    # _turns_since_memory and _user_turn_count are ALREADY incremented
    # in the run_conversation() pre-loop block (lines ~11793-11817) so we
    # do NOT touch them here — that would double-count.
    # Only _iters_since_skill needs explicit increment, since the
    # chat_completions loop bumps it per tool iteration (line ~12110)
    # and that loop is bypassed on this path.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )

    # Now check the skill nudge AFTER iters were incremented — same
    # pattern the chat_completions path uses (line ~15432).
    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider sync (mirrors line ~15439). Skipped on
    # interrupt/error to avoid feeding partial transcripts to memory.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review fork — same cadence + signature as the default
    # path (line ~15449). Only fires when a trigger actually tripped AND
    # we have a real final response.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,  # one app-server "turn" maps to one logical API call
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        "codex_thread_id": turn.thread_id,
        "codex_turn_id": turn.turn_id,
    }




def run_codex_stream(agent, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
    """Execute one streaming Responses API request and return the final response."""
    import httpx as _httpx

    active_client = client or agent._ensure_primary_openai_client(reason="codex_stream_direct")
    max_stream_retries = 1
    has_tool_calls = False
    first_delta_fired = False
    # Accumulate streamed text so we can recover if get_final_response()
    # returns empty output (e.g. chatgpt.com backend-api sends
    # response.incomplete instead of response.completed).
    agent._codex_streamed_text_parts: list = []
    for attempt in range(max_stream_retries + 1):
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted before Codex stream retry")
        collected_output_items: list = []
        try:
            with active_client.responses.stream(**api_kwargs) as stream:
                for event in stream:
                    agent._touch_activity("receiving stream response")
                    if agent._interrupt_requested:
                        break
                    event_type = getattr(event, "type", "")
                    # Fire callbacks on text content deltas (suppress during tool calls)
                    if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                        delta_text = getattr(event, "delta", "")
                        if delta_text:
                            agent._codex_streamed_text_parts.append(delta_text)
                        if delta_text and not has_tool_calls:
                            if not first_delta_fired:
                                first_delta_fired = True
                                if on_first_delta:
                                    try:
                                        on_first_delta()
                                    except Exception:
                                        pass
                            agent._fire_stream_delta(delta_text)
                    # Track tool calls to suppress text streaming
                    elif "function_call" in event_type:
                        has_tool_calls = True
                    # Fire reasoning callbacks
                    elif "reasoning" in event_type and "delta" in event_type:
                        reasoning_text = getattr(event, "delta", "")
                        if reasoning_text:
                            agent._fire_reasoning_delta(reasoning_text)
                    # Collect completed output items — some backends
                    # (chatgpt.com/backend-api/codex) stream valid items
                    # via response.output_item.done but the SDK's
                    # get_final_response() returns an empty output list.
                    elif event_type == "response.output_item.done":
                        done_item = getattr(event, "item", None)
                        if done_item is not None:
                            collected_output_items.append(done_item)
                    # Log non-completed terminal events for diagnostics
                    elif event_type in {"response.incomplete", "response.failed"}:
                        resp_obj = getattr(event, "response", None)
                        status = getattr(resp_obj, "status", None) if resp_obj else None
                        incomplete_details = getattr(resp_obj, "incomplete_details", None) if resp_obj else None
                        logger.warning(
                            "Codex Responses stream received terminal event %s "
                            "(status=%s, incomplete_details=%s, streamed_chars=%d). %s",
                            event_type, status, incomplete_details,
                            sum(len(p) for p in agent._codex_streamed_text_parts),
                            agent._client_log_context(),
                        )
                final_response = stream.get_final_response()
                # PATCH: ChatGPT Codex backend streams valid output items
                # but get_final_response() can return output=None or an empty
                # list. Backfill from collected items or synthesize from deltas.
                _out = getattr(final_response, "output", None)
                if _out is None or (isinstance(_out, list) and not _out):
                    if collected_output_items:
                        final_response.output = list(collected_output_items)
                        logger.debug(
                            "Codex stream: backfilled %d output items from stream events",
                            len(collected_output_items),
                        )
                    elif agent._codex_streamed_text_parts and not has_tool_calls:
                        assembled = "".join(agent._codex_streamed_text_parts)
                        final_response.output = [SimpleNamespace(
                            type="message",
                            role="assistant",
                            status="completed",
                            content=[SimpleNamespace(type="output_text", text=assembled)],
                        )]
                        logger.debug(
                            "Codex stream: synthesized output from %d text deltas (%d chars)",
                            len(agent._codex_streamed_text_parts), len(assembled),
                        )
                if getattr(final_response, "output", None) is None:
                    final_response.output = []
                return final_response
        except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
            if attempt < max_stream_retries:
                logger.debug(
                    "Codex Responses stream transport failed (attempt %s/%s); retrying. %s error=%s",
                    attempt + 1,
                    max_stream_retries + 1,
                    agent._client_log_context(),
                    exc,
                )
                continue
            logger.debug(
                "Codex Responses stream transport failed; falling back to create(stream=True). %s error=%s",
                agent._client_log_context(),
                exc,
            )
            return _create_stream_or_non_stream_fallback(
                agent,
                api_kwargs,
                active_client,
                reason="transport failure",
            )
        except RuntimeError as exc:
            err_text = str(exc)
            missing_completed = "response.completed" in err_text
            # The OpenAI SDK's Responses streaming state machine raises
            # ``RuntimeError("Expected to have received `response.created`
            # before `<event-type>`")`` when the first SSE event from the
            # server is anything other than ``response.created`` — and it
            # discards the event's payload before we can read it.  Three
            # real-world backends emit a different first frame:
            #
            #   * xAI on grok-4.x OAuth — sends ``error`` (issues
            #     reported around the May 2026 SuperGrok rollout when
            #     multi-turn conversations replay encrypted reasoning
            #     content the OAuth tier rejects)
            #   * codex-lb relays — send ``codex.rate_limits`` (#14634)
            #   * custom Responses relays — send ``response.in_progress``
            #     (#8133)
            #
            # In all three cases the underlying byte stream is still
            # readable: a non-stream ``responses.create(stream=True)``
            # fallback succeeds and surfaces the real provider error as
            # a normal exception with body+status_code attached, which
            # ``_summarize_api_error`` can then translate into a useful
            # user-facing line.  Treat ``response.created`` prelude
            # errors the same way we already treat ``response.completed``
            # postlude errors.
            prelude_error = (
                "Expected to have received `response.created`" in err_text
                or "Expected to have received \"response.created\"" in err_text
            )
            if (missing_completed or prelude_error) and attempt < max_stream_retries:
                logger.debug(
                    "Responses stream %s (attempt %s/%s); retrying. %s",
                    "prelude rejected" if prelude_error else "closed before completion",
                    attempt + 1,
                    max_stream_retries + 1,
                    agent._client_log_context(),
                )
                continue
            if missing_completed or prelude_error:
                logger.debug(
                    "Responses stream %s; falling back to create(stream=True). %s err=%s",
                    "rejected before response.created" if prelude_error else "did not emit response.completed",
                    agent._client_log_context(),
                    err_text,
                )
                return _create_stream_or_non_stream_fallback(
                    agent,
                    api_kwargs,
                    active_client,
                    reason="stream prelude/postlude failure",
                )
            raise
        except TypeError as exc:
            err_text = str(exc)
            none_iterable = _is_none_iterable_typeerror(exc)
            if none_iterable and attempt < max_stream_retries:
                logger.debug(
                    "Responses stream raised NoneType iterable TypeError "
                    "(attempt %s/%s); retrying. %s",
                    attempt + 1,
                    max_stream_retries + 1,
                    agent._client_log_context(),
                )
                continue
            if none_iterable:
                logger.debug(
                    "Responses stream raised NoneType iterable TypeError; "
                    "falling back to create(stream=True). %s err=%s",
                    agent._client_log_context(),
                    err_text,
                )
                return _create_stream_or_non_stream_fallback(
                    agent,
                    api_kwargs,
                    active_client,
                    reason="stream NoneType iterable TypeError",
                )
            raise



class _RawCodexNamespace(SimpleNamespace):
    def to_dict(self) -> dict:
        return _to_plain_dict(self)


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, _RawCodexNamespace):
        return {key: _to_plain_dict(item) for key, item in vars(value).items()}
    if isinstance(value, SimpleNamespace):
        return {key: _to_plain_dict(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    return value


def _to_raw_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return _RawCodexNamespace(**{key: _to_raw_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_raw_namespace(item) for item in value]
    return value


def _synthesize_message_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text}],
    }


def _read_httpx_stream_error_body(response: Any) -> str:
    try:
        raw_body = response.read()
    except Exception:
        return str(getattr(response, "text", "") or "")
    if isinstance(raw_body, bytes):
        encoding = getattr(response, "encoding", None) or "utf-8"
        return raw_body.decode(encoding, errors="replace")
    return str(raw_body or "")


def _header_items(source: Any) -> list[tuple[Any, Any]]:
    if source is None:
        return []
    try:
        return list(dict(source).items())
    except Exception:
        items = getattr(source, "items", None)
        if callable(items):
            try:
                return list(items())
            except Exception:
                return []
    return []


def run_codex_raw_sse_fallback(agent, api_kwargs: dict, client: Any = None):
    """Last-resort raw SSE parser when openai-python Responses streaming breaks."""
    import httpx as _httpx

    active_client = client or agent._ensure_primary_openai_client(reason="codex_raw_sse_fallback")
    fallback_kwargs = dict(api_kwargs)
    fallback_kwargs["stream"] = True
    fallback_kwargs = agent._get_transport().preflight_kwargs(fallback_kwargs, allow_stream=True)

    request_body = dict(fallback_kwargs)
    extra_headers = request_body.pop("extra_headers", None)
    extra_body = request_body.pop("extra_body", None)
    if isinstance(extra_body, dict):
        request_body.update(extra_body)

    headers: dict[str, str] = {}
    for source in (
        getattr(active_client, "default_headers", None),
        getattr(active_client, "auth_headers", None),
        extra_headers,
    ):
        headers.update({str(key): str(value) for key, value in _header_items(source) if value is not None})
    if not any(key.lower() == "content-type" for key in headers):
        headers["Content-Type"] = "application/json"

    base_url = str(getattr(active_client, "base_url", "")).rstrip("/")
    if not base_url:
        raise RuntimeError("Codex raw SSE fallback missing client base_url.")
    url = f"{base_url}/responses"

    terminal_response: dict[str, Any] | None = None
    collected_output_items: list[dict[str, Any]] = []
    collected_text_deltas: list[str] = []

    with _httpx.Client(timeout=600) as http_client:
        with http_client.stream("POST", url, headers=headers, json=request_body) as response:
            if response.status_code >= 400:
                body = _read_httpx_stream_error_body(response)
                try:
                    parsed = json.loads(body)
                    body = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {response.status_code}: {body}")

            for line in response.iter_lines():
                if agent._interrupt_requested:
                    raise InterruptedError("Agent interrupted during Codex raw SSE fallback")
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    logger.debug("Codex raw SSE fallback skipped non-JSON event: %s", payload[:200])
                    continue

                event_type = str(event.get("type") or "")
                agent._touch_activity("receiving raw SSE response")

                if event_type == "error":
                    err_message = str(event.get("message") or "stream emitted error event").strip()
                    from run_agent import _StreamErrorEvent
                    raise _StreamErrorEvent(
                        err_message,
                        code=event.get("code"),
                        param=event.get("param"),
                    )
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        collected_text_deltas.append(delta)
                        agent._fire_stream_delta(delta)
                elif event_type == "response.output_text.done":
                    text = event.get("text")
                    if isinstance(text, str) and text and not collected_text_deltas:
                        collected_text_deltas.append(text)
                elif event_type == "response.output_item.done":
                    item = event.get("item")
                    if isinstance(item, dict):
                        collected_output_items.append(item)
                elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                    response_obj = event.get("response")
                    if isinstance(response_obj, dict):
                        terminal_response = response_obj
                    if event_type == "response.completed":
                        break

    if terminal_response is None:
        terminal_response = {"status": "completed"}

    output = terminal_response.get("output")
    if not isinstance(output, list) or not output:
        if collected_output_items:
            terminal_response["output"] = collected_output_items
        elif collected_text_deltas:
            terminal_response["output"] = [_synthesize_message_item("".join(collected_text_deltas))]
        elif output is None:
            terminal_response["output"] = []

    if "output_text" not in terminal_response and collected_text_deltas:
        terminal_response["output_text"] = "".join(collected_text_deltas)

    return _to_raw_namespace(terminal_response)


def run_codex_create_stream_fallback(agent, api_kwargs: dict, client: Any = None):
    """Fallback path for stream completion edge cases on Codex-style Responses backends."""
    active_client = client or agent._ensure_primary_openai_client(reason="codex_create_stream_fallback")
    fallback_kwargs = dict(api_kwargs)
    fallback_kwargs["stream"] = True
    fallback_kwargs = agent._get_transport().preflight_kwargs(fallback_kwargs, allow_stream=True)
    stream_or_response = active_client.responses.create(**fallback_kwargs)

    # Compatibility shim for mocks or providers that still return a concrete response.
    if hasattr(stream_or_response, "output"):
        return stream_or_response
    if not hasattr(stream_or_response, "__iter__"):
        return stream_or_response

    terminal_response = None
    collected_output_items: list = []
    collected_text_deltas: list = []
    try:
        for event in stream_or_response:
            agent._touch_activity("receiving stream response")
            event_type = getattr(event, "type", None)
            if not event_type and isinstance(event, dict):
                event_type = event.get("type")

            # ``error`` SSE frames carry the provider's real failure
            # reason (subscription / quota / model-not-available /
            # rejected-reasoning-replay) but never appear in the
            # ``{completed, incomplete, failed}`` terminal set, so the
            # raw loop below would silently consume them and end with
            # "did not emit a terminal response".  xAI in particular
            # emits ``type=error`` as the FIRST frame for OAuth
            # accounts whose Grok subscription is missing/exhausted —
            # the SDK's stream helper raises ``RuntimeError(Expected
            # to have received response.created before error)`` which
            # the caller catches and routes here, expecting this
            # fallback to surface the message.  Synthesize an
            # APIError-shaped exception so ``_summarize_api_error``
            # and the credential-pool entitlement detector see the
            # real text instead of a generic RuntimeError.
            if event_type == "error":
                err_message = getattr(event, "message", None)
                if not err_message and isinstance(event, dict):
                    err_message = event.get("message")
                err_code = getattr(event, "code", None)
                if not err_code and isinstance(event, dict):
                    err_code = event.get("code")
                err_param = getattr(event, "param", None)
                if not err_param and isinstance(event, dict):
                    err_param = event.get("param")
                err_message = (err_message or "stream emitted error event").strip()
                from run_agent import _StreamErrorEvent
                raise _StreamErrorEvent(err_message, code=err_code, param=err_param)

            # Collect output items and text deltas for backfill
            if event_type == "response.output_item.done":
                done_item = getattr(event, "item", None)
                if done_item is None and isinstance(event, dict):
                    done_item = event.get("item")
                if done_item is not None:
                    collected_output_items.append(done_item)
            elif event_type in {"response.output_text.delta",}:
                delta = getattr(event, "delta", "")
                if not delta and isinstance(event, dict):
                    delta = event.get("delta", "")
                if delta:
                    collected_text_deltas.append(delta)

            if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                continue

            terminal_response = getattr(event, "response", None)
            if terminal_response is None and isinstance(event, dict):
                terminal_response = event.get("response")
            if terminal_response is not None:
                # Backfill empty/None output from collected stream events
                _out = getattr(terminal_response, "output", None)
                if _out is None or (isinstance(_out, list) and not _out):
                    if collected_output_items:
                        terminal_response.output = list(collected_output_items)
                        logger.debug(
                            "Codex fallback stream: backfilled %d output items",
                            len(collected_output_items),
                        )
                    elif collected_text_deltas:
                        assembled = "".join(collected_text_deltas)
                        terminal_response.output = [SimpleNamespace(
                            type="message", role="assistant",
                            status="completed",
                            content=[SimpleNamespace(type="output_text", text=assembled)],
                        )]
                        logger.debug(
                            "Codex fallback stream: synthesized from %d deltas (%d chars)",
                            len(collected_text_deltas), len(assembled),
                        )
                    elif _out is None:
                        terminal_response.output = []
                return terminal_response
    finally:
        close_fn = getattr(stream_or_response, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    if terminal_response is not None:
        return terminal_response
    raise RuntimeError("Responses create(stream=True) fallback did not emit a terminal response.")



__all__ = [
    "run_codex_app_server_turn",
    "run_codex_stream",
    "run_codex_raw_sse_fallback",
    "run_codex_create_stream_fallback",
]
