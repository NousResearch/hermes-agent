"""Plugin events for auxiliary LLM calls (#79733).

``pre_auxiliary_call`` / ``post_auxiliary_call`` fire once per physical provider attempt at the
relay boundary of ``agent.auxiliary_client`` — the funnel every auxiliary task (titling,
compression, MoA advisors/aggregator, vision, approval, ...) shares, retries and fallbacks
included — carrying the ``pre_api_request`` / ``post_api_request`` payload shape plus
``aux_task``. They are deliberately DISTINCT events: the main-loop ``*_api_request`` events stay
turn-scoped, so observability plugins keyed on turn identity never see auxiliary traffic unless
they subscribe to these. Observer-only (returns ignored) and fail-open: a raising or hung
callback is logged and the auxiliary call proceeds untouched.

``auxiliary_execution`` middleware (``hermes_cli.middleware``) is the one place a plugin can gate an
auxiliary attempt: it wraps the provider call of a non-streaming attempt and may return a
replacement response instead of calling ``next_call``; a response carrying a truthy
``auxiliary_refused`` attribute is a refusal and surfaces as ``AuxiliaryCallRefused`` (an error,
never a result). It cannot rewrite the request, and streaming attempts are not gated. Like every middleware kind it is fail-open on exceptions.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

PRE_AUXILIARY_CALL = "pre_auxiliary_call"
POST_AUXILIARY_CALL = "post_auxiliary_call"
AUXILIARY_EXECUTION = "auxiliary_execution"  # middleware kind; see hermes_cli.middleware


def _parent_turn_identity() -> dict[str, str]:
    """``session_id`` / ``task_id`` / ``turn_id`` / ``platform`` of the main turn this auxiliary
    call runs under, or empty strings for turn-less callers (cron, gateway idle work)."""
    ident = {"session_id": "", "task_id": "", "turn_id": "", "platform": ""}
    try:
        from agent.relay_runtime import current_turn

        turn = current_turn()
    except Exception:
        return ident
    if turn is None:
        return ident
    lease = getattr(turn, "lease", None)
    ident["session_id"] = str(getattr(lease, "session_id", "") or "")
    ident["platform"] = str(getattr(lease, "platform", "") or "")
    ident["task_id"] = str(getattr(turn, "task_id", "") or "")
    ident["turn_id"] = str(getattr(turn, "turn_id", "") or "")
    return ident


def _system_prompt(messages: Any, kwargs: dict[str, Any]) -> str:
    if isinstance(kwargs.get("system"), str):  # Anthropic
        return kwargs["system"]
    if isinstance(kwargs.get("instructions"), str):  # Responses
        return kwargs["instructions"]
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        first = messages[0]
        if first.get("role") == "system" and isinstance(first.get("content"), str):
            return first["content"]
    return ""


def _usage_summary(response: Any) -> Optional[dict[str, Any]]:
    raw_usage = getattr(response, "usage", None)
    if response is None or not raw_usage:
        return None
    from dataclasses import asdict

    from agent.usage_pricing import normalize_usage

    # Every auxiliary attempt returns from ``chat.completions.create``: native adapters
    # (Codex Responses, Anthropic Messages) translate their usage to prompt/completion
    # tokens first, so the native api_mode/provider shape would read zeros here.
    cu = normalize_usage(raw_usage, api_mode="chat_completions")
    summary = asdict(cu)
    summary.pop("raw_usage", None)
    summary["prompt_tokens"] = cu.prompt_tokens
    summary["total_tokens"] = cu.total_tokens
    return summary


def _first_choice_message(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return None, None
    choice = choices[0]
    if isinstance(choice, dict):
        return choice.get("message"), choice.get("finish_reason")
    return getattr(choice, "message", None), getattr(choice, "finish_reason", None)


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


class _AuxCallHooks:
    """Fires the pre/post pair for one provider attempt; the base payload is built once."""

    def __init__(
        self, *, aux_task: str, metadata: dict[str, Any], client: Any, kwargs: dict[str, Any],
        provider: str, model: str, api_mode: str, streaming: bool,
    ) -> None:
        self.streaming = streaming
        self.kwargs = kwargs
        self.started_at = time.time()
        self.base: dict[str, Any] = dict(_parent_turn_identity())
        self.base.update(
            aux_task=aux_task,
            api_request_id=str(metadata.get("api_request_id") or ""),
            retry_count=int(metadata.get("retry_count") or 0),
            api_call_count=int(metadata.get("retry_count") or 0) + 1,
            model=model,
            provider=provider,
            base_url=str(getattr(client, "base_url", "") or ""),
            api_mode=api_mode,
            streaming=streaming,
            started_at=self.started_at,
            message_count=len(kwargs.get("messages") or kwargs.get("input") or []),
        )

    def pre(self) -> None:
        if not _has_hook(PRE_AUXILIARY_CALL):
            return
        from agent.api_request_hooks import ApiRequestHooksMixin as _Sanitize

        kwargs = self.kwargs
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            messages = kwargs.get("input")  # Responses API
        if not isinstance(messages, list):
            messages = []
        body = {k: v for k, v in kwargs.items() if k not in {"timeout", "http_client"}}
        total_chars = sum(len(str(_field(m, "content") or "")) for m in messages)
        _fire(
            PRE_AUXILIARY_CALL, **self.base,
            request_messages=list(messages),
            system_prompt=_system_prompt(messages, kwargs),
            tool_count=len(kwargs.get("tools") or []),
            approx_input_tokens=total_chars // 4,
            request_char_count=total_chars,
            max_tokens=kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
            request=_Sanitize._sanitize_hook_payload({"method": "POST", "body": body}),
        )

    def post(self, response: Any = None, error: Optional[BaseException] = None) -> None:
        if not _has_hook(POST_AUXILIARY_CALL):
            return
        from agent.api_request_hooks import ApiRequestHooksMixin as _Sanitize

        ended_at = time.time()
        payload: dict[str, Any] = dict(
            self.base, ended_at=ended_at, api_duration=max(0.0, ended_at - self.started_at),
            error=None if error is None else f"{type(error).__name__}: {error}"[:2000],
            error_type=None if error is None else type(error).__name__,
        )
        # A streamed response is handed back unconsumed (the MoA facade owns reassembly), so
        # there is no usage/finish_reason to report yet; ``streaming`` tells the observer why.
        if error is not None or self.streaming:
            payload.update(finish_reason=None, response_model=None, usage=None, response=None,
                           assistant_content_chars=0, assistant_tool_call_count=0)
        else:
            message, finish_reason = _first_choice_message(response)
            content = _field(message, "content") if message is not None else None
            tool_calls = (_field(message, "tool_calls") if message is not None else None) or []
            payload.update(
                finish_reason=finish_reason,
                response_model=_field(response, "model"),
                usage=_usage_summary(response),
                response=_Sanitize._sanitize_hook_payload({
                    "model": _field(response, "model"),
                    "finish_reason": finish_reason,
                    "assistant_message": {
                        "role": (_field(message, "role") if message is not None else None) or "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "usage": payload.get("usage"),
                }),
                assistant_content_chars=len(content) if isinstance(content, str) else 0,
                assistant_tool_call_count=len(tool_calls),
            )
        _fire(POST_AUXILIARY_CALL, **payload)


def _has_hook(name: str) -> bool:
    try:
        from hermes_cli.lifecycle import has_hook

        return has_hook(name)
    except Exception:
        return False


def _fire(name: str, **payload: Any) -> None:
    """Dispatch one event; a failing subscriber is logged, never propagated (the aux task's
    result must not depend on an observer)."""
    try:
        from hermes_cli.lifecycle import invoke_hook

        invoke_hook(name, **payload)
    except Exception:
        logger.warning("%s plugin hook failed for aux_task=%s; continuing",
                       name, payload.get("aux_task"), exc_info=True)


def _hooks_or_none(**kw: Any) -> Optional[_AuxCallHooks]:
    if not (_has_hook(PRE_AUXILIARY_CALL) or _has_hook(POST_AUXILIARY_CALL)):
        return None
    try:
        hooks = _AuxCallHooks(**kw)
        hooks.pre()
    except Exception:
        logger.warning("pre_auxiliary_call payload build failed; continuing", exc_info=True)
        return None
    return hooks


def _post_safely(hooks: Optional[_AuxCallHooks], response: Any = None, error: Any = None) -> None:
    if hooks is None:
        return
    try:
        hooks.post(response, error)
    except Exception:
        logger.warning("post_auxiliary_call payload build failed; continuing", exc_info=True)


class AuxiliaryCallRefused(RuntimeError):
    """An ``auxiliary_execution`` middleware refused the attempt.

    A refusal is an error, never a result: handing refusal text back as a compression summary,
    a title or a vision answer would write it into the conversation. Raising it lets the
    caller's existing error handling (fallback chain, "no summary" paths) run instead.
    """


# A middleware refuses by returning a response object carrying this truthy attribute.
AUXILIARY_REFUSED_ATTR = "auxiliary_refused"


def _raise_if_refused(response: Any) -> None:
    if getattr(response, AUXILIARY_REFUSED_ATTR, False) is True:
        text = ""
        try:
            text = str(response.choices[0].message.content or "")
        except (AttributeError, IndexError, TypeError):
            pass
        raise AuxiliaryCallRefused(text or "auxiliary call refused by a plugin")


def _execution_context(
    aux_task: str, metadata: dict[str, Any], kwargs: dict[str, Any], provider: str, model: str, api_mode: str,
) -> dict[str, Any]:
    """Payload fields for ``auxiliary_execution`` middleware: the attempt plus its parent turn."""
    return dict(
        _parent_turn_identity(), aux_task=aux_task, provider=provider, model=model, api_mode=api_mode,
        api_request_id=str(metadata.get("api_request_id") or ""), retry_count=int(metadata.get("retry_count") or 0),
    )


def _has_execution_middleware() -> bool:
    try:
        from hermes_cli.plugins import has_middleware

        return has_middleware(AUXILIARY_EXECUTION)
    except Exception:
        logger.warning("auxiliary_execution middleware lookup failed; continuing", exc_info=True)
        return False


def run_with_aux_hooks(
    call: Callable[[], Any], *, aux_task: str, metadata: dict[str, Any], client: Any,
    kwargs: dict[str, Any], provider: str, model: str, api_mode: str, streaming: bool = False,
) -> Any:
    """Run one synchronous provider attempt between ``pre_auxiliary_call`` and
    ``post_auxiliary_call``; the exception (if any) is reported in ``post`` and re-raised.

    A non-streaming attempt also runs through ``auxiliary_execution`` middleware, which may
    replace the response instead of calling ``next_call`` (that is how a plugin refuses it).
    Streaming attempts are not gated: a refusal object would break the stream reassembly.
    """
    hooks = _hooks_or_none(aux_task=aux_task, metadata=metadata, client=client, kwargs=kwargs,
                           provider=provider, model=model, api_mode=api_mode, streaming=streaming)
    try:
        if streaming or not _has_execution_middleware():
            response = call()
        else:
            from hermes_cli.middleware import run_auxiliary_execution_middleware

            response = run_auxiliary_execution_middleware(
                kwargs, lambda _request: call(),
                **_execution_context(aux_task, metadata, kwargs, provider, model, api_mode))
            _raise_if_refused(response)
    except BaseException as exc:
        _post_safely(hooks, error=exc)
        raise
    _post_safely(hooks, response=response)
    return response


async def arun_with_aux_hooks(
    call: Callable[[], Awaitable[Any]], *, aux_task: str, metadata: dict[str, Any], client: Any,
    kwargs: dict[str, Any], provider: str, model: str, api_mode: str,
) -> Any:
    """Async twin of :func:`run_with_aux_hooks`.

    Middleware callbacks are synchronous: ``next_call`` hands back the awaitable, and a replacement
    response is returned as is, so the result is awaited only when it is awaitable.
    """
    hooks = _hooks_or_none(aux_task=aux_task, metadata=metadata, client=client, kwargs=kwargs,
                           provider=provider, model=model, api_mode=api_mode, streaming=False)
    try:
        if not _has_execution_middleware():
            response = await call()
        else:
            from hermes_cli.middleware import run_auxiliary_execution_middleware

            response = run_auxiliary_execution_middleware(
                kwargs, lambda _request: call(),
                **_execution_context(aux_task, metadata, kwargs, provider, model, api_mode))
            if inspect.isawaitable(response):
                response = await response
            _raise_if_refused(response)
    except BaseException as exc:
        _post_safely(hooks, error=exc)
        raise
    _post_safely(hooks, response=response)
    return response
