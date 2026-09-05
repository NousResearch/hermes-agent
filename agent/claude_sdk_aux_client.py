"""One-shot auxiliary client backed by the Claude Agent SDK.

LOCAL DIVERGENCE (2026-08-14).

Why this exists
---------------
``_resolve_auto_route`` fails closed when the MAIN provider is the
claude-agent-sdk (see auxiliary_client.py, "Fail-closed subscription lane",
#25267): auto-detection returns ``(None, None, "")`` so auxiliary tasks can
never be silently re-routed onto a METERED provider and break the
subscription billing contract through the side door.

That guard is correct, but it left ``auto`` meaning "no client at all" on the
SDK lane -- verified against a live runtime, ``web_extract``,
``tts_audio_tags`` and ``kanban_decomposer`` all resolved to ``None`` while
only explicitly-pinned channels worked.  The operator's escape hatch was an
explicit pin at ``auxiliary.<task>.provider``, which in practice meant
``claude-cli-live`` -- a pre-SDK shim that spawns and manages its own
persistent ``claude`` process.

This client closes that gap natively: it runs a ONE-SHOT
``claude_agent_sdk.query()`` against the SAME subscription the main lane
already uses.  Nothing metered is involved, so the billing contract the
fail-closed guard protects is preserved rather than bypassed, and ``auto``
can finally mean "the model actually in use".

Design constraints
------------------
* **Text only.**  Auxiliary tasks (compression, title generation, web
  extraction, ...) summarise text; they must never touch the filesystem or
  spawn child MCP servers.  ``tools=[]`` removes the built-in Claude Code
  tools (``allowed_tools`` is only a permission allowlist), while
  ``mcp_servers={}`` keeps MCP tools absent.  This also avoids the cost of
  booting MCP servers for a one-line summary.
* **No inherited settings.**  ``setting_sources=[]`` keeps user/project
  CLAUDE.md and settings.json out of an auxiliary prompt, so aux behaviour
  does not drift with the operator's editor config.
* **``permission_mode="dontAsk"``.**  With no tools enabled this is
  effectively moot, but it is the mode proven to work under root -- the
  ``bypassPermissions`` mode maps to ``--dangerously-skip-permissions``,
  which Claude Code refuses to run as root (repaired 2026-08-14 09:48).
* **OpenAI-shaped surface.**  Every aux caller goes through
  ``client.chat.completions.create(...)`` and reads
  ``resp.choices[0].message.content``; the return shape here mirrors
  ``claude_cli_live_client._LiveCompletions.create`` exactly.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, cast

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TIMEOUT = 600.0
# Advisory bound: asyncio's timeout cancellation must still be serviced by
# aclose(), so a cancellation-hostile SDK teardown can exceed this interval.
_QUERY_CLOSE_TIMEOUT = 5.0


class ClaudeSdkAuxError(RuntimeError):
    """Raised when a one-shot auxiliary SDK query cannot produce text."""


def _render_message_content(content: Any) -> str:
    """Render only the textual portion of an OpenAI-shaped message.

    Auxiliary SDK calls are deliberately text-only.  Image/file blocks are
    skipped rather than serialised into a misleading Python/JSON blob, while
    ordinary structured text and tool results keep their readable content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value.strip()
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts)
    return str(content).strip()


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten chat messages for the SDK's one-shot string query surface."""
    sections = [
        "You are performing a non-interactive auxiliary text task for Hermes. "
        "Return the requested answer directly; do not use tools or ask follow-up "
        "questions.",
    ]
    labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool result",
    }
    for message in messages:
        if not isinstance(message, dict):
            continue
        rendered = _render_message_content(message.get("content"))
        if not rendered:
            continue
        role = str(message.get("role") or "context").strip().lower()
        label = labels.get(role, "Context")
        sections.append(f"{label}:\n{rendered}")
    sections.append("Complete the auxiliary task described above.")
    return "\n\n".join(sections)


def _run_coro_blocking(coro, timeout: float):
    """Run ``coro`` to completion from sync code, loop-safe.

    Auxiliary clients are called from both sync paths (compression) and from
    inside a running event loop (gateway request handlers).  ``asyncio.run``
    raises if a loop is already running in this thread, so in that case the
    coroutine is handed to a dedicated worker thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            lambda: asyncio.run(asyncio.wait_for(coro, timeout=timeout))
        ).result(timeout=timeout + 30)


def _capture_aux_progress_hook() -> Callable[[], Any] | None:
    """Capture the request thread's progress hook for worker dispatch."""
    from agent import auxiliary_client

    hook = getattr(auxiliary_client._aux_progress, "hook", None)
    return hook if callable(hook) else None


async def _collect_text(
    prompt: str,
    *,
    model: str,
    cancel_check: Callable[[], Any] | None = None,
    progress_hook: Callable[[], Any] | None = None,
) -> tuple[str, Any, str]:
    """Run a one-shot SDK query and return (text, usage, stop_reason)."""
    from agent.auxiliary_client import (
        AuxiliaryExplicitCancellation,
        _aux_interrupt_cancel_requested,
        _captured_aux_cancel_requested,
        _notify_aux_progress,
        aux_progress_hook,
    )
    # Mirror the persistent session transport: make the SDK importable before
    # importing it. On a cold install the main turn may still be lazy-installing
    # claude-agent-sdk when an auxiliary task (title, compression) runs first;
    # importing without this races the install and dies with ModuleNotFoundError.
    from tools.lazy_deps import ensure as _lazy_ensure

    _lazy_ensure("provider.claude_agent_sdk", prompt=False)
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    # Import lazily to keep this lightweight facade importable without the
    # optional SDK extra.  The same override builder as the persistent lane is
    # load-bearing here: query() also spawns a CLI inheriting the parent env.
    from agent.transports.claude_agent_sdk_session import _sdk_env_overrides

    if cancel_check is None:
        cancel_requested = _aux_interrupt_cancel_requested
    else:
        # A captured generic decision may latch cancellation here. That is the
        # desired precedence, but this SDK seam performs no destructive timeout
        # cleanup and therefore never calls begin_timeout_cleanup().
        cancel_requested = lambda: _captured_aux_cancel_requested(cancel_check)

    def notify_progress() -> None:
        # Reinstall the owning request's hook in whichever worker consumes the
        # SDK stream. A None hook preserves any hook already on this thread.
        with aux_progress_hook(progress_hook):
            _notify_aux_progress()

    options = ClaudeAgentOptions(
        model=model,
        tools=[],
        allowed_tools=[],
        mcp_servers={},
        setting_sources=[],
        permission_mode="dontAsk",
        max_turns=1,
        include_partial_messages=True,
        env=_sdk_env_overrides(),
    )

    parts: list[str] = []
    usage: Any = None
    stop_reason = "stop"
    terminal_error: str | None = None
    saw_result = False

    if cancel_requested():
        raise AuxiliaryExplicitCancellation()

    stream = query(prompt=prompt, options=options)
    pending_error: BaseException | None = None
    try:
        async for message in stream:
            if cancel_requested():
                raise AuxiliaryExplicitCancellation()
            # The SDK query is internally streamed and therefore bypasses the
            # generic OpenAI chunk aggregator. Pulse Hermes's existing progress
            # hook for each consumed SDK message so long multi-call compression is
            # not mistaken for an idle/hung provider.
            notify_progress()
            if isinstance(message, AssistantMessage):
                for block in getattr(message, "content", None) or []:
                    # ThinkingBlock and friends are deliberately skipped -- aux
                    # callers want the answer text, not the reasoning trace.
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "") or ""
                        if text:
                            parts.append(text)
            elif isinstance(message, ResultMessage):
                saw_result = True
                usage = getattr(message, "usage", None)
                subtype = str(getattr(message, "subtype", None) or "")
                stop_reason = getattr(message, "stop_reason", None) or "stop"
                if getattr(message, "is_error", False) or subtype not in ("", "success"):
                    errors = getattr(message, "errors", None) or []
                    detail = (
                        "; ".join(str(error) for error in errors)
                        or str(
                            getattr(message, "result", None)
                            or subtype
                            or "unknown error"
                        )
                    )
                    terminal_error = redact_sensitive_text(detail, force=True)
    except BaseException as exc:
        pending_error = exc
        raise
    finally:
        close_stream = getattr(stream, "aclose", None)
        if callable(close_stream):
            close_timeout = asyncio.timeout(_QUERY_CLOSE_TIMEOUT)
            try:
                async with close_timeout:
                    await cast(Awaitable[Any], close_stream())
            except (
                AuxiliaryExplicitCancellation,
                KeyboardInterrupt,
                SystemExit,
            ):
                raise
            except TimeoutError as exc:
                if close_timeout.expired():
                    logger.warning(
                        "claude-agent-sdk auxiliary query close timed out "
                        "(model=%s, timeout=%gs)",
                        model,
                        _QUERY_CLOSE_TIMEOUT,
                    )
                else:
                    logger.warning(
                        "claude-agent-sdk auxiliary query close failed "
                        "(model=%s, error=%s)",
                        model,
                        type(exc).__name__,
                    )
            except asyncio.CancelledError:
                # A standalone outer/task cancellation intentionally outranks an
                # already recorded terminal SDK error; only an exception already
                # propagating from this stream body retains precedence. Production
                # callers always enter through _run_coro_blocking's wait_for wrapper;
                # suppressing its teardown cancellation preserves the earlier hard
                # cancel while the wrapper owns cancellation bookkeeping.
                if pending_error is None:
                    raise
                logger.debug(
                    "claude-agent-sdk auxiliary query close was cancelled",
                    exc_info=True,
                )
            except Exception as exc:
                logger.warning(
                    "claude-agent-sdk auxiliary query close failed "
                    "(model=%s, error=%s)",
                    model,
                    type(exc).__name__,
                )
            except BaseException as exc:
                if pending_error is None and terminal_error is None:
                    raise
                logger.warning(
                    "claude-agent-sdk auxiliary query close failed "
                    "(model=%s, error=%s)",
                    model,
                    type(exc).__name__,
                )

    if terminal_error is not None:
        # Fail closed even when partial assistant text preceded the terminal
        # error.  Returning that text as a successful compression/title can
        # silently persist a truncated result.
        raise ClaudeSdkAuxError(
            f"claude-agent-sdk auxiliary query failed: {terminal_error}"
        )
    if not saw_result:
        raise ClaudeSdkAuxError(
            "claude-agent-sdk auxiliary query ended without a terminal result"
        )

    return "".join(parts), usage, stop_reason


class _AuxCompletions:
    def __init__(self, owner: "ClaudeSdkAuxClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> SimpleNamespace:
        from agent.auxiliary_client import _capture_aux_cancel_check

        # Deliberately ungated: the predicate this replaces also reads an
        # installed source without consulting the thread-local active flag.
        return self._create(
            kwargs,
            cancel_check=_capture_aux_cancel_check(),
            progress_hook=_capture_aux_progress_hook(),
        )

    def _create(
        self,
        kwargs: dict[str, Any],
        *,
        cancel_check: Callable[[], Any] | None,
        progress_hook: Callable[[], Any] | None,
    ) -> SimpleNamespace:
        model = str(kwargs.get("model") or self._owner.default_model or DEFAULT_MODEL)
        messages = kwargs.get("messages") or []
        timeout = float(kwargs.get("timeout") or self._owner.timeout)

        if kwargs.get("stream"):
            # Auxiliary callers never need token streaming; refusing here is
            # clearer than silently returning a non-iterable.
            raise ClaudeSdkAuxError(
                "claude-agent-sdk auxiliary client does not support stream=True"
            )

        # Validate the CALLER'S messages, not the assembled prompt:
        # _messages_to_prompt always prepends the non-interactive UX guard, so
        # an assembled-prompt emptiness check can never fire and an empty
        # message list would burn a live subscription call sending boilerplate.
        if not any(
            str((m or {}).get("content") or "").strip()
            for m in messages
            if isinstance(m, dict)
        ):
            raise ClaudeSdkAuxError("refusing to send an empty auxiliary prompt")

        prompt = _messages_to_prompt(messages)

        try:
            text, usage, stop_reason = _run_coro_blocking(
                _collect_text(
                    prompt,
                    model=model,
                    cancel_check=cancel_check,
                    progress_hook=progress_hook,
                ),
                timeout,
            )
        except ClaudeSdkAuxError:
            raise
        except Exception as exc:
            safe_error = redact_sensitive_text(str(exc), force=True)
            raise ClaudeSdkAuxError(
                f"claude-agent-sdk auxiliary query failed: {safe_error}"
            ) from None

        if not text.strip():
            raise ClaudeSdkAuxError(
                f"claude-agent-sdk auxiliary query returned no text "
                f"(model={model}, stop_reason={stop_reason})"
            )

        message = SimpleNamespace(content=text, tool_calls=None, role="assistant")
        choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
        return SimpleNamespace(
            id=f"claude-agent-sdk-aux-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage,
            provider_data={"claude_agent_sdk_aux": {"stop_reason": stop_reason}},
        )


class _AuxChat:
    def __init__(self, owner: "ClaudeSdkAuxClient") -> None:
        self.completions = _AuxCompletions(owner)


class _AsyncAuxCompletions:
    """Awaitable adapter for the one-shot synchronous SDK facade."""

    def __init__(self, sync_adapter: _AuxCompletions) -> None:
        self._sync = sync_adapter

    async def create(self, **kwargs: Any) -> Any:
        import contextlib

        from agent.auxiliary_client import (
            _aux_interrupt_protected,
            _capture_aux_cancel_check,
            aux_interrupt_protection,
            aux_progress_hook,
        )

        cancel_active = _aux_interrupt_protected()
        # Canonical capture intentionally normalizes a cancel_event to its
        # bound is_set method. Every current cancellation reader needs only a
        # callable predicate; no worker mutates or waits on the Event object.
        cancel_check = _capture_aux_cancel_check()
        progress_hook = _capture_aux_progress_hook()

        # These threading-local values must be captured before to_thread().
        # This remains request-safe because production installers wrap a
        # synchronous aux call; none holds this context across an await on a
        # shared event-loop thread. A future async installer must use
        # task-local context rather than extending this adapter convention.

        def _call_sync_create() -> SimpleNamespace:
            cancel_scope = (
                aux_interrupt_protection(
                    active=cancel_active,
                    cancel_check=cancel_check,
                )
                if cancel_active or callable(cancel_check)
                else contextlib.nullcontext()
            )
            with aux_progress_hook(progress_hook), cancel_scope:
                # Restore the request context before entering the public sync
                # funnel so wrappers and future create-level logic still apply.
                return self._sync.create(**kwargs)

        return await asyncio.to_thread(_call_sync_create)


class _AsyncAuxChat:
    def __init__(self, sync_adapter: _AuxCompletions) -> None:
        self.completions = _AsyncAuxCompletions(sync_adapter)


class AsyncClaudeSdkAuxClient:
    """Async-compatible facade for subscription-safe one-shot SDK aux calls."""

    def __init__(self, sync_wrapper: "ClaudeSdkAuxClient") -> None:
        self.chat = _AsyncAuxChat(sync_wrapper.chat.completions)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        self.default_model = sync_wrapper.default_model

    async def close(self) -> None:  # pragma: no cover - no persistent client
        return None


class ClaudeSdkAuxClient:
    """OpenAI-shaped one-shot client over ``claude_agent_sdk.query()``."""

    def __init__(
        self,
        *,
        default_model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.default_model = default_model or DEFAULT_MODEL
        self.timeout = float(timeout or DEFAULT_TIMEOUT)
        # Parity with the other local facades: aux routing code reads these.
        self.base_url = ""
        self.api_key = "claude-subscription-oauth"
        self.chat = _AuxChat(self)

    def close(self) -> None:  # pragma: no cover - nothing persistent to release
        """No persistent process: each call is an independent one-shot query."""
        return None
