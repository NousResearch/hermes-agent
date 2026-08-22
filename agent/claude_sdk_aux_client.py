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
from typing import Any

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TIMEOUT = 600.0


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


async def _collect_text(prompt: str, *, model: str) -> tuple[str, Any, str]:
    """Run a one-shot SDK query and return (text, usage, stop_reason)."""
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

    options = ClaudeAgentOptions(
        model=model,
        tools=[],
        allowed_tools=[],
        mcp_servers={},
        setting_sources=[],
        permission_mode="dontAsk",
        max_turns=1,
        env=_sdk_env_overrides(),
    )

    parts: list[str] = []
    usage: Any = None
    stop_reason = "stop"
    terminal_error: str | None = None
    saw_result = False

    async for message in query(prompt=prompt, options=options):
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
                    or str(getattr(message, "result", None) or subtype or "unknown error")
                )
                terminal_error = redact_sensitive_text(detail, force=True)

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
                _collect_text(prompt, model=model), timeout
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
        return await asyncio.to_thread(self._sync.create, **kwargs)


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
