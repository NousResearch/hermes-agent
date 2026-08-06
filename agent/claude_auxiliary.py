"""One-shot Claude Agent SDK adapter for Hermes' auxiliary (side-LLM) work.

``agent/auxiliary_client.py`` runs everything that is not the conversation:
title generation, classification, extraction, vision, the compression fallback,
session search, and plugin LLM calls.  When the user is on the Claude
subscription runtime those calls must go through the official SDK like every
other Claude request — otherwise a side channel keeps hitting the pre-SDK
direct-OAuth HTTP path, which is exactly the billing surprise this work exists
to remove (``docs/design/claude-subscription-via-agent-sdk.md`` § 3).

Four properties define this adapter, and each of them is a decision, not a
detail:

* **One-shot ``query()``, never ``ClaudeSDKClient``.**  An auxiliary call is
  stateless.  ``query()`` spawns, answers, and exits; a client would mean a
  second long-lived CLI subprocess and a second event-loop thread per agent.
* **No tools, ever.**  ``tools=[]`` + ``mcp_servers={}`` +
  ``strict_mcp_config=True`` + ``setting_sources=[]``.  A title generator that
  can run Bash is a security bug, and a user/project ``.mcp.json`` must not be
  able to hand it one.
* **Never touches the conversation.**  No ``resume``, no ``continue_conversation``,
  no ``session_id``, no ``fork_session``, and no ``session_store``.  The SDK
  mints a throwaway session per call; the main session id is untouched.
* **Bounded.**  ``max_turns=1`` and a wall-clock timeout.  Auxiliary work that
  runs away is worse than auxiliary work that fails.

The returned object is the same OpenAI-shaped facade every other auxiliary
transport exposes (``client.chat.completions.create(...)`` →
``.choices[0].message.content``), so no call site in ``auxiliary_client.py``
needs to know which transport answered.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from agent.claude_sdk_input import (
    ClaudeImageInputUnsupported,
    VISION_PROVIDER_REQUIRED_MESSAGE,
    blocks_to_text,
    content_has_images,
    content_to_sdk_blocks,
    make_sdk_stream_prompt,
    sdk_supports_streaming_input,
)

logger = logging.getLogger(__name__)

# The provider id whose auxiliary work belongs here.  Kept as a module constant
# so ``auxiliary_client`` can route without importing the CLI provider module
# (which PR7 owns) on a hot path.
CLAUDE_CODE_PROVIDER_ID = "claude-code"

# An auxiliary prompt is a single question. Anything that wants a second turn
# is agentic work and does not belong on this path.
CLAUDE_AUX_MAX_TURNS = 1

# Wall-clock bound. Generous relative to an HTTP aux call because the SDK pays
# a CLI cold start, but far below the conversation runtime's 1800s.
DEFAULT_CLAUDE_AUX_TIMEOUT_SECONDS = 180.0

# How long we wait for the worker thread to unwind after its coroutine was
# cancelled, before giving up on a clean join.
_THREAD_JOIN_GRACE_SECONDS = 10.0

INSTALL_HINT = (
    "The Claude subscription runtime needs the optional `claude-code` extra. "
    "Install it with: pip install 'hermes-agent[claude-code]'"
)


class ClaudeAuxiliaryError(RuntimeError):
    """Raised when a one-shot Claude auxiliary call cannot be made or completed."""


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def _import_sdk() -> Any:
    """Import ``claude_agent_sdk`` lazily, with an actionable ImportError."""
    try:
        import claude_agent_sdk  # noqa: PLC0415 - optional extra, imported on use
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return claude_agent_sdk


def is_claude_subscription_provider(provider: Optional[str]) -> bool:
    """True when *provider* is the Claude subscription provider.

    Note what this does NOT do: it never answers True for ``anthropic``.  The
    two are separate providers on purpose — ``anthropic`` means "an API key you
    pasted, billed to your Console org".  While the subscription gate is closed
    ``hermes_cli.claude_code.legacy_alias_target`` rewrites ``claude-code`` to
    ``anthropic`` before resolution ever reaches here, and when it is open it
    stops rewriting, so the slug is unambiguous at this layer either way.
    """
    return (provider or "").strip().lower() == CLAUDE_CODE_PROVIDER_ID


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def build_claude_auxiliary_options(
    *,
    system_prompt: Optional[str],
    model: Optional[str],
    max_turns: int = CLAUDE_AUX_MAX_TURNS,
) -> Any:
    """Build the locked-down ``ClaudeAgentOptions`` for a one-shot aux call."""
    sdk = _import_sdk()
    return sdk.ClaudeAgentOptions(
        system_prompt=system_prompt or None,
        # No tools of any kind: not the SDK built-ins, not Hermes' MCP bridge,
        # and not whatever a project .mcp.json would like to add.
        tools=[],
        allowed_tools=[],
        mcp_servers={},
        strict_mcp_config=True,
        setting_sources=[],
        max_turns=max_turns,
        model=model or None,
        # Belt and braces: with no tools registered there is nothing to permit,
        # but "deny anything not pre-approved" is the right posture for a call
        # that must never execute anything.
        permission_mode="dontAsk",
        include_partial_messages=False,
        # Explicit rather than defaulted, because these four are precisely how
        # an auxiliary call would leak into the conversation's session.
        continue_conversation=False,
        resume=None,
        fork_session=False,
        session_store=None,
    )


# ---------------------------------------------------------------------------
# Message flattening
# ---------------------------------------------------------------------------


def split_messages(messages: Any) -> Tuple[Optional[str], Any]:
    """Split OpenAI-shaped *messages* into (system_prompt, prompt_content).

    The SDK carries the system prompt as an option, not as a message, and the
    one-shot path takes a single user prompt.  A multi-message auxiliary
    conversation (rare — a few tasks send a worked example) is flattened into
    one labelled prompt rather than dropped, so the example still reaches the
    model.
    """
    if not isinstance(messages, list):
        return None, str(messages or "")

    system_parts: List[str] = []
    turns: List[Tuple[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            turns.append(("user", str(message)))
            continue
        role = str(message.get("role") or "user").strip().lower()
        content = message.get("content")
        if role == "system":
            system_parts.append(blocks_to_text(content_to_sdk_blocks(content)))
        else:
            turns.append((role, content))

    system_prompt = "\n\n".join(p for p in system_parts if p) or None

    if len(turns) == 1:
        return system_prompt, turns[0][1]

    # More than one turn: keep any image parts, label the rest.
    flattened: List[Any] = []
    for role, content in turns:
        label = "User" if role == "user" else "Assistant"
        blocks = content_to_sdk_blocks(content)
        if any(b.get("type") == "image" for b in blocks):
            flattened.append({"type": "text", "text": f"{label}:"})
            flattened.extend(blocks)
        else:
            text = blocks_to_text(blocks)
            if text:
                flattened.append({"type": "text", "text": f"{label}: {text}"})
    if not flattened:
        return system_prompt, ""
    if not any(block.get("type") == "image" for block in flattened):
        # No image survived: send plain text, which is both the cheapest prompt
        # and the only shape that avoids the streaming-input path entirely.
        return system_prompt, "\n\n".join(
            block["text"] for block in flattened if block.get("type") == "text"
        )
    return system_prompt, flattened


def build_prompt(content: Any) -> Any:
    """Return the SDK prompt for auxiliary *content* (string or parts).

    Raises :class:`ClaudeAuxiliaryError` when the content carries an image the
    installed SDK cannot deliver, naming the one supported alternative.
    """
    if isinstance(content, str):
        return content
    blocks = content_to_sdk_blocks(content)
    if not content_has_images(content):
        return blocks_to_text(blocks)
    if not sdk_supports_streaming_input():
        raise ClaudeAuxiliaryError(VISION_PROVIDER_REQUIRED_MESSAGE)
    return make_sdk_stream_prompt(blocks)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


async def _collect(prompt: Any, options: Any) -> Dict[str, Any]:
    """Drive one ``query()`` and reduce the stream to a plain result dict."""
    sdk = _import_sdk()
    # Same sanitized spawn as the conversation runtime: options.env can
    # override but never delete an inherited credential, so removal has to
    # happen at the transport. query() accepts a pre-built transport. Only
    # built when the options came from the real package — a test stand-in
    # spawns nothing, and the real transport reads real-SDK option fields.
    transport = None
    if type(options).__module__.startswith("claude_agent_sdk"):
        from agent.transports.claude_sanitized_transport import (
            build_sanitized_transport,
        )

        transport = build_sanitized_transport(options, prompt=prompt)
    text_parts: List[str] = []
    result: Dict[str, Any] = {
        "text": "",
        "usage": None,
        "model": getattr(options, "model", None),
        "is_error": False,
        "error": None,
        "session_id": None,
    }
    async for message in sdk.query(prompt=prompt, options=options, transport=transport):
        # Dispatch on class name so this behaves identically against the real
        # extra and against a stand-in, matching agent/claude_runtime.py.
        kind = type(message).__name__
        if kind == "AssistantMessage":
            for block in getattr(message, "content", None) or []:
                if type(block).__name__ == "TextBlock":
                    text_parts.append(str(getattr(block, "text", "") or ""))
            model = getattr(message, "model", None)
            if model:
                result["model"] = str(model)
        elif kind == "ResultMessage":
            result["session_id"] = getattr(message, "session_id", None)
            usage = getattr(message, "usage", None)
            if isinstance(usage, dict):
                result["usage"] = usage
            if getattr(message, "is_error", False):
                result["is_error"] = True
                text = getattr(message, "result", None)
                result["error"] = str(text or "Claude Agent SDK reported an error")
            else:
                text = getattr(message, "result", None)
                if isinstance(text, str) and text.strip() and not text_parts:
                    text_parts.append(text)
    result["text"] = "".join(text_parts)
    return result


def run_claude_auxiliary_query(
    prompt: Any,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_turns: int = CLAUDE_AUX_MAX_TURNS,
) -> Dict[str, Any]:
    """Run one bounded, tool-less SDK query and return a plain result dict.

    Executes on a short-lived worker thread with its own event loop.  The
    conversation's :class:`~agent.transports.claude_agent_session.ClaudeAgentSession`
    owns a loop thread of its own and its anyio streams are bound to it; an
    auxiliary call must not borrow that loop, and it must not assume the caller
    has one.  A fresh loop per call is the only arrangement that is correct from
    every thread Hermes calls auxiliary work from.
    """
    # The same two guarantees the conversation runtime gives, because an
    # auxiliary call bills the same account: refuse when a higher-precedence
    # credential would win, and spawn the CLI from a sanitized environment.
    # Without these, a title-generation call with ANTHROPIC_API_KEY exported
    # silently bills metered API usage while the conversation is correctly
    # refused — the exact split-billing failure this provider exists to end.
    from agent.claude_billing import static_billing_refusal

    refusal = static_billing_refusal()
    if refusal is not None:
        raise ClaudeAuxiliaryError(refusal)

    options = build_claude_auxiliary_options(
        system_prompt=system_prompt, model=model, max_turns=max_turns
    )
    deadline = float(timeout or DEFAULT_CLAUDE_AUX_TIMEOUT_SECONDS)
    box: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(
                asyncio.wait_for(_collect(prompt, options), deadline)
            )
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller
            box["error"] = exc

    thread = threading.Thread(
        target=_runner, name="hermes-claude-aux", daemon=True
    )
    thread.start()
    # ``wait_for`` already bounds the coroutine; this join only covers the
    # unwind, so a wedged SDK cannot pin the calling thread for the full
    # deadline a second time.
    thread.join(deadline + _THREAD_JOIN_GRACE_SECONDS)
    if thread.is_alive():
        raise ClaudeAuxiliaryError(
            f"Claude auxiliary call did not finish within {deadline:.0f}s."
        )

    error = box.get("error")
    if isinstance(error, asyncio.TimeoutError):
        raise ClaudeAuxiliaryError(
            f"Claude auxiliary call timed out after {deadline:.0f}s."
        )
    if isinstance(error, BaseException):
        raise error

    result = box.get("value") or {}
    if result.get("is_error"):
        raise ClaudeAuxiliaryError(
            str(result.get("error") or "Claude Agent SDK reported an error")
        )
    return result


# ---------------------------------------------------------------------------
# OpenAI-shaped facade
# ---------------------------------------------------------------------------


def _usage_namespace(usage: Any) -> Optional[SimpleNamespace]:
    if not isinstance(usage, dict) or not usage:
        return None

    def _num(key: str) -> int:
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) and value > 0 else 0

    prompt_tokens = (
        _num("input_tokens")
        + _num("cache_read_input_tokens")
        + _num("cache_creation_input_tokens")
    )
    completion_tokens = _num("output_tokens")
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


class _ClaudeAuxiliaryCompletionsAdapter:
    """Translates ``chat.completions.create(**kwargs)`` into one SDK query."""

    def __init__(self, model: str, timeout: Optional[float] = None) -> None:
        self._model = model
        self._timeout = timeout

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("tools"):
            # Loud rather than silent: a caller asking for function calling on
            # this path would otherwise get a plausible-looking text answer with
            # the tool contract quietly dropped.
            raise ClaudeAuxiliaryError(
                "The Claude subscription auxiliary adapter runs with no tools; "
                "configure an explicit `auxiliary.<task>.provider` for "
                "tool-calling side tasks."
            )
        if kwargs.get("stream"):
            # Same contract as the Bedrock Converse shim: return a complete
            # response and let call_llm's consumer downgrade to non-live output.
            logger.debug(
                "ClaudeAuxiliaryClient: stream=True requested — returning a "
                "complete response (the one-shot SDK path does not stream)."
            )

        model = str(kwargs.get("model") or self._model or "") or None
        system_prompt, content = split_messages(kwargs.get("messages", []))
        prompt = build_prompt(content)
        timeout = kwargs.get("timeout") or self._timeout

        result = run_claude_auxiliary_query(
            prompt,
            system_prompt=system_prompt,
            model=model,
            timeout=float(timeout) if timeout else None,
        )

        message = SimpleNamespace(
            content=result.get("text") or "",
            tool_calls=None,
            reasoning=None,
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            model=result.get("model") or model,
            usage=_usage_namespace(result.get("usage")),
        )


class _ClaudeAuxiliaryChatShim:
    def __init__(self, adapter: _ClaudeAuxiliaryCompletionsAdapter) -> None:
        self.completions = adapter


class ClaudeAuxiliaryClient:
    """OpenAI-client-compatible facade over one-shot Claude Agent SDK calls.

    ``api_key`` is the empty string on purpose and is never read: Hermes holds
    no Claude credential.  ``base_url`` is the internal
    ``claude-sdk://subscription`` scheme — not a reachable endpoint — so nothing
    downstream can mistake this client for an HTTP transport.
    """

    def __init__(self, model: str, *, timeout: Optional[float] = None) -> None:
        self._model = model
        self.chat = _ClaudeAuxiliaryChatShim(
            _ClaudeAuxiliaryCompletionsAdapter(model, timeout=timeout)
        )
        self.api_key = ""
        self.base_url = "claude-sdk://subscription"

    def close(self) -> None:
        """No-op: every call owns and retires its own subprocess."""


class _AsyncClaudeAuxiliaryCompletionsAdapter:
    def __init__(self, sync_adapter: _ClaudeAuxiliaryCompletionsAdapter) -> None:
        self._sync = sync_adapter

    async def create(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncClaudeAuxiliaryChatShim:
    def __init__(self, adapter: _AsyncClaudeAuxiliaryCompletionsAdapter) -> None:
        self.completions = adapter


class AsyncClaudeAuxiliaryClient:
    """Async counterpart — the sync call runs on a worker thread."""

    def __init__(self, sync_wrapper: ClaudeAuxiliaryClient) -> None:
        self.chat = _AsyncClaudeAuxiliaryChatShim(
            _AsyncClaudeAuxiliaryCompletionsAdapter(sync_wrapper.chat.completions)
        )
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        self._real_client = sync_wrapper

    def close(self) -> None:
        """No-op — see :meth:`ClaudeAuxiliaryClient.close`."""


def build_claude_auxiliary_client(
    model: str,
    *,
    async_mode: bool = False,
    timeout: Optional[float] = None,
) -> Any:
    """Build the auxiliary client, raising an actionable error when unusable.

    The SDK import happens here rather than at first call so a missing extra is
    reported while the caller can still fall back, instead of mid-task.
    """
    _import_sdk()
    client = ClaudeAuxiliaryClient(model, timeout=timeout)
    return AsyncClaudeAuxiliaryClient(client) if async_mode else client


__all__ = [
    "CLAUDE_AUX_MAX_TURNS",
    "CLAUDE_CODE_PROVIDER_ID",
    "DEFAULT_CLAUDE_AUX_TIMEOUT_SECONDS",
    "AsyncClaudeAuxiliaryClient",
    "ClaudeAuxiliaryClient",
    "ClaudeAuxiliaryError",
    "ClaudeImageInputUnsupported",
    "build_claude_auxiliary_client",
    "build_claude_auxiliary_options",
    "build_prompt",
    "is_claude_subscription_provider",
    "run_claude_auxiliary_query",
    "split_messages",
]
