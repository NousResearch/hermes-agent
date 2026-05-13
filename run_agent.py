#!/usr/bin/env python3
"""
AI Agent Runner with Tool Calling

This module provides a clean, standalone agent that can execute AI models
with tool calling capabilities. It handles the conversation loop, tool execution,
and response management.

Features:
- Automatic tool calling loop until completion
- Configurable model parameters
- Error handling and recovery
- Message history management
- Support for multiple model providers

Usage:
    from run_agent import AIAgent
    
    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass

import asyncio
import base64
import concurrent.futures
import contextvars
import copy
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import random
import re
import ssl
import sys
import tempfile
import time
import threading
from types import SimpleNamespace
import urllib.request
import uuid
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs, urlunparse
# NOTE: `from openai import OpenAI` is deliberately NOT at module top — the
# SDK pulls ~240 ms of imports. We expose `OpenAI` as a thin proxy object
# that imports the SDK on first call/isinstance check. This preserves:
#   (a) the single in-module `OpenAI(**client_kwargs)` call site at
#       _create_openai_client, and
#   (b) `patch("run_agent.OpenAI", ...)` test patterns used by ~28 test files.
#
# NOTE: `fire` is ONLY used in the `__main__` block below (for running
# run_agent.py directly as a CLI) — it is NOT needed for library usage.
# It is imported there, not here, so that importing run_agent from a
# daemon thread (e.g. curator's forked review agent) never fails with
# ModuleNotFoundError on broken/partial installs where `fire` isn't present.
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home


_OPENAI_CLS_CACHE: Optional[type] = None


def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls
        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    """Module-level proxy that looks like ``openai.OpenAI`` but imports lazily."""

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return "<lazy openai.OpenAI proxy>"


OpenAI = _OpenAIProxy()

# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.timeouts import (
    get_provider_request_timeout,
    get_provider_stale_timeout,
)

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")


# Import our tool system
from model_tools import (
    get_tool_definitions,
    get_toolset_for_tool,
    handle_function_call,
    check_toolset_requirements,
)
from tools.terminal_tool import cleanup_vm, get_active_env, is_persistent_env
from tools.terminal_tool import (
    set_approval_callback as _set_approval_callback,
    set_sudo_password_callback as _set_sudo_password_callback,
    _get_approval_callback,
    _get_sudo_password_callback,
)
from tools.tool_result_storage import maybe_persist_tool_result, enforce_turn_budget
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser


# Agent internals extracted to agent/ package for modularity
from agent.memory_manager import StreamingContextScrubber, build_memory_context_block, sanitize_context
from agent.think_scrubber import StreamingThinkScrubber
from agent.retry_utils import jittered_backoff
from agent.error_classifier import classify_api_error, FailoverReason
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS,
    MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE,
    KANBAN_GUIDANCE,
    build_nous_subscription_prompt,
)
from agent.model_metadata import (
    fetch_model_metadata,
    estimate_tokens_rough, estimate_messages_tokens_rough, estimate_request_tokens_rough,
    get_next_probe_tier, parse_context_limit_from_error,
    parse_available_output_tokens_from_error,
    save_context_length, is_local_endpoint,
    query_ollama_num_ctx,
)
from agent.context_compressor import ContextCompressor
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.prompt_caching import apply_anthropic_cache_control
from agent.prompt_builder import build_skills_system_prompt, build_context_files_prompt, build_environment_hints, load_soul_md, TOOL_USE_ENFORCEMENT_GUIDANCE, TOOL_USE_ENFORCEMENT_MODELS, GOOGLE_MODEL_OPERATIONAL_GUIDANCE, OPENAI_MODEL_EXECUTION_GUIDANCE
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from agent.codex_responses_adapter import (
    _derive_responses_function_call_id as _codex_derive_responses_function_call_id,
    _deterministic_call_id as _codex_deterministic_call_id,
    _split_responses_tool_id as _codex_split_responses_tool_id,
    _summarize_user_message_for_log,
)
from agent.display import (
    KawaiiSpinner, build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
    _detect_tool_failure,
    get_tool_emoji as _get_tool_emoji,
)
from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolGuardrailDecision,
    append_toolguard_guidance,
    toolguard_synthetic_result,
)
from agent.trajectory import (
    convert_scratchpad_to_think, has_incomplete_scratchpad,
    save_trajectory as _save_trajectory_to_file,
)
from utils import atomic_json_write, base_url_host_matches, base_url_hostname, env_var_enabled, normalize_proxy_url
from hermes_cli.config import cfg_get



class _SafeWriter:
    """Transparent stdio wrapper that catches OSError/ValueError from broken pipes.

    When hermes-agent runs as a systemd service, Docker container, or headless
    daemon, the stdout/stderr pipe can become unavailable (idle timeout, buffer
    exhaustion, socket reset). Any print() call then raises
    ``OSError: [Errno 5] Input/output error``, which can crash agent setup or
    run_conversation() — especially via double-fault when an except handler
    also tries to print.

    Additionally, when subagents run in ThreadPoolExecutor threads, the shared
    stdout handle can close between thread teardown and cleanup, raising
    ``ValueError: I/O operation on closed file`` instead of OSError.

    This wrapper delegates all writes to the underlying stream and silently
    catches both OSError and ValueError. It is transparent when the wrapped
    stream is healthy.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _get_proxy_from_env() -> Optional[str]:
    """Read proxy URL from environment variables.

    Checks HTTPS_PROXY, HTTP_PROXY, ALL_PROXY (and lowercase variants) in order.
    Returns the first valid proxy URL found, or None if no proxy is configured.
    """
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = os.environ.get(key, "").strip()
        if value:
            return normalize_proxy_url(value)
    return None


def _get_proxy_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """Return an env-configured proxy unless NO_PROXY excludes this base URL."""
    proxy = _get_proxy_from_env()
    if not proxy or not base_url:
        return proxy

    host = base_url_hostname(base_url)
    if not host:
        return proxy

    try:
        if urllib.request.proxy_bypass_environment(host):
            return None
    except Exception:
        pass

    return proxy


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


class IterationBudget:
    """Thread-safe iteration counter for an agent.

    Each agent (parent or subagent) gets its own ``IterationBudget``.
    The parent's budget is capped at ``max_iterations`` (default 90).
    Each subagent gets an independent budget capped at
    ``delegation.max_iterations`` (default 50) — this means total
    iterations across parent + subagents can exceed the parent's cap.
    Users control the per-subagent limit via ``delegation.max_iterations``
    in config.yaml.

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


# Tools that must never run concurrently (interactive / user-facing).
# When any of these appear in a batch, we fall back to sequential execution.
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# Read-only tools with no shared mutable session state.
_PARALLEL_SAFE_TOOLS = frozenset({
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
})

# File tools can run concurrently when they target independent paths.
_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

# Tools that mutate files on disk.  Used by the per-turn verifier that
# surfaces silently-failed file edits so the model can't over-claim success.
_FILE_MUTATING_TOOLS = frozenset({"write_file", "patch"})

# Maximum number of concurrent worker threads for parallel tool execution.
_MAX_TOOL_WORKERS = 8

# Guard so the OpenRouter metadata pre-warm thread is only spawned once per
# process, not once per AIAgent instantiation.  Without this, long-running
# gateway processes leak one OS thread per incoming message and eventually
# exhaust the system thread limit (RuntimeError: can't start new thread).
_openrouter_prewarm_done = threading.Event()

# Patterns that indicate a terminal command may modify/delete files.
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        cp\s|install\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
# Output redirects that overwrite files (> but not >>)
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def _is_destructive_command(cmd: str) -> bool:
    """Heuristic: does this terminal command look like it modifies/deletes files?"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """Return True when a tool-call batch is safe to run concurrently."""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """Return the normalized file target for path-scoped tools."""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # Avoid resolve(); the file may not exist yet.
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        # Empty paths shouldn't reach here (guarded upstream), but be safe.
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]



_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')




def _is_multimodal_tool_result(value: Any) -> bool:
    """True if the value is a multimodal tool result envelope.

    Multimodal handlers (e.g. tools/computer_use) return a dict with
    `_multimodal=True`, a `content` key holding OpenAI-style content
    parts, and an optional `text_summary` for string-only fallbacks.
    """
    return (
        isinstance(value, dict)
        and value.get("_multimodal") is True
        and isinstance(value.get("content"), list)
    )


def _multimodal_text_summary(value: Any) -> str:
    """Extract a plain text view of a multimodal tool result.

    Used wherever downstream code needs a string — logging, previews,
    persistence size heuristics, fall-back content for providers that
    don't support multipart tool messages.
    """
    if _is_multimodal_tool_result(value):
        if value.get("text_summary"):
            return str(value["text_summary"])
        parts = []
        for p in value.get("content") or []:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        if parts:
            return "\n".join(parts)
        return "[multimodal tool result]"
    if isinstance(value, str):
        return value
    try:
        import json as _json
        return _json.dumps(value, default=str)
    except Exception:
        return str(value)


def _append_subdir_hint_to_multimodal(value: Dict[str, Any], hint: str) -> None:
    """Mutate a multimodal tool-result envelope to append a subdir hint.

    The hint is added to the first text part so the model sees it; image
    parts are left untouched. `text_summary` is also updated for
    string-fallback callers.
    """
    if not _is_multimodal_tool_result(value):
        return
    parts = value.get("content") or []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            p["text"] = str(p.get("text", "")) + hint
            break
    else:
        parts.insert(0, {"type": "text", "text": hint})
        value["content"] = parts
    if isinstance(value.get("text_summary"), str):
        value["text_summary"] = value["text_summary"] + hint


def _extract_file_mutation_targets(tool_name: str, args: Dict[str, Any]) -> List[str]:
    """Return the file paths a ``write_file`` or ``patch`` call is targeting.

    For ``write_file`` and ``patch`` in replace mode this is just ``args["path"]``.
    For ``patch`` in V4A patch mode we parse the patch content for
    ``*** Update File:`` / ``*** Add File:`` / ``*** Delete File:`` headers so
    the verifier can track each file in a multi-file patch separately.
    """
    if tool_name not in _FILE_MUTATING_TOOLS:
        return []
    if tool_name == "write_file":
        p = args.get("path")
        return [str(p)] if p else []
    # tool_name == "patch"
    mode = args.get("mode") or "replace"
    if mode == "replace":
        p = args.get("path")
        return [str(p)] if p else []
    if mode == "patch":
        body = args.get("patch") or ""
        if not isinstance(body, str) or not body:
            return []
        import re as _re
        paths: List[str] = []
        for _m in _re.finditer(
            r'^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$',
            body,
            _re.MULTILINE,
        ):
            p = _m.group(1).strip()
            if p:
                paths.append(p)
        return paths
    return []


def _extract_error_preview(result: Any, max_len: int = 180) -> str:
    """Pull a one-line error summary out of a tool result for footer display."""
    text = _multimodal_text_summary(result) if result is not None else ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    # Try to parse JSON and pull the ``error`` field — tool handlers return
    # ``{"success": false, "error": "..."}``; raw string wins if parse fails.
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            import json as _json
            data = _json.loads(stripped)
            if isinstance(data, dict) and isinstance(data.get("error"), str):
                text = data["error"]
        except Exception:
            pass
    # Collapse whitespace, trim to max_len.
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def _trajectory_normalize_msg(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Strip image blobs from a message for trajectory saving.

    Returns a shallow copy with multimodal tool results replaced by their
    text_summary, and image parts in content lists replaced by
    `[screenshot]` placeholders. Keeps the message schema otherwise intact.
    """
    if not isinstance(msg, dict):
        return msg
    content = msg.get("content")
    if _is_multimodal_tool_result(content):
        return {**msg, "content": _multimodal_text_summary(content)}
    if isinstance(content, list):
        cleaned = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in {"image", "image_url", "input_image"}:
                cleaned.append({"type": "text", "text": "[screenshot]"})
            else:
                cleaned.append(p)
        return {**msg, "content": cleaned}
    return msg


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate code points with U+FFFD (replacement character).

    Surrogates are invalid in UTF-8 and will crash ``json.dumps()`` inside the
    OpenAI SDK.  This is a fast no-op when the text contains no surrogates.
    """
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


# _summarize_user_message_for_log is imported from agent.codex_responses_adapter
# (see import block above). Remains importable from run_agent for backward compat.


def _sanitize_structure_surrogates(payload: Any) -> bool:
    """Replace surrogate code points in nested dict/list payloads in-place.

    Mirror of ``_sanitize_structure_non_ascii`` but for surrogate recovery.
    Used to scrub nested structured fields (e.g. ``reasoning_details`` — an
    array of dicts with ``summary``/``text`` strings) that flat per-field
    checks don't reach.  Returns True if any surrogates were replaced.
    """
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[key] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[idx] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found


def _sanitize_messages_surrogates(messages: list) -> bool:
    """Sanitize surrogate characters from all string content in a messages list.

    Walks message dicts in-place. Returns True if any surrogates were found
    and replaced, False otherwise. Covers content/text, name, tool call
    metadata/arguments, AND any additional string or nested structured fields
    (``reasoning``, ``reasoning_content``, ``reasoning_details``, etc.) so
    retries don't fail on a non-content field.  Byte-level reasoning models
    (xiaomi/mimo, kimi, glm) can emit lone surrogates in reasoning output
    that flow through to ``api_messages["reasoning_content"]`` on the next
    turn and crash json.dumps inside the OpenAI SDK.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
        name = msg.get("name")
        if isinstance(name, str) and _SURROGATE_RE.search(name):
            msg["name"] = _SURROGATE_RE.sub('\ufffd', name)
            found = True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and _SURROGATE_RE.search(tc_id):
                    tc["id"] = _SURROGATE_RE.sub('\ufffd', tc_id)
                    found = True
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and _SURROGATE_RE.search(fn_name):
                        fn["name"] = _SURROGATE_RE.sub('\ufffd', fn_name)
                        found = True
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str) and _SURROGATE_RE.search(fn_args):
                        fn["arguments"] = _SURROGATE_RE.sub('\ufffd', fn_args)
                        found = True
        # Walk any additional string / nested fields (reasoning,
        # reasoning_content, reasoning_details, etc.) — surrogates from
        # byte-level reasoning models (xiaomi/mimo, kimi, glm) can lurk
        # in these fields and aren't covered by the per-field checks above.
        # Matches _sanitize_messages_non_ascii's coverage (PR #10537).
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                if _SURROGATE_RE.search(value):
                    msg[key] = _SURROGATE_RE.sub('\ufffd', value)
                    found = True
            elif isinstance(value, (dict, list)):
                if _sanitize_structure_surrogates(value):
                    found = True
    return found


def _escape_invalid_chars_in_json_strings(raw: str) -> str:
    """Escape unescaped control chars inside JSON string values.

    Walks the raw JSON character-by-character, tracking whether we are
    inside a double-quoted string. Inside strings, replaces literal
    control characters (0x00-0x1F) that aren't already part of an escape
    sequence with their ``\\uXXXX`` equivalents. Pass-through for everything
    else.

    Ported from #12093 — complements the other repair passes in
    ``_repair_tool_call_arguments`` when ``json.loads(strict=False)`` is
    not enough (e.g. llama.cpp backends that emit literal apostrophes or
    tabs alongside other malformations).
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                # Already-escaped char — pass through as-is
                out.append(ch)
                out.append(raw[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
        i += 1
    return "".join(out)


def _repair_tool_call_arguments(raw_args: str, tool_name: str = "?") -> str:
    """Attempt to repair malformed tool_call argument JSON.

    Models like GLM-5.1 via Ollama can produce truncated JSON, trailing
    commas, Python ``None``, etc.  The API proxy rejects these with HTTP 400
    "invalid tool call arguments".  This function applies common repairs;
    if all fail it returns ``"{}"`` so the request succeeds (better than
    crashing the session).  All repairs are logged at WARNING level.
    """
    raw_stripped = raw_args.strip() if isinstance(raw_args, str) else ""

    # Fast-path: empty / whitespace-only -> empty object
    if not raw_stripped:
        logger.warning("Sanitized empty tool_call arguments for %s", tool_name)
        return "{}"

    # Python-literal None -> normalise to {}
    if raw_stripped == "None":
        logger.warning("Sanitized Python-None tool_call arguments for %s", tool_name)
        return "{}"

    # Repair pass 0: llama.cpp backends sometimes emit literal control
    # characters (tabs, newlines) inside JSON string values. json.loads
    # with strict=False accepts these and lets us re-serialise the
    # result into wire-valid JSON without any string surgery. This is
    # the most common local-model repair case (#12068).
    try:
        parsed = json.loads(raw_stripped, strict=False)
        reserialised = json.dumps(parsed, separators=(",", ":"))
        if reserialised != raw_stripped:
            logger.warning(
                "Repaired unescaped control chars in tool_call arguments for %s",
                tool_name,
            )
        return reserialised
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Attempt common JSON repairs
    fixed = raw_stripped
    # 1. Strip trailing commas before } or ]
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # 2. Close unclosed structures
    open_curly = fixed.count('{') - fixed.count('}')
    open_bracket = fixed.count('[') - fixed.count(']')
    if open_curly > 0:
        fixed += '}' * open_curly
    if open_bracket > 0:
        fixed += ']' * open_bracket
    # 3. Remove excess closing braces/brackets (bounded to 50 iterations)
    for _ in range(50):
        try:
            json.loads(fixed)
            break
        except json.JSONDecodeError:
            if fixed.endswith('}') and fixed.count('}') > fixed.count('{'):
                fixed = fixed[:-1]
            elif fixed.endswith(']') and fixed.count(']') > fixed.count('['):
                fixed = fixed[:-1]
            else:
                break

    try:
        json.loads(fixed)
        logger.warning(
            "Repaired malformed tool_call arguments for %s: %s → %s",
            tool_name, raw_stripped[:80], fixed[:80],
        )
        return fixed
    except json.JSONDecodeError:
        pass

    # Repair pass 4: escape unescaped control chars inside JSON strings,
    # then retry. Catches cases where strict=False alone fails because
    # other malformations are present too.
    try:
        escaped = _escape_invalid_chars_in_json_strings(fixed)
        if escaped != fixed:
            json.loads(escaped)
            logger.warning(
                "Repaired control-char-laced tool_call arguments for %s: %s → %s",
                tool_name, raw_stripped[:80], escaped[:80],
            )
            return escaped
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Last resort: replace with empty object so the API request doesn't
    # crash the entire session.
    logger.warning(
        "Unrepairable tool_call arguments for %s — "
        "replaced with empty object (was: %s)",
        tool_name, raw_stripped[:80],
    )
    return "{}"


def _strip_non_ascii(text: str) -> str:
    """Remove non-ASCII characters, replacing with closest ASCII equivalent or removing.

    Used as a last resort when the system encoding is ASCII and can't handle
    any non-ASCII characters (e.g. LANG=C on Chromebooks).
    """
    return text.encode('ascii', errors='ignore').decode('ascii')


def _sanitize_messages_non_ascii(messages: list) -> bool:
    """Strip non-ASCII characters from all string content in a messages list.

    This is a last-resort recovery for systems with ASCII-only encoding
    (LANG=C, Chromebooks, minimal containers).  Returns True if any
    non-ASCII content was found and sanitized.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # Sanitize content (string)
        content = msg.get("content")
        if isinstance(content, str):
            sanitized = _strip_non_ascii(content)
            if sanitized != content:
                msg["content"] = sanitized
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        sanitized = _strip_non_ascii(text)
                        if sanitized != text:
                            part["text"] = sanitized
                            found = True
        # Sanitize name field (can contain non-ASCII in tool results)
        name = msg.get("name")
        if isinstance(name, str):
            sanitized = _strip_non_ascii(name)
            if sanitized != name:
                msg["name"] = sanitized
                found = True
        # Sanitize tool_calls
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        fn_args = fn.get("arguments")
                        if isinstance(fn_args, str):
                            sanitized = _strip_non_ascii(fn_args)
                            if sanitized != fn_args:
                                fn["arguments"] = sanitized
                                found = True
        # Sanitize any additional top-level string fields (e.g. reasoning_content)
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                sanitized = _strip_non_ascii(value)
                if sanitized != value:
                    msg[key] = sanitized
                    found = True
    return found


def _sanitize_tools_non_ascii(tools: list) -> bool:
    """Strip non-ASCII characters from tool payloads in-place."""
    return _sanitize_structure_non_ascii(tools)


def _strip_images_from_messages(messages: list) -> bool:
    """Remove image_url content parts from all messages in-place.

    Called when a server signals it does not support images (e.g.
    "Only 'text' content type is supported.").  Mutates messages so the
    next API call sends text only.

    Preserves message alternation invariants:
      * ``tool``-role messages whose content was entirely images are replaced
        with a plaintext placeholder, NOT deleted — deleting them would leave
        the paired ``tool_call_id`` on the prior assistant message unmatched,
        which providers reject with HTTP 400.
      * Non-tool messages whose content becomes empty are dropped.  In
        practice this only hits synthetic image-only user messages appended
        for attachment delivery; real user turns always include text.

    Returns True if any image parts were removed.
    """
    found = False
    to_delete = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "image", "input_image"}:
                found = True
            else:
                new_parts.append(part)
        if len(new_parts) < len(content):
            if new_parts:
                msg["content"] = new_parts
            elif msg.get("role") == "tool":
                # Preserve tool_call_id linkage — providers require every
                # assistant tool_call to have a matching tool response.
                msg["content"] = "[image content removed — server does not support images]"
            else:
                # Synthetic image-only user/assistant message with no text;
                # safe to drop.
                to_delete.append(i)
    for i in reversed(to_delete):
        del messages[i]
    return found


def _sanitize_structure_non_ascii(payload: Any) -> bool:
    """Strip non-ASCII characters from nested dict/list payloads in-place."""
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[key] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[idx] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found





# =========================================================================
# Large tool result handler — save oversized output to temp file
# =========================================================================


# =========================================================================
# Qwen Portal headers — mimics QwenCode CLI for portal.qwen.ai compatibility.
# Extracted as a module-level helper so both __init__ and
# _apply_client_headers_for_base_url can share it.
# =========================================================================
_QWEN_CODE_VERSION = "0.14.1"


def _routermint_headers() -> dict:
    """Return the User-Agent RouterMint needs to avoid Cloudflare 1010 blocks."""
    from hermes_cli import __version__ as _HERMES_VERSION

    return {
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    }


def _pool_may_recover_from_rate_limit(
    pool, *, provider: str | None = None, base_url: str | None = None
) -> bool:
    """Decide whether to wait for credential-pool rotation instead of falling back.

    The existing pool-rotation path requires the pool to (1) exist and (2) have
    at least one entry not currently in exhaustion cooldown.  But rotation is
    only meaningful when the pool has more than one entry.

    With a single-credential pool (common for Gemini OAuth, Vertex service
    accounts, and any "one personal key" configuration), the primary entry
    just 429'd and there is nothing to rotate to.  Waiting for the pool
    cooldown to expire means retrying against the same exhausted quota — the
    daily-quota 429 will recur immediately, and the retry budget is burned.

    Additionally, Google CloudCode / Gemini CLI rate limits are ACCOUNT-level
    throttles — even a multi-entry pool shares the same quota window, so
    rotation won't recover.  Skip straight to the fallback for those (#13636).

    In those cases we must fall back to the configured ``fallback_model``
    instead.  Returns True only when rotation has somewhere to go.

    See issues #11314 and #13636.
    """
    if pool is None:
        return False
    if not pool.has_available():
        return False
    # CloudCode / Gemini CLI quotas are account-wide — all pool entries share
    # the same throttle window, so rotation can't recover.  Prefer fallback.
    if provider == "google-gemini-cli" or str(base_url or "").startswith("cloudcode-pa://"):
        return False
    return len(pool.entries()) > 1


def _qwen_portal_headers() -> dict:
    """Return default HTTP headers required by Qwen Portal API."""
    import platform as _plat

    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }


class AIAgent:
    """
    AI Agent with tool calling capabilities.

    This class manages the conversation flow, tool execution, and response handling
    for AI models that support function calling.
    """

    _TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER = (
        "[hermes-agent: tool call arguments were corrupted in this session and "
        "have been dropped to keep the conversation alive. See issue #15236.]"
    )

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""
        self._base_url_hostname = base_url_hostname(value)

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        acp_command: str = None,
        acp_args: list[str] | None = None,
        command: str = None,
        args: list[str] | None = None,
        model: str = "",
        max_iterations: int = 90,  # Default tool-calling iterations (shared with subagents)
        tool_delay: float = 1.0,
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
        provider_require_parameters: bool = False,
        provider_data_collection: str = None,
        openrouter_min_coding_score: Optional[float] = None,
        session_id: str = None,
        tool_progress_callback: callable = None,
        tool_start_callback: callable = None,
        tool_complete_callback: callable = None,
        thinking_callback: callable = None,
        reasoning_callback: callable = None,
        clarify_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None,
        status_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        service_tier: str = None,
        request_overrides: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        user_id: str = None,
        user_name: str = None,
        chat_id: str = None,
        chat_name: str = None,
        chat_type: str = None,
        thread_id: str = None,
        gateway_session_key: str = None,
        skip_context_files: bool = False,
        load_soul_identity: bool = False,
        skip_memory: bool = False,
        session_db=None,
        parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None,
        fallback_model: Dict[str, Any] = None,
        credential_pool=None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 20,
        checkpoint_max_total_size_mb: int = 500,
        checkpoint_max_file_size_mb: int = 10,
        pass_session_id: bool = False,
    ):
        """
        Initialize the AI Agent.

        Args:
            base_url (str): Base URL for the model API (optional)
            api_key (str): API key for authentication (optional, uses env var if not provided)
            provider (str): Provider identifier (optional; used for telemetry/routing hints)
            api_mode (str): API mode override: "chat_completions" or "codex_responses"
            model (str): Model name to use (default: "anthropic/claude-opus-4.6")
            max_iterations (int): Maximum number of tool calling iterations (default: 90)
            tool_delay (float): Delay between tool calls in seconds (default: 1.0)
            enabled_toolsets (List[str]): Only enable tools from these toolsets (optional)
            disabled_toolsets (List[str]): Disable tools from these toolsets (optional)
            save_trajectories (bool): Whether to save conversation trajectories to JSONL files (default: False)
            verbose_logging (bool): Enable verbose logging for debugging (default: False)
            quiet_mode (bool): Suppress progress output for clean CLI experience (default: False)
            ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
            log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 100)
            log_prefix (str): Prefix to add to all log messages for identification in parallel processing (default: "")
            providers_allowed (List[str]): OpenRouter providers to allow (optional)
            providers_ignored (List[str]): OpenRouter providers to ignore (optional)
            providers_order (List[str]): OpenRouter providers to try in order (optional)
            provider_sort (str): Sort providers by price/throughput/latency (optional)
            openrouter_min_coding_score (float): Coding-score floor (0.0-1.0) for the
                openrouter/pareto-code router. Only applied when model == "openrouter/pareto-code".
                None or empty = let OpenRouter pick the strongest available coder.
            session_id (str): Pre-generated session ID for logging (optional, auto-generated if not provided)
            tool_progress_callback (callable): Callback function(tool_name, args_preview) for progress notifications
            clarify_callback (callable): Callback function(question, choices) -> str for interactive user questions.
                Provided by the platform layer (CLI or gateway). If None, the clarify tool returns an error.
            max_tokens (int): Maximum tokens for model responses (optional, uses model default if not set)
            reasoning_config (Dict): OpenRouter reasoning configuration override (e.g. {"effort": "none"} to disable thinking).
                If None, defaults to {"enabled": True, "effort": "medium"} for OpenRouter. Set to disable/customize reasoning.
            prefill_messages (List[Dict]): Messages to prepend to conversation history as prefilled context.
                Useful for injecting a few-shot example or priming the model's response style.
                Example: [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
                NOTE: Anthropic Sonnet 4.6+ and Opus 4.6+ reject a conversation that ends on an
                assistant-role message (400 error).  For those models use structured outputs or
                output_config.format instead of a trailing-assistant prefill.
            platform (str): The interface platform the user is on (e.g. "cli", "telegram", "discord", "whatsapp").
                Used to inject platform-specific formatting hints into the system prompt.
            skip_context_files (bool): If True, skip auto-injection of SOUL.md, AGENTS.md, and .cursorrules
                into the system prompt. Use this for batch processing and data generation to avoid
                polluting trajectories with user-specific persona or project instructions.
            load_soul_identity (bool): If True, still use ~/.hermes/SOUL.md as the primary
                identity even when skip_context_files=True. Project context files from the cwd
                remain skipped.
        """
        _install_safe_stdio()

        self.model = model
        self.max_iterations = max_iterations
        # Shared iteration budget — parent creates, children inherit.
        # Consumed by every LLM turn across parent + all subagents.
        self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform  # "cli", "telegram", "discord", "whatsapp", etc.
        self._user_id = user_id  # Platform user identifier (gateway sessions)
        self._user_name = user_name
        self._chat_id = chat_id
        self._chat_name = chat_name
        self._chat_type = chat_type
        self._thread_id = thread_id
        self._gateway_session_key = gateway_session_key  # Stable per-chat key (e.g. agent:main:telegram:dm:123)
        # Pluggable print function — CLI replaces this with _cprint so that
        # raw ANSI status lines are routed through prompt_toolkit's renderer
        # instead of going directly to stdout where patch_stdout's StdoutProxy
        # would mangle the escape sequences.  None = use builtins.print.
        self._print_fn = None
        self.background_review_callback = None  # Optional sync callback for gateway delivery
        self.skip_context_files = skip_context_files
        self.load_soul_identity = load_soul_identity
        self.pass_session_id = pass_session_id
        self._credential_pool = credential_pool
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        # Store effective base URL for feature detection (prompt caching, reasoning, etc.)
        self.base_url = base_url or ""
        provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        self.provider = provider_name or ""
        self.acp_command = acp_command or command
        self.acp_args = list(acp_args or args or [])
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif self.provider == "xai":
            self.api_mode = "codex_responses"
        elif (provider_name is None) and (
            self._base_url_hostname == "chatgpt.com"
            and "/backend-api/codex" in self._base_url_lower
        ):
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif (provider_name is None) and self._base_url_hostname == "api.x.ai":
            self.api_mode = "codex_responses"
            self.provider = "xai"
        elif self.provider == "anthropic" or (provider_name is None and self._base_url_hostname == "api.anthropic.com"):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        elif self._base_url_lower.rstrip("/").endswith("/anthropic"):
            # Third-party Anthropic-compatible endpoints (e.g. MiniMax, DashScope)
            # use a URL convention ending in /anthropic. Auto-detect these so the
            # Anthropic Messages API adapter is used instead of chat completions.
            self.api_mode = "anthropic_messages"
        elif self.provider == "bedrock" or (
            self._base_url_hostname.startswith("bedrock-runtime.")
            and base_url_host_matches(self._base_url_lower, "amazonaws.com")
        ):
            # AWS Bedrock — auto-detect from provider name or base URL
            # (bedrock-runtime.<region>.amazonaws.com).
            self.api_mode = "bedrock_converse"
        else:
            self.api_mode = "chat_completions"

        # Eagerly warm the transport cache so import errors surface at init,
        # not mid-conversation.  Also validates the api_mode is registered.
        try:
            self._get_transport()
        except Exception:
            pass  # Non-fatal — transport may not exist for all modes yet

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS,
                normalize_model_for_provider,
            )

            if self.provider not in _AGGREGATOR_PROVIDERS:
                self.model = normalize_model_for_provider(self.model, self.provider)
        except Exception:
            pass

        # GPT-5.x models usually require the Responses API path, but some
        # providers have exceptions (for example Copilot's gpt-5-mini still
        # uses chat completions). Also auto-upgrade for direct OpenAI URLs
        # (api.openai.com) since all newer tool-calling models prefer
        # Responses there. ACP runtimes are excluded: CopilotACPClient
        # handles its own routing and does not implement the Responses API
        # surface.
        # When api_mode was explicitly provided, respect it — the user
        # knows what their endpoint supports (#10473).
        # Exception: Azure OpenAI serves gpt-5.x on /chat/completions and
        # does NOT support the Responses API — skip the upgrade for Azure
        # (openai.azure.com), even though it looks OpenAI-compatible.
        if (
            api_mode is None
            and self.api_mode == "chat_completions"
            and self.provider != "copilot-acp"
            and not str(self.base_url or "").lower().startswith("acp://copilot")
            and not str(self.base_url or "").lower().startswith("acp+tcp://")
            and not self._is_azure_openai_url()
            and (
                self._is_direct_openai_url()
                or self._provider_model_requires_responses_api(
                    self.model,
                    provider=self.provider,
                )
            )
        ):
            self.api_mode = "codex_responses"
            # Invalidate the eager-warmed transport cache — api_mode changed
            # from chat_completions to codex_responses after the warm at __init__.
            if hasattr(self, "_transport_cache"):
                self._transport_cache.clear()

        # Pre-warm OpenRouter model metadata cache in a background thread.
        # fetch_model_metadata() is cached for 1 hour; this avoids a blocking
        # HTTP request on the first API response when pricing is estimated.
        # Use a process-level Event so this thread is only spawned once — a new
        # AIAgent is created for every gateway request, so without the guard
        # each message leaks one OS thread and the process eventually exhausts
        # the system thread limit (RuntimeError: can't start new thread).
        if (self.provider == "openrouter" or self._is_openrouter_url()) and \
                not _openrouter_prewarm_done.is_set():
            _openrouter_prewarm_done.set()
            threading.Thread(
                target=fetch_model_metadata,
                daemon=True,
                name="openrouter-prewarm",
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.suppress_status_output = False
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self.interim_assistant_callback = interim_assistant_callback
        self.status_callback = status_callback
        self.tool_gen_callback = tool_gen_callback

        
        # Tool execution state — allows _vprint during tool execution
        # even when stream consumers are registered (no tokens streaming then)
        self._executing_tools = False
        self._tool_guardrails = ToolCallGuardrailController()
        self._tool_guardrail_halt_decision: ToolGuardrailDecision | None = None

        # Interrupt mechanism for breaking out of tool loops
        self._interrupt_requested = False
        self._interrupt_message = None  # Optional message that triggered interrupt
        self._execution_thread_id: int | None = None  # Set at run_conversation() start
        self._interrupt_thread_signal_pending = False
        self._client_lock = threading.RLock()

        # /steer mechanism — inject a user note into the next tool result
        # without interrupting the agent. Unlike interrupt(), steer() does
        # NOT set _interrupt_requested; it waits for the current tool batch
        # to finish naturally, then the drain hook appends the text to the
        # last tool result's content so the model sees it on its next
        # iteration. Message-role alternation is preserved (we modify an
        # existing tool message rather than inserting a new user turn).
        self._pending_steer: Optional[str] = None
        self._pending_steer_lock = threading.Lock()

        # Concurrent-tool worker thread tracking.  `_execute_tool_calls_concurrent`
        # runs each tool on its own ThreadPoolExecutor worker — those worker
        # threads have tids distinct from `_execution_thread_id`, so
        # `_set_interrupt(True, _execution_thread_id)` alone does NOT cause
        # `is_interrupted()` inside the worker to return True.  Track the
        # workers here so `interrupt()` / `clear_interrupt()` can fan out to
        # their tids explicitly.
        self._tool_worker_threads: set[int] = set()
        self._tool_worker_threads_lock = threading.Lock()
        
        # Subagent delegation state
        self._delegate_depth = 0        # 0 = top-level agent, incremented for children
        self._active_children = []      # Running child AIAgents (for interrupt propagation)
        self._active_children_lock = threading.Lock()
        
        # Store OpenRouter provider preferences
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.provider_require_parameters = provider_require_parameters
        self.provider_data_collection = provider_data_collection
        self.openrouter_min_coding_score = openrouter_min_coding_score

        # Store toolset filtering options
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # Model response configuration
        self.max_tokens = max_tokens  # None = use model default
        self.reasoning_config = reasoning_config  # None = use default (medium for OpenRouter)
        self.service_tier = service_tier
        self.request_overrides = dict(request_overrides or {})
        self.prefill_messages = prefill_messages or []  # Prefilled conversation turns
        self._force_ascii_payload = False
        
        # Anthropic prompt caching: auto-enabled for Claude models on native
        # Anthropic, OpenRouter, and third-party gateways that speak the
        # Anthropic protocol (``api_mode == 'anthropic_messages'``). Reduces
        # input costs by ~75% on multi-turn conversations. Uses system_and_3
        # strategy (4 breakpoints). See ``_anthropic_prompt_cache_policy``
        # for the layout-vs-transport decision.
        self._use_prompt_caching, self._use_native_cache_layout = (
            self._anthropic_prompt_cache_policy()
        )
        # Anthropic supports "5m" (default) and "1h" cache TTL tiers. Read from
        # config.yaml under prompt_caching.cache_ttl; unknown values keep "5m".
        # 1h tier costs 2x on write vs 1.25x for 5m, but amortizes across long
        # sessions with >5-minute pauses between turns (#14971).
        self._cache_ttl = "5m"
        try:
            from hermes_cli.config import load_config as _load_pc_cfg

            _pc_cfg = _load_pc_cfg().get("prompt_caching", {}) or {}
            _ttl = _pc_cfg.get("cache_ttl", "5m")
            if _ttl in {"5m", "1h"}:
                self._cache_ttl = _ttl
        except Exception:
            pass

        # Iteration budget: the LLM is only notified when it actually exhausts
        # the iteration budget (api_call_count >= max_iterations).  At that
        # point we inject ONE message, allow one final API call, and if the
        # model doesn't produce a text response, force a user-message asking
        # it to summarise.  No intermediate pressure warnings — they caused
        # models to "give up" prematurely on complex tasks (#7915).
        self._budget_exhausted_injected = False
        self._budget_grace_call = False

        # Activity tracking — updated on each API call, tool execution, and
        # stream chunk.  Used by the gateway timeout handler to report what the
        # agent was doing when it was killed, and by the "still working"
        # notifications to show progress.
        self._last_activity_ts: float = time.time()
        self._last_activity_desc: str = "initializing"
        self._current_tool: str | None = None
        self._api_call_count: int = 0

        # Rate limit tracking — updated from x-ratelimit-* response headers
        # after each API call.  Accessed by /usage slash command.
        self._rate_limit_state: Optional["RateLimitState"] = None

        # OpenRouter response cache hit counter — incremented when
        # X-OpenRouter-Cache-Status: HIT is seen in streaming response headers.
        self._or_cache_hits: int = 0

        # Centralized logging — agent.log (INFO+) and errors.log (WARNING+)
        # both live under ~/.hermes/logs/.  Idempotent, so gateway mode
        # (which creates a new AIAgent per message) won't duplicate handlers.
        from hermes_logging import setup_logging, setup_verbose_logging
        setup_logging(hermes_home=_hermes_home)

        if self.verbose_logging:
            setup_verbose_logging()
            logger.info("Verbose logging enabled (third-party library logs suppressed)")
        elif self.quiet_mode:
            # In quiet mode (CLI default), keep console output clean —
            # but DO NOT raise per-logger levels. Doing so prevents the
            # root logger's file handlers (agent.log, errors.log) from
            # ever seeing the records, because Python checks
            # logger.isEnabledFor() before handler propagation. We rely
            # on the fact that hermes_logging.setup_logging() does not
            # install a console StreamHandler in quiet mode — so INFO
            # records flow to the file handlers but never reach a
            # console. Any future noise reduction belongs at the
            # handler level inside hermes_logging.py, not here.
            pass
        
        # Internal stream callback (set during streaming TTS).
        # Initialized here so _vprint can reference it before run_conversation.
        self._stream_callback = None
        # Deferred paragraph break flag — set after tool iterations so a
        # single "\n\n" is prepended to the next real text delta.
        self._stream_needs_break = False
        # Stateful scrubber for <memory-context> spans split across stream
        # deltas (#5719).  sanitize_context() alone can't survive chunk
        # boundaries because the block regex needs both tags in one string.
        self._stream_context_scrubber = StreamingContextScrubber()
        # Stateful scrubber for reasoning/thinking tags in streamed deltas
        # (#17924).  Replaces the per-delta _strip_think_blocks regex that
        # destroyed downstream state (e.g. MiniMax-M2.7 streaming
        # '<think>' as delta1 and 'Let me check' as delta2 — the regex
        # erased delta1, so downstream state machines never learned a
        # block was open and leaked delta2 as content).
        self._stream_think_scrubber = StreamingThinkScrubber()
        # Visible assistant text already delivered through live token callbacks
        # during the current model response. Used to avoid re-sending the same
        # commentary when the provider later returns it as a completed interim
        # assistant message.
        self._current_streamed_assistant_text = ""

        # Optional current-turn user-message override used when the API-facing
        # user message intentionally differs from the persisted transcript
        # (e.g. CLI voice mode adds a temporary prefix for the live call only).
        self._persist_user_message_idx = None
        self._persist_user_message_override = None

        # Cache anthropic image-to-text fallbacks per image payload/URL so a
        # single tool loop does not repeatedly re-run auxiliary vision on the
        # same image history.
        self._anthropic_image_fallback_cache: Dict[str, str] = {}

        # Initialize LLM client via centralized provider router.
        # The router handles auth resolution, base URL, headers, and
        # Codex/Anthropic wrapping for all known providers.
        # raw_codex=True because the main agent needs direct responses.stream()
        # access for Codex Responses API streaming.
        self._anthropic_client = None
        self._is_anthropic_oauth = False

        # Resolve per-provider / per-model request timeout once up front so
        # every client construction path below (Anthropic native, OpenAI-wire,
        # router-based implicit auth) can apply it consistently.  Bedrock
        # Claude uses its own timeout path and is not covered here.
        _provider_timeout = get_provider_request_timeout(self.provider, self.model)

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            # Bedrock + Claude → use AnthropicBedrock SDK for full feature parity
            # (prompt caching, thinking budgets, adaptive thinking).
            _is_bedrock_anthropic = self.provider == "bedrock"
            if _is_bedrock_anthropic:
                from agent.anthropic_adapter import build_anthropic_bedrock_client
                _region_match = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
                _br_region = _region_match.group(1) if _region_match else "us-east-1"
                self._bedrock_region = _br_region
                self._anthropic_client = build_anthropic_bedrock_client(_br_region)
                self._anthropic_api_key = "aws-sdk"
                self._anthropic_base_url = base_url
                self._is_anthropic_oauth = False
                self.api_key = "aws-sdk"
                self.client = None
                self._client_kwargs = {}
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model} (AWS Bedrock + AnthropicBedrock SDK, {_br_region})")
            else:
                # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
                # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own API key.
                # Falling back would send Anthropic credentials to third-party endpoints (Fixes #1739, #minimax-401).
                _is_native_anthropic = self.provider == "anthropic"
                effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = base_url
                # Only mark the session as OAuth-authenticated when the token
                # genuinely belongs to native Anthropic.  Third-party providers
                # (MiniMax, Kimi, GLM, LiteLLM proxies) that accept the
                # Anthropic protocol must never trip OAuth code paths — doing
                # so injects Claude-Code identity headers and system prompts
                # that cause 401/403 on their endpoints.  Guards #1739 and
                # the third-party identity-injection bug.
                from agent.anthropic_adapter import _is_oauth_token as _is_oat
                self._is_anthropic_oauth = _is_oat(effective_key) if _is_native_anthropic else False
                self._anthropic_client = build_anthropic_client(effective_key, base_url, timeout=_provider_timeout)
                # No OpenAI client needed for Anthropic mode
                self.client = None
                self._client_kwargs = {}
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)")
                    if effective_key and len(effective_key) > 12:
                        print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        elif self.api_mode == "bedrock_converse":
            # AWS Bedrock — uses boto3 directly, no OpenAI client needed.
            # Region is extracted from the base_url or defaults to us-east-1.
            _region_match = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
            self._bedrock_region = _region_match.group(1) if _region_match else "us-east-1"
            # Guardrail config — read from config.yaml at init time.
            self._bedrock_guardrail_config = None
            try:
                from hermes_cli.config import load_config as _load_br_cfg
                _gr = _load_br_cfg().get("bedrock", {}).get("guardrail", {})
                if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
                    self._bedrock_guardrail_config = {
                        "guardrailIdentifier": _gr["guardrail_identifier"],
                        "guardrailVersion": _gr["guardrail_version"],
                    }
                    if _gr.get("stream_processing_mode"):
                        self._bedrock_guardrail_config["streamProcessingMode"] = _gr["stream_processing_mode"]
                    if _gr.get("trace"):
                        self._bedrock_guardrail_config["trace"] = _gr["trace"]
            except Exception:
                pass
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                _gr_label = " + Guardrails" if self._bedrock_guardrail_config else ""
                print(f"🤖 AI Agent initialized with model: {self.model} (AWS Bedrock, {self._bedrock_region}{_gr_label})")
        else:
            if api_key and base_url:
                # Explicit credentials from CLI/gateway — construct directly.
                # The runtime provider resolver already handled auth for us.
                # Extract query params (e.g. Azure api-version) from base_url
                # and pass via default_query to prevent loss during SDK URL
                # joining (httpx drops query string when joining paths).
                _parsed_url = urlparse(base_url)
                if _parsed_url.query:
                    _clean_url = urlunparse(_parsed_url._replace(query=""))
                    _query_params = {
                        k: v[0] for k, v in parse_qs(_parsed_url.query).items()
                    }
                    client_kwargs = {
                        "api_key": api_key,
                        "base_url": _clean_url,
                        "default_query": _query_params,
                    }
                else:
                    client_kwargs = {"api_key": api_key, "base_url": base_url}
                if _provider_timeout is not None:
                    client_kwargs["timeout"] = _provider_timeout
                if self.provider == "copilot-acp":
                    client_kwargs["command"] = self.acp_command
                    client_kwargs["args"] = self.acp_args
                effective_base = base_url
                if base_url_host_matches(effective_base, "openrouter.ai"):
                    from agent.auxiliary_client import build_or_headers
                    client_kwargs["default_headers"] = build_or_headers()
                elif base_url_host_matches(effective_base, "api.routermint.com"):
                    client_kwargs["default_headers"] = _routermint_headers()
                elif base_url_host_matches(effective_base, "api.githubcopilot.com"):
                    from hermes_cli.models import copilot_default_headers

                    client_kwargs["default_headers"] = copilot_default_headers()
                elif base_url_host_matches(effective_base, "api.kimi.com"):
                    client_kwargs["default_headers"] = {
                        "User-Agent": "claude-code/0.1.0",
                    }
                elif base_url_host_matches(effective_base, "portal.qwen.ai"):
                    client_kwargs["default_headers"] = _qwen_portal_headers()
                elif base_url_host_matches(effective_base, "chatgpt.com"):
                    from agent.auxiliary_client import _codex_cloudflare_headers
                    client_kwargs["default_headers"] = _codex_cloudflare_headers(api_key)
                elif "default_headers" not in client_kwargs:
                    # Fall back to profile.default_headers for providers that
                    # declare custom headers (e.g. Vercel AI Gateway attribution,
                    # Kimi User-Agent on non-kimi.com endpoints).
                    try:
                        from providers import get_provider_profile as _gpf
                        _ph = _gpf(self.provider)
                        if _ph and _ph.default_headers:
                            client_kwargs["default_headers"] = dict(_ph.default_headers)
                    except Exception:
                        pass
            else:
                # No explicit creds — use the centralized provider router
                from agent.auxiliary_client import resolve_provider_client
                _routed_client, _ = resolve_provider_client(
                    self.provider or "auto", model=self.model, raw_codex=True)
                if _routed_client is not None:
                    client_kwargs = {
                        "api_key": _routed_client.api_key,
                        "base_url": str(_routed_client.base_url),
                    }
                    if _provider_timeout is not None:
                        client_kwargs["timeout"] = _provider_timeout
                    # Preserve any default_headers the router set
                    if hasattr(_routed_client, '_default_headers') and _routed_client._default_headers:
                        client_kwargs["default_headers"] = dict(_routed_client._default_headers)
                else:
                    # When the user explicitly chose a non-OpenRouter provider
                    # but no credentials were found, fail fast with a clear
                    # message instead of silently routing through OpenRouter.
                    _explicit = (self.provider or "").strip().lower()
                    if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                        # Look up the actual env var name from the provider
                        # config — some providers use non-standard names
                        # (e.g. alibaba → DASHSCOPE_API_KEY, not ALIBABA_API_KEY).
                        _env_hint = f"{_explicit.upper()}_API_KEY"
                        try:
                            from hermes_cli.auth import PROVIDER_REGISTRY
                            _pcfg = PROVIDER_REGISTRY.get(_explicit)
                            if _pcfg and _pcfg.api_key_env_vars:
                                _env_hint = _pcfg.api_key_env_vars[0]
                        except Exception:
                            pass
                        # --- Init-time fallback (#17929) ---
                        _fb_entries = []
                        if isinstance(fallback_model, list):
                            _fb_entries = [
                                f for f in fallback_model
                                if isinstance(f, dict) and f.get("provider") and f.get("model")
                            ]
                        elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
                            _fb_entries = [fallback_model]
                        _fb_resolved = False
                        for _fb in _fb_entries:
                            _fb_explicit_key = (_fb.get("api_key") or "").strip() or None
                            if not _fb_explicit_key:
                                _fb_key_env = (_fb.get("key_env") or _fb.get("api_key_env") or "").strip()
                                if _fb_key_env:
                                    _fb_explicit_key = os.getenv(_fb_key_env, "").strip() or None
                            _fb_client, _fb_model = resolve_provider_client(
                                _fb["provider"], model=_fb["model"], raw_codex=True,
                                explicit_base_url=_fb.get("base_url"),
                                explicit_api_key=_fb_explicit_key,
                            )
                            if _fb_client is not None:
                                self.provider = _fb["provider"]
                                self.model = _fb_model or _fb["model"]
                                self._fallback_activated = True
                                client_kwargs = {
                                    "api_key": _fb_client.api_key,
                                    "base_url": str(_fb_client.base_url),
                                }
                                if _provider_timeout is not None:
                                    client_kwargs["timeout"] = _provider_timeout
                                if hasattr(_fb_client, "_default_headers") and _fb_client._default_headers:
                                    client_kwargs["default_headers"] = dict(_fb_client._default_headers)
                                _fb_resolved = True
                                break
                        if not _fb_resolved:
                            raise RuntimeError(
                                f"Provider '{_explicit}' is set in config.yaml but no API key "
                                f"was found. Set the {_env_hint} environment "
                                f"variable, or switch to a different provider with `hermes model`."
                            )
                    if not getattr(self, "_fallback_activated", False):
                        # No provider configured — reject with a clear message.
                        raise RuntimeError(
                            "No LLM provider configured. Run `hermes model` to "
                            "select a provider, or run `hermes setup` for first-time "
                            "configuration."
                        )
            
            self._client_kwargs = client_kwargs  # stored for rebuilding after interrupt

            # Enable fine-grained tool streaming for Claude on OpenRouter.
            # Without this, Anthropic buffers the entire tool call and goes
            # silent for minutes while thinking — OpenRouter's upstream proxy
            # times out during the silence.  The beta header makes Anthropic
            # stream tool call arguments token-by-token, keeping the
            # connection alive.
            _effective_base = str(client_kwargs.get("base_url", "")).lower()
            if base_url_host_matches(_effective_base, "openrouter.ai") and "claude" in (self.model or "").lower():
                headers = client_kwargs.get("default_headers") or {}
                existing_beta = headers.get("x-anthropic-beta", "")
                _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
                if _FINE_GRAINED not in existing_beta:
                    if existing_beta:
                        headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                    else:
                        headers["x-anthropic-beta"] = _FINE_GRAINED
                    client_kwargs["default_headers"] = headers

            self.api_key = client_kwargs.get("api_key", "")
            self.base_url = client_kwargs.get("base_url", self.base_url)
            try:
                self.client = self._create_openai_client(client_kwargs, reason="agent_init", shared=True)
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model}")
                    if base_url:
                        print(f"🔗 Using custom base URL: {base_url}")
                    # Always show API key info (masked) for debugging auth issues
                    key_used = client_kwargs.get("api_key", "none")
                    if key_used and key_used != "dummy-key" and len(key_used) > 12:
                        print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                    else:
                        print(f"⚠️  Warning: API key appears invalid or missing (got: '{key_used[:20] if key_used else 'none'}...')")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        # Provider fallback chain — ordered list of backup providers tried
        # when the primary is exhausted (rate-limit, overload, connection
        # failure).  Supports both legacy single-dict ``fallback_model`` and
        # new list ``fallback_providers`` format.
        if isinstance(fallback_model, list):
            self._fallback_chain = [
                f for f in fallback_model
                if isinstance(f, dict) and f.get("provider") and f.get("model")
            ]
        elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
            self._fallback_chain = [fallback_model]
        else:
            self._fallback_chain = []
        self._fallback_index = 0
        self._fallback_activated = getattr(self, "_fallback_activated", False)
        # Legacy attribute kept for backward compat (tests, external callers)
        self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None
        if self._fallback_chain and not self.quiet_mode:
            if len(self._fallback_chain) == 1:
                fb = self._fallback_chain[0]
                print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
            else:
                print(f"🔄 Fallback chain ({len(self._fallback_chain)} providers): " +
                      " → ".join(f"{f['model']} ({f['provider']})" for f in self._fallback_chain))

        # Get available tools with filtering
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=self.quiet_mode,
        )
        
        # Show tool configuration and store valid tool names for validation
        self.valid_tool_names = set()
        if self.tools:
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
            if not self.quiet_mode:
                print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")
                
                # Show filtering info if applied
                if enabled_toolsets:
                    print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
                if disabled_toolsets:
                    print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        elif not self.quiet_mode:
            print("🛠️  No tools loaded (all tools filtered out or unavailable)")
        
        # Check tool requirements
        if self.tools and not self.quiet_mode:
            requirements = check_toolset_requirements()
            missing_reqs = [name for name, available in requirements.items() if not available]
            if missing_reqs:
                print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")
        
        # Show trajectory saving status
        if self.save_trajectories and not self.quiet_mode:
            print("📝 Trajectory saving enabled")
        
        # Show ephemeral system prompt status
        if self.ephemeral_system_prompt and not self.quiet_mode:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")
        
        # Show prompt caching status
        if self._use_prompt_caching and not self.quiet_mode:
            if self._use_native_cache_layout and self.provider == "anthropic":
                source = "native Anthropic"
            elif self._use_native_cache_layout:
                source = "Anthropic-compatible endpoint"
            else:
                source = "Claude via OpenRouter"
            print(f"💾 Prompt caching: ENABLED ({source}, {self._cache_ttl} TTL)")
        
        # Session logging setup - auto-save conversation trajectories for debugging
        self.session_start = datetime.now()
        if session_id:
            # Use provided session ID (e.g., from CLI)
            self.session_id = session_id
        else:
            # Generate a new session ID
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"

        # Expose session ID to tools (terminal, execute_code) so agents can
        # reference their own session for --resume commands, cross-session
        # coordination, and logging.  Uses the ContextVar system from
        # session_context.py for concurrency safety (gateway runs multiple
        # sessions in one process).  Also writes os.environ as fallback for
        # CLI mode where ContextVars aren't used.
        os.environ["HERMES_SESSION_ID"] = self.session_id
        try:
            from gateway.session_context import _SESSION_ID
            _SESSION_ID.set(self.session_id)
        except Exception:
            pass  # CLI/test mode — ContextVar not needed

        # Session logs go into ~/.hermes/sessions/ alongside gateway sessions
        hermes_home = get_hermes_home()
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
        
        # Track conversation messages for session logging
        self._session_messages: List[Dict[str, Any]] = []
        self._memory_write_origin = "assistant_tool"
        self._memory_write_context = "foreground"
        
        # Cached system prompt -- built once per session, only rebuilt on compression
        self._cached_system_prompt: Optional[str] = None
        
        # Filesystem checkpoint manager (transparent — not a tool)
        from tools.checkpoint_manager import CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
            max_total_size_mb=checkpoint_max_total_size_mb,
            max_file_size_mb=checkpoint_max_file_size_mb,
        )
        
        # SQLite session store (optional -- provided by CLI or gateway)
        self._session_db = session_db
        self._parent_session_id = parent_session_id
        self._last_flushed_db_idx = 0  # tracks DB-write cursor to prevent duplicate writes
        self._session_db_created = False  # DB row deferred to run_conversation()
        self._session_init_model_config = {
            "max_iterations": self.max_iterations,
            "reasoning_config": reasoning_config,
            "max_tokens": max_tokens,
        }
        
        # In-memory todo list for task planning (one per agent/session)
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()
        
        # Load config once for memory, skills, and compression sections
        try:
            from hermes_cli.config import load_config as _load_agent_config
            _agent_cfg = _load_agent_config()
        except Exception:
            _agent_cfg = {}
        try:
            self._tool_guardrails = ToolCallGuardrailController(
                ToolCallGuardrailConfig.from_mapping(
                    _agent_cfg.get("tool_loop_guardrails", {})
                )
            )
        except Exception as _tlg_err:
            logger.warning("Tool loop guardrail config ignored: %s", _tlg_err)
        # Cache only the derived auxiliary compression context override that is
        # needed later by the startup feasibility check.  Avoid exposing a
        # broad pseudo-public config object on the agent instance.
        self._aux_compression_context_length_config = None

        # Persistent memory (MEMORY.md + USER.md) -- loaded from disk
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 10
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        if not skip_memory:
            try:
                mem_config = _agent_cfg.get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass  # Memory is optional -- don't break agent init
        


        # Memory provider plugin (external — one at a time, alongside built-in)
        # Reads memory.provider from config to select which plugin to activate.
        self._memory_manager = None
        if not skip_memory:
            try:
                _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

                if _mem_provider_name:
                    from agent.memory_manager import MemoryManager as _MemoryManager
                    from plugins.memory import load_memory_provider as _load_mem
                    self._memory_manager = _MemoryManager()
                    _mp = _load_mem(_mem_provider_name)
                    if _mp and _mp.is_available():
                        self._memory_manager.add_provider(_mp)
                    if self._memory_manager.providers:
                        _init_kwargs = {
                            "session_id": self.session_id,
                            "platform": platform or "cli",
                            "hermes_home": str(get_hermes_home()),
                            "agent_context": "primary",
                        }
                        # Thread session title for memory provider scoping
                        # (e.g. honcho uses this to derive chat-scoped session keys)
                        if self._session_db:
                            try:
                                _st = self._session_db.get_session_title(self.session_id)
                                if _st:
                                    _init_kwargs["session_title"] = _st
                            except Exception:
                                pass
                        # Thread gateway user identity for per-user memory scoping
                        if self._user_id:
                            _init_kwargs["user_id"] = self._user_id
                        if self._user_name:
                            _init_kwargs["user_name"] = self._user_name
                        if self._chat_id:
                            _init_kwargs["chat_id"] = self._chat_id
                        if self._chat_name:
                            _init_kwargs["chat_name"] = self._chat_name
                        if self._chat_type:
                            _init_kwargs["chat_type"] = self._chat_type
                        if self._thread_id:
                            _init_kwargs["thread_id"] = self._thread_id
                        # Thread gateway session key for stable per-chat Honcho session isolation
                        if self._gateway_session_key:
                            _init_kwargs["gateway_session_key"] = self._gateway_session_key
                        # Profile identity for per-profile provider scoping
                        try:
                            from hermes_cli.profiles import get_active_profile_name
                            _profile = get_active_profile_name()
                            _init_kwargs["agent_identity"] = _profile
                            _init_kwargs["agent_workspace"] = "hermes"
                        except Exception:
                            pass
                        self._memory_manager.initialize_all(**_init_kwargs)
                        logger.info("Memory provider '%s' activated", _mem_provider_name)
                    else:
                        logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                        self._memory_manager = None
            except Exception as _mpe:
                logger.warning("Memory provider plugin init failed: %s", _mpe)
                self._memory_manager = None

        # Inject memory provider tool schemas into the tool surface.
        # Skip tools whose names already exist (plugins may register the
        # same tools via ctx.register_tool(), which lands in self.tools
        # through get_tool_definitions()).  Duplicate function names cause
        # 400 errors on providers that enforce unique names (e.g. Xiaomi
        # MiMo via Nous Portal).
        if self._memory_manager and self.tools is not None:
            _existing_tool_names = {
                t.get("function", {}).get("name")
                for t in self.tools
                if isinstance(t, dict)
            }
            for _schema in self._memory_manager.get_all_tool_schemas():
                _tname = _schema.get("name", "")
                if _tname and _tname in _existing_tool_names:
                    continue  # already registered via plugin path
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                if _tname:
                    self.valid_tool_names.add(_tname)
                    _existing_tool_names.add(_tname)

        # Skills config: nudge interval for skill creation reminders
        self._skill_nudge_interval = 10
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
        except Exception:
            pass

        # Tool-use enforcement config: "auto" (default — matches hardcoded
        # model list), true (always), false (never), or list of substrings.
        _agent_section = _agent_cfg.get("agent", {})
        if not isinstance(_agent_section, dict):
            _agent_section = {}
        self._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

        # App-level API retry count (wraps each model API call).  Default 3,
        # overridable via agent.api_max_retries in config.yaml.  See #11616.
        try:
            _raw_api_retries = _agent_section.get("api_max_retries", 3)
            _api_retries = int(_raw_api_retries)
            _api_retries = max(_api_retries, 1)  # 1 = no retry (single attempt)
        except (TypeError, ValueError):
            _api_retries = 3
        self._api_max_retries = _api_retries

        # Initialize context compressor for automatic context management
        # Compresses conversation when approaching model's context limit
        # Configuration via config.yaml (compression section)
        _compression_cfg = _agent_cfg.get("compression", {})
        if not isinstance(_compression_cfg, dict):
            _compression_cfg = {}
        compression_threshold = float(_compression_cfg.get("threshold", 0.50))
        try:
            from agent.auxiliary_client import _compression_threshold_for_model as _cthresh_fn
            _model_cthresh = _cthresh_fn(self.model)
            if _model_cthresh is not None:
                compression_threshold = _model_cthresh
        except Exception:
            pass
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in {"true", "1", "yes"}
        compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))
        compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))

        # Read optional explicit context_length override for the auxiliary
        # compression model. Custom endpoints often cannot report this via
        # /models, so the startup feasibility check needs the config hint.
        try:
            _aux_cfg = cfg_get(_agent_cfg, "auxiliary", "compression", default={})
        except Exception:
            _aux_cfg = {}
        if isinstance(_aux_cfg, dict):
            _aux_context_config = _aux_cfg.get("context_length")
        else:
            _aux_context_config = None
        if _aux_context_config is not None:
            try:
                _aux_context_config = int(_aux_context_config)
            except (TypeError, ValueError):
                _aux_context_config = None
        self._aux_compression_context_length_config = _aux_context_config

        # Read explicit model output-token override from config when the
        # caller did not pass one directly.
        _model_cfg = _agent_cfg.get("model", {})
        if self.max_tokens is None and isinstance(_model_cfg, dict):
            _config_max_tokens = _model_cfg.get("max_tokens")
            if _config_max_tokens is not None:
                try:
                    if isinstance(_config_max_tokens, bool):
                        raise ValueError
                    _parsed_max_tokens = int(_config_max_tokens)
                    if _parsed_max_tokens <= 0:
                        raise ValueError
                    self.max_tokens = _parsed_max_tokens
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid model.max_tokens in config.yaml: %r — "
                        "must be a positive integer (e.g. 4096). "
                        "Falling back to provider default.",
                        _config_max_tokens,
                    )
                    print(
                        f"\n⚠ Invalid model.max_tokens in config.yaml: {_config_max_tokens!r}\n"
                        f"  Must be a positive integer (e.g. 4096).\n"
                        f"  Falling back to provider default.\n",
                        file=sys.stderr,
                    )
        self._session_init_model_config["max_tokens"] = self.max_tokens

        # Read explicit context_length override from model config
        if isinstance(_model_cfg, dict):
            _config_context_length = _model_cfg.get("context_length")
        else:
            _config_context_length = None
        if _config_context_length is not None:
            try:
                _config_context_length = int(_config_context_length)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid model.context_length in config.yaml: %r — "
                    "must be a plain integer (e.g. 256000, not '256K'). "
                    "Falling back to auto-detection.",
                    _config_context_length,
                )
                print(
                    f"\n⚠ Invalid model.context_length in config.yaml: {_config_context_length!r}\n"
                    f"  Must be a plain integer (e.g. 256000, not '256K').\n"
                    f"  Falling back to auto-detected context window.\n",
                    file=sys.stderr,
                )
                _config_context_length = None

        # Resolve custom_providers list once for reuse below (startup
        # context-length override and plugin context-engine init).
        try:
            from hermes_cli.config import get_compatible_custom_providers
            _custom_providers = get_compatible_custom_providers(_agent_cfg)
        except Exception:
            _custom_providers = _agent_cfg.get("custom_providers")
            if not isinstance(_custom_providers, list):
                _custom_providers = []

        # Check custom_providers per-model context_length
        if _config_context_length is None and _custom_providers:
            try:
                from hermes_cli.config import get_custom_provider_context_length
                _cp_ctx_resolved = get_custom_provider_context_length(
                    model=self.model,
                    base_url=self.base_url,
                    custom_providers=_custom_providers,
                )
                if _cp_ctx_resolved:
                    _config_context_length = int(_cp_ctx_resolved)
            except Exception:
                _cp_ctx_resolved = None

            # Surface a clear warning if the user set a context_length but it
            # wasn't a valid positive int — the helper silently skips those.
            if _config_context_length is None:
                _target = self.base_url.rstrip("/") if self.base_url else ""
                for _cp_entry in _custom_providers:
                    if not isinstance(_cp_entry, dict):
                        continue
                    _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                    if _target and _cp_url == _target:
                        _cp_models = _cp_entry.get("models", {})
                        if isinstance(_cp_models, dict):
                            _cp_model_cfg = _cp_models.get(self.model, {})
                            if isinstance(_cp_model_cfg, dict):
                                _cp_ctx = _cp_model_cfg.get("context_length")
                                if _cp_ctx is not None:
                                    try:
                                        _parsed = int(_cp_ctx)
                                        if _parsed <= 0:
                                            raise ValueError
                                    except (TypeError, ValueError):
                                        logger.warning(
                                            "Invalid context_length for model %r in "
                                            "custom_providers: %r — must be a positive "
                                            "integer (e.g. 256000, not '256K'). "
                                            "Falling back to auto-detection.",
                                            self.model, _cp_ctx,
                                        )
                                        print(
                                            f"\n⚠ Invalid context_length for model {self.model!r} in custom_providers: {_cp_ctx!r}\n"
                                            f"  Must be a positive integer (e.g. 256000, not '256K').\n"
                                            f"  Falling back to auto-detected context window.\n",
                                            file=sys.stderr,
                                        )
                        break

        # Persist for reuse on switch_model / fallback activation. Must come
        # AFTER the custom_providers branch so per-model overrides aren't lost.
        self._config_context_length = _config_context_length

        self._ensure_lmstudio_runtime_loaded(_config_context_length)



        # Select context engine: config-driven (like memory providers).
        # 1. Check config.yaml context.engine setting
        # 2. Check plugins/context_engine/<name>/ directory (repo-shipped)
        # 3. Check general plugin system (user-installed plugins)
        # 4. Fall back to built-in ContextCompressor
        _selected_engine = None
        _engine_name = "compressor"  # default
        try:
            _ctx_cfg = _agent_cfg.get("context", {}) if isinstance(_agent_cfg, dict) else {}
            _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
        except Exception:
            pass

        if _engine_name != "compressor":
            # Try loading from plugins/context_engine/<name>/
            try:
                from plugins.context_engine import load_context_engine
                _selected_engine = load_context_engine(_engine_name)
            except Exception as _ce_load_err:
                logger.debug("Context engine load from plugins/context_engine/: %s", _ce_load_err)

            # Try general plugin system as fallback
            if _selected_engine is None:
                try:
                    from hermes_cli.plugins import get_plugin_context_engine
                    _candidate = get_plugin_context_engine()
                    if _candidate and _candidate.name == _engine_name:
                        _selected_engine = _candidate
                except Exception:
                    pass

            if _selected_engine is None:
                logger.warning(
                    "Context engine '%s' not found — falling back to built-in compressor",
                    _engine_name,
                )
        # else: config says "compressor" — use built-in, don't auto-activate plugins

        if _selected_engine is not None:
            self.context_compressor = _selected_engine
            # Resolve context_length for plugin engines — mirrors switch_model() path
            from agent.model_metadata import get_model_context_length
            _plugin_ctx_len = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
                custom_providers=_custom_providers,
            )
            self.context_compressor.update_model(
                model=self.model,
                context_length=_plugin_ctx_len,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                provider=self.provider,
            )
            if not self.quiet_mode:
                logger.info("Using context engine: %s", _selected_engine.name)
        else:
            self.context_compressor = ContextCompressor(
                model=self.model,
                threshold_percent=compression_threshold,
                protect_first_n=3,
                protect_last_n=compression_protect_last,
                summary_target_ratio=compression_target_ratio,
                summary_model_override=None,
                quiet_mode=self.quiet_mode,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
                api_mode=self.api_mode,
            )
        self.compression_enabled = compression_enabled

        # Reject models whose context window is below the minimum required
        # for reliable tool-calling workflows (64K tokens).
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        _ctx = getattr(self.context_compressor, "context_length", 0)
        if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Model {self.model} has a context window of {_ctx:,} tokens, "
                f"which is below the minimum {MINIMUM_CONTEXT_LENGTH:,} required "
                f"by Hermes Agent.  Choose a model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context, or set "
                f"model.context_length in config.yaml to override."
            )

        # Inject context engine tool schemas (e.g. lcm_grep, lcm_describe, lcm_expand).
        # Skip names that are already present — the get_tool_definitions()
        # quiet_mode cache returned a shared list pre-#17335, so a stray
        # mutation here would poison subsequent agent inits in the same
        # Gateway process and trip provider-side 'duplicate tool name'
        # errors. Even with the cache fix, dedup is the right defense
        # against plugin paths that may register the same schemas via
        # ctx.register_tool(). Mirrors the memory tools dedup above.
        self._context_engine_tool_names: set = set()
        if hasattr(self, "context_compressor") and self.context_compressor and self.tools is not None:
            _existing_tool_names = {
                t.get("function", {}).get("name")
                for t in self.tools
                if isinstance(t, dict)
            }
            for _schema in self.context_compressor.get_tool_schemas():
                _tname = _schema.get("name", "")
                if _tname and _tname in _existing_tool_names:
                    continue  # already registered via plugin/cache path
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                if _tname:
                    self.valid_tool_names.add(_tname)
                    self._context_engine_tool_names.add(_tname)
                    _existing_tool_names.add(_tname)

        # Notify context engine of session start
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_start(
                    self.session_id,
                    hermes_home=str(get_hermes_home()),
                    platform=self.platform or "cli",
                    model=self.model,
                    context_length=getattr(self.context_compressor, "context_length", 0),
                )
            except Exception as _ce_err:
                logger.debug("Context engine on_session_start: %s", _ce_err)

        self._subdirectory_hints = SubdirectoryHintTracker(
            working_dir=os.getenv("TERMINAL_CWD") or None,
        )
        self._user_turn_count = 0

        # Cumulative token usage for the session
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # ── Ollama num_ctx injection ──
        # Ollama defaults to 2048 context regardless of the model's capabilities.
        # When running against an Ollama server, detect the model's max context
        # and pass num_ctx on every chat request so the full window is used.
        # User override: set model.ollama_num_ctx in config.yaml to cap VRAM use.
        # If model.context_length is set, it caps num_ctx so the user's VRAM
        # budget is respected even when GGUF metadata advertises a larger window.
        self._ollama_num_ctx: int | None = None
        _ollama_num_ctx_override = None
        if isinstance(_model_cfg, dict):
            _ollama_num_ctx_override = _model_cfg.get("ollama_num_ctx")
        if _ollama_num_ctx_override is not None:
            try:
                self._ollama_num_ctx = int(_ollama_num_ctx_override)
            except (TypeError, ValueError):
                logger.debug("Invalid ollama_num_ctx config value: %r", _ollama_num_ctx_override)
        if self._ollama_num_ctx is None and self.base_url and is_local_endpoint(self.base_url):
            try:
                _detected = query_ollama_num_ctx(self.model, self.base_url, api_key=self.api_key or "")
                if _detected and _detected > 0:
                    self._ollama_num_ctx = _detected
            except Exception as exc:
                logger.debug("Ollama num_ctx detection failed: %s", exc)
        # Cap auto-detected ollama_num_ctx to the user's explicit context_length.
        # Without this, GGUF metadata can advertise 256K+ which Ollama honours
        # by allocating that much VRAM — blowing up small GPUs even though the
        # user explicitly set a smaller context_length in config.yaml.
        if (
            self._ollama_num_ctx
            and _config_context_length
            and _ollama_num_ctx_override is None  # don't override explicit ollama_num_ctx
            and self._ollama_num_ctx > _config_context_length
        ):
            logger.info(
                "Ollama num_ctx capped: %d -> %d (model.context_length override)",
                self._ollama_num_ctx, _config_context_length,
            )
            self._ollama_num_ctx = _config_context_length
        if self._ollama_num_ctx and not self.quiet_mode:
            logger.info(
                "Ollama num_ctx: will request %d tokens (model max from /api/show)",
                self._ollama_num_ctx,
            )

        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")

        # Check immediately so CLI users see the warning at startup.
        # Gateway status_callback is not yet wired, so any warning is stored
        # in _compression_warning and replayed in the first run_conversation().
        self._compression_warning = None
        self._check_compression_model_feasibility()

        # Snapshot primary runtime for per-turn restoration.  When fallback
        # activates during a turn, the next turn restores these values so the
        # preferred model gets a fresh attempt each time.  Uses a single dict
        # so new state fields are easy to add without N individual attributes.
        _cc = self.context_compressor
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "use_native_cache_layout": self._use_native_cache_layout,
            # Context engine state that _try_activate_fallback() overwrites.
            # Use getattr for model/base_url/api_key/provider since plugin
            # engines may not have these (they're ContextCompressor-specific).
            "compressor_model": getattr(_cc, "model", self.model),
            "compressor_base_url": getattr(_cc, "base_url", self.base_url),
            "compressor_api_key": getattr(_cc, "api_key", ""),
            "compressor_provider": getattr(_cc, "provider", self.provider),
            "compressor_context_length": _cc.context_length,
            "compressor_threshold_tokens": _cc.threshold_tokens,
        }
        if self.api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

    def _get_session_db_for_recall(self):
        """Return a SessionDB for recall, lazily creating it if an entrypoint forgot.

        Most frontends pass ``session_db`` into ``AIAgent`` explicitly, but recall
        is important enough that a missing constructor argument should degrade by
        opening the default state DB instead of making the advertised
        ``session_search`` tool unusable.
        """
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state import SessionDB

            self._session_db = SessionDB()
            return self._session_db
        except Exception as exc:
            logger.debug("SessionDB unavailable for recall", exc_info=True)
            return None

    def _ensure_db_session(self) -> None:
        """Create session DB row on first use. Disables _session_db on failure."""
        if self._session_db_created or not self._session_db:
            return
        try:
            self._session_db.create_session(
                session_id=self.session_id,
                source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                model=self.model,
                model_config=self._session_init_model_config,
                system_prompt=self._cached_system_prompt,
                user_id=None,
                parent_session_id=self._parent_session_id,
            )
            self._session_db_created = True
        except Exception as e:
            # Transient failure (e.g. SQLite lock). Keep _session_db alive —
            # _session_db_created stays False so next run_conversation() retries.
            logger.warning(
                "Session DB creation failed (will retry next turn): %s", e
            )

    def reset_session_state(self):
        """Reset all session-scoped token counters to 0 for a fresh session.
        
        This method encapsulates the reset logic for all session-level metrics
        including:
        - Token usage counters (input, output, total, prompt, completion)
        - Cache read/write tokens
        - API call count
        - Reasoning tokens
        - Estimated cost tracking
        - Context compressor internal counters
        
        The method safely handles optional attributes (e.g., context compressor)
        using ``hasattr`` checks.
        
        This keeps the counter reset logic DRY and maintainable in one place
        rather than scattering it across multiple methods.
        """
        # Token usage counters
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0

        # Context engine reset (works for both built-in compressor and plugins)
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.on_session_reset()

    def _ensure_lmstudio_runtime_loaded(self, config_context_length: Optional[int] = None) -> None:
        """
        Preload the LM Studio model with at least Hermes' minimum context.
        """
        if (self.provider or "").strip().lower() != "lmstudio":
            return
        try:
            from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
            from hermes_cli.models import ensure_lmstudio_model_loaded
            if config_context_length is None:
                config_context_length = getattr(self, "_config_context_length", None)
            target_ctx = max(config_context_length or 0, MINIMUM_CONTEXT_LENGTH)
            loaded_ctx = ensure_lmstudio_model_loaded(
                self.model, self.base_url, getattr(self, "api_key", ""), target_ctx,
            )
            if loaded_ctx:
                # Push into the live compressor so the status bar reflects the
                # real loaded ctx the moment the load resolves, instead of
                # holding the previous model's value (or "ctx --") through the
                # next render tick.
                cc = getattr(self, "context_compressor", None)
                if cc is not None:
                    cc.update_model(
                        model=self.model,
                        context_length=loaded_ctx,
                        base_url=self.base_url,
                        api_key=getattr(self, "api_key", ""),
                        provider=self.provider,
                        api_mode=self.api_mode,
                    )
        except Exception as err:
            logger.debug("LM Studio preload skipped: %s", err)

    def switch_model(self, new_model, new_provider, api_key='', base_url='', api_mode=''):
        """Switch the model/provider in-place for a live agent.

        Called by the /model command handlers (CLI and gateway) after
        ``model_switch.switch_model()`` has resolved credentials and
        validated the model.  This method performs the actual runtime
        swap: rebuilding clients, updating caching flags, and refreshing
        the context compressor.

        The implementation mirrors ``_try_activate_fallback()`` for the
        client-swap logic but also updates ``_primary_runtime`` so the
        change persists across turns (unlike fallback which is
        turn-scoped).
        """
        from hermes_cli.providers import determine_api_mode

        # ── Determine api_mode if not provided ──
        if not api_mode:
            api_mode = determine_api_mode(new_provider, base_url)

        # Defense-in-depth: ensure OpenCode base_url doesn't carry a trailing
        # /v1 into the anthropic_messages client, which would cause the SDK to
        # hit /v1/v1/messages.  `model_switch.switch_model()` already strips
        # this, but we guard here so any direct callers (future code paths,
        # tests) can't reintroduce the double-/v1 404 bug.
        if (
            api_mode == "anthropic_messages"
            and new_provider in {"opencode-zen", "opencode-go"}
            and isinstance(base_url, str)
            and base_url
        ):
            base_url = re.sub(r"/v1/?$", "", base_url)

        old_model = self.model
        old_provider = self.provider

        # Clear the per-config context_length override so the new model's
        # actual context window is resolved via get_model_context_length()
        # instead of inheriting the stale value from the previous model.
        self._config_context_length = None

        # ── Swap core runtime fields ──
        self.model = new_model
        self.provider = new_provider
        # Use new base_url when provided; only fall back to current when the
        # new provider genuinely has no endpoint (e.g. native SDK providers).
        # Without this guard the old provider's URL (e.g. Ollama's localhost
        # address) would persist silently after switching to a cloud provider
        # that returns an empty base_url string.
        if base_url:
            self.base_url = base_url
        self.api_mode = api_mode
        # Invalidate transport cache — new api_mode may need a different transport
        if hasattr(self, "_transport_cache"):
            self._transport_cache.clear()
        if api_key:
            self.api_key = api_key

        # ── Build new client ──
        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
            # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own
            # API key — falling back would send Anthropic credentials to third-party endpoints.
            _is_native_anthropic = new_provider == "anthropic"
            effective_key = (api_key or self.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or self.api_key or "")
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url or getattr(self, "_anthropic_base_url", None)
            self._anthropic_client = build_anthropic_client(
                effective_key, self._anthropic_base_url,
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
            self._is_anthropic_oauth = _is_oauth_token(effective_key) if _is_native_anthropic else False
            self.client = None
            self._client_kwargs = {}
        else:
            effective_key = api_key or self.api_key
            effective_base = base_url or self.base_url
            self._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            _sm_timeout = get_provider_request_timeout(self.provider, self.model)
            if _sm_timeout is not None:
                self._client_kwargs["timeout"] = _sm_timeout
            self.client = self._create_openai_client(
                dict(self._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        # ── Re-evaluate prompt caching ──
        self._use_prompt_caching, self._use_native_cache_layout = (
            self._anthropic_prompt_cache_policy(
                provider=new_provider,
                base_url=self.base_url,
                api_mode=api_mode,
                model=new_model,
            )
        )

        # ── LM Studio: preload before probing context length ──
        self._ensure_lmstudio_runtime_loaded()

        # ── Update context compressor ──
        if hasattr(self, "context_compressor") and self.context_compressor:
            from agent.model_metadata import get_model_context_length
            # Re-read custom_providers from live config so per-model
            # context_length overrides are honored when switching to a
            # custom provider mid-session (closes #15779).
            _sm_custom_providers = None
            try:
                from hermes_cli.config import load_config, get_compatible_custom_providers
                _sm_cfg = load_config()
                _sm_custom_providers = get_compatible_custom_providers(_sm_cfg)
            except Exception:
                _sm_custom_providers = None
            new_context_length = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                provider=self.provider,
                config_context_length=getattr(self, "_config_context_length", None),
                custom_providers=_sm_custom_providers,
            )
            self.context_compressor.update_model(
                model=self.model,
                context_length=new_context_length,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                provider=self.provider,
                api_mode=self.api_mode,
            )

        # ── Invalidate cached system prompt so it rebuilds next turn ──
        self._cached_system_prompt = None

        # ── Update _primary_runtime so the change persists across turns ──
        _cc = self.context_compressor if hasattr(self, "context_compressor") and self.context_compressor else None
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "use_native_cache_layout": self._use_native_cache_layout,
            "compressor_model": getattr(_cc, "model", self.model) if _cc else self.model,
            "compressor_base_url": getattr(_cc, "base_url", self.base_url) if _cc else self.base_url,
            "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
            "compressor_provider": getattr(_cc, "provider", self.provider) if _cc else self.provider,
            "compressor_context_length": _cc.context_length if _cc else 0,
            "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
        }
        if api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

        # ── Reset fallback state ──
        self._fallback_activated = False
        self._fallback_index = 0

        # When the user deliberately swaps primary providers (e.g. openrouter
        # → anthropic), drop any fallback entries that target the OLD primary
        # or the NEW one.  The chain was seeded from config at agent init for
        # the original provider — without pruning, a failed turn on the new
        # primary silently re-activates the provider the user just rejected,
        # which is exactly what was reported during TUI v2 blitz testing
        # ("switched to anthropic, tui keeps trying openrouter").
        old_norm = (old_provider or "").strip().lower()
        new_norm = (new_provider or "").strip().lower()
        fallback_chain = list(getattr(self, "_fallback_chain", []) or [])
        if old_norm and new_norm and old_norm != new_norm:
            fallback_chain = [
                entry for entry in fallback_chain
                if (entry.get("provider") or "").strip().lower() not in {old_norm, new_norm}
            ]
        self._fallback_chain = fallback_chain
        self._fallback_model = fallback_chain[0] if fallback_chain else None

        logging.info(
            "Model switched in-place: %s (%s) -> %s (%s)",
            old_model, old_provider, new_model, new_provider,
        )

    def _safe_print(self, *args, **kwargs):
        """Print that silently handles broken pipes / closed stdout.

        In headless environments (systemd, Docker, nohup) stdout may become
        unavailable mid-session.  A raw ``print()`` raises ``OSError`` which
        can crash cron jobs and lose completed work.

        Internally routes through ``self._print_fn`` (default: builtin
        ``print``) so callers such as the CLI can inject a renderer that
        handles ANSI escape sequences properly (e.g. prompt_toolkit's
        ``print_formatted_text(ANSI(...))``) without touching this method.
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """Verbose print — suppressed when actively streaming tokens.

        Pass ``force=True`` for error/warning messages that should always be
        shown even during streaming playback (TTS or display).

        During tool execution (``_executing_tools`` is True), printing is
        allowed even with stream consumers registered because no tokens
        are being streamed at that point.

        After the main response has been delivered and the remaining tool
        calls are post-response housekeeping (``_mute_post_response``),
        all non-forced output is suppressed.

        ``suppress_status_output`` is a stricter CLI automation mode used by
        parseable single-query flows such as ``hermes chat -q``. In that mode,
        all status/diagnostic prints routed through ``_vprint`` are suppressed
        so stdout stays machine-readable.
        """
        if getattr(self, "suppress_status_output", False):
            return
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """Return True when quiet-mode spinner output has a safe sink.

        In headless/stdio-protocol environments, a raw spinner with no custom
        ``_print_fn`` falls back to ``sys.stdout`` and can corrupt protocol
        streams such as ACP JSON-RPC. Allow quiet spinners only when either:
        - output is explicitly rerouted via ``_print_fn``; or
        - stdout is a real TTY.
        """
        if self._print_fn is not None:
            return True
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _should_emit_quiet_tool_messages(self) -> bool:
        """Return True when quiet-mode tool summaries should print directly.

        Quiet mode is used by both the interactive CLI and embedded/library
        callers. The CLI may still want compact progress hints when no callback
        owns rendering. Embedded/library callers, on the other hand, expect
        quiet mode to be truly silent.
        """
        return (
            self.quiet_mode
            and not self.tool_progress_callback
            and getattr(self, "platform", "") == "cli"
        )

    def _emit_status(self, message: str) -> None:
        """Emit a lifecycle status message to both CLI and gateway channels.

        CLI users see the message via ``_vprint(force=True)`` so it is always
        visible regardless of verbose/quiet mode.  Gateway consumers receive
        it through ``status_callback("lifecycle", ...)``.

        This helper never raises — exceptions are swallowed so it cannot
        interrupt the retry/fallback logic.
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("lifecycle", message)
            except Exception:
                logger.debug("status_callback error in _emit_status", exc_info=True)

    def _emit_warning(self, message: str) -> None:
        """Emit a user-visible warning through the same status plumbing.

        Unlike debug logs, these warnings are meant for degraded side paths
        such as auxiliary compression or memory flushes where the main turn can
        continue but the user needs to know something important failed.
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("warn", message)
            except Exception:
                logger.debug("status_callback error in _emit_warning", exc_info=True)

    # Headers we capture from the dying stream's HTTP response so post-mortem
    # diagnosis can answer "which CF edge / which OpenRouter downstream
    # provider / which request id".  Lowercased; httpx returns CIMultiDict.
    _STREAM_DIAG_HEADERS = (
        "cf-ray",
        "cf-cache-status",
        "x-openrouter-provider",
        "x-openrouter-model",
        "x-openrouter-id",
        "x-request-id",
        "x-vercel-id",
        "via",
        "server",
        "x-forwarded-for",
    )

    @staticmethod
    def _stream_diag_init() -> Dict[str, Any]:
        """Return a fresh per-attempt diagnostic dict.

        Mutated in-place by the streaming functions and read from the retry
        block when a stream dies.  Lives on ``request_client_holder`` so it
        survives across the closure boundary.
        """
        return {
            "started_at": time.time(),
            "first_chunk_at": None,
            "chunks": 0,
            "bytes": 0,
            "headers": {},
            "http_status": None,
        }

    def _stream_diag_capture_response(
        self, diag: Dict[str, Any], http_response: Any
    ) -> None:
        """Snapshot interesting headers + HTTP status from the live stream.

        Called once at stream open (before iterating chunks) so the metadata
        survives even if the stream dies before any chunk arrives.  Failures
        are swallowed — diag is best-effort.
        """
        if http_response is None or not isinstance(diag, dict):
            return
        try:
            diag["http_status"] = getattr(http_response, "status_code", None)
        except Exception:
            pass
        try:
            headers = getattr(http_response, "headers", None) or {}
            captured: Dict[str, str] = {}
            for name in self._STREAM_DIAG_HEADERS:
                try:
                    val = headers.get(name)
                    if val:
                        # Truncate single-value to keep log lines bounded.
                        captured[name] = str(val)[:120]
                except Exception:
                    continue
            diag["headers"] = captured
        except Exception:
            pass

    @staticmethod
    def _flatten_exception_chain(error: BaseException) -> str:
        """Return a compact ``Outer(msg) <- Inner(msg) <- ...`` rendering.

        OpenAI SDK wraps httpx errors as ``APIConnectionError`` /
        ``APIError`` and only the wrapper's class is visible at the catch
        site — but the underlying ``RemoteProtocolError`` /
        ``ConnectError`` / ``ReadError`` is what tells us WHY the stream
        died.  Walks ``__cause__`` then ``__context__`` (deduped, max 4
        deep) to surface the chain in one line.
        """
        seen: List[BaseException] = []
        link: Optional[BaseException] = error
        while link is not None and len(seen) < 4:
            if link in seen:
                break
            seen.append(link)
            nxt = getattr(link, "__cause__", None) or getattr(
                link, "__context__", None
            )
            if nxt is None or nxt is link:
                break
            link = nxt
        parts: List[str] = []
        for e in seen:
            msg = str(e).strip().replace("\n", " ")
            if len(msg) > 140:
                msg = msg[:140] + "…"
            parts.append(f"{type(e).__name__}({msg})" if msg else type(e).__name__)
        return " <- ".join(parts) if parts else type(error).__name__

    def _log_stream_retry(
        self,
        *,
        kind: str,
        error: BaseException,
        attempt: int,
        max_attempts: int,
        mid_tool_call: bool,
        diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a transient stream-drop and retry to ``agent.log``.

        Always logs a structured WARNING so users have a breadcrumb regardless
        of UI verbosity.  Subagents in particular benefit because their
        retries no longer spam the parent's terminal — but the file log keeps
        full detail (provider, error class, attempt, base_url, subagent_id).

        When *diag* is provided (the per-attempt stream-diagnostic dict from
        ``_stream_diag_init``), the WARNING also captures upstream headers
        (cf-ray, x-openrouter-provider, x-openrouter-id), HTTP status, bytes
        streamed before the drop, and elapsed time on the dying attempt.
        These are the breadcrumbs needed to answer "is one CF edge / one
        downstream provider responsible, or is it random across runs?"
        """
        try:
            try:
                _summary = self._summarize_api_error(error)
            except Exception:
                _summary = str(error)
            if _summary and len(_summary) > 240:
                _summary = _summary[:240] + "…"

            # Inner-cause chain (httpx errors hide under openai.APIError).
            try:
                _chain = self._flatten_exception_chain(error)
            except Exception:
                _chain = type(error).__name__

            # Per-attempt counters and upstream headers.
            _now = time.time()
            _bytes = 0
            _chunks = 0
            _elapsed = 0.0
            _ttfb = None
            _headers_repr = "-"
            _http_status = "-"
            if isinstance(diag, dict):
                try:
                    _bytes = int(diag.get("bytes") or 0)
                    _chunks = int(diag.get("chunks") or 0)
                    _started = float(diag.get("started_at") or _now)
                    _elapsed = max(0.0, _now - _started)
                    _first = diag.get("first_chunk_at")
                    if _first is not None:
                        _ttfb = max(0.0, float(_first) - _started)
                    headers = diag.get("headers") or {}
                    if isinstance(headers, dict) and headers:
                        _headers_repr = " ".join(
                            f"{k}={v}" for k, v in headers.items()
                        )
                    if diag.get("http_status") is not None:
                        _http_status = str(diag.get("http_status"))
                except Exception:
                    pass

            logger.warning(
                "Stream %s on attempt %s/%s — retrying. "
                "subagent_id=%s depth=%s provider=%s base_url=%s "
                "error_type=%s error=%s "
                "chain=%s "
                "http_status=%s bytes=%d chunks=%d elapsed=%.2fs ttfb=%s "
                "upstream=[%s]",
                kind,
                attempt,
                max_attempts,
                getattr(self, "_subagent_id", None) or "-",
                getattr(self, "_delegate_depth", 0),
                self.provider or "-",
                self.base_url or "-",
                type(error).__name__,
                _summary,
                _chain,
                _http_status,
                _bytes,
                _chunks,
                _elapsed,
                f"{_ttfb:.2f}s" if _ttfb is not None else "-",
                _headers_repr,
                extra={"mid_tool_call": mid_tool_call},
            )
        except Exception:
            logger.debug("stream-retry log emit failed", exc_info=True)

    def _emit_stream_drop(
        self,
        *,
        error: BaseException,
        attempt: int,
        max_attempts: int,
        mid_tool_call: bool,
        diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a single user-visible line for a stream drop+retry.

        Both top-level agents and subagents announce drops in the UI — the
        parent prefixes subagent lines with ``[subagent-N]`` via ``log_prefix``
        so they're easy to attribute.  All cases also write a structured
        WARNING to ``agent.log`` via :meth:`_log_stream_retry` with the full
        diagnostic detail (subagent_id, provider, base_url, error_type,
        cf-ray, x-openrouter-provider, bytes/chunks, elapsed) for post-hoc
        analysis.

        The user-visible status line is intentionally compact: provider,
        error class, attempt N/M, plus ``after Xs`` when the stream dropped
        mid-flight.  Full diagnostic detail goes to ``agent.log`` only —
        ``hermes logs --level WARNING | grep "Stream drop"`` to inspect.
        """
        kind = "drop mid tool-call" if mid_tool_call else "drop"
        self._log_stream_retry(
            kind=kind,
            error=error,
            attempt=attempt,
            max_attempts=max_attempts,
            mid_tool_call=mid_tool_call,
            diag=diag,
        )
        provider = self.provider or "provider"
        # Compose a brief "after Xs" suffix when we have timing data — helps
        # the user distinguish "couldn't connect" (0s) from "died after 30s
        # of streaming" (likely upstream idle-kill or proxy timeout).
        _suffix = ""
        if isinstance(diag, dict):
            try:
                started = diag.get("started_at")
                if started is not None:
                    _suffix = f" after {max(0.0, time.time() - float(started)):.1f}s"
            except Exception:
                pass
        try:
            self._emit_status(
                f"⚠️ {provider} stream {kind} ({type(error).__name__}){_suffix} "
                f"— reconnecting, retry {attempt}/{max_attempts}"
            )
            self._touch_activity(
                f"stream retry {attempt}/{max_attempts} "
                f"after {type(error).__name__}"
            )
        except Exception:
            pass

    def _emit_auxiliary_failure(self, task: str, exc: BaseException) -> None:
        """Surface a compact warning for failed auxiliary work."""
        try:
            detail = self._summarize_api_error(exc)
        except Exception:
            detail = str(exc)
        detail = (detail or exc.__class__.__name__).strip()
        if len(detail) > 220:
            detail = detail[:217].rstrip() + "..."
        self._emit_warning(f"⚠ Auxiliary {task} failed: {detail}")

    def _current_main_runtime(self) -> Dict[str, str]:
        """Return the live main runtime for session-scoped auxiliary routing."""
        return {
            "model": getattr(self, "model", "") or "",
            "provider": getattr(self, "provider", "") or "",
            "base_url": getattr(self, "base_url", "") or "",
            "api_key": getattr(self, "api_key", "") or "",
            "api_mode": getattr(self, "api_mode", "") or "",
        }

    def _check_compression_model_feasibility(self) -> None:
        """Warn at session start if the auxiliary compression model's context
        window is smaller than the main model's compression threshold.

        When the auxiliary model cannot fit the content that needs summarising,
        compression will either fail outright (the LLM call errors) or produce
        a severely truncated summary.

        Called during ``__init__`` so CLI users see the warning immediately
        (via ``_vprint``).  The gateway sets ``status_callback`` *after*
        construction, so ``_replay_compression_warning()`` re-sends the
        stored warning through the callback on the first
        ``run_conversation()`` call.
        """
        if not self.compression_enabled:
            return
        try:
            from agent.auxiliary_client import (
                _resolve_task_provider_model,
                get_text_auxiliary_client,
            )
            from agent.model_metadata import (
                MINIMUM_CONTEXT_LENGTH,
                get_model_context_length,
            )

            client, aux_model = get_text_auxiliary_client(
                "compression",
                main_runtime=self._current_main_runtime(),
            )
            # Best-effort aux provider label for the warning message. The
            # configured provider may be "auto", in which case we fall back
            # to the client's base_url hostname so the user can still tell
            # where the compression model is actually being called.
            try:
                _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
            except Exception:
                _aux_cfg_provider = ""
            if client is None or not aux_model:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "No auxiliary LLM provider for compression — "
                    "summaries will be unavailable."
                )
                return

            aux_base_url = str(getattr(client, "base_url", ""))
            aux_api_key = str(getattr(client, "api_key", ""))

            aux_context = get_model_context_length(
                aux_model,
                base_url=aux_base_url,
                api_key=aux_api_key,
                config_context_length=getattr(self, "_aux_compression_context_length_config", None),
                # Each model must be resolved with its own provider so that
                # provider-specific paths (e.g. Bedrock static table, OpenRouter API)
                # are invoked for the correct client, not inherited from the main model.
                provider=(_aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(self, "provider", "")),
            )

            # Hard floor: the auxiliary compression model must have at least
            # MINIMUM_CONTEXT_LENGTH (64K) tokens of context.  The main model
            # is already required to meet this floor (checked earlier in
            # __init__), so the compression model must too — otherwise it
            # cannot summarise a full threshold-sized window of main-model
            # content.  Mirrors the main-model rejection pattern.
            if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
                raise ValueError(
                    f"Auxiliary compression model {aux_model} has a context "
                    f"window of {aux_context:,} tokens, which is below the "
                    f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                    f"Agent.  Choose a compression model with at least "
                    f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                    f"auxiliary.compression.model in config.yaml), or set "
                    f"auxiliary.compression.context_length to override the "
                    f"detected value if it is wrong."
                )

            threshold = self.context_compressor.threshold_tokens
            if aux_context < threshold:
                # Auto-correct: lower the live session threshold so
                # compression actually works this session.  The hard floor
                # above guarantees aux_context >= MINIMUM_CONTEXT_LENGTH,
                # so the new threshold is always >= 64K.
                #
                # The compression summariser sends a single user-role
                # prompt (no system prompt, no tools) to the aux model, so
                # new_threshold == aux_context is safe: the request is
                # the raw messages plus a small summarisation instruction.
                old_threshold = threshold
                new_threshold = aux_context
                self.context_compressor.threshold_tokens = new_threshold
                # Keep threshold_percent in sync so future main-model
                # context_length changes (update_model) re-derive from a
                # sensible number rather than the original too-high value.
                main_ctx = self.context_compressor.context_length
                if main_ctx:
                    self.context_compressor.threshold_percent = (
                        new_threshold / main_ctx
                    )
                safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
                # Build human-readable "model (provider)" labels for both
                # the main model and the compression model so users can
                # tell at a glance which provider each side is actually
                # using. When the configured provider is empty or "auto",
                # fall back to the client's base_url hostname.
                _main_model = getattr(self, "model", "") or "?"
                _main_provider = getattr(self, "provider", "") or ""
                _aux_provider_label = (
                    _aux_cfg_provider
                    if _aux_cfg_provider and _aux_cfg_provider != "auto"
                    else ""
                )
                if not _aux_provider_label:
                    try:
                        from urllib.parse import urlparse
                        _aux_provider_label = (
                            urlparse(aux_base_url).hostname or aux_base_url
                        )
                    except Exception:
                        _aux_provider_label = aux_base_url or "auto"
                _main_label = (
                    f"{_main_model} ({_main_provider})"
                    if _main_provider
                    else _main_model
                )
                _aux_label = f"{aux_model} ({_aux_provider_label})"
                msg = (
                    f"⚠ Compression model {_aux_label} context is "
                    f"{aux_context:,} tokens, but the main model "
                    f"{_main_label}'s compression threshold was "
                    f"{old_threshold:,} tokens. "
                    f"Auto-lowered this session's threshold to "
                    f"{new_threshold:,} tokens so compression can run.\n"
                    f"  To make this permanent, edit config.yaml — either:\n"
                    f"  1. Use a larger compression model:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{old_threshold:,}+-context>\n"
                    f"  2. Lower the compression threshold:\n"
                    f"       compression:\n"
                    f"         threshold: 0.{safe_pct:02d}"
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "Auxiliary compression model %s has %d token context, "
                    "below the main model's compression threshold of %d "
                    "tokens — auto-lowered session threshold to %d to "
                    "keep compression working.",
                    aux_model,
                    aux_context,
                    old_threshold,
                    new_threshold,
                )
        except ValueError:
            # Hard rejections (aux below minimum context) must propagate
            # so the session refuses to start.
            raise
        except Exception as exc:
            logger.debug(
                "Compression feasibility check failed (non-fatal): %s", exc
            )

    def _replay_compression_warning(self) -> None:
        """Re-send the compression warning through ``status_callback``.

        During ``__init__`` the gateway's ``status_callback`` is not yet
        wired, so ``_emit_status`` only reaches ``_vprint`` (CLI).  This
        method is called once at the start of the first
        ``run_conversation()`` — by then the gateway has set the callback,
        so every platform (Telegram, Discord, Slack, etc.) receives the
        warning.
        """
        msg = getattr(self, "_compression_warning", None)
        if msg and self.status_callback:
            try:
                self.status_callback("lifecycle", msg)
            except Exception:
                pass

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        if base_url is not None:
            hostname = base_url_hostname(base_url)
        else:
            hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
                getattr(self, "_base_url_lower", "")
            )
        return hostname == "api.openai.com"

    def _is_azure_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets Azure OpenAI.

        Azure OpenAI exposes an OpenAI-compatible endpoint at
        ``{resource}.openai.azure.com/openai/v1`` that accepts the
        standard ``openai`` Python client.  Unlike api.openai.com it
        does NOT support the Responses API — gpt-5.x models are served
        on the regular ``/chat/completions`` path — so routing decisions
        must treat Azure separately from direct OpenAI.
        """
        if base_url is not None:
            url = str(base_url).lower()
        else:
            url = getattr(self, "_base_url_lower", "") or ""
        return "openai.azure.com" in url

    def _is_github_copilot_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets GitHub Copilot's OpenAI-compatible API."""
        if base_url is not None:
            hostname = base_url_hostname(base_url)
        else:
            hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
                getattr(self, "_base_url_lower", "")
            )
        return hostname == "api.githubcopilot.com"

    def _resolved_api_call_timeout(self) -> float:
        """Resolve the effective per-call request timeout in seconds.

        Priority:
          1. ``providers.<id>.models.<model>.timeout_seconds`` (per-model override)
          2. ``providers.<id>.request_timeout_seconds`` (provider-wide)
          3. ``HERMES_API_TIMEOUT`` env var (legacy escape hatch)
          4. 1800.0s default

        Used by OpenAI-wire chat completions (streaming and non-streaming) so
        the per-provider config knob wins over the 1800s default.  Without this
        helper, the hardcoded ``HERMES_API_TIMEOUT`` fallback would always be
        passed as a per-call ``timeout=`` kwarg, overriding the client-level
        timeout the AIAgent.__init__ path configured.
        """
        cfg = get_provider_request_timeout(self.provider, self.model)
        if cfg is not None:
            return cfg
        return float(os.getenv("HERMES_API_TIMEOUT", 1800.0))

    def _resolved_api_call_stale_timeout_base(self) -> tuple[float, bool]:
        """Resolve the base non-stream stale timeout and whether it is implicit.

        Priority:
          1. ``providers.<id>.models.<model>.stale_timeout_seconds``
          2. ``providers.<id>.stale_timeout_seconds``
          3. ``HERMES_API_CALL_STALE_TIMEOUT`` env var
          4. 300.0s default

        Returns ``(timeout_seconds, uses_implicit_default)`` so the caller can
        preserve legacy behaviors that only apply when the user has *not*
        explicitly configured a stale timeout, such as auto-disabling the
        detector for local endpoints.
        """
        cfg = get_provider_stale_timeout(self.provider, self.model)
        if cfg is not None:
            return cfg, False

        env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT")
        if env_timeout is not None:
            return float(env_timeout), False

        return 300.0, True

    def _compute_non_stream_stale_timeout(self, messages: list[dict[str, Any]]) -> float:
        """Compute the effective non-stream stale timeout for this request."""
        stale_base, uses_implicit_default = self._resolved_api_call_stale_timeout_base()
        base_url = getattr(self, "_base_url", None) or self.base_url or ""
        if uses_implicit_default and base_url and is_local_endpoint(base_url):
            return float("inf")

        est_tokens = sum(len(str(v)) for v in messages) // 4
        if est_tokens > 100_000:
            return max(stale_base, 600.0)
        if est_tokens > 50_000:
            return max(stale_base, 450.0)
        return stale_base

    def _is_openrouter_url(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return base_url_host_matches(self._base_url_lower, "openrouter.ai")

    def _anthropic_prompt_cache_policy(
        self,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[bool, bool]:
        """Decide whether to apply Anthropic prompt caching and which layout to use.

        Returns ``(should_cache, use_native_layout)``:
          * ``should_cache`` — inject ``cache_control`` breakpoints for this
            request (applies to OpenRouter Claude, native Anthropic, and
            third-party gateways that speak the native Anthropic protocol).
          * ``use_native_layout`` — place markers on the *inner* content
            blocks (native Anthropic accepts and requires this layout);
            when False markers go on the message envelope (OpenRouter and
            OpenAI-wire proxies expect the looser layout).

        Third-party providers using the native Anthropic transport
        (``api_mode == 'anthropic_messages'`` + Claude-named model) get
        caching with the native layout so they benefit from the same
        cost reduction as direct Anthropic callers, provided their
        gateway implements the Anthropic cache_control contract
        (MiniMax, Zhipu GLM, LiteLLM's Anthropic proxy mode all do).

        Qwen / Alibaba-family models on OpenCode, OpenCode Go, and direct
        Alibaba (DashScope) also honour Anthropic-style ``cache_control``
        markers on OpenAI-wire chat completions. Upstream pi-mono #3392 /
        pi #3393 documented this for opencode-go Qwen. Without markers
        these providers serve zero cache hits, re-billing the full prompt
        on every turn.
        """
        eff_provider = (provider if provider is not None else self.provider) or ""
        eff_base_url = base_url if base_url is not None else (self.base_url or "")
        eff_api_mode = api_mode if api_mode is not None else (self.api_mode or "")
        eff_model = (model if model is not None else self.model) or ""

        model_lower = eff_model.lower()
        provider_lower = eff_provider.lower()
        is_claude = "claude" in model_lower
        is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
        # Nous Portal proxies to OpenRouter behind the scenes — identical
        # OpenAI-wire envelope cache_control semantics. Treat it as an
        # OpenRouter-equivalent endpoint for caching layout purposes.
        is_nous_portal = "nousresearch" in eff_base_url.lower()
        is_anthropic_wire = eff_api_mode == "anthropic_messages"
        is_native_anthropic = (
            is_anthropic_wire
            and (eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com")
        )

        if is_native_anthropic:
            return True, True
        if (is_openrouter or is_nous_portal) and is_claude:
            return True, False
        # Nous Portal Qwen (e.g. qwen3.6-plus) takes the same envelope-layout
        # cache_control path as Portal Claude. Portal proxies to OpenRouter
        # and the upstream Qwen route accepts cache_control markers; without
        # this branch the alibaba-family check below only matches
        # provider=opencode/alibaba and Portal traffic falls through to
        # (False, False), serving 0% cache hits and re-billing the full
        # prompt on every turn.
        if is_nous_portal and "qwen" in model_lower:
            return True, False
        if is_anthropic_wire and is_claude:
            # Third-party Anthropic-compatible gateway.
            return True, True

        # MiniMax on its Anthropic-compatible endpoint serves its own
        # model family (MiniMax-M2.7, M2.5, M2.1, M2) with documented
        # cache_control support (0.1× read pricing, 5-minute TTL).  The
        # blanket is_claude gate above excludes these — opt them in
        # explicitly via provider id or host match so users on
        # provider=minimax / minimax-cn (or custom endpoints pointing at
        # api.minimax.io/anthropic / api.minimaxi.com/anthropic) get the
        # same cost reduction as Claude traffic.
        # Docs: https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache
        if is_anthropic_wire:
            is_minimax_provider = provider_lower in {"minimax", "minimax-cn"}
            is_minimax_host = (
                base_url_host_matches(eff_base_url, "api.minimax.io")
                or base_url_host_matches(eff_base_url, "api.minimaxi.com")
            )
            if is_minimax_provider or is_minimax_host:
                return True, True

        # Qwen/Alibaba on OpenCode (Zen/Go) and native DashScope: OpenAI-wire
        # transport that accepts Anthropic-style cache_control markers and
        # rewards them with real cache hits.  Without this branch
        # qwen3.6-plus on opencode-go reports 0% cached tokens and burns
        # through the subscription on every turn.
        model_is_qwen = "qwen" in model_lower
        provider_is_alibaba_family = provider_lower in {
            "opencode", "opencode-zen", "opencode-go", "alibaba",
        }
        if provider_is_alibaba_family and model_is_qwen:
            # Envelope layout (native_anthropic=False): markers on inner
            # content parts, not top-level tool messages.  Matches
            # pi-mono's "alibaba" cacheControlFormat.
            return True, False

        return False, False

    @staticmethod
    def _model_requires_responses_api(model: str) -> bool:
        """Return True for models that require the Responses API path.

        GPT-5.x models are rejected on /v1/chat/completions by both
        OpenAI and OpenRouter (error: ``unsupported_api_for_model``).
        Detect these so the correct api_mode is set regardless of
        which provider is serving the model.
        """
        m = model.lower()
        # Strip vendor prefix (e.g. "openai/gpt-5.4" → "gpt-5.4")
        if "/" in m:
            m = m.rsplit("/", 1)[-1]
        return m.startswith("gpt-5")

    @staticmethod
    def _provider_model_requires_responses_api(
        model: str,
        *,
        provider: Optional[str] = None,
    ) -> bool:
        """Return True when this provider/model pair should use Responses API."""
        normalized_provider = (provider or "").strip().lower()
        # Nous serves GPT-5.x models via its OpenAI-compatible chat
        # completions endpoint; its /v1/responses endpoint returns 404.
        if normalized_provider == "nous":
            return False
        if normalized_provider == "copilot":
            try:
                from hermes_cli.models import _should_use_copilot_responses_api
                return _should_use_copilot_responses_api(model)
            except Exception:
                # Fall back to the generic GPT-5 rule if Copilot-specific
                # logic is unavailable for any reason.
                pass
        return AIAgent._model_requires_responses_api(model)

    def _max_tokens_param(self, value: int) -> dict:
        """Return the correct max tokens kwarg for the current provider.

        OpenAI's newer models (gpt-4o, o-series, gpt-5+) require
        'max_completion_tokens'. Azure OpenAI also requires
        'max_completion_tokens' for gpt-5.x models served via the
        OpenAI-compatible endpoint. OpenRouter, local models, and older
        OpenAI models use 'max_tokens'.
        """
        if self._is_direct_openai_url() or self._is_azure_openai_url() or self._is_github_copilot_url():
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    def _has_content_after_think_block(self, content: str) -> bool:
        """
        Check if content has actual text after any reasoning/thinking blocks.

        This detects cases where the model only outputs reasoning but no actual
        response, which indicates an incomplete generation that should be retried.
        Must stay in sync with _strip_think_blocks() tag variants.

        Args:
            content: The assistant message content to check

        Returns:
            True if there's meaningful content after think blocks, False otherwise
        """
        if not content:
            return False

        # Remove all reasoning tag variants (must match _strip_think_blocks)
        cleaned = self._strip_think_blocks(content)

        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())

    def _strip_think_blocks(self, content: str) -> str:
        """Remove reasoning/thinking blocks from content, returning only visible text.

        Handles four cases:
          1. Closed tag pairs (``<think>…</think>``) — the common path when
             the provider emits complete reasoning blocks.
          2. Unterminated open tag at a block boundary (start of text or
             after a newline) — e.g. MiniMax M2.7 / NIM endpoints where the
             closing tag is dropped.  Everything from the open tag to end
             of string is stripped.  The block-boundary check mirrors
             ``gateway/stream_consumer.py``'s filter so models that mention
             ``<think>`` in prose aren't over-stripped.
          3. Stray orphan open/close tags that slip through.
          4. Tag variants: ``<think>``, ``<thinking>``, ``<reasoning>``,
             ``<REASONING_SCRATCHPAD>``, ``<thought>`` (Gemma 4), all
             case-insensitive.

        Additionally strips standalone tool-call XML blocks that some open
        models (notably Gemma variants on OpenRouter) emit inside assistant
        content instead of via the structured ``tool_calls`` field:
          * ``<tool_call>…</tool_call>``
          * ``<tool_calls>…</tool_calls>``
          * ``<tool_result>…</tool_result>``
          * ``<function_call>…</function_call>``
          * ``<function_calls>…</function_calls>``
          * ``<function name="…">…</function>`` (Gemma style)
        Ported from openclaw/openclaw#67318. The ``<function>`` variant is
        boundary-gated (only strips when the tag sits at start-of-line or
        after punctuation and carries a ``name="..."`` attribute) so prose
        mentions like "Use <function> in JavaScript" are preserved.
        """
        if not content:
            return ""
        # 1. Closed tag pairs — case-insensitive for all variants so
        #    mixed-case tags (<THINK>, <Thinking>) don't slip through to
        #    the unterminated-tag pass and take trailing content with them.
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 1b. Tool-call XML blocks (openclaw/openclaw#67318). Handle the
        #     generic tag names first — they have no attribute gating since
        #     a literal <tool_call> in prose is already vanishingly rare.
        for _tc_name in ("tool_call", "tool_calls", "tool_result",
                          "function_call", "function_calls"):
            content = re.sub(
                rf'<{_tc_name}\b[^>]*>.*?</{_tc_name}>',
                '',
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )
        # 1c. <function name="...">...</function> — Gemma-style standalone
        #     tool call. Only strip when the tag sits at a block boundary
        #     (start of text, after a newline, or after sentence-ending
        #     punctuation) AND carries a name="..." attribute. This keeps
        #     prose mentions like "Use <function> to declare" safe.
        content = re.sub(
            r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
            r'<function\b[^>]*\bname\s*=[^>]*>'
            r'(?:(?:(?!</function>).)*)</function>',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # 2. Unterminated reasoning block — open tag at a block boundary
        #    (start of text, or after a newline) with no matching close.
        #    Strip from the tag to end of string.  Fixes #8878 / #9568
        #    (MiniMax M2.7 leaking raw reasoning into assistant content).
        content = re.sub(
            r'(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # 3. Stray orphan open/close tags that slipped through.
        content = re.sub(
            r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*',
            '',
            content,
            flags=re.IGNORECASE,
        )
        # 3b. Stray tool-call closers. (We do NOT strip bare <function> or
        #     unterminated <function name="..."> because a truncated tail
        #     during streaming may still be valuable to the user; matches
        #     OpenClaw's intentional asymmetry.)
        content = re.sub(
            r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*',
            '',
            content,
            flags=re.IGNORECASE,
        )
        return content

    @staticmethod
    def _has_natural_response_ending(content: str) -> bool:
        """Heuristic: does visible assistant text look intentionally finished?"""
        if not content:
            return False
        stripped = content.rstrip()
        if not stripped:
            return False
        if stripped.endswith("```"):
            return True
        return stripped[-1] in '.!?:)"\']}。！？：）】」』》'

    def _is_ollama_glm_backend(self) -> bool:
        """Detect the narrow backend family affected by Ollama/GLM stop misreports."""
        model_lower = (self.model or "").lower()
        provider_lower = (self.provider or "").lower()
        if "glm" not in model_lower and provider_lower != "zai":
            return False
        if "ollama" in self._base_url_lower or ":11434" in self._base_url_lower:
            return True
        return bool(self.base_url and is_local_endpoint(self.base_url))

    def _should_treat_stop_as_truncated(
        self,
        finish_reason: str,
        assistant_message,
        messages: Optional[list] = None,
    ) -> bool:
        """Detect conservative stop->length misreports for Ollama-hosted GLM models."""
        if finish_reason != "stop" or self.api_mode != "chat_completions":
            return False
        if not self._is_ollama_glm_backend():
            return False
        if not any(
            isinstance(msg, dict) and msg.get("role") == "tool"
            for msg in (messages or [])
        ):
            return False
        if assistant_message is None or getattr(assistant_message, "tool_calls", None):
            return False

        content = getattr(assistant_message, "content", None)
        if not isinstance(content, str):
            return False

        visible_text = self._strip_think_blocks(content).strip()
        if not visible_text:
            return False
        if len(visible_text) < 20 or not re.search(r"\s", visible_text):
            return False

        return not self._has_natural_response_ending(visible_text)

    def _looks_like_codex_intermediate_ack(
        self,
        user_message: str,
        assistant_content: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """Detect a planning/ack message that should continue instead of ending the turn."""
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
            return False

        assistant_text = self._strip_think_blocks(assistant_content or "").strip().lower()
        if not assistant_text:
            return False
        if len(assistant_text) > 1200:
            return False

        has_future_ack = bool(
            re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
        )
        if not has_future_ack:
            return False

        action_markers = (
            "look into",
            "look at",
            "inspect",
            "scan",
            "check",
            "analyz",
            "review",
            "explore",
            "read",
            "open",
            "run",
            "test",
            "fix",
            "debug",
            "search",
            "find",
            "walkthrough",
            "report back",
            "summarize",
        )
        workspace_markers = (
            "directory",
            "current directory",
            "current dir",
            "cwd",
            "repo",
            "repository",
            "codebase",
            "project",
            "folder",
            "filesystem",
            "file tree",
            "files",
            "path",
        )

        user_text = (user_message or "").strip().lower()
        user_targets_workspace = (
            any(marker in user_text for marker in workspace_markers)
            or "~/" in user_text
            or "/" in user_text
        )
        assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
        assistant_targets_workspace = any(
            marker in assistant_text for marker in workspace_markers
        )
        return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action


    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        Extract reasoning/thinking content from an assistant message.
        
        OpenRouter and various providers can return reasoning in multiple formats:
        1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
        2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
        3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)
        
        Args:
            assistant_message: The assistant message object from the API response
            
        Returns:
            Combined reasoning text, or None if no reasoning found
        """
        reasoning_parts = []
        
        # Check direct reasoning field
        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)
        
        # Check reasoning_content field (alternative name used by some providers)
        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            # Don't duplicate if same as reasoning
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)
        
        # Check reasoning_details array (OpenRouter unified format)
        # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # Extract summary from reasoning detail object
                    summary = (
                        detail.get('summary')
                        or detail.get('thinking')
                        or detail.get('content')
                        or detail.get('text')
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # Some providers embed reasoning directly inside assistant content
        # instead of returning structured reasoning fields.  Only fall back
        # to inline extraction when no structured reasoning was found.
        content = getattr(assistant_message, "content", None)
        if not reasoning_parts and isinstance(content, list):
            # DeepSeek V4 Pro (and compatible providers) return content as a
            # list of typed blocks, e.g.:
            #   [{"type": "thinking", "thinking": "..."}, {"type": "output", ...}]
            # Without this branch the thinking text is silently dropped and the
            # next turn fails with HTTP 400 ("thinking must be passed back").
            # Refs #21944.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking") or block.get("text") or ""
                    thinking_text = thinking_text.strip()
                    if thinking_text and thinking_text not in reasoning_parts:
                        reasoning_parts.append(thinking_text)
        if not reasoning_parts and isinstance(content, str) and content:
            inline_patterns = (
                r"<think>(.*?)</think>",
                r"<thinking>(.*?)</thinking>",
                r"<thought>(.*?)</thought>",
                r"<reasoning>(.*?)</reasoning>",
                r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
            )
            for pattern in inline_patterns:
                flags = re.DOTALL | re.IGNORECASE
                for block in re.findall(pattern, content, flags=flags):
                    cleaned = block.strip()
                    if cleaned and cleaned not in reasoning_parts:
                        reasoning_parts.append(cleaned)
        
        # Combine all reasoning parts
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None

    def _cleanup_task_resources(self, task_id: str) -> None:
        """Clean up VM and browser resources for a given task.

        Skips ``cleanup_vm`` when the active terminal environment is marked
        persistent (``persistent_filesystem=True``) so that long-lived sandbox
        containers survive between turns. The idle reaper in
        ``terminal_tool._cleanup_inactive_envs`` still tears them down once
        ``terminal.lifetime_seconds`` is exceeded. Non-persistent backends are
        torn down per-turn as before to prevent resource leakage (the original
        intent of this hook for the Morph backend, see commit fbd3a2fd).
        """
        try:
            if is_persistent_env(task_id):
                if self.verbose_logging:
                    logging.debug(
                        f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                        f"idle reaper will handle it."
                    )
            else:
                cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # Background memory/skill review
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "Review the conversation above and consider saving to memory if appropriate.\n\n"
        "Focus on:\n"
        "1. Has the user revealed things about themselves — their persona, desires, "
        "preferences, or personal details worth remembering?\n"
        "2. Has the user expressed expectations about how you should behave, their work "
        "style, or ways they want you to operate?\n\n"
        "If something stands out, save it using the memory tool. "
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _SKILL_REVIEW_PROMPT = (
        "Review the conversation above and update the skill library. Be "
        "ACTIVE — most sessions produce at least one skill update, even if "
        "small. A pass that does nothing is a missed learning opportunity, "
        "not a neutral outcome.\n\n"
        "Target shape of the library: CLASS-LEVEL skills, each with a rich "
        "SKILL.md and a `references/` directory for session-specific detail. "
        "Not a long flat list of narrow one-session-one-skill entries. This "
        "shapes HOW you update, not WHETHER you update.\n\n"
        "Signals to look for (any one of these warrants action):\n"
        "  • User corrected your style, tone, format, legibility, or "
        "verbosity. Frustration signals like 'stop doing X', 'this is too "
        "verbose', 'don't format like this', 'why are you explaining', "
        "'just give me the answer', 'you always do Y and I hate it', or an "
        "explicit 'remember this' are FIRST-CLASS skill signals, not just "
        "memory signals. Update the relevant skill(s) to embed the "
        "preference so the next session starts already knowing.\n"
        "  • User corrected your workflow, approach, or sequence of steps. "
        "Encode the correction as a pitfall or explicit step in the skill "
        "that governs that class of task.\n"
        "  • Non-trivial technique, fix, workaround, debugging path, or "
        "tool-usage pattern emerged that a future session would benefit "
        "from. Capture it.\n"
        "  • A skill that got loaded or consulted this session turned out "
        "to be wrong, missing a step, or outdated. Patch it NOW.\n\n"
        "Preference order — prefer the earliest action that fits, but do "
        "pick one when a signal above fired:\n"
        "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the "
        "conversation for skills the user loaded via /skill-name or you "
        "read via skill_view. If any of them covers the territory of the "
        "new learning, PATCH that one first. It is the skill that was in "
        "play, so it's the right one to extend.\n"
        "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). "
        "If no loaded skill fits but an existing class-level skill does, "
        "patch it. Add a subsection, a pitfall, or broaden a trigger.\n"
        "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be "
        "packaged with three kinds of support files — use the right "
        "directory per kind:\n"
        "     • `references/<topic>.md` — session-specific detail (error "
        "transcripts, reproduction recipes, provider quirks) AND "
        "condensed knowledge banks: quoted research, API docs, external "
        "authoritative excerpts, or domain notes you found while working "
        "on the problem. Write it concise and for the value of the task, "
        "not as a full mirror of upstream docs.\n"
        "     • `templates/<name>.<ext>` — starter files meant to be "
        "copied and modified (boilerplate configs, scaffolding, a "
        "known-good example the agent can `reproduce with modifications`).\n"
        "     • `scripts/<name>.<ext>` — statically re-runnable actions "
        "the skill can invoke directly (verification scripts, fixture "
        "generators, deterministic probes, anything the agent should run "
        "rather than hand-type each time).\n"
        "     Add support files via skill_manage action=write_file with "
        "file_path starting 'references/', 'templates/', or 'scripts/'. "
        "The umbrella's SKILL.md should gain a one-line pointer to any "
        "new support file so future agents know it exists.\n"
        "  4. CREATE A NEW CLASS-LEVEL UMBRELLA SKILL when no existing "
        "skill covers the class. The name MUST be at the class level. "
        "The name MUST NOT be a specific PR number, error string, feature "
        "codename, library-alone name, or 'fix-X / debug-Y / audit-Z-today' "
        "session artifact. If the proposed name only makes sense for "
        "today's task, it's wrong — fall back to (1), (2), or (3).\n\n"
        "User-preference embedding (important): when the user expressed a "
        "style/format/workflow preference, the update belongs in the "
        "SKILL.md body, not just in memory. Memory captures 'who the user "
        "is and what the current situation and state of your operations "
        "are'; skills capture 'how to do this class of task for this "
        "user'. When they complain about how you handled a task, the "
        "skill that governs that task needs to carry the lesson.\n\n"
        "If you notice two existing skills that overlap, note it in your "
        "reply — the background curator handles consolidation at scale.\n\n"
        "Do NOT capture (these become persistent self-imposed constraints "
        "that bite you later when the environment changes):\n"
        "  • Environment-dependent failures: missing binaries, fresh-install "
        "errors, post-migration path mismatches, 'command not found', "
        "unconfigured credentials, uninstalled packages. The user can fix "
        "these — they are not durable rules.\n"
        "  • Negative claims about tools or features ('browser tools do not "
        "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
        "harden into refusals the agent cites against itself for months "
        "after the actual problem was fixed.\n"
        "  • Session-specific transient errors that resolved before the "
        "conversation ended. If retrying worked, the lesson is the retry "
        "pattern, not the original failure.\n"
        "  • One-off task narratives. A user asking 'summarize today's "
        "market' or 'analyze this PR' is not a class of work that warrants "
        "a skill.\n\n"
        "If a tool failed because of setup state, capture the FIX (install "
        "command, config step, env var to set) under an existing setup or "
        "troubleshooting skill — never 'this tool does not work' as a "
        "standalone constraint.\n\n"
        "'Nothing to save.' is a real option but should NOT be the "
        "default. If the session ran smoothly with no corrections and "
        "produced no new technique, just say 'Nothing to save.' and stop. "
        "Otherwise, act."
    )

    _COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and update two things:\n\n"
        "**Memory**: who the user is. Did the user reveal persona, "
        "desires, preferences, personal details, or expectations about "
        "how you should behave? Save facts about the user and durable "
        "preferences with the memory tool.\n\n"
        "**Skills**: how to do this class of task. Be ACTIVE — most "
        "sessions produce at least one skill update. A pass that does "
        "nothing is a missed learning opportunity, not a neutral outcome.\n\n"
        "Target shape of the skill library: CLASS-LEVEL skills with a rich "
        "SKILL.md and a `references/` directory for session-specific detail. "
        "Not a long flat list of narrow one-session-one-skill entries.\n\n"
        "Signals that warrant a skill update (any one is enough):\n"
        "  • User corrected your style, tone, format, legibility, "
        "verbosity, or approach. Frustration is a FIRST-CLASS skill "
        "signal, not just a memory signal. 'stop doing X', 'don't format "
        "like this', 'I hate when you Y' — embed the lesson in the skill "
        "that governs that task so the next session starts fixed.\n"
        "  • Non-trivial technique, fix, workaround, or debugging path "
        "emerged.\n"
        "  • A skill that was loaded or consulted turned out wrong, "
        "missing, or outdated — patch it now.\n\n"
        "Preference order for skills — pick the earliest that fits:\n"
        "  1. UPDATE A CURRENTLY-LOADED SKILL. Check what skills were "
        "loaded via /skill-name or skill_view in the conversation. If one "
        "of them covers the learning, PATCH it first. It was in play; "
        "it's the right place.\n"
        "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to "
        "find the right one). Patch it.\n"
        "  3. ADD A SUPPORT FILE under an existing umbrella via "
        "skill_manage action=write_file. Three kinds: "
        "`references/<topic>.md` for session-specific detail OR condensed "
        "knowledge banks (quoted research, API docs excerpts, domain "
        "notes) written concise and task-focused; `templates/<name>.<ext>` "
        "for starter files meant to be copied and modified; "
        "`scripts/<name>.<ext>` for statically re-runnable actions "
        "(verification, fixture generators, probes). Add a one-line "
        "pointer in SKILL.md so future agents find them.\n"
        "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. "
        "Name at the class level — NOT a PR number, error string, "
        "codename, library-alone name, or 'fix-X / debug-Y' session "
        "artifact. If the name only fits today's task, fall back to (1), "
        "(2), or (3).\n\n"
        "User-preference embedding: when the user complains about how "
        "you handled a task, update the skill that governs that task — "
        "memory alone isn't enough. Memory says 'who the user is and "
        "what the current situation and state of your operations are'; "
        "skills say 'how to do this class of task for this user'. Both "
        "should carry user-preference lessons when relevant.\n\n"
        "If you notice overlapping existing skills, mention it — the "
        "background curator handles consolidation.\n\n"
        "Do NOT capture as skills (these become persistent self-imposed "
        "constraints that bite you later when the environment changes):\n"
        "  • Environment-dependent failures: missing binaries, fresh-install "
        "errors, post-migration path mismatches, 'command not found', "
        "unconfigured credentials, uninstalled packages. The user can fix "
        "these — they are not durable rules.\n"
        "  • Negative claims about tools or features ('browser tools do not "
        "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
        "harden into refusals the agent cites against itself for months "
        "after the actual problem was fixed.\n"
        "  • Session-specific transient errors that resolved before the "
        "conversation ended. If retrying worked, the lesson is the retry "
        "pattern, not the original failure.\n"
        "  • One-off task narratives. A user asking 'summarize today's "
        "market' or 'analyze this PR' is not a class of work that warrants "
        "a skill.\n\n"
        "If a tool failed because of setup state, capture the FIX (install "
        "command, config step, env var to set) under an existing setup or "
        "troubleshooting skill — never 'this tool does not work' as a "
        "standalone constraint.\n\n"
        "Act on whichever of the two dimensions has real signal. If "
        "genuinely nothing stands out on either, say 'Nothing to save.' "
        "and stop — but don't reach for that conclusion as a default."
    )

    @staticmethod
    def _summarize_background_review_actions(
        review_messages: List[Dict],
        prior_snapshot: List[Dict],
    ) -> List[str]:
        """Build the human-facing action summary for a background review pass.

        Walks the review agent's session messages and collects "successful tool
        action" descriptions to surface to the user (e.g. "Memory updated").
        Tool messages already present in ``prior_snapshot`` are skipped so we
        don't re-surface stale results from the prior conversation that the
        review agent inherited via ``conversation_history`` (issue #14944).

        Matching is by ``tool_call_id`` when available, with a content-equality
        fallback for tool messages that lack one.
        """
        existing_tool_call_ids = set()
        existing_tool_contents = set()
        for prior in prior_snapshot or []:
            if not isinstance(prior, dict) or prior.get("role") != "tool":
                continue
            tcid = prior.get("tool_call_id")
            if tcid:
                existing_tool_call_ids.add(tcid)
            else:
                content = prior.get("content")
                if isinstance(content, str):
                    existing_tool_contents.add(content)

        actions: List[str] = []
        for msg in review_messages or []:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            tcid = msg.get("tool_call_id")
            if tcid and tcid in existing_tool_call_ids:
                continue
            if not tcid:
                content_str = msg.get("content")
                if isinstance(content_str, str) and content_str in existing_tool_contents:
                    continue
            try:
                data = json.loads(msg.get("content", "{}"))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict) or not data.get("success"):
                continue
            message = data.get("message", "")
            target = data.get("target", "")
            if "created" in message.lower():
                actions.append(message)
            elif "updated" in message.lower():
                actions.append(message)
            elif "added" in message.lower() or (target and "add" in message.lower()):
                label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                actions.append(f"{label} updated")
            elif "Entry added" in message:
                label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                actions.append(f"{label} updated")
            elif "removed" in message.lower() or "replaced" in message.lower():
                label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                actions.append(f"{label} updated")
        return actions

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """Spawn a background thread to review the conversation for memory/skill saves.

        Creates a full AIAgent fork with the same model, tools, and context as the
        main session. The review prompt is appended as the next user turn in the
        forked conversation. Writes directly to the shared memory/skill stores.
        Never modifies the main conversation history or produces user-visible output.
        """
        import threading

        # Pick the right prompt based on which triggers fired
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib
            # Install a non-interactive approval callback on this worker
            # thread so any dangerous-command guard the review agent trips
            # resolves to "deny" instead of falling back to input() -- which
            # deadlocks against the parent's prompt_toolkit TUI (#15216).
            # Same pattern as _subagent_auto_deny in tools/delegate_tool.py.
            def _bg_review_auto_deny(command, description, **kwargs):
                logger.warning(
                    "Background review auto-denied dangerous command: %s (%s)",
                    command, description,
                )
                return "deny"
            try:
                _set_approval_callback(_bg_review_auto_deny)
            except Exception:
                pass
            review_agent = None
            try:
                with open(os.devnull, "w", encoding="utf-8") as _devnull, \
                     contextlib.redirect_stdout(_devnull), \
                     contextlib.redirect_stderr(_devnull):
                    # Inherit the parent agent's live runtime (provider, model,
                    # base_url, api_key, api_mode) so the fork uses the exact
                    # same credentials the main turn is using.  Without this,
                    # AIAgent.__init__ re-runs auto-resolution from env vars,
                    # which fails for OAuth-only providers, session-scoped
                    # creds, or credential-pool setups where the resolver can't
                    # reconstruct auth from scratch -- producing the spurious
                    # "No LLM provider configured" warning at end of turn.
                    _parent_runtime = self._current_main_runtime()
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=16,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                        api_mode=_parent_runtime.get("api_mode") or None,
                        base_url=_parent_runtime.get("base_url") or None,
                        api_key=_parent_runtime.get("api_key") or None,
                        credential_pool=getattr(self, "_credential_pool", None),
                        parent_session_id=self.session_id,
                        enabled_toolsets=["memory", "skills"],
                    )
                    review_agent._memory_write_origin = "background_review"
                    review_agent._memory_write_context = "background_review"
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0
                    # Suppress all status/warning emits from the fork so the
                    # user only sees the final successful-action summary.
                    # Without this, mid-review "Iteration budget exhausted",
                    # rate-limit retries, compression warnings, and other
                    # lifecycle messages bubble up through _emit_status ->
                    # _vprint and leak past the stdout redirect (they go via
                    # _print_fn/status_callback, which bypass sys.stdout).
                    review_agent.suppress_status_output = True

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # Scan the review agent's messages for successful tool actions
                # and surface a compact summary to the user. Tool messages
                # already present in messages_snapshot must be skipped, since
                # the review agent inherits that history and would otherwise
                # re-surface stale "created"/"updated" messages from the prior
                # conversation as if they just happened (issue #14944).
                actions = self._summarize_background_review_actions(
                    getattr(review_agent, "_session_messages", []),
                    messages_snapshot,
                )

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(
                        f"  💾 Self-improvement review: {summary}"
                    )
                    _bg_cb = self.background_review_callback
                    if _bg_cb:
                        try:
                            _bg_cb(
                                f"💾 Self-improvement review: {summary}"
                            )
                        except Exception:
                            pass

            except Exception as e:
                logger.warning("Background memory/skill review failed: %s", e)
                self._emit_auxiliary_failure("background review", e)
            finally:
                # Background review agents can initialize memory providers
                # (for example Hindsight) that own their own network clients.
                # Explicitly stop those providers before closing the agent so
                # their aiohttp sessions do not leak until GC/process exit.
                # Then close all remaining resources (httpx client,
                # subprocesses, etc.) so GC doesn't try to clean them up on a
                # dead asyncio event loop (which produces "Event loop is
                # closed" errors).
                if review_agent is not None:
                    try:
                        review_agent.shutdown_memory_provider()
                    except Exception:
                        pass
                    try:
                        review_agent.close()
                    except Exception:
                        pass
                # Clear the approval callback on this bg-review thread so a
                # recycled thread-id doesn't inherit a stale reference.
                try:
                    _set_approval_callback(None)
                except Exception:
                    pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()

    def _build_memory_write_metadata(
        self,
        *,
        write_origin: Optional[str] = None,
        execution_context: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build provenance metadata for external memory-provider mirrors."""
        metadata: Dict[str, Any] = {
            "write_origin": write_origin or getattr(self, "_memory_write_origin", "assistant_tool"),
            "execution_context": (
                execution_context
                or getattr(self, "_memory_write_context", "foreground")
            ),
            "session_id": self.session_id or "",
            "parent_session_id": self._parent_session_id or "",
            "platform": self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
            "tool_name": "memory",
        }
        if task_id:
            metadata["task_id"] = task_id
        if tool_call_id:
            metadata["tool_call_id"] = tool_call_id
        return {k: v for k, v in metadata.items() if v not in {None, ""}}

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message before persistence/return.

        Some call paths need an API-only user-message variant without letting
        that synthetic text leak into persisted transcripts or resumed session
        history. When an override is configured for the active turn, mutate the
        in-memory messages list in place so both persistence and returned
        history stay clean.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite on any exit path.

        Ensures conversations are never lost, even on errors or early returns.
        """
        self._drop_trailing_empty_response_scaffolding(messages)
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)

    def _drop_trailing_empty_response_scaffolding(self, messages: List[Dict]) -> None:
        """Remove private empty-response retry/failure scaffolding from transcript tails.

        Also rewinds past any trailing tool-result / assistant(tool_calls) pair
        that the failed iteration left hanging. Without this, the tail ends at
        a raw ``tool`` message and the next user turn lands as
        ``...tool, user, user`` — a protocol-invalid sequence that most
        providers silently reject (returns empty content), causing the
        empty-retry loop to fire forever. See #<TBD>.
        """
        # Pass 1: strip the flagged scaffolding messages themselves.
        dropped_scaffolding = False
        while (
            messages
            and isinstance(messages[-1], dict)
            and (
                messages[-1].get("_empty_recovery_synthetic")
                or messages[-1].get("_empty_terminal_sentinel")
            )
        ):
            messages.pop()
            dropped_scaffolding = True

        # Pass 2: if we stripped scaffolding, rewind through any trailing
        # tool-result messages plus the assistant(tool_calls) message that
        # produced them. This preserves role alternation so the next user
        # message follows a user or assistant message, not an orphan tool
        # result. Only runs when scaffolding was actually present — normal
        # conversation tails (real tool loops mid-progress) are untouched.
        if not dropped_scaffolding:
            return

        # Drop any trailing tool-result messages
        while (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "tool"
        ):
            messages.pop()

        # Drop the assistant message that issued the tool calls, if the tail
        # now ends in an assistant-with-tool_calls (the pair that owned the
        # just-popped tool results). Without this, the tail is
        # ``assistant(tool_calls=...)`` with no tool answers, which some
        # providers also reject.
        if (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            messages.pop()

    def _repair_message_sequence(self, messages: List[Dict]) -> int:
        """Collapse malformed role-alternation left in the live history.

        Providers (OpenAI, OpenRouter, Anthropic) expect strict alternation:
        after the system message, user/tool alternates with assistant, with
        no two consecutive user messages and no tool-result that doesn't
        follow an assistant-with-tool_calls. Violations cause silent empty
        responses on most providers, which triggers the empty-retry loop.

        This runs right before the API call as a defensive belt — by the
        time it fires, the scaffolding strip should already have prevented
        most shapes, but external callers (gateway multi-queue replay,
        session resume, cron, explicit conversation_history passed in by
        host code) can feed in already-broken histories.

        Repairs applied:
          1. Stray ``tool`` messages whose ``tool_call_id`` doesn't match
             any preceding assistant tool_call — dropped.
          2. Consecutive ``user`` messages — merged with newline separator
             so no user input is lost.

        Deliberately does NOT rewind orphan ``assistant(tool_calls)+tool``
        pairs that precede a user message — that pattern IS valid when the
        previous turn completed normally and the user jumped in to redirect
        before the model got a continuation turn (the ongoing dialog
        pattern). The empty-response scaffolding stripper handles the
        genuinely-broken variant via its flag-gated rewind.

        Returns the number of repairs made (for logging/telemetry).
        """
        if not messages:
            return 0

        repairs = 0

        # Pass 1: drop stray tool messages that don't follow a known
        # assistant tool_call_id. Uses a rolling set of known ids refreshed
        # on each assistant message.
        known_tool_ids: set = set()
        filtered: List[Dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                filtered.append(msg)
                continue
            role = msg.get("role")
            if role == "assistant":
                known_tool_ids = set()
                for tc in (msg.get("tool_calls") or []):
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id:
                        known_tool_ids.add(tc_id)
                filtered.append(msg)
            elif role == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id and tc_id in known_tool_ids:
                    filtered.append(msg)
                else:
                    repairs += 1
            else:
                if role == "user":
                    # A user turn closes the tool-result run; subsequent
                    # tool messages without a fresh assistant tool_call
                    # are orphans.
                    known_tool_ids = set()
                filtered.append(msg)

        # Pass 2: merge consecutive user messages. Preserves all user input
        # so nothing the user typed is lost.
        merged: List[Dict] = []
        for msg in filtered:
            if (
                merged
                and isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(merged[-1], dict)
                and merged[-1].get("role") == "user"
            ):
                prev = merged[-1]
                prev_content = prev.get("content", "")
                new_content = msg.get("content", "")
                # Only merge plain-text content; leave multimodal (list)
                # content alone — collapsing image/audio blocks risks
                # mangling the attachment structure.
                if isinstance(prev_content, str) and isinstance(new_content, str):
                    prev["content"] = (
                        (prev_content + "\n\n" + new_content)
                        if prev_content and new_content
                        else (prev_content or new_content)
                    )
                    repairs += 1
                    continue
            merged.append(msg)

        if repairs > 0:
            # Rewrite in place so downstream paths (persistence, return
            # value, session DB flush) see the repaired sequence.
            messages[:] = merged

        return repairs

    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Persist any un-flushed messages to the SQLite session store.

        Uses _last_flushed_db_idx to track which messages have already been
        written, so repeated calls (from multiple exit paths) only write
        truly new messages — preventing the duplicate-write bug (#860).
        """
        if not self._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # Retry row creation if the earlier attempt failed transiently.
            if not self._session_db_created:
                self._ensure_db_session()
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                # Persist multimodal tool results as their text summary only —
                # base64 images would bloat the session DB and aren't useful
                # for cross-session replay.
                if _is_multimodal_tool_result(content):
                    content = _multimodal_text_summary(content)
                elif isinstance(content, list):
                    # List of OpenAI-style content parts: strip images, keep text.
                    _txt = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            _txt.append(str(p.get("text", "")))
                        elif isinstance(p, dict) and p.get("type") in {"image", "image_url", "input_image"}:
                            _txt.append("[screenshot]")
                    content = "\n".join(_txt) if _txt else None
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and isinstance(msg.tool_calls, list) and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_content=msg.get("reasoning_content") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                    codex_message_items=msg.get("codex_message_items") if role == "assistant" else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        Get messages up to (but not including) the last assistant turn.
        
        This is used when we need to "roll back" to the last successful point
        in the conversation, typically when the final assistant message is
        incomplete or malformed.
        
        Args:
            messages: Full message list
            
        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []
        
        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()
        
        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]

    def _format_tools_for_system_message(self) -> str:
        """
        Format tool definitions for the system message in the trajectory format.
        
        Returns:
            str: JSON string representation of tool definitions
        """
        if not self.tools:
            return "[]"
        
        # Convert tool definitions to the format expected in trajectories
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools, ensure_ascii=False)

    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """
        Convert internal message format to trajectory format for saving.
        
        Args:
            messages (List[Dict]): Internal message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
            
        Returns:
            List[Dict]: Messages in trajectory format
        """
        # Normalize multimodal tool results — trajectories are text-only, so
        # replace image-bearing tool messages with their text_summary to avoid
        # embedding ~1MB base64 blobs into every saved trajectory.
        messages = [_trajectory_normalize_msg(m) for m in messages]
        trajectory = []
        
        # Add system message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({
            "from": "system",
            "value": system_msg
        })
        
        # Add the actual user prompt (from the dataset) as the first human message
        trajectory.append({
            "from": "human",
            "value": user_query
        })
        
        # Skip the first message (the user query) since we already added it above.
        # Prefill messages are injected at API-call time only (not in the messages
        # list), so no offset adjustment is needed here.
        i = 1
        
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                # Check if this message has tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Format assistant message with tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    if msg.get("content") and msg["content"].strip():
                        # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                        # (used when native thinking is disabled and model reasons via XML)
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"
                    
                    # Add tool calls wrapped in XML tags
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        # Parse arguments - should always succeed since we validate during conversation
                        # but keep try-except as safety net
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            # This shouldn't happen since we validate and retry during conversation,
                            # but if it does, log warning and use empty dict
                            logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    # so the format is consistent for training data
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })
                    
                    # Collect all subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # Format tool response with XML tags
                        tool_response = "<tool_response>\n"
                        
                        # Try to parse tool content as JSON if it looks like JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # Keep as string if not valid JSON
                        
                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": tool_name,
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    # Add all tool responses as a single message
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1  # Skip the tool messages we just processed
                
                else:
                    # Regular assistant message without tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)
                    
                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
                    })
            
            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })
            
            i += 1
        
        return trajectory

    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """
        Save conversation trajectory to JSONL file.
        
        Args:
            messages (List[Dict]): Complete message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
        """
        if not self.save_trajectories:
            return
        
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)

    @staticmethod
    def _summarize_api_error(error: Exception) -> str:
        """Extract a human-readable one-liner from an API error.

        Handles Cloudflare HTML error pages (502, 503, etc.) by pulling the
        <title> tag instead of dumping raw HTML.  Falls back to a truncated
        str(error) for everything else.
        """
        raw = str(error)

        # Cloudflare / proxy HTML pages: grab the <title> for a clean summary
        if "<!DOCTYPE" in raw or "<html" in raw:
            m = re.search(r"<title[^>]*>([^<]+)</title>", raw, re.IGNORECASE)
            title = m.group(1).strip() if m else "HTML error page (title not found)"
            # Also grab Cloudflare Ray ID if present
            ray = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
            ray_id = ray.group(1).strip() if ray else None
            status_code = getattr(error, "status_code", None)
            parts = []
            if status_code:
                parts.append(f"HTTP {status_code}")
            parts.append(title)
            if ray_id:
                parts.append(f"Ray {ray_id}")
            return " — ".join(parts)

        # JSON body errors from OpenAI/Anthropic SDKs
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                return f"{prefix}{msg[:300]}"

        # Fallback: truncate the raw string but give more room than 200 chars
        status_code = getattr(error, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return f"{prefix}{raw[:500]}"

    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def _clean_error_message(self, error_msg: str) -> str:
        """
        Clean up error messages for user display, removing HTML content and truncating.
        
        Args:
            error_msg: Raw error message from API or exception
            
        Returns:
            Clean, user-friendly error message
        """
        if not error_msg:
            return "Unknown error"
            
        # Remove HTML content (common with CloudFlare and gateway error pages)
        if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
            return "Service temporarily unavailable (HTML error page returned)"
            
        # Remove newlines and excessive whitespace
        cleaned = ' '.join(error_msg.split())
        
        # Truncate if too long
        if len(cleaned) > 150:
            cleaned = cleaned[:150] + "..."
            
        return cleaned

    @staticmethod
    def _extract_api_error_context(error: Exception) -> Dict[str, Any]:
        """Extract structured rate-limit details from provider errors."""
        context: Dict[str, Any] = {}

        body = getattr(error, "body", None)
        payload = None
        if isinstance(body, dict):
            payload = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(payload, dict):
            reason = payload.get("code") or payload.get("error")
            if isinstance(reason, str) and reason.strip():
                context["reason"] = reason.strip()
            message = payload.get("message") or payload.get("error_description")
            if isinstance(message, str) and message.strip():
                context["message"] = message.strip()
            for key in ("resets_at", "reset_at"):
                value = payload.get(key)
                if value not in {None, ""}:
                    context["reset_at"] = value
                    break
            retry_after = payload.get("retry_after")
            if retry_after not in {None, ""} and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass
            ratelimit_reset = headers.get("x-ratelimit-reset")
            if ratelimit_reset and "reset_at" not in context:
                context["reset_at"] = ratelimit_reset

        if "message" not in context:
            raw_message = str(error).strip()
            if raw_message:
                context["message"] = raw_message[:500]

        if "reset_at" not in context:
            message = context.get("message") or ""
            if isinstance(message, str):
                delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
                if delay_match:
                    value = float(delay_match.group(1))
                    seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                    context["reset_at"] = time.time() + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

        return context

    def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[Dict[str, Any]]:
        """Token buckets for ``post_api_request`` plugins (no raw ``response`` object)."""
        if response is None:
            return None
        raw_usage = getattr(response, "usage", None)
        if not raw_usage:
            return None
        from dataclasses import asdict

        cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
        summary = asdict(cu)
        summary.pop("raw_usage", None)
        summary["prompt_tokens"] = cu.prompt_tokens
        summary["total_tokens"] = cu.total_tokens
        return summary

    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        Dump a debug-friendly HTTP request record for the active inference API.

        Captures the request body from api_kwargs (excluding transport-only keys
        like timeout). Intended for debugging provider-side 4xx failures where
        retries are not useful.
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(response_obj, "status_code", None)
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        Save the full raw session to a JSON file.

        Stores every message exactly as the agent sees it: user messages,
        assistant messages (with reasoning, finish_reason, tool_calls),
        tool responses (with tool_call_id, tool_name), and injected system
        messages (compression summaries, todo snapshots, etc.).

        REASONING_SCRATCHPAD tags are converted to <think> blocks for consistency.
        Overwritten after each turn so it always reflects the latest state.
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # Clean assistant content for session logs
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            # Guard: never overwrite a larger session log with fewer messages.
            # This protects against data loss when --resume loads a session whose
            # messages weren't fully written to SQLite — the resumed agent starts
            # with partial history and would otherwise clobber the full JSON log.
            if self.session_log_file.exists():
                try:
                    existing = json.loads(self.session_log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # corrupted existing file — allow the overwrite

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")

    def interrupt(self, message: str = None) -> None:
        """
        Request the agent to interrupt its current tool-calling loop.
        
        Call this from another thread (e.g., input handler, message receiver)
        to gracefully stop the agent and process a new message.
        
        Also signals long-running tool executions (e.g. terminal commands)
        to terminate early, so the agent can respond immediately.
        
        Args:
            message: Optional new message that triggered the interrupt.
                     If provided, the agent will include this in its response context.
        
        Example (CLI):
            # In a separate input thread:
            if user_typed_something:
                agent.interrupt(user_input)
        
        Example (Messaging):
            # When new message arrives for active session:
            if session_has_running_agent:
                running_agent.interrupt(new_message.text)
        """
        self._interrupt_requested = True
        self._interrupt_message = message
        # Signal all tools to abort any in-flight operations immediately.
        # Scope the interrupt to this agent's execution thread so other
        # agents running in the same process (gateway) are not affected.
        if self._execution_thread_id is not None:
            _set_interrupt(True, self._execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            # The interrupt arrived before run_conversation() finished
            # binding the agent to its execution thread. Defer the tool-level
            # interrupt signal until startup completes instead of targeting
            # the caller thread by mistake.
            self._interrupt_thread_signal_pending = True
        # Fan out to concurrent-tool worker threads.  Those workers run tools
        # on their own tids (ThreadPoolExecutor workers), so `is_interrupted()`
        # inside a tool only sees an interrupt when their specific tid is in
        # the `_interrupted_threads` set.  Without this propagation, an
        # already-running concurrent tool (e.g. a terminal command hung on
        # network I/O) never notices the interrupt and has to run to its own
        # timeout.  See `_run_tool` for the matching entry/exit bookkeeping.
        # `getattr` fallback covers test stubs that build AIAgent via
        # object.__new__ and skip __init__.
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(True, _wtid)
                except Exception:
                    pass
        # Propagate interrupt to any running child agents (subagent delegation)
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))

    def clear_interrupt(self) -> None:
        """Clear any pending interrupt request and the per-thread tool interrupt signal."""
        self._interrupt_requested = False
        self._interrupt_message = None
        self._interrupt_thread_signal_pending = False
        if self._execution_thread_id is not None:
            _set_interrupt(False, self._execution_thread_id)
        # Also clear any concurrent-tool worker thread bits.  Tracked
        # workers normally clear their own bit on exit, but an explicit
        # clear here guarantees no stale interrupt can survive a turn
        # boundary and fire on a subsequent, unrelated tool call that
        # happens to get scheduled onto the same recycled worker tid.
        # `getattr` fallback covers test stubs that build AIAgent via
        # object.__new__ and skip __init__.
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(False, _wtid)
                except Exception:
                    pass
        # A hard interrupt supersedes any pending /steer — the steer was
        # meant for the agent's next tool-call iteration, which will no
        # longer happen. Drop it instead of surprising the user with a
        # late injection on the post-interrupt turn.
        _steer_lock = getattr(self, "_pending_steer_lock", None)
        if _steer_lock is not None:
            with _steer_lock:
                self._pending_steer = None

    def steer(self, text: str) -> bool:
        """
        Inject a user message into the next tool result without interrupting.

        Unlike interrupt(), this does NOT stop the current tool call. The
        text is stashed and the agent loop appends it to the LAST tool
        result's content once the current tool batch finishes. The model
        sees the steer as part of the tool output on its next iteration.

        Thread-safe: callable from gateway/CLI/TUI threads. Multiple calls
        before the drain point concatenate with newlines.

        Args:
            text: The user text to inject. Empty strings are ignored.

        Returns:
            True if the steer was accepted, False if the text was empty.
        """
        if not text or not text.strip():
            return False
        cleaned = text.strip()
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            # Test stubs that built AIAgent via object.__new__ skip __init__.
            # Fall back to direct attribute set; no concurrent callers expected
            # in those stubs.
            existing = getattr(self, "_pending_steer", None)
            self._pending_steer = (existing + "\n" + cleaned) if existing else cleaned
            return True
        with _lock:
            if self._pending_steer:
                self._pending_steer = self._pending_steer + "\n" + cleaned
            else:
                self._pending_steer = cleaned
        return True

    def _drain_pending_steer(self) -> Optional[str]:
        """Return the pending steer text (if any) and clear the slot.

        Safe to call from the agent execution thread after appending tool
        results. Returns None when no steer is pending.
        """
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            text = getattr(self, "_pending_steer", None)
            self._pending_steer = None
            return text
        with _lock:
            text = self._pending_steer
            self._pending_steer = None
        return text

    def _record_file_mutation_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        is_error: bool,
    ) -> None:
        """Record a ``write_file`` / ``patch`` outcome for the turn-end verifier.

        On failure, store ``{path: {error_preview, tool}}`` entries.  On
        success, remove any prior failure entries for the same paths (the
        model recovered within the turn).  Silently no-ops if the per-turn
        state dict hasn't been initialised yet (e.g. a tool dispatched
        outside ``run_conversation``).
        """
        if tool_name not in _FILE_MUTATING_TOOLS:
            return
        state = getattr(self, "_turn_failed_file_mutations", None)
        if state is None:
            return
        targets = _extract_file_mutation_targets(tool_name, args)
        if not targets:
            return
        if is_error:
            preview = _extract_error_preview(result)
            for path in targets:
                # Keep the FIRST error we saw for a given path unless we
                # later see success.  A repeated failure with a different
                # message shouldn't silently overwrite the original.
                if path not in state:
                    state[path] = {
                        "tool": tool_name,
                        "error_preview": preview,
                    }
        else:
            for path in targets:
                state.pop(path, None)

    def _file_mutation_verifier_enabled(self) -> bool:
        """Check whether the per-turn file-mutation verifier footer is on.

        Config path: ``display.file_mutation_verifier`` (bool, default True).
        ``HERMES_FILE_MUTATION_VERIFIER`` env var overrides config.  Exposed
        as a method so tests can patch a single seam without reaching into
        the private ``_turn_failed_file_mutations`` state dict.
        """
        try:
            import os as _os
            env = _os.environ.get("HERMES_FILE_MUTATION_VERIFIER")
            if env is not None:
                return env.strip().lower() not in ("0", "false", "no", "off")
            # Read from the persisted config.yaml so gateway and CLI share
            # the same setting.  Import lazily to avoid a startup-time cycle.
            try:
                from hermes_cli.config import load_config as _load_config
                _cfg = _load_config() or {}
            except Exception:
                _cfg = {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "file_mutation_verifier" in _display:
                return bool(_display.get("file_mutation_verifier"))
        except Exception:
            pass
        return True  # safe default: verifier on

    @staticmethod
    def _format_file_mutation_failure_footer(failed: Dict[str, Dict[str, Any]]) -> str:
        """Render the per-turn failed-mutation dict as a user-facing footer.

        Displays up to 10 paths with their first error preview, then a
        count of any additional failures.  Returns an empty string when
        the dict is empty so callers can concatenate unconditionally.
        """
        if not failed:
            return ""
        lines = [
            "⚠️ File-mutation verifier: "
            f"{len(failed)} file(s) were NOT modified this turn despite any "
            "wording above that may suggest otherwise. Run `git status` or "
            "`read_file` to confirm."
        ]
        shown = 0
        for path, info in failed.items():
            if shown >= 10:
                break
            preview = (info.get("error_preview") or "").strip()
            tool = info.get("tool") or "patch"
            if preview:
                lines.append(f"  • {path} — [{tool}] {preview}")
            else:
                lines.append(f"  • {path} — [{tool}] failed")
            shown += 1
        remaining = len(failed) - shown
        if remaining > 0:
            lines.append(f"  • … and {remaining} more")
        return "\n".join(lines)

    def _apply_pending_steer_to_tool_results(self, messages: list, num_tool_msgs: int) -> None:
        """Append any pending /steer text to the last tool result in this turn.

        Called at the end of a tool-call batch, before the next API call.
        The steer is appended to the last ``role:"tool"`` message's content
        with a clear marker so the model understands it came from the user
        and NOT from the tool itself. Role alternation is preserved —
        nothing new is inserted, we only modify existing content.

        Args:
            messages: The running messages list.
            num_tool_msgs: Number of tool results appended in this batch;
                used to locate the tail slice safely.
        """
        if num_tool_msgs <= 0 or not messages:
            return
        steer_text = self._drain_pending_steer()
        if not steer_text:
            return
        # Find the last tool-role message in the recent tail. Skipping
        # non-tool messages defends against future code appending
        # something else at the boundary.
        target_idx = None
        for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
            msg = messages[j]
            if isinstance(msg, dict) and msg.get("role") == "tool":
                target_idx = j
                break
        if target_idx is None:
            # No tool result in this batch (e.g. all skipped by interrupt);
            # put the steer back so the caller's fallback path can deliver
            # it as a normal next-turn user message.
            _lock = getattr(self, "_pending_steer_lock", None)
            if _lock is not None:
                with _lock:
                    if self._pending_steer:
                        self._pending_steer = self._pending_steer + "\n" + steer_text
                    else:
                        self._pending_steer = steer_text
            else:
                existing = getattr(self, "_pending_steer", None)
                self._pending_steer = (existing + "\n" + steer_text) if existing else steer_text
            return
        marker = f"\n\nUser guidance: {steer_text}"
        existing_content = messages[target_idx].get("content", "")
        if not isinstance(existing_content, str):
            # Anthropic multimodal content blocks — preserve them and append
            # a text block at the end.
            try:
                blocks = list(existing_content) if existing_content else []
                blocks.append({"type": "text", "text": marker.lstrip()})
                messages[target_idx]["content"] = blocks
            except Exception:
                # Fall back to string replacement if content shape is unexpected.
                messages[target_idx]["content"] = f"{existing_content}{marker}"
        else:
            messages[target_idx]["content"] = existing_content + marker
        logger.info(
            "Delivered /steer to agent after tool batch (%d chars): %s",
            len(steer_text),
            steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
        )

    def _touch_activity(self, desc: str) -> None:
        """Update the last-activity timestamp and description (thread-safe)."""
        self._last_activity_ts = time.time()
        self._last_activity_desc = desc

    def _capture_rate_limits(self, http_response: Any) -> None:
        """Parse x-ratelimit-* headers from an HTTP response and cache the state.

        Called after each streaming API call.  The httpx Response object is
        available on the OpenAI SDK Stream via ``stream.response``.
        """
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        try:
            from agent.rate_limit_tracker import parse_rate_limit_headers
            state = parse_rate_limit_headers(headers, provider=self.provider)
            if state is not None:
                self._rate_limit_state = state
        except Exception:
            pass  # Never let header parsing break the agent loop

    def get_rate_limit_state(self):
        """Return the last captured RateLimitState, or None."""
        return self._rate_limit_state

    def _check_openrouter_cache_status(self, http_response: Any) -> None:
        """Read X-OpenRouter-Cache-Status from response headers and log it.

        Increments ``_or_cache_hits`` on HIT so callers can report savings.
        """
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        try:
            status = headers.get("x-openrouter-cache-status")
            if not status:
                return
            if status.upper() == "HIT":
                self._or_cache_hits += 1
                logger.info("OpenRouter response cache HIT (total: %d)", self._or_cache_hits)
            else:
                logger.debug("OpenRouter response cache %s", status.upper())
        except Exception:
            pass  # Never let header parsing break the agent loop

    def get_activity_summary(self) -> dict:
        """Return a snapshot of the agent's current activity for diagnostics.

        Called by the gateway timeout handler to report what the agent was doing
        when it was killed, and by the periodic "still working" notifications.
        """
        elapsed = time.time() - self._last_activity_ts
        return {
            "last_activity_ts": self._last_activity_ts,
            "last_activity_desc": self._last_activity_desc,
            "seconds_since_activity": round(elapsed, 1),
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "budget_used": self.iteration_budget.used,
            "budget_max": self.iteration_budget.max_total,
        }

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down the memory provider and context engine — call at actual session boundaries.

        This calls on_session_end() then shutdown_all() on the memory
        manager, and on_session_end() on the context engine.
        NOT called per-turn — only at CLI exit, /reset, gateway
        session expiry, etc.
        """
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
            try:
                self._memory_manager.shutdown_all()
            except Exception:
                pass
        # Notify context engine of session end (flush DAG, close DBs, etc.)
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass

    def commit_memory_session(self, messages: list = None) -> None:
        """Trigger end-of-session extraction without tearing providers down.
        Called when session_id rotates (e.g. /new, context compression);
        providers keep their state and continue running under the old
        session_id — they just flush pending extraction now."""
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
        # Notify context engine of session end too — same lifecycle moment as
        # the memory manager's on_session_end. Without this, engines that
        # accumulate per-session state (DAGs, summaries) leak that state from
        # the rotated-out session into whatever comes next under the same
        # compressor instance. Mirrors the call in shutdown_memory_provider().
        # See issue #22394.
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass

    def _sync_external_memory_for_turn(
        self,
        *,
        original_user_message: Any,
        final_response: Any,
        interrupted: bool,
    ) -> None:
        """Mirror a completed turn into external memory providers.

        Called at the end of ``run_conversation`` with the cleaned user
        message (``original_user_message``) and the finalised assistant
        response.  The external memory backend gets both ``sync_all`` (to
        persist the exchange) and ``queue_prefetch_all`` (to start
        warming context for the next turn) in one shot.

        Uses ``original_user_message`` rather than ``user_message``
        because the latter may carry injected skill content that bloats
        or breaks provider queries.

        Interrupted turns are skipped entirely (#15218).  A partial
        assistant output, an aborted tool chain, or a mid-stream reset
        is not durable conversational truth — mirroring it into an
        external memory backend pollutes future recall with state the
        user never saw completed.  The prefetch is gated on the same
        flag: the user's next message is almost certainly a retry of
        the same intent, and a prefetch keyed on the interrupted turn
        would fire against stale context.

        Normal completed turns still sync as before.  The whole body is
        wrapped in ``try/except Exception`` because external memory
        providers are strictly best-effort — a misconfigured or offline
        backend must not block the user from seeing their response.
        """
        if interrupted:
            return
        if not (self._memory_manager and final_response and original_user_message):
            return
        try:
            self._memory_manager.sync_all(
                original_user_message, final_response,
                session_id=self.session_id or "",
            )
            self._memory_manager.queue_prefetch_all(
                original_user_message,
                session_id=self.session_id or "",
            )
        except Exception:
            pass

    def release_clients(self) -> None:
        """Release LLM client resources WITHOUT tearing down session tool state.

        Used by the gateway when evicting this agent from _agent_cache for
        memory-management reasons (LRU cap or idle TTL) — the session may
        resume at any time with a freshly-built AIAgent that reuses the
        same task_id / session_id, so we must NOT kill:
          - process_registry entries for task_id (user's bg shells)
          - terminal sandbox for task_id (cwd, env, shell state)
          - browser daemon for task_id (open tabs, cookies)
          - memory provider (has its own lifecycle; keeps running)

        We DO close:
          - OpenAI/httpx client pool (big chunk of held memory + sockets;
            the rebuilt agent gets a fresh client anyway)
          - Active child subagents (per-turn artefacts; safe to drop)

        Safe to call multiple times.  Distinct from close() — which is the
        hard teardown for actual session boundaries (/new, /reset, session
        expiry).
        """
        # Close active child agents (per-turn; no cross-turn persistence).
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.release_clients()
                except Exception:
                    # Fall back to full close on children; they're per-turn.
                    try:
                        child.close()
                    except Exception:
                        pass
        except Exception:
            pass

        # Close the OpenAI/httpx client to release sockets immediately.
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="cache_evict", shared=True)
                self.client = None
        except Exception:
            pass

    def close(self) -> None:
        """Release all resources held by this agent instance.

        Cleans up subprocess resources that would otherwise become orphans:
        - Background processes tracked in ProcessRegistry
        - Terminal sandbox environments
        - Browser daemon sessions
        - Active child agents (subagent delegation)
        - OpenAI/httpx client connections

        Safe to call multiple times (idempotent).  Each cleanup step is
        independently guarded so a failure in one does not prevent the rest.
        """
        task_id = getattr(self, "session_id", None) or ""

        # 1. Kill background processes for this task
        try:
            from tools.process_registry import process_registry
            process_registry.kill_all(task_id=task_id)
        except Exception:
            pass

        # 2. Clean terminal sandbox environments
        try:
            cleanup_vm(task_id)
        except Exception:
            pass

        # 3. Clean browser daemon sessions
        try:
            cleanup_browser(task_id)
        except Exception:
            pass

        # 4. Close active child agents
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.close()
                except Exception:
                    pass
        except Exception:
            pass

        # 5. Close the OpenAI/httpx client
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="agent_close", shared=True)
                self.client = None
        except Exception:
            pass

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        Recover todo state from conversation history.
        
        The gateway creates a fresh AIAgent per message, so the in-memory
        TodoStore is empty. We scan the history for the most recent todo
        tool response and replay it to reconstruct the state.
        """
        # Walk history backwards to find the most recent todo tool response
        last_todo_response = None
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Quick check: todo responses contain "todos" key
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if last_todo_response:
            # Replay the items into the store (replace mode)
            self._todo_store.write(last_todo_response, merge=False)
            if not self.quiet_mode:
                self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)

    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested










    def _build_system_prompt_parts(self, system_message: str = None) -> Dict[str, str]:
        """Assemble the system prompt as three ordered parts.

        Returns a dict with three keys:
          * ``stable``   — identity, tool guidance, skills prompt,
            environment hints, platform hints, model-family operational
            guidance.
          * ``context``  — context files (AGENTS.md, .cursorrules, etc.)
            and caller-supplied system_message.
          * ``volatile`` — memory snapshot, user profile, external
            memory provider block, timestamp line.

        Joined into a single string by ``_build_system_prompt`` and
        cached on ``_cached_system_prompt`` for the lifetime of the
        AIAgent.  Hermes never re-renders parts of this string mid-
        session — that's the only way to keep upstream prompt caches
        warm across turns.
        """
        # ── Stable tier ────────────────────────────────────────────────
        stable_parts: List[str] = []

        # Try SOUL.md as primary identity unless the caller explicitly skipped it.
        # Some execution modes (cron) still want HERMES_HOME persona while keeping
        # cwd project instructions disabled.
        _soul_loaded = False
        if self.load_soul_identity or not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                stable_parts.append(_soul_content)
                _soul_loaded = True

        if not _soul_loaded:
            # Fallback to hardcoded identity
            stable_parts.append(DEFAULT_AGENT_IDENTITY)

        # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
        stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)

        # Tool-aware behavioral guidance: only inject when the tools are loaded
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        # Kanban worker/orchestrator lifecycle — only present when the
        # dispatcher spawned this process (kanban_show check_fn gates on
        # HERMES_KANBAN_TASK env var). Normal chat sessions never see
        # this block.
        if "kanban_show" in self.valid_tool_names:
            tool_guidance.append(KANBAN_GUIDANCE)
        if tool_guidance:
            stable_parts.append(" ".join(tool_guidance))

        # Computer-use (macOS) — goes in as its own block rather than being
        # merged into tool_guidance because the content is multi-paragraph.
        if "computer_use" in self.valid_tool_names:
            from agent.prompt_builder import COMPUTER_USE_GUIDANCE
            stable_parts.append(COMPUTER_USE_GUIDANCE)

        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            stable_parts.append(nous_subscription_prompt)
        # Tool-use enforcement: tells the model to actually call tools instead
        # of describing intended actions.  Controlled by config.yaml
        # agent.tool_use_enforcement:
        #   "auto" (default) — matches TOOL_USE_ENFORCEMENT_MODELS
        #   true  — always inject (all models)
        #   false — never inject
        #   list  — custom model-name substrings to match
        if self.valid_tool_names:
            _enforce = self._tool_use_enforcement
            _inject = False
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in {"true", "always", "yes", "on"}):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in {"false", "never", "no", "off"}):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (self.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                # "auto" or any unrecognised value — use hardcoded defaults
                model_lower = (self.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                stable_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (self.model or "").lower()
                # Google model operational guidance (conciseness, absolute
                # paths, parallel tool calls, verify-before-edit, etc.)
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    stable_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex execution discipline (tool persistence,
                # prerequisite checks, verification, anti-hallucination).
                if "gpt" in _model_lower or "codex" in _model_lower:
                    stable_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in self.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            stable_parts.append(skills_prompt)

        # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
        # of the requested model. Inject explicit model identity into the system prompt
        # so the agent can correctly report which model it is (workaround for API bug).
        # Stable for the lifetime of an agent instance — model and provider are fixed
        # at construction time.
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            stable_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        # Environment hints (WSL, Termux, etc.) — tell the agent about the
        # execution environment so it can translate paths and adapt behavior.
        # Stable for the lifetime of the process.
        _env_hints = build_environment_hints()
        if _env_hints:
            stable_parts.append(_env_hints)

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            stable_parts.append(PLATFORM_HINTS[platform_key])
        elif platform_key:
            # Check plugin registry for platform-specific LLM guidance
            try:
                from gateway.platform_registry import platform_registry
                _entry = platform_registry.get(platform_key)
                if _entry and _entry.platform_hint:
                    stable_parts.append(_entry.platform_hint)
            except Exception:
                pass

        # ── Context tier (cwd-dependent, may change between sessions) ─
        context_parts: List[str] = []

        # Note: ephemeral_system_prompt is NOT included here. It's injected at
        # API-call time only so it stays out of the cached/stored system prompt.
        if system_message is not None:
            context_parts.append(system_message)

        if not self.skip_context_files:
            # Use TERMINAL_CWD for context file discovery when set (gateway
            # mode).  The gateway process runs from the hermes-agent install
            # dir, so os.getcwd() would pick up the repo's AGENTS.md and
            # other dev files — inflating token usage by ~10k for no benefit.
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                context_parts.append(context_files_prompt)

        # ── Volatile tier (changes per session/turn — never cached) ───
        volatile_parts: List[str] = []

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    volatile_parts.append(mem_block)
            # USER.md is always included when enabled.
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    volatile_parts.append(user_block)

        # External memory provider system prompt block (additive to built-in)
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    volatile_parts.append(_ext_mem_block)
            except Exception:
                pass

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        volatile_parts.append(timestamp_line)

        return {
            "stable":   "\n\n".join(p.strip() for p in stable_parts   if p and p.strip()),
            "context":  "\n\n".join(p.strip() for p in context_parts  if p and p.strip()),
            "volatile": "\n\n".join(p.strip() for p in volatile_parts if p and p.strip()),
        }

    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        Assemble the full system prompt from all layers.

        Called once per session (cached on self._cached_system_prompt) and only
        rebuilt after context compression events. This ensures the system prompt
        is stable across all turns in a session, maximizing prefix cache hits.

        Layers are ordered cache-friendly: stable identity/guidance first,
        then session-stable context files, then per-call volatile content
        (memory, USER profile, timestamp).  The whole string is treated as
        one cached block — Hermes never rebuilds or reinjects parts of it
        mid-session, which is the only way to keep upstream prompt caches
        warm across turns.
        """
        parts = self._build_system_prompt_parts(system_message=system_message)
        joined = "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)
        return joined

    # =========================================================================
    # Pre/post-call guardrails (inspired by PR #1321 — @alireza78a)
    # =========================================================================

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """Extract call ID from a tool_call entry (dict or object)."""
        if isinstance(tc, dict):
            return tc.get("call_id", "") or tc.get("id", "") or ""
        return getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""

    @staticmethod
    def _get_tool_call_name_static(tc) -> str:
        """Extract function name from a tool_call entry (dict or object).

        Gemini's OpenAI-compatibility endpoint requires every `role: tool`
        message to carry the matching function name. OpenAI/Anthropic/ollama
        tolerate its absence, so the field is best-effort: callers fall back
        to "" and the message still works elsewhere.
        """
        if isinstance(tc, dict):
            fn = tc.get("function")
            if isinstance(fn, dict):
                return fn.get("name", "") or ""
            return ""
        fn = getattr(tc, "function", None)
        return getattr(fn, "name", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs before every LLM call.

        Runs unconditionally — not gated on whether the context compressor
        is present — so orphans from session loading or manual message
        manipulation are always caught.
        """
        # --- Role allowlist: drop messages with roles the API won't accept ---
        filtered = []
        for msg in messages:
            role = msg.get("role")
            if role not in AIAgent._VALID_API_ROLES:
                logger.debug(
                    "Pre-call sanitizer: dropping message with invalid role %r",
                    role,
                )
                continue
            filtered.append(msg)
        messages = filtered

        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = AIAgent._get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Drop tool results with no matching assistant call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            logger.debug(
                "Pre-call sanitizer: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. Inject stub results for calls whose result was dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = AIAgent._get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "name": AIAgent._get_tool_call_name_static(tc),
                                "content": "[Result unavailable — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            logger.debug(
                "Pre-call sanitizer: added %d stub tool result(s)",
                len(missing_results),
            )
        return messages

    @staticmethod
    def _is_thinking_only_assistant(msg: Dict[str, Any]) -> bool:
        """Return True if ``msg`` is an assistant turn whose only payload is reasoning.

        "Thinking-only" means the model emitted reasoning (``reasoning`` or
        ``reasoning_content``) but no visible text and no tool_calls. When sent
        back to providers that convert reasoning into thinking blocks (native
        Anthropic, OpenRouter Anthropic, third-party Anthropic-compatible
        gateways), the resulting message has only thinking blocks — which
        Anthropic rejects with HTTP 400 "The final block in an assistant
        message cannot be `thinking`."

        Symmetric with Claude Code's ``filterOrphanedThinkingOnlyMessages``
        (src/utils/messages.ts). We drop the whole turn from the API copy
        rather than fabricating stub text — the message log (UI transcript)
        keeps the reasoning block; only the wire copy is cleaned.
        """
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return False
        if msg.get("tool_calls"):
            return False
        # Does it have any actual output?
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                return False
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if block:  # non-empty non-dict string etc.
                        return False
                    continue
                btype = block.get("type")
                if btype in {"thinking", "redacted_thinking"}:
                    continue
                if btype == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return False
                    continue
                # tool_use, image, document, etc. — real payload
                return False
        elif content is not None and content != "":
            return False
        # Content is empty-ish. Is there reasoning to make it thinking-only?
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return True
        # reasoning_details list form
        rd = msg.get("reasoning_details")
        if isinstance(rd, list) and rd:
            return True
        return False

    @staticmethod
    def _drop_thinking_only_and_merge_users(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop thinking-only assistant turns; merge any adjacent user messages left behind.

        Runs on the per-call ``api_messages`` copy only. The stored
        conversation history (``self.messages``) is never mutated, so the
        user still sees the thinking block in the CLI/gateway transcript and
        session persistence keeps the full trace. Only the wire copy sent to
        the provider is cleaned.

        Why drop-and-merge rather than inject stub text:
        - Fabricating ``"."`` / ``"(continued)"`` text lies in the history
          and makes future turns see model output the model didn't emit.
        - Dropping the turn preserves honesty; merging adjacent user messages
          preserves the provider's role-alternation invariant.
        - This is the pattern used by Claude Code's ``normalizeMessagesForAPI``
          (filterOrphanedThinkingOnlyMessages + mergeAdjacentUserMessages).
        """
        if not messages:
            return messages

        # Pass 1: drop thinking-only assistant turns.
        kept = [m for m in messages if not AIAgent._is_thinking_only_assistant(m)]
        dropped = len(messages) - len(kept)
        if dropped == 0:
            return messages

        # Pass 2: merge any newly-adjacent user messages.
        merged: List[Dict[str, Any]] = []
        merges = 0
        for m in kept:
            prev = merged[-1] if merged else None
            if (
                prev is not None
                and prev.get("role") == "user"
                and m.get("role") == "user"
            ):
                prev_content = prev.get("content", "")
                cur_content = m.get("content", "")
                # Work on a copy of ``prev`` so the caller's input dicts are
                # never mutated. ``_sanitize_api_messages`` upstream already
                # hands us per-call copies, but staying pure here means we
                # can be called safely from anywhere (tests, other loops).
                prev_copy = dict(prev)
                # Only string-content merge is meaningful for role-alternation
                # purposes. If either side is a list (multimodal), append as a
                # separate block rather than collapsing.
                if isinstance(prev_content, str) and isinstance(cur_content, str):
                    sep = "\n\n" if prev_content and cur_content else ""
                    prev_copy["content"] = prev_content + sep + cur_content
                elif isinstance(prev_content, list) and isinstance(cur_content, list):
                    prev_copy["content"] = list(prev_content) + list(cur_content)
                elif isinstance(prev_content, list) and isinstance(cur_content, str):
                    if cur_content:
                        prev_copy["content"] = list(prev_content) + [
                            {"type": "text", "text": cur_content}
                        ]
                    else:
                        prev_copy["content"] = list(prev_content)
                elif isinstance(prev_content, str) and isinstance(cur_content, list):
                    new_blocks: List[Dict[str, Any]] = []
                    if prev_content:
                        new_blocks.append({"type": "text", "text": prev_content})
                    new_blocks.extend(cur_content)
                    prev_copy["content"] = new_blocks
                else:
                    # Unknown content shape — fall back to appending separately
                    # (violates alternation, but safer than raising in a hot path).
                    merged.append(m)
                    continue
                merged[-1] = prev_copy
                merges += 1
            else:
                merged.append(m)

        logger.debug(
            "Pre-call sanitizer: dropped %d thinking-only assistant turn(s), "
            "merged %d adjacent user message(s)",
            dropped,
            merges,
        )
        return merged

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """Truncate excess delegate_task calls to max_concurrent_children.

        The delegate_tool caps the task list inside a single call, but the
        model can emit multiple separate delegate_task tool_calls in one
        turn.  This truncates the excess, preserving all non-delegate calls.

        Returns the original list if no truncation was needed.
        """
        from tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < max_children:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "Truncated %d excess delegate_task call(s) to enforce "
            "max_concurrent_children=%d limit",
            delegate_count - max_children, max_children,
        )
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """Remove duplicate (tool_name, arguments) pairs within a single turn.

        Only the first occurrence of each unique pair is kept.
        Returns the original list if no duplicates were found.
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    def _repair_tool_call(self, tool_name: str) -> str | None:
        """Attempt to repair a mismatched tool name before aborting.

        Models sometimes emit variants of a tool name that differ only
        in casing, separators, or class-like suffixes. Normalize
        aggressively before falling back to fuzzy match:

        1. Lowercase direct match.
        2. Lowercase + hyphens/spaces -> underscores.
        3. CamelCase -> snake_case (TodoTool -> todo_tool).
        4. Strip trailing ``_tool`` / ``-tool`` / ``tool`` suffix that
           Claude-style models sometimes tack on (TodoTool_tool ->
           TodoTool -> Todo -> todo). Applied twice so double-tacked
           suffixes like ``TodoTool_tool`` reduce all the way.
        5. Fuzzy match (difflib, cutoff=0.7).

        See #14784 for the original reports (TodoTool_tool, Patch_tool,
        BrowserClick_tool were all returning "Unknown tool" before).

        Returns the repaired name if found in valid_tool_names, else None.
        """
        import re
        from difflib import get_close_matches

        if not tool_name:
            return None

        def _norm(s: str) -> str:
            return s.lower().replace("-", "_").replace(" ", "_")

        def _camel_snake(s: str) -> str:
            return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

        def _strip_tool_suffix(s: str) -> str | None:
            lc = s.lower()
            for suffix in ("_tool", "-tool", "tool"):
                if lc.endswith(suffix):
                    return s[: -len(suffix)].rstrip("_-")
            return None

        # Cheap fast-paths first — these cover the common case.
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered
        normalized = _norm(tool_name)
        if normalized in self.valid_tool_names:
            return normalized

        # Build the full candidate set for class-like emissions.
        cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
        # Strip trailing tool-suffix up to twice — TodoTool_tool needs it.
        for _ in range(2):
            extra: set[str] = set()
            for c in cands:
                stripped = _strip_tool_suffix(c)
                if stripped:
                    extra.add(stripped)
                    extra.add(_norm(stripped))
                    extra.add(_camel_snake(stripped))
            cands |= extra

        for c in cands:
            if c and c in self.valid_tool_names:
                return c

        # Fuzzy match as last resort.
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None

    def _invalidate_system_prompt(self):
        """
        Invalidate the cached system prompt, forcing a rebuild on the next turn.
        
        Called after context compression events. Also reloads memory from disk
        so the rebuilt prompt captures any writes from this session.
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()

    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        """Generate a deterministic call_id from tool call content.

        Used as a fallback when the API doesn't provide a call_id.
        Deterministic IDs prevent cache invalidation — random UUIDs would
        make every API call's prefix unique, breaking OpenAI's prompt cache.
        """
        return _codex_deterministic_call_id(fn_name, arguments, index)

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """Split a stored tool id into (call_id, response_item_id)."""
        return _codex_split_responses_tool_id(raw_id)

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """Build a valid Responses `function_call.id` (must start with `fc_`)."""
        return _codex_derive_responses_function_call_id(call_id, response_item_id)

    def _thread_identity(self) -> str:
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """Check if an OpenAI client is closed.

        Handles both property and method forms of is_closed:
        - httpx.Client.is_closed is a bool property
        - openai.OpenAI.is_closed is a method returning bool

        Prior bug: getattr(client, "is_closed", False) returned the bound method,
        which is always truthy, causing unnecessary client recreation on every call.
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # Handle method (openai SDK) vs property (httpx)
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    @staticmethod
    def _build_keepalive_http_client(base_url: str = "") -> Any:
        try:
            import httpx as _httpx
            import socket as _socket

            _sock_opts = [(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)]
            if hasattr(_socket, "TCP_KEEPIDLE"):
                _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE, 30))
                _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL, 10))
                _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, 3))
            elif hasattr(_socket, "TCP_KEEPALIVE"):
                _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPALIVE, 30))
            # When a custom transport is provided, httpx won't auto-read proxy
            # from env vars (allow_env_proxies = trust_env and transport is None).
            # Explicitly read proxy settings while still honoring NO_PROXY for
            # loopback / local endpoints such as a locally hosted sub2api.
            _proxy = _get_proxy_for_base_url(base_url)
            return _httpx.Client(
                transport=_httpx.HTTPTransport(socket_options=_sock_opts),
                proxy=_proxy,
            )
        except Exception:
            return None

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
        # Treat client_kwargs as read-only. Callers pass self._client_kwargs (or shallow
        # copies of it) in; any in-place mutation leaks back into the stored dict and is
        # reused on subsequent requests. #10933 hit this by injecting an httpx.Client
        # transport that was torn down after the first request, so the next request
        # wrapped a closed transport and raised "Cannot send a request, as the client
        # has been closed" on every retry. The revert resolved that specific path; this
        # copy locks the contract so future transport/keepalive work can't reintroduce
        # the same class of bug.
        client_kwargs = dict(client_kwargs)
        _validate_proxy_env_urls()
        _validate_base_url(client_kwargs.get("base_url"))
        if self.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        if self.provider == "google-gemini-cli" or str(client_kwargs.get("base_url", "")).startswith("cloudcode-pa://"):
            from agent.gemini_cloudcode_adapter import GeminiCloudCodeClient

            # Strip OpenAI-specific kwargs the Gemini client doesn't accept
            safe_kwargs = {
                k: v for k, v in client_kwargs.items()
                if k in {"api_key", "base_url", "default_headers", "project_id", "timeout"}
            }
            client = GeminiCloudCodeClient(**safe_kwargs)
            logger.info(
                "Gemini Cloud Code Assist client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        if self.provider == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

            base_url = str(client_kwargs.get("base_url", "") or "")
            if is_native_gemini_base_url(base_url):
                safe_kwargs = {
                    k: v for k, v in client_kwargs.items()
                    if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
                }
                if "http_client" not in safe_kwargs:
                    keepalive_http = self._build_keepalive_http_client(base_url)
                    if keepalive_http is not None:
                        safe_kwargs["http_client"] = keepalive_http
                client = GeminiNativeClient(**safe_kwargs)
                logger.info(
                    "Gemini native client created (%s, shared=%s) %s",
                    reason,
                    shared,
                    self._client_log_context(),
                )
                return client
        # Inject TCP keepalives so the kernel detects dead provider connections
        # instead of letting them sit silently in CLOSE-WAIT (#10324).  Without
        # this, a peer that drops mid-stream leaves the socket in a state where
        # epoll_wait never fires, ``httpx`` read timeout may not trigger, and
        # the agent hangs until manually killed.  Probes after 30s idle, retry
        # every 10s, give up after 3 → dead peer detected within ~60s.
        #
        # Safety against #10933: the ``client_kwargs = dict(client_kwargs)``
        # above means this injection only lands in the local per-call copy,
        # never back into ``self._client_kwargs``.  Each ``_create_openai_client``
        # invocation therefore gets its OWN fresh ``httpx.Client`` whose
        # lifetime is tied to the OpenAI client it is passed to.  When the
        # OpenAI client is closed (rebuild, teardown, credential rotation),
        # the paired ``httpx.Client`` closes with it, and the next call
        # constructs a fresh one — no stale closed transport can be reused.
        # Tests in ``tests/run_agent/test_create_openai_client_reuse.py`` and
        # ``tests/run_agent/test_sequential_chats_live.py`` pin this invariant.
        if "http_client" not in client_kwargs:
            keepalive_http = self._build_keepalive_http_client(client_kwargs.get("base_url", ""))
            if keepalive_http is not None:
                client_kwargs["http_client"] = keepalive_http
        # Uses the module-level `OpenAI` name, resolved lazily on first
        # access via __getattr__ below. Tests patch via `run_agent.OpenAI`.
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """Force-close underlying TCP sockets to prevent CLOSE-WAIT accumulation.

        When a provider drops a connection mid-stream, httpx's ``client.close()``
        performs a graceful shutdown which leaves sockets in CLOSE-WAIT until the
        OS times them out (often minutes).  This method walks the httpx transport
        pool and issues ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()`` to
        force an immediate TCP RST, freeing the file descriptors.

        Returns the number of sockets force-closed.
        """
        import socket as _socket

        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return 0
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return 0
            # httpx uses httpcore connection pools; connections live in
            # _connections (list) or _pool (list) depending on version.
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            for conn in list(connections):
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        # Force-close TCP sockets first to prevent CLOSE-WAIT accumulation,
        # then do the graceful SDK-level close.
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(self._client_kwargs, reason=reason, shared=True)
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _cleanup_dead_connections(self) -> bool:
        """Detect and clean up dead TCP connections on the primary client.

        Inspects the httpx connection pool for sockets in unhealthy states
        (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
        and rebuilds the primary client from scratch.

        Returns True if dead connections were found and cleaned up.
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # Check for connections that are idle but have closed sockets
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # Probe socket health with a non-blocking recv peek
                import socket as _socket
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # No data available — socket is healthy
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    @staticmethod
    def _api_kwargs_have_image_parts(api_kwargs: dict) -> bool:
        """Return True when the outbound request still contains native image parts."""
        if not isinstance(api_kwargs, dict):
            return False
        candidates = []
        messages = api_kwargs.get("messages")
        if isinstance(messages, list):
            candidates.extend(messages)
        # Responses API payloads use `input`; after conversion, image parts can
        # still be present there instead of in `messages`.
        response_input = api_kwargs.get("input")
        if isinstance(response_input, list):
            candidates.extend(response_input)

        def _contains_image(value: Any) -> bool:
            if isinstance(value, dict):
                ptype = value.get("type")
                if ptype in {"image_url", "input_image"}:
                    return True
                return any(_contains_image(v) for v in value.values())
            if isinstance(value, list):
                return any(_contains_image(v) for v in value)
            return False

        return any(_contains_image(item) for item in candidates)

    def _copilot_headers_for_request(self, *, is_vision: bool) -> dict:
        from hermes_cli.copilot_auth import copilot_request_headers

        return copilot_request_headers(is_agent_turn=True, is_vision=is_vision)

    def _create_request_openai_client(self, *, reason: str, api_kwargs: Optional[dict] = None) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        # Per-request OpenAI-wire clients (used by both the non-streaming
        # chat-completions path and the streaming chat-completions path
        # in `_interruptible_api_call`) should not run the SDK's built-in
        # retry loop: the agent's outer loop owns retries with credential
        # rotation, provider fallback, and backoff that the SDK can't
        # see. Leaving SDK retries on (default 2) compounds with our outer
        # retries and lets a single hung provider request stretch to ~3x
        # the per-call timeout before our stale detector reports it.
        # Shared/primary clients and Anthropic / Bedrock paths are
        # unaffected (they don't go through here).
        request_kwargs["max_retries"] = 0
        if (
            base_url_host_matches(str(request_kwargs.get("base_url", "")), "api.githubcopilot.com")
            and self._api_kwargs_have_image_parts(api_kwargs or {})
        ):
            request_kwargs["default_headers"] = self._copilot_headers_for_request(is_vision=True)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        self._close_openai_client(client, reason=reason, shared=False)

    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """Execute one streaming Responses API request and return the final response."""
        import httpx as _httpx

        active_client = client or self._ensure_primary_openai_client(reason="codex_stream_direct")
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        # Accumulate streamed text so we can recover if get_final_response()
        # returns empty output (e.g. chatgpt.com backend-api sends
        # response.incomplete instead of response.completed).
        self._codex_streamed_text_parts: list = []
        for attempt in range(max_stream_retries + 1):
            if self._interrupt_requested:
                raise InterruptedError("Agent interrupted before Codex stream retry")
            collected_output_items: list = []
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        self._touch_activity("receiving stream response")
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # Fire callbacks on text content deltas (suppress during tool calls)
                        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", "")
                            if delta_text:
                                self._codex_streamed_text_parts.append(delta_text)
                            if delta_text and not has_tool_calls:
                                if not first_delta_fired:
                                    first_delta_fired = True
                                    if on_first_delta:
                                        try:
                                            on_first_delta()
                                        except Exception:
                                            pass
                                self._fire_stream_delta(delta_text)
                        # Track tool calls to suppress text streaming
                        elif "function_call" in event_type:
                            has_tool_calls = True
                        # Fire reasoning callbacks
                        elif "reasoning" in event_type and "delta" in event_type:
                            reasoning_text = getattr(event, "delta", "")
                            if reasoning_text:
                                self._fire_reasoning_delta(reasoning_text)
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
                                sum(len(p) for p in self._codex_streamed_text_parts),
                                self._client_log_context(),
                            )
                    final_response = stream.get_final_response()
                    # PATCH: ChatGPT Codex backend streams valid output items
                    # but get_final_response() can return an empty output list.
                    # Backfill from collected items or synthesize from deltas.
                    _out = getattr(final_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            final_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex stream: backfilled %d output items from stream events",
                                len(collected_output_items),
                            )
                        elif self._codex_streamed_text_parts and not has_tool_calls:
                            assembled = "".join(self._codex_streamed_text_parts)
                            final_response.output = [SimpleNamespace(
                                type="message",
                                role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex stream: synthesized output from %d text deltas (%d chars)",
                                len(self._codex_streamed_text_parts), len(assembled),
                            )
                    return final_response
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed (attempt %s/%s); retrying. %s error=%s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                        exc,
                    )
                    continue
                logger.debug(
                    "Codex Responses stream transport failed; falling back to create(stream=True). %s error=%s",
                    self._client_log_context(),
                    exc,
                )
                return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
            except RuntimeError as exc:
                err_text = str(exc)
                missing_completed = "response.completed" in err_text
                if missing_completed and attempt < max_stream_retries:
                    logger.debug(
                        "Responses stream closed before completion (attempt %s/%s); retrying. %s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                    )
                    continue
                if missing_completed:
                    logger.debug(
                        "Responses stream did not emit response.completed; falling back to create(stream=True). %s",
                        self._client_log_context(),
                    )
                    return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
                raise

    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """Fallback path for stream completion edge cases on Codex-style Responses backends."""
        active_client = client or self._ensure_primary_openai_client(reason="codex_create_stream_fallback")
        fallback_kwargs = dict(api_kwargs)
        fallback_kwargs["stream"] = True
        fallback_kwargs = self._get_transport().preflight_kwargs(fallback_kwargs, allow_stream=True)
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
                self._touch_activity("receiving stream response")
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")

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
                    # Backfill empty output from collected stream events
                    _out = getattr(terminal_response, "output", None)
                    if isinstance(_out, list) and not _out:
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

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous requests should not inherit OpenRouter-only attribution headers.
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_copilot_client_credentials(self) -> bool:
        """Refresh Copilot credentials and rebuild the shared OpenAI client.

        Copilot tokens may remain the same string across refreshes (`gh auth token`
        returns a stable OAuth token in many setups). We still rebuild the client
        on 401 so retries recover from stale auth/client state without requiring
        a session restart.
        """
        if self.provider != "copilot":
            return False

        try:
            from hermes_cli.copilot_auth import resolve_copilot_token

            new_token, token_source = resolve_copilot_token()
        except Exception as exc:
            logger.debug("Copilot credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False

        new_token = new_token.strip()

        self.api_key = new_token
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(str(self.base_url or ""))

        if not self._replace_primary_openai_client(reason="copilot_credential_refresh"):
            return False

        logger.info("Copilot credentials refreshed from %s", token_source)
        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # Only refresh credentials for the native Anthropic provider.
        # Other anthropic_messages providers (MiniMax, Alibaba, etc.) use their own keys.
        if self.provider != "anthropic":
            return False
        # Azure endpoints use static API keys — OAuth token rotation doesn't apply.
        # Refreshing would pick up ~/.claude/.credentials.json OAuth token and break auth.
        _base = getattr(self, "_anthropic_base_url", "") or ""
        if "azure.com" in _base:
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(
                new_token,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # Update OAuth flag — token type may have changed (API key ↔ OAuth).
        # Only treat as OAuth on native Anthropic; third-party endpoints using
        # the Anthropic protocol must not trip OAuth paths (#1739 & third-party
        # identity-injection guard).
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token) if self.provider == "anthropic" else False
        return True

    def _apply_client_headers_for_base_url(self, base_url: str) -> None:
        from agent.auxiliary_client import _AI_GATEWAY_HEADERS, build_or_headers

        if base_url_host_matches(base_url, "openrouter.ai"):
            self._client_kwargs["default_headers"] = build_or_headers()
        elif base_url_host_matches(base_url, "ai-gateway.vercel.sh"):
            self._client_kwargs["default_headers"] = dict(_AI_GATEWAY_HEADERS)
        elif base_url_host_matches(base_url, "api.routermint.com"):
            self._client_kwargs["default_headers"] = _routermint_headers()
        elif base_url_host_matches(base_url, "api.githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif base_url_host_matches(base_url, "api.kimi.com"):
            self._client_kwargs["default_headers"] = {"User-Agent": "claude-code/0.1.0"}
        elif base_url_host_matches(base_url, "portal.qwen.ai"):
            self._client_kwargs["default_headers"] = _qwen_portal_headers()
        elif base_url_host_matches(base_url, "chatgpt.com"):
            from agent.auxiliary_client import _codex_cloudflare_headers
            self._client_kwargs["default_headers"] = _codex_cloudflare_headers(
                self._client_kwargs.get("api_key", "")
            )
        else:
            # No URL-specific headers — check profile.default_headers before clearing.
            _ph_headers = None
            try:
                from providers import get_provider_profile as _gpf2
                _ph2 = _gpf2(self.provider)
                if _ph2 and _ph2.default_headers:
                    _ph_headers = dict(_ph2.default_headers)
            except Exception:
                pass
            if _ph_headers:
                self._client_kwargs["default_headers"] = _ph_headers
            else:
                self._client_kwargs.pop("default_headers", None)

    def _swap_credential(self, entry) -> None:
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base
            self._anthropic_client = build_anthropic_client(
                runtime_key, runtime_base,
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
            self._is_anthropic_oauth = _is_oauth_token(runtime_key) if self.provider == "anthropic" else False
            self.api_key = runtime_key
            self.base_url = runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(self.base_url)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        classified_reason: Optional[FailoverReason] = None,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """Attempt credential recovery via pool rotation.

        Returns (recovered, has_retried_429).
        On rate limits: first occurrence retries same credential (sets flag True).
                        second consecutive failure rotates to next credential.
        On billing exhaustion: immediately rotates.
        On auth failures: attempts token refresh before rotating.

        `classified_reason` lets the recovery path honor the structured error
        classifier instead of relying only on raw HTTP codes. This matters for
        providers that surface billing/rate-limit/auth conditions under a
        different status code, such as Anthropic returning HTTP 400 for
        "out of extra usage".
        """
        pool = self._credential_pool
        if pool is None:
            return False, has_retried_429

        effective_reason = classified_reason
        if effective_reason is None:
            if status_code == 402:
                effective_reason = FailoverReason.billing
            elif status_code == 429:
                effective_reason = FailoverReason.rate_limit
            elif status_code in {401, 403}:
                effective_reason = FailoverReason.auth

        if effective_reason == FailoverReason.billing:
            rotate_status = status_code if status_code is not None else 402
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (billing) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, has_retried_429

        if effective_reason == FailoverReason.rate_limit:
            if not has_retried_429:
                return False, True
            rotate_status = status_code if status_code is not None else 429
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (rate limit) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, True

        if effective_reason == FailoverReason.auth:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                logger.info(f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}")
                self._swap_credential(refreshed)
                return True, has_retried_429
            # Refresh failed — rotate to next credential instead of giving up.
            # The failed entry is already marked exhausted by try_refresh_current().
            rotate_status = status_code if status_code is not None else 401
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (auth refresh failed) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False

        return False, has_retried_429

    def _credential_pool_may_recover_rate_limit(self) -> bool:
        """Whether a rate-limit retry should wait for same-provider credentials."""
        pool = self._credential_pool
        if pool is None:
            return False
        if (
            self.provider == "google-gemini-cli"
            or str(getattr(self, "base_url", "")).startswith("cloudcode-pa://")
        ):
            # CloudCode/Gemini quota windows are usually account-level throttles.
            # Prefer the configured fallback immediately instead of waiting out
            # Retry-After while a pooled OAuth credential may still appear usable.
            return False
        return pool.has_available()

    def _anthropic_messages_create(self, api_kwargs: dict):
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        return self._anthropic_client.messages.create(**api_kwargs)

    def _rebuild_anthropic_client(self) -> None:
        """Rebuild the Anthropic client after an interrupt or stale call.

        Handles both direct Anthropic and Bedrock-hosted Anthropic models
        correctly — rebuilding with the Bedrock SDK when provider is bedrock,
        rather than always falling back to build_anthropic_client() which
        requires a direct Anthropic API key.

        Honors ``self._oauth_1m_beta_disabled`` (set by the reactive recovery
        path when an OAuth subscription rejects the 1M-context beta) so the
        rebuilt client carries the reduced beta set.
        """
        _drop_1m = bool(getattr(self, "_oauth_1m_beta_disabled", False))
        if getattr(self, "provider", None) == "bedrock":
            from agent.anthropic_adapter import build_anthropic_bedrock_client
            region = getattr(self, "_bedrock_region", "us-east-1") or "us-east-1"
            self._anthropic_client = build_anthropic_bedrock_client(region)
        else:
            from agent.anthropic_adapter import build_anthropic_client
            self._anthropic_client = build_anthropic_client(
                self._anthropic_api_key,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
                drop_context_1m_beta=_drop_1m,
            )

    def _interruptible_api_call(self, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.

        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.

        Includes a stale-call detector: if no response arrives within the
        configured timeout, the connection is killed and an error raised so
        the main retry loop can try again with backoff / credential rotation /
        provider fallback.
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = self._create_request_openai_client(
                        reason="codex_stream_request",
                        api_kwargs=api_kwargs,
                    )
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                elif self.api_mode == "bedrock_converse":
                    # Bedrock uses boto3 directly — no OpenAI client needed.
                    # normalize_converse_response produces an OpenAI-compatible
                    # SimpleNamespace so the rest of the agent loop can treat
                    # bedrock responses like chat_completions responses.
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        invalidate_runtime_client,
                        is_stale_connection_error,
                        normalize_converse_response,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    try:
                        raw_response = client.converse(**api_kwargs)
                    except Exception as _bedrock_exc:
                        # Evict the cached client on stale-connection failures
                        # so the outer retry loop builds a fresh client/pool.
                        if is_stale_connection_error(_bedrock_exc):
                            invalidate_runtime_client(region)
                        raise
                    result["response"] = normalize_converse_response(raw_response)
                else:
                    request_client_holder["client"] = self._create_request_openai_client(
                        reason="chat_completion_request",
                        api_kwargs=api_kwargs,
                    )
                    result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="request_complete")

        # ── Stale-call timeout (mirrors streaming stale detector) ────────
        # Non-streaming calls return nothing until the full response is
        # ready.  Without this, a hung provider can block for the full
        # httpx timeout (default 1800s) with zero feedback.  The stale
        # detector kills the connection early so the main retry loop can
        # apply richer recovery (credential rotation, provider fallback).
        _stale_timeout = self._compute_non_stream_stale_timeout(
            api_kwargs.get("messages", [])
        )

        _call_start = time.time()
        self._touch_activity("waiting for non-streaming API response")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        _poll_count = 0
        while t.is_alive():
            t.join(timeout=0.3)
            _poll_count += 1

            # Touch activity every ~30s so the gateway's inactivity
            # monitor knows we're alive while waiting for the response.
            if _poll_count % 100 == 0:  # 100 × 0.3s = 30s
                _elapsed = time.time() - _call_start
                self._touch_activity(
                    f"waiting for non-streaming response ({int(_elapsed)}s elapsed)"
                )

            # Stale-call detector: kill the connection if no response
            # arrives within the configured timeout.
            _elapsed = time.time() - _call_start
            if _elapsed > _stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Non-streaming API call stale for %.0fs (threshold %.0fs). "
                    "model=%s context=~%s tokens. Killing connection.",
                    _elapsed, _stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_elapsed)}s "
                    f"(non-streaming, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Aborting call."
                )
                try:
                    if self.api_mode == "anthropic_messages":
                        self._anthropic_client.close()
                        self._rebuild_anthropic_client()
                    else:
                        rc = request_client_holder.get("client")
                        if rc is not None:
                            self._close_request_openai_client(rc, reason="stale_call_kill")
                except Exception:
                    pass
                self._touch_activity(
                    f"stale non-streaming call killed after {int(_elapsed)}s"
                )
                # Wait briefly for the thread to notice the closed connection.
                t.join(timeout=2.0)
                if result["error"] is None and result["response"] is None:
                    result["error"] = TimeoutError(
                        f"Non-streaming API call timed out after {int(_elapsed)}s "
                        f"with no response (threshold: {int(_stale_timeout)}s)"
                    )
                break

            if self._interrupt_requested:
                # Force-close the in-flight worker-local HTTP connection to stop
                # token generation without poisoning the shared client used to
                # seed future retries.
                try:
                    if self.api_mode == "anthropic_messages":
                        self._anthropic_client.close()
                        self._rebuild_anthropic_client()
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Unified streaming API call ─────────────────────────────────────────

    def _reset_stream_delivery_tracking(self) -> None:
        """Reset tracking for text delivered during the current model response."""
        # Flush any benign partial-tag tail held by the think scrubber
        # first (#17924): an innocent '<' at the end of the stream that
        # turned out not to be a tag prefix should reach the UI.  Then
        # flush the context scrubber.  Order matters — the think
        # scrubber's output feeds into the context scrubber's state.
        think_scrubber = getattr(self, "_stream_think_scrubber", None)
        if think_scrubber is not None:
            think_tail = think_scrubber.flush()
            if think_tail:
                # Route the tail through the context scrubber too so a
                # memory-context span straddling the final boundary is
                # still caught.
                ctx_scrubber = getattr(self, "_stream_context_scrubber", None)
                if ctx_scrubber is not None:
                    think_tail = ctx_scrubber.feed(think_tail)
                if think_tail:
                    callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
                    for cb in callbacks:
                        try:
                            cb(think_tail)
                        except Exception:
                            pass
                    self._record_streamed_assistant_text(think_tail)
        # Flush any benign partial-tag tail held by the context scrubber so it
        # reaches the UI before we clear state for the next model call.  If
        # the scrubber is mid-span, flush() drops the orphaned content.
        scrubber = getattr(self, "_stream_context_scrubber", None)
        if scrubber is not None:
            tail = scrubber.flush()
            if tail:
                callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
                for cb in callbacks:
                    try:
                        cb(tail)
                    except Exception:
                        pass
                self._record_streamed_assistant_text(tail)
        self._current_streamed_assistant_text = ""

    def _record_streamed_assistant_text(self, text: str) -> None:
        """Accumulate visible assistant text emitted through stream callbacks."""
        if isinstance(text, str) and text:
            self._current_streamed_assistant_text = (
                getattr(self, "_current_streamed_assistant_text", "") + text
            )

    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _interim_content_was_streamed(self, content: str) -> bool:
        visible_content = self._normalize_interim_visible_text(
            self._strip_think_blocks(content or "")
        )
        if not visible_content:
            return False
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(getattr(self, "_current_streamed_assistant_text", "") or "")
        )
        return bool(streamed) and streamed == visible_content

    def _emit_interim_assistant_message(self, assistant_msg: Dict[str, Any]) -> None:
        """Surface a real mid-turn assistant commentary message to the UI layer."""
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None or not isinstance(assistant_msg, dict):
            return
        content = assistant_msg.get("content")
        visible = self._strip_think_blocks(content or "").strip()
        if not visible or visible == "(empty)":
            return
        already_streamed = self._interim_content_was_streamed(visible)
        try:
            cb(visible, already_streamed=already_streamed)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        # If a tool iteration set the break flag, prepend a single paragraph
        # break before the first real text delta.  This prevents the original
        # problem (text concatenation across tool boundaries) without stacking
        # blank lines when multiple tool iterations run back-to-back.
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
            prepended_break = True
        else:
            prepended_break = False
        if isinstance(text, str):
            # Suppress reasoning/thinking blocks via the stateful
            # scrubber (#17924).  Earlier versions ran _strip_think_blocks
            # per-delta here, which destroyed downstream state machines
            # when a tag was split across deltas (e.g. MiniMax-M2.7
            # sends '<think>' and its content as separate deltas —
            # regex case 2 erased the first delta, so the CLI/gateway
            # state machine never saw the open tag and leaked the
            # reasoning content as regular response text).
            think_scrubber = getattr(self, "_stream_think_scrubber", None)
            if think_scrubber is not None:
                text = think_scrubber.feed(text or "")
            else:
                # Defensive: legacy callers without the scrubber attribute.
                text = self._strip_think_blocks(text or "")
            # Then feed through the stateful context scrubber so memory-context
            # spans split across chunks cannot leak to the UI (#5719).
            scrubber = getattr(self, "_stream_context_scrubber", None)
            if scrubber is not None:
                text = scrubber.feed(text)
            else:
                # Defensive: legacy callers without the scrubber attribute.
                text = sanitize_context(text)
            # Only strip leading newlines on the first delta — mid-stream "\n" is legitimate markdown.
            if not prepended_break and not getattr(
                self, "_current_streamed_assistant_text", ""
            ):
                text = text.lstrip("\n")
        if not text:
            return
        callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
        delivered = False
        for cb in callbacks:
            try:
                cb(text)
                delivered = True
            except Exception:
                pass
        if delivered:
            self._record_streamed_assistant_text(text)

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered."""
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """Notify display layer that the model is generating tool call arguments.

        Fires once per tool name when the streaming response begins producing
        tool_call / tool_use tokens.  Gives the TUI a chance to show a spinner
        or status line so the user isn't staring at a frozen screen while a
        large tool payload (e.g. a 45 KB write_file) is being generated.
        """
        cb = self.tool_gen_callback
        if cb is not None:
            try:
                cb(tool_name)
            except Exception:
                pass

    def _has_stream_consumers(self) -> bool:
        """Return True if any streaming consumer is registered."""
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None
    ):
        """Streaming variant of _interruptible_api_call for real-time token delivery.

        Handles all three api_modes:
        - chat_completions: stream=True on OpenAI-compatible endpoints
        - anthropic_messages: client.messages.stream() via Anthropic SDK
        - codex_responses: delegates to _run_codex_stream (already streaming)

        Fires stream_delta_callback and _stream_callback for each text token.
        Tool-call turns suppress the callback — only text-only final responses
        stream to the consumer.  Returns a SimpleNamespace that mimics the
        non-streaming response shape so the rest of the agent loop is unchanged.

        Falls back to _interruptible_api_call on provider errors indicating
        streaming is not supported.
        """
        if self._interrupt_requested:
            raise InterruptedError("Agent interrupted before streaming API call")

        if self.api_mode == "codex_responses":
            # Codex streams internally via _run_codex_stream. The main dispatch
            # in _interruptible_api_call already calls it; we just need to
            # ensure on_first_delta reaches it. Store it on the instance
            # temporarily so _run_codex_stream can pick it up.
            self._codex_on_first_delta = on_first_delta
            try:
                return self._interruptible_api_call(api_kwargs)
            finally:
                self._codex_on_first_delta = None

        # Bedrock Converse uses boto3's converse_stream() with real-time delta
        # callbacks — same UX as Anthropic and chat_completions streaming.
        if self.api_mode == "bedrock_converse":
            result = {"response": None, "error": None}
            first_delta_fired = {"done": False}
            deltas_were_sent = {"yes": False}

            def _fire_first():
                if not first_delta_fired["done"] and on_first_delta:
                    first_delta_fired["done"] = True
                    try:
                        on_first_delta()
                    except Exception:
                        pass

            def _bedrock_call():
                try:
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        invalidate_runtime_client,
                        is_stale_connection_error,
                        stream_converse_with_callbacks,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    try:
                        raw_response = client.converse_stream(**api_kwargs)
                    except Exception as _bedrock_exc:
                        # Evict the cached client on stale-connection failures
                        # so the outer retry loop builds a fresh client/pool.
                        if is_stale_connection_error(_bedrock_exc):
                            invalidate_runtime_client(region)
                        raise

                    def _on_text(text):
                        _fire_first()
                        self._fire_stream_delta(text)
                        deltas_were_sent["yes"] = True

                    def _on_tool(name):
                        _fire_first()
                        self._fire_tool_gen_started(name)

                    def _on_reasoning(text):
                        _fire_first()
                        self._fire_reasoning_delta(text)

                    result["response"] = stream_converse_with_callbacks(
                        raw_response,
                        on_text_delta=_on_text if self._has_stream_consumers() else None,
                        on_tool_start=_on_tool,
                        on_reasoning_delta=_on_reasoning if self.reasoning_callback or self.stream_delta_callback else None,
                        on_interrupt_check=lambda: self._interrupt_requested,
                    )
                except Exception as e:
                    result["error"] = e

            t = threading.Thread(target=_bedrock_call, daemon=True)
            t.start()
            while t.is_alive():
                t.join(timeout=0.3)
                if self._interrupt_requested:
                    raise InterruptedError("Agent interrupted during Bedrock API call")
            if result["error"] is not None:
                raise result["error"]
            return result["response"]

        result = {"response": None, "error": None, "partial_tool_names": []}
        request_client_holder = {"client": None, "diag": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)
        # Wall-clock timestamp of the last real streaming chunk.  The outer
        # poll loop uses this to detect stale connections that keep receiving
        # SSE keep-alive pings but no actual data.
        last_chunk_time = {"t": time.time()}

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """Stream a chat completions response."""
            import httpx as _httpx
            # Per-provider / per-model request_timeout_seconds (from config.yaml)
            # wins over the HERMES_API_TIMEOUT env default if the user set it.
            _provider_timeout_cfg = get_provider_request_timeout(self.provider, self.model)
            _base_timeout = (
                _provider_timeout_cfg
                if _provider_timeout_cfg is not None
                else float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            )
            # Read timeout: config wins here too.  Otherwise use
            # HERMES_STREAM_READ_TIMEOUT (default 120s) for cloud providers.
            if _provider_timeout_cfg is not None:
                _stream_read_timeout = _provider_timeout_cfg
            else:
                _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
                # Local providers (Ollama, llama.cpp, vLLM) can take minutes for
                # prefill on large contexts before producing the first token.
                # Auto-increase the httpx read timeout unless the user explicitly
                # overrode HERMES_STREAM_READ_TIMEOUT.
                if _stream_read_timeout == 120.0 and self.base_url and is_local_endpoint(self.base_url):
                    _stream_read_timeout = _base_timeout
                    logger.debug(
                        "Local provider detected (%s) — stream read timeout raised to %.0fs",
                        self.base_url, _stream_read_timeout,
                    )
            stream_kwargs = {
                **api_kwargs,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": _httpx.Timeout(
                    connect=30.0,
                    read=_stream_read_timeout,
                    write=_base_timeout,
                    pool=30.0,
                ),
            }
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request",
                api_kwargs=stream_kwargs,
            )
            # Reset stale-stream timer so the detector measures from this
            # attempt's start, not a previous attempt's last chunk.
            last_chunk_time["t"] = time.time()
            self._touch_activity("waiting for provider response (streaming)")
            # Initialize per-attempt stream diagnostics so the retry block can
            # reach for them after the stream dies.  Lives on
            # ``request_client_holder["diag"]`` for closure access.
            _diag = self._stream_diag_init()
            request_client_holder["diag"] = _diag
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            # Capture rate limit headers from the initial HTTP response.
            # The OpenAI SDK Stream object exposes the underlying httpx
            # response via .response before any chunks are consumed.
            self._capture_rate_limits(getattr(stream, "response", None))
            # Snapshot diagnostic headers (cf-ray, x-openrouter-provider, etc.)
            # so they survive even when the stream dies before any chunk
            # arrives.  Best-effort; never raises.
            self._stream_diag_capture_response(_diag, getattr(stream, "response", None))

            # Log OpenRouter response cache status when present.
            self._check_openrouter_cache_status(getattr(stream, "response", None))

            content_parts: list = []
            tool_calls_acc: dict = {}
            tool_gen_notified: set = set()
            # Ollama-compatible endpoints reuse index 0 for every tool call
            # in a parallel batch, distinguishing them only by id.  Track
            # the last seen id per raw index so we can detect a new tool
            # call starting at the same index and redirect it to a fresh slot.
            _last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
            _active_slot_by_idx: dict = {}  # raw_index -> current slot in tool_calls_acc
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None
            for chunk in stream:
                last_chunk_time["t"] = time.time()
                self._touch_activity("receiving stream response")

                # Update per-attempt diagnostic counters.  Best-effort —
                # failures are swallowed so the streaming hot path is never
                # interrupted by diagnostic accounting.
                try:
                    _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                    if _diag.get("first_chunk_at") is None:
                        _diag["first_chunk_at"] = last_chunk_time["t"]
                    # Approximate byte size from the chunk's repr — exact wire
                    # bytes aren't exposed by the SDK, but len(repr(chunk)) is
                    # a stable proxy for "how much content arrived" that
                    # survives stub provider differences.
                    try:
                        _diag["bytes"] = int(_diag.get("bytes", 0)) + len(repr(chunk))
                    except Exception:
                        pass
                except Exception:
                    pass

                if self._interrupt_requested:
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # Usage comes in the final chunk with empty choices
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # Accumulate reasoning content
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    _fire_first_delta()
                    self._fire_reasoning_delta(reasoning_text)

                # Accumulate text content — fire callback only when no tool calls
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True
                    # Tool calls suppress regular content streaming (avoids
                    # displaying chatty "I'll use the tool..." text alongside
                    # tool calls).  But reasoning tags embedded in suppressed
                    # content should still reach the display — otherwise the
                    # reasoning box only appears as a post-response fallback,
                    # rendering it confusingly after the already-streamed
                    # response.  Route suppressed content through the stream
                    # delta callback so its tag extraction can fire the
                    # reasoning display.  Non-reasoning text is harmlessly
                    # suppressed by the CLI's _stream_delta when the stream
                    # box is already closed (tool boundary flush).
                    elif self.stream_delta_callback:
                        try:
                            self.stream_delta_callback(delta.content)
                            self._record_streamed_assistant_text(delta.content)
                        except Exception:
                            pass

                # Accumulate tool call deltas — notify display on first name
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        raw_idx = tc_delta.index if tc_delta.index is not None else 0
                        delta_id = tc_delta.id or ""

                        # Ollama fix: detect a new tool call reusing the same
                        # raw index (different id) and redirect to a fresh slot.
                        if raw_idx not in _active_slot_by_idx:
                            _active_slot_by_idx[raw_idx] = raw_idx
                        if (
                            delta_id
                            and raw_idx in _last_id_at_idx
                            and delta_id != _last_id_at_idx[raw_idx]
                        ):
                            new_slot = max(tool_calls_acc, default=-1) + 1
                            _active_slot_by_idx[raw_idx] = new_slot
                        if delta_id:
                            _last_id_at_idx[raw_idx] = delta_id
                        idx = _active_slot_by_idx[raw_idx]

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "extra_content": None,
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                # Use assignment, not +=.  Function names are
                                # atomic identifiers delivered complete in the
                                # first chunk (OpenAI spec).  Some providers
                                # (MiniMax M2.7 via NVIDIA NIM) resend the full
                                # name in every chunk; concatenation would
                                # produce "read_fileread_file".  Assignment
                                # (matching the OpenAI Node SDK / LiteLLM /
                                # Vercel AI patterns) is immune to this.
                                entry["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                        extra = getattr(tc_delta, "extra_content", None)
                        if extra is None and hasattr(tc_delta, "model_extra"):
                            extra = (tc_delta.model_extra or {}).get("extra_content")
                        if extra is not None:
                            if hasattr(extra, "model_dump"):
                                extra = extra.model_dump()
                            entry["extra_content"] = extra
                        # Fire once per tool when the full name is available
                        name = entry["function"]["name"]
                        if name and idx not in tool_gen_notified:
                            tool_gen_notified.add(idx)
                            _fire_first_delta()
                            self._fire_tool_gen_started(name)
                            # Record the partial tool-call name so the outer
                            # stub-builder can surface a user-visible warning
                            # if streaming dies before this tool's arguments
                            # are fully delivered.  Without this, a stall
                            # during tool-call JSON generation lets the stub
                            # at line ~6107 return `tool_calls=None`, silently
                            # discarding the attempted action.
                            result["partial_tool_names"].append(name)

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage in the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # Build mock response matching non-streaming shape
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            has_truncated_tool_args = False
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    arguments = tc["function"]["arguments"]
                    tool_name = tc["function"]["name"] or "?"
                    if arguments and arguments.strip():
                        try:
                            json.loads(arguments)
                        except json.JSONDecodeError:
                            # Attempt repair before flagging as truncated.
                            # Models like GLM-5.1 via Ollama produce trailing
                            # commas, unclosed brackets, Python None, etc.
                            # Without repair, these hit the truncation handler
                            # and kill the session.  _repair_tool_call_arguments
                            # returns "{}" for unrepairable args, which is far
                            # better than a crashed session.
                            repaired = _repair_tool_call_arguments(arguments, tool_name)
                            if repaired != "{}":
                                # Successfully repaired — use the fixed args
                                arguments = repaired
                            else:
                                # Unrepairable — flag for truncation handling
                                has_truncated_tool_args = True
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        extra_content=tc.get("extra_content"),
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=arguments,
                        ),
                    ))

            effective_finish_reason = finish_reason or "stop"
            if has_truncated_tool_args:
                effective_finish_reason = "length"

            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=mock_tool_calls,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=effective_finish_reason,
            )
            return SimpleNamespace(
                id="stream-" + str(uuid.uuid4()),
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
            )

        def _call_anthropic():
            """Stream an Anthropic Messages API response.

            Fires delta callbacks for real-time token delivery, but returns
            the native Anthropic Message object from get_final_message() so
            the rest of the agent loop (validation, tool extraction, etc.)
            works unchanged.
            """
            has_tool_use = False

            # Reset stale-stream timer for this attempt
            last_chunk_time["t"] = time.time()
            # Per-attempt diagnostic dict for the retry block to consume.
            _diag = self._stream_diag_init()
            request_client_holder["diag"] = _diag
            # Use the Anthropic SDK's streaming context manager
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                # The Anthropic SDK exposes the raw httpx response on
                # ``stream.response``.  Snapshot diagnostic headers
                # immediately so they survive a stream that dies before the
                # first event.
                try:
                    self._stream_diag_capture_response(
                        _diag, getattr(stream, "response", None)
                    )
                except Exception:
                    pass
                for event in stream:
                    # Update stale-stream timer on every event so the
                    # outer poll loop knows data is flowing.  Without
                    # this, the detector kills healthy long-running
                    # Opus streams after 180 s even when events are
                    # actively arriving (the chat_completions path
                    # already does this at the top of its chunk loop).
                    last_chunk_time["t"] = time.time()
                    self._touch_activity("receiving stream response")

                    # Update per-attempt diagnostic counters (best-effort).
                    try:
                        _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                        if _diag.get("first_chunk_at") is None:
                            _diag["first_chunk_at"] = last_chunk_time["t"]
                        try:
                            _diag["bytes"] = int(_diag.get("bytes", 0)) + len(repr(event))
                        except Exception:
                            pass
                    except Exception:
                        pass

                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True
                            tool_name = getattr(block, "name", None)
                            if tool_name:
                                _fire_first_delta()
                                self._fire_tool_gen_started(tool_name)

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                                    deltas_were_sent["yes"] = True
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    _fire_first_delta()
                                    self._fire_reasoning_delta(thinking_text)

                # Return the native Anthropic Message for downstream processing
                return stream.get_final_message()

        def _call():
            import httpx as _httpx

            _max_stream_retries = int(os.getenv("HERMES_STREAM_RETRIES", 2))

            try:
                for _stream_attempt in range(_max_stream_retries + 1):
                    # Check for interrupt before each retry attempt.  Without
                    # this, /stop closes the HTTP connection (outer poll loop),
                    # but the retry loop opens a FRESH connection — negating the
                    # interrupt entirely.  On slow providers (ollama-cloud) each
                    # retry can block for the full stream-read timeout (120s+),
                    # causing multi-minute delays between /stop and response.
                    if self._interrupt_requested:
                        raise InterruptedError("Agent interrupted before stream retry")
                    try:
                        if self.api_mode == "anthropic_messages":
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        else:
                            result["response"] = _call_chat_completions()
                        return  # success
                    except Exception as e:
                        _is_timeout = isinstance(
                            e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                        )
                        _is_conn_err = isinstance(
                            e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                        )

                        # If the stream died AFTER some tokens were delivered:
                        # normally we don't retry (the user already saw text,
                        # retrying would duplicate it).  BUT: if a tool call
                        # was in-flight when the stream died, silently aborting
                        # discards the tool call entirely.  In that case we
                        # prefer to retry — the user sees a brief
                        # "reconnecting" marker + duplicated preamble text,
                        # which is strictly better than a failed action with
                        # a "retry manually" message.  Limit this to transient
                        # connection errors (Clawdbot-style narrow gate): no
                        # tool has executed yet within this API call, so
                        # silent retry is safe wrt side-effects.
                        if deltas_were_sent["yes"]:
                            _partial_tool_in_flight = bool(
                                result.get("partial_tool_names")
                            )
                            _is_sse_conn_err_preview = False
                            if not _is_timeout and not _is_conn_err:
                                from openai import APIError as _APIError
                                if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                    _err_lower_preview = str(e).lower()
                                    _SSE_PREVIEW_PHRASES = (
                                        "connection lost",
                                        "connection reset",
                                        "connection closed",
                                        "connection terminated",
                                        "network error",
                                        "network connection",
                                        "terminated",
                                        "peer closed",
                                        "broken pipe",
                                        "upstream connect error",
                                    )
                                    _is_sse_conn_err_preview = any(
                                        phrase in _err_lower_preview
                                        for phrase in _SSE_PREVIEW_PHRASES
                                    )
                            _is_transient = (
                                _is_timeout or _is_conn_err or _is_sse_conn_err_preview
                            )
                            _can_silent_retry = (
                                _partial_tool_in_flight
                                and _is_transient
                                and _stream_attempt < _max_stream_retries
                            )
                            if not _can_silent_retry:
                                # Either no tool call was in-flight (so the
                                # turn was a pure text response — current
                                # stub-with-recovered-text behaviour is
                                # correct), or retries are exhausted, or the
                                # error isn't transient.  Fall through to the
                                # stub path.
                                logger.warning(
                                    "Streaming failed after partial delivery, not retrying: %s", e
                                )
                                result["error"] = e
                                return
                            # Tool call was in-flight AND error is transient:
                            # retry silently.  Clear per-attempt state so the
                            # next stream starts clean.  Fire a "reconnecting"
                            # marker so the user sees why the preamble is
                            # about to be re-streamed.  Structured WARNING is
                            # emitted by ``_emit_stream_drop`` below; no
                            # additional INFO line needed.
                            try:
                                self._fire_stream_delta(
                                    "\n\n⚠ Connection dropped mid tool-call; "
                                    "reconnecting…\n\n"
                                )
                            except Exception:
                                pass
                            # Reset the streamed-text buffer so the retry's
                            # fresh preamble doesn't get double-recorded in
                            # _current_streamed_assistant_text (which would
                            # pollute the interim-visible-text comparison).
                            try:
                                self._reset_stream_delivery_tracking()
                            except Exception:
                                pass
                            # Reset in-memory accumulators so the next
                            # attempt's chunks don't concat onto the dead
                            # stream's partial JSON.
                            result["partial_tool_names"] = []
                            deltas_were_sent["yes"] = False
                            first_delta_fired["done"] = False
                            self._emit_stream_drop(
                                error=e,
                                attempt=_stream_attempt + 2,
                                max_attempts=_max_stream_retries + 1,
                                mid_tool_call=True,
                                diag=request_client_holder.get("diag"),
                            )
                            stale = request_client_holder.get("client")
                            if stale is not None:
                                self._close_request_openai_client(
                                    stale, reason="stream_mid_tool_retry_cleanup"
                                )
                                request_client_holder["client"] = None
                            try:
                                self._replace_primary_openai_client(
                                    reason="stream_mid_tool_retry_pool_cleanup"
                                )
                            except Exception:
                                pass
                            continue

                        # SSE error events from proxies (e.g. OpenRouter sends
                        # {"error":{"message":"Network connection lost."}}) are
                        # raised as APIError by the OpenAI SDK.  These are
                        # semantically identical to httpx connection drops —
                        # the upstream stream died — and should be retried with
                        # a fresh connection.  Distinguish from HTTP errors:
                        # APIError from SSE has no status_code, while
                        # APIStatusError (4xx/5xx) always has one.
                        _is_sse_conn_err = False
                        if not _is_timeout and not _is_conn_err:
                            from openai import APIError as _APIError
                            if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                _err_lower_sse = str(e).lower()
                                _SSE_CONN_PHRASES = (
                                    "connection lost",
                                    "connection reset",
                                    "connection closed",
                                    "connection terminated",
                                    "network error",
                                    "network connection",
                                    "terminated",
                                    "peer closed",
                                    "broken pipe",
                                    "upstream connect error",
                                )
                                _is_sse_conn_err = any(
                                    phrase in _err_lower_sse
                                    for phrase in _SSE_CONN_PHRASES
                                )

                        if _is_timeout or _is_conn_err or _is_sse_conn_err:
                            # Transient network / timeout error. Retry the
                            # streaming request with a fresh connection first.
                            if _stream_attempt < _max_stream_retries:
                                self._emit_stream_drop(
                                    error=e,
                                    attempt=_stream_attempt + 2,
                                    max_attempts=_max_stream_retries + 1,
                                    mid_tool_call=False,
                                    diag=request_client_holder.get("diag"),
                                )
                                # Close the stale request client before retry
                                stale = request_client_holder.get("client")
                                if stale is not None:
                                    self._close_request_openai_client(
                                        stale, reason="stream_retry_cleanup"
                                    )
                                    request_client_holder["client"] = None
                                # Also rebuild the primary client to purge
                                # any dead connections from the pool.
                                try:
                                    self._replace_primary_openai_client(
                                        reason="stream_retry_pool_cleanup"
                                    )
                                except Exception:
                                    pass
                                continue
                            # Retries exhausted. Log the final failure with
                            # full diagnostic detail (chain, headers,
                            # bytes/elapsed) via the same helper used for
                            # mid-flight retries — subagent lines get the
                            # ``[subagent-N]`` log_prefix so the parent can
                            # attribute them.
                            self._log_stream_retry(
                                kind="exhausted",
                                error=e,
                                attempt=_max_stream_retries + 1,
                                max_attempts=_max_stream_retries + 1,
                                mid_tool_call=False,
                                diag=request_client_holder.get("diag"),
                            )
                            self._emit_status(
                                "❌ Connection to provider failed after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                        else:
                            _err_lower = str(e).lower()
                            _is_stream_unsupported = (
                                "stream" in _err_lower
                                and "not supported" in _err_lower
                            )
                            if _is_stream_unsupported:
                                self._disable_streaming = True
                                self._safe_print(
                                    "\n⚠  Streaming is not supported for this "
                                    "model/provider. Switching to non-streaming.\n"
                                    "   To avoid this delay, set display.streaming: false "
                                    "in config.yaml\n"
                                )
                            logger.info(
                                "Streaming failed before delivery: %s",
                                e,
                            )

                        # Propagate the error to the main retry loop instead of
                        # falling back to non-streaming inline.  The main loop has
                        # richer recovery: credential rotation, provider fallback,
                        # backoff, and — for "stream not supported" — will switch
                        # to non-streaming on the next attempt via _disable_streaming.
                        result["error"] = e
                        return
            except InterruptedError as e:
                # The interrupt may be noticed inside the worker thread before
                # the polling loop sees it. Surface it through the normal result
                # channel so callers never miss a fast pre-retry interrupt.
                result["error"] = e
                return
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
        # Local providers (Ollama, oMLX, llama-cpp) can take 300+ seconds
        # for prefill on large contexts.  Disable the stale detector unless
        # the user explicitly set HERMES_STREAM_STALE_TIMEOUT.
        if _stream_stale_timeout_base == 180.0 and self.base_url and is_local_endpoint(self.base_url):
            _stream_stale_timeout = float("inf")
            logger.debug("Local provider detected (%s) — stale stream timeout disabled", self.base_url)
        else:
            # Scale the stale timeout for large contexts: slow models (like Opus)
            # can legitimately think for minutes before producing the first token
            # when the context is large.  Without this, the stale detector kills
            # healthy connections during the model's thinking phase, producing
            # spurious RemoteProtocolError ("peer closed connection").
            _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
            if _est_tokens > 100_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
            elif _est_tokens > 50_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
            else:
                _stream_stale_timeout = _stream_stale_timeout_base

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        _last_heartbeat = time.time()
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        while t.is_alive():
            t.join(timeout=0.3)

            # Periodic heartbeat: touch the agent's activity tracker so the
            # gateway's inactivity monitor knows we're alive while waiting
            # for stream chunks.  Without this, long thinking pauses (e.g.
            # reasoning models) or slow prefill on local providers (Ollama)
            # trigger false inactivity timeouts.  The _call thread touches
            # activity on each chunk, but the gap between API call start
            # and first chunk can exceed the gateway timeout — especially
            # when the stale-stream timeout is disabled (local providers).
            _hb_now = time.time()
            if _hb_now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                _last_heartbeat = _hb_now
                _waiting_secs = int(_hb_now - last_chunk_time["t"])
                self._touch_activity(
                    f"waiting for stream response ({_waiting_secs}s, no chunks yet)"
                )

            # Detect stale streams: connections kept alive by SSE pings
            # but delivering no real chunks.  Kill the client so the
            # inner retry loop can start a fresh connection.
            _stale_elapsed = time.time() - last_chunk_time["t"]
            if _stale_elapsed > _stream_stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                    "model=%s context=~%s tokens. Killing connection.",
                    _stale_elapsed, _stream_stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                    f"(model: {api_kwargs.get('model', 'unknown')}, "
                    f"context: ~{_est_ctx:,} tokens). "
                    f"Reconnecting..."
                )
                try:
                    rc = request_client_holder.get("client")
                    if rc is not None:
                        self._close_request_openai_client(rc, reason="stale_stream_kill")
                except Exception:
                    pass
                # Rebuild the primary client too — its connection pool
                # may hold dead sockets from the same provider outage.
                try:
                    self._replace_primary_openai_client(reason="stale_stream_pool_cleanup")
                except Exception:
                    pass
                # Reset the timer so we don't kill repeatedly while
                # the inner thread processes the closure.
                last_chunk_time["t"] = time.time()
                self._touch_activity(
                    f"stale stream detected after {int(_stale_elapsed)}s, reconnecting"
                )

            if self._interrupt_requested:
                try:
                    if self.api_mode == "anthropic_messages":
                        self._anthropic_client.close()
                        self._rebuild_anthropic_client()
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during streaming API call")
        if result["error"] is not None:
            if deltas_were_sent["yes"]:
                # Streaming failed AFTER some tokens were already delivered to
                # the platform.  Re-raising would let the outer retry loop make
                # a new API call, creating a duplicate message.  Return a
                # partial "stop" response instead so the outer loop treats this
                # turn as complete (no retry, no fallback).
                # Recover whatever content was already streamed to the user.
                # _current_streamed_assistant_text accumulates text fired
                # through _fire_stream_delta, so it has exactly what the
                # user saw before the connection died.
                _partial_text = (
                    getattr(self, "_current_streamed_assistant_text", "") or ""
                ).strip() or None

                # If the stream died while the model was emitting a tool call,
                # the stub below will silently set `tool_calls=None` and the
                # agent loop will treat the turn as complete — the attempted
                # action is lost with no user-facing signal.  Append a
                # human-visible warning to the stub content so (a) the user
                # knows something failed, and (b) the next turn's model sees
                # in conversation history what was attempted and can retry.
                _partial_names = list(result.get("partial_tool_names") or [])
                if _partial_names:
                    _name_str = ", ".join(_partial_names[:3])
                    if len(_partial_names) > 3:
                        _name_str += f", +{len(_partial_names) - 3} more"
                    _warn = (
                        f"\n\n⚠ Stream stalled mid tool-call "
                        f"({_name_str}); the action was not executed. "
                        f"Ask me to retry if you want to continue."
                    )
                    _partial_text = (_partial_text or "") + _warn
                    # Also fire as a streaming delta so the user sees it now
                    # instead of only in the persisted transcript.
                    try:
                        self._fire_stream_delta(_warn)
                    except Exception:
                        pass
                    logger.warning(
                        "Partial stream dropped tool call(s) %s after %s chars "
                        "of text; surfaced warning to user: %s",
                        _partial_names, len(_partial_text or ""), result["error"],
                    )
                else:
                    logger.warning(
                        "Partial stream delivered before error; returning stub "
                        "response with %s chars of recovered content to prevent "
                        "duplicate messages: %s",
                        len(_partial_text or ""),
                        result["error"],
                    )
                _stub_msg = SimpleNamespace(
                    role="assistant", content=_partial_text, tool_calls=None,
                    reasoning_content=None,
                )
                return SimpleNamespace(
                    id="partial-stream-stub",
                    model=getattr(self, "model", "unknown"),
                    choices=[SimpleNamespace(
                        index=0, message=_stub_msg, finish_reason="stop",
                    )],
                    usage=None,
                )
            raise result["error"]
        return result["response"]

    # ── Provider fallback ──────────────────────────────────────────────────

    def _try_activate_fallback(self, reason: "FailoverReason | None" = None) -> bool:
        """Switch to the next fallback model/provider in the chain.

        Called when the current model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  Advances through the chain on
        each call; returns False when exhausted.

        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.
        """
        if reason in {FailoverReason.rate_limit, FailoverReason.billing}:
            # Only start cooldown when leaving the primary provider.  If we're
            # already on a fallback and chain-switching, the primary wasn't the
            # source of the 429 so the cooldown should not be reset/extended.
            fallback_already_active = bool(getattr(self, "_fallback_activated", False))
            current_provider = (getattr(self, "provider", "") or "").strip().lower()
            primary_provider = ((self._primary_runtime or {}).get("provider") or "").strip().lower()
            if (not fallback_already_active) or (primary_provider and current_provider == primary_provider):
                self._rate_limited_until = time.monotonic() + 60
        if self._fallback_index >= len(self._fallback_chain):
            return False

        fb = self._fallback_chain[self._fallback_index]
        self._fallback_index += 1
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return self._try_activate_fallback()  # skip invalid, try next

        # Skip entries that resolve to the current (provider, model) — falling
        # back to the same backend that just failed loops the failure. Compare
        # base_url too so two distinct custom_providers entries pointing at the
        # same shim/proxy URL also dedup. See issue #22548.
        current_provider = (getattr(self, "provider", "") or "").strip().lower()
        current_model = (getattr(self, "model", "") or "").strip()
        current_base_url = str(getattr(self, "base_url", "") or "").rstrip("/").lower()
        fb_base_url_for_dedup = (fb.get("base_url") or "").strip().rstrip("/").lower()
        if fb_provider == current_provider and fb_model == current_model:
            logging.warning(
                "Fallback skip: chain entry %s/%s matches current provider/model",
                fb_provider, fb_model,
            )
            return self._try_activate_fallback()
        if (
            fb_base_url_for_dedup
            and current_base_url
            and fb_base_url_for_dedup == current_base_url
            and fb_model == current_model
        ):
            logging.warning(
                "Fallback skip: chain entry base_url %s matches current backend",
                fb_base_url_for_dedup,
            )
            return self._try_activate_fallback()

        # Use centralized router for client construction.
        # raw_codex=True because the main agent needs direct responses.stream()
        # access for Codex providers.
        try:
            from agent.auxiliary_client import resolve_provider_client
            # Pass base_url and api_key from fallback config so custom
            # endpoints (e.g. Ollama Cloud) resolve correctly instead of
            # falling through to OpenRouter defaults.
            fb_base_url_hint = (fb.get("base_url") or "").strip() or None
            fb_api_key_hint = (fb.get("api_key") or "").strip() or None
            if not fb_api_key_hint:
                # key_env and api_key_env are both documented aliases (see
                # _normalize_custom_provider_entry in hermes_cli/config.py).
                fb_key_env = (fb.get("key_env") or fb.get("api_key_env") or "").strip()
                if fb_key_env:
                    fb_api_key_hint = os.getenv(fb_key_env, "").strip() or None
            # For Ollama Cloud endpoints, pull OLLAMA_API_KEY from env
            # when no explicit key is in the fallback config. Host match
            # (not substring) — see GHSA-76xc-57q6-vm5m.
            if fb_base_url_hint and base_url_host_matches(fb_base_url_hint, "ollama.com") and not fb_api_key_hint:
                fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
            fb_client, _resolved_fb_model = resolve_provider_client(
                fb_provider, model=fb_model, raw_codex=True,
                explicit_base_url=fb_base_url_hint,
                explicit_api_key=fb_api_key_hint)
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured",
                    fb_provider)
                return self._try_activate_fallback()  # try next in chain
            try:
                from hermes_cli.model_normalize import normalize_model_for_provider

                fb_model = normalize_model_for_provider(fb_model, fb_provider)
            except Exception:
                pass

            # Determine api_mode from provider / base URL / model
            fb_api_mode = "chat_completions"
            fb_base_url = str(fb_client.base_url)
            _fb_is_azure = self._is_azure_openai_url(fb_base_url)
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith("/anthropic"):
                fb_api_mode = "anthropic_messages"
            elif _fb_is_azure:
                # Azure OpenAI serves gpt-5.x on /chat/completions — does NOT
                # support the Responses API. Stay on chat_completions.
                fb_api_mode = "chat_completions"
            elif self._is_direct_openai_url(fb_base_url):
                fb_api_mode = "codex_responses"
            elif self._provider_model_requires_responses_api(
                fb_model,
                provider=fb_provider,
            ):
                # GPT-5.x models usually need Responses API, but keep
                # provider-specific exceptions like Copilot gpt-5-mini on
                # chat completions.
                fb_api_mode = "codex_responses"
            elif fb_provider == "bedrock" or (
                base_url_hostname(fb_base_url).startswith("bedrock-runtime.")
                and base_url_host_matches(fb_base_url, "amazonaws.com")
            ):
                fb_api_mode = "bedrock_converse"

            old_model = self.model

            # Clear the per-config context_length override so the fallback
            # model's actual context window is resolved instead of inheriting
            # the stale value from the previous model.  See #22387.
            self._config_context_length = None
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            if hasattr(self, "_transport_cache"):
                self._transport_cache.clear()
            self._fallback_activated = True

            # Honor per-provider / per-model request_timeout_seconds for the
            # fallback target (same knob the primary client uses).  None = use
            # SDK default.
            _fb_timeout = get_provider_request_timeout(fb_provider, fb_model)

            if fb_api_mode == "anthropic_messages":
                # Build native Anthropic client instead of using OpenAI client
                from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token
                effective_key = (fb_client.api_key or resolve_anthropic_token() or "") if fb_provider == "anthropic" else (fb_client.api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = fb_base_url
                self._anthropic_client = build_anthropic_client(
                    effective_key, self._anthropic_base_url, timeout=_fb_timeout,
                )
                self._is_anthropic_oauth = _is_oauth_token(effective_key) if fb_provider == "anthropic" else False
                self.client = None
                self._client_kwargs = {}
            else:
                # Swap OpenAI client and config in-place
                self.api_key = fb_client.api_key
                self.client = fb_client
                # Preserve provider-specific headers that
                # resolve_provider_client() may have baked into
                # fb_client via the default_headers kwarg.  The OpenAI
                # SDK stores these in _custom_headers.  Without this,
                # subsequent request-client rebuilds (via
                # _create_request_openai_client) drop the headers,
                # causing 403s from providers like Kimi Coding that
                # require a User-Agent sentinel.
                fb_headers = getattr(fb_client, "_custom_headers", None)
                if not fb_headers:
                    fb_headers = getattr(fb_client, "default_headers", None)
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                    **({"default_headers": dict(fb_headers)} if fb_headers else {}),
                }
                if _fb_timeout is not None:
                    self._client_kwargs["timeout"] = _fb_timeout
                    # Rebuild the shared OpenAI client so the configured
                    # timeout takes effect on the very next fallback request,
                    # not only after a later credential-rotation rebuild.
                    self._replace_primary_openai_client(reason="fallback_timeout_apply")

            # Re-evaluate prompt caching for the new provider/model
            self._use_prompt_caching, self._use_native_cache_layout = (
                self._anthropic_prompt_cache_policy(
                    provider=fb_provider,
                    base_url=fb_base_url,
                    api_mode=fb_api_mode,
                    model=fb_model,
                )
            )

            # LM Studio: preload before probing the fallback's context length.
            self._ensure_lmstudio_runtime_loaded()

            # Update context compressor limits for the fallback model.
            # Without this, compression decisions use the primary model's
            # context window (e.g. 200K) instead of the fallback's (e.g. 32K),
            # causing oversized sessions to overflow the fallback.
            # Also pass _config_context_length so the explicit config override
            # (model.context_length in config.yaml) is respected — without this,
            # the fallback activation drops to 128K even when config says 204800.
            if hasattr(self, 'context_compressor') and self.context_compressor:
                from agent.model_metadata import get_model_context_length
                fb_context_length = get_model_context_length(
                    self.model, base_url=self.base_url,
                    api_key=self.api_key, provider=self.provider,
                    config_context_length=getattr(self, "_config_context_length", None),
                )
                self.context_compressor.update_model(
                    model=self.model,
                    context_length=fb_context_length,
                    base_url=self.base_url,
                    api_key=getattr(self, "api_key", ""),
                    provider=self.provider,
                )

            self._emit_status(
                f"🔄 Primary model failed — switching to fallback: "
                f"{fb_model} via {fb_provider}"
            )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model, fb_model, fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback %s: %s", fb_model, e)
            return self._try_activate_fallback()  # try next in chain

    # ── Per-turn primary restoration ─────────────────────────────────────

    def _restore_primary_runtime(self) -> bool:
        """Restore the primary runtime at the start of a new turn.

        In long-lived CLI sessions a single AIAgent instance spans multiple
        turns.  Without restoration, one transient failure pins the session
        to the fallback provider for every subsequent turn.  Calling this at
        the top of ``run_conversation()`` makes fallback turn-scoped.

        The gateway caches agents across messages (``_agent_cache`` in
        ``gateway/run.py``), so this restoration IS needed there too.
        """
        if not self._fallback_activated:
            return False

        if getattr(self, "_rate_limited_until", 0) > time.monotonic():
            return False  # primary still in rate-limit cooldown, stay on fallback

        rt = self._primary_runtime
        try:
            # ── Core runtime state ──
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]           # setter updates _base_url_lower
            self.api_mode = rt["api_mode"]
            if hasattr(self, "_transport_cache"):
                self._transport_cache.clear()
            self.api_key = rt["api_key"]
            self._client_kwargs = dict(rt["client_kwargs"])
            self._use_prompt_caching = rt["use_prompt_caching"]
            # Default to native layout when the restored snapshot predates the
            # native-vs-proxy split (older sessions saved before this PR).
            self._use_native_cache_layout = rt.get(
                "use_native_cache_layout",
                self.api_mode == "anthropic_messages" and self.provider == "anthropic",
            )

            # ── Rebuild client for the primary provider ──
            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                    timeout=get_provider_request_timeout(self.provider, self.model),
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="restore_primary",
                    shared=True,
                )

            # ── Restore context engine state ──
            cc = self.context_compressor
            cc.update_model(
                model=rt["compressor_model"],
                context_length=rt["compressor_context_length"],
                base_url=rt["compressor_base_url"],
                api_key=rt["compressor_api_key"],
                provider=rt["compressor_provider"],
            )

            # ── Reset fallback chain for the new turn ──
            self._fallback_activated = False
            self._fallback_index = 0

            logging.info(
                "Primary runtime restored for new turn: %s (%s)",
                self.model, self.provider,
            )
            return True
        except Exception as e:
            logging.warning("Failed to restore primary runtime: %s", e)
            return False

    # Which error types indicate a transient transport failure worth
    # one more attempt with a rebuilt client / connection pool.
    _TRANSIENT_TRANSPORT_ERRORS = frozenset({
        "ReadTimeout", "ConnectTimeout", "PoolTimeout",
        "ConnectError", "RemoteProtocolError",
        "APIConnectionError", "APITimeoutError",
    })

    def _try_recover_primary_transport(
        self, api_error: Exception, *, retry_count: int, max_retries: int,
    ) -> bool:
        """Attempt one extra primary-provider recovery cycle for transient transport failures.

        After ``max_retries`` exhaust, rebuild the primary client (clearing
        stale connection pools) and give it one more attempt before falling
        back.  This is most useful for direct endpoints (custom, Z.AI,
        Anthropic, OpenAI, local models) where a TCP-level hiccup does not
        mean the provider is down.

        Skipped for proxy/aggregator providers (OpenRouter, Nous) which
        already manage connection pools and retries server-side — if our
        retries through them are exhausted, one more rebuilt client won't help.
        """
        if self._fallback_activated:
            return False

        # Only for transient transport errors
        error_type = type(api_error).__name__
        if error_type not in self._TRANSIENT_TRANSPORT_ERRORS:
            return False

        # Skip for aggregator providers — they manage their own retry infra
        if self._is_openrouter_url():
            return False
        provider_lower = (self.provider or "").strip().lower()
        if provider_lower in {"nous", "nous-research"}:
            return False

        try:
            # Close existing client to release stale connections
            if getattr(self, "client", None) is not None:
                try:
                    self._close_openai_client(
                        self.client, reason="primary_recovery", shared=True,
                    )
                except Exception:
                    pass

            # Rebuild from primary snapshot
            rt = self._primary_runtime
            self._client_kwargs = dict(rt["client_kwargs"])
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]
            self.api_mode = rt["api_mode"]
            if hasattr(self, "_transport_cache"):
                self._transport_cache.clear()
            self.api_key = rt["api_key"]

            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                    timeout=get_provider_request_timeout(self.provider, self.model),
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="primary_recovery",
                    shared=True,
                )

            wait_time = min(3 + retry_count, 8)
            self._vprint(
                f"{self.log_prefix}🔁 Transient {error_type} on {self.provider} — "
                f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
                force=True,
            )
            time.sleep(wait_time)
            return True
        except Exception as e:
            logging.warning("Primary transport recovery failed: %s", e)
            return False

    # ── End provider fallback ──────────────────────────────────────────────

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        try:
            with tmp:
                tmp.write(base64.b64decode(data))
        except Exception:
            # delete=False means a corrupt/unsupported data URL would otherwise
            # leak a zero-byte temp file on every failed materialization.
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        path = Path(tmp.name)
        return str(path), path

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, "user")
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, UI, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        vision_source = str(image_url or "")
        cleanup_path: Optional[Path] = None
        if vision_source.startswith("data:"):
            vision_source, cleanup_path = self._materialize_data_url_for_vision(vision_source)

        description = ""
        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(
                vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt)
            )
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"Image analysis failed: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        if not description:
            description = "Image analysis failed."

        note = f"[The {role_label} attached an image. Here's what it contains:\n{description}]"
        if vision_source and not str(image_url or "").startswith("data:"):
            note += (
                f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"
            )

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _model_supports_vision(self) -> bool:
        """Return True if the active provider+model reports native vision.

        Used to decide whether to strip image content parts from API-bound
        messages (for non-vision models) or let the provider adapter handle
        them natively (for vision-capable models).
        """
        try:
            from agent.models_dev import get_model_capabilities
            provider = (getattr(self, "provider", "") or "").strip()
            model = (getattr(self, "model", "") or "").strip()
            if not provider or not model:
                return False
            caps = get_model_capabilities(provider, model)
            if caps is None:
                return False
            return bool(caps.supports_vision)
        except Exception:
            return False

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
                if image_url:
                    image_notes.append(self._describe_image_for_anthropic_fallback(image_url, role))
                else:
                    image_notes.append("[An image was attached but no image source was available.]")
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return "[A multimodal message was converted to text for Anthropic compatibility.]"

    def _get_transport(self, api_mode: str = None):
        """Return the cached transport for the given (or current) api_mode.

        Lazy-initializes on first call per api_mode. Returns None if no
        transport is registered for the mode.
        """
        mode = api_mode or self.api_mode
        cache = getattr(self, "_transport_cache", None)
        if cache is None:
            cache = {}
            self._transport_cache = cache
        t = cache.get(mode)
        if t is None:
            from agent.transports import get_transport
            t = get_transport(mode)
            cache[mode] = t
        return t

    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        # Fast exit when no message carries image content at all.
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        # The Anthropic adapter (agent/anthropic_adapter.py:_convert_content_part_to_anthropic)
        # already translates OpenAI-style image_url/input_image parts into
        # native Anthropic ``{"type": "image", "source": ...}`` blocks. When
        # the active model supports vision we let the adapter do its job and
        # skip this legacy text-fallback preprocessor entirely.
        if self._model_supports_vision():
            return api_messages

        # Non-vision Anthropic model (rare today, but keep the fallback for
        # compat): replace each image part with a vision_analyze text note.
        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _prepare_messages_for_non_vision_model(self, api_messages: list) -> list:
        """Strip native image parts when the active model lacks vision.

        Runs on the chat.completions / codex_responses paths. Vision-capable
        models pass through unchanged (provider and any downstream translator
        handle the image parts natively). Non-vision models get each image
        replaced by a cached vision_analyze text description so the turn
        doesn't fail with "model does not support image input".
        """
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        if self._model_supports_vision():
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            # Reuse the Anthropic text-fallback preprocessor — the behaviour is
            # identical (walk content parts, replace images with cached
            # descriptions, merge back into a single text or structured
            # content). Naming is historical.
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _try_shrink_image_parts_in_messages(self, api_messages: list) -> bool:
        """Re-encode all native image parts at a smaller size to recover from
        image-too-large errors (Anthropic 5 MB, unknown other providers).

        Mutates ``api_messages`` in place. Returns True if any image part was
        actually replaced, False if there were no image parts to shrink or
        Pillow couldn't help (caller should surface the original error).

        Strategy: look for ``image_url`` / ``input_image`` parts carrying a
        ``data:image/...;base64,...`` payload.  For each one whose encoded
        size exceeds 4 MB (a safe target that slides under Anthropic's 5 MB
        ceiling with header overhead), write the base64 to a tempfile, call
        ``vision_tools._resize_image_for_vision`` to produce a smaller data
        URL, and substitute it in place.

        Non-data-URL images (http/https URLs) are not touched — the provider
        fetches those itself and the size limit is different.
        """
        if not api_messages:
            return False

        try:
            from tools.vision_tools import _resize_image_for_vision
        except Exception as exc:
            logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
            return False

        # 4 MB target leaves comfortable headroom under Anthropic's 5 MB.
        # Non-Anthropic providers we haven't observed rejecting are fine with
        # much larger; shrinking to 4 MB here loses quality but only fires
        # after a confirmed provider rejection, so the alternative is failure.
        target_bytes = 4 * 1024 * 1024
        changed_count = 0

        def _shrink_data_url(url: str) -> Optional[str]:
            """Return a smaller data URL, or None if shrink can't help."""
            if not isinstance(url, str) or not url.startswith("data:"):
                return None
            if len(url) <= target_bytes:
                # This specific image wasn't the oversized one.
                return None
            try:
                header, _, data = url.partition(",")
                mime = "image/jpeg"
                if header.startswith("data:"):
                    mime_part = header[len("data:"):].split(";", 1)[0].strip()
                    if mime_part.startswith("image/"):
                        mime = mime_part
                import base64 as _b64
                raw = _b64.b64decode(data)
                suffix = {
                    "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
                    "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/bmp": ".bmp",
                }.get(mime, ".jpg")
                tmp = tempfile.NamedTemporaryFile(
                    prefix="hermes_shrink_", suffix=suffix, delete=False,
                )
                try:
                    tmp.write(raw)
                    tmp.close()
                    resized = _resize_image_for_vision(
                        Path(tmp.name),
                        mime_type=mime,
                        max_base64_bytes=target_bytes,
                    )
                finally:
                    try:
                        Path(tmp.name).unlink(missing_ok=True)
                    except Exception:
                        pass
                if not resized or len(resized) >= len(url):
                    # Shrink didn't help (or made it bigger — corrupt input?).
                    return None
                return resized
            except Exception as exc:
                logger.warning("image-shrink recovery: re-encode failed — %s", exc)
                return None

        for msg in api_messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype not in {"image_url", "input_image"}:
                    continue
                image_value = part.get("image_url")
                # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
                # OpenAI Responses: {"image_url": "data:..."}
                if isinstance(image_value, dict):
                    url = image_value.get("url", "")
                    resized = _shrink_data_url(url)
                    if resized:
                        image_value["url"] = resized
                        changed_count += 1
                elif isinstance(image_value, str):
                    resized = _shrink_data_url(image_value)
                    if resized:
                        part["image_url"] = resized
                        changed_count += 1

        if changed_count:
            logger.info(
                "image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB",
                changed_count, target_bytes / (1024 * 1024),
            )
        return changed_count > 0

    def _anthropic_preserve_dots(self) -> bool:
        """True when using an anthropic-compatible endpoint that preserves dots in model names.
        Alibaba/DashScope keeps dots (e.g. qwen3.5-plus).
        MiniMax keeps dots (e.g. MiniMax-M2.7).
        Xiaomi MiMo keeps dots (e.g. mimo-v2.5, mimo-v2.5-pro).
        OpenCode Go/Zen keeps dots for non-Claude models (e.g. minimax-m2.5-free).
        ZAI/Zhipu keeps dots (e.g. glm-4.7, glm-5.1).
        AWS Bedrock uses dotted inference-profile IDs
        (e.g. ``global.anthropic.claude-opus-4-7``,
        ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``) and rejects
        the hyphenated form with
        ``HTTP 400 The provided model identifier is invalid``.
        Regression for #11976; mirrors the opencode-go fix for #5211
        (commit f77be22c), which extended this same allowlist."""
        if (getattr(self, "provider", "") or "").lower() in {
            "alibaba", "minimax", "minimax-cn",
            "opencode-go", "opencode-zen",
            "zai", "bedrock",
            "xiaomi",
        }:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return (
            "dashscope" in base
            or "aliyuncs" in base
            or "minimax" in base
            or "opencode.ai/zen/" in base
            or "bigmodel.cn" in base
            or "xiaomimimo.com" in base
            # AWS Bedrock runtime endpoints — defense-in-depth when
            # ``provider`` is unset but ``base_url`` still names Bedrock.
            or "bedrock-runtime." in base
        )

    def _is_qwen_portal(self) -> bool:
        """Return True when the base URL targets Qwen Portal."""
        return base_url_host_matches(self._base_url_lower, "portal.qwen.ai")

    def _qwen_prepare_chat_messages(self, api_messages: list) -> list:
        prepared = copy.deepcopy(api_messages)
        if not prepared:
            return prepared

        for msg in prepared:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # Normalize: convert bare strings to text dicts, keep dicts as-is.
                # deepcopy already created independent copies, no need for dict().
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        # Inject cache_control on the last part of the system message.
        for msg in prepared:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return prepared

    def _qwen_prepare_chat_messages_inplace(self, messages: list) -> None:
        """In-place variant — mutates an already-copied message list."""
        if not messages:
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the active API mode."""
        tools_for_api = self.tools

        if self.api_mode == "anthropic_messages":
            _transport = self._get_transport()
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            ctx_len = getattr(self, "context_compressor", None)
            ctx_len = ctx_len.context_length if ctx_len else None
            ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
            if ephemeral_out is not None:
                self._ephemeral_max_output_tokens = None  # consume immediately
            return _transport.build_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=tools_for_api,
                max_tokens=ephemeral_out if ephemeral_out is not None else self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=self._is_anthropic_oauth,
                preserve_dots=self._anthropic_preserve_dots(),
                context_length=ctx_len,
                base_url=getattr(self, "_anthropic_base_url", None),
                fast_mode=(self.request_overrides or {}).get("speed") == "fast",
                drop_context_1m_beta=bool(getattr(self, "_oauth_1m_beta_disabled", False)),
            )

        # AWS Bedrock native Converse API — bypasses the OpenAI client entirely.
        # The adapter handles message/tool conversion and boto3 calls directly.
        if self.api_mode == "bedrock_converse":
            _bt = self._get_transport()
            region = getattr(self, "_bedrock_region", None) or "us-east-1"
            guardrail = getattr(self, "_bedrock_guardrail_config", None)
            return _bt.build_kwargs(
                model=self.model,
                messages=api_messages,
                tools=tools_for_api,
                max_tokens=self.max_tokens or 4096,
                region=region,
                guardrail_config=guardrail,
            )

        if self.api_mode == "codex_responses":
            _ct = self._get_transport()
            is_github_responses = (
                base_url_host_matches(self.base_url, "models.github.ai")
                or base_url_host_matches(self.base_url, "api.githubcopilot.com")
            )
            is_codex_backend = (
                self.provider == "openai-codex"
                or (
                    self._base_url_hostname == "chatgpt.com"
                    and "/backend-api/codex" in self._base_url_lower
                )
            )
            is_xai_responses = self.provider == "xai" or self._base_url_hostname == "api.x.ai"
            _msgs_for_codex = self._prepare_messages_for_non_vision_model(api_messages)
            return _ct.build_kwargs(
                model=self.model,
                messages=_msgs_for_codex,
                tools=tools_for_api,
                reasoning_config=self.reasoning_config,
                session_id=getattr(self, "session_id", None),
                max_tokens=self.max_tokens,
                request_overrides=self.request_overrides,
                is_github_responses=is_github_responses,
                is_codex_backend=is_codex_backend,
                is_xai_responses=is_xai_responses,
                github_reasoning_extra=self._github_models_reasoning_extra_body() if is_github_responses else None,
            )

        # ── chat_completions (default) ─────────────────────────────────────
        _ct = self._get_transport()

        # Provider detection flags
        _is_qwen = self._is_qwen_portal()
        _is_or = self._is_openrouter_url()
        _is_gh = (
            base_url_host_matches(self._base_url_lower, "models.github.ai")
            or base_url_host_matches(self._base_url_lower, "api.githubcopilot.com")
        )
        _is_nous = "nousresearch" in self._base_url_lower
        _is_nvidia = "integrate.api.nvidia.com" in self._base_url_lower
        _is_kimi = (
            base_url_host_matches(self.base_url, "api.kimi.com")
            or base_url_host_matches(self.base_url, "moonshot.ai")
            or base_url_host_matches(self.base_url, "moonshot.cn")
        )
        _is_tokenhub = base_url_host_matches(self._base_url_lower, "tokenhub.tencentmaas.com")
        _is_lmstudio = (self.provider or "").strip().lower() == "lmstudio"

        # Temperature: _fixed_temperature_for_model may return OMIT_TEMPERATURE
        # sentinel (temperature omitted entirely), a numeric override, or None.
        try:
            from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
            _ft = _fixed_temperature_for_model(self.model, self.base_url)
            _omit_temp = _ft is OMIT_TEMPERATURE
            _fixed_temp = _ft if not _omit_temp else None
        except Exception:
            _omit_temp = False
            _fixed_temp = None

        # Provider preferences (OpenRouter-style)
        _prefs: Dict[str, Any] = {}
        if self.providers_allowed:
            _prefs["only"] = self.providers_allowed
        if self.providers_ignored:
            _prefs["ignore"] = self.providers_ignored
        if self.providers_order:
            _prefs["order"] = self.providers_order
        if self.provider_sort:
            _prefs["sort"] = self.provider_sort
        if self.provider_require_parameters:
            _prefs["require_parameters"] = True
        if self.provider_data_collection:
            _prefs["data_collection"] = self.provider_data_collection

        # Claude max-output override on aggregators
        _ant_max = None
        if (_is_or or _is_nous) and "claude" in (self.model or "").lower():
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                _ant_max = _get_anthropic_max_output(self.model)
            except Exception:
                pass

        # Qwen session metadata
        _qwen_meta = None
        if _is_qwen:
            _qwen_meta = {
                "sessionId": self.session_id or "hermes",
                "promptId": str(uuid.uuid4()),
            }

        # ── Provider profile path (registered providers) ───────────────────
        # Profiles handle per-provider quirks via hooks. When a profile is
        # found, delegate fully; otherwise fall through to the legacy flag path.
        try:
            from providers import get_provider_profile
            _profile = get_provider_profile(self.provider)
        except Exception:
            _profile = None

        if _profile:
            _ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
            if _ephemeral_out is not None:
                self._ephemeral_max_output_tokens = None

            return _ct.build_kwargs(
                model=self.model,
                messages=api_messages,
                tools=tools_for_api,
                base_url=self.base_url,
                timeout=self._resolved_api_call_timeout(),
                max_tokens=self.max_tokens,
                ephemeral_max_output_tokens=_ephemeral_out,
                max_tokens_param_fn=self._max_tokens_param,
                reasoning_config=self.reasoning_config,
                request_overrides=self.request_overrides,
                session_id=getattr(self, "session_id", None),
                provider_profile=_profile,
                ollama_num_ctx=self._ollama_num_ctx,
                # Context forwarded to profile hooks:
                provider_preferences=_prefs or None,
                openrouter_min_coding_score=self.openrouter_min_coding_score,
                anthropic_max_output=_ant_max,
                supports_reasoning=self._supports_reasoning_extra_body(),
                qwen_session_metadata=_qwen_meta,
            )

        # ── Legacy flag path ────────────────────────────────────────────
        # Reached only when get_provider_profile() returns None — i.e. a
        # completely unknown provider not in providers/ registry.
        _ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
        if _ephemeral_out is not None:
            self._ephemeral_max_output_tokens = None

        # Strip image parts for non-vision models (no-op when vision-capable).
        _msgs_for_chat = self._prepare_messages_for_non_vision_model(api_messages)

        return _ct.build_kwargs(
            model=self.model,
            messages=_msgs_for_chat,
            tools=tools_for_api,
            base_url=self.base_url,
            timeout=self._resolved_api_call_timeout(),
            max_tokens=self.max_tokens,
            ephemeral_max_output_tokens=_ephemeral_out,
            max_tokens_param_fn=self._max_tokens_param,
            reasoning_config=self.reasoning_config,
            request_overrides=self.request_overrides,
            session_id=getattr(self, "session_id", None),
            model_lower=(self.model or "").lower(),
            is_openrouter=_is_or,
            is_nous=_is_nous,
            is_qwen_portal=_is_qwen,
            is_github_models=_is_gh,
            is_nvidia_nim=_is_nvidia,
            is_kimi=_is_kimi,
            is_tokenhub=_is_tokenhub,
            is_lmstudio=_is_lmstudio,
            is_custom_provider=self.provider == "custom",
            ollama_num_ctx=self._ollama_num_ctx,
            provider_preferences=_prefs or None,
            openrouter_min_coding_score=self.openrouter_min_coding_score,
            qwen_prepare_fn=self._qwen_prepare_chat_messages if _is_qwen else None,
            qwen_prepare_inplace_fn=self._qwen_prepare_chat_messages_inplace if _is_qwen else None,
            qwen_session_metadata=_qwen_meta,
            fixed_temperature=_fixed_temp,
            omit_temperature=_omit_temp,
            supports_reasoning=self._supports_reasoning_extra_body(),
            github_reasoning_extra=self._github_models_reasoning_extra_body() if _is_gh else None,
            lmstudio_reasoning_options=self._lmstudio_reasoning_options_cached() if _is_lmstudio else None,
            anthropic_max_output=_ant_max,
            provider_name=self.provider,
        )

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body fields to upstream providers.
        Some providers/routes reject `reasoning` with 400s, so gate it to
        known reasoning-capable model families and direct Nous Portal.
        """
        if base_url_host_matches(self._base_url_lower, "nousresearch.com"):
            return True
        if base_url_host_matches(self._base_url_lower, "ai-gateway.vercel.sh"):
            return True
        if (
            base_url_host_matches(self._base_url_lower, "models.github.ai")
            or base_url_host_matches(self._base_url_lower, "api.githubcopilot.com")
        ):
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if (self.provider or "").strip().lower() == "lmstudio":
            opts = self._lmstudio_reasoning_options_cached()
            # "off-only" (or absent) means no real reasoning capability.
            return any(opt and opt != "off" for opt in opts)
        if "openrouter" not in self._base_url_lower:
            return False
        if "api.mistral.ai" in self._base_url_lower:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
            "tencent/hy3-preview",
            "xiaomi/",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _lmstudio_reasoning_options_cached(self) -> list[str]:
        """Probe LM Studio's published reasoning ``allowed_options`` once per
        (model, base_url). The list (e.g. ``["off","on"]`` or
        ``["off","minimal","low"]``) is needed both for the supports-reasoning
        gate and for clamping the emitted ``reasoning_effort`` so toggle-style
        models don't 400 on ``high``. Cache is keyed on (model, base_url) so
        ``/model`` swaps and base-URL changes don't reuse a stale list.
        Non-empty results are cached permanently (model capabilities don't
        change). Empty results (transient probe failure OR genuinely
        non-reasoning model) are cached with a 60-second TTL to avoid an
        HTTP round-trip on every turn while still retrying reasonably soon.
        """
        import time as _time

        cache = getattr(self, "_lm_reasoning_opts_cache", None)
        if cache is None:
            cache = self._lm_reasoning_opts_cache = {}
        key = (self.model, self.base_url)
        cached = cache.get(key)
        if cached is not None:
            opts, ts = cached
            # Non-empty → permanent. Empty → 60s TTL.
            if opts or (_time.monotonic() - ts) < 60:
                return opts
        try:
            from hermes_cli.models import lmstudio_model_reasoning_options
            opts = lmstudio_model_reasoning_options(
                self.model, self.base_url, getattr(self, "api_key", ""),
            )
        except Exception:
            opts = []
        cache[key] = (opts, _time.monotonic())
        return opts

    def _resolve_lmstudio_summary_reasoning_effort(self) -> Optional[str]:
        """Resolve a safe top-level ``reasoning_effort`` for LM Studio.

        The iteration-limit summary path calls ``chat.completions.create()``
        directly, bypassing the transport. Share the helper so the two paths
        can't drift on effort resolution and clamping.
        """
        from agent.lmstudio_reasoning import resolve_lmstudio_effort
        return resolve_lmstudio_effort(
            self.reasoning_config,
            self._lmstudio_reasoning_options_cached(),
        )

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        assistant_tool_calls = getattr(assistant_message, "tool_calls", None)
        reasoning_text = self._extract_reasoning(assistant_message)
        _from_structured = bool(reasoning_text)

        # Fallback: extract inline <think> blocks from content when no structured
        # reasoning fields are present (some models/providers embed thinking
        # directly in the content rather than returning separate API fields).
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

        if reasoning_text and self.reasoning_callback:
            # Skip callback when streaming is active — reasoning was already
            # displayed during the stream via one of two paths:
            #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
            #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
            # When streaming is NOT active, always fire so non-streaming modes
            # (gateway, batch, quiet) still get reasoning.
            # Any reasoning that wasn't shown during streaming is caught by the
            # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
            if not self.stream_delta_callback and not self._stream_callback:
                try:
                    self.reasoning_callback(reasoning_text)
                except Exception:
                    pass

        # Sanitize surrogates from API response — some models (e.g. Kimi/GLM via Ollama)
        # can return invalid surrogate code points that crash json.dumps() on persist.
        _raw_content = assistant_message.content or ""
        _san_content = _sanitize_surrogates(_raw_content)
        if reasoning_text:
            reasoning_text = _sanitize_surrogates(reasoning_text)

        # Strip inline reasoning tags (<think>…</think> etc.) from the stored
        # assistant content.  Reasoning was already captured into
        # ``reasoning_text`` above (either from structured fields or the
        # inline-block fallback), so the raw tags in content are redundant.
        # Leaving them in place caused reasoning to leak to messaging
        # platforms (#8878, #9568), inflate context on subsequent turns
        # (#9306 observed 16% content-size reduction on a real MiniMax
        # session), and pollute generated session titles.  One strip at the
        # storage boundary cleans content for every downstream consumer:
        # API replay, session transcript, gateway delivery, CLI display,
        # compression, title generation.
        if isinstance(_san_content, str) and _san_content:
            _san_content = self._strip_think_blocks(_san_content).strip()

        msg = {
            "role": "assistant",
            "content": _san_content,
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        raw_reasoning_content = getattr(assistant_message, "reasoning_content", None)
        if raw_reasoning_content is None and hasattr(assistant_message, "model_extra"):
            model_extra = getattr(assistant_message, "model_extra", None) or {}
            if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
                raw_reasoning_content = model_extra["reasoning_content"]
        if raw_reasoning_content is not None:
            msg["reasoning_content"] = _sanitize_surrogates(raw_reasoning_content)
        elif assistant_tool_calls and self._needs_thinking_reasoning_pad():
            # DeepSeek v4 thinking mode and Kimi / Moonshot thinking mode
            # both require reasoning_content on every assistant tool-call
            # message. Without it, replaying the persisted message causes
            # HTTP 400 ("The reasoning_content in the thinking mode must
            # be passed back to the API"). Include streamed reasoning
            # text when captured; otherwise pad with a single space —
            # DeepSeek V4 Pro tightened validation and rejects empty
            # string ("The reasoning content in the thinking mode must
            # be passed back to the API"). A space satisfies non-empty
            # checks everywhere without leaking fabricated reasoning.
            # Refs #15250, #17400, #17341.
            msg["reasoning_content"] = reasoning_text or " "

        # Additive fallback (refs #16844, #16884). Streaming-only providers
        # (glm, MiniMax, gpt-5.x via aigw, Anthropic via openai-compat shims)
        # accumulate reasoning through ``delta.reasoning_content`` chunks
        # but never land it on the message object as a top-level attribute,
        # so neither branch above fires and the chain-of-thought is stored
        # only under the internal ``reasoning`` key. When the user later
        # replays that history through a DeepSeek-v4 / Kimi thinking model,
        # the missing ``reasoning_content`` causes HTTP 400 ("The
        # reasoning_content in the thinking mode must be passed back to the
        # API.").
        #
        # Promote the already-sanitized streamed ``reasoning_text`` to
        # ``reasoning_content`` at write time, but ONLY when no prior branch
        # already set it AND we actually captured reasoning text. This
        # preserves every existing behavior:
        #   - SDK-exposed ``reasoning_content`` (OpenAI/Moonshot/DeepSeek SDK)
        #     still wins.
        #   - DeepSeek tool-call ""-pad (#15250) still fires.
        #   - Non-thinking turns with no reasoning leave the field absent,
        #     so ``_copy_reasoning_content_for_api``'s cross-provider leak
        #     guard (#15748) and ``reasoning``→``reasoning_content``
        #     promotion tiers still apply at replay time.
        if "reasoning_content" not in msg and reasoning_text:
            msg["reasoning_content"] = reasoning_text

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            # Pass reasoning_details back unmodified so providers (OpenRouter,
            # Anthropic, OpenAI) can maintain reasoning continuity across turns.
            # Each provider may include opaque fields (signature, encrypted_content)
            # that must be preserved exactly.
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API: preserve encrypted reasoning items for
        # multi-turn continuity. These get replayed as input on the next turn.
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        # Codex Responses API: preserve exact assistant message items (with
        # id/phase) so follow-up turns can replay structured items instead of
        # flattening to plain text. This is required for prefix cache hits.
        codex_message_items = getattr(assistant_message, "codex_message_items", None)
        if codex_message_items:
            msg["codex_message_items"] = codex_message_items

        if assistant_tool_calls:
            tool_calls = []
            for tool_call in assistant_tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        _fn = getattr(tool_call, "function", None)
                        _fn_name = getattr(_fn, "name", "") if _fn else ""
                        _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                        call_id = self._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if not isinstance(response_item_id, str) or not response_item_id.strip():
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                }
                # Preserve extra_content (e.g. Gemini thought_signature) so it
                # is sent back on subsequent API calls.  Without this, Gemini 3
                # thinking models reject the request with a 400 error.
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg

    def _needs_thinking_reasoning_pad(self) -> bool:
        """Return True when the active provider enforces reasoning_content echo-back.

        DeepSeek v4 thinking and Kimi / Moonshot thinking both reject replays
        of assistant tool-call messages that omit ``reasoning_content`` (refs
        #15250, #17400).
        """
        return (
            self._needs_deepseek_tool_reasoning()
            or self._needs_kimi_tool_reasoning()
        )

    def _needs_kimi_tool_reasoning(self) -> bool:
        """Return True when the current provider is Kimi / Moonshot thinking mode.

        Kimi ``/coding`` and Moonshot thinking mode both require
        ``reasoning_content`` on every assistant tool-call message; omitting
        it causes the next replay to fail with HTTP 400.
        """
        return (
            self.provider in {"kimi-coding", "kimi-coding-cn"}
            or base_url_host_matches(self.base_url, "api.kimi.com")
            or base_url_host_matches(self.base_url, "moonshot.ai")
            or base_url_host_matches(self.base_url, "moonshot.cn")
        )

    def _needs_deepseek_tool_reasoning(self) -> bool:
        """Return True when the current provider is DeepSeek thinking mode.

        DeepSeek V4 thinking mode requires ``reasoning_content`` on every
        assistant tool-call turn; omitting it causes HTTP 400 when the
        message is replayed in a subsequent API request (#15250).
        """
        provider = (self.provider or "").lower()
        model = (self.model or "").lower()
        return (
            provider == "deepseek"
            or "deepseek" in model
            or base_url_host_matches(self.base_url, "api.deepseek.com")
        )

    def _copy_reasoning_content_for_api(self, source_msg: dict, api_msg: dict) -> None:
        """Copy provider-facing reasoning fields onto an API replay message."""
        if source_msg.get("role") != "assistant":
            return

        # 1. Explicit reasoning_content already set — preserve it verbatim
        # (includes DeepSeek/Kimi's own space-placeholder written at creation
        # time, and any valid reasoning content from the same provider).
        #
        # Exception: sessions persisted BEFORE #17341 have empty-string
        # placeholders pinned at creation time. DeepSeek V4 Pro rejects
        # those with HTTP 400. When the active provider enforces the
        # thinking-mode echo, upgrade "" → " " on replay so stale history
        # doesn't 400 the user on the next turn.
        existing = source_msg.get("reasoning_content")
        if isinstance(existing, str):
            if existing == "" and self._needs_thinking_reasoning_pad():
                api_msg["reasoning_content"] = " "
            else:
                api_msg["reasoning_content"] = existing
            return

        needs_thinking_pad = self._needs_thinking_reasoning_pad()

        # 2. Cross-provider poisoned history (#15748): on DeepSeek/Kimi,
        # if the source turn has tool_calls AND a 'reasoning' field but no
        # 'reasoning_content' key, the 'reasoning' text was written by a
        # prior provider (e.g. MiniMax) — DeepSeek's own _build_assistant_message
        # pins reasoning_content at creation time for tool-call turns, so the
        # shape (reasoning set, reasoning_content absent, tool_calls present)
        # is unreachable from same-provider DeepSeek history after this fix.
        # Inject a single space to satisfy the API without leaking another
        # provider's chain of thought to DeepSeek/Kimi. Space (not "")
        # because DeepSeek V4 Pro rejects empty-string reasoning_content
        # in thinking mode (refs #17341).
        normalized_reasoning = source_msg.get("reasoning")
        if (
            needs_thinking_pad
            and source_msg.get("tool_calls")
            and isinstance(normalized_reasoning, str)
            and normalized_reasoning
        ):
            api_msg["reasoning_content"] = " "
            return

        # 3. Healthy session: promote 'reasoning' field to 'reasoning_content'
        # for providers that use the internal 'reasoning' key.
        # This must happen before the unconditional empty-string fallback so
        # genuine reasoning content is not overwritten (#15812 regression in
        # PR #15478).
        if isinstance(normalized_reasoning, str) and normalized_reasoning:
            api_msg["reasoning_content"] = normalized_reasoning
            return

        # 4. DeepSeek / Kimi thinking mode: all assistant messages need
        # reasoning_content. Inject a single space to satisfy the provider's
        # requirement when no explicit reasoning content is present. Covers
        # both tool-call turns (already-poisoned history with no reasoning
        # at all) and plain text turns. Space (not "") because DeepSeek V4
        # Pro tightened validation and rejects empty string with HTTP 400
        # ("The reasoning content in the thinking mode must be passed back
        # to the API"). Refs #17341.
        if needs_thinking_pad:
            api_msg["reasoning_content"] = " "
            return

        # 5. reasoning_content was present but not a string (e.g. None after
        # context compaction).  Don't pass null to the API.
        api_msg.pop("reasoning_content", None)

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
        validate the Chat Completions schema and reject unknown fields (call_id,
        response_item_id) with 400 or 422 errors. These fields are preserved in
        the internal message history — this method only modifies the outgoing
        API copy.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).

        Fields stripped: call_id, response_item_id
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    @staticmethod
    def _sanitize_tool_call_arguments(
        messages: list,
        *,
        logger=None,
        session_id: str = None,
    ) -> int:
        """Repair corrupted assistant tool-call argument JSON in-place."""
        log = logger or logging.getLogger(__name__)
        if not isinstance(messages, list):
            return 0

        repaired = 0
        marker = AIAgent._TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER

        def _prepend_marker(tool_msg: dict) -> None:
            existing = tool_msg.get("content")
            if isinstance(existing, str):
                if not existing:
                    tool_msg["content"] = marker
                elif not existing.startswith(marker):
                    tool_msg["content"] = f"{marker}\n{existing}"
                return
            if existing is None:
                tool_msg["content"] = marker
                return
            try:
                existing_text = json.dumps(existing)
            except TypeError:
                existing_text = str(existing)
            tool_msg["content"] = f"{marker}\n{existing_text}"

        message_index = 0
        while message_index < len(messages):
            msg = messages[message_index]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                message_index += 1
                continue

            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                message_index += 1
                continue

            insert_at = message_index + 1
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue

                arguments = function.get("arguments")
                if arguments is None or arguments == "":
                    function["arguments"] = "{}"
                    continue
                if isinstance(arguments, str) and not arguments.strip():
                    function["arguments"] = "{}"
                    continue
                if not isinstance(arguments, str):
                    continue

                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    tool_call_id = tool_call.get("id")
                    function_name = function.get("name", "?")
                    preview = arguments[:80]
                    log.warning(
                        "Corrupted tool_call arguments repaired before request "
                        "(session=%s, message_index=%s, tool_call_id=%s, function=%s, preview=%r)",
                        session_id or "-",
                        message_index,
                        tool_call_id or "-",
                        function_name,
                        preview,
                    )
                    function["arguments"] = "{}"

                    existing_tool_msg = None
                    scan_index = message_index + 1
                    while scan_index < len(messages):
                        candidate = messages[scan_index]
                        if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                            break
                        if candidate.get("tool_call_id") == tool_call_id:
                            existing_tool_msg = candidate
                            break
                        scan_index += 1

                    if existing_tool_msg is None:
                        messages.insert(
                            insert_at,
                            {
                                "role": "tool",
                                "name": function_name if function_name != "?" else "",
                                "tool_call_id": tool_call_id,
                                "content": marker,
                            },
                        )
                        insert_at += 1
                    else:
                        _prepend_marker(existing_tool_msg)

                    repaired += 1

            message_index += 1

        return repaired

    def _should_sanitize_tool_calls(self) -> bool:
        """Determine if tool_calls need sanitization for strict APIs.

        Codex Responses API uses fields like call_id and response_item_id
        that are not part of the standard Chat Completions schema. These
        fields must be stripped when calling any other API to avoid
        validation errors (400 Bad Request).

        Returns:
            bool: True if sanitization is needed (non-Codex API), False otherwise.
        """
        return self.api_mode != "codex_responses"

    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default", focus_topic: str = None) -> tuple:
        """Compress conversation context and split the session in SQLite.

        Args:
            focus_topic: Optional focus string for guided compression — the
                summariser will prioritise preserving information related to
                this topic.  Inspired by Claude Code's ``/compact <focus>``.

        Returns:
            (compressed_messages, new_system_prompt) tuple
        """
        _pre_msg_count = len(messages)
        logger.info(
            "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
            self.session_id or "none", _pre_msg_count,
            f"{approx_tokens:,}" if approx_tokens else "unknown", self.model,
            focus_topic,
        )

        # Notify external memory provider before compression discards context
        if self._memory_manager:
            try:
                self._memory_manager.on_pre_compress(messages)
            except Exception:
                pass

        try:
            compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic)
        except TypeError:
            # Plugin context engine with strict signature that doesn't accept
            # focus_topic — fall back to calling without it.
            compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens)

        summary_error = getattr(self.context_compressor, "_last_summary_error", None)
        if summary_error:
            if getattr(self, "_last_compression_summary_warning", None) != summary_error:
                self._last_compression_summary_warning = summary_error
                self._emit_warning(
                    f"⚠ Compression summary failed: {summary_error}. "
                    "Inserted a fallback context marker."
                )
        else:
            # No hard failure — but did the configured aux model error out
            # and get recovered by retrying on main?  Surface that so users
            # know their auxiliary.compression.model setting is broken even
            # though compression succeeded.
            _aux_fail_model = getattr(self.context_compressor, "_last_aux_model_failure_model", None)
            _aux_fail_err = getattr(self.context_compressor, "_last_aux_model_failure_error", None)
            if _aux_fail_model:
                # Dedup on (model, error) so we don't spam on every compaction
                _aux_key = (_aux_fail_model, _aux_fail_err)
                if getattr(self, "_last_aux_fallback_warning_key", None) != _aux_key:
                    self._last_aux_fallback_warning_key = _aux_key
                    self._emit_warning(
                        f"ℹ Configured compression model '{_aux_fail_model}' failed "
                        f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                        "check auxiliary.compression.model in config.yaml."
                    )

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # Propagate title to the new session with auto-numbering
                old_title = self._session_db.get_session_title(self.session_id)
                # Trigger memory extraction on the old session before it rotates.
                self.commit_memory_session(messages)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                os.environ["HERMES_SESSION_ID"] = self.session_id
                try:
                    from gateway.session_context import _SESSION_ID
                    _SESSION_ID.set(self.session_id)
                except Exception:
                    pass
                # Update session_log_file to point to the new session's JSON file
                self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
                self._session_db_created = False
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config=self._session_init_model_config,
                    parent_session_id=old_session_id,
                )
                self._session_db_created = True
                # Auto-number the title for the continuation session
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(old_title)
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(self.session_id, new_system_prompt)
                # Reset flush cursor — new session starts with no messages written
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # Notify the context engine that the session_id rotated because of
        # compression (not a fresh /new). Plugin engines (e.g. hermes-lcm) use
        # boundary_reason="compression" to preserve DAG lineage across the
        # rollover instead of re-initializing fresh per-session state.
        # See hermes-lcm#68. Built-in ContextCompressor ignores kwargs.
        try:
            _old_sid = locals().get("old_session_id")
            if _old_sid and hasattr(self.context_compressor, "on_session_start"):
                self.context_compressor.on_session_start(
                    self.session_id or "",
                    boundary_reason="compression",
                    old_session_id=_old_sid,
                )
        except Exception as _ce_err:
            logger.debug("context engine on_session_start (compression): %s", _ce_err)

        # Notify memory providers of the compression-driven session_id rotation
        # so provider-cached per-session state (Hindsight's _document_id,
        # accumulated turn buffers, counters) refreshes. reset=False because
        # the logical conversation continues; only the id and DB row rolled
        # over. See #6672.
        try:
            _old_sid = locals().get("old_session_id")
            if _old_sid and self._memory_manager:
                self._memory_manager.on_session_switch(
                    self.session_id or "",
                    parent_session_id=_old_sid,
                    reset=False,
                    reason="compression",
                )
        except Exception as _me_err:
            logger.debug("memory manager on_session_switch (compression): %s", _me_err)

        # Warn on repeated compressions (quality degrades with each pass)
        _cc = self.context_compressor.compression_count
        if _cc >= 2:
            self._vprint(
                f"{self.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh.",
                force=True,
            )

        # Update token estimate after compaction so pressure calculations
        # use the post-compression count, not the stale pre-compression one.
        # Use estimate_request_tokens_rough() so tool schemas are included —
        # with 50+ tools enabled, schemas alone can add 20-30K tokens, and
        # omitting them delays the next compression cycle far past the
        # configured threshold (issue #14695).
        _compressed_est = estimate_request_tokens_rough(
            compressed,
            system_prompt=new_system_prompt or "",
            tools=self.tools or None,
        )
        self.context_compressor.last_prompt_tokens = _compressed_est
        self.context_compressor.last_completion_tokens = 0

        # Clear the file-read dedup cache.  After compression the original
        # read content is summarised away — if the model re-reads the same
        # file it needs the full content, not a "file unchanged" stub.
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d tokens=~%s",
            self.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt

    def _set_tool_guardrail_halt(self, decision: ToolGuardrailDecision) -> None:
        """Record the first guardrail decision that should stop this turn."""
        if decision.should_halt and self._tool_guardrail_halt_decision is None:
            self._tool_guardrail_halt_decision = decision

    def _toolguard_controlled_halt_response(self, decision: ToolGuardrailDecision) -> str:
        tool = decision.tool_name or "a tool"
        return (
            f"I stopped retrying {tool} because it hit the tool-call guardrail "
            f"({decision.code}) after {decision.count} repeated non-progressing "
            "attempts. The last tool result explains the blocker; the next step is "
            "to change strategy instead of repeating the same call."
        )

    def _append_guardrail_observation(
        self,
        tool_name: str,
        function_args: dict,
        function_result: str,
        *,
        failed: bool,
    ) -> str:
        decision = self._tool_guardrails.after_call(
            tool_name,
            function_args,
            function_result,
            failed=failed,
        )
        if decision.action in {"warn", "halt"}:
            function_result = append_toolguard_guidance(function_result, decision)
        if decision.should_halt:
            self._set_tool_guardrail_halt(decision)
        return function_result

    def _guardrail_block_result(self, decision: ToolGuardrailDecision) -> str:
        self._set_tool_guardrail_halt(decision)
        return toolguard_synthetic_result(decision)

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        Dispatches to concurrent execution only for batches that look
        independent: read-only tools may always share the parallel path, while
        file reads/writes may do so only when their target paths do not overlap.
        """
        tool_calls = assistant_message.tool_calls

        # Allow _vprint during tool execution even with stream consumers
        self._executing_tools = True
        try:
            if not _should_parallelize_tool_batch(tool_calls):
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            return self._execute_tool_calls_concurrent(
                assistant_message, messages, effective_task_id, api_call_count
            )
        finally:
            self._executing_tools = False

    def _dispatch_delegate_task(self, function_args: dict) -> str:
        """Single call site for delegate_task dispatch.

        New DELEGATE_TASK_SCHEMA fields only need to be added here to reach all
        invocation paths (concurrent, sequential, inline).
        """
        from tools.delegate_tool import delegate_task as _delegate_task
        return _delegate_task(
            goal=function_args.get("goal"),
            context=function_args.get("context"),
            toolsets=function_args.get("toolsets"),
            tasks=function_args.get("tasks"),
            max_iterations=function_args.get("max_iterations"),
            acp_command=function_args.get("acp_command"),
            acp_args=function_args.get("acp_args"),
            role=function_args.get("role"),
            parent_agent=self,
        )

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None, messages: list = None,
                     pre_tool_block_checked: bool = False) -> str:
        """Invoke a single tool and return the result string. No display logic.

        Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
        tools. Used by the concurrent execution path; the sequential path retains
        its own inline invocation for backward-compatible display handling.
        """
        # Check plugin hooks for a block directive before executing anything.
        block_message: Optional[str] = None
        if not pre_tool_block_checked:
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                block_message = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                pass
        if block_message is not None:
            return json.dumps({"error": block_message}, ensure_ascii=False)

        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            session_db = self._get_session_db_for_recall()
            if not session_db:
                from hermes_state import format_session_db_unavailable
                return json.dumps({"success": False, "error": format_session_db_unavailable()})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # Bridge: notify external memory provider of built-in memory writes
            if self._memory_manager and function_args.get("action") in {"add", "replace"}:
                try:
                    self._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                        metadata=self._build_memory_write_metadata(
                            task_id=effective_task_id,
                            tool_call_id=tool_call_id,
                        ),
                    )
                except Exception:
                    pass
            return result
        elif self._memory_manager and self._memory_manager.has_tool(function_name):
            return self._memory_manager.handle_tool_call(function_name, function_args)
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            return self._dispatch_delegate_task(function_args)
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                tool_call_id=tool_call_id,
                session_id=self.session_id or "",
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
            )

    @staticmethod
    def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
        """Word-wrap verbose tool output to fit the terminal width.

        Splits *text* on existing newlines and wraps each line individually,
        preserving intentional line breaks (e.g. pretty-printed JSON).
        Returns a ready-to-print string with *label* on the first line and
        continuation lines indented.
        """
        import shutil as _shutil
        import textwrap as _tw
        cols = _shutil.get_terminal_size((120, 24)).columns
        wrap_width = max(40, cols - len(indent))
        out_lines: list[str] = []
        for raw_line in text.split("\n"):
            if len(raw_line) <= wrap_width:
                out_lines.append(raw_line)
            else:
                wrapped = _tw.wrap(raw_line, width=wrap_width,
                                   break_long_words=True,
                                   break_on_hyphens=False)
                out_lines.extend(wrapped or [raw_line])
        body = ("\n" + indent).join(out_lines)
        return f"{indent}{label}{body}"

    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages so the API sees them in the expected sequence.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── Pre-flight: interrupt check ──────────────────────────────────
        if self._interrupt_requested:
            print(f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "name": tc.function.name,
                    "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                    "tool_call_id": tc.id,
                })
            return

        # ── Parse args + pre-execution bookkeeping ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Reset nudge counters
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Checkpoint for file-mutating tools
            if function_name in {"write_file", "patch"} and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

            block_result = None
            blocked_by_guardrail = False
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                block_message = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                block_message = None

            if block_message is not None:
                block_result = json.dumps({"error": block_message}, ensure_ascii=False)
            else:
                guardrail_decision = self._tool_guardrails.before_call(function_name, function_args)
                if not guardrail_decision.allows_execution:
                    block_result = self._guardrail_block_result(guardrail_decision)
                    blocked_by_guardrail = True

            parsed_calls.append((tool_call, function_name, function_args, block_result, blocked_by_guardrail))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _, _, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(self._wrap_verbose("Args: ", json.dumps(args, indent=2, ensure_ascii=False)))
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for tc, name, args, block_result, blocked_by_guardrail in parsed_calls:
            if block_result is not None:
                continue
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args, block_result, blocked_by_guardrail in parsed_calls:
            if block_result is not None:
                continue
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # Each slot holds (function_name, function_args, function_result, duration, error_flag, blocked_flag)
        results = [None] * num_tools
        for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls):
            if block_result is not None:
                results[i] = (name, args, block_result, 0.0, True, True)

        # Touch activity before launching workers so the gateway knows
        # we're executing tools (not stuck).
        self._current_tool = tool_names_str
        self._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

        # Capture CLI callbacks from the agent thread so worker threads can
        # register them locally.  Without this, _get_approval_callback() in
        # terminal_tool returns None in ThreadPoolExecutor workers, causing
        # the dangerous-command prompt to fall back to input() — which
        # deadlocks against prompt_toolkit's raw terminal mode (#13617).
        _parent_approval_cb = _get_approval_callback()
        _parent_sudo_cb = _get_sudo_password_callback()

        def _run_tool(index, tool_call, function_name, function_args):
            """Worker function executed in a thread."""
            # Register this worker tid so the agent can fan out an interrupt
            # to it — see AIAgent.interrupt().  Must happen first thing, and
            # must be paired with discard + clear in the finally block.
            _worker_tid = threading.current_thread().ident
            with self._tool_worker_threads_lock:
                self._tool_worker_threads.add(_worker_tid)
            # Race: if the agent was interrupted between fan-out (which
            # snapshotted an empty/earlier set) and our registration, apply
            # the interrupt to our own tid now so is_interrupted() inside
            # the tool returns True on the next poll.
            if self._interrupt_requested:
                try:
                    _set_interrupt(True, _worker_tid)
                except Exception:
                    pass
            # Set the activity callback on THIS worker thread so
            # _wait_for_process (terminal commands) can fire heartbeats.
            # The callback is thread-local; the main thread's callback
            # is invisible to worker threads.
            try:
                from tools.environments.base import set_activity_callback
                set_activity_callback(self._touch_activity)
            except Exception:
                pass
            # Propagate approval/sudo callbacks to this worker thread.
            # Mirrors cli.py run_agent() pattern (GHSA-qg5c-hvr5-hjgr).
            if _parent_approval_cb is not None:
                try:
                    _set_approval_callback(_parent_approval_cb)
                except Exception:
                    pass
            if _parent_sudo_cb is not None:
                try:
                    _set_sudo_password_callback(_parent_sudo_cb)
                except Exception:
                    pass
            start = time.time()
            try:
                result = self._invoke_tool(
                    function_name,
                    function_args,
                    effective_task_id,
                    tool_call.id,
                    messages=messages,
                    pre_tool_block_checked=True,
                )
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error, False)
            # Tear down worker-tid tracking.  Clear any interrupt bit we may
            # have set so the next task scheduled onto this recycled tid
            # starts with a clean slate.
            with self._tool_worker_threads_lock:
                self._tool_worker_threads.discard(_worker_tid)
            try:
                _set_interrupt(False, _worker_tid)
            except Exception:
                pass
            # Clear thread-local callbacks so a recycled worker thread
            # doesn't hold stale references to a disposed CLI instance.
            try:
                _set_approval_callback(None)
                _set_sudo_password_callback(None)
            except Exception:
                pass

        # Start spinner for CLI mode (skip when TUI handles tool progress)
        spinner = None
        if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.get_waiting_faces())
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=self._print_fn)
            spinner.start()

        try:
            runnable_calls = [
                (i, tc, name, args)
                for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls)
                if block_result is None
            ]
            futures = []
            if runnable_calls:
                max_workers = min(len(runnable_calls), _MAX_TOOL_WORKERS)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for i, tc, name, args in runnable_calls:
                        # Propagate ContextVars (e.g. _approval_session_key); mirrors asyncio.to_thread.
                        ctx = contextvars.copy_context()
                        f = executor.submit(ctx.run, _run_tool, i, tc, name, args)
                        futures.append(f)

                    # Wait for all to complete with periodic heartbeats so the
                    # gateway's inactivity monitor doesn't kill us during long
                    # concurrent tool batches. Also check for user interrupts
                    # so we don't block indefinitely when the user sends /stop
                    # or a new message during concurrent tool execution.
                    _conc_start = time.time()
                    _interrupt_logged = False
                    while True:
                        done, not_done = concurrent.futures.wait(
                            futures, timeout=5.0,
                        )
                        if not not_done:
                            break

                        # Check for interrupt — the per-thread interrupt signal
                        # already causes individual tools (terminal, execute_code)
                        # to abort, but tools without interrupt checks (web_search,
                        # read_file) will run to completion. Cancel any futures
                        # that haven't started yet so we don't block on them.
                        if self._interrupt_requested:
                            if not _interrupt_logged:
                                _interrupt_logged = True
                                self._vprint(
                                    f"{self.log_prefix}⚡ Interrupt: cancelling "
                                    f"{len(not_done)} pending concurrent tool(s)",
                                    force=True,
                                )
                            for f in not_done:
                                f.cancel()
                            # Give already-running tools a moment to notice the
                            # per-thread interrupt signal and exit gracefully.
                            concurrent.futures.wait(not_done, timeout=3.0)
                            break

                        _conc_elapsed = int(time.time() - _conc_start)
                        # Heartbeat every ~30s (6 × 5s poll intervals)
                        if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                            _still_running = [
                                parsed_calls[futures.index(f)][1]
                                for f in not_done
                                if f in futures
                            ]
                            self._touch_activity(
                                f"concurrent tools running ({_conc_elapsed}s, "
                                f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                            )
        finally:
            if spinner:
                # Build a summary message for the spinner stop
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls):
            r = results[i]
            blocked = False
            if r is None:
                # Tool was cancelled (interrupt) or thread didn't return
                if self._interrupt_requested:
                    function_result = f"[Tool execution cancelled — {name} was skipped due to user interrupt]"
                else:
                    function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error, blocked = r

                if not blocked:
                    function_result = self._append_guardrail_observation(
                        function_name,
                        function_args,
                        function_result,
                        failed=is_error,
                    )

                if is_error:
                    _err_text = _multimodal_text_summary(function_result)
                    result_preview = _err_text[:200] if len(_err_text) > 200 else _err_text
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                # Track file-mutation outcome for the turn-end verifier.
                # `blocked` calls never actually ran — don't let a guardrail
                # block count as either a failure or a success.
                if not blocked:
                    try:
                        self._record_file_mutation_result(
                            function_name, function_args, function_result, is_error,
                        )
                    except Exception as _ver_err:
                        logging.debug("file-mutation verifier record failed: %s", _ver_err)

                if not blocked and self.tool_progress_callback:
                    try:
                        self.tool_progress_callback(
                            "tool.completed", function_name, None, None,
                            duration=tool_duration, is_error=is_error,
                        )
                    except Exception as cb_err:
                        logging.debug(f"Tool progress callback error: {cb_err}")

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # Print cute message per tool
            if self._should_emit_quiet_tool_messages():
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                _preview_str = _multimodal_text_summary(function_result)
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(self._wrap_verbose("Result: ", _preview_str))
                else:
                    response_preview = _preview_str[:self.log_prefix_chars] + "..." if len(_preview_str) > self.log_prefix_chars else _preview_str
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if not blocked and self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=name,
                tool_use_id=tc.id,
                env=get_active_env(effective_task_id),
            ) if not _is_multimodal_tool_result(function_result) else function_result

            subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
            if subdir_hints:
                if _is_multimodal_tool_result(function_result):
                    # Append the hint to the text summary part so the model
                    # still sees it; don't touch the image blocks.
                    _append_subdir_hint_to_multimodal(function_result, subdir_hints)
                else:
                    function_result += subdir_hints

            # Unwrap _multimodal dicts to an OpenAI-style content list so any
            # vision-capable provider receives [{type:text},{type:image_url}]
            # rather than a raw Python dict.  The Anthropic adapter already
            # accepts content lists; vision-capable OpenAI-compatible servers
            # (mlx-vlm, GPT-4o, …) accept image_url in tool messages natively.
            # Text-only servers that reject images are handled by the adaptive
            # _vision_supported recovery in the API retry loop.
            # String results pass through unchanged.
            _tool_content = (
                function_result["content"]
                if _is_multimodal_tool_result(function_result)
                else function_result
            )
            tool_msg = {
                "role": "tool",
                "name": name,
                "content": _tool_content,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

            # ── Per-tool /steer drain ───────────────────────────────────
            # Same as the sequential path: drain between each collected
            # result so the steer lands as early as possible.
            self._apply_pending_steer_to_tool_results(messages, 1)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools = len(parsed_calls)
        if num_tools > 0:
            turn_tool_msgs = messages[-num_tools:]
            enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id))

        # ── /steer injection ──────────────────────────────────────────────
        # Append any pending user steer text to the last tool result so the
        # agent sees it on its next iteration. Runs AFTER budget enforcement
        # so the steer marker is never truncated. See steer() for details.
        if num_tools > 0:
            self._apply_pending_steer_to_tool_results(messages, num_tools)

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls sequentially (original behavior). Used for single calls or interactive tools."""
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # SAFETY: check interrupt BEFORE starting each tool.
            # If the user sent "stop" during a previous tool's execution,
            # do NOT start any more tools -- skip them all immediately.
            if self._interrupt_requested:
                remaining_calls = assistant_message.tool_calls[i-1:]
                if remaining_calls:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "name": skipped_name,
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Check plugin hooks for a block directive before executing.
            _block_msg: Optional[str] = None
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                _block_msg = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                pass

            _guardrail_block_decision: ToolGuardrailDecision | None = None
            if _block_msg is None:
                guardrail_decision = self._tool_guardrails.before_call(function_name, function_args)
                if not guardrail_decision.allows_execution:
                    _guardrail_block_decision = guardrail_decision

            _execution_blocked = _block_msg is not None or _guardrail_block_decision is not None

            if _execution_blocked:
                # Tool blocked by plugin or guardrail policy — skip counters,
                # callbacks, checkpointing, activity mutation, and real execution.
                pass
            # Reset nudge counters when the relevant tool is actually used
            elif function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(self._wrap_verbose("Args: ", json.dumps(function_args, indent=2, ensure_ascii=False)))
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            if not _execution_blocked:
                self._current_tool = function_name
                self._touch_activity(f"executing tool: {function_name}")

            # Set activity callback for long-running tool execution (terminal
            # commands, etc.) so the gateway's inactivity monitor doesn't kill
            # the agent while a command is running.
            if not _execution_blocked:
                try:
                    from tools.environments.base import set_activity_callback
                    set_activity_callback(self._touch_activity)
                except Exception:
                    pass

            if not _execution_blocked and self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback("tool.started", function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if not _execution_blocked and self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # Checkpoint: snapshot working dir before file-mutating tools
            if not _execution_blocked and function_name in {"write_file", "patch"} and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(
                            work_dir, f"before {function_name}"
                        )
                except Exception:
                    pass  # never block tool execution

            # Checkpoint before destructive terminal commands
            if not _execution_blocked and function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass  # never block tool execution

            tool_start_time = time.time()

            if _block_msg is not None:
                # Tool blocked by plugin policy — return error without executing.
                function_result = json.dumps({"error": _block_msg}, ensure_ascii=False)
                tool_duration = 0.0
            elif _guardrail_block_decision is not None:
                # Tool blocked by tool-loop guardrail — synthesize exactly one
                # tool result for the original tool_call_id without executing.
                function_result = self._guardrail_block_result(_guardrail_block_decision)
                tool_duration = 0.0
            elif function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                session_db = self._get_session_db_for_recall()
                if not session_db:
                    from hermes_state import format_session_db_unavailable
                    function_result = json.dumps({"success": False, "error": format_session_db_unavailable()})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            elif function_name == "memory":
                target = function_args.get("target", "memory")
                from tools.memory_tool import memory_tool as _memory_tool
                function_result = _memory_tool(
                    action=function_args.get("action"),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                # Bridge: notify external memory provider of built-in memory writes
                if self._memory_manager and function_args.get("action") in {"add", "replace"}:
                    try:
                        self._memory_manager.on_memory_write(
                            function_args.get("action", ""),
                            target,
                            function_args.get("content", ""),
                            metadata=self._build_memory_write_metadata(
                                task_id=effective_task_id,
                                tool_call_id=getattr(tool_call, "id", None),
                            ),
                        )
                    except Exception:
                        pass
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = self._dispatch_delegate_task(function_args)
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._context_engine_tool_names and function_name in self._context_engine_tool_names:
                # Context engine tools (lcm_grep, lcm_describe, lcm_expand, etc.)
                spinner = None
                if self._should_emit_quiet_tool_messages():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _ce_result = None
                try:
                    function_result = self.context_compressor.handle_tool_call(function_name, function_args, messages=messages)
                    _ce_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                    logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._memory_manager and self._memory_manager.has_tool(function_name):
                # Memory provider tools (hindsight_retain, honcho_search, etc.)
                # These are not in the tool registry — route through MemoryManager.
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _mem_result = None
                try:
                    function_result = self._memory_manager.handle_tool_call(function_name, function_args)
                    _mem_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                    logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode:
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        skip_pre_tool_call_hook=True,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        skip_pre_tool_call_hook=True,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            if isinstance(function_result, str):
                result_preview = function_result if self.verbose_logging else (
                    function_result[:200] if len(function_result) > 200 else function_result
                )
                _result_len = len(function_result)
            else:
                # Multimodal dict result (_multimodal=True) — not sliceable as string
                result_preview = function_result
                _result_len = len(str(function_result))

            # Log tool errors to the persistent error log so [error] tags
            # in the UI always have a corresponding detailed entry on disk.
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if not _execution_blocked:
                function_result = self._append_guardrail_observation(
                    function_name,
                    function_args,
                    function_result,
                    failed=_is_error_result,
                )
                result_preview = function_result if self.verbose_logging else (
                    function_result[:200] if len(function_result) > 200 else function_result
                )
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, _result_len)

            # Track file-mutation outcome for the turn-end verifier.  See
            # the concurrent path for the rationale; both paths must feed
            # the same state so the footer reflects every tool call in the
            # turn, not just the parallel ones.
            if not _execution_blocked:
                try:
                    self._record_file_mutation_result(
                        function_name, function_args, function_result, _is_error_result,
                    )
                except Exception as _ver_err:
                    logging.debug("file-mutation verifier record failed: %s", _ver_err)

            if not _execution_blocked and self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=_is_error_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                _log_result = _multimodal_text_summary(function_result)
                logging.debug(f"Tool result ({len(_log_result)} chars): {_log_result}")

            if not _execution_blocked and self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=function_name,
                tool_use_id=tool_call.id,
                env=get_active_env(effective_task_id),
            ) if not _is_multimodal_tool_result(function_result) else function_result

            # Discover subdirectory context files from tool arguments
            subdir_hints = self._subdirectory_hints.check_tool_call(function_name, function_args)
            if subdir_hints:
                if _is_multimodal_tool_result(function_result):
                    _append_subdir_hint_to_multimodal(function_result, subdir_hints)
                else:
                    function_result += subdir_hints

            # Unwrap _multimodal dicts to an OpenAI-style content list
            # (see parallel path for rationale). String results pass through.
            _tool_content = (
                function_result["content"]
                if _is_multimodal_tool_result(function_result)
                else function_result
            )
            tool_msg = {
                "role": "tool",
                "name": function_name,
                "content": _tool_content,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            # ── Per-tool /steer drain ───────────────────────────────────
            # Drain pending steer BETWEEN individual tool calls so the
            # injection lands as soon as a tool finishes — not after the
            # entire batch.  The model sees it on the next API iteration.
            self._apply_pending_steer_to_tool_results(messages, 1)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(self._wrap_verbose("Result: ", function_result))
                else:
                    _fr_str = function_result if isinstance(function_result, str) else str(function_result)
                    response_preview = _fr_str[:self.log_prefix_chars] + "..." if len(_fr_str) > self.log_prefix_chars else _fr_str
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "name": skipped_name,
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools_seq = len(assistant_message.tool_calls)
        if num_tools_seq > 0:
            enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id))

        # ── /steer injection ──────────────────────────────────────────────
        # See _execute_tool_calls_parallel for the rationale. Same hook,
        # applied to sequential execution as well.
        if num_tools_seq > 0:
            self._apply_pending_steer_to_tool_results(messages, num_tools_seq)


    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """Request a summary when max iterations are reached. Returns the final response text."""
        print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # Build API messages, stripping internal-only fields
            # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                self._copy_reasoning_content_for_api(msg, api_msg)
                for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                    api_msg.pop(internal_field, None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # Same safety net as the main loop: repair tool-call/result
            # pairing before asking for a final summary.  Compression and
            # session resume can leave a tool result whose parent assistant
            # tool_call was summarized away; Responses API rejects that as
            # "No tool call found for function call output".
            api_messages = self._sanitize_api_messages(api_messages)

            # Same safety net as the main loop: drop thinking-only assistant
            # turns so Anthropic-family providers don't 400 the summary call.
            api_messages = self._drop_thinking_only_and_merge_users(api_messages)

            summary_extra_body = {}
            try:
                from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE as _OMIT_TEMP
            except Exception:
                _fixed_temperature_for_model = None
                _OMIT_TEMP = None
            _raw_summary_temp = (
                _fixed_temperature_for_model(self.model, self.base_url)
                if _fixed_temperature_for_model is not None
                else None
            )
            _omit_summary_temperature = _raw_summary_temp is _OMIT_TEMP
            _summary_temperature = None if _omit_summary_temperature else _raw_summary_temp
            _is_nous = "nousresearch" in self._base_url_lower
            # LM Studio uses top-level `reasoning_effort` (not extra_body.reasoning).
            # Mirror ChatCompletionsTransport.build_kwargs() so the summary path
            # — which calls chat.completions.create() directly without going
            # through the transport — sends the same shape the transport does.
            _is_lmstudio_summary = (
                (self.provider or "").strip().lower() == "lmstudio"
                and self._supports_reasoning_extra_body()
            )
            _lm_reasoning_effort: str | None = (
                self._resolve_lmstudio_summary_reasoning_effort()
                if _is_lmstudio_summary else None
            )
            if not _is_lmstudio_summary and self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                from agent.portal_tags import nous_portal_tags as _portal_tags
                summary_extra_body["tags"] = _portal_tags()

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                _ct_sum = self._get_transport()
                _cnr_sum = _ct_sum.normalize_response(summary_response)
                final_response = (_cnr_sum.content or "").strip()
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if _summary_temperature is not None:
                    summary_kwargs["temperature"] = _summary_temperature
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                if _lm_reasoning_effort is not None:
                    summary_kwargs["reasoning_effort"] = _lm_reasoning_effort

                # Include provider routing preferences
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences and (
                    (self.provider or "").strip().lower() == "openrouter"
                    or self._is_openrouter_url()
                ):
                    summary_extra_body["provider"] = provider_preferences

                # Pareto Code router plugin — model-gated. Same shape as
                # the main-loop emission so summary calls on
                # openrouter/pareto-code respect the user's coding-score floor.
                if (
                    self.model == "openrouter/pareto-code"
                    and (
                        (self.provider or "").strip().lower() == "openrouter"
                        or self._is_openrouter_url()
                    )
                    and self.openrouter_min_coding_score is not None
                    and self.openrouter_min_coding_score != ""
                ):
                    try:
                        _ps = float(self.openrouter_min_coding_score)
                    except (TypeError, ValueError):
                        _ps = None
                    if _ps is not None and 0.0 <= _ps <= 1.0:
                        summary_extra_body["plugins"] = [
                            {"id": "pareto-router", "min_coding_score": _ps}
                        ]

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    _tsum = self._get_transport()
                    _ant_kw = _tsum.build_kwargs(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=self._is_anthropic_oauth,
                                   preserve_dots=self._anthropic_preserve_dots())
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _summary_result = _tsum.normalize_response(summary_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_summary_result.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)
                    _summary_result = self._get_transport().normalize_response(summary_response)
                    final_response = (_summary_result.content or "").strip()

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                # Retry summary generation
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    _ct_retry = self._get_transport()
                    _cnr_retry = _ct_retry.normalize_response(retry_response)
                    final_response = (_cnr_retry.content or "").strip()
                elif self.api_mode == "anthropic_messages":
                    _tretry = self._get_transport()
                    _ant_kw2 = _tretry.build_kwargs(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=self._is_anthropic_oauth,
                                    max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                    preserve_dots=self._anthropic_preserve_dots())
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_result = _tretry.normalize_response(retry_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_retry_result.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if _summary_temperature is not None:
                        summary_kwargs["temperature"] = _summary_temperature
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if _lm_reasoning_effort is not None:
                        summary_kwargs["reasoning_effort"] = _lm_reasoning_effort
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)
                    _retry_result = self._get_transport().normalize_response(summary_response)
                    final_response = (_retry_result.content or "").strip()

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    if final_response:
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling until completion.

        Args:
            user_message (str): The user's message/question
            system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
            conversation_history (List[Dict]): Previous conversation messages (optional)
            task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)
            stream_callback: Optional callback invoked with each text delta during streaming.
                Used by the TTS pipeline to start audio generation before the full response.
                When None (default), API calls use the standard non-streaming path.
            persist_user_message: Optional clean user message to store in
                transcripts/history when user_message contains API-only
                synthetic prefixes.
                    or queuing follow-up prefetch work.

        Returns:
            Dict: Complete conversation result with final response and message history
        """
        # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
        # Installed once, transparent when streams are healthy, prevents crash on write.
        _install_safe_stdio()

        self._ensure_db_session()

        # Tell auxiliary_client what the live main provider/model are for
        # this turn. Used by tools whose behaviour depends on the active
        # main model (e.g. vision_analyze's native fast path) so they see
        # the CLI/gateway override instead of the stale config.yaml
        # default. Idempotent — fine to call every turn.
        try:
            from agent.auxiliary_client import set_runtime_main
            set_runtime_main(
                getattr(self, "provider", "") or "",
                getattr(self, "model", "") or "",
            )
        except Exception:
            pass

        # Tag all log records on this thread with the session ID so
        # ``hermes logs --session <id>`` can filter a single conversation.
        from hermes_logging import set_session_context
        set_session_context(self.session_id)

        # Bind the skill write-origin ContextVar for this thread so tool
        # handlers (e.g. skill_manage create) can tell whether they are
        # running inside the background self-improvement review fork vs.
        # a foreground user-directed turn. Set at the top of each call;
        # the review fork runs on its own thread with a fresh context,
        # so the foreground value here does not leak into it.
        from tools.skill_provenance import set_current_write_origin
        set_current_write_origin(getattr(self, "_memory_write_origin", "assistant_tool"))

        # If the previous turn activated fallback, restore the primary
        # runtime so this turn gets a fresh attempt with the preferred model.
        # No-op when _fallback_activated is False (gateway, first turn, etc.).
        self._restore_primary_runtime()

        # Sanitize surrogate characters from user input.  Clipboard paste from
        # rich-text editors (Google Docs, Word, etc.) can inject lone surrogates
        # that are invalid UTF-8 and crash JSON serialization in the OpenAI SDK.
        if isinstance(user_message, str):
            user_message = _sanitize_surrogates(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = _sanitize_surrogates(persist_user_message)

        # Store stream callback for _interruptible_api_call to pick up
        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message
        # Generate unique task_id if not provided to isolate VMs between concurrent tasks
        effective_task_id = task_id or str(uuid.uuid4())
        # Expose the active task_id so tools running mid-turn (e.g. delegate_task
        # in delegate_tool.py) can identify this agent for the cross-agent file
        # state registry.  Set BEFORE any tool dispatch so snapshots taken at
        # child-launch time see the parent's real id, not None.
        self._current_task_id = effective_task_id
        
        # Reset retry counters and iteration budget at the start of each turn
        # so subagent usage from a previous turn doesn't eat into the next one.
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._thinking_prefill_retries = 0
        self._post_tool_empty_retried = False
        self._last_content_with_tools = None
        self._last_content_tools_all_housekeeping = False
        self._mute_post_response = False
        self._unicode_sanitization_passes = 0
        self._tool_guardrails.reset_for_turn()
        self._tool_guardrail_halt_decision = None
        # True until the server rejects an image_url content part with an error
        # like "Only 'text' content type is supported."  Set to False on first
        # rejection and kept False for the rest of the session so we never re-send
        # images to a text-only endpoint.  Scoped per `_run()` call, not per instance.
        self._vision_supported = True

        # Pre-turn connection health check: detect and clean up dead TCP
        # connections left over from provider outages or dropped streams.
        # This prevents the next API call from hanging on a zombie socket.
        if self.api_mode != "anthropic_messages":
            try:
                if self._cleanup_dead_connections():
                    self._emit_status(
                        "🔌 Detected stale connections from a previous provider "
                        "issue — cleaned up automatically. Proceeding with fresh "
                        "connection."
                    )
            except Exception:
                pass
        # Replay compression warning through status_callback for gateway
        # platforms (the callback was not wired during __init__).
        if self._compression_warning:
            self._replay_compression_warning()
            self._compression_warning = None  # send once

        # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
        # They are initialized in __init__ and must persist across run_conversation
        # calls so that nudge logic accumulates correctly in CLI mode.
        self.iteration_budget = IterationBudget(self.max_iterations)

        # Log conversation turn start for debugging/observability
        _preview_text = _summarize_user_message_for_log(user_message)
        _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
        _msg_preview = _msg_preview.replace("\n", " ")
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            self.session_id or "none", self.model, self.provider or "unknown",
            self.platform or "unknown", len(conversation_history or []),
            _msg_preview,
        )

        # Initialize conversation (copy to avoid mutating the caller's list)
        messages = list(conversation_history) if conversation_history else []

        # Hydrate todo store from conversation history (gateway creates a fresh
        # AIAgent per message, so the in-memory store is empty -- we need to
        # recover the todo state from the most recent todo tool response in history)
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)

        # Hydrate per-session nudge counters from persisted history.
        # Gateway creates a fresh AIAgent per inbound message (cache miss /
        # 1h idle eviction / config-signature mismatch / process restart), so
        # _turns_since_memory and _user_turn_count start at 0 every turn and
        # the memory.nudge_interval trigger may never be reached. Reconstruct
        # an effective count from prior user turns in conversation_history.
        # Idempotent: a cached agent that already accumulated counters keeps
        # them; only a freshly-built agent with empty in-memory state hydrates.
        # See issue #22357.
        if conversation_history and self._user_turn_count == 0:
            prior_user_turns = sum(
                1 for m in conversation_history if m.get("role") == "user"
            )
            if prior_user_turns > 0:
                self._user_turn_count = prior_user_turns
                if self._memory_nudge_interval > 0 and self._turns_since_memory == 0:
                    # % preserves original 1-in-N cadence rather than firing a
                    # review immediately on resume (which would surprise users
                    # whose session happened to land just past a multiple of N).
                    self._turns_since_memory = prior_user_turns % self._memory_nudge_interval


        # Prefill messages (few-shot priming) are injected at API-call time only,
        # never stored in the messages list. This keeps them ephemeral: they won't
        # be saved to session DB, session logs, or batch trajectories, but they're
        # automatically re-applied on every API call (including session continuations).
        
        # Track user turns for memory flush and periodic nudge logic
        self._user_turn_count += 1

        # Reset the streaming context scrubber at the top of each turn so a
        # hung span from a prior interrupted stream can't taint this turn's
        # output.
        scrubber = getattr(self, "_stream_context_scrubber", None)
        if scrubber is not None:
            scrubber.reset()
        # Reset the think scrubber for the same reason — an interrupted
        # prior stream may have left us inside an unterminated block.
        think_scrubber = getattr(self, "_stream_think_scrubber", None)
        if think_scrubber is not None:
            think_scrubber.reset()

        # Preserve the original user message (no nudge injection).
        original_user_message = persist_user_message if persist_user_message is not None else user_message

        # Track memory nudge trigger (turn-based, checked here).
        # Skill trigger is checked AFTER the agent loop completes, based on
        # how many tool iterations THIS turn used.
        _should_review_memory = False
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                _should_review_memory = True
                self._turns_since_memory = 0

        # Add user message
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx
        
        if not self.quiet_mode:
            _print_preview = _summarize_user_message_for_log(user_message)
            self._safe_print(f"💬 Starting conversation: '{_print_preview[:60]}{'...' if len(_print_preview) > 60 else ''}'")
        
        # ── System prompt (cached per session for prefix caching) ──
        # Built once on first call, reused for all subsequent calls.
        # Only rebuilt after context compression events (which invalidate
        # the cache and reload memory from disk).
        #
        # For continuing sessions (gateway creates a fresh AIAgent per
        # message), we load the stored system prompt from the session DB
        # instead of rebuilding.  Rebuilding would pick up memory changes
        # from disk that the model already knows about (it wrote them!),
        # producing a different system prompt and breaking the Anthropic
        # prefix cache.
        if self._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and self._session_db:
                try:
                    session_row = self._session_db.get_session(self.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass  # Fall through to build fresh

            if stored_prompt:
                # Continuing session — reuse the exact system prompt from
                # the previous turn so the Anthropic cache prefix matches.
                self._cached_system_prompt = stored_prompt
            else:
                # First turn of a new session — build from scratch.
                self._cached_system_prompt = self._build_system_prompt(system_message)
                # Plugin hook: on_session_start
                # Fired once when a brand-new session is created (not on
                # continuation).  Plugins can use this to initialise
                # session-scoped state (e.g. warm a memory cache).
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_start",
                        session_id=self.session_id,
                        model=self.model,
                        platform=getattr(self, "platform", None) or "",
                    )
                except Exception as exc:
                    logger.warning("on_session_start hook failed: %s", exc)

                # Store the system prompt snapshot in SQLite
                if self._session_db:
                    try:
                        self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                    except Exception as e:
                        logger.debug("Session DB update_system_prompt failed: %s", e)

        active_system_prompt = self._cached_system_prompt

        # ── Preflight context compression ──
        # Before entering the main loop, check if the loaded conversation
        # history already exceeds the model's context threshold.  This handles
        # cases where a user switches to a model with a smaller context window
        # while having a large existing session — compress proactively rather
        # than waiting for an API error (which might be caught as a non-retryable
        # 4xx and abort the request entirely).
        if (
            self.compression_enabled
            and len(messages) > self.context_compressor.protect_first_n
                                + self.context_compressor.protect_last_n + 1
        ):
            # Include tool schema tokens — with many tools these can add
            # 20-30K+ tokens that the old sys+msg estimate missed entirely.
            _preflight_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=self.tools or None,
            )

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                self._emit_status(
                    f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                    f">= {self.context_compressor.threshold_tokens:,} threshold. "
                    "This may take a moment."
                )
                # May need multiple passes for very large sessions with small
                # context windows (each pass summarises the middle N turns).
                for _pass in range(3):
                    _orig_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=_preflight_tokens,
                        task_id=effective_task_id,
                    )
                    if len(messages) >= _orig_len:
                        break  # Cannot compress further
                    # Compression created a new session — clear the history
                    # reference so _flush_messages_to_session_db writes ALL
                    # compressed messages to the new session's SQLite, not
                    # skipping them because conversation_history is still the
                    # pre-compression length.
                    conversation_history = None
                    # Fix: reset retry counters after compression so the model
                    # gets a fresh budget on the compressed context.  Without
                    # this, pre-compression retries carry over and the model
                    # hits "(empty)" immediately after compression-induced
                    # context loss.
                    self._empty_content_retries = 0
                    self._thinking_prefill_retries = 0
                    self._last_content_with_tools = None
                    self._last_content_tools_all_housekeeping = False
                    self._mute_post_response = False
                    # Re-estimate after compression
                    _preflight_tokens = estimate_request_tokens_rough(
                        messages,
                        system_prompt=active_system_prompt or "",
                        tools=self.tools or None,
                    )
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # Under threshold

        # Plugin hook: pre_llm_call
        # Fired once per turn before the tool-calling loop.  Plugins can
        # return a dict with a ``context`` key (or a plain string) whose
        # value is appended to the current turn's user message.
        #
        # Context is ALWAYS injected into the user message, never the
        # system prompt.  This preserves the prompt cache prefix — the
        # system prompt stays identical across turns so cached tokens
        # are reused.  The system prompt is Hermes's territory; plugins
        # contribute context alongside the user's input.
        #
        # All injected context is ephemeral (not persisted to session DB).
        _plugin_user_context = ""
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _pre_results = _invoke_hook(
                "pre_llm_call",
                session_id=self.session_id,
                user_message=original_user_message,
                conversation_history=list(messages),
                is_first_turn=(not bool(conversation_history)),
                model=self.model,
                platform=getattr(self, "platform", None) or "",
                sender_id=getattr(self, "_user_id", None) or "",
            )
            _ctx_parts: list[str] = []
            for r in _pre_results:
                if isinstance(r, dict) and r.get("context"):
                    _ctx_parts.append(str(r["context"]))
                elif isinstance(r, str) and r.strip():
                    _ctx_parts.append(r)
            if _ctx_parts:
                _plugin_user_context = "\n\n".join(_ctx_parts)
        except Exception as exc:
            logger.warning("pre_llm_call hook failed: %s", exc)

        # Main conversation loop
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_tool_call_retries = 0
        truncated_response_prefix = ""
        compression_attempts = 0
        _turn_exit_reason = "unknown"  # Diagnostic: why the loop ended

        # Per-turn file-mutation verifier state.  Keyed by resolved path;
        # each failed ``write_file`` / ``patch`` call records the error
        # preview.  Later successful writes to the same path remove the
        # entry (the model recovered).  At end-of-turn, any entries still
        # present are surfaced in an advisory footer so the model cannot
        # over-claim success while the file is actually unchanged on disk.
        self._turn_failed_file_mutations: Dict[str, Dict[str, Any]] = {}
        
        # Record the execution thread so interrupt()/clear_interrupt() can
        # scope the tool-level interrupt signal to THIS agent's thread only.
        # Must be set before any thread-scoped interrupt syncing.
        self._execution_thread_id = threading.current_thread().ident

        # Always clear stale per-thread state from a previous turn. If an
        # interrupt arrived before startup finished, preserve it and bind it
        # to this execution thread now instead of dropping it on the floor.
        _set_interrupt(False, self._execution_thread_id)
        if self._interrupt_requested:
            _set_interrupt(True, self._execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            self._interrupt_message = None
            self._interrupt_thread_signal_pending = False

        # Notify memory providers of the new turn so cadence tracking works.
        # Must happen BEFORE prefetch_all() so providers know which turn it is
        # and can gate context/dialectic refresh via contextCadence/dialecticCadence.
        if self._memory_manager:
            try:
                _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
                self._memory_manager.on_turn_start(self._user_turn_count, _turn_msg)
            except Exception:
                pass

        # External memory provider: prefetch once before the tool loop.
        # Reuse the cached result on every iteration to avoid re-calling
        # prefetch_all() on each tool call (10 tool calls = 10x latency + cost).
        # Use original_user_message (clean input) — user_message may contain
        # injected skill content that bloats / breaks provider queries.
        _ext_prefetch_cache = ""
        if self._memory_manager:
            try:
                _query = original_user_message if isinstance(original_user_message, str) else ""
                _ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""
            except Exception:
                pass

        while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) or self._budget_grace_call:
            # Reset per-turn checkpoint dedup so each iteration can take one snapshot
            self._checkpoint_mgr.new_turn()

            # Check for interrupt request (e.g., user sent new message)
            if self._interrupt_requested:
                interrupted = True
                _turn_exit_reason = "interrupted_by_user"
                if not self.quiet_mode:
                    self._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1
            self._api_call_count = api_call_count
            self._touch_activity(f"starting API call #{api_call_count}")

            # Grace call: the budget is exhausted but we gave the model one
            # more chance.  Consume the grace flag so the loop exits after
            # this iteration regardless of outcome.
            if self._budget_grace_call:
                self._budget_grace_call = False
            elif not self.iteration_budget.consume():
                _turn_exit_reason = "budget_exhausted"
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
                break

            # Fire step_callback for gateway hooks (agent:step event)
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _idx, _m in enumerate(reversed(messages)):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            _fwd_start = len(messages) - _idx
                            _results_by_id = {}
                            for _tm in messages[_fwd_start:]:
                                if _tm.get("role") != "tool":
                                    break
                                _tcid = _tm.get("tool_call_id")
                                if _tcid:
                                    _results_by_id[_tcid] = _tm.get("content", "")
                            prev_tools = [
                                {
                                    "name": tc["function"]["name"],
                                    "result": _results_by_id.get(tc.get("id")),
                                    "arguments": tc["function"].get("arguments"),
                                }
                                for tc in _m["tool_calls"]
                                if isinstance(tc, dict)
                            ]
                            break
                    self.step_callback(api_call_count, prev_tools)
                except Exception as _step_err:
                    logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

            # Track tool-calling iterations for skill nudge.
            # Counter resets whenever skill_manage is actually used.
            if (self._skill_nudge_interval > 0
                    and "skill_manage" in self.valid_tool_names):
                self._iters_since_skill += 1
            
            # ── Pre-API-call /steer drain ──────────────────────────────────
            # If a /steer arrived during the previous API call (while the model
            # was thinking), drain it now — before we build api_messages — so
            # the model sees the steer text on THIS iteration.  Without this,
            # steers sent during an API call only land after the NEXT tool batch,
            # which may never come if the model returns a final response.
            #
            # We scan backwards for the last tool-role message in the messages
            # list.  If found, the steer is appended there.  If not (first
            # iteration, no tools yet), the steer stays pending for the next
            # tool batch — injecting into a user message would break role
            # alternation, and there's no tool output to piggyback on.
            _pre_api_steer = self._drain_pending_steer()
            if _pre_api_steer:
                _injected = False
                for _si in range(len(messages) - 1, -1, -1):
                    _sm = messages[_si]
                    if isinstance(_sm, dict) and _sm.get("role") == "tool":
                        marker = f"\n\nUser guidance: {_pre_api_steer}"
                        existing = _sm.get("content", "")
                        if isinstance(existing, str):
                            _sm["content"] = existing + marker
                        else:
                            # Multimodal content blocks — append text block
                            try:
                                blocks = list(existing) if existing else []
                                blocks.append({"type": "text", "text": marker})
                                _sm["content"] = blocks
                            except Exception:
                                pass
                        _injected = True
                        logger.debug(
                            "Pre-API-call steer drain: injected into tool msg at index %d",
                            _si,
                        )
                        break
                if not _injected:
                    # No tool message to inject into — put it back so
                    # the post-tool-execution drain picks it up later.
                    _lock = getattr(self, "_pending_steer_lock", None)
                    if _lock is not None:
                        with _lock:
                            if self._pending_steer:
                                self._pending_steer = self._pending_steer + "\n" + _pre_api_steer
                            else:
                                self._pending_steer = _pre_api_steer
                    else:
                        existing = getattr(self, "_pending_steer", None)
                        self._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

            # Prepare messages for API call
            # If we have an ephemeral system prompt, prepend it to the messages
            # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
            # However, providers like Moonshot AI require a separate 'reasoning_content' field
            # on assistant messages with tool_calls. We handle both cases here.
            request_logger = getattr(self, "logger", None) or logging.getLogger(__name__)
            repaired_tool_calls = self._sanitize_tool_call_arguments(
                messages,
                logger=request_logger,
                session_id=self.session_id,
            )
            if repaired_tool_calls > 0:
                request_logger.info(
                    "Sanitized %s corrupted tool_call arguments before request (session=%s)",
                    repaired_tool_calls,
                    self.session_id or "-",
                )

            # Defensive: repair malformed role-alternation before API call.
            # Catches cases where the history got wedged into a
            # ``tool → user`` or ``user → user`` tail (e.g. after empty-
            # response scaffolding was stripped and a new user message
            # landed after an orphan tool result). Most providers return
            # empty content on malformed sequences, which would otherwise
            # retrigger the empty-retry loop indefinitely.
            repaired_seq = self._repair_message_sequence(messages)
            if repaired_seq > 0:
                request_logger.info(
                    "Repaired %s message-alternation violations before request (session=%s)",
                    repaired_seq,
                    self.session_id or "-",
                )

            api_messages = []
            for idx, msg in enumerate(messages):
                api_msg = msg.copy()

                # Inject ephemeral context into the current turn's user message.
                # Sources: memory manager prefetch + plugin pre_llm_call hooks
                # with target="user_message" (the default).  Both are
                # API-call-time only — the original message in `messages` is
                # never mutated, so nothing leaks into session persistence.
                if idx == current_turn_user_idx and msg.get("role") == "user":
                    _injections = []
                    if _ext_prefetch_cache:
                        _fenced = build_memory_context_block(_ext_prefetch_cache)
                        if _fenced:
                            _injections.append(_fenced)
                    if _plugin_user_context:
                        _injections.append(_plugin_user_context)
                    if _injections:
                        _base = api_msg.get("content", "")
                        if isinstance(_base, str):
                            api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

                # For ALL assistant messages, pass reasoning back to the API
                # This ensures multi-turn reasoning context is preserved
                self._copy_reasoning_content_for_api(msg, api_msg)

                # Remove 'reasoning' field - it's for trajectory storage only
                # We've copied it to 'reasoning_content' for the API above
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
                if "finish_reason" in api_msg:
                    api_msg.pop("finish_reason")
                # Strip internal thinking-prefill marker
                api_msg.pop("_thinking_prefill", None)
                # Strip Codex Responses API fields (call_id, response_item_id) for
                # strict providers like Mistral, Fireworks, etc. that reject unknown fields.
                # Uses new dicts so the internal messages list retains the fields
                # for Codex Responses compatibility.
                if self._should_sanitize_tool_calls():
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
                # The signature field helps maintain reasoning continuity
                api_messages.append(api_msg)

            # Build the final system message: cached prompt + ephemeral system prompt.
            # Ephemeral additions are API-call-time only (not persisted to session DB).
            # External recall context is injected into the user message, not the system
            # prompt, so the stable cache prefix remains unchanged.
            #
            # NOTE: Plugin context from pre_llm_call hooks is injected into the
            # user message (see injection block above), NOT the system prompt.
            # This is intentional — system prompt modifications break the prompt
            # cache prefix.  The system prompt is reserved for Hermes internals.
            #
            # Hermes invariant: the system prompt is built ONCE per session
            # (cached on ``_cached_system_prompt``) and replayed verbatim on
            # every turn.  We send it as a single content string so the
            # bytes are byte-stable across turns and upstream prompt caches
            # stay warm.
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages

            # Inject ephemeral prefill messages right after the system prompt
            # but before conversation history. Same API-call-time-only pattern.
            if self.prefill_messages:
                sys_offset = 1 if (api_messages and api_messages[0].get("role") == "system") else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # Apply Anthropic prompt caching for Claude models on native
            # Anthropic, OpenRouter, and third-party Anthropic-compatible
            # gateways. Auto-detected: if ``_use_prompt_caching`` is set,
            # inject cache_control breakpoints (system + last 3 messages)
            # to reduce input token costs by ~75% on multi-turn
            # conversations.
            if self._use_prompt_caching:
                api_messages = apply_anthropic_cache_control(
                    api_messages,
                    cache_ttl=self._cache_ttl,
                    native_anthropic=self._use_native_cache_layout,
                )

            # Safety net: strip orphaned tool results / add stubs for missing
            # results before sending to the API.  Runs unconditionally — not
            # gated on context_compressor — so orphans from session loading or
            # manual message manipulation are always caught.
            api_messages = self._sanitize_api_messages(api_messages)

            # Drop thinking-only assistant turns (reasoning but no visible
            # output and no tool_calls) and merge any adjacent user messages
            # left behind. Prevents Anthropic 400s ("The final block in an
            # assistant message cannot be `thinking`.") and equivalent errors
            # from third-party Anthropic-compatible gateways that can't replay
            # a thinking-only turn. Runs on the per-call copy only — the
            # stored conversation history keeps the reasoning block for the
            # UI transcript and session persistence.
            api_messages = self._drop_thinking_only_and_merge_users(api_messages)

            # Normalize message whitespace and tool-call JSON for consistent
            # prefix matching.  Ensures bit-perfect prefixes across turns,
            # which enables KV cache reuse on local inference servers
            # (llama.cpp, vLLM, Ollama) and improves cache hit rates for
            # cloud providers.  Operates on api_messages (the API copy) so
            # the original conversation history in `messages` is untouched.
            for am in api_messages:
                if isinstance(am.get("content"), str):
                    am["content"] = am["content"].strip()
            for am in api_messages:
                tcs = am.get("tool_calls")
                if not tcs:
                    continue
                new_tcs = []
                for tc in tcs:
                    if isinstance(tc, dict) and "function" in tc:
                        try:
                            args_obj = json.loads(tc["function"]["arguments"])
                            tc = {**tc, "function": {
                                **tc["function"],
                                "arguments": json.dumps(
                                    args_obj, separators=(",", ":"),
                                    sort_keys=True,
                                ),
                            }}
                        except Exception:
                            tc["function"]["arguments"] = _repair_tool_call_arguments(
                                tc["function"]["arguments"],
                                tc["function"].get("name", "?"),
                            )
                    new_tcs.append(tc)
                am["tool_calls"] = new_tcs

            # Proactively strip any surrogate characters before the API call.
            # Models served via Ollama (Kimi K2.5, GLM-5, Qwen) can return
            # lone surrogates (U+D800-U+DFFF) that crash json.dumps() inside
            # the OpenAI SDK. Sanitizing here prevents the 3-retry cycle.
            _sanitize_messages_surrogates(api_messages)

            # Calculate approximate request size for logging
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = estimate_messages_tokens_rough(api_messages)
            
            # Thinking spinner for quiet mode (animated during API call)
            thinking_spinner = None
            
            if not self.quiet_mode:
                self._vprint(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
                self._vprint(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
                self._vprint(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            else:
                # Animated thinking spinner in quiet mode
                face = random.choice(KawaiiSpinner.get_thinking_faces())
                verb = random.choice(KawaiiSpinner.get_thinking_verbs())
                if self.thinking_callback:
                    # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                    # (works in both streaming and non-streaming modes)
                    self.thinking_callback(f"{face} {verb}...")
                elif not self._has_stream_consumers() and self._should_start_quiet_spinner():
                    # Raw KawaiiSpinner only when no streaming consumers and the
                    # spinner output has a safe sink.
                    spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                    thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=self._print_fn)
                    thinking_spinner.start()
            
            # Log request details if verbose
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = self._api_max_retries
            primary_recovery_attempted = False
            max_compression_attempts = 3
            codex_auth_retry_attempted=False
            anthropic_auth_retry_attempted=False
            nous_auth_retry_attempted=False
            copilot_auth_retry_attempted=False
            thinking_sig_retry_attempted = False
            image_shrink_retry_attempted = False
            oauth_1m_beta_retry_attempted = False
            llama_cpp_grammar_retry_attempted = False
            has_retried_429 = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # Guard against UnboundLocalError if all retries fail
            api_kwargs = None  # Guard against UnboundLocalError in except handler

            while retry_count < max_retries:
                # ── Nous Portal rate limit guard ──────────────────────
                # If another session already recorded that Nous is rate-
                # limited, skip the API call entirely.  Each attempt
                # (including SDK-level retries) counts against RPH and
                # deepens the rate limit hole.
                if self.provider == "nous":
                    try:
                        from agent.nous_rate_guard import (
                            nous_rate_limit_remaining,
                            format_remaining as _fmt_nous_remaining,
                        )
                        _nous_remaining = nous_rate_limit_remaining()
                        if _nous_remaining is not None and _nous_remaining > 0:
                            _nous_msg = (
                                f"Nous Portal rate limit active — "
                                f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                            )
                            self._vprint(
                                f"{self.log_prefix}⏳ {_nous_msg} Trying fallback...",
                                force=True,
                            )
                            self._emit_status(f"⏳ {_nous_msg}")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue
                            # No fallback available — return with clear message
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": (
                                    f"⏳ {_nous_msg}\n\n"
                                    "No fallback provider available. "
                                    "Try again after the reset, or add a "
                                    "fallback provider in config.yaml."
                                ),
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": _nous_msg,
                            }
                    except ImportError:
                        pass
                    except Exception:
                        pass  # Never let rate guard break the agent loop

                try:
                    self._reset_stream_delivery_tracking()
                    api_kwargs = self._build_api_kwargs(api_messages)
                    if self._force_ascii_payload:
                        _sanitize_structure_non_ascii(api_kwargs)
                    if self.api_mode == "codex_responses":
                        api_kwargs = self._get_transport().preflight_kwargs(api_kwargs, allow_stream=False)

                    try:
                        from hermes_cli.plugins import invoke_hook as _invoke_hook
                        _invoke_hook(
                            "pre_api_request",
                            task_id=effective_task_id,
                            session_id=self.session_id or "",
                            platform=self.platform or "",
                            model=self.model,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_mode=self.api_mode,
                            api_call_count=api_call_count,
                            message_count=len(api_messages),
                            tool_count=len(self.tools or []),
                            approx_input_tokens=approx_tokens,
                            request_char_count=total_chars,
                            max_tokens=self.max_tokens,
                        )
                    except Exception:
                        pass

                    if env_var_enabled("HERMES_DUMP_REQUESTS"):
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    # Always prefer the streaming path — even without stream
                    # consumers.  Streaming gives us fine-grained health
                    # checking (90s stale-stream detection, 60s read timeout)
                    # that the non-streaming path lacks.  Without this,
                    # subagents and other quiet-mode callers can hang
                    # indefinitely when the provider keeps the connection
                    # alive with SSE pings but never delivers a response.
                    # The streaming path is a no-op for callbacks when no
                    # consumers are registered, and falls back to non-
                    # streaming automatically if the provider doesn't
                    # support it.
                    def _stop_spinner():
                        nonlocal thinking_spinner
                        if thinking_spinner:
                            thinking_spinner.stop("")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")

                    _use_streaming = True
                    # Provider signaled "stream not supported" on a previous
                    # attempt — switch to non-streaming for the rest of this
                    # session instead of re-failing every retry.
                    if getattr(self, "_disable_streaming", False):
                        _use_streaming = False
                    # CopilotACPClient communicates via subprocess stdio and
                    # returns a plain SimpleNamespace — not an iterable
                    # stream.  Mirror the ACP exclusion used for Responses
                    # API upgrade (lines ~1083-1085).
                    elif (
                        self.provider == "copilot-acp"
                        or str(self.base_url or "").lower().startswith("acp://copilot")
                        or str(self.base_url or "").lower().startswith("acp+tcp://")
                    ):
                        _use_streaming = False
                    elif not self._has_stream_consumers():
                        # No display/TTS consumer. Still prefer streaming for
                        # health checking, but skip for Mock clients in tests
                        # (mocks return SimpleNamespace, not stream iterators).
                        from unittest.mock import Mock
                        if isinstance(getattr(self, "client", None), Mock):
                            _use_streaming = False

                    if _use_streaming:
                        response = self._interruptible_streaming_api_call(
                            api_kwargs, on_first_delta=_stop_spinner
                        )
                    else:
                        response = self._interruptible_api_call(api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    
                    # Stop thinking spinner silently -- the response box or tool
                    # execution messages that follow are more informative.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                    
                    if self.verbose_logging:
                        # Log response with provider info if available
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    # Validate response shape before proceeding
                    response_invalid = False
                    error_details = []
                    if self.api_mode == "codex_responses":
                        _ct_v = self._get_transport()
                        if not _ct_v.validate_response(response):
                            if response is None:
                                response_invalid = True
                                error_details.append("response is None")
                            else:
                                # Provider returned a terminal failure (e.g. quota exhaustion).
                                # Treat as invalid so the fallback chain is triggered instead of
                                # letting the error bubble up outside the retry/fallback loop.
                                _codex_resp_status = str(getattr(response, "status", "") or "").strip().lower()
                                if _codex_resp_status in {"failed", "cancelled"}:
                                    _codex_error_obj = getattr(response, "error", None)
                                    _codex_error_msg = (
                                        _codex_error_obj.get("message") if isinstance(_codex_error_obj, dict)
                                        else str(_codex_error_obj) if _codex_error_obj
                                        else f"Responses API returned status '{_codex_resp_status}'"
                                    )
                                    logging.warning(
                                        "Codex response status='%s' (error=%s). Routing to fallback. %s",
                                        _codex_resp_status, _codex_error_msg,
                                        self._client_log_context(),
                                    )
                                    response_invalid = True
                                    error_details.append(f"response.status={_codex_resp_status}: {_codex_error_msg}")
                                else:
                                    # output_text fallback: stream backfill may have failed
                                    # but normalize can still recover from output_text
                                    _out_text = getattr(response, "output_text", None)
                                    _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                                    if _out_text_stripped:
                                        logger.debug(
                                            "Codex response.output is empty but output_text is present "
                                            "(%d chars); deferring to normalization.",
                                            len(_out_text_stripped),
                                        )
                                    else:
                                        _resp_status = getattr(response, "status", None)
                                        _resp_incomplete = getattr(response, "incomplete_details", None)
                                        logger.warning(
                                            "Codex response.output is empty after stream backfill "
                                            "(status=%s, incomplete_details=%s, model=%s). %s",
                                            _resp_status, _resp_incomplete,
                                            getattr(response, "model", None),
                                            f"api_mode={self.api_mode} provider={self.provider}",
                                        )
                                        response_invalid = True
                                        error_details.append("response.output is empty")
                    elif self.api_mode == "anthropic_messages":
                        _tv = self._get_transport()
                        if not _tv.validate_response(response):
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            else:
                                error_details.append("response.content invalid (not a non-empty list)")
                    elif self.api_mode == "bedrock_converse":
                        _btv = self._get_transport()
                        if not _btv.validate_response(response):
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            else:
                                error_details.append("Bedrock response invalid (no output or choices)")
                    else:
                        _ctv = self._get_transport()
                        if not _ctv.validate_response(response):
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            elif not hasattr(response, 'choices'):
                                error_details.append("response has no 'choices' attribute")
                            elif response.choices is None:
                                error_details.append("response.choices is None")
                            else:
                                error_details.append("response.choices is empty")

                    if response_invalid:
                        # Stop spinner before printing error messages
                        if thinking_spinner:
                            thinking_spinner.stop("(´;ω;`) oops, retrying...")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")
                        
                        # Invalid response — could be rate limiting, provider timeout,
                        # upstream server error, or malformed response.
                        retry_count += 1
                        
                        # Eager fallback: empty/malformed responses are a common
                        # rate-limit symptom.  Switch to fallback immediately
                        # rather than retrying with extended backoff.
                        if self._fallback_index < len(self._fallback_chain):
                            self._emit_status("⚠️ Empty/malformed response — switching to fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                        # Check for error field in response (some providers include this)
                        error_msg = "Unknown"
                        provider_name = "Unknown"
                        if response and hasattr(response, 'error') and response.error:
                            error_msg = str(response.error)
                            # Try to extract provider from error metadata
                            if hasattr(response.error, 'metadata') and response.error.metadata:
                                provider_name = response.error.metadata.get('provider_name', 'Unknown')
                        elif response and hasattr(response, 'message') and response.message:
                            error_msg = str(response.message)
                        
                        # Try to get provider from model field (OpenRouter often returns actual model used)
                        if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                            provider_name = f"model={response.model}"
                        
                        # Check for x-openrouter-provider or similar metadata
                        if provider_name == "Unknown" and response:
                            # Log all response attributes for debugging
                            resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                            if self.verbose_logging:
                                logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                        
                        # Extract error code from response for contextual diagnostics
                        _resp_error_code = None
                        if response and hasattr(response, 'error') and response.error:
                            _code_raw = getattr(response.error, 'code', None)
                            if _code_raw is None and isinstance(response.error, dict):
                                _code_raw = response.error.get('code')
                            if _code_raw is not None:
                                try:
                                    _resp_error_code = int(_code_raw)
                                except (TypeError, ValueError):
                                    pass

                        # Build a human-readable failure hint from the error code
                        # and response time, instead of always assuming rate limiting.
                        if _resp_error_code == 524:
                            _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                        elif _resp_error_code == 504:
                            _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                        elif _resp_error_code == 429:
                            _failure_hint = f"rate limited by upstream provider (429)"
                        elif _resp_error_code in {500, 502}:
                            _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                        elif _resp_error_code in {503, 529}:
                            _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                        elif _resp_error_code is not None:
                            _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                        elif api_duration < 10:
                            _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                        elif api_duration > 60:
                            _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                        else:
                            _failure_hint = f"response time {api_duration:.1f}s"

                        self._vprint(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                        self._vprint(f"{self.log_prefix}   🏢 Provider: {provider_name}", force=True)
                        cleaned_provider_error = self._clean_error_message(error_msg)
                        self._vprint(f"{self.log_prefix}   📝 Provider message: {cleaned_provider_error}", force=True)
                        self._vprint(f"{self.log_prefix}   ⏱️  {_failure_hint}", force=True)
                        
                        if retry_count >= max_retries:
                            # Try fallback before giving up
                            self._emit_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue
                            self._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                            logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                                "failed": True  # Mark as failure for filtering
                            }
                        
                        # Backoff before retry — jittered exponential: 5s base, 120s cap
                        wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                        self._vprint(f"{self.log_prefix}⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...", force=True)
                        logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                        
                        # Sleep in small increments to stay responsive to interrupts
                        sleep_end = time.time() + wait_time
                        _backoff_touch_counter = 0
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                                self._persist_session(messages, conversation_history)
                                self.clear_interrupt()
                                return {
                                    "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
                            # Touch activity every ~30s so the gateway's inactivity
                            # monitor knows we're alive during backoff waits.
                            _backoff_touch_counter += 1
                            if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                                self._touch_activity(
                                    f"retry backoff ({retry_count}/{max_retries}), "
                                    f"{int(sleep_end - time.time())}s remaining"
                                )
                        continue  # Retry the API call

                    # Check finish_reason before proceeding
                    if self.api_mode == "codex_responses":
                        status = getattr(response, "status", None)
                        incomplete_details = getattr(response, "incomplete_details", None)
                        incomplete_reason = None
                        if isinstance(incomplete_details, dict):
                            incomplete_reason = incomplete_details.get("reason")
                        else:
                            incomplete_reason = getattr(incomplete_details, "reason", None)
                        if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"
                    elif self.api_mode == "anthropic_messages":
                        _tfr = self._get_transport()
                        finish_reason = _tfr.map_finish_reason(response.stop_reason)
                    elif self.api_mode == "bedrock_converse":
                        # Bedrock response already normalized at dispatch — use transport
                        _bt_fr = self._get_transport()
                        _bedrock_result = _bt_fr.normalize_response(response)
                        finish_reason = _bedrock_result.finish_reason
                    else:
                        _cc_fr = self._get_transport()
                        _finish_result = _cc_fr.normalize_response(response)
                        finish_reason = _finish_result.finish_reason
                        assistant_message = _finish_result
                        if self._should_treat_stop_as_truncated(
                            finish_reason,
                            assistant_message,
                            messages,
                        ):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                                force=True,
                            )
                            finish_reason = "length"

                    if finish_reason == "length":
                        self._vprint(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

                        # Normalize the truncated response to a single OpenAI-style
                        # message shape so text-continuation and tool-call retry
                        # work uniformly across chat_completions, bedrock_converse,
                        # and anthropic_messages.  For Anthropic we use the same
                        # adapter the agent loop already relies on so the rebuilt
                        # interim assistant message is byte-identical to what
                        # would have been appended in the non-truncated path.
                        _trunc_msg = None
                        _trunc_transport = self._get_transport()
                        if self.api_mode == "anthropic_messages":
                            _trunc_result = _trunc_transport.normalize_response(
                                response, strip_tool_prefix=self._is_anthropic_oauth
                            )
                        else:
                            _trunc_result = _trunc_transport.normalize_response(response)
                        _trunc_msg = _trunc_result

                        _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                        _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

                        # ── Detect thinking-budget exhaustion ──────────────
                        # When the model spends ALL output tokens on reasoning
                        # and has none left for the response, continuation
                        # retries are pointless.  Detect this early and give a
                        # targeted error instead of wasting 3 API calls.
                        # A response is "thinking exhausted" only when the model
                        # actually produced reasoning blocks but no visible text after
                        # them.  Models that do not use <think> tags (e.g. GLM-4.7 on
                        # NVIDIA Build, minimax) may return content=None or an empty
                        # string for unrelated reasons — treat those as normal
                        # truncations that deserve continuation retries, not as
                        # thinking-budget exhaustion.
                        _has_think_tags = bool(
                            _trunc_content and re.search(
                                r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                                _trunc_content,
                                re.IGNORECASE,
                            )
                        )
                        _thinking_exhausted = (
                            not _trunc_has_tool_calls
                            and _has_think_tags
                            and (
                                (_trunc_content is not None and not self._has_content_after_think_block(_trunc_content))
                                or _trunc_content is None
                            )
                        )

                        if _thinking_exhausted:
                            _exhaust_error = (
                                "Model used all output tokens on reasoning with none left "
                                "for the response. Try lowering reasoning effort or "
                                "increasing max_tokens."
                            )
                            self._vprint(
                                f"{self.log_prefix}💭 Reasoning exhausted the output token budget — "
                                f"no visible response was produced.",
                                force=True,
                            )
                            # Return a user-friendly message as the response so
                            # CLI (response box) and gateway (chat message) both
                            # display it naturally instead of a suppressed error.
                            _exhaust_response = (
                                "⚠️ **Thinking Budget Exhausted**\n\n"
                                "The model used all its output tokens on reasoning "
                                "and had none left for the actual response.\n\n"
                                "To fix this:\n"
                                "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                                "→ Or switch to a larger/non-reasoning model with `/model`"
                            )
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": _exhaust_response,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": _exhaust_error,
                            }

                        if self.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                            assistant_message = _trunc_msg
                            if assistant_message is not None and not _trunc_has_tool_calls:
                                length_continue_retries += 1
                                interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                                messages.append(interim_msg)
                                if assistant_message.content:
                                    truncated_response_prefix += assistant_message.content

                                if length_continue_retries < 3:
                                    self._vprint(
                                        f"{self.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )
                                    continue_msg = {
                                        "role": "user",
                                        "content": (
                                            "[System: Your previous response was truncated by the output "
                                            "length limit. Continue exactly where you left off. Do not "
                                            "restart or repeat prior text. Finish the answer directly.]"
                                        ),
                                    }
                                    messages.append(continue_msg)
                                    self._session_messages = messages
                                    self._save_session_log(messages)
                                    restart_with_length_continuation = True
                                    break

                                partial_response = self._strip_think_blocks(truncated_response_prefix).strip()
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": partial_response or None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response remained truncated after 3 continuation attempts",
                                }

                        if self.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                            assistant_message = _trunc_msg
                            if assistant_message is not None and _trunc_has_tool_calls:
                                if truncated_tool_call_retries < 1:
                                    truncated_tool_call_retries += 1
                                    self._vprint(
                                        f"{self.log_prefix}⚠️  Truncated tool call detected — retrying API call...",
                                        force=True,
                                    )
                                    # Don't append the broken response to messages;
                                    # just re-run the same API call from the current
                                    # message state, giving the model another chance.
                                    continue
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                    force=True,
                                )
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response truncated due to output length limit",
                                }

                        # If we have prior messages, roll back to last complete state
                        if len(messages) > 1:
                            self._vprint(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)

                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)

                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit"
                            }
                        else:
                            # First message was truncated - mark as failed
                            self._vprint(f"{self.log_prefix}❌ First response truncated - cannot recover", force=True)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": "First response truncated due to output length limit"
                            }
                    
                    # Track actual token usage from response for context management
                    if hasattr(response, 'usage') and response.usage:
                        canonical_usage = normalize_usage(
                            response.usage,
                            provider=self.provider,
                            api_mode=self.api_mode,
                        )
                        prompt_tokens = canonical_usage.prompt_tokens
                        completion_tokens = canonical_usage.output_tokens
                        total_tokens = canonical_usage.total_tokens
                        usage_dict = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                        self.context_compressor.update_from_response(usage_dict)

                        # Cache discovered context length after successful call.
                        # Only persist limits confirmed by the provider (parsed
                        # from the error message), not guessed probe tiers.
                        if getattr(self.context_compressor, "_context_probed", False):
                            ctx = self.context_compressor.context_length
                            if getattr(self.context_compressor, "_context_probe_persistable", False):
                                save_context_length(self.model, self.base_url, ctx)
                                self._safe_print(f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}")
                            self.context_compressor._context_probed = False
                            self.context_compressor._context_probe_persistable = False

                        self.session_prompt_tokens += prompt_tokens
                        self.session_completion_tokens += completion_tokens
                        self.session_total_tokens += total_tokens
                        self.session_api_calls += 1
                        self.session_input_tokens += canonical_usage.input_tokens
                        self.session_output_tokens += canonical_usage.output_tokens
                        self.session_cache_read_tokens += canonical_usage.cache_read_tokens
                        self.session_cache_write_tokens += canonical_usage.cache_write_tokens
                        self.session_reasoning_tokens += canonical_usage.reasoning_tokens

                        # Log API call details for debugging/observability
                        _cache_pct = ""
                        if canonical_usage.cache_read_tokens and prompt_tokens:
                            _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                        logger.info(
                            "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                            self.session_api_calls, self.model, self.provider or "unknown",
                            prompt_tokens, completion_tokens, total_tokens,
                            api_duration, _cache_pct,
                        )

                        cost_result = estimate_usage_cost(
                            self.model,
                            canonical_usage,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                        )
                        if cost_result.amount_usd is not None:
                            self.session_estimated_cost_usd += float(cost_result.amount_usd)
                        self.session_cost_status = cost_result.status
                        self.session_cost_source = cost_result.source

                        # Persist token counts to session DB for /insights.
                        # Do this for every platform with a session_id so non-CLI
                        # sessions (gateway, cron, delegated runs) cannot lose
                        # token/accounting data if a higher-level persistence path
                        # is skipped or fails. Gateway/session-store writes use
                        # absolute totals, so they safely overwrite these per-call
                        # deltas instead of double-counting them.
                        if self._session_db and self.session_id:
                            try:
                                # Ensure the session row exists before attempting UPDATE.
                                # Under concurrent load (cron/kanban), the initial
                                # _ensure_db_session() may have failed due to SQLite
                                # locking.  Retry here so per-call token deltas are
                                # not silently lost (UPDATE on a non-existent row
                                # affects 0 rows without error).
                                if not self._session_db_created:
                                    self._ensure_db_session()
                                self._session_db.update_token_counts(
                                    self.session_id,
                                    input_tokens=canonical_usage.input_tokens,
                                    output_tokens=canonical_usage.output_tokens,
                                    cache_read_tokens=canonical_usage.cache_read_tokens,
                                    cache_write_tokens=canonical_usage.cache_write_tokens,
                                    reasoning_tokens=canonical_usage.reasoning_tokens,
                                    estimated_cost_usd=float(cost_result.amount_usd)
                                    if cost_result.amount_usd is not None else None,
                                    cost_status=cost_result.status,
                                    cost_source=cost_result.source,
                                    billing_provider=self.provider,
                                    billing_base_url=self.base_url,
                                    billing_mode="subscription_included"
                                    if cost_result.status == "included" else None,
                                    model=self.model,
                                    api_call_count=1,
                                )
                            except Exception as e:
                                # Log token persistence failures so they're
                                # visible in agent.log — silent loss here is
                                # the root cause of undercounted analytics.
                                logger.debug(
                                    "Token persistence failed (session=%s, tokens=%d): %s",
                                    self.session_id, total_tokens, e,
                                )
                        
                        if self.verbose_logging:
                            logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                        
                        # Surface cache hit stats for any provider that reports
                        # them — not just those where we inject cache_control
                        # markers.  OpenAI/Kimi/DeepSeek/Qwen all do automatic
                        # server-side prefix caching and return
                        # ``prompt_tokens_details.cached_tokens``; users
                        # previously could not see their cache % because this
                        # line was gated on ``_use_prompt_caching``, which is
                        # only True for Anthropic-style marker injection.
                        # ``canonical_usage`` is already normalised from all
                        # three API shapes (Anthropic / Codex / OpenAI-chat)
                        # so we can rely on its values directly.
                        cached = canonical_usage.cache_read_tokens
                        written = canonical_usage.cache_write_tokens
                        prompt = usage_dict["prompt_tokens"]
                        if (cached or written) and not self.quiet_mode:
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            self._vprint(
                                f"{self.log_prefix}   💾 Cache: "
                                f"{cached:,}/{prompt:,} tokens "
                                f"({hit_pct:.0f}% hit, {written:,} written)"
                            )
                    
                    has_retried_429 = False  # Reset on success
                    # Clear Nous rate limit state on successful request —
                    # proves the limit has reset and other sessions can
                    # resume hitting Nous.
                    if self.provider == "nous":
                        try:
                            from agent.nous_rate_guard import clear_nous_rate_limit
                            clear_nous_rate_limit()
                        except Exception:
                            pass
                    self._touch_activity(f"API call #{api_call_count} completed")
                    break  # Success, exit retry loop

                except InterruptedError:
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    api_elapsed = time.time() - api_start_time
                    self._vprint(f"{self.log_prefix}⚡ Interrupted during API call.", force=True)
                    self._persist_session(messages, conversation_history)
                    interrupted = True
                    final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                    break

                except Exception as api_error:
                    # Stop spinner before printing error messages
                    if thinking_spinner:
                        thinking_spinner.stop("(╥_╥) error, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    # -----------------------------------------------------------
                    # UnicodeEncodeError recovery.  Two common causes:
                    #   1. Lone surrogates (U+D800..U+DFFF) from clipboard paste
                    #      (Google Docs, rich-text editors) — sanitize and retry.
                    #   2. ASCII codec on systems with LANG=C or non-UTF-8 locale
                    #      (e.g. Chromebooks) — any non-ASCII character fails.
                    #      Detect via the error message mentioning 'ascii' codec.
                    # We sanitize messages in-place and may retry twice:
                    # first to strip surrogates, then once more for pure
                    # ASCII-only locale sanitization if needed.
                    # -----------------------------------------------------------
                    if isinstance(api_error, UnicodeEncodeError) and getattr(self, '_unicode_sanitization_passes', 0) < 2:
                        _err_str = str(api_error).lower()
                        _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                        # Detect surrogate errors — utf-8 codec refusing to
                        # encode U+D800..U+DFFF.  The error text is:
                        #   "'utf-8' codec can't encode characters in position
                        #    N-M: surrogates not allowed"
                        _is_surrogate_error = (
                            "surrogate" in _err_str
                            or ("'utf-8'" in _err_str and not _is_ascii_codec)
                        )
                        # Sanitize surrogates from both the canonical `messages`
                        # list AND `api_messages` (the API-copy, which may carry
                        # `reasoning_content`/`reasoning_details` transformed
                        # from `reasoning` — fields the canonical list doesn't
                        # have directly).  Also clean `api_kwargs` if built and
                        # `prefill_messages` if present.  Mirrors the ASCII
                        # codec recovery below.
                        _surrogates_found = _sanitize_messages_surrogates(messages)
                        if isinstance(api_messages, list):
                            if _sanitize_messages_surrogates(api_messages):
                                _surrogates_found = True
                        if isinstance(api_kwargs, dict):
                            if _sanitize_structure_surrogates(api_kwargs):
                                _surrogates_found = True
                        if isinstance(getattr(self, "prefill_messages", None), list):
                            if _sanitize_messages_surrogates(self.prefill_messages):
                                _surrogates_found = True
                        # Gate the retry on the error type, not on whether we
                        # found anything — _force_ascii_payload / the extended
                        # surrogate walker above cover all known paths, but a
                        # new transformed field could still slip through.  If
                        # the error was a surrogate encode failure, always let
                        # the retry run; the proactive sanitizer at line ~8781
                        # runs again on the next iteration.  Bounded by
                        # _unicode_sanitization_passes < 2 (outer guard).
                        if _surrogates_found or _is_surrogate_error:
                            self._unicode_sanitization_passes += 1
                            if _surrogates_found:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Stripped invalid surrogate characters from messages. Retrying...",
                                    force=True,
                                )
                            else:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Surrogate encoding error — retrying after full-payload sanitization...",
                                    force=True,
                                )
                            continue
                        if _is_ascii_codec:
                            self._force_ascii_payload = True
                            # ASCII codec: the system encoding can't handle
                            # non-ASCII characters at all. Sanitize all
                            # non-ASCII content from messages/tool schemas and retry.
                            # Sanitize both the canonical `messages` list and
                            # `api_messages` (the API-copy built before the retry
                            # loop, which may contain extra fields like
                            # reasoning_content that are not in `messages`).
                            _messages_sanitized = _sanitize_messages_non_ascii(messages)
                            if isinstance(api_messages, list):
                                _sanitize_messages_non_ascii(api_messages)
                            # Also sanitize the last api_kwargs if already built,
                            # so a leftover non-ASCII value in a transformed field
                            # (e.g. extra_body, reasoning_content) doesn't survive
                            # into the next attempt via _build_api_kwargs cache paths.
                            if isinstance(api_kwargs, dict):
                                _sanitize_structure_non_ascii(api_kwargs)
                            _prefill_sanitized = False
                            if isinstance(getattr(self, "prefill_messages", None), list):
                                _prefill_sanitized = _sanitize_messages_non_ascii(self.prefill_messages)

                            _tools_sanitized = False
                            if isinstance(getattr(self, "tools", None), list):
                                _tools_sanitized = _sanitize_tools_non_ascii(self.tools)

                            _system_sanitized = False
                            if isinstance(active_system_prompt, str):
                                _sanitized_system = _strip_non_ascii(active_system_prompt)
                                if _sanitized_system != active_system_prompt:
                                    active_system_prompt = _sanitized_system
                                    self._cached_system_prompt = _sanitized_system
                                    _system_sanitized = True
                            if isinstance(getattr(self, "ephemeral_system_prompt", None), str):
                                _sanitized_ephemeral = _strip_non_ascii(self.ephemeral_system_prompt)
                                if _sanitized_ephemeral != self.ephemeral_system_prompt:
                                    self.ephemeral_system_prompt = _sanitized_ephemeral
                                    _system_sanitized = True

                            _headers_sanitized = False
                            _default_headers = (
                                self._client_kwargs.get("default_headers")
                                if isinstance(getattr(self, "_client_kwargs", None), dict)
                                else None
                            )
                            if isinstance(_default_headers, dict):
                                _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                            # Sanitize the API key — non-ASCII characters in
                            # credentials (e.g. ʋ instead of v from a bad
                            # copy-paste) cause httpx to fail when encoding
                            # the Authorization header as ASCII.  This is the
                            # most common cause of persistent UnicodeEncodeError
                            # that survives message/tool sanitization (#6843).
                            _credential_sanitized = False
                            _raw_key = getattr(self, "api_key", None) or ""
                            if _raw_key:
                                _clean_key = _strip_non_ascii(_raw_key)
                                if _clean_key != _raw_key:
                                    self.api_key = _clean_key
                                    if isinstance(getattr(self, "_client_kwargs", None), dict):
                                        self._client_kwargs["api_key"] = _clean_key
                                    # Also update the live client — it holds its
                                    # own copy of api_key which auth_headers reads
                                    # dynamically on every request.
                                    if getattr(self, "client", None) is not None and hasattr(self.client, "api_key"):
                                        self.client.api_key = _clean_key
                                    _credential_sanitized = True
                                    self._vprint(
                                        f"{self.log_prefix}⚠️  API key contained non-ASCII characters "
                                        f"(bad copy-paste?) — stripped them. If auth fails, "
                                        f"re-copy the key from your provider's dashboard.",
                                        force=True,
                                    )

                            # Always retry on ASCII codec detection —
                            # _force_ascii_payload guarantees the full
                            # api_kwargs payload is sanitized on the
                            # next iteration (line ~8475).  Even when
                            # per-component checks above find nothing
                            # (e.g. non-ASCII only in api_messages'
                            # reasoning_content), the flag catches it.
                            # Bounded by _unicode_sanitization_passes < 2.
                            self._unicode_sanitization_passes += 1
                            _any_sanitized = (
                                _messages_sanitized
                                or _prefill_sanitized
                                or _tools_sanitized
                                or _system_sanitized
                                or _headers_sanitized
                                or _credential_sanitized
                            )
                            if _any_sanitized:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                                    force=True,
                                )
                            else:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                                    force=True,
                                )
                            continue

                    # ── Image-rejection recovery ──────────────────────────────
                    # Some providers (mlx-lm, text-only endpoints, text-only
                    # fallbacks on multimodal models) reject any message that
                    # contains image_url content with a 4xx error like
                    # "Only 'text' content type is supported."  On first hit,
                    # strip all images from the message list, mark the session
                    # as vision-unsupported, and retry with text only.
                    #
                    # Detection is best-effort English phrase matching — a
                    # locale-translated or heavily-reworded upstream error
                    # will bypass this guard and fall through to the normal
                    # error handler.  Expand the phrase list when new
                    # provider wordings are observed in the wild.
                    _err_body = ""
                    try:
                        _err_body = str(getattr(api_error, "body", None) or
                                        getattr(api_error, "message", None) or
                                        str(api_error))
                    except Exception:
                        pass
                    _err_status = getattr(api_error, "status_code", None)
                    _IMAGE_REJECTION_PHRASES = (
                        "only 'text' content type is supported",
                        "only text content type is supported",
                        "image_url is not supported",
                        "image content is not supported",
                        "multimodal is not supported",
                        "multimodal content is not supported",
                        "multimodal input is not supported",
                        "vision is not supported",
                        "vision input is not supported",
                        "does not support images",
                        "does not support image input",
                        "does not support multimodal",
                        "does not support vision",
                        "model does not support image",
                        # ChatGPT-account Codex backend
                        # (https://chatgpt.com/backend-api/codex) rejects
                        # data:image/...base64 URLs in input_image fields
                        # with HTTP 400 "Invalid 'input[N].content[K].image_url'.
                        # Expected a valid URL, but got a value with an
                        # invalid format." The OpenAI Responses API on the
                        # public endpoint accepts data URLs, but the
                        # ChatGPT-account variant does not. Without this
                        # phrase the agent cascaded into compression /
                        # context-too-large recovery instead of just
                        # stripping the images. Match is narrow on
                        # purpose — keyed on the field-path apostrophe so
                        # we don't false-trip on other URL validation
                        # errors. (issue #23570)
                        "image_url'. expected",
                    )
                    _err_lower = _err_body.lower()
                    _looks_like_image_rejection = any(
                        p in _err_lower for p in _IMAGE_REJECTION_PHRASES
                    )
                    # 4xx-only gate: never interpret 5xx/timeout as "server
                    # said no to images" — those are transient and must
                    # route to the normal retry path.
                    _status_ok = _err_status is None or (400 <= int(_err_status) < 500)
                    if (
                        getattr(self, "_vision_supported", True)
                        and _looks_like_image_rejection
                        and _status_ok
                    ):
                        self._vision_supported = False
                        _imgs_removed = _strip_images_from_messages(messages)
                        if isinstance(api_messages, list):
                            _strip_images_from_messages(api_messages)
                        self._vprint(
                            f"{self.log_prefix}⚠️  Server rejected image content — "
                            f"switching to text-only mode for this session"
                            + (". Stripped images from history and retrying." if _imgs_removed else "."),
                            force=True,
                        )
                        continue

                    status_code = getattr(api_error, "status_code", None)
                    error_context = self._extract_api_error_context(api_error)

                    # ── Classify the error for structured recovery decisions ──
                    _compressor = getattr(self, "context_compressor", None)
                    _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                    classified = classify_api_error(
                        api_error,
                        provider=getattr(self, "provider", "") or "",
                        model=getattr(self, "model", "") or "",
                        approx_tokens=approx_tokens,
                        context_length=_ctx_len,
                        num_messages=len(api_messages) if api_messages else 0,
                    )
                    logger.debug(
                        "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                        classified.reason.value, classified.status_code,
                        classified.retryable, classified.should_compress,
                        classified.should_rotate_credential, classified.should_fallback,
                    )

                    recovered_with_pool, has_retried_429 = self._recover_with_credential_pool(
                        status_code=status_code,
                        has_retried_429=has_retried_429,
                        classified_reason=classified.reason,
                        error_context=error_context,
                    )
                    if recovered_with_pool:
                        continue

                    # Image-too-large recovery: shrink oversized native image
                    # parts in-place and retry once.  Triggered by Anthropic's
                    # per-image 5 MB ceiling (400 with "image exceeds 5 MB
                    # maximum") or any other provider that complains about
                    # image size.  If shrink fails or a second attempt still
                    # fails, fall through to normal error handling.
                    if (
                        classified.reason == FailoverReason.image_too_large
                        and not image_shrink_retry_attempted
                    ):
                        image_shrink_retry_attempted = True
                        if self._try_shrink_image_parts_in_messages(api_messages):
                            self._vprint(
                                f"{self.log_prefix}📐 Image(s) exceeded provider size limit — "
                                f"shrank and retrying...",
                                force=True,
                            )
                            continue
                        else:
                            logger.info(
                                "image-shrink recovery: no data-URL image parts found "
                                "or shrink didn't reduce size; surfacing original error."
                            )

                    # Anthropic OAuth subscription rejected the 1M-context beta
                    # header ("long context beta is not yet available for this
                    # subscription"). Disable the beta for the rest of this
                    # session, rebuild the client, and retry once.  1M-capable
                    # subscriptions never hit this branch — they accept the
                    # beta and keep full 1M context.  See PR #17680 for the
                    # original report (we chose reactive recovery over the
                    # proposed unconditional omit so capable subscriptions
                    # don't silently lose the capability).
                    if (
                        classified.reason == FailoverReason.oauth_long_context_beta_forbidden
                        and self.api_mode == "anthropic_messages"
                        and self._is_anthropic_oauth
                        and not oauth_1m_beta_retry_attempted
                    ):
                        oauth_1m_beta_retry_attempted = True
                        if not getattr(self, "_oauth_1m_beta_disabled", False):
                            self._oauth_1m_beta_disabled = True
                            try:
                                self._anthropic_client.close()
                            except Exception:
                                pass
                            self._rebuild_anthropic_client()
                            self._vprint(
                                f"{self.log_prefix}🔕 OAuth subscription doesn't support "
                                f"the 1M-context beta — disabled for this session and retrying...",
                                force=True,
                            )
                            continue

                    if (
                        self.api_mode == "codex_responses"
                        and self.provider == "openai-codex"
                        and status_code == 401
                        and not codex_auth_retry_attempted
                    ):
                        codex_auth_retry_attempted = True
                        if self._try_refresh_codex_client_credentials(force=True):
                            self._vprint(f"{self.log_prefix}🔐 Codex auth refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "chat_completions"
                        and self.provider == "nous"
                        and status_code == 401
                        and not nous_auth_retry_attempted
                    ):
                        nous_auth_retry_attempted = True
                        if self._try_refresh_nous_client_credentials(force=True):
                            print(f"{self.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                            continue
                        # Credential refresh didn't help — show diagnostic info.
                        # Most common causes: Portal OAuth expired/revoked,
                        # account out of credits, or agent key blocked.
                        from hermes_constants import display_hermes_home as _dhh_fn
                        _dhh = _dhh_fn()
                        _body_text = ""
                        try:
                            _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
                            if _body is not None:
                                _body_text = str(_body)[:200]
                        except Exception:
                            pass
                        print(f"{self.log_prefix}🔐 Nous 401 — Portal authentication failed.")
                        if _body_text:
                            print(f"{self.log_prefix}   Response: {_body_text}")
                        print(f"{self.log_prefix}   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
                        print(f"{self.log_prefix}   Troubleshooting:")
                        print(f"{self.log_prefix}     • Re-authenticate: hermes login --provider nous")
                        print(f"{self.log_prefix}     • Check credits / billing: https://portal.nousresearch.com")
                        print(f"{self.log_prefix}     • Verify stored credentials: {_dhh}/auth.json")
                        print(f"{self.log_prefix}     • Switch providers temporarily: /model <model> --provider openrouter")
                    if (
                        self.provider == "copilot"
                        and status_code == 401
                        and not copilot_auth_retry_attempted
                    ):
                        copilot_auth_retry_attempted = True
                        if self._try_refresh_copilot_client_credentials():
                            self._vprint(f"{self.log_prefix}🔐 Copilot credentials refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "anthropic_messages"
                        and status_code == 401
                        and hasattr(self, '_anthropic_api_key')
                        and not anthropic_auth_retry_attempted
                    ):
                        anthropic_auth_retry_attempted = True
                        from agent.anthropic_adapter import _is_oauth_token
                        if self._try_refresh_anthropic_client_credentials():
                            print(f"{self.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                            continue
                        # Credential refresh didn't help — show diagnostic info
                        key = self._anthropic_api_key
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                        print(f"{self.log_prefix}🔐 Anthropic 401 — authentication failed.")
                        print(f"{self.log_prefix}   Auth method: {auth_method}")
                        print(f"{self.log_prefix}   Token prefix: {key[:12]}..." if key and len(key) > 12 else f"{self.log_prefix}   Token: (empty or short)")
                        print(f"{self.log_prefix}   Troubleshooting:")
                        from hermes_constants import display_hermes_home as _dhh_fn
                        _dhh = _dhh_fn()
                        print(f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                        print(f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                        print(f"{self.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                        print(f"{self.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                        print(f"{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                        print(f"{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                    # ── Thinking block signature recovery ─────────────────
                    # Anthropic signs thinking blocks against the full turn
                    # content.  Any upstream mutation (context compression,
                    # session truncation, message merging) invalidates the
                    # signature → HTTP 400.  Recovery: strip reasoning_details
                    # from all messages so the next retry sends no thinking
                    # blocks at all.  One-shot — don't retry infinitely.
                    if (
                        classified.reason == FailoverReason.thinking_signature
                        and not thinking_sig_retry_attempted
                    ):
                        thinking_sig_retry_attempted = True
                        for _m in messages:
                            if isinstance(_m, dict):
                                _m.pop("reasoning_details", None)
                        self._vprint(
                            f"{self.log_prefix}⚠️  Thinking block signature invalid — "
                            f"stripped all thinking blocks, retrying...",
                            force=True,
                        )
                        logging.warning(
                            "%sThinking block signature recovery: stripped "
                            "reasoning_details from %d messages",
                            self.log_prefix, len(messages),
                        )
                        continue

                    # ── llama.cpp grammar-parse recovery ──────────────────
                    # llama.cpp's ``json-schema-to-grammar`` converter rejects
                    # regex escape classes (``\d``, ``\w``, ``\s``) and most
                    # ``format`` values in tool schemas.  MCP servers emit
                    # these routinely for date/phone/email params.  Recovery:
                    # strip ``pattern``/``format`` from ``self.tools`` and
                    # retry once.  We keep the keywords by default so cloud
                    # providers get the full prompting hints; this branch
                    # fires only for users on llama.cpp's OAI server.
                    if (
                        classified.reason == FailoverReason.llama_cpp_grammar_pattern
                        and not llama_cpp_grammar_retry_attempted
                    ):
                        llama_cpp_grammar_retry_attempted = True
                        try:
                            from tools.schema_sanitizer import strip_pattern_and_format
                            _, _stripped = strip_pattern_and_format(self.tools)
                        except Exception as _strip_exc:  # pragma: no cover — defensive
                            logging.warning(
                                "%sllama.cpp grammar recovery: strip helper failed: %s",
                                self.log_prefix, _strip_exc,
                            )
                            _stripped = 0
                        if _stripped:
                            self._vprint(
                                f"{self.log_prefix}⚠️  llama.cpp rejected tool schema grammar — "
                                f"stripped {_stripped} pattern/format keyword(s), retrying...",
                                force=True,
                            )
                            logging.warning(
                                "%sllama.cpp grammar recovery: stripped %d "
                                "pattern/format keyword(s) from tool schemas",
                                self.log_prefix, _stripped,
                            )
                            continue
                        # No keywords found to strip — fall through to normal
                        # retry path rather than loop forever on the same error.
                        logging.warning(
                            "%sllama.cpp grammar error but no pattern/format "
                            "keywords to strip — falling through to normal retry",
                            self.log_prefix,
                        )

                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    self._touch_activity(
                        f"API error recovery (attempt {retry_count}/{max_retries})"
                    )
                    
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    _error_summary = self._summarize_api_error(api_error)
                    logger.warning(
                        "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                        retry_count,
                        max_retries,
                        error_type,
                        self._client_log_context(),
                        _error_summary,
                    )

                    _provider = getattr(self, "provider", "unknown")
                    _base = getattr(self, "base_url", "unknown")
                    _model = getattr(self, "model", "unknown")
                    _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                    self._vprint(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}", force=True)
                    self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    self._vprint(f"{self.log_prefix}   📝 Error: {_error_summary}", force=True)
                    if status_code and status_code < 500:
                        _err_body = getattr(api_error, "body", None)
                        _err_body_str = str(_err_body)[:300] if _err_body else None
                        if _err_body_str:
                            self._vprint(f"{self.log_prefix}   📋 Details: {_err_body_str}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                    # Actionable hint for OpenRouter "no tool endpoints" error.
                    # This fires regardless of whether fallback succeeds — the
                    # user needs to know WHY their model failed so they can fix
                    # their provider routing, not just silently fall back.
                    if (
                        self._is_openrouter_url()
                        and "support tool use" in error_msg
                    ):
                        self._vprint(
                            f"{self.log_prefix}   💡 No OpenRouter providers for {_model} support tool calling with your current settings.",
                            force=True,
                        )
                        if self.providers_allowed:
                            self._vprint(
                                f"{self.log_prefix}      Your provider_routing.only restriction is filtering out tool-capable providers.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      Try removing the restriction or adding providers that support tools for this model.",
                                force=True,
                            )
                        self._vprint(
                            f"{self.log_prefix}      Check which providers support tools: https://openrouter.ai/models/{_model}",
                            force=True,
                        )

                    # Check for interrupt before deciding to retry
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: handling API error ({error_type}: {self._clean_error_message(str(api_error))}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    
                    # Check for 413 payload-too-large BEFORE generic 4xx handler.
                    # A 413 is a payload-size error — the correct response is to
                    # compress history and retry, not abort immediately.
                    status_code = getattr(api_error, "status_code", None)

                    # ── Anthropic Sonnet long-context tier gate ───────────
                    # Anthropic returns HTTP 429 "Extra usage is required for
                    # long context requests" when a Claude Max (or similar)
                    # subscription doesn't include the 1M-context tier.  This
                    # is NOT a transient rate limit — retrying or switching
                    # credentials won't help.  Reduce context to 200k (the
                    # standard tier) and compress.
                    if classified.reason == FailoverReason.long_context_tier:
                        _reduced_ctx = 200000
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length
                        if old_ctx > _reduced_ctx:
                            compressor.update_model(
                                model=self.model,
                                context_length=_reduced_ctx,
                                base_url=self.base_url,
                                api_key=getattr(self, "api_key", ""),
                                provider=self.provider,
                            )
                            # Context probing flags — only set on built-in
                            # compressor (plugin engines manage their own).
                            if hasattr(compressor, "_context_probed"):
                                compressor._context_probed = True
                                # Don't persist — this is a subscription-tier
                                # limitation, not a model capability.  If the
                                # user later enables extra usage the 1M limit
                                # should come back automatically.
                                compressor._context_probe_persistable = False
                            self._vprint(
                                f"{self.log_prefix}⚠️  Anthropic long-context tier "
                                f"requires extra usage — reducing context: "
                                f"{old_ctx:,} → {_reduced_ctx:,} tokens",
                                force=True,
                            )

                        compression_attempts += 1
                        if compression_attempts <= max_compression_attempts:
                            original_len = len(messages)
                            messages, active_system_prompt = self._compress_context(
                                messages, system_message,
                                approx_tokens=approx_tokens,
                                task_id=effective_task_id,
                            )
                            # Compression created a new session — clear history
                            # so _flush_messages_to_session_db writes compressed
                            # messages to the new session, not skipping them.
                            conversation_history = None
                            if len(messages) < original_len or old_ctx > _reduced_ctx:
                                self._emit_status(
                                    f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                    f"(was {old_ctx:,}), retrying..."
                                )
                                time.sleep(2)
                                restart_with_compressed_messages = True
                                break
                        # Fall through to normal error handling if compression
                        # is exhausted or didn't help.

                    # Eager fallback for rate-limit errors (429 or quota exhaustion).
                    # When a fallback model is configured, switch immediately instead
                    # of burning through retries with exponential backoff -- the
                    # primary provider won't recover within the retry window.
                    is_rate_limited = classified.reason in {
                        FailoverReason.rate_limit,
                        FailoverReason.billing,
                    }
                    if is_rate_limited and self._fallback_index < len(self._fallback_chain):
                        # Don't eagerly fallback if credential pool rotation may
                        # still recover.  See _pool_may_recover_from_rate_limit
                        # for the single-credential-pool and CloudCode-quota
                        # exceptions.  Fixes #11314 and #13636.
                        pool_may_recover = _pool_may_recover_from_rate_limit(
                            self._credential_pool,
                            provider=self.provider,
                            base_url=getattr(self, "base_url", None),
                        )
                        if not pool_may_recover:
                            self._emit_status("⚠️ Rate limited — switching to fallback provider...")
                            if self._try_activate_fallback(reason=classified.reason):
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue

                    # ── Nous Portal: record rate limit & skip retries ─────
                    # When Nous returns a 429 that is a genuine account-
                    # level rate limit, record the reset time to a shared
                    # file so ALL sessions (cron, gateway, auxiliary) know
                    # not to pile on, then skip further retries -- each
                    # one burns another RPH request and deepens the hole.
                    # The retry loop's top-of-iteration guard will catch
                    # this on the next pass and try fallback or bail.
                    #
                    # IMPORTANT: Nous Portal multiplexes multiple upstream
                    # providers (DeepSeek, Kimi, MiMo, Hermes).  A 429 can
                    # also mean an UPSTREAM provider is out of capacity
                    # for one specific model -- transient, clears in
                    # seconds, nothing to do with the caller's quota.
                    # Tripping the cross-session breaker on that would
                    # block every Nous model for minutes.  We use
                    # ``is_genuine_nous_rate_limit`` to tell the two
                    # apart via the 429's own x-ratelimit-* headers and
                    # the last-known-good state captured on the previous
                    # successful response.
                    if (
                        is_rate_limited
                        and self.provider == "nous"
                        and classified.reason == FailoverReason.rate_limit
                        and not recovered_with_pool
                    ):
                        _genuine_nous_rate_limit = False
                        try:
                            from agent.nous_rate_guard import (
                                is_genuine_nous_rate_limit,
                                record_nous_rate_limit,
                            )
                            _err_resp = getattr(api_error, "response", None)
                            _err_hdrs = (
                                getattr(_err_resp, "headers", None)
                                if _err_resp else None
                            )
                            _genuine_nous_rate_limit = is_genuine_nous_rate_limit(
                                headers=_err_hdrs,
                                last_known_state=self._rate_limit_state,
                            )
                            if _genuine_nous_rate_limit:
                                record_nous_rate_limit(
                                    headers=_err_hdrs,
                                    error_context=error_context,
                                )
                            else:
                                logging.info(
                                    "Nous 429 looks like upstream capacity "
                                    "(no exhausted bucket in headers or "
                                    "last-known state) -- not tripping "
                                    "cross-session breaker."
                                )
                        except Exception:
                            pass
                        if _genuine_nous_rate_limit:
                            # Skip straight to max_retries -- the
                            # top-of-loop guard will handle fallback or
                            # bail cleanly.
                            retry_count = max_retries
                            continue
                        # Upstream capacity 429: fall through to normal
                        # retry logic.  A different model (or the same
                        # model a moment later) will typically succeed.

                    is_payload_too_large = (
                        classified.reason == FailoverReason.payload_too_large
                    )

                    if is_payload_too_large:
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        self._emit_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        conversation_history = None

                        if len(messages) < original_len:
                            self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            self._vprint(f"{self.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 payload too large. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Request payload too large (413). Cannot compress further.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }

                    # Check for context-length errors BEFORE generic 4xx handler.
                    # The classifier detects context overflow from: explicit error
                    # messages, generic 400 + large session heuristic (#1630), and
                    # server disconnect + large session pattern (#2153).
                    is_context_length_error = (
                        classified.reason == FailoverReason.context_overflow
                    )

                    if is_context_length_error:
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length

                        # ── Distinguish two very different errors ───────────
                        # 1. "Prompt too long": the INPUT exceeds the context window.
                        #    Fix: reduce context_length + compress history.
                        # 2. "max_tokens too large": input is fine, but
                        #    input_tokens + requested max_tokens > context_window.
                        #    Fix: reduce max_tokens (the OUTPUT cap) for this call.
                        #    Do NOT shrink context_length — the window is unchanged.
                        #
                        # Note: max_tokens = output token cap (one response).
                        #       context_length = total window (input + output combined).
                        available_out = parse_available_output_tokens_from_error(error_msg)
                        if available_out is not None:
                            # Error is purely about the output cap being too large.
                            # Cap output to the available space and retry without
                            # touching context_length or triggering compression.
                            safe_out = max(1, available_out - 64)  # small safety margin
                            self._ephemeral_max_output_tokens = safe_out
                            self._vprint(
                                f"{self.log_prefix}⚠️  Output cap too large for current prompt — "
                                f"retrying with max_tokens={safe_out:,} "
                                f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})",
                                force=True,
                            )
                            # Still count against compression_attempts so we don't
                            # loop forever if the error keeps recurring.
                            compression_attempts += 1
                            if compression_attempts > max_compression_attempts:
                                self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                                self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                                logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                                self._persist_session(messages, conversation_history)
                                return {
                                    "messages": messages,
                                    "completed": False,
                                    "api_calls": api_call_count,
                                    "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                    "partial": True,
                                    "failed": True,
                                    "compression_exhausted": True,
                                }
                            restart_with_compressed_messages = True
                            break

                        # Error is about the INPUT being too large — reduce context_length.
                        # Try to parse the actual limit from the error message
                        parsed_limit = parse_context_limit_from_error(error_msg)
                        _provider_lower = (getattr(self, "provider", "") or "").lower()
                        _base_lower = (getattr(self, "base_url", "") or "").rstrip("/").lower()
                        is_minimax_provider = (
                            _provider_lower in {"minimax", "minimax-cn"}
                            or _base_lower.startswith((
                                "https://api.minimax.io/anthropic",
                                "https://api.minimaxi.com/anthropic",
                            ))
                        )
                        minimax_delta_only_overflow = (
                            is_minimax_provider
                            and parsed_limit is None
                            and "context window exceeds limit (" in error_msg
                        )
                        if parsed_limit and parsed_limit < old_ctx:
                            new_ctx = parsed_limit
                            self._vprint(f"{self.log_prefix}Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
                        elif minimax_delta_only_overflow:
                            new_ctx = old_ctx
                            self._vprint(
                                f"{self.log_prefix}Provider reported overflow amount only; "
                                f"keeping context_length at {old_ctx:,} tokens and compressing.",
                                force=True,
                            )
                        else:
                            # Step down to the next probe tier
                            new_ctx = get_next_probe_tier(old_ctx)

                        if new_ctx and new_ctx < old_ctx:
                            compressor.update_model(
                                model=self.model,
                                context_length=new_ctx,
                                base_url=self.base_url,
                                api_key=getattr(self, "api_key", ""),
                                provider=self.provider,
                            )
                            # Context probing flags — only set on built-in
                            # compressor (plugin engines manage their own).
                            if hasattr(compressor, "_context_probed"):
                                compressor._context_probed = True
                                # Only persist limits parsed from the provider's
                                # error message (a real number).  Guessed fallback
                                # tiers from get_next_probe_tier() should stay
                                # in-memory only — persisting them pollutes the
                                # cache with wrong values.
                                compressor._context_probe_persistable = bool(
                                    parsed_limit and parsed_limit == new_ctx
                                )
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        self._emit_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        conversation_history = None

                        if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                            if len(messages) < original_len:
                                self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            # Can't compress further and already at minimum tier
                            self._vprint(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                            logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }

                    # Check for non-retryable client errors.  The classifier
                    # already accounts for 413, 429, 529 (transient), context
                    # overflow, and generic-400 heuristics.  Local validation
                    # errors (ValueError, TypeError) are programming bugs.
                    # Exclude UnicodeEncodeError — it's a ValueError subclass
                    # but is handled separately by the surrogate sanitization
                    # path above.  Exclude json.JSONDecodeError — also a
                    # ValueError subclass, but it indicates a transient
                    # provider/network failure (malformed response body,
                    # truncated stream, routing layer corruption), not a
                    # local programming bug, and should be retried (#14782).
                    is_local_validation_error = (
                        isinstance(api_error, (ValueError, TypeError))
                        and not isinstance(
                            api_error, (UnicodeEncodeError, json.JSONDecodeError)
                        )
                        # ssl.SSLError (and its subclass SSLCertVerificationError)
                        # inherits from OSError *and* ValueError via Python MRO,
                        # so the isinstance(ValueError) check above would
                        # misclassify a TLS transport failure as a local
                        # programming bug and abort without retrying.  Exclude
                        # ssl.SSLError explicitly so the error classifier's
                        # retryable=True mapping takes effect instead.
                        and not isinstance(api_error, ssl.SSLError)
                    )
                    is_client_error = (
                        is_local_validation_error
                        or (
                            not classified.retryable
                            and not classified.should_compress
                            and classified.reason not in {
                                FailoverReason.rate_limit,
                                FailoverReason.billing,
                                FailoverReason.overloaded,
                                FailoverReason.context_overflow,
                                FailoverReason.payload_too_large,
                                FailoverReason.long_context_tier,
                                FailoverReason.thinking_signature,
                            }
                        )
                    ) and not is_context_length_error

                    if is_client_error:
                        # Try fallback before aborting — a different provider
                        # may not have the same issue (rate limit, auth, etc.)
                        self._emit_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        if api_kwargs is not None:
                            self._dump_api_request_debug(
                                api_kwargs, reason="non_retryable_client_error", error=api_error,
                            )
                        self._emit_status(
                            f"❌ Non-retryable error (HTTP {status_code}): "
                            f"{self._summarize_api_error(api_error)}"
                        )
                        self._vprint(f"{self.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                        self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                        self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                        # Actionable guidance for common auth errors
                        if classified.is_auth or classified.reason == FailoverReason.billing:
                            if _provider == "openai-codex" and status_code == 401:
                                self._vprint(f"{self.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                self._vprint(f"{self.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                self._vprint(f"{self.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                self._vprint(f"{self.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            else:
                                self._vprint(f"{self.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                                self._vprint(f"{self.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                                self._vprint(f"{self.log_prefix}      • Does your account have access to {_model}?", force=True)
                                if base_url_host_matches(str(_base), "openrouter.ai"):
                                    self._vprint(f"{self.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                        logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")
                        # Skip session persistence when the error is likely
                        # context-overflow related (status 400 + large session).
                        # Persisting the failed user message would make the
                        # session even larger, causing the same failure on the
                        # next attempt. (#1630)
                        if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Skipping session persistence "
                                f"for large failed session to prevent growth loop.",
                                force=True,
                            )
                        else:
                            self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": str(api_error),
                        }

                    if retry_count >= max_retries:
                        # Before falling back, try rebuilding the primary
                        # client once for transient transport errors (stale
                        # connection pool, TCP reset).  Only attempted once
                        # per API call block.
                        if not primary_recovery_attempted and self._try_recover_primary_transport(
                            api_error, retry_count=retry_count, max_retries=max_retries,
                        ):
                            primary_recovery_attempted = True
                            retry_count = 0
                            continue
                        # Try fallback before giving up entirely
                        self._emit_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        _final_summary = self._summarize_api_error(api_error)
                        if is_rate_limited:
                            self._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                        else:
                            self._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                        self._vprint(f"{self.log_prefix}   💀 Final error: {_final_summary}", force=True)

                        # Detect SSE stream-drop pattern (e.g. "Network
                        # connection lost") and surface actionable guidance.
                        # This typically happens when the model generates a
                        # very large tool call (write_file with huge content)
                        # and the proxy/CDN drops the stream mid-response.
                        _is_stream_drop = (
                            not getattr(api_error, "status_code", None)
                            and any(p in error_msg for p in (
                                "connection lost", "connection reset",
                                "connection closed", "network connection",
                                "network error", "terminated",
                            ))
                        )
                        if _is_stream_drop:
                            self._vprint(
                                f"{self.log_prefix}   💡 The provider's stream "
                                f"connection keeps dropping. This often happens "
                                f"when the model tries to write a very large "
                                f"file in a single tool call.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      Try asking the model "
                                f"to use execute_code with Python's open() for "
                                f"large files, or to write the file in smaller "
                                f"sections.",
                                force=True,
                            )

                        logging.error(
                            "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                            self.log_prefix, max_retries, _final_summary,
                            _provider, _model, len(api_messages), f"{approx_tokens:,}",
                        )
                        if api_kwargs is not None:
                            self._dump_api_request_debug(
                                api_kwargs, reason="max_retries_exhausted", error=api_error,
                            )
                        self._persist_session(messages, conversation_history)
                        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                        if _is_stream_drop:
                            _final_response += (
                                "\n\nThe provider's stream connection keeps "
                                "dropping — this often happens when generating "
                                "very large tool call responses (e.g. write_file "
                                "with long content). Try asking me to use "
                                "execute_code with Python's open() for large "
                                "files, or to write in smaller sections."
                            )
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _final_summary,
                        }

                    # For rate limits, respect the Retry-After header if present
                    _retry_after = None
                    if is_rate_limited:
                        _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                        if _resp_headers and hasattr(_resp_headers, "get"):
                            _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                            if _ra_raw:
                                try:
                                    _retry_after = min(float(_ra_raw), 120)  # Cap at 2 minutes
                                except (TypeError, ValueError):
                                    pass
                    wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                    if is_rate_limited:
                        self._emit_status(f"⏱️ Rate limited. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries})...")
                    else:
                        self._emit_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                    logger.warning(
                        "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                        wait_time,
                        retry_count,
                        max_retries,
                        self._client_log_context(),
                        api_error,
                    )
                    # Sleep in small increments so we can respond to interrupts quickly
                    # instead of blocking the entire wait_time in one sleep() call
                    sleep_end = time.time() + wait_time
                    _backoff_touch_counter = 0
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)  # Check interrupt every 200ms
                        # Touch activity every ~30s so the gateway's inactivity
                        # monitor knows we're alive during backoff waits.
                        _backoff_touch_counter += 1
                        if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                            self._touch_activity(
                                f"error retry backoff ({retry_count}/{max_retries}), "
                                f"{int(sleep_end - time.time())}s remaining"
                            )
            
            # If the API call was interrupted, skip response processing
            if interrupted:
                _turn_exit_reason = "interrupted_during_api_call"
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                # Count compression restarts toward the retry limit to prevent
                # infinite loops when compression reduces messages but not enough
                # to fit the context window.
                retry_count += 1
                restart_with_compressed_messages = False
                continue

            if restart_with_length_continuation:
                # Progressively boost the output token budget on each retry.
                # Retry 1 → 2× base, retry 2 → 3× base, capped at 32 768.
                # Applies to all providers via _ephemeral_max_output_tokens.
                _boost_base = self.max_tokens if self.max_tokens else 4096
                _boost = _boost_base * (length_continue_retries + 1)
                self._ephemeral_max_output_tokens = min(_boost, 32768)
                continue

            # Guard: if all retries exhausted without a successful response
            # (e.g. repeated context-length errors that exhausted retry_count),
            # the `response` variable is still None. Break out cleanly.
            if response is None:
                _turn_exit_reason = "all_retries_exhausted_no_response"
                print(f"{self.log_prefix}❌ All API retries exhausted with no successful response.")
                self._persist_session(messages, conversation_history)
                break

            try:
                _transport = self._get_transport()
                _normalize_kwargs = {}
                if self.api_mode == "anthropic_messages":
                    _normalize_kwargs["strip_tool_prefix"] = self._is_anthropic_oauth
                normalized = _transport.normalize_response(response, **_normalize_kwargs)
                assistant_message = normalized
                finish_reason = normalized.finish_reason
                
                # Normalize content to string — some OpenAI-compatible servers
                # (llama-server, etc.) return content as a dict or list instead
                # of a plain string, which crashes downstream .strip() calls.
                if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                    raw = assistant_message.content
                    if isinstance(raw, dict):
                        assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                    elif isinstance(raw, list):
                        # Multimodal content list — extract text parts
                        parts = []
                        for part in raw:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                        assistant_message.content = "\n".join(parts)
                    else:
                        assistant_message.content = str(raw)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                    _assistant_text = assistant_message.content or ""
                    _invoke_hook(
                        "post_api_request",
                        task_id=effective_task_id,
                        session_id=self.session_id or "",
                        platform=self.platform or "",
                        model=self.model,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_mode=self.api_mode,
                        api_call_count=api_call_count,
                        api_duration=api_duration,
                        finish_reason=finish_reason,
                        message_count=len(api_messages),
                        response_model=getattr(response, "model", None),
                        usage=self._usage_summary_for_api_request_hook(response),
                        assistant_content_chars=len(_assistant_text),
                        assistant_tool_call_count=len(_assistant_tool_calls),
                    )
                except Exception:
                    pass

                # Handle assistant response
                if assistant_message.content and not self.quiet_mode:
                    if self.verbose_logging:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
                    else:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

                # Notify progress callback of model's thinking (used by subagent
                # delegation to relay the child's reasoning to the parent display).
                if (assistant_message.content and self.tool_progress_callback):
                    _think_text = assistant_message.content.strip()
                    # Strip reasoning XML tags that shouldn't leak to parent display
                    _think_text = re.sub(
                        r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                    ).strip()
                    # For subagents: relay first line to parent display (existing behaviour).
                    # For all agents with a structured callback: emit reasoning.available event.
                    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                    if first_line and getattr(self, '_delegate_depth', 0) > 0:
                        try:
                            self.tool_progress_callback("_thinking", first_line)
                        except Exception:
                            pass
                    elif _think_text:
                        try:
                            self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                        except Exception:
                            pass
                
                # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
                # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    self._incomplete_scratchpad_retries += 1
                    
                    self._vprint(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                    
                    if self._incomplete_scratchpad_retries <= 2:
                        self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                        # Don't add the broken message, just retry
                        continue
                    else:
                        # Max retries - discard this turn and save as partial
                        self._vprint(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                        self._incomplete_scratchpad_retries = 0
                        
                        rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)
                        
                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                        }
                
                # Reset incomplete scratchpad counter on clean response
                self._incomplete_scratchpad_retries = 0

                if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                    self._codex_incomplete_retries += 1

                    interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                    interim_has_content = bool((interim_msg.get("content") or "").strip())
                    interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
                    interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))

                    if (
                        interim_has_content
                        or interim_has_reasoning
                        or interim_has_codex_reasoning
                        or interim_has_codex_message_items
                    ):
                        last_msg = messages[-1] if messages else None
                        # Duplicate detection: two consecutive incomplete assistant
                        # messages with identical content AND reasoning are collapsed.
                        # For provider-state-only changes (encrypted reasoning
                        # items or replayable message ids/phases/statuses differ
                        # while visible content/reasoning are unchanged), compare
                        # those opaque payloads too so we don't silently drop the
                        # newer continuation state.
                        last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                        interim_codex_items = interim_msg.get("codex_reasoning_items")
                        last_codex_message_items = last_msg.get("codex_message_items") if isinstance(last_msg, dict) else None
                        interim_codex_message_items = interim_msg.get("codex_message_items")
                        duplicate_interim = (
                            isinstance(last_msg, dict)
                            and last_msg.get("role") == "assistant"
                            and last_msg.get("finish_reason") == "incomplete"
                            and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                            and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                            and last_codex_items == interim_codex_items
                            and last_codex_message_items == interim_codex_message_items
                        )
                        if not duplicate_interim:
                            messages.append(interim_msg)
                            self._emit_interim_assistant_message(interim_msg)

                    if self._codex_incomplete_retries < 3:
                        if not self.quiet_mode:
                            self._vprint(f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)")
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    self._codex_incomplete_retries = 0
                    self._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Codex response remained incomplete after 3 continuation attempts",
                    }
                elif hasattr(self, "_codex_incomplete_retries"):
                    self._codex_incomplete_retries = 0
                
                # Check for tool calls
                if assistant_message.tool_calls:
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
                    # Validate tool call names - detect model hallucinations
                    # Repair mismatched tool names before validating
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in self.valid_tool_names:
                            repaired = self._repair_tool_call(tc.function.name)
                            if repaired:
                                print(f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                                tc.function.name = repaired
                    invalid_tool_calls = [
                        tc.function.name for tc in assistant_message.tool_calls
                        if tc.function.name not in self.valid_tool_names
                    ]
                    if invalid_tool_calls:
                        # Track retries for invalid tool calls
                        self._invalid_tool_retries += 1

                        # Return helpful error to model — model can self-correct next turn
                        available = ", ".join(sorted(self.valid_tool_names))
                        invalid_name = invalid_tool_calls[0]
                        invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                        self._vprint(f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)")

                        if self._invalid_tool_retries >= 3:
                            self._vprint(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                            self._invalid_tool_retries = 0
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}"
                            }

                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        messages.append(assistant_msg)
                        for tc in assistant_message.tool_calls:
                            if tc.function.name not in self.valid_tool_names:
                                content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                            else:
                                content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                            messages.append({
                                "role": "tool",
                                "name": tc.function.name,
                                "tool_call_id": tc.id,
                                "content": content,
                            })
                        continue
                    # Reset retry counter on successful tool call validation
                    self._invalid_tool_retries = 0
                    
                    # Validate tool call arguments are valid JSON
                    # Handle empty strings as empty objects (common model quirk)
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, (dict, list)):
                            tc.function.arguments = json.dumps(args)
                            continue
                        if args is not None and not isinstance(args, str):
                            tc.function.arguments = str(args)
                            args = tc.function.arguments
                        # Treat empty/whitespace strings as empty object
                        if not args or not args.strip():
                            tc.function.arguments = "{}"
                            continue
                        try:
                            json.loads(args)
                        except json.JSONDecodeError as e:
                            invalid_json_args.append((tc.function.name, str(e)))
                    
                    if invalid_json_args:
                        # Check if the invalid JSON is due to truncation rather
                        # than a model formatting mistake.  Routers sometimes
                        # rewrite finish_reason from "length" to "tool_calls",
                        # hiding the truncation from the length handler above.
                        # Detect truncation: args that don't end with } or ]
                        # (after stripping whitespace) are cut off mid-stream.
                        _truncated = any(
                            not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                            for tc in assistant_message.tool_calls
                            if tc.function.name in {n for n, _ in invalid_json_args}
                        )
                        if _truncated:
                            self._vprint(
                                f"{self.log_prefix}⚠️  Truncated tool call arguments detected "
                                f"(finish_reason={finish_reason!r}) — refusing to execute.",
                                force=True,
                            )
                            self._invalid_json_retries = 0
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit",
                            }

                        # Track retries for invalid JSON arguments
                        self._invalid_json_retries += 1

                        tool_name, error_msg = invalid_json_args[0]
                        self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                        if self._invalid_json_retries < 3:
                            self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            # Instead of returning partial, inject tool error results so the model can recover.
                            # Using tool results (not user messages) preserves role alternation.
                            self._vprint(f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                            self._invalid_json_retries = 0  # Reset for next attempt
                            
                            # Append the assistant message with its (broken) tool_calls
                            recovery_assistant = self._build_assistant_message(assistant_message, finish_reason)
                            messages.append(recovery_assistant)
                            
                            # Respond with tool error results for each tool call
                            invalid_names = {name for name, _ in invalid_json_args}
                            for tc in assistant_message.tool_calls:
                                if tc.function.name in invalid_names:
                                    err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                    tool_result = (
                                        f"Error: Invalid JSON arguments. {err}. "
                                        f"For tools with no required parameters, use an empty object: {{}}. "
                                        f"Please retry with valid JSON."
                                    )
                                else:
                                    tool_result = "Skipped: other tool call in this response had invalid JSON."
                                messages.append({
                                    "role": "tool",
                                    "name": tc.function.name,
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                })
                            continue
                    
                    # Reset retry counter on successful JSON validation
                    self._invalid_json_retries = 0

                    # ── Post-call guardrails ──────────────────────────
                    assistant_message.tool_calls = self._cap_delegate_task_calls(
                        assistant_message.tool_calls
                    )
                    assistant_message.tool_calls = self._deduplicate_tool_calls(
                        assistant_message.tool_calls
                    )

                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    # If this turn has both content AND tool_calls, capture the content
                    # as a fallback final response. Common pattern: model delivers its
                    # answer and calls memory/skill tools as a side-effect in the same
                    # turn. If the follow-up turn after tools is empty, we use this.
                    turn_content = assistant_message.content or ""
                    if turn_content and self._has_content_after_think_block(turn_content):
                        self._last_content_with_tools = turn_content
                        # Only mute subsequent output when EVERY tool call in
                        # this turn is post-response housekeeping (memory, todo,
                        # skill_manage, etc.).  If any substantive tool is present
                        # (search_files, read_file, write_file, terminal, ...),
                        # keep output visible so the user sees progress.
                        _HOUSEKEEPING_TOOLS = frozenset({
                            "memory", "todo", "skill_manage", "session_search",
                        })
                        _all_housekeeping = all(
                            tc.function.name in _HOUSEKEEPING_TOOLS
                            for tc in assistant_message.tool_calls
                        )
                        self._last_content_tools_all_housekeeping = _all_housekeeping
                        if _all_housekeeping and self._has_stream_consumers():
                            self._mute_post_response = True
                        elif self._should_emit_quiet_tool_messages():
                            clean = self._strip_think_blocks(turn_content).strip()
                            if clean:
                                self._vprint(f"  ┊ 💬 {clean}")
                    
                    # Pop thinking-only prefill message(s) before appending
                    # (tool-call path — same rationale as the final-response path).
                    _had_prefill = False
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()
                        _had_prefill = True

                    # Reset prefill counter when tool calls follow a prefill
                    # recovery.  Without this, the counter accumulates across
                    # the whole conversation — a model that intermittently
                    # empties (empty → prefill → tools → empty → prefill →
                    # tools) burns both prefill attempts and the third empty
                    # gets zero recovery.  Resetting here treats each tool-
                    # call success as a fresh start.
                    if _had_prefill:
                        self._thinking_prefill_retries = 0
                        self._empty_content_retries = 0
                    # Successful tool execution — reset the post-tool nudge
                    # flag so it can fire again if the model goes empty on
                    # a LATER tool round.
                    self._post_tool_empty_retried = False

                    messages.append(assistant_msg)
                    self._emit_interim_assistant_message(assistant_msg)

                    # Close any open streaming display (response box, reasoning
                    # box) before tool execution begins.  Intermediate turns may
                    # have streamed early content that opened the response box;
                    # flushing here prevents it from wrapping tool feed lines.
                    # Only signal the display callback — TTS (_stream_callback)
                    # should NOT receive None (it uses None as end-of-stream).
                    if self.stream_delta_callback:
                        try:
                            self.stream_delta_callback(None)
                        except Exception:
                            pass

                    self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                    if self._tool_guardrail_halt_decision is not None:
                        decision = self._tool_guardrail_halt_decision
                        _turn_exit_reason = "guardrail_halt"
                        final_response = self._toolguard_controlled_halt_response(decision)
                        self._emit_status(
                            f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}"
                        )
                        messages.append({"role": "assistant", "content": final_response})
                        break

                    # Reset per-turn retry counters after successful tool
                    # execution so a single truncation doesn't poison the
                    # entire conversation.
                    truncated_tool_call_retries = 0

                    # Signal that a paragraph break is needed before the next
                    # streamed text.  We don't emit it immediately because
                    # multiple consecutive tool iterations would stack up
                    # redundant blank lines.  Instead, _fire_stream_delta()
                    # will prepend a single "\n\n" the next time real text
                    # arrives.
                    self._stream_needs_break = True

                    # Refund the iteration if the ONLY tool(s) called were
                    # execute_code (programmatic tool calling).  These are
                    # cheap RPC-style calls that shouldn't eat the budget.
                    _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                    if _tc_names == {"execute_code"}:
                        self.iteration_budget.refund()
                    
                    # Use real token counts from the API response to decide
                    # compression.  prompt_tokens + completion_tokens is the
                    # actual context size the provider reported plus the
                    # assistant turn — a tight lower bound for the next prompt.
                    # Tool results appended above aren't counted yet, but the
                    # threshold (default 50%) leaves ample headroom; if tool
                    # results push past it, the next API call will report the
                    # real total and trigger compression then.
                    #
                    # If last_prompt_tokens is 0 (stale after API disconnect
                    # or provider returned no usage data), fall back to rough
                    # estimate to avoid missing compression.  Without this,
                    # a session can grow unbounded after disconnects because
                    # should_compress(0) never fires.  (#2153)
                    _compressor = self.context_compressor
                    if _compressor.last_prompt_tokens > 0:
                        # Only use prompt_tokens — completion/reasoning
                        # tokens don't consume context window space.
                        # Thinking models (GLM-5.1, QwQ, DeepSeek R1)
                        # inflate completion_tokens with reasoning,
                        # causing premature compression.  (#12026)
                        _real_tokens = _compressor.last_prompt_tokens
                    else:
                        # Include tool schemas — with 50+ tools enabled
                        # these add 20-30K tokens the messages-only
                        # estimate misses, which can skip compression
                        # past the configured threshold (#14695).
                        _real_tokens = estimate_request_tokens_rough(
                            messages, tools=self.tools or None
                        )

                    if self.compression_enabled and _compressor.should_compress(_real_tokens):
                        self._safe_print("  ⟳ compacting context…")
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=self.context_compressor.last_prompt_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history so
                        # _flush_messages_to_session_db writes compressed messages
                        # to the new session (see preflight compression comment).
                        conversation_history = None
                    
                    # Save session log incrementally (so progress is visible even if interrupted)
                    self._session_messages = messages
                    self._save_session_log(messages)
                    
                    # Continue loop for next response
                    continue
                
                else:
                    # No tool calls - this is the final response
                    final_response = assistant_message.content or ""
                    
                    # Fix: unmute output when entering the no-tool-call branch
                    # so the user can see empty-response warnings and recovery
                    # status messages.  _mute_post_response was set during a
                    # prior housekeeping tool turn and should not silence the
                    # final response path.
                    self._mute_post_response = False
                    
                    # Check if response only has think block with no actual content after it
                    if not self._has_content_after_think_block(final_response):
                        # ── Partial stream recovery ─────────────────────
                        # If content was already streamed to the user before
                        # the connection died, use it as the final response
                        # instead of falling through to prior-turn fallback
                        # or wasting API calls on retries.
                        _partial_streamed = (
                            getattr(self, "_current_streamed_assistant_text", "") or ""
                        )
                        if self._has_content_after_think_block(_partial_streamed):
                            _turn_exit_reason = "partial_stream_recovery"
                            _recovered = self._strip_think_blocks(_partial_streamed).strip()
                            logger.info(
                                "Partial stream content delivered (%d chars) "
                                "— using as final response",
                                len(_recovered),
                            )
                            self._emit_status(
                                "↻ Stream interrupted — using delivered content "
                                "as final response"
                            )
                            final_response = _recovered
                            self._response_was_previewed = True
                            break

                        # If the previous turn already delivered real content alongside
                        # HOUSEKEEPING tool calls (e.g. "You're welcome!" + memory save),
                        # the model has nothing more to say. Use the earlier content
                        # immediately instead of wasting API calls on retries.
                        # NOTE: Only use this shortcut when ALL tools in that turn were
                        # housekeeping (memory, todo, etc.).  When substantive tools
                        # were called (terminal, search_files, etc.), the content was
                        # likely mid-task narration ("I'll scan the directory...") and
                        # the empty follow-up means the model choked — let the
                        # post-tool nudge below handle that instead of exiting early.
                        fallback = getattr(self, '_last_content_with_tools', None)
                        if fallback and getattr(self, '_last_content_tools_all_housekeeping', False):
                            _turn_exit_reason = "fallback_prior_turn_content"
                            logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                            self._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                            self._last_content_with_tools = None
                            self._last_content_tools_all_housekeeping = False
                            self._empty_content_retries = 0
                            # Do NOT modify the assistant message content — the
                            # old code injected "Calling the X tools..." which
                            # poisoned the conversation history.  Just use the
                            # fallback text as the final response and break.
                            final_response = self._strip_think_blocks(fallback).strip()
                            self._response_was_previewed = True
                            break

                        # ── Post-tool-call empty response nudge ───────────
                        # The model returned empty after executing tool calls.
                        # This covers two cases:
                        #  (a) No prior-turn content at all — model went silent
                        #  (b) Prior turn had content + SUBSTANTIVE tools (the
                        #      fallback above was skipped because the content
                        #      was mid-task narration, not a final answer)
                        # Instead of giving up, nudge the model to continue by
                        # appending a user-level hint.  This is the #9400 case:
                        # weaker models (mimo-v2-pro, GLM-5, etc.) sometimes
                        # return empty after tool results instead of continuing
                        # to the next step.  One retry with a nudge usually
                        # fixes it.
                        _prior_was_tool = any(
                            m.get("role") == "tool"
                            for m in messages[-5:]  # check recent messages
                        )
                        # Detect Qwen3/Ollama-style in-content thinking blocks.
                        # Ollama puts <think> in the content field (not in
                        # reasoning_content), so _has_structured below would
                        # miss it.  We check here so thinking-only responses
                        # after tool calls route to prefill instead of nudge.
                        _has_inline_thinking = bool(
                            re.search(
                                r'<think>|<thinking>|<reasoning>',
                                final_response or "",
                                re.IGNORECASE,
                            )
                        )
                        if (
                            _prior_was_tool
                            and not getattr(self, "_post_tool_empty_retried", False)
                            and not _has_inline_thinking  # thinking model still working — let prefill handle
                        ):
                            self._post_tool_empty_retried = True
                            # Clear stale narration so it doesn't resurface
                            # on a later empty response after the nudge.
                            self._last_content_with_tools = None
                            self._last_content_tools_all_housekeeping = False
                            logger.info(
                                "Empty response after tool calls — nudging model "
                                "to continue processing"
                            )
                            self._emit_status(
                                "⚠️ Model returned empty after tool calls — "
                                "nudging to continue"
                            )
                            # Append the empty assistant message first so the
                            # message sequence stays valid:
                            #   tool(result) → assistant("(empty)") → user(nudge)
                            # Without this, we'd have tool → user which most
                            # APIs reject as an invalid sequence.
                            _nudge_msg = self._build_assistant_message(assistant_message, finish_reason)
                            _nudge_msg["content"] = "(empty)"
                            _nudge_msg["_empty_recovery_synthetic"] = True
                            messages.append(_nudge_msg)
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You just executed tool calls but returned an "
                                    "empty response. Please process the tool "
                                    "results above and continue with the task."
                                ),
                                "_empty_recovery_synthetic": True,
                            })
                            continue

                        # ── Thinking-only prefill continuation ──────────
                        # The model produced structured reasoning (via API
                        # fields) but no visible text content.  Rather than
                        # giving up, append the assistant message as-is and
                        # continue — the model will see its own reasoning
                        # on the next turn and produce the text portion.
                        # Inspired by clawdbot's "incomplete-text" recovery.
                        # Also covers Qwen3/Ollama in-content <think> blocks
                        # (detected above as _has_inline_thinking).
                        _has_structured = bool(
                            getattr(assistant_message, "reasoning", None)
                            or getattr(assistant_message, "reasoning_content", None)
                            or getattr(assistant_message, "reasoning_details", None)
                            or _has_inline_thinking
                        )
                        if _has_structured and self._thinking_prefill_retries < 2:
                            self._thinking_prefill_retries += 1
                            logger.info(
                                "Thinking-only response (no visible content) — "
                                "prefilling to continue (%d/2)",
                                self._thinking_prefill_retries,
                            )
                            self._emit_status(
                                f"↻ Thinking-only response — prefilling to continue "
                                f"({self._thinking_prefill_retries}/2)"
                            )
                            interim_msg = self._build_assistant_message(
                                assistant_message, "incomplete"
                            )
                            interim_msg["_thinking_prefill"] = True
                            messages.append(interim_msg)
                            self._session_messages = messages
                            self._save_session_log(messages)
                            continue

                        # ── Empty response retry ──────────────────────
                        # Model returned nothing usable.  Retry up to 3
                        # times before attempting fallback.  This covers
                        # both truly empty responses (no content, no
                        # reasoning) AND reasoning-only responses after
                        # prefill exhaustion — models like mimo-v2-pro
                        # always populate reasoning fields via OpenRouter,
                        # so the old `not _has_structured` guard blocked
                        # retries for every reasoning model after prefill.
                        _truly_empty = not self._strip_think_blocks(
                            final_response
                        ).strip()
                        _prefill_exhausted = (
                            _has_structured
                            and self._thinking_prefill_retries >= 2
                        )
                        if _truly_empty and (not _has_structured or _prefill_exhausted) and self._empty_content_retries < 3:
                            self._empty_content_retries += 1
                            logger.warning(
                                "Empty response (no content or reasoning) — "
                                "retry %d/3 (model=%s)",
                                self._empty_content_retries, self.model,
                            )
                            self._emit_status(
                                f"⚠️ Empty response from model — retrying "
                                f"({self._empty_content_retries}/3)"
                            )
                            continue

                        # ── Exhausted retries — try fallback provider ──
                        # Before giving up with "(empty)", attempt to
                        # switch to the next provider in the fallback
                        # chain.  This covers the case where a model
                        # (e.g. GLM-4.5-Air) consistently returns empty
                        # due to context degradation or provider issues.
                        if _truly_empty and self._fallback_chain:
                            logger.warning(
                                "Empty response after %d retries — "
                                "attempting fallback (model=%s, provider=%s)",
                                self._empty_content_retries, self.model,
                                self.provider,
                            )
                            self._emit_status(
                                "⚠️ Model returning empty responses — "
                                "switching to fallback provider..."
                            )
                            if self._try_activate_fallback():
                                self._empty_content_retries = 0
                                self._emit_status(
                                    f"↻ Switched to fallback: {self.model} "
                                    f"({self.provider})"
                                )
                                logger.info(
                                    "Fallback activated after empty responses: "
                                    "now using %s on %s",
                                    self.model, self.provider,
                                )
                                continue

                        # Exhausted retries and fallback chain (or no
                        # fallback configured).  Fall through to the
                        # "(empty)" terminal.
                        _turn_exit_reason = "empty_response_exhausted"
                        reasoning_text = self._extract_reasoning(assistant_message)
                        self._drop_trailing_empty_response_scaffolding(messages)
                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        assistant_msg["content"] = "(empty)"
                        # This is a user-facing failure sentinel for the gateway,
                        # not real assistant content. Persisting it makes later
                        # "continue" turns replay assistant("(empty)") as if it
                        # were a meaningful model response, which can keep long
                        # tool-heavy sessions stuck in empty-response loops.
                        assistant_msg["_empty_terminal_sentinel"] = True
                        messages.append(assistant_msg)

                        if reasoning_text:
                            reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                            logger.warning(
                                "Reasoning-only response (no visible content) "
                                "after exhausting retries and fallback. "
                                "Reasoning: %s", reasoning_preview,
                            )
                            self._emit_status(
                                "⚠️ Model produced reasoning but no visible "
                                "response after all retries. Returning empty."
                            )
                        else:
                            logger.warning(
                                "Empty response (no content or reasoning) "
                                "after %d retries. No fallback available. "
                                "model=%s provider=%s",
                                self._empty_content_retries, self.model,
                                self.provider,
                            )
                            self._emit_status(
                                "❌ Model returned no content after all retries"
                                + (" and fallback attempts." if self._fallback_chain else
                                   ". No fallback providers configured.")
                            )

                        final_response = "(empty)"
                        break
                    
                    # Reset retry counter/signature on successful content
                    self._empty_content_retries = 0
                    self._thinking_prefill_retries = 0

                    if (
                        self.api_mode == "codex_responses"
                        and self.valid_tool_names
                        and codex_ack_continuations < 2
                        and self._looks_like_codex_intermediate_ack(
                            user_message=user_message,
                            assistant_content=final_response,
                            messages=messages,
                        )
                    ):
                        codex_ack_continuations += 1
                        interim_msg = self._build_assistant_message(assistant_message, "incomplete")
                        messages.append(interim_msg)
                        self._emit_interim_assistant_message(interim_msg)

                        continue_msg = {
                            "role": "user",
                            "content": (
                                "[System: Continue now. Execute the required tool calls and only "
                                "send your final answer after completing the task.]"
                            ),
                        }
                        messages.append(continue_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    codex_ack_continuations = 0

                    if truncated_response_prefix:
                        final_response = truncated_response_prefix + final_response
                        truncated_response_prefix = ""
                        length_continue_retries = 0
                    
                    final_response = self._strip_think_blocks(final_response).strip()
                    
                    final_msg = self._build_assistant_message(assistant_message, finish_reason)

                    # Pop thinking-only prefill and empty-response retry
                    # scaffolding before appending the final response.  These
                    # internal turns are only for the next API retry and should
                    # not become durable transcript context.
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and (
                            messages[-1].get("_thinking_prefill")
                            or messages[-1].get("_empty_recovery_synthetic")
                            or messages[-1].get("_empty_terminal_sentinel")
                        )
                    ):
                        messages.pop()

                    messages.append(final_msg)
                    
                    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    logger.error(error_msg)
                
                logger.debug("Outer loop error in API call #%d", api_call_count, exc_info=True)
                
                # If an assistant message with tool_calls was already appended,
                # the API expects a role="tool" result for every tool_call_id.
                # Fill in error results for any that weren't answered yet.
                for idx in range(len(messages) - 1, -1, -1):
                    msg = messages[idx]
                    if not isinstance(msg, dict):
                        break
                    if msg.get("role") == "tool":
                        continue
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        answered_ids = {
                            m["tool_call_id"]
                            for m in messages[idx + 1:]
                            if isinstance(m, dict) and m.get("role") == "tool"
                        }
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict): continue
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "name": AIAgent._get_tool_call_name_static(tc),
                                    "tool_call_id": tc["id"],
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                messages.append(err_msg)
                    break
                
                # Non-tool errors don't need a synthetic message injected.
                # The error is already printed to the user (line above), and
                # the retry loop continues.  Injecting a fake user/assistant
                # message pollutes history, burns tokens, and risks violating
                # role-alternation invariants.

                # If we're near the limit, break to avoid infinite loops
                if api_call_count >= self.max_iterations - 1:
                    _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    # Append as assistant so the history stays valid for
                    # session resume (avoids consecutive user messages).
                    messages.append({"role": "assistant", "content": final_response})
                    break
        
        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ):
            # Budget exhausted — ask the model for a summary via one extra
            # API call with tools stripped.  _handle_max_iterations injects a
            # user message and makes a single toolless request.
            _turn_exit_reason = f"max_iterations_reached({api_call_count}/{self.max_iterations})"
            self._emit_status(
                f"⚠️ Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                "— asking model to summarise"
            )
            if not self.quiet_mode:
                self._safe_print(
                    f"\n⚠️  Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                    "— requesting summary..."
                )
            final_response = self._handle_max_iterations(messages, api_call_count)

            # If running as a kanban worker, block the task so the dispatcher
            # knows the worker could not complete (rather than treating it as a
            # protocol violation).  The agent loop strips tools before calling
            # _handle_max_iterations, so the model cannot call kanban_block
            # itself — we must do it on its behalf.
            _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
            if _kanban_task:
                try:
                    handle_function_call(
                        "kanban_block",
                        {
                            "task_id": _kanban_task,
                            "reason": (
                                f"Iteration budget exhausted "
                                f"({api_call_count}/{self.max_iterations}) — "
                                "task could not complete within the allowed "
                                "iterations"
                            ),
                        },
                        task_id=effective_task_id,
                    )
                    logger.info(
                        "kanban_block called for task %s after iteration "
                        "exhaustion (%d/%d)",
                        _kanban_task, api_call_count, self.max_iterations,
                    )
                except Exception:
                    logger.warning(
                        "Failed to call kanban_block after iteration "
                        "exhaustion for task %s",
                        _kanban_task,
                        exc_info=True,
                    )

        # Determine if conversation completed successfully
        completed = final_response is not None and api_call_count < self.max_iterations

        # Save trajectory if enabled.  ``user_message`` may be a multimodal
        # list of parts; the trajectory format wants a plain string.
        self._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)

        # Clean up VM and browser for this task after conversation completes
        self._cleanup_task_resources(effective_task_id)

        # Persist session to both JSON log and SQLite only after private retry
        # scaffolding has been removed. Otherwise a later user "continue" turn
        # can replay assistant("(empty)") / recovery nudges and fall into the
        # same empty-response loop again.
        self._drop_trailing_empty_response_scaffolding(messages)
        self._persist_session(messages, conversation_history)

        # ── Turn-exit diagnostic log ─────────────────────────────────────
        # Always logged at INFO so agent.log captures WHY every turn ended.
        # When the last message is a tool result (agent was mid-work), log
        # at WARNING — this is the "just stops" scenario users report.
        _last_msg_role = messages[-1].get("role") if messages else None
        _last_tool_name = None
        if _last_msg_role == "tool":
            # Walk back to find the assistant message with the tool call
            for _m in reversed(messages):
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    _tcs = _m["tool_calls"]
                    if _tcs and isinstance(_tcs[0], dict):
                        _last_tool_name = _tcs[-1].get("function", {}).get("name")
                    break

        _turn_tool_count = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
        )
        _resp_len = len(final_response) if final_response else 0
        _budget_used = self.iteration_budget.used if self.iteration_budget else 0
        _budget_max = self.iteration_budget.max_total if self.iteration_budget else 0

        _diag_msg = (
            "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
            "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
        )
        _diag_args = (
            _turn_exit_reason, self.model, api_call_count, self.max_iterations,
            _budget_used, _budget_max,
            _turn_tool_count, _last_msg_role, _resp_len,
            self.session_id or "none",
        )

        if _last_msg_role == "tool" and not interrupted:
            # Agent was mid-work — this is the "just stops" case.
            logger.warning(
                "Turn ended with pending tool result (agent may appear stuck). "
                + _diag_msg + " last_tool=%s",
                *_diag_args, _last_tool_name,
            )
        else:
            logger.info(_diag_msg, *_diag_args)

        # File-mutation verifier footer.
        # If one or more ``write_file`` / ``patch`` calls failed during this
        # turn and were never superseded by a successful write to the same
        # path, append an advisory footer to the assistant response.  This
        # catches the specific case — reported by Ben Eng (#15524-adjacent)
        # — where a model issues a batch of parallel patches, half of them
        # fail with "Could not find old_string", and the model summarises
        # the turn claiming every file was edited.  The user then has to
        # manually run ``git status`` to catch the lie.  With this footer
        # the truth is surfaced on every turn, so over-claiming is
        # structurally impossible past the model.
        #
        # Gate: only applied when a real text response exists for this
        # turn and the user didn't interrupt.  Empty/interrupted turns
        # already have other surface text that shouldn't be augmented.
        if final_response and not interrupted:
            try:
                _failed = getattr(self, "_turn_failed_file_mutations", None) or {}
                if _failed and self._file_mutation_verifier_enabled():
                    footer = self._format_file_mutation_failure_footer(_failed)
                    if footer:
                        final_response = final_response.rstrip() + "\n\n" + footer
            except Exception as _ver_err:
                logger.debug("file-mutation verifier footer failed: %s", _ver_err)

        # Plugin hook: transform_llm_output
        # Fired once per turn after the tool-calling loop completes.
        # Plugins can transform the LLM's output text before it's returned.
        # First hook to return a string wins; None/empty return leaves text unchanged.
        if final_response and not interrupted:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _transform_results = _invoke_hook(
                    "transform_llm_output",
                    response_text=final_response,
                    session_id=self.session_id or "",
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
                for _hook_result in _transform_results:
                    if isinstance(_hook_result, str) and _hook_result:
                        final_response = _hook_result
                        break  # First non-empty string wins
            except Exception as exc:
                logger.warning("transform_llm_output hook failed: %s", exc)

        # Plugin hook: post_llm_call
        # Fired once per turn after the tool-calling loop completes.
        # Plugins can use this to persist conversation data (e.g. sync
        # to an external memory system).
        if final_response and not interrupted:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "post_llm_call",
                    session_id=self.session_id,
                    user_message=original_user_message,
                    assistant_response=final_response,
                    conversation_history=list(messages),
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("post_llm_call hook failed: %s", exc)

        # Extract reasoning from the CURRENT turn only.  Walk backwards
        # but stop at the user message that started this turn — anything
        # earlier is from a prior turn and must not leak into the reasoning
        # box (confusing stale display; #17055).  Within the current turn
        # we still want the *most recent* non-empty reasoning: many
        # providers (Claude thinking, DeepSeek v4, Codex Responses) emit
        # reasoning on the tool-call step and leave the final-answer step
        # with reasoning=None, so picking only the last assistant would
        # silently drop legitimate same-turn reasoning.
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                break  # turn boundary — don't cross into prior turns
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        # Build result with interrupt info if applicable
        result = {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "turn_exit_reason": _turn_exit_reason,
            "partial": False,  # True only when stopped due to invalid tool calls
            "interrupted": interrupted,
            "response_previewed": getattr(self, "_response_was_previewed", False),
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "input_tokens": self.session_input_tokens,
            "output_tokens": self.session_output_tokens,
            "cache_read_tokens": self.session_cache_read_tokens,
            "cache_write_tokens": self.session_cache_write_tokens,
            "reasoning_tokens": self.session_reasoning_tokens,
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "last_prompt_tokens": getattr(self.context_compressor, "last_prompt_tokens", 0) or 0,
            "estimated_cost_usd": self.session_estimated_cost_usd,
            "cost_status": self.session_cost_status,
            "cost_source": self.session_cost_source,
        }
        if self._tool_guardrail_halt_decision is not None:
            result["guardrail"] = self._tool_guardrail_halt_decision.to_metadata()
        # If a /steer landed after the final assistant turn (no more tool
        # batches to drain into), hand it back to the caller so it can be
        # delivered as the next user turn instead of being silently lost.
        _leftover_steer = self._drain_pending_steer()
        if _leftover_steer:
            result["pending_steer"] = _leftover_steer
        self._response_was_previewed = False
        
        # Include interrupt message if one triggered the interrupt
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        # Clear interrupt state after handling
        self.clear_interrupt()

        # Clear stream callback so it doesn't leak into future calls
        self._stream_callback = None

        # Check skill trigger NOW — based on how many tool iterations THIS turn used.
        _should_review_skills = False
        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            _should_review_skills = True
            self._iters_since_skill = 0

        # External memory provider: sync the completed turn + queue next prefetch.
        self._sync_external_memory_for_turn(
            original_user_message=original_user_message,
            final_response=final_response,
            interrupted=interrupted,
        )

        # Background memory/skill review — runs AFTER the response is delivered
        # so it never competes with the user's task for model attention.
        if final_response and not interrupted and (_should_review_memory or _should_review_skills):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                )
            except Exception:
                pass  # Background review is best-effort

        # Note: Memory provider on_session_end() + shutdown_all() are NOT
        # called here — run_conversation() is called once per user message in
        # multi-turn sessions. Shutting down after every turn would kill the
        # provider before the second message. Actual session-end cleanup is
        # handled by the CLI (atexit / /reset) and gateway (session expiry /
        # _reset_session).

        # Plugin hook: on_session_end
        # Fired at the very end of every run_conversation call.
        # Plugins can use this for cleanup, flushing buffers, etc.
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_end",
                session_id=self.session_id,
                completed=completed,
                interrupted=interrupted,
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("on_session_end hook failed: %s", exc)

        return result

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """
        Simple chat interface that returns just the final response.

        Args:
            message (str): User message
            stream_callback: Optional callback invoked with each text delta during streaming.

        Returns:
            str: Final assistant response
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20
):
    """
    Main function for running the agent directly.

    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use (OpenRouter format: provider/model). Defaults to anthropic/claude-sonnet-4.6.
        api_key (str): API key for authentication. Uses OPENROUTER_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://openrouter.ai/api/v1
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_toolsets (str): Comma-separated list of toolsets to enable. Supports predefined
                              toolsets (e.g., "research", "development", "safe").
                              Multiple toolsets can be combined: "web,vision"
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files (appends to trajectory_samples.jsonl). Defaults to False.
        save_sample (bool): Save a single trajectory sample to a UUID-named JSONL file for inspection. Defaults to False.
        verbose (bool): Enable verbose logging for debugging. Defaults to False.
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses. Defaults to 20.

    Toolset Examples:
        - "research": Web search, extract, crawl + vision tools
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    
    # Handle tool listing
    if list_tools:
        from model_tools import get_all_tool_names, get_available_toolsets
        from toolsets import get_all_toolsets, get_toolset_info
        
        print("📋 Available Tools & Toolsets:")
        print("-" * 50)
        
        # Show new toolsets system
        print("\n🎯 Predefined Toolsets (New System):")
        print("-" * 40)
        all_toolsets = get_all_toolsets()
        
        # Group by category
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []
        
        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in {"web", "terminal", "vision", "creative", "reasoning"}:
                    basic_toolsets.append(entry)
                elif name in {"research", "development", "analysis", "content_creation", "full_stack"}:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)
        
        # Print basic toolsets
        print("\n📌 Basic Toolsets:")
        for name, info in basic_toolsets:
            tools_str = ', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Tools: {tools_str}")
        
        # Print composite toolsets
        print("\n📂 Composite Toolsets (built from other toolsets):")
        for name, info in composite_toolsets:
            includes_str = ', '.join(info['includes']) if info['includes'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Includes: {includes_str}")
            print(f"    Total tools: {info['tool_count']}")
        
        # Print scenario-specific toolsets
        print("\n🎭 Scenario-Specific Toolsets:")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    Total tools: {info['tool_count']}")
        
        
        # Show legacy toolset compatibility
        print("\n📦 Legacy Toolsets (for backward compatibility):")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")
        
        # Show individual tools
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")
        
        print("\n💡 Usage Examples:")
        print("  # Use predefined toolsets")
        print("  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print("  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print("  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print("  ")
        print("  # Combine multiple toolsets")
        print("  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print("  ")
        print("  # Disable toolsets")
        print("  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print("  ")
        print("  # Run with trajectory saving enabled")
        print("  python run_agent.py --save_trajectories --query='your question here'")
        return
    
    # Parse toolset selection arguments
    enabled_toolsets_list = None
    disabled_toolsets_list = None
    
    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")
    
    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")
    
    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")
    
    # Initialize agent with provided parameters
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            log_prefix_chars=log_prefix_chars
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Use provided query or default to Python 3.13 example
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query
    
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)
    
    # Run conversation
    result = agent.run_conversation(user_query)
    
    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")
    
    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result['final_response'])
    
    # Save sample trajectory to UUID-named file if requested
    if save_sample:
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.json"
        
        # Convert messages to trajectory format (same as batch_runner)
        trajectory = agent._convert_to_trajectory_format(
            result['messages'], 
            user_query, 
            result['completed']
        )
        
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result['completed'],
            "query": user_query
        }
        
        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                # Pretty-print JSON with indent for readability
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")
    
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
