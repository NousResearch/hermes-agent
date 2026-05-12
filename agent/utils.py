"""Standalone utility functions and classes extracted from run_agent.py.

Pure functions and standalone classes that don't depend on AIAgent state.
Extracted to reduce the God class size and improve testability.

Previously at lines 77-1027 of run_agent.py.
"""

from __future__ import annotations

import copy
import json
import sys
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

__all__ = [
    "IterationBudget",
    "OpenAI",
    "_DESTRUCTIVE_PATTERNS",
    "_MAX_TOOL_WORKERS",
    "_NEVER_PARALLEL_TOOLS",
    "_OpenAIProxy",
    "_PARALLEL_SAFE_TOOLS",
    "_PATH_SCOPED_TOOLS",
    "_QWEN_CODE_VERSION",
    "_REDIRECT_OVERWRITE",
    "_SURROGATE_RE",
    "_SafeWriter",
    "_append_subdir_hint_to_multimodal",
    "_escape_invalid_chars_in_json_strings",
    "_extract_parallel_scope_path",
    "_get_proxy_for_base_url",
    "_get_proxy_from_env",
    "_hermes_home",
    "_install_safe_stdio",
    "_is_destructive_command",
    "_is_multimodal_tool_result",
    "_load_openai_cls",
    "_loaded_env_paths",
    "_multimodal_text_summary",
    "_openrouter_prewarm_done",
    "_paths_overlap",
    "_pool_may_recover_from_rate_limit",
    "_project_env",
    "_qwen_portal_headers",
    "_repair_tool_call_arguments",
    "_routermint_headers",
    "_sanitize_messages_non_ascii",
    "_sanitize_messages_surrogates",
    "_sanitize_structure_non_ascii",
    "_sanitize_structure_surrogates",
    "_sanitize_surrogates",
    "_sanitize_tools_non_ascii",
    "_should_parallelize_tool_batch",
    "_strip_images_from_messages",
    "_strip_non_ascii",
    "_trajectory_normalize_msg",
    "_OPENAI_CLS_CACHE",
    "_set_interrupt",
    "_codex_derive_responses_function_call_id",
    "_codex_split_responses_tool_id",
    "_detect_tool_failure",
    "_get_cute_tool_message_impl",
    "handle_function_call",
    "base_url_host_matches",
    "base_url_hostname",
    "atomic_json_write",
    "env_var_enabled",
    "normalize_proxy_url",
    "is_local_endpoint",
    "save_context_length",
    "query_ollama_num_ctx",
    "logger",
    "ContextCompressor",
    "StreamingContextScrubber",
    "StreamingThinkScrubber",
    "SubdirectoryHintTracker",
    "apply_anthropic_cache_control",
    "build_memory_context_block",
    "build_skills_system_prompt",
    "build_context_files_prompt",
    "build_environment_hints",
    "classify_api_error",
    "cfg_get",
    "cleanup_browser",
    "cleanup_vm",
    "enforce_turn_budget",
    "estimate_usage_cost",
    "get_active_env",
    "get_hermes_home",
    "is_persistent_env",
    "jittered_backoff",
    "load_hermes_dotenv",
    "load_soul_md",
    "maybe_persist_tool_result",
    "normalize_usage",
    "sanitize_context",
    "get_provider_request_timeout",
    "parse_qs",
    "urlparse",
    "handle_function_call",
    "_get_cute_tool_message_impl",
    "FailoverReason",
    "GOOGLE_MODEL_OPERATIONAL_GUIDANCE",
    "OPENAI_MODEL_EXECUTION_GUIDANCE",
    "TOOL_USE_ENFORCEMENT_GUIDANCE",
    "TOOL_USE_ENFORCEMENT_MODELS",
    "TOOL_USE_ENFORCEMENT_MODELS",
    "DEFAULT_AGENT_IDENTITY",
    "HERMES_AGENT_HELP_GUIDANCE",
    "KANBAN_GUIDANCE",
    "KawaiiSpinner",
    "MEMORY_GUIDANCE",
    "PLATFORM_HINTS",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
    "ToolCallGuardrailConfig",
    "ToolCallGuardrailController",
    "ToolGuardrailDecision",
    "_build_tool_preview",
    "_codex_deterministic_call_id",
    "_get_approval_callback",
    "_get_sudo_password_callback",
    "_get_tool_emoji",
    "_save_trajectory_to_file",
    "_set_approval_callback",
    "_set_sudo_password_callback",
    "_summarize_user_message_for_log",
    "append_toolguard_guidance",
    "build_nous_subscription_prompt",
    "check_toolset_requirements",
    "convert_scratchpad_to_think",
    "estimate_messages_tokens_rough",
    "estimate_request_tokens_rough",
    "estimate_tokens_rough",
    "fetch_model_metadata",
    "get_next_probe_tier",
    "get_provider_stale_timeout",
    "get_tool_definitions",
    "get_toolset_for_tool",
    "has_incomplete_scratchpad",
    "parse_available_output_tokens_from_error",
    "parse_context_limit_from_error",
    "toolguard_synthetic_result",
]

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
            if isinstance(p, dict) and p.get("type") in ("image", "image_url", "input_image"):
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
            if isinstance(part, dict) and part.get("type") in ("image_url", "image", "input_image"):
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


