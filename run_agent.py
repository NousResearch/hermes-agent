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
import copy
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import re
import sys
import tempfile
import time
import threading
import uuid
import warnings
from typing import List, Dict, Any, Optional, Callable
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
from types import SimpleNamespace

from hermes_constants import get_hermes_home


def _launch_cwd_for_session(source: str) -> Optional[str]:
    """Working directory to stamp on a new session row, or None.

    Only local CLI sessions get a recorded cwd: the directory the process was
    launched from is meaningful for ``hermes -c`` / ``--resume`` (relaunch
    where you left off). Gateway/cron/remote-backend sessions have no stable
    host cwd to restore, so they record nothing.

    ``TERMINAL_ENV`` is set by the CLI's config bridge (``load_cli_config``);
    a non-"local" backend (docker/ssh/modal/...) means the host cwd is
    irrelevant to the agent's tools, so we skip it there too.
    """
    if source != "cli":
        return None
    backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
    if backend and backend != "local":
        return None
    try:
        return os.getcwd()
    except OSError:
        # cwd was unlinked out from under us — nothing meaningful to record.
        return None


def _session_source_for_agent(platform: Optional[str]) -> str:
    try:
        from gateway.session_context import get_session_env

        source = get_session_env("HERMES_SESSION_SOURCE", "")
    except Exception:
        source = os.environ.get("HERMES_SESSION_SOURCE", "")
    source = str(source or "").strip()
    if source:
        return source
    return platform or "cli"


def _gateway_origin_json(agent: "AIAgent") -> Optional[str]:
    """Build the gateway routing ``origin_json`` for a session row.

    Mirrors the shape of ``SessionSource.to_dict()`` (platform, chat_id,
    chat_name, chat_type, user_id, user_name, thread_id, optional
    user_id_alt / profile) so consumers that read ``origin_json`` from
    state.db (channel directory, mcp_serve, mirror) see the same fields the
    gateway's own ``record_gateway_session_peer`` would write. Returns None
    when the agent carries no gateway identity (plain CLI session), matching
    the previous identity-less creation.
    """
    chat_id = getattr(agent, "_chat_id", None)
    session_key = getattr(agent, "_gateway_session_key", None)
    user_id = getattr(agent, "_user_id", None)
    if not (chat_id or session_key or user_id):
        return None
    origin: Dict[str, Any] = {
        "platform": getattr(agent, "platform", None) or "",
        "chat_id": chat_id,
        "chat_name": getattr(agent, "_chat_name", None),
        "chat_type": getattr(agent, "_chat_type", None) or "dm",
        "user_id": user_id,
        "user_name": getattr(agent, "_user_name", None),
        "thread_id": getattr(agent, "_thread_id", None),
    }
    user_id_alt = getattr(agent, "_user_id_alt", None)
    if user_id_alt:
        origin["user_id_alt"] = user_id_alt
    profile = getattr(agent, "_profile_name", None)
    if not profile:
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile == "default":
                profile = None
        except Exception:
            profile = None
    if profile:
        origin["profile"] = profile
    try:
        return json.dumps(origin)
    except Exception:
        return None


# OpenAI lazy proxy + safe stdio + proxy URL helpers — see agent/process_bootstrap.py.
# `OpenAI` is re-exported here so `patch("run_agent.OpenAI", ...)` in tests works.
# The other `# noqa: F401` re-exports below cover names accessed via
# `mock.patch("run_agent.<X>")`, `from run_agent import <X>` in production
# siblings, or the `_ra().<X>` indirection in agent/system_prompt.py — none
# of which ruff's in-module usage scan can see.
from agent.process_bootstrap import (
    OpenAI,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.OpenAI")
    _SafeWriter,  # noqa: F401  # re-exported for tests that `from run_agent import _SafeWriter`
    _get_proxy_for_base_url,
)
from agent.iteration_budget import IterationBudget
from agent.interrupt_compat import request_hard_interrupt


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
    get_tool_definitions,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.get_tool_definitions")
    get_toolset_for_tool,
    handle_function_call,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.handle_function_call")
    check_toolset_requirements,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.check_toolset_requirements")
)
from tools.terminal_tool import cleanup_vm, get_active_env
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser


# Agent internals extracted to agent/ package for modularity
from agent.memory_manager import sanitize_context
from agent.memory_provider import is_trivial_prompt
from agent.error_classifier import FailoverReason
from agent.redact import redact_sensitive_text
from agent.message_content import flatten_message_text
from agent.session_activity import ActivityProvenance
from agent.model_metadata import (
    estimate_request_tokens_rough,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.estimate_request_tokens_rough")
    is_local_endpoint,
)
from agent.usage_pricing import normalize_usage
# Re-exported for tests that monkeypatch these symbols on run_agent.
from agent.context_compressor import (  # noqa: F401
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    user_originated_turn_view,
)
from agent.retry_utils import jittered_backoff  # noqa: F401
from agent.prompt_builder import (  # noqa: F401  # re-exported via _ra() / mock.patch("run_agent.<name>") / from run_agent import <name>
    DEFAULT_AGENT_IDENTITY,
    build_skills_system_prompt,
    build_context_files_prompt,
    build_environment_hints,
    load_soul_md,
)
from agent.process_bootstrap import _get_proxy_from_env  # noqa: F401
from agent.message_sanitization import (  # noqa: F401
    _SURROGATE_RE,
    _sanitize_surrogates,
    _sanitize_structure_surrogates,
    _sanitize_messages_surrogates,
    _escape_invalid_chars_in_json_strings,
    _repair_tool_call_arguments,
    _strip_non_ascii,
    _sanitize_messages_non_ascii,
    _sanitize_tools_non_ascii,
    _looks_like_image_content_rejection,
    _strip_images_from_messages,
    _sanitize_structure_non_ascii,
    coalesce_tool_call_id as _sanitize_coalesce_tool_call_id,
    uniquify_tool_call_ids as _sanitize_uniquify_tool_call_ids,
)
from agent.codex_responses_adapter import (
    _derive_responses_function_call_id as _codex_derive_responses_function_call_id,
    _deterministic_call_id as _codex_deterministic_call_id,
    _split_responses_tool_id as _codex_split_responses_tool_id,
    _summarize_user_message_for_log,  # also used by _sync_external_memory_for_turn (memory boundary)
)
from agent.tool_guardrails import (
    ToolGuardrailDecision,
    append_toolguard_guidance,
    toolguard_synthetic_result,
)
from agent.tool_result_classification import (
    FILE_MUTATING_TOOL_NAMES as _FILE_MUTATING_TOOLS,
    file_mutation_result_landed,
)
from agent.trajectory import (
    convert_scratchpad_to_think,
    save_trajectory as _save_trajectory_to_file,
)
from agent.tool_dispatch_helpers import (
    _should_parallelize_tool_batch,  # noqa: F401  # re-exported for tests that `from run_agent import _should_parallelize_tool_batch`
    _is_destructive_command,  # noqa: F401  # re-exported for tests that access `run_agent._is_destructive_command`
    _extract_parallel_scope_path,  # noqa: F401  # re-exported for tests that `from run_agent import _extract_parallel_scope_path`
    _paths_overlap,  # noqa: F401  # re-exported for tests that `from run_agent import _paths_overlap`
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _append_subdir_hint_to_multimodal,  # noqa: F401  # re-exported for tests that `from run_agent import _append_subdir_hint_to_multimodal`
    _extract_file_mutation_targets,
    _extract_landed_file_mutation_paths,
    _extract_error_preview,
    _trajectory_normalize_msg,  # noqa: F401  # re-exported for tests that `from run_agent import _trajectory_normalize_msg`
)
from utils import atomic_json_write, base_url_host_matches, base_url_hostname, env_float, is_truthy_value, model_forces_max_completion_tokens


# Internal flags that mark a message as ephemeral empty-response/prefill
# recovery scaffolding: the synthetic assistant "(empty)" turn and user nudge
# injected after an empty response, the terminal "(empty)" sentinel, and the
# thinking-only prefill placeholder. These exist only to drive the next API
# retry; the in-memory loop pops them before appending the real response.
# Persistence must mirror that, otherwise an append-only flush can commit them
# to the session store and a resumed session replays synthetic "(empty)"/nudge
# turns as if they were genuine context.
_EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_empty_recovery_synthetic",
    "_empty_terminal_sentinel",
    "_thinking_prefill",
    # verify-on-stop and pre_verify nudges append a synthetic user nudge to
    # keep the agent going one more turn before it can claim completion.
    # The nudge exists only to drive the verification loop; persisting it
    # poisons the resumed transcript and breaks prompt-prefix cache reuse
    # on later turns. The assistant candidate is NOT synthetic — it is
    # persisted and emitted as an interim message (#65919).
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    # kanban worker stop-guard: narrated exit without kanban_complete/block
    "_kanban_stop_synthetic",
    # dropped tool-call re-prompt pair (finish_reason=tool_calls with an
    # empty tool_calls array): the interim narration-only assistant turn
    # and the "issue the actual tool call now" user nudge exist only to
    # drive the bounded retry. Persisting them would replay the internal
    # retry instruction as user-authored context on resume.
    "_dropped_toolcall_nudge",
)


def _is_ephemeral_scaffolding(msg: Any) -> bool:
    """Return True when ``msg`` is internal recovery scaffolding that must never
    be persisted to the durable transcript (SQLite session store or JSON log)."""
    return isinstance(msg, dict) and any(
        msg.get(flag) for flag in _EPHEMERAL_SCAFFOLDING_FLAGS
    )


_MAX_TOOL_WORKERS = 8

# Intrinsic marker stamped on a message dict once it has been written to the
# SQLite session store.  Used by ``_flush_messages_to_session_db`` to decide
# what is already durable.  An object-identity (``id(msg)``) dedup set cannot be
# trusted across turns: once a flushed message dict is dropped from the live
# list (e.g. by scaffolding rewind or in-place compaction) and garbage-
# collected, CPython is free to hand its address to a brand-new assistant/tool
# message, whose ``id()`` then collides with the stale entry and the real turn
# is silently never persisted.  A marker bound to the dict itself cannot be
# aliased that way.  The ``_`` prefix is mandatory: the wire sanitizers
# (agent/transports/chat_completions.py, agent/chat_completion_helpers.py) strip
# every top-level ``_``-prefixed key before the request leaves the process, so
# this never reaches a strict OpenAI-compatible gateway.
#
# CONTRACT (#92231): the marker asserts "this dict's CONTENT is durable as
# written". Loaded rows are stamped at materialization time
# (hermes_state._rows_to_conversation), so any code that mutates a loaded or
# flushed dict's content in place and needs the change persisted MUST pop the
# marker (and invalidate _db_flush_scan_prefix if the dict may sit inside the
# bounded-scan prefix) — see agent/turn_finalizer.py (fill-empty-tail) and
# agent/context_compressor.py (micro-compaction defrag) for the two canonical
# pop sites. Mutating without popping leaves the DB silently stale.
_DB_PERSISTED_MARKER = "_db_persisted"


# Guard so the OpenRouter metadata pre-warm thread is only spawned once per
# process, not once per AIAgent instantiation.  Without this, long-running
# gateway processes leak one OS thread per incoming message and eventually
# exhaust the system thread limit (RuntimeError: can't start new thread).
_openrouter_prewarm_done = threading.Event()

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


def _pool_may_recover_from_rate_limit(pool) -> bool:
    """Decide whether to wait for credential-pool rotation instead of falling back.

    The existing pool-rotation path requires the pool to (1) exist and (2) have
    at least one entry not currently in exhaustion cooldown.  But rotation is
    only meaningful when the pool has more than one entry.

    With a single-credential pool (common for Vertex service accounts and any
    "one personal key" configuration), the primary entry just 429'd and there
    is nothing to rotate to.  Waiting for the pool cooldown to expire means
    retrying against the same exhausted quota — the daily-quota 429 will recur
    immediately, and the retry budget is burned.

    In that case we must fall back to the configured ``fallback_model``
    instead.  Returns True only when rotation has somewhere to go.

    See issues #11314 and #13636.
    """
    if pool is None:
        return False
    if not pool.has_available():
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


def _safe_session_filename_component(session_id: str) -> str:
    """Return a stable, path-safe filename component for a session ID.

    Session IDs can originate from untrusted input (e.g. the
    ``X-Hermes-Session-Id`` API header) and are otherwise interpolated raw
    into on-disk artifact filenames under ``~/.hermes/sessions/``.  Without
    sanitization, a traversal-shaped ID such as ``../../../../etc/pwned``
    would let a caller write the session snapshot / request dump outside the
    sessions directory.  This collapses every non ``[A-Za-z0-9_-]`` character
    to ``_`` (so no path separators or ``.`` survive), caps the length, and —
    when sanitization changed the string — appends a short content hash so two
    distinct IDs that sanitize to the same component don't collide.  The
    result is always a single, traversal-free path segment.
    """
    raw = str(session_id or "").strip()
    sanitized = re.sub(r"[^\w-]", "_", raw).strip("._")
    sanitized = sanitized[:96] or "session"
    if raw and sanitized == raw:
        return sanitized
    digest = hashlib.sha256(
        raw.encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:12]
    return f"{sanitized}_{digest}"


class _StreamErrorEvent(Exception):
    """Synthesized provider error surfaced from a Responses ``error`` SSE frame.

    Some Codex-style Responses backends (xAI for subscription/quota
    failures, custom relays under malformed-tool-call conditions) emit a
    standalone ``type=error`` frame instead of routing the failure
    through ``response.failed`` or returning an HTTP 4xx.  The fallback
    streaming path raises this exception so ``_summarize_api_error`` and
    ``_extract_api_error_context`` see a familiar ``.body`` /
    ``.status_code`` shape and the entitlement detector can match the
    underlying provider message ("do not have an active Grok
    subscription", etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        param: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param
        self.status_code = status_code
        # OpenAI SDK-shaped body so _extract_api_error_context /
        # _summarize_api_error / classify_api_error all pick it up.
        self.body: Dict[str, Any] = {
            "error": {
                "message": message,
                "code": code,
                "param": param,
                "type": "error",
            }
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
        max_iterations: int = sys.maxsize,  # Default: unlimited tool-calling iterations (shared with subagents)
        tool_delay: float = None,  # Deprecated: accepted for compatibility, ignored
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        tool_progress_mode: str = "all",
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
        read_terminal_callback: callable = None,
        read_preview_callback: callable = None,
        drive_preview_callback: callable = None,
        read_window_below_callback: callable = None,
        setup_mcp_callback: callable = None,
        tour_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None,
        status_callback: callable = None,
        notice_callback: callable = None,
        notice_clear_callback: callable = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
        reaction_callback: Optional[Callable[[str], None]] = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        service_tier: str = None,
        request_overrides: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        user_id: str = None,
        user_id_alt: str = None,
        user_name: str = None,
        chat_id: str = None,
        chat_name: str = None,
        chat_type: str = None,
        thread_id: str = None,
        gateway_session_key: str = None,
        skip_context_files: bool = False,
        load_soul_identity: bool = False,
        skip_memory: bool = False,
        skip_background_review: bool = False,
        session_db=None,
        parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None,
        run_budget_seconds: Optional[float] = None,
        fallback_model: Dict[str, Any] = None,
        credential_pool=None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 20,
        checkpoint_max_total_size_mb: int = 500,
        checkpoint_max_file_size_mb: int = 10,
        pass_session_id: bool = False,
        requested_provider: str = None,
        capabilities: Dict[str, bool] | None = None,
    ):
        """Forwarder — see ``agent.agent_init.init_agent``."""
        if tool_delay is not None:
            warnings.warn(
                "tool_delay is deprecated and ignored; sequential tool calls "
                "no longer sleep between executions.",
                DeprecationWarning,
                stacklevel=2,
            )
        from agent.agent_init import init_agent
        init_agent(
            self,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            requested_provider=requested_provider,
            capabilities=capabilities,
            api_mode=api_mode,
            acp_command=acp_command,
            acp_args=acp_args,
            command=command,
            args=args,
            model=model,
            max_iterations=max_iterations,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            save_trajectories=save_trajectories,
            verbose_logging=verbose_logging,
            quiet_mode=quiet_mode,
            tool_progress_mode=tool_progress_mode,
            ephemeral_system_prompt=ephemeral_system_prompt,
            log_prefix_chars=log_prefix_chars,
            log_prefix=log_prefix,
            providers_allowed=providers_allowed,
            providers_ignored=providers_ignored,
            providers_order=providers_order,
            provider_sort=provider_sort,
            provider_require_parameters=provider_require_parameters,
            provider_data_collection=provider_data_collection,
            openrouter_min_coding_score=openrouter_min_coding_score,
            session_id=session_id,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            thinking_callback=thinking_callback,
            reasoning_callback=reasoning_callback,
            clarify_callback=clarify_callback,
            read_terminal_callback=read_terminal_callback,
            read_preview_callback=read_preview_callback,
            drive_preview_callback=drive_preview_callback,
            read_window_below_callback=read_window_below_callback,
            setup_mcp_callback=setup_mcp_callback,
            tour_callback=tour_callback,
            step_callback=step_callback,
            stream_delta_callback=stream_delta_callback,
            interim_assistant_callback=interim_assistant_callback,
            tool_gen_callback=tool_gen_callback,
            status_callback=status_callback,
            notice_callback=notice_callback,
            notice_clear_callback=notice_clear_callback,
            event_callback=event_callback,
            reaction_callback=reaction_callback,
            max_tokens=max_tokens,
            reasoning_config=reasoning_config,
            service_tier=service_tier,
            request_overrides=request_overrides,
            prefill_messages=prefill_messages,
            platform=platform,
            user_id=user_id,
            user_id_alt=user_id_alt,
            user_name=user_name,
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            thread_id=thread_id,
            gateway_session_key=gateway_session_key,
            skip_context_files=skip_context_files,
            load_soul_identity=load_soul_identity,
            skip_memory=skip_memory,
            skip_background_review=skip_background_review,
            session_db=session_db,
            parent_session_id=parent_session_id,
            iteration_budget=iteration_budget,
            run_budget_seconds=run_budget_seconds,
            fallback_model=fallback_model,
            credential_pool=credential_pool,
            checkpoints_enabled=checkpoints_enabled,
            checkpoint_max_snapshots=checkpoint_max_snapshots,
            checkpoint_max_total_size_mb=checkpoint_max_total_size_mb,
            checkpoint_max_file_size_mb=checkpoint_max_file_size_mb,
            pass_session_id=pass_session_id,
        )

    def _get_session_db_for_recall(self):
        """Return a SessionDB for recall, lazily creating it if an entrypoint forgot.

        Most frontends pass ``session_db`` into ``AIAgent`` explicitly, but recall
        is important enough that a missing constructor argument should degrade by
        opening the default state DB instead of making the advertised
        ``session_search`` tool unusable.
        """
        # Persistence-isolated forks (background review) must not lazily open the
        # canonical state DB: doing so would re-arm _flush_messages_to_session_db
        # to write the fork's harness turn into the user's real session. Recall
        # degrades to None for them (they don't use session_search anyway).
        if getattr(self, "_persist_disabled", False):
            return None
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state import get_shared_session_db

            self._session_db = get_shared_session_db()
            # We opened it here, so nothing else holds a reference — this agent
            # is its only owner and close() must release it.
            self._owns_session_db = True
            return self._session_db
        except Exception:
            logger.debug("SessionDB unavailable for recall", exc_info=True)
            return None

    def _ensure_db_session(self) -> None:
        """Create session DB row on first use. Disables _session_db on failure."""
        if getattr(self, "_persist_disabled", False):
            return
        if self._session_db_created or not self._session_db:
            return
        source = _session_source_for_agent(self.platform)
        try:
            try:
                from hermes_cli.profiles import get_active_profile_name
                _profile_for_session = get_active_profile_name()
                # Persist the profile name EXPLICITLY, including "default".
                # NULL used to stand in for the default profile, but the
                # #94724 legacy-owner backfill already stamps literal
                # "default" onto old rows, and profile-keyed consumers
                # (sidebar scope matching, @session:<profile>/<id> deep
                # links) treat NULL as unowned — rows minted NULL after the
                # one-shot backfill vanished from the sidebar (#99222).
            except Exception:
                _profile_for_session = None
            # Carry the live YOLO bypass into the creation-time model_config so
            # a session whose /yolo was toggled BEFORE the row existed (the row
            # is created lazily on the first turn) still persists the flag for
            # `hermes --resume`. set_session_yolo() no-ops on a missing row, so
            # this is the only chance to record a pre-first-turn toggle.
            _init_model_config = self._session_init_model_config
            try:
                from tools.approval import is_session_yolo_enabled
                if is_session_yolo_enabled(self.session_id):
                    _init_model_config = dict(_init_model_config or {})
                    _init_model_config["yolo_mode"] = True
            except Exception:
                pass
            # Persist the gateway routing identity with the row. The gateway's
            # SessionStore normally creates the row first (db_create_kwargs) and
            # record_gateway_session_peer self-heals a missing row (#82616), but
            # when the default/global state.db is corrupt or unavailable at
            # gateway startup the SessionStore degrades to a JSONL fallback
            # (_db=None) and the peer recorder no-ops. In that degraded mode
            # this lazy creation is the ONLY durable write for the session, so
            # it must carry session_key/chat_id/chat_type/thread_id/user_id/
            # display_name/origin_json or the row is identity-less and
            # unrecoverable by find_latest_gateway_session_for_peer (regression:
            # Telegram rows with chat_id=NULL/session_key=NULL under multiplexed
            # profile routes).
            self._session_db.create_session(
                session_id=self.session_id,
                source=source,
                model=self.model,
                model_config=_init_model_config,
                system_prompt=self._cached_system_prompt,
                user_id=getattr(self, "_user_id", None),
                session_key=getattr(self, "_gateway_session_key", None),
                chat_id=getattr(self, "_chat_id", None),
                chat_type=getattr(self, "_chat_type", None),
                thread_id=getattr(self, "_thread_id", None),
                display_name=(
                    getattr(self, "_chat_name", None)
                    or getattr(self, "_user_name", None)
                ),
                origin_json=_gateway_origin_json(self),
                parent_session_id=self._parent_session_id,
                cwd=_launch_cwd_for_session(source),
                profile_name=_profile_for_session,
            )
            self._session_db_created = True
        except Exception as e:
            # Transient failure (e.g. SQLite lock). Keep _session_db alive —
            # _session_db_created stays False so next run_conversation() retries.
            logger.warning(
                "Session DB creation failed (will retry next turn): %s", e
            )

    def _transition_context_engine_session(
        self,
        *,
        old_session_id: Optional[str] = None,
        new_session_id: Optional[str] = None,
        previous_messages: Optional[list] = None,
        carry_over_context: bool = False,
        reset_engine: bool = True,
        **extra_context,
    ) -> None:
        """Notify the active context engine about a host session transition.

        Generic host-side lifecycle helper. The built-in compressor keeps its
        existing reset behavior; plugin engines that implement richer hooks
        (``on_session_end``, ``on_session_reset``, ``on_session_start``,
        ``carry_over_new_session_context``) can flush old-session state,
        reset runtime counters, bind to the new session, and optionally
        carry retained context forward.
        """
        engine = getattr(self, "context_compressor", None)
        if not engine:
            return

        if old_session_id and previous_messages is not None and hasattr(engine, "on_session_end"):
            try:
                engine.on_session_end(old_session_id, previous_messages)
            except Exception as exc:
                logger.debug("context engine on_session_end during transition: %s", exc)

        if reset_engine and hasattr(engine, "on_session_reset"):
            try:
                engine.on_session_reset()
            except Exception as exc:
                logger.debug("context engine on_session_reset during transition: %s", exc)

        should_start = bool(
            old_session_id
            or previous_messages is not None
            or carry_over_context
            or extra_context
        )
        target_session_id = new_session_id or getattr(self, "session_id", "") or ""
        if should_start and target_session_id and hasattr(engine, "on_session_start"):
            start_context = {
                "old_session_id": old_session_id,
                "carry_over_context": carry_over_context,
                "platform": _session_source_for_agent(getattr(self, "platform", None)),
                "model": getattr(self, "model", ""),
                "context_length": getattr(engine, "context_length", None),
                "conversation_id": getattr(self, "_gateway_session_key", None),
            }
            start_context.update(extra_context)
            start_context = {k: v for k, v in start_context.items() if v not in (None, "")}
            try:
                engine.on_session_start(target_session_id, **start_context)
            except Exception as exc:
                logger.debug("context engine on_session_start during transition: %s", exc)

        if (
            carry_over_context
            and old_session_id
            and target_session_id
            and hasattr(engine, "carry_over_new_session_context")
        ):
            try:
                engine.carry_over_new_session_context(old_session_id, target_session_id)
            except Exception as exc:
                logger.debug("context engine carry_over_new_session_context during transition: %s", exc)

    def reset_session_state(
        self,
        previous_messages: Optional[list] = None,
        old_session_id: Optional[str] = None,
        carry_over_context: bool = False,
    ):
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

        When ``previous_messages`` / ``old_session_id`` / ``carry_over_context``
        are provided, the active context engine is notified through the
        full transition lifecycle (``_transition_context_engine_session``)
        instead of a bare reset. Default callers pass nothing and keep the
        existing reset-only behavior.
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

        # Session boundary: the usage anchor describes the OLD session's
        # transcript — a fresh/branched/resumed session must fall back to
        # full estimation until its first provider response re-anchors.
        self._usage_anchor = None
        
        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0

        # Copilot x-initiator: True for the first API call of a user turn,
        # False for tool-loop follow-ups (#3040).
        self._is_user_initiated_turn = False

        # Context engine reset/transition (works for built-in compressor and plugins)
        self._transition_context_engine_session(
            old_session_id=old_session_id,
            new_session_id=getattr(self, "session_id", None),
            previous_messages=previous_messages,
            carry_over_context=carry_over_context,
            reset_engine=True,
        )

        # Reset-only session switches (/new, /resume, /branch) update
        # agent.session_id before calling reset_session_state(). The built-in
        # compressor keeps durable cooldown state keyed by its bound session,
        # so rebind it when the active session changed but no full start hook ran.
        engine = getattr(self, "context_compressor", None)
        target_session_id = getattr(self, "session_id", "") or ""
        bound_session_id = getattr(engine, "_session_id", "") if engine is not None else ""
        if (
            engine is not None
            and hasattr(engine, "bind_session_state")
            and target_session_id
            and target_session_id != bound_session_id
        ):
            try:
                engine.bind_session_state(getattr(self, "_session_db", None), target_session_id)
            except Exception as exc:
                logger.debug("context engine bind_session_state during reset: %s", exc)

    @staticmethod
    def _effective_lmstudio_context_length(
        config_context_length: Optional[int],
        runtime_context_length: Any,
    ) -> Optional[int]:
        """Return a safe context budget from explicit intent and verified runtime."""
        explicit = (
            config_context_length
            if isinstance(config_context_length, int)
            and not isinstance(config_context_length, bool)
            and config_context_length > 0
            else None
        )
        runtime_value = getattr(runtime_context_length, "context_length", runtime_context_length)
        runtime = (
            runtime_value
            if isinstance(runtime_value, int)
            and not isinstance(runtime_value, bool)
            and runtime_value > 0
            else None
        )
        if bool(getattr(runtime_context_length, "rejected", False)) or (
            bool(getattr(runtime_context_length, "load_attempted", False))
            and runtime is None
        ):
            return None
        if runtime is not None and explicit is not None:
            return min(runtime, explicit)
        return runtime if runtime is not None else explicit

    @staticmethod
    def _lmstudio_load_was_unverified(load_result: Any) -> bool:
        """Return true when a management load was rejected or unverifiable."""
        return bool(getattr(load_result, "rejected", False)) or (
            bool(getattr(load_result, "load_attempted", False))
            and getattr(load_result, "context_length", None) is None
        )

    def _ensure_lmstudio_runtime_loaded(
        self,
        config_context_length: Optional[int] = None,
    ) -> Any:
        """Preload LM Studio unless configured to rely on JIT loading."""
        if (self.provider or "").strip().lower() != "lmstudio":
            return None
        if (getattr(self, "lmstudio_load_mode", "explicit") or "explicit").strip().lower() == "jit":
            logger.debug("LM Studio explicit preload skipped: lmstudio_load_mode=jit")
            return None

        from hermes_cli.models import ensure_lmstudio_model_loaded

        if config_context_length is None:
            config_context_length = getattr(self, "_config_context_length", None)
        return ensure_lmstudio_model_loaded(
            self.model,
            self.base_url,
            getattr(self, "api_key", ""),
            config_context_length,
            return_load_result=True,
        )

    def switch_model(
        self,
        new_model,
        new_provider,
        api_key='',
        base_url='',
        api_mode='',
        capabilities=None,
    ):
        """Forwarder — see ``agent.agent_runtime_helpers.switch_model``."""
        from agent.agent_runtime_helpers import switch_model
        return switch_model(
            self,
            new_model,
            new_provider,
            api_key,
            base_url,
            api_mode,
            capabilities,
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

        ``suppress_status_output`` (the strict machine-readable mode used by
        ``hermes chat -Q``) always wins: those flows neutralize the rendering
        callbacks, and without this gate the "no callback owns rendering"
        fallback would print ``[tool]``/``[done]`` spinner lines into the
        captured stdout it exists to keep clean (#93220).
        """
        if getattr(self, "suppress_status_output", False):
            return False
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

    def _warn_context_overflow_blocked(
        self, reason: str, preflight_tokens: int, threshold_tokens: int
    ) -> None:
        """Surface a deduped warning when the context is over the compression
        threshold but compression is blocked (summary-LLM cooldown or
        anti-thrashing).

        Without this signal the session keeps growing until the model silently
        stops answering — the conversation hits the hard provider token limit
        with no explanation. Centralised here so every caller that checks
        ``should_compress_info`` (turn-context preflight, conversation-loop
        guards) shares identical dedup/reset logic.

        Dedup is on the *kind* of block (``cooldown`` / ``ineffective``), not the
        exact countdown string, so a cooldown ticking down 30→29→… doesn't
        re-fire the warning every turn. The dedup key is cleared when the block
        clears (see ``_clear_context_overflow_warn``), so the warning can fire
        again on the next blocked-over-threshold turn.
        """
        _warn_kind = (reason or "unknown").split(":", 1)[0]
        _warn_key = ("ctx_overflow_blocked", _warn_kind)
        if getattr(self, "_last_ctx_overflow_warn", None) != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            from agent.conversation_compression import (
                CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE,
            )
            # cooldown + anti-thrash (ineffective) are both "compression blocked".
            if _warn_kind in ("cooldown", "ineffective"):
                self._touch_activity(
                    f"compression blocked ({reason})",
                    provenance=ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
                )
            self._emit_warning(
                CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
                    tokens=preflight_tokens,
                    threshold=threshold_tokens,
                    reason=reason,
                )
            )

    def _warn_uncompressed_context_overflow(
        self, preflight_tokens: int, context_length: int
    ) -> None:
        """Surface a deduped warning when uncompressed context exceeds model limit.

        When compression is explicitly disabled (compression.enabled: false), long
        sessions can grow past the model context window with no compression to shrink
        them (#89297). Surface an actionable warning so the user knows to run /compact
        or enable compression.
        """
        _warn_key = ("uncompressed_ctx_overflow", context_length)
        if getattr(self, "_last_ctx_overflow_warn", None) != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            self._emit_warning(
                f"⚠️ Session context (~{preflight_tokens:,} tokens) exceeds the model "
                f"context window (~{context_length:,} tokens) with compression disabled "
                f"(compression.enabled: false). Use /compact to compress history or "
                f"enable compression in config.yaml."
            )

    def _clear_context_overflow_warn(self) -> None:
        """Reset the dedup state for the blocked-overflow warning.

        Call this whenever compression is no longer blocked while the context
        is over threshold (e.g. the cooldown elapsed, or compression ran
        successfully), so the warning can re-fire on the next blocked turn.
        """
        self._last_ctx_overflow_warn = None

    def _emit_notice(self, notice) -> None:
        """Fire a structured ``AgentNotice`` to the active driver (TUI / CLI).

        Driver-agnostic: the bound ``notice_callback`` renders it however that
        driver does (TUI status-bar override, CLI console line). Swallows all
        callback errors — a notice must NEVER break the agent loop (D-D fail-open).
        """
        if self.notice_callback:
            try:
                self.notice_callback(notice)
            except Exception:
                logger.debug("notice_callback error in _emit_notice", exc_info=True)

    def _emit_notice_clear(self, key: str) -> None:
        """Clear a previously-fired sticky notice by ``key`` (e.g. on recovery)."""
        if self.notice_clear_callback:
            try:
                self.notice_clear_callback(key)
            except Exception:
                logger.debug("notice_clear_callback error in _emit_notice_clear", exc_info=True)

    def _emit_wait_notice(self, text: str) -> None:
        """Surface a live wait-state explanation on every driver.

        Long provider waits (slow/overloaded backend, no first byte, reasoning
        model thinking for minutes) used to leave the user staring at a generic
        "cogitating..." spinner with no hint of what the agent was waiting on.
        This helper rewrites the live status line with an explanation:

        - CLI: ``thinking_callback`` updates the prompt_toolkit spinner text.
        - TUI / Desktop: the same callback is bridged to the ``thinking.delta``
          event, which both render as the live spinner/status line.
        - Gateway: ``_touch_activity`` stores the text as the activity
          description, which the "⏳ Working — N min" heartbeat includes.

        Never raises — a wait notice must not break the API-call wait loop.
        """
        self._touch_activity(text)
        _thinking_cb = getattr(self, "thinking_callback", None)
        if _thinking_cb:
            try:
                _thinking_cb(text)
            except Exception:
                logger.debug("thinking_callback error in _emit_wait_notice", exc_info=True)

    # ── Buffered retry/fallback status ────────────────────────────────────
    # Retry and fallback chains were flooding the CLI/gateway with status
    # noise that users found confusing: a single transient 429 could produce
    # 10+ "Provider/Endpoint/Retrying in 5s..." lines before the request
    # eventually succeeded.  The buffered helpers below capture these
    # status messages instead of emitting them immediately.  They are
    # flushed (shown to the user) ONLY when every retry and fallback has
    # been exhausted; on success they are silently dropped.  Backend logs
    # (agent.log) are unaffected — every individual emission site still
    # writes to ``logger.warning`` / ``logger.info`` for diagnosis.

    def _buffer_status(self, message: str) -> None:
        """Buffer a retry/fallback status message.

        Stored as a (kind, text) tuple where ``kind`` is one of:
        - ``"status"``  -> replays via ``_emit_status``
        - ``"vprint"``  -> replays via ``_vprint(force=True)``
        - ``"warn"``    -> replays via ``_emit_warning``
        Used to defer noisy retry chatter until we know whether the
        turn ultimately recovered or failed.
        """
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf is None:
                buf = []
                self._retry_status_buffer = buf
            buf.append(("status", message))
        except Exception:
            # Never break the retry loop on a buffer hiccup.
            pass

    def _buffer_vprint(self, message: str) -> None:
        """Buffer a vprint(force=True) retry/fallback line."""
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf is None:
                buf = []
                self._retry_status_buffer = buf
            buf.append(("vprint", message))
        except Exception:
            pass

    def _clear_status_buffer(self) -> None:
        """Drop buffered retry messages — call on successful recovery."""
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf:
                buf.clear()
        except Exception:
            pass

    def _emit_pending_fallback_notice(self) -> None:
        """Surface the one-shot fallback-switch notice on successful recovery.

        A provider/model switch is a durable state change operators must see,
        unlike transient retry chatter that ``_clear_status_buffer`` drops.
        ``try_activate_fallback`` records the switch in
        ``self._pending_fallback_notice``; this emits it exactly once via
        ``_emit_status`` and then clears it, so a successful fallback still
        produces one visible notice.  On terminal failure the buffered switch
        line is flushed instead (and this notice discarded) — see
        ``_flush_status_buffer`` — so the user always sees the switch once.
        """
        try:
            notice = getattr(self, "_pending_fallback_notice", None)
            if notice:
                # Clear before emitting so a (swallowed) callback error can't
                # leave the notice set for a stale re-emit on a later turn.
                self._pending_fallback_notice = None
                notices = notice if isinstance(notice, list) else [notice]
                for item in notices:
                    try:
                        self._emit_status(str(item))
                    except Exception:
                        # A single surface callback failure must not hide later
                        # switches from the same fallback chain.
                        continue
        except Exception:
            # Never break the conversation loop on a notice hiccup.
            pass

    def _flush_status_buffer(self) -> None:
        """Emit buffered retry messages — call on terminal failure.

        Surfaces the full retry/fallback trace so the user can see what
        was tried before the turn gave up.
        """
        try:
            # The buffered trace already carries the fallback switch line, so
            # drop any one-shot fallback notice to avoid a stale duplicate
            # leaking into a later successful turn.
            self._pending_fallback_notice = None
            buf = getattr(self, "_retry_status_buffer", None)
            if not buf:
                return
            # Drain first so a callback exception doesn't double-emit.
            messages = list(buf)
            buf.clear()
            for kind, msg in messages:
                try:
                    if kind == "status":
                        self._emit_status(msg)
                    elif kind == "warn":
                        self._emit_warning(msg)
                    else:
                        self._vprint(f"{self.log_prefix}{msg}", force=True)
                except Exception:
                    pass
        except Exception:
            pass

    def _disable_codex_reasoning_replay(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """Disable Responses encrypted reasoning replay and strip cached state.

        Called from the conversation_loop retry path when the provider
        rejects a replayed ``codex_reasoning_items`` blob with HTTP 400
        ``invalid_encrypted_content``.  Sets ``self._codex_reasoning_replay_enabled``
        to ``False`` (consumed by ``codex_responses_adapter._chat_messages_to_responses_input``
        and ``transports/codex.py`` to drop ``reasoning.encrypted_content``
        from subsequent requests) and pops ``codex_reasoning_items`` from
        every assistant message in ``messages`` so they cannot be replayed
        again later in the session.

        Returns a small stats dict ``{"messages": int, "items": int}``
        counting what was stripped — purely for diagnostic logging.
        """
        stripped_messages = 0
        stripped_items = 0
        target_messages = messages if isinstance(messages, list) else []

        for msg in target_messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            items = msg.pop("codex_reasoning_items", None)
            if isinstance(items, list) and items:
                stripped_messages += 1
                stripped_items += len(items)

        self._codex_reasoning_replay_enabled = False
        return {"messages": stripped_messages, "items": stripped_items}

    # Stream-diagnostic class header preserved for backward compat —
    # actual list lives in ``agent.stream_diag.STREAM_DIAG_HEADERS``.
    from agent.stream_diag import STREAM_DIAG_HEADERS as _STREAM_DIAG_HEADERS  # noqa: E402

    @staticmethod
    def _stream_diag_init() -> Dict[str, Any]:
        """Forwarder — see ``agent.stream_diag.stream_diag_init``."""
        from agent.stream_diag import stream_diag_init
        return stream_diag_init()

    def _stream_diag_capture_response(
        self, diag: Dict[str, Any], http_response: Any
    ) -> None:
        """Forwarder — see ``agent.stream_diag.stream_diag_capture_response``."""
        from agent.stream_diag import stream_diag_capture_response
        stream_diag_capture_response(self, diag, http_response)

    @staticmethod
    def _flatten_exception_chain(error: BaseException) -> str:
        """Forwarder — see ``agent.stream_diag.flatten_exception_chain``."""
        from agent.stream_diag import flatten_exception_chain
        return flatten_exception_chain(error)

    def _is_provider_stream_parse_error(self, error: BaseException) -> bool:
        """Return True for malformed provider streaming data from SDK parsers.

        Some Anthropic-compatible streaming providers can send a malformed
        event-stream frame.  The Anthropic SDK surfaces that as a plain
        ``ValueError`` such as ``expected ident at line 1 column 149``.  That
        is provider wire-format trouble, not local request validation, so it
        should follow the same retry path as a truncated JSON body.
        """
        if getattr(self, "api_mode", None) != "anthropic_messages":
            return False
        if not isinstance(error, ValueError):
            return False
        if isinstance(error, (UnicodeEncodeError, json.JSONDecodeError)):
            return False
        message = str(error).strip().lower()
        return "expected ident at line" in message

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
        """Forwarder — see ``agent.stream_diag.log_stream_retry``."""
        from agent.stream_diag import log_stream_retry
        log_stream_retry(
            self, kind=kind, error=error, attempt=attempt,
            max_attempts=max_attempts, mid_tool_call=mid_tool_call, diag=diag,
        )

    def _emit_stream_drop(
        self,
        *,
        error: BaseException,
        attempt: int,
        max_attempts: int,
        mid_tool_call: bool,
        diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Forwarder — see ``agent.stream_diag.emit_stream_drop``."""
        from agent.stream_diag import emit_stream_drop
        emit_stream_drop(
            self, error=error, attempt=attempt, max_attempts=max_attempts,
            mid_tool_call=mid_tool_call, diag=diag,
        )

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
            "auth_mode": getattr(self, "auth_mode", "") or "",
        }

    def _check_compression_model_feasibility(self) -> None:
        """Forwarder — see ``agent.conversation_compression.check_compression_model_feasibility``."""
        from agent.conversation_compression import check_compression_model_feasibility
        check_compression_model_feasibility(self)

    def _replay_compression_warning(self) -> None:
        """Forwarder — see ``agent.conversation_compression.replay_compression_warning``."""
        from agent.conversation_compression import replay_compression_warning
        replay_compression_warning(self)

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
        return base_url_host_matches(url, "openai.azure.com")

    def _is_github_copilot_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets GitHub Copilot's OpenAI-compatible API."""
        if base_url is not None:
            hostname = base_url_hostname(base_url)
        else:
            hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
                getattr(self, "_base_url_lower", "")
            )
        if not hostname:
            return False
        return hostname == "api.githubcopilot.com" or hostname.endswith(".githubcopilot.com")

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
        return env_float("HERMES_API_TIMEOUT", 1800.0)

    def _resolved_api_call_stale_timeout_base(self) -> tuple[float, bool]:
        """Resolve the base non-stream stale timeout and whether it is implicit.

        Priority:
          1. ``providers.<id>.models.<model>.stale_timeout_seconds``
          2. ``providers.<id>.stale_timeout_seconds``
          3. ``HERMES_API_CALL_STALE_TIMEOUT`` env var
          4. 90.0s default (time-to-first-byte for non-streaming / Codex
             internal-streaming requests; lowered from 300s in May 2026 so
             fallback providers kick in faster when upstream providers
             stall).  The detector still scales up for large contexts in
             ``_compute_non_stream_stale_timeout``.

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

        # Reasoning-model floor: auto-mitigation for known reasoning models
        # (Nemotron 3 Ultra, OpenAI o1/o3, Anthropic Opus 4.x thinking,
        # DeepSeek R1, Qwen QwQ, xAI Grok reasoning, etc.) whose cloud
        # gateways idle-kill before the model's thinking phase ends.
        # uses_implicit_default is False here so the local-endpoint
        # short-circuit in _compute_non_stream_stale_timeout does not
        # disable stale detection for users running reasoning models on a
        # local NIM endpoint.
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        reasoning_floor = get_reasoning_stale_timeout_floor(self.model)
        if reasoning_floor is not None:
            return reasoning_floor, False

        return 90.0, True

    def _compute_non_stream_stale_timeout(self, api_payload: Any) -> float:
        """Compute the effective non-stream stale timeout for this request.

        Accepts either the full ``api_kwargs`` dict (Chat Completions or
        Responses API) or a legacy ``messages`` list.  Context-size scaling
        applies the same way to both shapes via
        :func:`agent.chat_completion_helpers.estimate_request_context_tokens`.
        """
        stale_base, uses_implicit_default = self._resolved_api_call_stale_timeout_base()
        base_url = getattr(self, "_base_url", None) or self.base_url or ""
        if uses_implicit_default and base_url and is_local_endpoint(base_url):
            return float("inf")

        from agent.chat_completion_helpers import estimate_request_context_tokens
        est_tokens = estimate_request_context_tokens(api_payload)
        if est_tokens > 100_000:
            timeout = max(stale_base, 240.0)
        elif est_tokens > 50_000:
            timeout = max(stale_base, 150.0)
        else:
            timeout = stale_base

        # Wall-clock run budget cap: when a run budget is active, an implicit
        # (floor-/default-derived) stale timeout is capped at half the
        # remaining budget (>= 60s) so a single hung provider call cannot eat
        # the whole run — e.g. deepseek-v4-pro's 600s reasoning floor inside a
        # 900s eval ceiling. NEVER raises the timeout above what it would
        # otherwise be, and an explicit user-configured stale_timeout_seconds
        # (or env var) still wins untouched.
        run_budget = getattr(self, "run_budget_seconds", None)
        if run_budget and not self._stale_timeout_is_explicit():
            started = getattr(self, "_run_budget_started_at", None)
            if started:
                remaining = float(run_budget) - (time.time() - started)
                deadline_cap = max(60.0, remaining * 0.5)
                if deadline_cap < timeout:
                    timeout = deadline_cap
        return timeout

    def _stale_timeout_is_explicit(self) -> bool:
        """True when the user explicitly configured the non-stream stale timeout.

        Explicit = provider/model ``stale_timeout_seconds`` in config.yaml or
        the ``HERMES_API_CALL_STALE_TIMEOUT`` env var. Reasoning-model floors
        and the 90s default are implicit — they yield to the wall-clock run
        budget cap; explicit user configuration never does.
        """
        if get_provider_stale_timeout(self.provider, self.model) is not None:
            return True
        return os.getenv("HERMES_API_CALL_STALE_TIMEOUT") is not None

    def _codex_silent_hang_hint(self, model: Optional[str] = None) -> Optional[str]:
        """Return an actionable hint when this request matches a known
        Codex silent-reject configuration, else ``None``.

        The ChatGPT Codex backend (``chatgpt.com/backend-api/codex``) has
        historically silently dropped certain model requests: the connection
        is accepted but no stream events are emitted and no error is raised.
        The stale-call detector ends the hang, but a generic "timed out"
        message gives the user no path forward.

        This helper substitutes an actionable hint into the stale-timeout
        warning when the request matches a known silent-reject pattern.
        Currently flagged: ``gpt-5.5`` family on the Codex backend.  See
        hermes-agent #21444 for the symptom history.  The upstream backend
        behavior has historically come and gone with ChatGPT entitlement
        changes — the heuristic stays in place as future-proofing even when
        the symptom is dormant.

        Does NOT fix the backend issue.  Only converts an opaque stale-timeout
        into actionable text so users learn the workaround in seconds rather
        than digging through logs.
        """
        if self.api_mode != "codex_responses":
            return None
        from agent.codex_responses_adapter import classify_responses_route

        if not classify_responses_route(self).is_codex_backend:
            return None
        eff_model = (model if model is not None else self.model) or ""
        model_lower = eff_model.lower()
        # Match the gpt-5.5 family — bare ``gpt-5.5``, ``gpt-5.5-codex``,
        # vendor-prefixed variants like ``openai/gpt-5.5``, and any future
        # ``gpt-5.5-*`` SKU.  Anchor at a word boundary on either side so
        # unrelated tokens like ``gpt-5.50`` do not match.
        if not re.search(r"(?:^|[/\-_])gpt-5\.5(?:$|[\-_])", model_lower):
            return None
        return (
            f"Codex backend appears to be silently rejecting {eff_model!r} "
            "on chatgpt.com/backend-api/codex (no stream events, no error). "
            "This is a known backend-side pattern that has affected ChatGPT "
            "Plus accounts intermittently. "
            "Workaround: try `gpt-5.4` on the same OAuth profile, or `gpt-5.3-codex`, "
            "or switch to a different model/provider in your fallback chain. "
            "Some ChatGPT Codex accounts do not support `gpt-5.4-codex`. "
            "See hermes-agent#21444 for symptom history."
        )

    def _is_openrouter_url(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return base_url_host_matches(self._base_url_lower, "openrouter.ai")

    def _is_copilot_url(self) -> bool:
        """Return True when the base URL targets GitHub Copilot or GitHub Models."""
        return (
            base_url_host_matches(self._base_url_lower, "api.githubcopilot.com")
            or base_url_host_matches(self._base_url_lower, "models.github.ai")
        )

    def _is_copilot_provider(self) -> bool:
        """True when the active provider is GitHub Copilot, however spelled.

        ``self.provider`` is not always the normalized slug: ``/model`` and
        profile configs can leave the alias ``github-copilot`` (or ``github``)
        in place — a single session log can show both ``provider=copilot`` and
        ``provider=github-copilot`` for the same account. A bare
        ``provider == "copilot"`` gate silently skips credential recovery for
        the alias spellings, so this is the single owner of the check; the
        Copilot base URL is accepted as a fallback signal.
        """
        if (self.provider or "").strip().lower() in {"copilot", "github-copilot", "github"}:
            return True
        return self._is_copilot_url()

    def _is_codex_backend(self) -> bool:
        """Return True for the ChatGPT OAuth Codex Responses backend."""
        return (
            getattr(self, "api_mode", None) == "codex_responses"
            and getattr(self, "_base_url_hostname", "") == "chatgpt.com"
            and "/backend-api/codex"
            in (getattr(self, "_base_url_lower", "") or "")
        )

    def _anthropic_prompt_cache_policy(
        self,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[bool, bool]:
        """Forwarder — see ``agent.agent_runtime_helpers.anthropic_prompt_cache_policy``."""
        from agent.agent_runtime_helpers import anthropic_prompt_cache_policy
        return anthropic_prompt_cache_policy(self, provider=provider, base_url=base_url, api_mode=api_mode, model=model)

    def _direct_native_anthropic_tool_cache_capability(
        self,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bool:
        """Forwarder for the request-local native Anthropic tool capability."""
        from agent.agent_runtime_helpers import _direct_native_anthropic_tool_cache_capability
        return _direct_native_anthropic_tool_cache_capability(
            self,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
            model=model,
        )

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
        if normalized_provider == "custom":
            # Generic custom endpoints are conservative by default. They may
            # relay GPT-5 models without full Responses semantics, so only
            # direct OpenAI/xAI URL detection should auto-upgrade them.
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

        OpenAI's newer models (gpt-4o, gpt-4.1, gpt-5+, o-series) require
        'max_completion_tokens'. Azure OpenAI and GitHub Copilot also require
        'max_completion_tokens' for those families served via their
        OpenAI-compatible endpoints. OpenRouter, local models, and older
        OpenAI models use 'max_tokens'.

        The check is URL-first (api.openai.com / Azure / Copilot all use the
        new kwarg), then falls back to a model-name check so third-party
        OpenAI-compatible endpoints fronting those models are recognised —
        URL-only detection misses that case and silently sends the wrong
        kwarg, which the upstream model rejects with a 400.
        """
        if (
            self._is_direct_openai_url()
            or self._is_azure_openai_url()
            or self._is_github_copilot_url()
            or model_forces_max_completion_tokens(self.model)
        ):
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    @staticmethod
    def _requested_output_cap_from_api_kwargs(api_kwargs: Any) -> Optional[int]:
        """Extract the outgoing response token cap from a prepared request."""
        if not isinstance(api_kwargs, dict):
            return None
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            raw = api_kwargs.get(key)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

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
        """Forwarder — see ``agent.agent_runtime_helpers.strip_think_blocks``."""
        from agent.agent_runtime_helpers import strip_think_blocks
        return strip_think_blocks(self, content)

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
        if stripped.endswith('^'):
            return True
        last = stripped[-1]
        if last in '.!?:)"\']}。！？：）】」』》^':
            return True
        # Emoji ranges (Misc Symbols, Dingbats, Emoticons, Supplemental, etc.)
        if ord(last) >= 0x1F300:
            return True
        return False

    def _is_ollama_glm_backend(self) -> bool:
        """Detect Ollama-hosted GLM models affected by stop misreports.

        Ollama can misreport truncated output as finish_reason='stop'.
        Detection relies on explicit Ollama signatures:
        - Port 11434 (Ollama default)
        - "ollama" in the base URL (e.g. ollama.local, /ollama/ path)
        - provider explicitly set to "ollama"

        Crucially it does NOT match arbitrary local/private endpoints
        (LiteLLM/sglang/vLLM/LM Studio proxies, Tailscale boxes), which
        report finish_reason correctly and were the source of #13971's
        false-positive truncation continuations.
        """
        model_lower = (self.model or "").lower()
        provider_lower = (self.provider or "").lower()
        if "glm" not in model_lower and provider_lower != "zai":
            return False
        if "ollama" in self._base_url_lower or ":11434" in self._base_url_lower:
            return True
        return provider_lower == "ollama"

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
        require_workspace: bool = True,
    ) -> bool:
        """Forwarder — see ``agent.agent_runtime_helpers.looks_like_codex_intermediate_ack``."""
        from agent.agent_runtime_helpers import looks_like_codex_intermediate_ack
        return looks_like_codex_intermediate_ack(
            self, user_message, assistant_content, messages, require_workspace
        )

    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """Forwarder — see ``agent.agent_runtime_helpers.extract_reasoning``."""
        from agent.agent_runtime_helpers import extract_reasoning
        return extract_reasoning(self, assistant_message)

    def _cleanup_task_resources(self, task_id: str) -> None:
        """Forwarder — see ``agent.chat_completion_helpers.cleanup_task_resources``."""
        from agent.chat_completion_helpers import cleanup_task_resources
        return cleanup_task_resources(self, task_id)

    # ------------------------------------------------------------------
    # Background memory/skill review — prompts live in agent.background_review
    # ------------------------------------------------------------------
    from agent.background_review import (
        _MEMORY_REVIEW_PROMPT,
        _SKILL_REVIEW_PROMPT,
        _COMBINED_REVIEW_PROMPT,
    )

    @staticmethod
    def _summarize_background_review_actions(
        review_messages: List[Dict],
        prior_snapshot: List[Dict],
        notification_mode: str = "on",
    ) -> List[str]:
        """Forwarder — see ``agent.background_review.summarize_background_review_actions``."""
        from agent.background_review import summarize_background_review_actions
        return summarize_background_review_actions(
            review_messages,
            prior_snapshot,
            notification_mode=notification_mode,
        )

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
        focus: Optional[str] = None,
    ) -> None:
        """Spawn the background memory/skill review thread.

        Thin wrapper — the heavy lifting lives in
        ``agent.background_review.spawn_background_review_thread`` which
        returns the thread target.  ``threading.Thread`` is constructed
        here so existing tests that patch ``run_agent.threading.Thread``
        keep working.

        ``focus`` is optional user-supplied steering (from ``/refine``)
        appended to the review prompt — e.g. "save the deploy workflow as a
        skill". The automatic post-turn triggers never set it.
        """
        # A delegation subagent (``_delegate_depth > 0``) must not run the
        # automatic post-turn review. Subagents are ephemeral workers already
        # barred from writing shared MEMORY.md (``DELEGATE_BLOCKED_TOOLS``) and
        # are spawned with ``skip_memory=True``, so a review here has little to
        # persist — yet it inherits the subagent's (often premium) delegation
        # model and replays the whole conversation at premium rates, silently
        # inflating token cost (#85859). An explicit ``/refine`` (``focus`` set)
        # is a deliberate user request and still runs.
        if focus is None and getattr(self, "_delegate_depth", 0) > 0:
            return
        # Explicit off-switch for automatic post-turn forks
        # (``auxiliary.background_review.enabled: false``). Manual ``/refine``
        # still works — same contract as zeroing the nudge intervals (#87250).
        # Load the task block once here and pass it into the spawn path so
        # aux routing does not re-read config.
        task_cfg = None
        if focus is None:
            from agent.background_review import load_background_review_settings
            enabled, task_cfg = load_background_review_settings()
            if not enabled:
                return
        from agent.background_review import (
            finish_background_review_run,
            prepare_background_review_run,
            spawn_background_review_thread,
        )
        from tools.thread_context import propagate_context_to_thread

        review_run = prepare_background_review_run(self)
        if review_run is None:
            return
        try:
            target, _prompt = spawn_background_review_thread(
                self,
                messages_snapshot,
                review_memory=review_memory,
                review_skills=review_skills,
                focus=focus,
                task_cfg=task_cfg,
                review_run=review_run,
            )
            # Carry the active profile into the review thread so MEMORY.md /
            # skill review writes land in the right profile (#54937).
            t = threading.Thread(
                target=propagate_context_to_thread(target),
                daemon=True,
                name="bg-review",
            )
            t.start()
        except Exception:
            finish_background_review_run(self, review_run)
            raise

    def _build_memory_write_metadata(
        self,
        *,
        write_origin: Optional[str] = None,
        execution_context: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forwarder — see ``agent.background_review.build_memory_write_metadata``."""
        from agent.background_review import build_memory_write_metadata
        return build_memory_write_metadata(
            self,
            write_origin=write_origin,
            execution_context=execution_context,
            task_id=task_id,
            tool_call_id=tool_call_id,
        )

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message before persistence/return.

        Some call paths need an API-only user-message variant without letting
        that synthetic text leak into persisted transcripts or resumed session
        history. When an override is configured for the active turn, mutate the
        in-memory messages list in place so both persistence and returned
        history stay clean.  A paired timestamp override preserves the platform
        event time as message metadata, rather than embedding it in content.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        timestamp = getattr(self, "_persist_user_message_timestamp", None)
        platform_id = getattr(self, "_persist_user_message_platform_id", None)
        if idx is None or (
            override is None and timestamp is None and platform_id is None
        ):
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                # Text-only call paths may pass a synthetic API-facing prompt
                # and a cleaner transcript string separately. Before the API
                # call, a plain-text override must not replace native image/audio
                # blocks. A list override, however, is the original clean
                # multimodal payload (for example before a queued /model note)
                # and must replace the API-local list once the turn is final.
                # Preflight compaction can re-anchor this index at a message
                # whose content was MERGED with the compaction summary
                # (merge-summary-into-tail).  That is not an accident:
                # ``reanchor_current_turn_user_idx`` falls back to the last
                # user row precisely BECAUSE the merge rewrote the content and
                # the exact-match lookup misses.  Overwriting it with the clean
                # text would drop the summary from the continuation history the
                # next turn is built from — the same hazard the DB-write twin
                # below already refuses (see the sibling guard in
                # ``_flush_messages_to_session_db_unlocked``).
                if (
                    override is not None
                    and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                    and (
                        not isinstance(msg.get("content"), list)
                        or isinstance(override, list)
                    )
                ):
                    msg["content"] = override
                if timestamp is not None:
                    msg["timestamp"] = timestamp
                # Platform-side message id (e.g. the Discord/Telegram message
                # id) — metadata, load-bearing for restart drain-window
                # recovery dedup: it lets a recovery pass ask
                # ``has_platform_message_id`` whether an interrupted turn
                # already reached the transcript. Stamped here in addition to
                # ``build_turn_context`` so it survives the override path.
                if platform_id is not None:
                    msg["platform_message_id"] = platform_id

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite on any exit path.

        Ensures conversations are never lost, even on errors or early returns.

        Trailing empty-response scaffolding is dropped from the live list in
        place (it is ephemeral junk the real transcript should shed). The
        persist user-message *override* is NOT applied here — it is resolved
        inside ``_flush_messages_to_session_db`` and written only to the DB row,
        never mutating the live message list used by the API call (#48677 is
        thus closed for every persist caller, not just this one).
        """
        # Scaffolding removal mutates the live list (desired — ephemeral
        # retry/failure sentinels must not survive into the real transcript).
        # Close and turn-start persistence can run on separate CLI threads; the
        # marker test-and-append below must be one critical section or both can
        # observe the same unmarked dict and write duplicate durable rows.
        from agent.agent_runtime_helpers import note_turn_persisted

        persist_lock = getattr(self, "_session_persist_lock", None)

        def _persist_and_drain() -> None:
            self._drop_trailing_empty_response_scaffolding(messages)
            self._session_messages = messages
            self._save_session_log(messages)
            self._flush_messages_to_session_db(messages, conversation_history)
            # Drain async token-accounting deltas at every persist point (turn
            # finalize + error exits) so a crash after this line loses at most
            # the in-flight API call's delta. Cheap no-op when nothing queued.
            if self._session_db is not None:
                self._session_db.flush_token_counts()
            note_turn_persisted(self)

        if persist_lock is None:
            _persist_and_drain()
            return

        with persist_lock:
            _persist_and_drain()

    def _drop_trailing_empty_response_scaffolding(self, messages: List[Dict]) -> None:
        """Remove private empty-response retry/failure scaffolding from transcript tails.

        Also rewinds past any trailing tool-result / assistant(tool_calls) pair
        that the failed iteration left hanging. Without this, the tail ends at
        a raw ``tool`` message and the next user turn lands as
        ``...tool, user, user`` — a protocol-invalid sequence that most
        providers silently reject (returns empty content), causing the
        empty-retry loop to fire forever. (issue number to be backfilled once filed)
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
        """Forwarder — see ``agent.agent_runtime_helpers.repair_message_sequence``."""
        from agent.agent_runtime_helpers import repair_message_sequence
        return repair_message_sequence(self, messages)

    def _flush_messages_to_session_db(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
    ):
        """Serialize direct and turn-boundary session flushes per agent."""
        persist_lock = getattr(self, "_session_persist_lock", None)
        if persist_lock is None:
            return self._flush_messages_to_session_db_unlocked(messages, conversation_history)
        with persist_lock:
            return self._flush_messages_to_session_db_unlocked(messages, conversation_history)

    def _flush_messages_to_session_db_unlocked(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        _adoption_budget: int = 1,
    ):
        """Persist any un-flushed messages to the SQLite session store.

        Deduplicates via an intrinsic ``_DB_PERSISTED_MARKER`` stamped on each
        written message dict, so repeated calls (from multiple exit paths) only
        write truly new messages — preventing the duplicate-write bug (#860)
        without relying on positional slices that can drift after
        message-sequence repair, and without a retained ``id(msg)`` set that
        CPython could alias onto a freed-then-reused address (#50372). The
        ``_flushed_db_message_ids`` attribute is now only a one-shot seed
        (translated to markers, then cleared each flush), not a persisted set.

        Note: the marker is stamped on the live/shared conversation dict, which
        correctly makes re-persistence idempotent across turns. No code path
        edits a persisted message's content/role in place expecting a re-write
        (in-place compaction resets the seed and re-diffs by identity).
        """
        # Persistence-isolated agents (e.g. the background skill/memory review
        # fork) must NEVER write into the canonical session store. The fork
        # shares the parent's session_id for prompt-cache warmth, so any write
        # here would land its harness turn ("Review the conversation above and
        # update the skill library…") inside the user's real session history,
        # where the next live turn re-reads it as an instruction and the agent
        # "becomes" the curator. Hard-stop before any DB touch.
        if getattr(self, "_persist_disabled", False):
            return None
        if not self._session_db:
            return None
        # Persist user-message override (#48677 chokepoint): historically this
        # mutated the live `messages` list in place, which — on the early
        # crash-resilience persist that runs BEFORE the API call is built —
        # stripped observed group-chat context off the live user message and
        # silently dropped it. Instead, resolve the override here and apply it
        # ONLY to the value written to the DB (see the write loop below); the
        # live dict is never mutated, so every caller (early persist, mid-loop
        # flush, /resume, /branch) is protected uniformly. Timestamp override is
        # metadata and is likewise applied only to the written row.
        _ov_idx = getattr(self, "_persist_user_message_idx", None)
        _ov_content = getattr(self, "_persist_user_message_override", None)
        _ov_timestamp = getattr(self, "_persist_user_message_timestamp", None)
        try:
            # Retry row creation if the earlier attempt failed transiently.
            if not self._session_db_created:
                self._ensure_db_session()
            # Positional flushing used to slice at
            # max(len(conversation_history), _last_flushed_db_idx). That
            # assumes the live `messages` list is the original history plus a
            # new tail. repair_message_sequence can shrink/merge the history
            # copy before the final flush, making len(conversation_history)
            # larger than len(messages); the slice is then empty and delivered
            # assistant responses never reach state.db (#46053).
            #
            # Track persistence with an intrinsic per-message marker rather than
            # id(msg). `messages` is a shallow copy of `conversation_history`, so
            # history dicts are skipped by identity, and new dicts appended
            # during this turn are written once even if repair compacts the list
            # around them. Unlike an id()-keyed set, a marker bound to the dict
            # cannot be aliased onto a freed-then-reused address, so a real turn
            # can never be silently skipped (see _DB_PERSISTED_MARKER).
            #
            # `self._flushed_db_message_ids` is still honoured as a *one-shot*
            # seed: external callers (gateway shutdown, tests) populate it with
            # {id(m) for m in already_persisted} immediately before the flush,
            # while those objects are alive — so the ids are valid at that
            # instant. We translate the seed into durable markers and then clear
            # the set, so stale ids can never accumulate across turns and alias a
            # future message.
            current_session_id = getattr(self, "session_id", None)
            flushed_session_id = getattr(self, "_flushed_db_message_session_id", None)
            if flushed_session_id != current_session_id or self._last_flushed_db_idx == 0:
                seed_ids = set()
            else:
                seed_ids = getattr(self, "_flushed_db_message_ids", None)
                if not isinstance(seed_ids, set):
                    seed_ids = set()
            self._flushed_db_message_session_id = current_session_id
            history_ids = {
                id(item) for item in (conversation_history or [])
                if isinstance(item, dict)
            }

            # Bounded scan: skip the longest identity-matched prefix of the
            # list snapshot taken at the end of the previous successful flush.
            # Every message in that snapshot was already given its final
            # disposition (written+stamped, stamped as durable history, or
            # skipped as ephemeral scaffolding / non-dict), and no code path
            # pops _DB_PERSISTED_MARKER from a live dict in place (compression
            # strips markers on fresh copies, which breaks identity here and
            # forces a full re-scan). Identity match ⇒ identical skip decision,
            # so starting after the matched prefix is behavior-preserving.
            _scan_start = 0
            _prev_prefix = getattr(self, "_db_flush_scan_prefix", None)
            if isinstance(_prev_prefix, list):
                _limit = min(len(_prev_prefix), len(messages))
                while (
                    _scan_start < _limit
                    and messages[_scan_start] is _prev_prefix[_scan_start]
                    and bool(messages[_scan_start].get(_DB_PERSISTED_MARKER))
                ):
                    _scan_start += 1

            # Collect this flush's new rows and write them in ONE transaction
            # at the end of the scan (see append_messages_batch).
            _batch_rows: List[Dict[str, Any]] = []
            _batch_msgs: List[Dict] = []
            for _msg_idx in range(_scan_start, len(messages)):
                msg = messages[_msg_idx]
                if not isinstance(msg, dict):
                    continue
                # Never write ephemeral recovery scaffolding to the session
                # store. The flush is append-only (it only advances
                # _last_flushed_db_idx via identity tracking), so a synthetic
                # message committed by a mid-turn persist cannot be un-written
                # when the end-of-turn drop removes it from the in-memory list —
                # the resumed transcript would then replay synthetic
                # "(empty)"/nudge/thinking-prefill turns as if they were genuine
                # context. Skip regardless of position: an answered nudge leaves
                # the synthetic pair buried mid-list, not just at the tail.
                if _is_ephemeral_scaffolding(msg):
                    continue
                if msg.get(_DB_PERSISTED_MARKER):
                    continue
                # Already-durable messages: either carried over from the loaded
                # history copy, or seeded by a caller. Stamp them so future
                # flushes skip them without consulting any id() set again.
                if id(msg) in history_ids or id(msg) in seed_ids:
                    msg[_DB_PERSISTED_MARKER] = True
                    continue
                role = msg.get("role", "unknown")
                content = msg.get("content")
                # api_content sidecar: the exact bytes sent to the API when
                # they differ from the clean content (stamped by the turn
                # prologue for prefetch/plugin injections). Written verbatim
                # so replay can reproduce the sent prefix byte-for-byte.
                _row_api_content = msg.get("api_content")
                if not isinstance(_row_api_content, str):
                    _row_api_content = None
                _row_timestamp = msg.get("timestamp")
                # Apply the persist override to THIS row's written values only
                # (never to the live dict). A multimodal override is a complete
                # clean replacement for an API-local noted payload. Preserve the
                # historical text-only guard for a list payload, though: a plain
                # text override must not erase its image/audio transcript summary.
                # The close safety-net may flush a shortened snapshot while
                # turn setup still owns its staged CLI dict. In that shape the
                # normal turn index refers to the full history, not this list;
                # preserve the API-local override by recognizing the same dict.
                pending_cli_message = getattr(self, "_pending_cli_user_message", None)
                is_current_turn_user = (
                    _ov_idx == _msg_idx or msg is pending_cli_message
                )
                if is_current_turn_user and msg.get("role") == "user":
                    # Preflight compaction can re-anchor the override index at
                    # a message whose content was MERGED with the compaction
                    # summary (merge-summary-into-tail). Overwriting that with
                    # the clean gateway text would silently drop the summary
                    # from the durable transcript. The wire is already
                    # consistent — the merge popped the sidecar and the merged
                    # content is what gets sent — so keep it.
                    if (
                        _ov_content is not None
                        and (not isinstance(content, list) or isinstance(_ov_content, list))
                        and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                    ):
                        # The live content is what the API call sends; the
                        # override is the cleaned transcript value. If they
                        # differ and no injection already stamped the sidecar,
                        # keep the sent bytes in api_content so replay matches
                        # the wire (#48677 divergence, closed for the cache
                        # prefix too).
                        if (
                            _row_api_content is None
                            and isinstance(content, str)
                            and content != _ov_content
                        ):
                            _row_api_content = content
                        content = _ov_content
                    if _ov_timestamp is not None:
                        _row_timestamp = _ov_timestamp
                # Store the sidecar only when it actually differs.
                if _row_api_content == content:
                    _row_api_content = None
                # Load-time sanitize divergence: get_messages_as_conversation
                # replays user/assistant rows through
                # ``sanitize_context(content).strip()``, so content that
                # sanitize would rewrite (echoed/pasted <memory-context>
                # fences or system notes) replays different bytes after a
                # session reload even though THIS turn sent it verbatim.
                # Capture the sent bytes in the sidecar so a reloaded session
                # replays what was actually on the wire. Compared in wire form
                # (both sides .strip()-ed — the api_messages build strips
                # every outgoing content string) so plain surrounding
                # whitespace doesn't grow redundant sidecars.
                if (
                    _row_api_content is None
                    and role in ("user", "assistant")
                    and isinstance(content, str)
                    and content
                    and sanitize_context(content).strip() != content.strip()
                ):
                    _row_api_content = content
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
                _row = {
                    "role": role,
                    "content": content,
                    "tool_name": msg.get("tool_name"),
                    "tool_calls": tool_calls_data,
                    "tool_call_id": msg.get("tool_call_id"),
                    "finish_reason": msg.get("finish_reason"),
                    # Reasoning/codex fields are role-gated (assistant-only)
                    # inside _insert_message_rows — pass through untouched.
                    "reasoning": msg.get("reasoning"),
                    "reasoning_content": msg.get("reasoning_content"),
                    "reasoning_details": msg.get("reasoning_details"),
                    "codex_reasoning_items": msg.get("codex_reasoning_items"),
                    "codex_message_items": msg.get("codex_message_items"),
                    "_compressed_summary": bool(msg.get(COMPRESSED_SUMMARY_METADATA_KEY)),
                    "timestamp": _row_timestamp,
                    "api_content": _row_api_content,
                    # Standalone reference handoffs are always hidden, even
                    # when the summarized transcript contained a user turn —
                    # otherwise they occupy the active user slot in
                    # retry/undo/session dispatch (#80622). Merge-into-tail
                    # carriers keep prior visibility rules so preserved tail
                    # content stays readable.
                    "display_kind": (
                        "hidden"
                        if (
                            msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                            and user_originated_turn_view(msg) is None
                            and (
                                ContextCompressor.classify_summary_content(
                                    msg.get("content")
                                )
                                == "standalone"
                                or not msg.get(
                                    "_compressed_summary_has_user_turn"
                                )
                            )
                        )
                        else msg.get("display_kind")
                    ),
                    "display_metadata": msg.get("display_metadata"),
                    # Platform-side message id (e.g. the Discord/Telegram
                    # message id). _insert_message_rows reads it off the row
                    # dict; load-bearing for restart drain-window recovery
                    # dedup via has_platform_message_id.
                    "platform_message_id": msg.get("platform_message_id"),
                }
                if isinstance(msg.get("_row_id"), int):
                    _row["_row_id"] = msg["_row_id"]
                _batch_rows.append(_row)
                _batch_msgs.append(msg)
            # One transaction for the whole turn's new rows (typically 3-8
            # messages): one BEGIN IMMEDIATE / commit — and, off WAL, one
            # fsync — instead of one per row. All-or-nothing pairs exactly
            # with the marker stamping below: on failure NO rows landed and
            # NO markers were stamped, so the next flush re-scans and
            # re-writes the whole tail (same recovery contract as before,
            # minus the partial-prefix case that could double-pay counters).
            if _batch_rows:
                self._session_db.append_messages_batch(
                    session_id=self.session_id,
                    messages=_batch_rows,
                    compression_lock_holder=getattr(
                        self, "_active_compression_lock_holder", None
                    ),
                    turn_lease_holder=getattr(
                        self, "_active_session_turn_lease_holder", None
                    ),
                    turn_lease_ttl_seconds=getattr(
                        self, "_active_session_turn_lease_ttl_seconds", 300.0
                    )
                    or 300.0,
                )
                from agent.transcript_repair import sync_flushed_message_markers

                sync_flushed_message_markers(_batch_msgs, _batch_rows)
            # The intrinsic markers are now the sole source of truth. Reset the
            # one-shot seed so no id() outlives this flush to alias a message
            # allocated next turn at a recycled address.
            self._flushed_db_message_ids = set()
            self._last_flushed_db_idx = len(messages)
            # Snapshot for the bounded scan above — only on full success, so
            # a partially-processed list can never be treated as settled.
            self._db_flush_scan_prefix = messages[:]
            return True
        except Exception as e:
            # Force a full re-scan on the next flush: an exception mid-loop
            # leaves messages with mixed dispositions.
            self._db_flush_scan_prefix = None
            # This is the one place the underlying SQLite error is visible
            # before it is swallowed into a bare ``False`` — classify it here
            # so the turn-end explanation can distinguish lock contention
            # ("storage was busy, send it again") from disk-full/read-only.
            from hermes_state import (
                CompressionSessionClosedError,
                StateDbReplacedError,
                classify_persistence_error,
                divert_session_transcript_jsonl,
            )

            self._last_persistence_error_cause = classify_persistence_error(e)
            if isinstance(e, StateDbReplacedError):
                try:
                    divert_session_transcript_jsonl(
                        getattr(self, "session_id", "") or "",
                        _batch_rows,
                    )
                except Exception:
                    logger.warning(
                        "JSONL divert failed after state.db replace for %s",
                        getattr(self, "session_id", None),
                        exc_info=True,
                    )
            if isinstance(e, CompressionSessionClosedError):
                # Compression race: another path rotated this session while
                # this turn was still writing against it. The store resolves
                # the continuation chain transitively via the canonical API
                # ``get_compression_tip`` (bounded walk, excludes branch/
                # delegate/tool children, prefers live children over stale
                # closed siblings such as ``ws_orphan_reap``). Adopt the tip
                # ONLY when it is a different row AND still live, and retry
                # the flush exactly once (adoption budget) — a second
                # closed-parent write must fail closed, never loop. The tip
                # walk returns the input id when no continuation exists, so
                # ``tip == session_id`` means fail closed.
                if _adoption_budget > 0:
                    old_id = self.session_id
                    tip = None
                    try:
                        tip = self._session_db.get_compression_tip(old_id)
                    except Exception as tip_exc:
                        logger.warning(
                            "compression tip lookup failed for %s: %s",
                            old_id,
                            tip_exc,
                        )
                    if tip and tip != old_id:
                        tip_row = None
                        try:
                            tip_row = self._session_db.get_session(tip)
                        except Exception:
                            tip_row = None
                        if tip_row is not None and tip_row.get("ended_at") is None:
                            logger.warning(
                                "Adopted live compression tip %s for closed "
                                "session %s; retrying flush once",
                                tip,
                                old_id,
                            )
                            self.session_id = tip
                            self._flushed_db_message_ids = set()
                            self._last_flushed_db_idx = 0
                            self._compression_adoption_failed = False
                            return self._flush_messages_to_session_db_unlocked(
                                messages,
                                conversation_history,
                                _adoption_budget=0,
                            )
                # No live tip (or budget exhausted): fail closed — never guess
                # a target session. The per-turn diagnostic flag lets the
                # turn-completion explanation name compression rotation
                # instead of the historical (misleading) full-disk advice.
                self._compression_adoption_failed = True
                logger.warning("Session DB append_message failed: %s", e)
                return False
            logger.warning("Session DB append_message failed: %s", e)
            return False

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
        """Forwarder — see ``agent.system_prompt.format_tools_for_system_message``."""
        from agent.system_prompt import format_tools_for_system_message
        return format_tools_for_system_message(self)

    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """Forwarder — see ``agent.agent_runtime_helpers.convert_to_trajectory_format``."""
        from agent.agent_runtime_helpers import convert_to_trajectory_format
        return convert_to_trajectory_format(self, messages, user_query, completed)

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
    def _is_entitlement_failure(
        error_context: Optional[Dict[str, Any]],
        status_code: Optional[int],
    ) -> bool:
        """Detect subscription/entitlement 403s that masquerade as auth failures.

        Returned True only when the body text matches a known entitlement
        shape AND the status is 401/403.  Refreshing an OAuth token cannot
        fix an unsubscribed account, so callers should surface the error
        instead of looping the credential pool.

        Current matches:
          * xAI OAuth: "do not have an active Grok subscription" /
            "out of available resources" / "does not have permission" + "grok"

        Disambiguator for xAI (#29344): the same ``code`` text ("The caller
        does not have permission to execute the specified operation") is
        returned for BOTH an unsubscribed account AND a stale OAuth access
        token.  xAI ships an explicit signal in the ``error`` field that
        tells the two apart: a ``[WKE=unauthenticated:...]`` suffix (and/or
        the ``OAuth2 access token could not be validated`` phrasing) means
        the credentials failed validation — that's recoverable by refreshing
        the token, NOT by surfacing an entitlement message.  When either
        signal is present we return False eagerly so the credential-pool
        refresh path runs, letting long-running TUI sessions recover from
        stale tokens without an exit/reopen cycle.

        Extend here for new providers as we discover them (Anthropic's
        Claude Max OAuth entitlement errors look distinct enough today that
        the existing 1M-context-beta branch handles them; revisit if other
        subscription tiers start producing the same loop signature).
        """
        if status_code not in {401, 403, None}:
            return False
        if not isinstance(error_context, dict):
            return False
        # Build a single lowercase haystack covering every field shape the
        # body might land in.  ``_extract_api_error_context`` normalises to
        # ``message``/``reason``, but callers (and the test suite) may also
        # hand us the raw body with ``code``/``error`` keys; cover both so
        # the WKE disambiguator below fires regardless of entry point.
        message = str(error_context.get("message") or "").lower()
        reason = str(error_context.get("reason") or "").lower()
        code = str(error_context.get("code") or "").lower()
        err = str(error_context.get("error") or "").lower()
        haystack = f"{message} {reason} {code} {err}"
        if not haystack.strip():
            return False
        # xAI's authoritative disambiguator for "stale token" vs
        # "unsubscribed account".  Both conditions share the same
        # permission-denied ``code`` text; only one carries this suffix.
        # Bail out before the entitlement keyword checks so a stale OAuth
        # token routes through the credential-refresh path instead of the
        # surface-error-as-entitlement path.  See #29344 for the long-
        # running TUI failure mode this closes.
        if "[wke=unauthenticated:" in haystack:
            return False
        if "oauth2 access token could not be validated" in haystack:
            return False
        if "do not have an active grok subscription" in haystack:
            return True
        if "out of available resources" in haystack and "grok" in haystack:
            return True
        if "does not have permission" in haystack and "grok" in haystack:
            return True
        return False

    @staticmethod
    def _decorate_xai_entitlement_error(detail: str) -> str:
        """Append a neutral hint when xAI's OAuth surface returns the
        permission-denied 403.

        xAI's ``/v1/responses`` endpoint replies to several distinct failure
        modes with the SAME body::

            {"code": "The caller does not have permission to execute the
             specified operation", "error": "You have either run out of
             available resources or do not have an active Grok subscription.
             Manage subscriptions at https://grok.com/?_s=usage or subscribe
             at https://grok.com/supergrok"}

        That body covers several real causes we cannot distinguish without
        more info from xAI.  The most common (and least obvious) one is
        that **X Premium+ does NOT include API access** — only standalone
        SuperGrok subscribers can use Hermes against xai-oauth.  Lots of
        users see Grok in their X app, assume it works here too, and hit
        this 403 with no idea why.  Lead the hint with that.

        Other possible causes:
          * No Grok subscription at all
          * SuperGrok tier doesn't include the requested model (e.g.
            grok-4.3 may need a higher tier)
          * Monthly quota exhausted (the ``?_s=usage`` URL hints at this)

        Surface the raw xAI text verbatim and point at
        https://grok.com/?_s=usage where the user can see WHICH applies.

        Matched once per detail string — won't double-decorate if the
        upstream already concatenated the same text.
        """
        if not detail:
            return detail
        lower = detail.lower()
        is_entitlement = (
            "do not have an active grok subscription" in lower
            or ("out of available resources" in lower and "grok" in lower)
            or ("does not have permission" in lower and "grok" in lower)
        )
        if not is_entitlement:
            return detail
        hint = (
            " — xAI rejected this OAuth account. NOTE: X Premium+ does NOT "
            "include xAI API access — only standalone SuperGrok subscribers "
            "can use this provider. Other possible causes: no Grok "
            "subscription, your tier doesn't include this model, or your "
            "quota is exhausted. Check https://grok.com/?_s=usage to see "
            "which, or run `/model` to switch providers."
        )
        # Idempotency: detect prior decoration by a substring unique to the
        # hint (not present in xAI's own body text).
        if "X Premium+ does NOT include" in detail:
            return detail
        return f"{detail}{hint}"

    @staticmethod
    def _coerce_api_error_detail(value: Any) -> str:
        """Return a display-safe string for structured provider error fields."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("message", "detail", "error", "code", "type"):
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    return nested
            for key in ("message", "detail", "error", "code", "type"):
                if key in value:
                    nested_detail = AIAgent._coerce_api_error_detail(value[key])
                    if nested_detail:
                        return nested_detail
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(value)
        if isinstance(value, (list, tuple)):
            parts = [
                AIAgent._coerce_api_error_detail(item)
                for item in value
            ]
            return "; ".join(part for part in parts if part)
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _summarize_api_error(error: Exception) -> str:
        """Extract a human-readable one-liner from an API error.

        Handles Cloudflare HTML error pages (502, 503, etc.) by pulling the
        <title> tag instead of dumping raw HTML. Network/DNS failures are
        translated into an offline hint, including when an SDK wraps the
        original OS error. Falls back to a truncated str(error) otherwise.
        """
        raw = str(error)

        # Linux, macOS, and Windows use different low-level messages when DNS
        # cannot resolve the provider while the device is offline. SDKs often
        # wrap that OSError in a generic "Connection error", so inspect the
        # exception chain before showing the top-level message to the user.
        network_resolution_markers = (
            "temporary failure in name resolution",
            "name or service not known",
            "nodename nor servname provided, or not known",
            "getaddrinfo failed",
            "no address associated with hostname",
            "network is unreachable",
        )
        current: Optional[BaseException] = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if any(
                marker in str(current).lower()
                for marker in network_resolution_markers
            ):
                return (
                    "Hermes can't reach the model provider. You may be offline. "
                    "Check your internet connection and try again."
                )
            current = current.__cause__ or current.__context__

        if (
            isinstance(error, ValueError)
            and "expected ident at line" in raw.lower()
        ):
            return f"Malformed provider streaming response: {raw[:300]}"

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

        # GeminiAPIError (agent/gemini_native_adapter.py) already composes a
        # clean one-liner and may have appended actionable guidance (free-tier
        # 429, legacy Standard-key 401). Prefer its message over re-extracting
        # the raw response body below, which would strip that guidance.
        if type(error).__name__ == "GeminiAPIError":
            return redact_sensitive_text(raw[:1000])

        # JSON body errors from OpenAI/Anthropic SDKs
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                msg = AIAgent._coerce_api_error_detail(msg)
                return AIAgent._decorate_xai_entitlement_error(f"{prefix}{msg[:300]}")

        # SDK may leave body empty while httpx still has the payload (#36109).
        # Redact before returning: the raw provider/proxy error body is
        # attacker-influenced and may echo Authorization / x-api-key / request
        # JSON, which would otherwise leak into final_response + logs (this path
        # widens exposure vs the old empty-body "HTTP 400" string).
        response = getattr(error, "response", None)
        if response is not None:
            try:
                snippet = (getattr(response, "text", None) or "").strip()
            except Exception:
                snippet = ""
            if snippet:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                try:
                    payload = json.loads(snippet)
                except (json.JSONDecodeError, TypeError):
                    payload = None
                if isinstance(payload, dict):
                    err = payload.get("error")
                    if isinstance(err, dict) and err.get("message"):
                        return redact_sensitive_text(f"{prefix}{str(err['message'])[:300]}")
                    if payload.get("message"):
                        return redact_sensitive_text(f"{prefix}{str(payload['message'])[:300]}")
                return redact_sensitive_text(f"{prefix}{snippet[:300]}")

        # Fallback: truncate the raw string but give more room than 200 chars
        status_code = getattr(error, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return AIAgent._decorate_xai_entitlement_error(f"{prefix}{raw[:500]}")

    def _mask_api_key_for_logs(self, key: Any) -> Optional[str]:
        # Azure Foundry Entra ID bearer providers are callables — never
        # invoke them in log paths; identify the auth surface instead.
        if callable(key) and not isinstance(key, str):
            return "<entra-id-bearer>"
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
        """Forwarder — see ``agent.agent_runtime_helpers.extract_api_error_context``."""
        from agent.agent_runtime_helpers import extract_api_error_context
        return extract_api_error_context(error)

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

    @staticmethod
    def _hook_payload_max_chars() -> int:
        raw = os.getenv("HERMES_PLUGIN_PAYLOAD_MAX_CHARS", "50000")
        try:
            return max(1000, int(raw))
        except (TypeError, ValueError):
            return 50000

    @staticmethod
    def _is_sensitive_hook_key(key: Any) -> bool:
        if not isinstance(key, str):
            return False
        lowered = key.lower().replace("-", "_")
        exact = {
            "api_key",
            "authorization",
            "proxy_authorization",
            "cookie",
            "set_cookie",
        }
        return lowered in exact or lowered.endswith("_api_key")

    @classmethod
    def _hook_jsonable(
        cls,
        value: Any,
        *,
        depth: int = 0,
        max_depth: int = 8,
        max_string: int = 8000,
        max_sequence: int = 200,
    ) -> Any:
        if depth > max_depth:
            return f"<{type(value).__name__} depth limit>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if len(value) > max_string:
                return value[:max_string] + f"...[truncated {len(value) - max_string} chars]"
            return value
        if isinstance(value, (bytes, bytearray)):
            return f"<{len(value)} bytes>"
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for idx, (key, item) in enumerate(value.items()):
                if idx >= max_sequence:
                    out["_truncated_items"] = len(value) - max_sequence
                    break
                str_key = str(key)
                if cls._is_sensitive_hook_key(str_key):
                    out[str_key] = "<redacted>"
                else:
                    out[str_key] = cls._hook_jsonable(
                        item,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_string=max_string,
                        max_sequence=max_sequence,
                    )
            return out
        if isinstance(value, (list, tuple, set)):
            seq = list(value)
            out = [
                cls._hook_jsonable(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_string=max_string,
                    max_sequence=max_sequence,
                )
                for item in seq[:max_sequence]
            ]
            if len(seq) > max_sequence:
                out.append({"_truncated_items": len(seq) - max_sequence})
            return out
        try:
            if hasattr(value, "model_dump"):
                try:
                    # warnings=False: pydantic's serializer UserWarnings on
                    # generic-union SDK models (Anthropic ParsedMessage etc.)
                    # would otherwise leak to the terminal mid-response.
                    dumped = value.model_dump(mode="json", warnings=False)
                except TypeError:
                    try:
                        dumped = value.model_dump(mode="json")
                    except TypeError:
                        dumped = value.model_dump()
                return cls._hook_jsonable(
                    dumped,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_string=max_string,
                    max_sequence=max_sequence,
                )
        except Exception:
            pass
        try:
            from dataclasses import asdict, is_dataclass
            if is_dataclass(value):
                return cls._hook_jsonable(
                    asdict(value),
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_string=max_string,
                    max_sequence=max_sequence,
                )
        except Exception:
            pass
        if isinstance(value, SimpleNamespace):
            return cls._hook_jsonable(
                vars(value),
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
        if hasattr(value, "__dict__"):
            try:
                public_attrs = {
                    k: v
                    for k, v in vars(value).items()
                    if not str(k).startswith("_")
                }
                return cls._hook_jsonable(
                    public_attrs,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_string=max_string,
                    max_sequence=max_sequence,
                )
            except Exception:
                pass
        return str(value)[:max_string]

    @classmethod
    def _sanitize_hook_payload(cls, value: Any) -> Any:
        payload = cls._hook_jsonable(value)
        limit = cls._hook_payload_max_chars()
        try:
            encoded = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return str(payload)[:limit]
        if len(encoded) <= limit:
            return payload
        payload = cls._hook_jsonable(value, max_string=1000, max_sequence=50)
        try:
            encoded = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return str(payload)[:limit]
        if len(encoded) <= limit:
            return payload
        return {
            "_truncated": True,
            "original_type": type(value).__name__,
            "preview": encoded[:limit],
        }

    def _api_request_payload_for_hook(self, api_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        body = {
            key: value
            for key, value in (api_kwargs or {}).items()
            if key not in {"timeout", "http_client"}
        }
        return self._sanitize_hook_payload(
            {
                "method": "POST",
                "body": body,
            }
        )

    def _api_response_payload_for_hook(
        self,
        response: Any,
        assistant_message: Any,
        *,
        finish_reason: Optional[str],
    ) -> Dict[str, Any]:
        # ``tool_calls`` is the raw list of provider SDK objects (e.g.
        # OpenAI ``ChatCompletionMessageToolCall``).  We deliberately hand
        # the raw objects to ``_sanitize_hook_payload`` and rely on
        # ``_hook_jsonable`` to normalise them via ``model_dump`` /
        # ``__dict__`` / dataclass introspection — a future refactor of
        # the sanitiser MUST preserve that capability or hook subscribers
        # will receive opaque ``str(obj)`` blobs here.
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        return self._sanitize_hook_payload(
            {
                "model": getattr(response, "model", None),
                "finish_reason": finish_reason,
                "assistant_message": {
                    "role": getattr(assistant_message, "role", "assistant"),
                    "content": getattr(assistant_message, "content", None),
                    "tool_calls": tool_calls,
                },
                "usage": self._usage_summary_for_api_request_hook(response),
            }
        )

    def _invoke_api_request_error_hook(
        self,
        *,
        task_id: str,
        turn_id: str,
        api_request_id: str,
        api_call_count: int,
        api_start_time: float,
        api_kwargs: Optional[Dict[str, Any]],
        error_type: str,
        error_message: str,
        status_code: Optional[int] = None,
        retry_count: Optional[int] = None,
        max_retries: Optional[int] = None,
        retryable: Optional[bool] = None,
        reason: Optional[str] = None,
    ) -> None:
        # Lazy module import (not from-import) so tests can replace lifecycle
        # dispatch at this call site. After first call the import is a
        # ``sys.modules`` dict lookup, so retries don't repay any real cost.
        try:
            from hermes_cli import lifecycle as _lifecycle

            if not _lifecycle.has_hook("api_request_error"):
                return
            ended_at = time.time()
            _lifecycle.invoke_hook(
                "api_request_error",
                task_id=task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                session_id=self.session_id or "",
                platform=self.platform or "",
                model=self.model,
                provider=self.provider,
                base_url=self.base_url,
                api_mode=self.api_mode,
                api_call_count=api_call_count,
                api_duration=ended_at - api_start_time,
                started_at=api_start_time,
                ended_at=ended_at,
                status_code=status_code,
                retry_count=retry_count,
                max_retries=max_retries,
                retryable=retryable,
                reason=reason,
                error={
                    "type": error_type,
                    "message": error_message,
                },
                request=self._api_request_payload_for_hook(api_kwargs),
            )
        except Exception:
            pass

    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """Forwarder — see ``agent.agent_runtime_helpers.dump_api_request_debug``."""
        from agent.agent_runtime_helpers import dump_api_request_debug
        return dump_api_request_debug(self, api_kwargs, reason=reason, error=error)

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    @staticmethod
    def _redact_message_content(content):
        """Apply secret redaction to message content (str or list-of-parts).

        Handles both plain-string content and the OpenAI/Anthropic multimodal
        shape where ``content`` is a list of ``{"type": "text", "text": ...}``
        / ``{"type": "image_url", ...}`` / ``{"type": "input_text", "content": ...}``
        parts. Image / binary parts are left untouched; only text fields are
        passed through ``redact_sensitive_text``.

        Respects ``HERMES_REDACT_SECRETS`` via ``redact_sensitive_text`` —
        when disabled the helper is effectively a no-op.
        """
        if content is None:
            return content
        if isinstance(content, str):
            return redact_sensitive_text(content)
        if isinstance(content, list):
            redacted = []
            for part in content:
                if isinstance(part, dict):
                    part = dict(part)
                    if isinstance(part.get("text"), str):
                        part["text"] = redact_sensitive_text(part["text"])
                    if isinstance(part.get("content"), str):
                        part["content"] = redact_sensitive_text(part["content"])
                redacted.append(part)
            return redacted
        return content

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """Optional per-session JSON snapshot writer.

        Gated by ``sessions.write_json_snapshots`` (default False).  state.db
        is the canonical message store; this writer exists only for users
        whose external tooling consumes ``~/.hermes/sessions/session_{sid}.json``
        directly.  When the flag is off this is a fast no-op.

        When enabled, rewrites the snapshot after every persistence point with
        the full message list (assistant content normalized via
        ``_clean_session_content`` to convert REASONING_SCRATCHPAD to think
        tags).  The truncation guard ("don't overwrite a larger log with
        fewer messages") is preserved so resume + branch don't clobber a
        fuller existing snapshot.
        """
        if not getattr(self, "_session_json_enabled", False):
            return
        messages = messages or self._session_messages
        if not messages:
            return

        # Re-derive the target path each call so /branch and /compress
        # session-id changes land in the right file without any re-point
        # bookkeeping at the call sites.  Sanitize the session ID into a
        # single traversal-free path segment — session IDs can come from
        # untrusted input (X-Hermes-Session-Id header) and must not escape
        # the sessions directory.
        try:
            safe_sid = _safe_session_filename_component(self.session_id)
            log_file = self.logs_dir / f"session_{safe_sid}.json"
        except Exception:
            return

        try:
            cleaned = []
            for msg in messages:
                # Mirror the SQLite flush: ephemeral recovery scaffolding is
                # internal retry state, never durable transcript content.
                if _is_ephemeral_scaffolding(msg):
                    continue
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                # Defence-in-depth: redact credentials from every message
                # content before persistence. Catches PATs / API keys / Bearer
                # tokens that may have leaked into assistant responses, tool
                # output, or user paste. Respects HERMES_REDACT_SECRETS via
                # redact_sensitive_text — no-op when disabled. (#19798, #19845)
                if "content" in msg:
                    msg = dict(msg)
                    msg["content"] = self._redact_message_content(msg.get("content"))
                cleaned.append(msg)

            # Guard: never overwrite a larger session log with fewer messages.
            # Protects against data loss when a resumed agent starts with
            # partial history and would otherwise clobber the full JSON log.
            if log_file.exists():
                try:
                    existing = json.loads(log_file.read_text(encoding="utf-8"))
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
                "system_prompt": redact_sensitive_text(self._cached_system_prompt or ""),
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")


    def interrupt(
        self,
        message: Optional[str] = None,
        *,
        hard_cancel: bool = False,
        tool_reason: Optional[str] = None,
        require_generation: Optional[int] = None,
    ) -> bool:
        """
        Request the agent to interrupt its current tool-calling loop.
        
        Call this from another thread (e.g., input handler, message receiver)
        to gracefully stop the agent and process a new message.
        
        Also signals long-running tool executions (e.g. terminal commands)
        to terminate early, so the agent can respond immediately.
        
        Args:
            message: Optional new message that triggered the interrupt.
                     If provided, the agent will include this in its response context.
            hard_cancel: Mark this as an explicit stop rather than a redirect or
                         incoming-message interrupt. Compression may honor this
                         atomic signal even while ordinary interrupts are
                         masked. With a generation claim in play, the
                         destructive compression-fence cancellation is
                         deferred until after the claim survives, so a
                         declined abort never cancels a legitimate pending
                         compression.
            tool_reason: Trusted fixed category safe to expose in tool output.
                         Arbitrary diagnostic or caller text belongs in message.
            require_generation: Optional activity-generation claim (#95663).
                         When set, the interrupt is published only if the
                         turn's activity generation still equals this value
                         at the final mutation edge. The claim is RESERVED
                         under the activity lock — ``_touch_activity``
                         invalidates the reservation the instant real
                         progress lands — survives every blocking boundary in
                         between (including the compression commit fence),
                         and is CONSUMED in ONE lock critical section
                         together with the first observable publication
                         (``_interrupt_requested`` / ``_interrupt_message`` /
                         ``_tool_interrupt_reason`` and the hard-cancel
                         event). If the turn resumed in the window, the call
                         abandons itself without publishing anything (no
                         flag, no hard-cancel event, no tool signal).

        Returns:
            True when the interrupt was published, False when a
            ``require_generation`` claim no longer matched the live activity
            clock and the call was abandoned without publishing.
        
        Example (CLI):
            # In a separate input thread:
            if user_typed_something:
                agent.interrupt(user_input)
        
        Example (Messaging):
            # When new message arrives for active session:
            if session_has_running_agent:
                running_agent.interrupt(new_message.text)
        """
        if require_generation is not None:
            # RESERVE the abort's generation claim under the SAME lock
            # `_touch_activity` stamps the clock with. Real progress
            # invalidates the reservation the instant it lands, and the
            # claim is CONSUMED at the final mutation edge — after every
            # blocking boundary — in ONE critical section with the first
            # observable publication. A resumed turn therefore abandons
            # the abort instead of being hard-cancelled by a stale proof.
            with self._liveness_activity_lock():
                if (
                    getattr(self, "_turn_liveness_activity_generation", 0)
                    != require_generation
                ):
                    return False
                self._turn_liveness_abort_claim = require_generation

        # A hard stop and redirect share one lock so /stop cannot race with an
        # accepted correction and accidentally turn itself into a retry.
        def _wait_for_compression_commit() -> None:
            # Pre-claim half of hard-cancel admission (#99758 P1): wait out
            # a commit that ALREADY crossed its boundary, so the interrupt
            # is published only after the in-flight SessionDB mutation has
            # finished — but mutate NOTHING. Cancelling a pending commit is
            # a destructive, irreversible fence mutation (``begin_commit``
            # refuses a cancelled fence forever), so it must not run while
            # a generation claim can still be vetoed: an abort that declines
            # after the fence was cancelled would have killed the recovered
            # turn's legitimate pending compression. The destructive half
            # runs in _cancel_pending_compression_commit(), only after the
            # claim survived the final mutation edge.
            fence = vars(self).get("_active_compression_commit_fence")
            if fence is None:
                return
            if not getattr(fence, "commit_in_flight", False):
                # No commit crossed its boundary — nothing to wait out,
                # and calling cancel_before_commit here WOULD cancel the
                # pending commit (the production fence's
                # cancel_before_commit sets _cancelled whenever no commit
                # has started). Skip it; the destructive half handles it.
                return
            cancel_before_commit = getattr(
                type(fence), "cancel_before_commit", None
            )
            if callable(cancel_before_commit):
                try:
                    # A commit is in flight (it holds the fence lock
                    # through finish_commit), so this call blocks until
                    # the commit finishes and returns False WITHOUT
                    # setting _cancelled — the started-commit branch of
                    # the production fence never cancels.
                    cancel_before_commit(fence)
                except Exception:
                    logger.debug(
                        "Compression hard-cancel fence wait failed",
                        exc_info=True,
                    )

        def _cancel_pending_compression_commit() -> None:
            # Destructive half of hard-cancel admission (#99758 P1): runs
            # only AFTER the generation claim survived the final mutation
            # edge, so an abort that declines can never leave the active
            # compression fence cancelled. Waiting for an in-flight commit
            # already happened in _wait_for_compression_commit(); if a
            # commit crossed its boundary in between, it can no longer be
            # fence-cancelled (it owns the fence until finish_commit and
            # completes on its own), so only a still-pending commit is
            # cancelled here.
            fence = vars(self).get("_active_compression_commit_fence")
            if fence is None:
                return
            if getattr(fence, "commit_in_flight", False):
                return
            cancel_before_commit = getattr(
                type(fence), "cancel_before_commit", None
            )
            if callable(cancel_before_commit):
                try:
                    # Marks the fence cancelled (or waits out a commit
                    # that started between the wait above and now) without
                    # setting the hard-stop Event, which was already
                    # published at the final claim edge.
                    cancel_before_commit(fence)
                except Exception:
                    logger.debug(
                        "Compression hard-cancel fence admission failed",
                        exc_info=True,
                    )

        def _publish_interrupt_state() -> None:
            self._interrupt_requested = True
            self._interrupt_message = message
            self._tool_interrupt_reason = tool_interrupt_reason
            if hard_cancel:
                _hard_event = getattr(
                    self, "_hard_interrupt_requested", None
                )
                if _hard_event is not None:
                    _hard_event.set()

        def _consume_claim_and_publish_first_state() -> bool:
            # Final mutation edge: when a generation claim is in play,
            # claim consumption and the FIRST observable interrupt
            # publication are ONE activity-lock critical section — the
            # same lock `_touch_activity` stamps the clock with. The
            # generation winner is therefore total: either the claim
            # survives and the interrupt state commits under the lock
            # BEFORE any later activity stamp, or the stamp landed first
            # and the abort declines without publishing anything. (A
            # consume-then-release-then-publish split would let a turn
            # that resumed in the consume→publication window be
            # hard-cancelled by an already-consumed claim.)
            if require_generation is None:
                # No claim to race the activity clock against: publish
                # WITHOUT touching the liveness lock. ``AIAgent``
                # stand-ins used by unrelated suites (e.g. the
                # start-order gate `_Stub`) do not carry the liveness
                # seam, and an unconditional
                # ``_liveness_activity_lock()`` acquisition here
                # regresses them with AttributeError.
                _publish_interrupt_state()
                return True
            with self._liveness_activity_lock():
                if (
                    getattr(self, "_turn_liveness_abort_claim", None)
                    != require_generation
                ):
                    return False
                self._turn_liveness_abort_claim = None
                _publish_interrupt_state()
            return True

        # Keep tool cancellation attribution separate from _interrupt_message:
        # ordinary interrupts may carry the user's full next message, which
        # must not be copied into tool output.
        tool_interrupt_reason = (
            (tool_reason or "explicit stop requested")
            if hard_cancel
            else ("user sent a new message" if message else "user interrupt")
        )

        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is not None:
            with _redirect_lock:
                # The (potentially blocking) in-flight-commit wait runs
                # BEFORE the atomic claim/publication edge; the redirect
                # lock is still held across it, exactly as before, so /stop
                # cannot race with an accepted correction. The destructive
                # pending-commit cancellation runs AFTER the claim survives
                # (#99758 P1) so a declined abort can never cancel the
                # recovered turn's legitimate compression.
                if hard_cancel:
                    _wait_for_compression_commit()
                if not _consume_claim_and_publish_first_state():
                    return False
                if hard_cancel:
                    _cancel_pending_compression_commit()
                self._pending_redirect = None
        else:
            if hard_cancel:
                _wait_for_compression_commit()
            if not _consume_claim_and_publish_first_state():
                return False
            if hard_cancel:
                _cancel_pending_compression_commit()
            self._pending_redirect = None

        # Codex app-server owns its model/tool loop and watches a private
        # interrupt event rather than Hermes' per-thread flag.
        if getattr(self, "api_mode", None) == "codex_app_server":
            _codex_session = getattr(self, "_codex_session", None)
            _request_interrupt = getattr(_codex_session, "request_interrupt", None)
            if callable(_request_interrupt):
                try:
                    _request_interrupt()
                except Exception:
                    logger.debug(
                        "Failed to interrupt Codex app-server turn",
                        exc_info=True,
                    )

        # A cron turn performs its API request on the conversation thread to
        # avoid the nested interrupt-worker deadlock.  Unlike the normal worker
        # path, its client is registered here so this cross-thread interrupt can
        # still shut down the active sockets promptly.
        _abort_active_request = getattr(self, "_active_request_abort", None)
        if callable(_abort_active_request):
            try:
                _abort_active_request("interrupt_abort")
            except Exception:
                logger.debug("Failed to abort active inline request", exc_info=True)
        # Signal all tools to abort any in-flight operations immediately.
        # Scope the interrupt to this agent's execution thread so other
        # agents running in the same process (gateway) are not affected.
        if self._execution_thread_id is not None:
            _set_interrupt(
                True,
                self._execution_thread_id,
                reason=tool_interrupt_reason,
            )
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
                    _set_interrupt(True, _wtid, reason=tool_interrupt_reason)
                except Exception:
                    pass
        # Propagate interrupt to any running child agents (subagent delegation)
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                if hard_cancel:
                    request_hard_interrupt(
                        child,
                        message,
                        tool_reason=tool_interrupt_reason,
                    )
                else:
                    child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
        return True

    def hard_interrupt(
        self,
        message: Optional[str] = None,
        *,
        tool_reason: Optional[str] = None,
    ) -> None:
        """Request an explicit stop while preserving ``interrupt()`` ABI.

        Frontends can feature-detect this method and fall back to the legacy
        ``interrupt()`` signature for synthetic or third-party agents.
        """
        # Deliberately bypass dynamic dispatch: subclasses written against the
        # legacy interrupt(message=None) ABI may override interrupt without the
        # newer keyword-only hard_cancel argument.
        AIAgent.interrupt(
            self,
            message,
            hard_cancel=True,
            tool_reason=tool_reason,
        )

    def clear_interrupt(self, *, preserve_redirect: bool = False) -> bool:
        """Clear the interrupt request and per-thread tool signal.

        ``preserve_redirect`` is used only by the conversation loop after it
        intentionally cancels a model request to rebuild that same logical
        turn. Public hard-stop paths keep the default and clear everything.
        """
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is not None:
            with _redirect_lock:
                if preserve_redirect and not self._pending_redirect:
                    return False
                self._interrupt_requested = False
                self._interrupt_message = None
                self._tool_interrupt_reason = None
                getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
                if not preserve_redirect:
                    self._pending_redirect = None
        else:
            if preserve_redirect and not getattr(self, "_pending_redirect", None):
                return False
            self._interrupt_requested = False
            self._interrupt_message = None
            self._tool_interrupt_reason = None
            getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
            if not preserve_redirect:
                self._pending_redirect = None
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
        return True

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

    def redirect(self, text: str) -> bool:
        """Redirect the active turn without converting it into a new task.

        During a normal Hermes model request this cancels only that request;
        the conversation loop retains completed messages/tool results, records
        the displayed partial reasoning as plain assistant context, appends the
        correction as a real user message, and retries. During tool execution
        it degrades to ``steer()`` so the tool can finish at a safe boundary.
        Codex app-server has a native ``turn/steer`` operation and uses it
        directly instead of cancelling.

        Returns ``False`` when there is no live turn or the text is empty, so
        surfaces can fall back to their existing next-turn queue.
        """
        if not text or not text.strip():
            return False
        cleaned = text.strip()

        # Codex owns its internal reasoning/tool loop, so use its first-class
        # active-turn steering protocol rather than interrupting the subprocess.
        if getattr(self, "api_mode", None) == "codex_app_server":
            _codex_session = getattr(self, "_codex_session", None)
            _native_steer = getattr(_codex_session, "request_steer", None)
            if callable(_native_steer):
                _redirect_lock = getattr(self, "_pending_redirect_lock", None)
                if _redirect_lock is not None:
                    with _redirect_lock:
                        if self._interrupt_requested:
                            return False
                elif self._interrupt_requested:
                    return False
                try:
                    return bool(_native_steer(cleaned))
                except Exception:
                    logger.debug("Codex app-server turn/steer failed", exc_info=True)
                    return False

        # Never kill a tool merely to deliver conversational guidance. The
        # existing steer drain puts it on the final tool result before the next
        # model decision, including delegate_task children.
        if getattr(self, "_executing_tools", False):
            return self.steer(cleaned)

        _model_active = getattr(self, "_model_request_active", None)
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            if _model_active is None or not _model_active.is_set():
                return False
            existing = getattr(self, "_pending_redirect", None)
            if self._interrupt_requested and not existing:
                return False
            self._pending_redirect = (
                f"{existing}\n\n[Additional user correction]\n{cleaned}"
                if existing
                else cleaned
            )
            self._interrupt_requested = True
            self._interrupt_message = None
        else:
            with _redirect_lock:
                if _model_active is None or not _model_active.is_set():
                    # The response completed before we acquired the state lock.
                    # Reject so the surface queues a new turn.
                    return False
                if self._interrupt_requested and not self._pending_redirect:
                    return False
                if self._pending_redirect:
                    self._pending_redirect = (
                        f"{self._pending_redirect}\n\n"
                        f"[Additional user correction]\n{cleaned}"
                    )
                else:
                    self._pending_redirect = cleaned
                self._interrupt_requested = True
                self._interrupt_message = None

        # Interrupt only the model request. Do not fan out to tool workers or
        # child agents as interrupt() does.
        _execution_thread_id = getattr(self, "_execution_thread_id", None)
        if _execution_thread_id is not None:
            _set_interrupt(True, _execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            self._interrupt_thread_signal_pending = True
        _abort_active_request = getattr(self, "_active_request_abort", None)
        if callable(_abort_active_request):
            try:
                _abort_active_request("redirect_abort")
            except Exception:
                logger.debug("Failed to abort request for redirect", exc_info=True)
        return True

    def _has_pending_redirect(self) -> bool:
        """Return whether an active-turn redirect is waiting to be applied."""
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            return bool(getattr(self, "_pending_redirect", None))
        with _redirect_lock:
            return bool(self._pending_redirect)

    def _drain_pending_redirect(self) -> Optional[str]:
        """Return and clear pending active-turn correction text."""
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            text = getattr(self, "_pending_redirect", None)
            self._pending_redirect = None
            return text
        with _redirect_lock:
            text = self._pending_redirect
            self._pending_redirect = None
        return text

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
        landed = file_mutation_result_landed(tool_name, result)
        if landed:
            landed_paths = _extract_landed_file_mutation_paths(tool_name, args, result)
            changed = getattr(self, "_turn_file_mutation_paths", None)
            if changed is not None:
                changed.update(landed_paths)
            # Feed the checkpoint agent-write ledger so /rollback's safe mode
            # can tell Hermes-authored content from later user hand-edits.
            mgr = getattr(self, "_checkpoint_mgr", None)
            if mgr is not None and getattr(mgr, "enabled", False):
                for _p in landed_paths:
                    try:
                        mgr.record_agent_write(_p)
                    except Exception:
                        pass
        if is_error and not landed:
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

        The config lookup is read once per agent and cached (mirroring
        ``_credits_notices_enabled``) — the footer gate runs at the end of
        every turn, and a config flip applying on the next session is fine.
        The env-var override stays authoritative on every call and is never
        cached, so tests and operators can still flip it at runtime.
        """
        try:
            import os as _os
            env = _os.environ.get("HERMES_FILE_MUTATION_VERIFIER")
            if env is not None:
                return env.strip().lower() not in {"0", "false", "no", "off"}
            cached = getattr(self, "_file_mutation_verifier_enabled_cache", None)
            if cached is not None:
                return cached
            # Read from the persisted config.yaml so gateway and CLI share
            # the same setting.  Import lazily to avoid a startup-time cycle.
            try:
                from hermes_cli.config import load_config as _load_config
                _cfg = _load_config() or {}
            except Exception:
                _cfg = {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "file_mutation_verifier" in _display:
                enabled = bool(_display.get("file_mutation_verifier"))
            else:
                enabled = True  # safe default: verifier on
            self._file_mutation_verifier_enabled_cache = enabled
            return enabled
        except Exception:
            pass
        return True  # safe default: verifier on

    # Bare absolute / home / Windows-drive file paths in a footer line.
    # Anchors mirror the gateway's ``extract_local_files`` bare-path
    # detector so that anything the gateway WOULD auto-attach is wrapped
    # in inline-code backticks here first (the extractor skips paths inside
    # `code` spans).  Defense-in-depth: even if a future error message
    # echoes a credential path (config.yaml, .env, auth.json) into the
    # user-facing footer, it can never be matched as a deliverable bare
    # path and silently uploaded to a messaging channel (#35584).
    _FOOTER_PATH_RE = re.compile(
        r"(?<![/:\w.`])(?:~/|/|[A-Za-z]:[/\\])(?:[\w.\-]+[/\\])*[\w.\-]+\.[\w]+",
    )

    @classmethod
    def _neutralize_footer_paths(cls, text: str) -> str:
        """Wrap bare file paths in backticks so they aren't auto-delivered.

        The gateway's ``extract_local_files`` scans response text for bare
        absolute/home paths ending in a deliverable extension and uploads
        any that exist on disk as native attachments — but it explicitly
        skips paths inside inline-code (`` `...` ``) spans.  Backticking
        every path the footer renders defeats that auto-detection while
        keeping the path fully human-readable.  Paths already wrapped in a
        backtick (the negative lookbehind excludes a preceding `` ` ``) are
        left untouched so we never double-wrap.
        """
        if not text:
            return text
        return cls._FOOTER_PATH_RE.sub(lambda m: f"`{m.group(0)}`", text)

    @classmethod
    def _format_file_mutation_failure_footer(cls, failed: Dict[str, Dict[str, Any]]) -> str:
        """Render the per-turn failed-mutation dict as a user-facing footer.

        Displays up to 10 paths with their first error preview, then a
        count of any additional failures.  Returns an empty string when
        the dict is empty so callers can concatenate unconditionally.

        Every file path that reaches the user-facing text — both the bullet
        path and any path echoed inside the tool's error preview — is
        backtick-wrapped via ``_neutralize_footer_paths`` so the gateway's
        bare-path media extractor can never auto-attach a protected file
        (e.g. ``~/.hermes/config.yaml``) to a messaging channel (#35584).
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
                lines.append(f"  • `{path}` — [{tool}] {preview}")
            else:
                lines.append(f"  • `{path}` — [{tool}] failed")
            shown += 1
        remaining = len(failed) - shown
        if remaining > 0:
            lines.append(f"  • … and {remaining} more")
        # Neutralize any path the preview text echoed (the bullet path is
        # already backticked above; the lookbehind keeps it from being
        # double-wrapped).
        return cls._neutralize_footer_paths("\n".join(lines))

    def _turn_completion_explainer_enabled(self) -> bool:
        """Check whether the end-of-turn completion explainer footer is on.

        Config path: ``display.turn_completion_explainer`` (bool, default
        True).  ``HERMES_TURN_COMPLETION_EXPLAINER`` env var overrides
        config.  Exposed as a method so tests can patch a single seam,
        mirroring ``_file_mutation_verifier_enabled``.

        The config lookup is read once per agent and cached (mirroring
        ``_credits_notices_enabled``) — the gate runs at the end of every
        turn, and a config flip applying on the next session is fine.
        The env-var override stays authoritative on every call and is never
        cached, so tests and operators can still flip it at runtime.
        """
        try:
            import os as _os
            env = _os.environ.get("HERMES_TURN_COMPLETION_EXPLAINER")
            if env is not None:
                return env.strip().lower() not in {"0", "false", "no", "off"}
            cached = getattr(self, "_turn_completion_explainer_enabled_cache", None)
            if cached is not None:
                return cached
            # Read from the persisted config.yaml so gateway and CLI share
            # the same setting.  Import lazily to avoid a startup-time cycle.
            try:
                from hermes_cli.config import load_config as _load_config
                _cfg = _load_config() or {}
            except Exception:
                _cfg = {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "turn_completion_explainer" in _display:
                enabled = bool(_display.get("turn_completion_explainer"))
            else:
                enabled = True  # safe default: explainer on
            self._turn_completion_explainer_enabled_cache = enabled
            return enabled
        except Exception:
            pass
        return True  # safe default: explainer on

    @staticmethod
    def _format_turn_completion_explanation(
        turn_exit_reason: str, persistence_cause: Optional[str] = None
    ) -> str:
        """Render a user-facing explanation for an abnormal turn ending.

        Maps the internal ``turn_exit_reason`` to a short, actionable
        message so a turn that produced no usable assistant reply (empty
        content after retries, a partial/truncated stream, a still-pending
        tool result, or an iteration/budget limit) is never silent from
        the UI's perspective — the symptom users report in #34452.

        ``persistence_cause`` refines the ``session_persistence_failed``
        wording (see ``classify_persistence_error``): lock contention gets
        "storage was busy, send it again" instead of the disk-space advice,
        which was a misdiagnosis for that failure mode. It is optional and
        ignored for every other reason, so one-argument callers keep the
        exact behavior they had before.

        Returns an empty string for reasons that are NOT abnormal (e.g.
        a normal ``text_response(...)`` exit), so callers can concatenate
        or substitute unconditionally without warning on healthy turns
        like a terse ``Done.``.
        """
        if not turn_exit_reason:
            return ""
        reason = str(turn_exit_reason)

        # Normal completion — stay quiet.  ``text_response(...)`` is the
        # healthy terminal; anything that produced a real reply is fine.
        if reason.startswith("text_response"):
            return ""

        prefix = "⚠️ No reply: "
        if reason == "empty_response_exhausted":
            return (
                prefix
                + "the model returned empty content after retries and any "
                "fallback providers. Try `continue`, switch model/provider, "
                "or inspect the tool output above."
            )
        if reason == "all_retries_exhausted_no_response":
            return (
                prefix
                + "all API retries were exhausted before a response was "
                "produced (provider errors / rate limits). Try `continue` "
                "or switch provider."
            )
        if reason == "partial_stream_recovery":
            return (
                prefix
                + "streaming stopped early and only a partial response was "
                "recovered. Send `continue` to resume from where it stopped."
            )
        if reason == "fallback_prior_turn_content":
            return (
                prefix
                + "no new content was produced this turn; showing recovered "
                "prior context. Send `continue` to retry."
            )
        if reason == "interrupted_during_api_call":
            return (
                prefix
                + "the request was interrupted mid-call before a reply was "
                "received. Send `continue` to retry."
            )
        if reason == "budget_exhausted":
            return (
                prefix
                + "the per-turn iteration/cost budget was exhausted before a "
                "final answer. Send `continue` to keep going."
            )
        if reason == "ollama_runtime_context_too_small":
            return (
                prefix
                + "the local model's context window was too small to finish. "
                "Increase the context size or use a larger model."
            )
        if reason.startswith("max_iterations_reached"):
            return (
                prefix
                + "the maximum tool-iteration limit was reached before a "
                "final answer. Send `continue` to keep going, or raise "
                "`max_iterations`."
            )
        if reason.startswith("error_near_max_iterations"):
            return (
                prefix
                + "an error occurred near the iteration limit before a final "
                "answer. Check the tool output above, then send `continue`."
            )
        if reason.startswith("repeated_outer_errors"):
            return (
                prefix
                + "the turn kept failing with repeated errors and was stopped "
                "early instead of retrying forever. Check the errors above, "
                "then send `continue` to retry."
            )
        if reason == "pending_tool_result":
            return (
                prefix
                + "the turn stopped while a tool result was still pending and "
                "the model produced no follow-up text. Send `continue` to "
                "let it summarize."
            )
        if reason == "session_persistence_failed":
            cause = persistence_cause or "unknown"
            if cause == "compression":
                return (
                    prefix
                    + "the turn was stopped because another process was "
                    "compressing this session. Your message should already be "
                    "saved — please send it again after compression completes."
                )
            if cause == "compression_closed":
                return (
                    prefix
                    + "the turn was stopped because this session was rotated "
                    "by context compression and its live continuation could "
                    "not be adopted. The storage itself is healthy — refresh "
                    "the client (or start a new turn) so it picks up the new "
                    "session id, then send your message again."
                )
            if cause == "turn_lease":
                return (
                    prefix
                    + "the turn was stopped because another Hermes process "
                    "took over this session. Your reply was not saved — wait "
                    "for the other process to finish, then send your message "
                    "again."
                )
            if cause == "locked":
                return (
                    prefix
                    + "the turn was stopped because session storage was busy "
                    "(another Hermes process was writing to the state "
                    "database). Your message should already be saved — "
                    "please send it again in a moment."
                )
            if cause == "replaced":
                return (
                    prefix
                    + "the turn was stopped because the state database file "
                    "was replaced underneath this process. Do not run "
                    "`hermes doctor --fix` or in-place FTS repair — stop "
                    "the process, restore the intended state.db, then "
                    "restart. Unwritten messages were diverted to "
                    "sessions/<session_id>.jsonl and, on the gateway, "
                    "pending_messages/pending-*.json."
                )
            if cause == "corrupt":
                return (
                    prefix
                    + "the turn was stopped because the state database "
                    "reported structural corruption (the transcript would "
                    "have been lost on restart). Freeing disk space will "
                    "not help. Recovery options:\n"
                    "1. Run `hermes doctor --fix`\n"
                    "2. Salvage with: sqlite3 ~/.hermes/state.db \".recover\" "
                    "(then replace state.db)\n"
                    "3. Restore from a backup in ~/.hermes/backups/\n"
                    "Then send your message again."
                )
            if cause == "disk":
                return (
                    prefix
                    + "the turn was stopped because session storage could not "
                    "be written (the transcript would have been lost on "
                    "restart). This is often a full disk — free some space "
                    "(or fix state.db permissions), then send your message "
                    "again."
                )
            return (
                prefix
                + "the turn was stopped because session storage could not be "
                "written (the transcript would have been lost on restart). "
                "Check the state database health (`hermes doctor`), then "
                "send your message again."
            )
        # Unknown/diagnostic-only reasons (e.g. "unknown", guardrail_halt
        # which already surfaces its own message) — don't second-guess.
        return ""

    def _apply_pending_steer_to_tool_results(self, messages: list, num_tool_msgs: int) -> None:
        """Forwarder — see ``agent.agent_runtime_helpers.apply_pending_steer_to_tool_results``."""
        from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
        return apply_pending_steer_to_tool_results(self, messages, num_tool_msgs)

    def _liveness_activity_lock(self) -> "threading.Lock":
        """Shared lock for the activity clock and its generation counter.

        ``_touch_activity`` stamps the clock under this lock; the turn
        liveness watchdog (``agent/turn_liveness.py``) samples and commits
        under the same lock, so a stall observation can never abort a turn
        that resumed between the sample and the commit (#95663 review).
        Created lazily so ``AIAgent.__new__``-based test doubles keep
        working.
        """
        _lock = getattr(self, "_turn_liveness_activity_lock", None)
        if _lock is None:
            _lock = threading.Lock()
            self._turn_liveness_activity_lock = _lock
        return _lock

    def _touch_activity(
        self,
        desc: str,
        *,
        provenance: Optional[ActivityProvenance] = None,
        force_persist: bool = False,
    ) -> None:
        """Update the last-activity timestamp and description (thread-safe).

        The clock stamp is synchronized on ``_liveness_activity_lock`` and
        bumps a monotonic generation counter, so concurrent readers (the
        turn liveness watchdog, #95548) can bind their stall observation to
        the exact ``(generation, timestamp)`` pair they sampled and
        revalidate it at the commit point.

        Also bridges to the kanban board's heartbeat fields when this
        process is a dispatcher-spawned worker (HERMES_KANBAN_TASK set),
        so the dispatcher watchdog doesn't reclaim an actively-running
        worker as stale (#31752). Bridge is rate-limited (60s) and
        best-effort — it never raises into the agent loop.

        Separately, rate-limits a durable SessionDB activity projection
        (``last_activity_at`` + bounded description/provenance) so
        CLI/Gateway consumers share one observation source (#72016 / #72039).

        ``provenance`` defaults to ``unknown`` (the ordinary agent activity
        clock). Named values are for special writers (e.g. compression);
        ordinary call sites should leave the default.

        ``force_persist`` bypasses the 60s SessionDB rate limit so a
        terminal stamp (e.g. compression completed) is not dropped.
        """
        from agent.session_activity import (
            bound_activity_description,
            normalize_activity_provenance,
            reset_session_activity_persist_window,
        )

        # Lazy per-instance lock (inline so bare doubles like
        # types.SimpleNamespace fixtures keep working — they bind
        # _touch_activity without the class, so they cannot call
        # self._liveness_activity_lock(); see
        # tests/run_agent/test_session_activity_persist.py).
        _clock_lock = getattr(self, "_turn_liveness_activity_lock", None)
        if _clock_lock is None:
            _clock_lock = threading.Lock()
            self._turn_liveness_activity_lock = _clock_lock
        with _clock_lock:
            self._turn_liveness_activity_generation = (
                getattr(self, "_turn_liveness_activity_generation", 0) + 1
            )
            self._last_activity_ts = time.time()
            self._last_activity_desc = bound_activity_description(desc)
            self._last_activity_provenance = normalize_activity_provenance(provenance)
            # Real progress invalidates any reserved abort claim. A watchdog
            # interrupt that is still in flight (e.g. parked inside the
            # compression commit fence) must abandon itself at the final
            # mutation edge instead of publishing against a generation the
            # turn has already left behind.
            self._turn_liveness_abort_claim = None
        if os.environ.get("HERMES_KANBAN_TASK"):
            try:
                from tools.kanban_tools import (
                    heartbeat_current_worker_from_env,
                    inject_new_comments_from_env,
                )
                heartbeat_current_worker_from_env()
                # Fold any new operator notes into the running turn (OUT-OF-BAND
                # steer) so the user can talk to a live task without a restart.
                inject_new_comments_from_env(self)
            except Exception:
                # Never let the bridge break the agent loop.  The function
                # already swallows exceptions internally; this outer guard
                # covers import-time failures (kanban_tools unavailable,
                # etc.) on niche deployment surfaces.
                pass
        if force_persist:
            reset_session_activity_persist_window(self)
        self._persist_session_activity_if_due()

    def _persist_session_activity_if_due(self) -> None:
        """Best-effort durable activity heartbeat for SessionDB consumers.

        Cadence is pinned by SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS
        (>=30s per session, config-independent — see agent/session_activity.py).
        The write rides the standard SessionDB ``_execute_write`` patience
        path via ``touch_session_activity``. Fail-open: a failed heartbeat
        write must NEVER raise into the agent loop (swallow + debug-log).
        """
        session_id = getattr(self, "session_id", None)
        session_db = getattr(self, "_session_db", None)
        if not session_id or session_db is None:
            return
        touch = getattr(session_db, "touch_session_activity", None)
        if not callable(touch):
            return
        from agent.session_activity import (
            SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS,
            normalize_activity_provenance,
        )

        now_mono = time.monotonic()
        last_mono = getattr(self, "_session_activity_last_persist_mono", 0.0)
        if (now_mono - last_mono) < SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS:
            return
        self._session_activity_last_persist_mono = now_mono
        try:
            touch(
                session_id,
                getattr(self, "_last_activity_ts", None),
                description=getattr(self, "_last_activity_desc", None),
                provenance=normalize_activity_provenance(
                    getattr(self, "_last_activity_provenance", None)
                ),
            )
        except Exception:
            # Never let durable heartbeat I/O break the agent loop. The
            # heartbeat is an observation-only projection; the next due
            # window retries naturally.
            logger.debug(
                "session activity heartbeat write failed (ignored)",
                exc_info=True,
            )

    def _reset_activity_labels_after_turn(self) -> None:
        """Drop mid-turn activity labels once the turn is no longer running.

        Keeps ``_last_activity_ts`` so idle/watchdog clocks stay continuous
        across interrupt-recursive turns (#15654) and between turns. Clears
        description + provenance so idle cached agents / SessionDB listings
        do not keep advertising the last mid-turn stamp (e.g. compression
        or tool execution) after the turn ended (#72039).
        """
        from agent.session_activity import ActivityProvenance

        self._last_activity_desc = ""
        self._last_activity_provenance = ActivityProvenance.UNKNOWN
        session_id = getattr(self, "session_id", None)
        session_db = getattr(self, "_session_db", None)
        if not session_id or session_db is None:
            return
        clear = getattr(session_db, "clear_session_activity_labels", None)
        if not callable(clear):
            return
        try:
            clear(session_id)
        except Exception:
            # Never let durable cleanup I/O break turn teardown.
            pass

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

    def _capture_anthropic_response_headers(self, http_response: Any) -> None:
        """Capture out-of-band state from Anthropic Messages response headers.

        The Anthropic SDK's aggregated ``Message`` drops HTTP headers. Portal
        (and other providers) put rate-limit and credits state there — the same
        families the OpenAI-wire streaming path captures via
        ``stream.response``. Fail-open: each capture swallows its own errors.
        """
        self._capture_rate_limits(http_response)
        self._capture_credits(http_response)

    def _capture_credits(self, http_response: Any) -> None:
        """Parse x-nous-credits-* headers, cache CreditsState, fire threshold notices.

        Fail-open throughout — header issues never break the agent loop. The PARSE is
        swallowed (any error → treated as a miss → keep last-known). The notice
        EVALUATION/EMIT is a SEPARATE block that WARNS on failure (R1-M2): a bug in the
        depletion-notice path must not vanish silently under the parse swallow.
        """
        # Dev test fixture (HERMES_DEV_CREDITS_FIXTURE): inject a chosen notice state
        # each turn for repeatable testing, bypassing real headers. Throwaway scaffolding.
        try:
            from agent.credits_tracker import dev_fixture_credits_state
            _fixture = dev_fixture_credits_state()
        except Exception:
            _fixture = None
        if _fixture is not None:
            self._credits_state = _fixture
            if self._credits_session_start_micros is None:
                self._credits_session_start_micros = _fixture.remaining_micros
            _latch = getattr(self, "_credits_latch", None)
            if isinstance(_latch, dict):
                # Only seen_below_90 — never seen_grant_unspent (priming it would
                # fire grant_spent on a fixture's first observation, the exact
                # every-session nag the gate exists to prevent).
                _latch["seen_below_90"] = True  # let warn90 fire without a real crossing
            _used = _fixture.used_fraction
            logger.info(
                "credits ▸ [FIXTURE] remaining=%d (%s) · paid=%s · denom=%s · used=%s "
                "(real headers bypassed — `echo clear` / unset HERMES_DEV_CREDITS_FIXTURE to restore)",
                _fixture.remaining_micros,
                _fixture.remaining_usd or "?",
                _fixture.paid_access,
                _fixture.denominator_kind,
                ("%.0f%%" % (_used * 100)) if _used is not None else "n/a",
            )
            self._emit_credits_notices()
            return
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        _dev = is_truthy_value(os.environ.get("HERMES_DEV_CREDITS"))

        # ── Parse (fail-open → miss; never overwrite good state with None) ──
        try:
            from agent.credits_tracker import parse_credits_headers
            state = parse_credits_headers(headers, provider=self.provider)
        except Exception:
            return  # parse error → treat as a miss, keep last-known
        if state is None:
            if _dev:
                logger.info(
                    "credits ▸ response had no valid x-nous-credits-* headers "
                    "(miss — producer off / non-Nous path / >TTL stale)"
                )
            return

        # retain-last-known: only overwrite on a fresh valid parse
        self._credits_state = state
        # Latch session-start remaining the first time we ever see a header
        if self._credits_session_start_micros is None:
            self._credits_session_start_micros = state.remaining_micros
        if _dev:
            # HERMES_DEV_CREDITS: stream each capture to agent.log — watch live with
            # `hermes logs -f` (grep 'credits ▸'). Dev-only; silent for normal users.
            spent = self.get_credits_spent_micros()
            used = state.used_fraction
            logger.info(
                "credits ▸ remaining=%d (%s) · paid=%s · denom=%s · used=%s "
                "· Δspent=%s · age=%s%s",
                state.remaining_micros,
                state.remaining_usd or "?",
                state.paid_access,
                state.denominator_kind,
                ("%.0f%%" % (used * 100)) if used is not None else "n/a",
                ("%.1f¢" % (spent / 10000)) if spent is not None else "n/a",
                ("%.0fs" % state.age_seconds) if state.age_seconds != float("inf") else "n/a",
                (" · disabled=%s" % state.disabled_reason) if state.disabled_reason else "",
            )

        # Threshold notices — shared with the cold-start seed (see _emit_credits_notices).
        self._emit_credits_notices()

    def _emit_credits_notices(self) -> None:
        """Run the threshold policy on the current credits state and emit notices.

        Shared by the warm path (_capture_credits) and the L3 cold-start seed, so a
        session that opens already depleted warns immediately — not only after the first
        inference header. Runs only when a notice consumer is bound (messaging binds none
        → state still cached for /usage, no policy). WARNS on failure rather than
        swallowing (R1-M2): a depletion-path bug must not vanish silently. Emits clears
        FIRST, then shows (so depleted lands last in a latest-wins slot).
        """
        if getattr(self, "notice_callback", None) is None and getattr(self, "notice_clear_callback", None) is None:
            return
        if not self._credits_notices_enabled():
            return
        state = getattr(self, "_credits_state", None)
        if state is None:
            return
        try:
            from agent.credits_tracker import evaluate_credits_notices, is_free_tier_model, new_credits_latch
            latch = getattr(self, "_credits_latch", None)
            if latch is None:
                latch = self._credits_latch = new_credits_latch()
            # Free-model gate: a depleted account on a free model can still
            # inference, so the depleted error banner is suppressed. Local-data
            # only (":free" suffix, "stealth/" prefix + pricing-cache peek) —
            # never a network call.
            model_is_free = is_free_tier_model(
                getattr(self, "model", "") or "",
                getattr(self, "base_url", "") or "",
            )
            to_show, to_clear = evaluate_credits_notices(state, latch, model_is_free=model_is_free)
            for key in to_clear:        # clears FIRST …
                self._emit_notice_clear(key)
            for notice in to_show:      # … then shows (depleted lands last in a latest-wins slot)
                self._emit_notice(notice)
        except Exception:
            logger.warning("credits notice evaluation/emit failed", exc_info=True)

    def _credits_notices_enabled(self) -> bool:
        """Whether credits notices are enabled (config display.credits_notices).

        Read once per agent and cached — the policy runs after every API
        response, and the setting governs UI noise, not correctness, so a
        config flip applying on the next session is fine.  Fail-open True
        (preserve current behaviour) on any config error.
        """
        cached = getattr(self, "_credits_notices_enabled_cache", None)
        if cached is not None:
            return cached
        enabled = True
        try:
            from hermes_cli.config import load_config as _load_config
            _cfg = _load_config() or {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "credits_notices" in _display:
                enabled = bool(_display.get("credits_notices"))
        except Exception:
            enabled = True
        self._credits_notices_enabled_cache = enabled
        return enabled

    def get_credits_state(self):
        """Return the last captured CreditsState, or None."""
        return self._credits_state

    def get_credits_spent_micros(self):
        """Session-cumulative micros spent = first_seen_remaining - current_remaining. None if no data."""
        if self._credits_session_start_micros is None or self._credits_state is None:
            return None
        return self._credits_session_start_micros - self._credits_state.remaining_micros

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

        Exposes the shared activity observation contract
        (``last_activity_at`` / ``last_activity_description`` /
        ``last_activity_provenance``) plus short aliases
        (``last_activity_ts`` / ``last_activity_desc`` / …) for existing
        gateway and delegate readers.
        """
        from agent.session_activity import (
            ActivityProvenance,
            build_activity_snapshot,
        )

        provenance = getattr(self, "_last_activity_provenance", None)
        if provenance is None:
            provenance = ActivityProvenance.UNKNOWN
        return build_activity_snapshot(
            last_activity_at=getattr(self, "_last_activity_ts", None),
            last_activity_description=getattr(self, "_last_activity_desc", None) or "",
            last_activity_provenance=provenance,
            extra={
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "budget_used": self.iteration_budget.used,
            "budget_max": self.iteration_budget.max_total,
            },
        )

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down the memory provider and context engine at session end.

        Idempotent: gateway cleanup and AIAgent.close() may share this
        ownership boundary.
        """
        if getattr(self, "_memory_provider_shutdown", False):
            return
        self._memory_provider_shutdown = True
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception as e:
                logger.warning("Memory provider on_session_end failed during shutdown: %s", e, exc_info=True)
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
        messages: list | None = None,
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
        # Multimodal turns carry content as a list of typed parts; providers
        # expect plain strings, so flatten to text first (newline-joined for
        # memory, vs the default space-join used for log/trajectory previews).
        user_text = _summarize_user_message_for_log(original_user_message, sep="\n")
        response_text = _summarize_user_message_for_log(final_response, sep="\n")
        if not (user_text and response_text):
            return
        try:
            sync_kwargs = {"session_id": self.session_id or ""}
            if messages is not None:
                sync_kwargs["messages"] = messages
            self._memory_manager.sync_all(
                user_text,
                response_text,
                **sync_kwargs,
            )
            # Sibling of the build_turn_context() prefetch gate: warming the
            # next turn's recall with a trivial prompt ("hi", "thanks") keys
            # provider searches on zero-signal text — skip it. The sync above
            # still runs so the turn itself is persisted.
            if not is_trivial_prompt(user_text):
                self._memory_manager.queue_prefetch_all(
                    user_text,
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
          - computer-use backend for task_id (native target and browser refs)
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

        # Retire the OpenAI/httpx client to release sockets immediately.
        # #70773: eviction runs on the gateway's memory-manager thread — a
        # cross-thread hard close of the shared client can release TLS FDs
        # under a still-unwinding worker (FD-recycle → SQLite corruption).
        # Retirement shuts the pooled sockets down (the memory/socket win we
        # want here) and lets GC release the FDs once no thread holds them.
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._retire_shared_openai_client(client, reason="cache_evict")
                self.client = None
        except Exception:
            pass

        # Also drop the cached per-request wire client (reused across
        # sequential LLM calls) — same socket/memory rationale as above.
        try:
            self._close_cached_request_openai_client(reason="cache_evict")
        except Exception:
            pass
        try:
            self._close_cached_request_anthropic_client(reason="cache_evict")
        except Exception:
            pass

    def close(self) -> None:
        """Release all resources held by this agent instance.

        Cleans up subprocess resources that would otherwise become orphans:
        - Background processes tracked in ProcessRegistry
        - Terminal sandbox environments
        - Browser daemon sessions
        - Computer-use backend sessions and target/ref state
        - Active child agents (subagent delegation)
        - OpenAI/httpx client connections

        Safe to call multiple times (idempotent).  Each cleanup step is
        independently guarded so a failure in one does not prevent the rest.
        """
        # AIAgent.close() is the hard owner boundary. Gateway cleanup may
        # call shutdown_memory_provider() first; its idempotence prevents
        # duplicate extraction while direct callers cannot skip provider close.
        try:
            session_messages = getattr(self, "_session_messages", None)
            self.shutdown_memory_provider(
                session_messages if isinstance(session_messages, list) else None
            )
        except Exception:
            pass

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

        # 4. Release the session-owned computer-use backend.  This ends the
        # exact cua-driver session, drops typed-browser refs/grants, and stops
        # a private embedded daemon when Hermes YOLO selected unrestricted
        # mode.  The import is lazy so sessions without computer_use retain
        # the narrow core footprint.
        try:
            from tools.computer_use import release_computer_use_session

            release_computer_use_session(task_id)
        except Exception:
            pass

        # 5. Close active child agents
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

        # 6. Close the OpenAI/httpx client
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="agent_close", shared=True)
                self.client = None
        except Exception:
            pass

        # 6b. Close the cached per-request wire client (reused across
        # sequential LLM calls; see _create_request_openai_client).
        try:
            self._close_cached_request_openai_client(reason="agent_close")
        except Exception:
            pass
        try:
            self._close_cached_request_anthropic_client(reason="agent_close")
        except Exception:
            pass

        # 6c. Close the Codex app-server session. The runtime already drops
        # it on turn crash / retirement (agent/codex_runtime.py), but hard
        # teardown had no owner — a /new, /reset, or session expiry left the
        # app-server child process running until interpreter exit. Clear the
        # attribute BEFORE close() so a concurrent reader can't grab a
        # half-closed session, and so a raising close() can't strand a stale
        # reference behind.
        try:
            codex_session = getattr(self, "_codex_session", None)
            if codex_session is not None:
                self._codex_session = None
                codex_session.close()
        except Exception:
            pass

        # 7. Free conversation history.  Mirrors _release_evicted_agent_soft's
        # soft-eviction clear — close() is the hard teardown for true session
        # boundaries (/new, /reset, session expiry), so the message list won't
        # be reused.  Drops the reference proactively rather than waiting for
        # the agent object itself to be collected, which matters when a caller
        # still holds the closed agent (e.g. a draining background task).
        try:
            self._session_messages = []
        except Exception:
            pass

        # The references above are now gone; on Linux/glibc, return their free
        # heap pages immediately instead of retaining the process RSS high-water
        # mark until exit.  This helper is a safe no-op on other allocators.
        try:
            from hermes_cli.mem_trim import trim_memory
            trim_memory(force=True, reason="agent close")
        except Exception:
            pass

        # 8. Finalize the owned SQLite session row unless this agent is only a
        # temporary helper that deliberately handed session ownership forward
        # (manual compression helpers that rotate to a continuation session_id,
        # or background-review forks that share the live parent's session_id and
        # must leave it open). end_session() is first-reason-wins and no-ops on
        # an already-ended row, so this never clobbers a 'compression' /
        # 'cron_complete' / 'cli_close' reason set by an earlier terminal path.
        session_db = getattr(self, "_session_db", None)
        try:
            if getattr(self, "_end_session_on_close", True):
                session_id = getattr(self, "session_id", None)
                if session_db and session_id:
                    session_db.end_session(session_id, "agent_close")
        except Exception:
            pass

        # 9. Close the SQLite handle itself, but ONLY when this agent owns it.
        # end_session() above finalizes the session ROW; it does not release the
        # connection. For the shared launch handle that is correct — it outlives
        # every agent — so _owns_session_db defaults False and this is a no-op.
        # A DEDICATED handle (the gateway's per-profile state.db opens, and the
        # lazy self-open in _get_session_db_for_recall) has no other owner: left
        # unclosed it keeps its db/-wal/-shm fds and its background token-writer
        # thread, and once that writer has started the instance pins ITSELF via
        # atexit.register(_drain_token_queue_at_exit) — which only close()
        # unregisters — so it survives for the life of the process.
        # Cleared first so the documented idempotency of close() holds.
        try:
            if getattr(self, "_owns_session_db", False) and session_db is not None:
                self._owns_session_db = False
                # Shared instances no-op on close(); release the refcount
                # so the registry can close when the last caller is done (#90837).
                from hermes_state import release_or_close
                release_or_close(session_db)
        except Exception:
            pass

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        Recover todo state from conversation history.
        
        The gateway creates a fresh AIAgent per message, so the in-memory
        TodoStore is empty. We scan the history for the most recent todo
        tool response and replay it to reconstruct the state.

        Hydration is restricted to tool results that are paired with an
        earlier assistant ``todo`` tool call. The gateway/API server accepts
        caller-supplied ``conversation_history``, so a forged bare
        ``role: tool`` message carrying a ``todos`` array must not be able to
        seed the store without a matching canonical tool call
        (GHSA-5g4g-6jrg-mw3g).
        """
        from tools.todo_tool import MAX_TODO_RESULT_CHARS

        # Walk history backwards to find the most recent todo tool response
        last_todo_response = None
        last_todo_revision = 0
        for idx in range(len(history) - 1, -1, -1):
            msg = history[idx]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            # Only accept tool results paired with a prior assistant todo call.
            if not self._tool_response_matches_todo_call(history, idx):
                continue
            if len(content) > MAX_TODO_RESULT_CHARS:
                logger.warning(
                    "Skipping oversized todo tool response during hydration: "
                    "session=%s chars=%d",
                    self.session_id or "none",
                    len(content),
                )
                continue
            # Quick check: todo responses contain "todos" key
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    last_todo_revision = data.get("revision", 1)
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        if last_todo_response is not None:
            # Restore only when history carries a newer revision than the
            # store already holds (a live store re-hydrated in place must not
            # be rolled back by older history). Sessions that predate
            # revisions default to 1 so they still hydrate. Empty lists
            # matter: they are an authoritative clear after an earlier
            # non-empty plan.
            current_revision = int(
                self._todo_store.snapshot().get("revision", 0) or 0
            )
            try:
                history_revision = max(0, int(last_todo_revision or 0))
            except (TypeError, ValueError):
                history_revision = 1
            if history_revision > current_revision:
                self._todo_store.restore(
                    last_todo_response,
                    revision=history_revision,
                )
                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)

    @classmethod
    def _tool_response_matches_todo_call(
        cls,
        history: List[Dict[str, Any]],
        tool_index: int,
    ) -> bool:
        """Return True when a tool result belongs to a prior assistant todo call.

        Scans backwards from the tool result to the nearest assistant message
        and confirms it issued a ``todo`` tool call whose id matches this
        result's ``tool_call_id``. A ``user``/``system`` boundary (or a missing
        id) means the result is unpaired and must not hydrate the store.
        """
        if tool_index < 0 or tool_index >= len(history):
            return False
        tool_msg = history[tool_index]
        tool_call_id = tool_msg.get("tool_call_id")
        if not tool_call_id:
            return False

        for prior_idx in range(tool_index - 1, -1, -1):
            prior = history[prior_idx]
            role = prior.get("role")
            if role == "assistant":
                return cls._assistant_has_todo_tool_call(prior, tool_call_id)
            if role in {"user", "system"}:
                return False
        return False

    @classmethod
    def _assistant_has_todo_tool_call(
        cls,
        assistant_msg: Dict[str, Any],
        tool_call_id: str,
    ) -> bool:
        """True when the assistant message issued a ``todo`` call with this id."""
        tool_calls = assistant_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return False

        for tool_call in tool_calls:
            if cls._get_tool_call_id_static(tool_call) != tool_call_id:
                continue
            if cls._get_tool_call_name_static(tool_call) == "todo":
                return True
        return False

    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested










    def _build_system_prompt_parts(self, system_message: str = None) -> Dict[str, str]:
        """Forwarder — see ``agent.system_prompt.build_system_prompt_parts``."""
        from agent.system_prompt import build_system_prompt_parts
        return build_system_prompt_parts(self, system_message=system_message)

    def _build_system_prompt(self, system_message: str = None) -> str:
        """Forwarder — see ``agent.system_prompt.build_system_prompt``."""
        from agent.system_prompt import build_system_prompt
        return build_system_prompt(self, system_message=system_message)

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """Extract call ID from a tool_call entry (dict or object).

        Forwarder — policy owner is
        ``agent.message_sanitization.coalesce_tool_call_id`` (audit F4).
        """
        return _sanitize_coalesce_tool_call_id(tc)

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
        """Forwarder — see ``agent.agent_runtime_helpers.sanitize_api_messages``."""
        from agent.agent_runtime_helpers import sanitize_api_messages
        return sanitize_api_messages(messages)

    @staticmethod
    def _is_thinking_only_assistant(
        msg: Dict[str, Any],
        *,
        drop_codex_reasoning_items: bool = True,
    ) -> bool:
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
        # Prefill stubs are thinking-only by construction; check before content
        # inspection since repair_empty_non_final_messages may have healed content.
        if msg.get("_thinking_prefill"):
            return True
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
        # A native compaction checkpoint makes a carrier never thinking-only,
        # regardless of api_mode or which reasoning field is populated. The
        # checkpoint is the server-side stand-in for already-pruned history
        # and exists in exactly one place; the codex_responses adapter also
        # surfaces commentary text via msg["reasoning"], so the string branch
        # below would otherwise drop a carrier before the sidecar is ever
        # inspected. Checked here — above every reasoning branch — so no
        # carrier shape can fall into a drop path (#82108 review finding).
        from agent.native_compaction import has_compaction_checkpoint

        if has_compaction_checkpoint(msg.get("codex_reasoning_items")):
            return False
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return True
        # reasoning_details list form
        rd = msg.get("reasoning_details")
        if isinstance(rd, list) and rd:
            return True
        # Codex Responses stores encrypted reasoning state under a separate
        # assistant-message key. Treat only real reasoning items as
        # thinking-only; empty/junk lists should fall through to the generic
        # empty-turn handling instead of being dropped here.
        codex_items = msg.get("codex_reasoning_items")
        if drop_codex_reasoning_items and isinstance(codex_items, list):
            return any(
                isinstance(item, dict) and item.get("type") == "reasoning"
                for item in codex_items
            )
        return False

    @staticmethod
    def _drop_thinking_only_and_merge_users(
        messages: List[Dict[str, Any]],
        *,
        drop_codex_reasoning_items: bool = True,
    ) -> List[Dict[str, Any]]:
        """Forwarder — see ``agent.agent_runtime_helpers.drop_thinking_only_and_merge_users``."""
        from agent.agent_runtime_helpers import drop_thinking_only_and_merge_users
        return drop_thinking_only_and_merge_users(
            messages,
            drop_codex_reasoning_items=drop_codex_reasoning_items,
        )

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

        Valid JSON arguments are canonicalized so equivalent objects do not
        evade deduplication merely because their keys or whitespace differ.
        Malformed arguments retain their raw representation rather than being
        repaired here. Only the first occurrence of each unique pair is kept.
        Returns the original list if no duplicates were found.
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            arguments = tc.function.arguments
            try:
                arguments = json.dumps(
                    json.loads(arguments), separators=(",", ":"), sort_keys=True
                )
            except (TypeError, ValueError):
                pass
            key = (tc.function.name, arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    @staticmethod
    def _uniquify_tool_call_ids(tool_calls: list) -> list:
        """Ensure every tool call in a single assistant turn has a distinct id.

        Forwarder — policy owner is
        ``agent.message_sanitization.uniquify_tool_call_ids`` (audit F4).
        First occurrence keeps its id; later collisions get a deterministic
        ``<id>_d<n>`` suffix (never uuid4 — prompt-cache prefix stability).
        Mutates entries in place and returns the same list.
        """
        return _sanitize_uniquify_tool_call_ids(tool_calls)

    def _repair_tool_call(self, tool_name: str) -> str | None:
        """Forwarder — see ``agent.agent_runtime_helpers.repair_tool_call``."""
        from agent.agent_runtime_helpers import repair_tool_call
        return repair_tool_call(self, tool_name)

    def _invalidate_system_prompt(self):
        """Forwarder — see ``agent.system_prompt.invalidate_system_prompt``."""
        from agent.system_prompt import invalidate_system_prompt
        invalidate_system_prompt(self)

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
    def _build_keepalive_http_client(base_url: str = "", *, verify: Any = True) -> Any:
        """Build the shared OpenAI httpx client used by main and aux paths."""
        from agent.process_bootstrap import build_keepalive_http_client

        return build_keepalive_http_client(base_url, verify=verify)

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        """Forwarder — see ``agent.agent_runtime_helpers.create_openai_client``."""
        from agent.agent_runtime_helpers import create_openai_client
        return create_openai_client(self, client_kwargs, reason=reason, shared=shared)

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """Forwarder — see ``agent.agent_runtime_helpers.force_close_tcp_sockets``."""
        from agent.agent_runtime_helpers import force_close_tcp_sockets
        return force_close_tcp_sockets(client)

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

    def _retire_shared_openai_client(self, client: Any, *, reason: str) -> None:
        """Ownership-safe retirement of a replaced shared OpenAI client.

        #70773 / #67142 / #29507: ``client.close()`` releases the pool's raw
        FDs from the *calling* thread. The shared primary client has no single
        owning thread — worker threads from stale-killed attempts may still be
        unwinding their SSL BIOs, and the codex-direct / MoA paths stream on
        the shared client itself. If we release an FD while another thread's
        SSL layer still caches the raw integer fd, the kernel can recycle it
        into an unrelated ``open()`` (e.g. ``kanban.db``) and the unwinding
        TLS flush then writes an application-data record into that file — the
        SQLite-header corruption documented in #29507/#70773.

        Only an owner may release FDs, and a replaced shared client has none.
        So nobody calls ``close()``: we ``shutdown()`` the pooled sockets
        (FD-safe from any thread; unblocks in-flight readers with EOF/EPIPE)
        and defer the actual FD release to garbage collection. Refcounting
        guarantees the underlying sockets are only collected once every
        thread that borrowed the client has unwound — GC *is* the ownership
        handshake. In the common case (no borrower) the refcount hits zero on
        this line and the FDs are released immediately anyway.
        """
        if client is None:
            return
        try:
            shutdown_count = self._force_close_tcp_sockets(client)
            logger.info(
                "Shared OpenAI client retired (%s, tcp_shutdown=%d, "
                "fd_release=deferred_to_gc) %s",
                reason,
                shutdown_count,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "Shared OpenAI client retire failed (%s) %s error=%s",
                reason,
                self._client_log_context(),
                exc,
            )

    def _drain_transports_after_abandonment(self, *, reason: str) -> int:
        """FD-safe transport drain for an abandoned (timed-out) worker (#94248).

        A delegation deadline abandons this agent's daemon worker while it may
        still be blocked inside an in-flight OpenSSL ``read`` (Codex Responses
        stream, httpx request). The timeout thread must never hard-close those
        transports — ``client.close()`` releases raw FDs under a live SSL BIO,
        the #29507 / #67142 / #70773 native-corruption family and the SIGSEGV
        shape reported in #94248. This helper only ``shutdown()``s pooled
        sockets (safe from any thread), settling blocked reads with EOF/EPIPE
        so the worker can unwind and run the real close from its own thread.

        Returns the number of sockets shut down across all transports.
        """
        drained = 0
        # Shared primary client (codex-direct / MoA stream on it directly).
        try:
            client = getattr(self, "client", None)
            if client is not None:
                drained += self._force_close_tcp_sockets(client)
        except Exception:
            logger.debug("Abandoned-worker drain: shared client sweep failed",
                         exc_info=True)
        # Cached per-request wire clients: abort (shutdown + poison the reuse
        # slot) so the unwinding worker discards them instead of re-caching.
        try:
            with self._openai_client_lock():
                cache = getattr(self, "_request_client_cache", None)
                cached = cache["client"] if cache else None
            if cached is not None:
                self._abort_request_openai_client(cached, reason=reason)
        except Exception:
            logger.debug("Abandoned-worker drain: request client abort failed",
                         exc_info=True)
        try:
            with self._openai_client_lock():
                cache = getattr(self, "_request_anthropic_client_cache", None)
                cached = cache["client"] if cache else None
            if cached is not None:
                self._abort_request_anthropic_client(cached, reason=reason)
        except Exception:
            logger.debug("Abandoned-worker drain: anthropic client abort failed",
                         exc_info=True)
        # Codex app-server session watches a private interrupt event.
        try:
            codex_session = getattr(self, "_codex_session", None)
            request_interrupt = getattr(codex_session, "request_interrupt", None)
            if callable(request_interrupt):
                request_interrupt()
        except Exception:
            logger.debug("Abandoned-worker drain: codex interrupt failed",
                         exc_info=True)
        # Inline (cron-style) request abort hook, when registered.
        try:
            abort_active = getattr(self, "_active_request_abort", None)
            if callable(abort_active):
                abort_active(reason)
        except Exception:
            logger.debug("Abandoned-worker drain: active request abort failed",
                         exc_info=True)
        logger.info(
            "Abandoned-worker transports drained (%s, tcp_shutdown=%d, "
            "fd_release=deferred_to_worker) %s",
            reason,
            drained,
            self._client_log_context(),
        )
        return drained

    def _build_primary_client_for_active_provider(self, *, reason: str) -> Any:
        """Build the shared client shape required by the active provider.

        MoA is a virtual provider whose ``client`` is an in-process facade,
        not an OpenAI SDK client. Generic rebuild paths (credential rotation,
        timeout application, and dead-connection cleanup) still pass through
        this helper, so they must preserve that provider/client invariant.
        """
        if (getattr(self, "provider", "") or "").strip().lower() == "moa":
            from agent.moa_loop import build_moa_facade

            return build_moa_facade(self, self.model)
        return self._create_openai_client(
            self._client_kwargs,
            reason=reason,
            shared=True,
        )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._build_primary_client_for_active_provider(
                    reason=reason,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared primary client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        # #70773: never hard-close the replaced shared client from here — the
        # caller may not be the thread whose request is still unwinding on the
        # old pool (credential rotation and dead-connection cleanup run on the
        # turn thread while stale-killed workers unwind; the codex-direct path
        # streams on the shared client itself). Retire it instead: sockets are
        # shut down (FD-safe), FD release deferred to GC.
        self._retire_shared_openai_client(old_client, reason=f"replace:{reason}")
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client
            old_client = client
            try:
                new_client = self._create_openai_client(
                    self._client_kwargs, reason=reason, shared=True
                )
            except Exception as exc:
                logger.warning(
                    "Failed to recreate closed OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                raise RuntimeError("Failed to recreate closed OpenAI client") from exc
            self.client = new_client

        logger.warning(
            "Detected closed shared OpenAI client; recreated before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return new_client

    def _cleanup_dead_connections(self) -> bool:
        """Forwarder — see ``agent.agent_runtime_helpers.cleanup_dead_connections``."""
        from agent.agent_runtime_helpers import cleanup_dead_connections
        return cleanup_dead_connections(self)

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

    # Close reasons the request workers' own ``finally`` unwind reports for
    # a request that produced a response — the only closes that both come
    # from the thread that owns the pool's FDs AND attest a healthy pool.
    # Only these may keep the wire client for the next call, and poisoning
    # still wins: a cross-thread abort (#29507) marks the slot so even a
    # worker-finally close discards it. Every other reason (error cleanups,
    # stale/interrupt kills, retry cleanups) gets a real close, so a retry
    # after a request error always builds a fresh pool.
    _REQUEST_CLIENT_REUSE_REASONS = frozenset({
        "request_complete",
        "stream_request_complete",
    })

    def _request_client_cache_ref(self) -> dict:
        # Lazy init — tests build agents via AIAgent.__new__ without __init__.
        cache = getattr(self, "_request_client_cache", None)
        if cache is None:
            cache = {"client": None, "kwargs": None, "poisoned": False, "in_use": False}
            self._request_client_cache = cache
        return cache

    def _create_request_openai_client(self, *, reason: str, api_kwargs: Optional[dict] = None) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if self.provider == "moa":
            return primary_client
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
            base_url_host_matches(str(request_kwargs.get("base_url", "")), "githubcopilot.com")
            and self._api_kwargs_have_image_parts(api_kwargs or {})
        ):
            request_kwargs["default_headers"] = self._copilot_headers_for_request(is_vision=True)
        # Reuse the cached wire client while the effective kwargs are
        # unchanged: constructing openai.OpenAI + its httpx pool costs
        # ~19-35ms per LLM call (fresh TCP+TLS handshake), ~5x per turn.
        # The cache is a single checked-out slot: `in_use` prevents two
        # concurrent calls from sharing one pool's close/abort lifecycle
        # (a second concurrent call gets a fresh untracked client with
        # the old build-per-request behavior).
        stale = None
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            cached = cache["client"]
            if cached is not None and not cache["in_use"]:
                if (
                    not cache["poisoned"]
                    and cache["kwargs"] == request_kwargs
                    and not self._is_openai_client_closed(cached)
                ):
                    cache["in_use"] = True
                    return cached
                # kwargs changed (credential rotation, provider failover),
                # poisoned by a cross-thread abort (#29507), or externally
                # closed — never reuse; discard and rebuild below.
                stale = cached
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
        if stale is not None:
            # Safe to close from this thread: in_use was False, so no
            # worker thread owns the pool's FDs (#29507 concerns clients
            # with an in-flight request on another thread).
            self._close_openai_client(stale, reason=f"reuse_evict:{reason}", shared=False)
        client = self._create_openai_client(request_kwargs, reason=reason, shared=False)
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is None:
                cache["client"] = client
                # Snapshot nested dicts (default_headers): rotation sites
                # assign fresh inner dicts today, but an aliased inner
                # object would compare equal even after in-place mutation.
                cache["kwargs"] = {
                    k: dict(v) if isinstance(v, dict) else v
                    for k, v in request_kwargs.items()
                }
                cache["poisoned"] = False
                cache["in_use"] = True
            # else: a concurrent call holds the slot — hand this client
            # out untracked; _close_request_openai_client fully closes
            # untracked clients, preserving the per-request lifecycle.
        return client

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is client:
                if reason in self._REQUEST_CLIENT_REUSE_REASONS and not cache["poisoned"]:
                    # Clean finish on the owning thread — keep the wire client
                    # (and its warm httpx pool) for the next sequential call.
                    cache["in_use"] = False
                    return
                # Failure / kill / abort outcome: drop the slot and fall
                # through to a real close. This runs on the owning worker
                # thread, which is where the FD release belongs (#29507).
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        self._close_openai_client(client, reason=reason, shared=False)

    def _close_cached_request_openai_client(self, *, reason: str) -> None:
        """Teardown hook: really close the cached per-request wire client."""
        with self._openai_client_lock():
            cache = getattr(self, "_request_client_cache", None)
            client = cache["client"] if cache else None
            in_use = bool(cache["in_use"]) if cache else False
            if cache is not None:
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        if client is None:
            return
        if in_use:
            # A worker thread has this client checked out for an in-flight
            # request (workers can outlive turns — see interruptible_api_call).
            # client.close() here would release its FDs from a stranger thread,
            # the #29507 race teardown must not reintroduce. Abort the sockets
            # instead; the slot is already cleared, so the worker's own finally
            # sees an untracked client and does the real close on its thread.
            self._abort_request_openai_client(client, reason=f"{reason}_in_flight")
            return
        self._close_openai_client(client, reason=reason, shared=False)

    def _abort_request_openai_client(self, client: Any, *, reason: str) -> None:
        """Cross-thread abort: shut sockets down without releasing FDs.

        Companion to :meth:`_close_request_openai_client` for stranger-thread
        callers (interrupt-check loop, stale-call detector). Calling
        ``client.close()`` from a thread that does not own the active httpx
        connection raced the still-live SSL BIO and corrupted unrelated file
        descriptors when the kernel recycled the just-freed TCP FD (#29507).

        Here we only ``shutdown(SHUT_RDWR)`` the pool's sockets. That unblocks
        the owning worker thread's pending ``recv``/``send`` with an EOF or
        ``EPIPE`` so it can unwind and close ``client`` from its own context
        — which is where the FD release belongs.
        """
        if client is None:
            return
        # A pool whose sockets were shut down from a stranger thread must
        # never be reused: poison the cache slot so the owner-thread close
        # discards it and the next create builds a fresh client.
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is client:
                cache["poisoned"] = True
        try:
            shutdown_count = self._force_close_tcp_sockets(client)
            # tcp_force_closed=0 means the stranger-thread abort found no
            # sockets to shut down — the worker stays blocked in recv and the
            # provider keeps the slot (#72975). Surface that as WARNING so it
            # cannot be mistaken for a successful abort in the logs.
            _log = logger.warning if shutdown_count == 0 else logger.info
            _log(
                "OpenAI client aborted (%s, shared=False, tcp_force_closed=%d, "
                "deferred_close=stranger_thread) %s%s",
                reason,
                shutdown_count,
                self._client_log_context(),
                (
                    " — no sockets found; in-flight request may keep running "
                    "until the provider finishes"
                    if shutdown_count == 0
                    else ""
                ),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client abort failed (%s, shared=False) %s error=%s",
                reason,
                self._client_log_context(),
                exc,
            )

    def _request_anthropic_client_cache_ref(self) -> dict:
        # Lazy init — tests build agents via AIAgent.__new__ without __init__.
        cache = getattr(self, "_request_anthropic_client_cache", None)
        if cache is None:
            cache = {"client": None, "key": None, "poisoned": False, "in_use": False}
            self._request_anthropic_client_cache = cache
        return cache

    def _request_anthropic_client_key(self) -> tuple:
        """Cache key covering everything that forces a fresh client: credential
        rotation, base URL / region changes, timeout changes (model switch),
        and the 1M-context beta flag."""
        if getattr(self, "provider", None) == "bedrock":
            region = getattr(self, "_bedrock_region", "us-east-1") or "us-east-1"
            return ("bedrock", region)
        return (
            "direct",
            self._anthropic_api_key,
            getattr(self, "_anthropic_base_url", None),
            get_provider_request_timeout(self.provider, self.model),
            bool(getattr(self, "_oauth_1m_beta_disabled", False)),
        )

    def _create_request_anthropic_client(self, *, reason: str) -> Any:
        """Build (or reuse) a request-local Anthropic client for one in-flight call.

        The shared ``_anthropic_client`` stays the long-lived primary, but the
        stale/interrupt watchdog runs on the poll thread and must never call
        ``close()`` on the client whose TLS socket a worker thread is still
        reading: releasing that FD from a stranger thread lets the kernel
        recycle it under a still-live SSL BIO, which then writes a TLS record
        into an unrelated SQLite header (#29507 / #67142). A per-request client
        lets the stranger thread ``shutdown()`` the socket while the owning
        worker performs the SDK-level close from its own context — the same
        ownership contract the OpenAI-wire path already uses.

        Also mirrors the OpenAI-wire path's single-slot cache
        (``_create_request_openai_client``): building ``anthropic.Anthropic``
        means a fresh httpx pool and TCP+TLS handshake per call, so the client
        is kept warm across sequential calls whose cache key (credentials,
        base URL/region, timeout, 1M-beta flag) hasn't changed. ``in_use``
        keeps a second concurrent call from sharing one pool's close/abort
        lifecycle — it gets a fresh untracked client instead.

        Mirrors ``_rebuild_anthropic_client`` construction (direct + Bedrock,
        1M-beta drop) but returns a fresh/cached client instead of swapping
        the shared one.
        """
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        key = self._request_anthropic_client_key()

        stale = None
        with self._openai_client_lock():
            cache = self._request_anthropic_client_cache_ref()
            cached = cache["client"]
            if cached is not None and not cache["in_use"]:
                if (
                    not cache["poisoned"]
                    and cache["key"] == key
                    and not self._is_openai_client_closed(cached)
                ):
                    cache["in_use"] = True
                    return cached
                # Key changed (credential rotation, base URL/region, timeout,
                # 1M-beta flip), poisoned by a cross-thread abort, or
                # externally closed — never reuse; discard and rebuild below.
                stale = cached
                cache["client"] = None
                cache["key"] = None
                cache["poisoned"] = False
        if stale is not None:
            # Safe to close from this thread: in_use was False, so no worker
            # thread owns the pool's FDs (same #29507 reasoning as OpenAI).
            self._close_request_anthropic_client(stale, reason=f"reuse_evict:{reason}")

        if key[0] == "bedrock":
            from agent.anthropic_adapter import build_anthropic_bedrock_client
            client = build_anthropic_bedrock_client(key[1])
        else:
            from agent.anthropic_adapter import build_anthropic_client
            client = build_anthropic_client(
                self._anthropic_api_key,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
                drop_context_1m_beta=key[4],
            )
        logger.debug(
            "Anthropic request client created (%s, shared=False) provider=%s model=%s",
            reason,
            getattr(self, "provider", None),
            getattr(self, "model", None),
        )
        with self._openai_client_lock():
            cache = self._request_anthropic_client_cache_ref()
            if cache["client"] is None:
                cache["client"] = client
                cache["key"] = key
                cache["poisoned"] = False
                cache["in_use"] = True
            # else: a concurrent call holds the slot — hand this client out
            # untracked; _close_request_anthropic_client fully closes
            # untracked clients, preserving the per-request lifecycle.
        return client

    def _close_request_anthropic_client(self, client: Any, *, reason: str) -> None:
        """Owner-thread close of a request-local Anthropic client.

        On a clean finish (``reason`` in ``_REQUEST_CLIENT_REUSE_REASONS``)
        the pool is kept warm in the cache slot for the next sequential call,
        mirroring ``_close_request_openai_client``. Any other outcome
        (error / kill / abort / stale-slot eviction) force-closes the pool's
        TCP sockets first (CLOSE-WAIT hygiene, parity with
        ``_close_openai_client``), then does the graceful SDK close. Safe
        because the caller owns the connection.
        """
        if client is None:
            return
        with self._openai_client_lock():
            cache = self._request_anthropic_client_cache_ref()
            if cache["client"] is client:
                if reason in self._REQUEST_CLIENT_REUSE_REASONS and not cache["poisoned"]:
                    cache["in_use"] = False
                    return
                cache["client"] = None
                cache["key"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        try:
            self._force_close_tcp_sockets(client)
            client.close()
            logger.info(
                "Anthropic client closed (%s, shared=False) provider=%s model=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
            )
        except Exception as exc:
            logger.debug(
                "Anthropic client close failed (%s, shared=False) provider=%s model=%s error=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                exc,
            )

    def _close_cached_request_anthropic_client(self, *, reason: str) -> None:
        """Teardown hook: really close the cached per-request Anthropic client."""
        with self._openai_client_lock():
            cache = getattr(self, "_request_anthropic_client_cache", None)
            client = cache["client"] if cache else None
            in_use = bool(cache["in_use"]) if cache else False
            if cache is not None:
                cache["client"] = None
                cache["key"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        if client is None:
            return
        if in_use:
            # A worker thread has this client checked out for an in-flight
            # request — same #29507 reasoning as the OpenAI teardown hook.
            self._abort_request_anthropic_client(client, reason=f"{reason}_in_flight")
            return
        try:
            self._force_close_tcp_sockets(client)
            client.close()
        except Exception:
            pass

    def _abort_request_anthropic_client(self, client: Any, *, reason: str) -> None:
        """Cross-thread abort for request-local Anthropic clients.

        Stranger threads (the interrupt-check / stale-stream detector loop)
        must not call the SDK ``close()`` — that races the owning worker's live
        SSL BIO and can recycle a TLS FD into a SQLite header (#29507 /
        #67142). Only ``shutdown(SHUT_RDWR)`` the pool's sockets so the worker
        unblocks and releases the FD from its own thread.
        """
        if client is None:
            return
        # A pool whose sockets were shut down from a stranger thread must
        # never be reused: poison the cache slot so the owner-thread close
        # discards it and the next create builds a fresh client.
        with self._openai_client_lock():
            cache = self._request_anthropic_client_cache_ref()
            if cache["client"] is client:
                cache["poisoned"] = True
        try:
            shutdown_count = self._force_close_tcp_sockets(client)
            # Same visibility contract as the OpenAI abort path (#72975):
            # zero sockets shut down means the abort did not unblock the
            # worker — log WARNING, not a success-shaped INFO.
            _log = logger.warning if shutdown_count == 0 else logger.info
            _log(
                "Anthropic client aborted (%s, shared=False, tcp_force_closed=%d, "
                "deferred_close=stranger_thread) provider=%s model=%s%s",
                reason,
                shutdown_count,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                (
                    " — no sockets found; in-flight request may keep running "
                    "until the provider finishes"
                    if shutdown_count == 0
                    else ""
                ),
            )
        except Exception as exc:
            logger.debug(
                "Anthropic client abort failed (%s, shared=False) provider=%s model=%s error=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                exc,
            )

    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """Forwarder — see ``agent.codex_runtime.run_codex_stream``."""
        from agent.codex_runtime import run_codex_stream
        return run_codex_stream(self, api_kwargs, client, on_first_delta)

    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """Forwarder — see ``agent.codex_runtime.run_codex_create_stream_fallback``."""
        from agent.codex_runtime import run_codex_create_stream_fallback
        return run_codex_create_stream_fallback(self, api_kwargs, client)

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider not in {"openai-codex", "xai-oauth"}:
            return False

        # Guard against silent account swap.
        #
        # When an agent is using a non-singleton credential — e.g. a manual
        # pool entry (``hermes auth add xai-oauth``) whose tokens belong to
        # a different account than the device_code singleton, or an agent
        # constructed with an explicit ``api_key=`` arg — force-refreshing
        # the singleton here and adopting its tokens silently re-routes the
        # rest of the conversation onto the singleton's account.  The
        # credential pool's reactive recovery (``_recover_with_credential_pool``)
        # is the right channel for that case; this path is the
        # singleton-only fallback used when the pool can't recover, and
        # MUST only fire when the agent really is on singleton tokens.
        try:
            if self.provider == "openai-codex":
                from hermes_cli.auth import resolve_codex_runtime_credentials

                singleton_now = resolve_codex_runtime_credentials(
                    refresh_if_expiring=False,
                )
            else:
                from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

                singleton_now = resolve_xai_oauth_runtime_credentials(
                    refresh_if_expiring=False,
                )
        except Exception as exc:
            logger.debug("%s singleton read failed: %s", self.provider, exc)
            return False

        singleton_key = str(singleton_now.get("api_key") or "").strip()
        active_key = str(self.api_key or "").strip()
        if singleton_key and active_key and singleton_key != active_key:
            logger.debug(
                "%s singleton tokens differ from the active api_key; "
                "skipping singleton force-refresh to avoid silent account swap. "
                "Reactive credential rotation should go through the pool.",
                self.provider,
            )
            return False

        try:
            if self.provider == "openai-codex":
                from hermes_cli.auth import resolve_codex_runtime_credentials

                old_key = str(self.api_key or "").strip()
                creds = resolve_codex_runtime_credentials(force_refresh=force)
            else:
                from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

                old_key = str(self.api_key or "").strip()
                creds = resolve_xai_oauth_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("%s credential refresh failed: %s", self.provider, exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        # Defect 2 fix: return False when no NEW token was actually minted.
        # resolve_codex_runtime_credentials returns the same stale token
        # when the underlying refresh fails (failure is debug-only).
        # Comparing the access token (api_key) before/after detects this.
        new_key = api_key.strip()
        if old_key and new_key == old_key:
            logger.debug(
                "%s credential refresh returned the same token; "
                "refresh likely failed silently",
                self.provider,
            )
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason=f"{self.provider}_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(
        self,
        *,
        force: bool = True,
    ) -> bool:
        if self.provider != "nous":
            return False
        # Portal serves anthropic/* on the native Messages route, so a session
        # can be holding either client kind when its short-lived invoke JWT
        # expires. Both need the refresh or the turn dies on a 401.
        if self.api_mode not in ("chat_completions", "anthropic_messages"):
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
                force_refresh=force,
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

        if self.api_mode == "anthropic_messages":
            self._anthropic_api_key = self.api_key
            self._anthropic_base_url = self.base_url
            self._rebuild_anthropic_client()
            return True

        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous requests should not inherit OpenRouter-only attribution headers.
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_env_client_credentials(self) -> bool:
        """Adopt ~/.hermes/.env credential/base-url edits at the turn boundary.

        A Settings save (desktop ``PUT /api/env``, ``hermes setup``) updates
        ``.env`` and the *saving* process's os.environ, but a live session
        worker keeps the base_url/api_key captured at agent init until it
        restarts — so an open chat silently keeps calling the old endpoint
        (#67821). Called at the start of each conversation turn, this
        re-resolves the provider's env-sourced credentials (load_env() is
        mtime-memoized, so an unchanged file costs one stat()) and rebuilds
        the client when the user edited them.

        Reacts only to env *edits* (resolved values changed since the last
        look), never to mere divergence from the agent's current values —
        credential-pool rotation and failover legitimately move the session
        off the env credential, and stomping those back every turn would
        flap. A config.yaml ``model.base_url`` (or a pool entry with a
        custom endpoint) also wins: edits are only adopted while the
        session's current base_url is still the registry default or the
        previously-seen env value.

        Covers api-key registry providers and named custom providers with a
        ``key_env`` (#67935) — the latter resolve to ``provider="custom"``
        with no registry entry, so they are matched through the runtime
        provider's config lookup instead.
        """
        if self.api_mode != "chat_completions":
            return False
        if getattr(self, "_fallback_activated", False):
            return False
        try:
            from agent.credential_pool import get_env_prefer_dotenv
            from hermes_cli.auth import PROVIDER_REGISTRY
        except ImportError:
            return False

        pconfig = PROVIDER_REGISTRY.get(self.provider)
        if (
            pconfig
            and getattr(pconfig, "auth_type", "") == "api_key"
            and getattr(pconfig, "api_key_env_vars", ())
        ):
            api_key = ""
            for env_var in pconfig.api_key_env_vars:
                api_key = get_env_prefer_dotenv(env_var).strip()
                if api_key:
                    break
            if not api_key:
                return False

            env_url = ""
            if pconfig.base_url_env_var:
                env_url = get_env_prefer_dotenv(pconfig.base_url_env_var).strip().rstrip("/")
            default_base = (pconfig.inference_base_url or "").strip().rstrip("/")
            base_url = env_url or default_base
            if self.provider == "kimi-coding":
                from hermes_cli.auth import _resolve_kimi_base_url

                base_url = _resolve_kimi_base_url(
                    api_key, pconfig.inference_base_url, env_url
                ).rstrip("/")
            elif self.provider == "zai":
                from hermes_cli.auth import _resolve_zai_base_url

                base_url = _resolve_zai_base_url(
                    api_key, pconfig.inference_base_url, env_url
                ).rstrip("/")
        elif self.provider == "custom":
            # Named custom provider (#67935): identity lives in config
            # (``providers.<name>`` / ``custom_providers``), the credential in
            # the env var it names via ``key_env``. Re-resolve through the
            # same config lookup the runtime resolver uses; entries without
            # ``key_env`` (inline ``api_key``, pool-backed) have no
            # env-sourced credential to watch.
            try:
                from hermes_cli.runtime_provider import _get_named_custom_provider
            except ImportError:
                return False
            custom_provider = _get_named_custom_provider(
                getattr(self, "requested_provider", "") or ""
            )
            if not custom_provider:
                return False
            key_env = str(custom_provider.get("key_env") or "").strip()
            if not key_env:
                return False
            api_key = get_env_prefer_dotenv(key_env).strip()
            if not api_key:
                return False
            # Custom providers pin their endpoint in config, not env — the
            # config base_url is both the resolved and the "default" base, so
            # only key edits are ever adopted here.
            default_base = str(custom_provider.get("base_url") or "").strip().rstrip("/")
            base_url = default_base
        else:
            return False

        if not base_url:
            return False

        resolved = (base_url, api_key)
        prev = getattr(self, "_env_creds_seen", None)
        current_base = (self.base_url or "").strip().rstrip("/")

        if prev is None:
            # First look — no baseline to diff against. Adopt only the
            # boot-default case (worker spawned before the user saved an
            # override); anything else is unattributable on turn one.
            adopt = current_base == default_base and not (
                base_url == current_base and api_key == self.api_key
            )
            # #79156: if the session already holds a pool-rotated key, do
            # not treat that divergence as a boot-time env adoption. First
            # look would otherwise stomp the rotated key with the env value
            # while leaving ``_credential_pool_entry_id`` on the fallback.
            if (
                adopt
                and api_key != self.api_key
                and getattr(self, "_credential_pool", None) is not None
                and getattr(self, "_credential_pool_entry_id", None)
            ):
                adopt = False
        else:
            # Env unchanged → no-op; any drift from self.* is rotation/
            # failover or config precedence — leave it alone. An edit is
            # only adopted while the session still runs on the registry
            # default or the previously-seen env value.
            adopt = (
                resolved != prev
                and current_base in {default_base, prev[0]}
                and not (base_url == current_base and api_key == self.api_key)
            )

        if not adopt:
            self._env_creds_seen = resolved
            return False

        from hermes_cli.route_identity import normalize_route_base_url

        route_changed = normalize_route_base_url(self.base_url) != normalize_route_base_url(
            base_url
        )
        prior_api_key = self.api_key
        prior_base_url = self.base_url
        prior_client_kwargs = dict(self._client_kwargs)

        self.api_key = api_key
        self.base_url = base_url
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # A base-url change moves the route: TLS material and default
        # headers derived from the old endpoint must be recomputed, exactly
        # as on credential-pool rotation.
        self._reapply_route_client_config(route_changed=route_changed)

        if not self._replace_primary_openai_client(reason="env_credential_refresh"):
            # Leave the baseline un-advanced so the unchanged edit is
            # retried next turn, and roll the agent back so its state keeps
            # matching the still-live old client.
            self.api_key = prior_api_key
            self.base_url = prior_base_url
            self._client_kwargs.clear()
            self._client_kwargs.update(prior_client_kwargs)
            return False

        # Rebind the pool entry id to the key we just adopted. Leaving a
        # stale id after a key rewrite makes mark_exhausted_and_rotate
        # quarantine the wrong credential on the next 429 (#79156).
        try:
            from agent.agent_runtime_helpers import sync_credential_pool_entry_id

            sync_credential_pool_entry_id(self)
        except Exception:
            logger.debug(
                "sync_credential_pool_entry_id after env refresh failed",
                exc_info=True,
            )

        self._env_creds_seen = resolved
        logger.info(
            "Applied updated .env credentials for %s: endpoint %s",
            self.provider,
            self.base_url,
        )
        return True

    def _try_refresh_vertex_client_credentials(self) -> bool:
        """Re-mint the Vertex OAuth2 access token and rebuild the OpenAI client.

        Vertex tokens live ~1 hour. On a long-lived agent (gateway session) a
        cached client's bearer token will expire mid-session, producing a 401.
        This re-resolves credentials via the adapter (which refreshes the
        underlying google-auth Credentials object when near expiry), swaps the
        new token into the client kwargs, and rebuilds the primary OpenAI
        client. Returns True when a usable token+base_url were obtained.
        """
        if self.api_mode != "chat_completions" or self.provider != "vertex":
            return False

        try:
            from agent.vertex_adapter import get_vertex_config

            token, base_url = get_vertex_config()
        except Exception as exc:
            logger.debug("Vertex credential refresh failed: %s", exc)
            return False

        if not isinstance(token, str) or not token.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = token.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="vertex_credential_refresh"):
            return False

        logger.info("Vertex AI OAuth token refreshed")
        return True

    def _try_refresh_copilot_client_credentials(self) -> bool:
        """Refresh Copilot credentials and rebuild the shared OpenAI client.

        The raw GitHub OAuth token (`gh auth token`) is usually stable, but the
        short-TTL *exchanged* IDE token minted from it is what Copilot actually
        authenticates — and it expires mid-session. A heavy/long turn whose
        request straddles that expiry gets a clean `401 IDE token expired:
        unauthorized: token expired`. Simply re-resolving the (unchanged) raw
        token and rebuilding the client leaves the SAME expired IDE token on the
        wire, so the retry 401s again and the turn aborts as non-retryable —
        only a gateway restart helped, because a cold process re-runs the
        exchange. Fix: force a fresh exchange (evict the cached exchanged JWT,
        then mint a new one) so the retry carries a valid IDE token. Mirrors the
        400 stale-credential recovery; the caller enforces the single-shot guard.
        """
        if not self._is_copilot_provider():
            return False

        try:
            from hermes_cli.copilot_auth import (
                resolve_copilot_token,
                get_copilot_api_token,
                evict_cached_exchanged_token,
            )

            new_token, token_source = resolve_copilot_token()
        except Exception as exc:
            logger.debug("Copilot credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False

        new_token = new_token.strip()

        # Force a fresh IDE-token exchange: the cached exchanged JWT is the thing
        # that expired ("401 IDE token expired"), so evict it and re-mint before
        # rebuilding the client. Fall back to the resolved (raw) token only if the
        # exchange itself is unavailable (network blip) — a client rebuild on the
        # raw token still clears stale client state and may recover on enterprise
        # seats where headers matter.
        try:
            evict_cached_exchanged_token(new_token)
            api_token, enterprise_base_url = get_copilot_api_token(new_token)
            if isinstance(api_token, str) and api_token.strip():
                new_token = api_token.strip()
                if enterprise_base_url:
                    self.base_url = enterprise_base_url.rstrip("/")
        except Exception as exc:
            logger.debug("Copilot 401 re-exchange failed, using resolved token: %s", exc)

        self.api_key = new_token
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(str(self.base_url or ""))

        if not self._replace_primary_openai_client(reason="copilot_credential_refresh"):
            return False

        logger.info("Copilot credentials refreshed from %s", token_source)
        return True

    def _try_recover_stale_copilot_credential(self) -> bool:
        """Force a fresh Copilot token exchange + client rebuild after a 400.

        Copilot surfaces a stale/degraded credential as a
        ``400 model_not_available_for_integrator`` /
        ``model_not_supported`` — NOT a clean 401 — so the normal 401 refresh
        path never fires. The most common trigger is a raw ``ghu_`` OAuth token
        that got seeded (and cached) when the startup token exchange degraded:
        the raw token routes the request to the restricted
        ``copilot-language-server`` integrator whose allowlist omits
        enterprise-only models (e.g. ``claude-opus-4.8``).

        Recovery = evict the poisoned cache entry, force a fresh exchange to
        mint the real ~437-char API token, re-apply the Copilot headers, and
        rebuild the shared client. Single-shot (guarded by the caller) so a
        genuinely unavailable model can't loop.
        """
        if not self._is_copilot_provider():
            return False

        try:
            from hermes_cli.copilot_auth import (
                resolve_copilot_token,
                get_copilot_api_token,
                evict_cached_exchanged_token,
            )

            raw_token, token_source = resolve_copilot_token()
            if not isinstance(raw_token, str) or not raw_token.strip():
                return False
            raw_token = raw_token.strip()

            # Drop any cached (possibly degraded/raw) exchanged token so the
            # next exchange hits the network and mints a fresh one.
            evict_cached_exchanged_token(raw_token)

            api_token, enterprise_base_url = get_copilot_api_token(raw_token)
        except Exception as exc:
            logger.debug("Copilot stale-credential recovery failed: %s", exc)
            return False

        if not isinstance(api_token, str) or not api_token.strip():
            return False

        # If the exchange STILL degraded to the raw token, a rebuild won't help
        # — don't burn the single-shot retry on an identical request.
        if api_token == raw_token and not enterprise_base_url:
            logger.warning(
                "Copilot stale-credential recovery: exchange still degraded to "
                "raw token; skipping retry (network/exchange endpoint unavailable)."
            )
            return False

        self.api_key = api_token.strip()
        if enterprise_base_url:
            self.base_url = enterprise_base_url.rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(str(self.base_url or ""))

        if not self._replace_primary_openai_client(reason="copilot_stale_credential_recovery"):
            return False

        logger.info("Copilot credentials re-exchanged after stale-credential 400 (source=%s)", token_source)
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
        if base_url_host_matches(_base, "azure.com"):
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

    def _apply_client_headers_for_base_url(
        self,
        base_url: str,
        *,
        apply_user_headers: bool = True,
    ) -> None:
        from agent.auxiliary_client import (
            _AI_GATEWAY_HEADERS,
            build_nvidia_nim_headers,
            build_or_headers,
        )

        if base_url_host_matches(base_url, "openrouter.ai"):
            self._client_kwargs["default_headers"] = build_or_headers()
        elif base_url_host_matches(base_url, "ai-gateway.vercel.sh"):
            self._client_kwargs["default_headers"] = dict(_AI_GATEWAY_HEADERS)
        elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
            self._client_kwargs["default_headers"] = build_nvidia_nim_headers(base_url)
        elif base_url_host_matches(base_url, "api.routermint.com"):
            self._client_kwargs["default_headers"] = _routermint_headers()
        elif base_url_host_matches(base_url, "githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif base_url_host_matches(base_url, "api.kimi.com"):
            from agent.auxiliary_client import _AI_GATEWAY_HEADERS
            self._client_kwargs["default_headers"] = dict(_AI_GATEWAY_HEADERS)
        elif base_url_host_matches(base_url, "portal.qwen.ai"):
            self._client_kwargs["default_headers"] = _qwen_portal_headers()
        elif base_url_host_matches(base_url, "chatgpt.com"):
            from agent.codex_headers import codex_cloudflare_headers
            self._client_kwargs["default_headers"] = codex_cloudflare_headers(
                self._client_kwargs.get("api_key", ""), base_url=base_url,
            )
        elif base_url_host_matches(base_url, "x.ai"):
            # Cover both provider=xai and provider=xai-oauth (api.x.ai).
            from tools.xai_http import hermes_xai_default_headers

            self._client_kwargs["default_headers"] = hermes_xai_default_headers()
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

        # User-configured overrides win over URL/profile defaults for the same
        # route. A credential swap to another endpoint must not inherit them.
        if apply_user_headers:
            self._apply_user_default_headers()

        # Per-provider extra HTTP headers (providers.<name>.extra_headers /
        # custom_providers[].extra_headers) — applied last so the most
        # specific config level survives credential swaps and rebuilds too.
        # SECURITY: values may carry credentials — never log them.
        if self.api_mode not in ("anthropic_messages", "bedrock_converse"):
            try:
                from hermes_cli.config import (
                    apply_custom_provider_extra_headers_to_client_kwargs,
                )

                apply_custom_provider_extra_headers_to_client_kwargs(
                    self._client_kwargs, base_url,
                )
            except Exception:
                logger.debug("custom-provider extra_headers skipped", exc_info=True)

    def _apply_user_default_headers(self) -> None:
        """Merge user-configured request headers onto the OpenAI client.

        Reads ``model.default_headers`` from config.yaml and merges it onto
        ``self._client_kwargs["default_headers"]``, with user values taking
        precedence over provider- and SDK-supplied defaults.

        This exists for ``custom`` OpenAI-compatible endpoints sitting behind
        a gateway/WAF that rejects the OpenAI Python SDK's identifying headers
        (``User-Agent: OpenAI/Python ...``, ``X-Stainless-*``). Setting e.g.
        ``model.default_headers: {User-Agent: curl/8.7.1}`` lets the request
        reach such an upstream instead of failing with an opaque 4xx/502 even
        though the same body works under ``curl``. (#40033)

        Delegates the config read + merge to
        ``agent.auxiliary_client._apply_user_default_headers`` so the main and
        auxiliary clients can never drift on precedence or value handling.

        No-op for Anthropic/Bedrock modes, which don't use the OpenAI client,
        and when no overrides are configured.
        """
        if self.api_mode in ("anthropic_messages", "bedrock_converse"):
            return
        from agent.auxiliary_client import (
            _apply_user_default_headers as _merge_user_headers,
        )
        merged = _merge_user_headers(self._client_kwargs.get("default_headers"))
        if merged:
            self._client_kwargs["default_headers"] = merged

    def _swap_credential(self, entry) -> None:
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url
        self._credential_pool_entry_id = getattr(entry, "id", None)
        from hermes_cli.route_identity import normalize_route_base_url

        route_changed = normalize_route_base_url(self.base_url) != normalize_route_base_url(
            runtime_base
        )

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
            self._anthropic_client = build_anthropic_client(
                runtime_key, self._anthropic_base_url,
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
            self._is_anthropic_oauth = _is_oauth_token(runtime_key) if self.provider == "anthropic" else False
            self.api_key = runtime_key
            self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._reapply_route_client_config(route_changed=route_changed)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _reapply_route_client_config(self, *, route_changed: bool) -> None:
        """Recompute route-derived client kwargs for the current ``self.base_url``.

        TLS material (``ssl_verify``/``ssl_ca_cert``) and default headers are
        derived from the endpoint, not the credential — any client rebuild
        that may have moved ``base_url`` must recompute them or the new
        endpoint inherits configuration computed for the old one. Shared by
        credential-pool rotation and the per-turn env refresh so the two
        paths cannot drift.
        """
        self._client_kwargs.pop("ssl_verify", None)
        self._client_kwargs.pop("ssl_ca_cert", None)
        try:
            from hermes_cli.config import (
                apply_custom_provider_tls_to_client_kwargs,
                get_compatible_custom_providers,
                load_config_readonly,
            )

            apply_custom_provider_tls_to_client_kwargs(
                self._client_kwargs,
                str(self.base_url or ""),
                get_compatible_custom_providers(load_config_readonly()),
            )
        except Exception:
            logger.debug(
                "custom-provider TLS resolution skipped on credential rotation",
                exc_info=True,
            )
        self._apply_client_headers_for_base_url(
            self.base_url,
            apply_user_headers=not route_changed,
        )

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        classified_reason: Optional[FailoverReason] = None,
        error_context: Optional[Dict[str, Any]] = None,
        billing_unverified: bool = False,
    ) -> tuple[bool, bool]:
        """Forwarder — see ``agent.agent_runtime_helpers.recover_with_credential_pool``."""
        from agent.agent_runtime_helpers import recover_with_credential_pool
        return recover_with_credential_pool(self, status_code=status_code, has_retried_429=has_retried_429, classified_reason=classified_reason, error_context=error_context, billing_unverified=billing_unverified)

    def _credential_pool_may_recover_rate_limit(self) -> bool:
        """Whether a rate-limit retry should wait for same-provider credentials."""
        pool = self._credential_pool
        if pool is None:
            return False
        return pool.has_available()

    def _anthropic_messages_create(self, api_kwargs: dict, *, client: Any = None):
        # When a request-local client is supplied it was already credential-
        # refreshed in ``_create_request_anthropic_client``; only the shared
        # fallback path refreshes here.
        if client is None and self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        # Defensive: strip Responses-only kwargs that can leak in under an
        # api_mode-flip race (the Anthropic SDK raises a non-retryable
        # TypeError on them). See #31673.
        from agent.anthropic_adapter import create_anthropic_message
        return create_anthropic_message(
            client or self._anthropic_client,
            api_kwargs,
            log_prefix=getattr(self, "log_prefix", ""),
            prefer_stream=not bool(getattr(self, "_disable_streaming", False)),
            # Rate-limit + credits state live in response headers, which the
            # parsed Message drops. No-ops on providers that don't send the
            # matching header families (x-ratelimit-* / x-nous-credits-*).
            on_response=self._capture_anthropic_response_headers,
        )

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
        """Forwarder — see ``agent.chat_completion_helpers.interruptible_api_call``."""
        from agent.chat_completion_helpers import interruptible_api_call
        return interruptible_api_call(self, api_kwargs)

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
        # Single-writer guard (#65991): a superseded stream must not pollute the
        # turn's accumulated text (which also feeds the interim-visible-text
        # de-dup comparison), even when a caller reaches this directly (the
        # tool-suppressed content path) rather than through _fire_stream_delta.
        if self._stream_writer_superseded():
            return
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
        # Prefix match (not exact equality): the final response may be the
        # streamed text plus a trailing delta, or the stream may have been
        # partial when the verify nudge fired.  In both cases the streamed
        # content is a prefix of the final — that's enough to mark it
        # previewed (fails safe to a benign duplicate, never loses text).
        # The reverse direction (streamed longer than final) is NOT matched:
        # that could suppress a needed resend in the gateway path where
        # already_streamed=True calls on_segment_break() instead of
        # on_commentary() (#65919 review).
        return bool(streamed) and visible_content.startswith(streamed)

    def _extract_codex_interim_visible_parts(
        self,
        assistant_msg: Dict[str, Any],
    ) -> List[str]:
        """Extract visible Codex commentary as one string per message item.

        Codex Responses can keep user-facing mid-turn narration as structured
        ``phase=commentary`` message items while final answer text remains in
        assistant ``content``.  Non-streaming gateway surfaces need that
        commentary through the interim assistant callback before tool calls run.
        ``phase=analysis`` remains hidden because it is provider scratchpad.
        """
        if not getattr(self, "show_commentary", True):
            # display.show_commentary=false — commentary stays on the
            # reasoning channel (pre-commentary-channel behavior).
            return []
        items = assistant_msg.get("codex_message_items")
        if not isinstance(items, list):
            return []

        messages: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            phase = item.get("phase")
            if not isinstance(phase, str) or phase.strip().lower() != "commentary":
                continue
            content_parts = item.get("content")
            if not isinstance(content_parts, list):
                continue
            item_parts: List[str] = []
            for part in content_parts:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "output_text":
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    item_parts.append(text)
            visible = "".join(item_parts).strip()
            if visible:
                visible = self._strip_think_blocks(visible).strip()
                visible = redact_sensitive_text(visible)
            if visible:
                messages.append(visible)
        return messages

    def _extract_codex_interim_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
        """Extract all visible Codex commentary for comparison/fallback."""
        return "\n\n".join(
            self._extract_codex_interim_visible_parts(assistant_msg)
        ).strip()

    def _interim_assistant_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
        """Return the exact assistant text eligible for interim delivery.

        Prefer structured Codex commentary over top-level content. A Codex
        response can contain both commentary and a partial/final-answer message
        while tools are still pending; treating top-level content as progress
        in that shape leaks the answer before the tool call runs.

        Content may be a string or a structured parts list (e.g. after vision
        turns or context compaction), so flatten it before stripping reasoning.
        """
        visible = self._extract_codex_interim_visible_text(assistant_msg)
        if visible:
            return visible
        content = assistant_msg.get("content")
        return self._strip_think_blocks(flatten_message_text(content)).strip()

    def _interim_text_was_delivered(self, text: str) -> bool:
        normalized = self._normalize_interim_visible_text(text)
        if not normalized:
            return False
        return normalized in getattr(self, "_delivered_interim_texts", set())

    def _record_delivered_interim_text(self, text: str) -> None:
        normalized = self._normalize_interim_visible_text(text)
        if normalized:
            delivered = getattr(self, "_delivered_interim_texts", None)
            if not isinstance(delivered, set):
                delivered = set()
                self._delivered_interim_texts = delivered
            delivered.add(normalized)

    def _fire_streamed_codex_commentary(self, text: str) -> None:
        """Deliver a completed live Codex commentary message immediately."""
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None or not isinstance(text, str):
            return
        visible = self._strip_think_blocks(text).strip()
        if visible:
            visible = redact_sensitive_text(visible)
        if not visible or visible == "(empty)" or self._interim_text_was_delivered(visible):
            return
        try:
            cb(visible, already_streamed=False)
            self._record_delivered_interim_text(visible)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _emit_interim_assistant_message(
        self, assistant_msg: Dict[str, Any]
    ) -> None:
        """Surface a real mid-turn assistant commentary message to the UI layer.

        Does NOT set ``_response_was_previewed`` — that flag means "the final
        response was already shown to the user," but this helper is called for
        ordinary tool-call narration, intermediate acknowledgements, and
        verification candidates alike. Setting it here would cause the CLI to
        suppress a *different* final summary (e.g. from ``_handle_max_iterations``)
        when the only streamed text was unrelated mid-turn commentary. (#65919
        review: response-loss blocker)
        """
        if not isinstance(assistant_msg, dict):
            return
        commentary_parts = self._extract_codex_interim_visible_parts(assistant_msg)
        undelivered_parts: List[str] = []
        pending_keys: set[str] = set()
        for part in commentary_parts:
            key = self._normalize_interim_visible_text(part)
            if (
                not key
                or key in pending_keys
                or self._interim_text_was_delivered(part)
            ):
                continue
            pending_keys.add(key)
            undelivered_parts.append(part)
        visible = (
            "\n\n".join(undelivered_parts).strip()
            if commentary_parts
            else self._interim_assistant_visible_text(assistant_msg)
        )
        if (
            not visible
            or visible == "(empty)"
            or self._interim_text_was_delivered(visible)
        ):
            return
        already_streamed = self._interim_content_was_streamed(visible)
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(
                "on_interim_message",
                turn_id=getattr(self, "_current_turn_id", "") or "",
                iteration=int(getattr(self, "_api_call_count", 0) or 0),
                session_id=self.session_id or "",
                model=self.model or "",
                provider=self.provider or "",
                surface=self.platform or "cli",
                text=visible,
                already_streamed=already_streamed,
            )
        except Exception:
            logger.debug("on_interim_message plugin hook enqueue failed", exc_info=True)
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None:
            return
        try:
            cb(visible, already_streamed=already_streamed)
            if undelivered_parts:
                for part in undelivered_parts:
                    self._record_delivered_interim_text(part)
            else:
                self._record_delivered_interim_text(visible)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _ensure_stream_writer_state(self) -> None:
        """Lazily create the single-writer guard fields (#65991).

        The fields are normally set in ``agent_init``, but agents constructed
        via ``AIAgent.__new__`` (test doubles, legacy/partially-initialized
        instances) skip that path. Claiming/checking the writer must not crash
        those agents, so initialize the fields on first use.
        """
        if getattr(self, "_stream_writer_lock", None) is None:
            self._stream_writer_lock = threading.Lock()
        if not hasattr(self, "_stream_writer_token"):
            self._stream_writer_token = 0
        if getattr(self, "_stream_writer_tls", None) is None:
            self._stream_writer_tls = threading.local()
        if not hasattr(self, "_stream_writer_dropped"):
            self._stream_writer_dropped = 0

    def _claim_stream_writer(self) -> int:
        """Claim exclusive ownership of the streaming delta sink for the calling
        stream attempt and return its monotonic writer token (#65991).

        Every streaming attempt (each provider path, each retry) calls this
        right before it begins consuming its stream. Claiming bumps the shared
        token, so any earlier attempt still alive on another thread is
        immediately superseded: its cached token no longer matches and the sink
        fences its late chunks out. The token is stored per-thread, so a thread
        that never claimed (a non-streaming caller) is never treated as a
        writer and can never be fenced.
        """
        self._ensure_stream_writer_state()
        with self._stream_writer_lock:
            self._stream_writer_token += 1
            token = self._stream_writer_token
        self._stream_writer_tls.token = token
        return token

    def _stream_writer_is_current(self, token: int) -> bool:
        """True when ``token`` (from a prior _claim_stream_writer) is still the
        active writer — i.e. no newer stream attempt has claimed the sink since
        (#65991). Lets a stream loop bail out the instant it is superseded."""
        return token == getattr(self, "_stream_writer_token", token)

    def _stream_writer_superseded(self) -> bool:
        """True when the calling thread claimed the delta sink but a newer
        stream attempt has since claimed it — i.e. this thread is a stale
        writer whose chunks must be dropped (#65991).

        A thread that never claimed (``token is None``) is not a writer and is
        never reported as superseded, so non-streaming delta callers are
        unaffected.
        """
        tls = getattr(self, "_stream_writer_tls", None)
        token = getattr(tls, "token", None) if tls is not None else None
        if token is None:
            return False
        return token != getattr(self, "_stream_writer_token", token)

    def _note_dropped_stream_writer(self, where: str) -> None:
        """Record + log that a superseded stream's delta was discarded."""
        try:
            self._stream_writer_dropped = int(getattr(self, "_stream_writer_dropped", 0)) + 1
        except Exception:
            self._stream_writer_dropped = 1
        # Log sparsely (first drop, then powers of two) so a chatty superseded
        # stream can't flood the log, but a real provider problem is still
        # visible. A silent discard would hide genuine failures.
        _n = self._stream_writer_dropped
        if _n == 1 or (_n & (_n - 1)) == 0:
            logger.warning(
                "Dropped delta from a superseded stream writer at %s "
                "(discarded=%d this turn) — a stale stream tried to write into "
                "the turn after a retry superseded it.",
                where, _n,
            )

    def _stream_hook_base_payload(self) -> Dict[str, Any]:
        return {
            "turn_id": getattr(self, "_current_turn_id", "") or "",
            "iteration": int(getattr(self, "_api_call_count", 0) or 0),
            "session_id": self.session_id or "",
            "model": self.model or "",
            "provider": self.provider or "",
            "surface": self.platform or "cli",
        }

    def _emit_stream_start(self) -> None:
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook("on_stream_start", **self._stream_hook_base_payload())
        except Exception:
            logger.debug("on_stream_start plugin hook enqueue failed", exc_info=True)

    def _emit_stream_end(self, *, final_text: str, finished: bool, error: str | None) -> None:
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(
                "on_stream_end",
                **self._stream_hook_base_payload(),
                final_text=final_text,
                finished=finished,
                error=error,
            )
        except Exception:
            logger.debug("on_stream_end plugin hook enqueue failed", exc_info=True)

    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        # Single-writer guard (#65991): a superseded stream must not interleave
        # its tokens into the turn alongside the retry that replaced it.
        if self._stream_writer_superseded():
            self._note_dropped_stream_writer("_fire_stream_delta")
            return
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
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(
                "on_stream_delta",
                **self._stream_hook_base_payload(),
                delta=text,
                kind="text",
            )
        except Exception:
            logger.debug("on_stream_delta plugin hook enqueue failed", exc_info=True)
        if delivered:
            self._record_streamed_assistant_text(text)

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered."""
        # Single-writer guard (#65991): fence out a superseded stream's
        # reasoning deltas the same way as content deltas.
        if self._stream_writer_superseded():
            self._note_dropped_stream_writer("_fire_reasoning_delta")
            return
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook, stream_reasoning_deltas_enabled

            if stream_reasoning_deltas_enabled():
                enqueue_plugin_stream_hook(
                    "on_stream_delta",
                    **self._stream_hook_base_payload(),
                    delta=text,
                    kind="reasoning",
                )
        except Exception:
            logger.debug("reasoning on_stream_delta plugin hook enqueue failed", exc_info=True)

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
        try:
            from agent.plugin_stream_hooks import has_stream_observer_hooks

            if has_stream_observer_hooks():
                return True
        except Exception:
            logger.debug("plugin stream hook consumer check failed", exc_info=True)
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None
    ):
        """Forwarder — see ``agent.chat_completion_helpers.interruptible_streaming_api_call``."""
        from agent.chat_completion_helpers import interruptible_streaming_api_call
        return interruptible_streaming_api_call(self, api_kwargs, on_first_delta=on_first_delta)

    def _try_activate_fallback(self, reason: "FailoverReason | None" = None) -> bool:
        """Forwarder — see ``agent.chat_completion_helpers.try_activate_fallback``."""
        from agent.chat_completion_helpers import try_activate_fallback
        return try_activate_fallback(self, reason)

    def _has_pending_fallback(self) -> bool:
        """Whether a fallback provider is actually available to switch to.

        Used to gate user-facing "trying fallback..." status so we don't
        announce a fallback that will never be attempted (the user has no
        fallback chain configured).  Mirrors the early-return guard in
        ``try_activate_fallback`` (#35314, #17446).
        """
        chain = getattr(self, "_fallback_chain", None) or []
        index = getattr(self, "_fallback_index", 0)
        return index < len(chain)

    # ── Per-turn primary restoration ─────────────────────────────────────

    def _restore_primary_runtime(self) -> bool:
        """Forwarder — see ``agent.agent_runtime_helpers.restore_primary_runtime``."""
        from agent.agent_runtime_helpers import restore_primary_runtime
        return restore_primary_runtime(self)

    def _try_recover_primary_transport(
        self, api_error: Exception, *, retry_count: int, max_retries: int,
    ) -> bool:
        """Forwarder — see ``agent.agent_runtime_helpers.try_recover_primary_transport``."""
        from agent.agent_runtime_helpers import try_recover_primary_transport
        return try_recover_primary_transport(self, api_error, retry_count=retry_count, max_retries=max_retries)

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    # 20 MB base64 ≈ 15 MB decoded image — generous but prevents OOM from an
    # oversized data: URL (a 100 MB+ payload creates ~275 MB of memory pressure,
    # and gateway users sharing the same process can trivially OOM it).
    _MAX_DATA_URL_BASE64_BYTES = 20 * 1024 * 1024

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        if len(data) > AIAgent._MAX_DATA_URL_BASE64_BYTES:
            logger.warning(
                "data-URL payload too large (%d bytes), skipping", len(data)
            )
            return "", None
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

        Resolution order (see ``agent.image_routing._supports_vision_override``):
          1. ``model.supports_vision`` (top-level, single-model shortcut)
          2. ``providers.<provider>.models.<model>.supports_vision``
          3. models.dev capability lookup
        Custom/local models absent from models.dev would otherwise be
        misclassified as non-vision and have their images stripped.
        """
        try:
            from hermes_cli.config import load_config
            from agent.image_routing import _lookup_supports_vision
            cfg = load_config()
            provider = (getattr(self, "provider", "") or "").strip()
            model = (getattr(self, "model", "") or "").strip()
            return _lookup_supports_vision(provider, model, cfg) is True
        except Exception:
            return False

    def _provider_supports_vision_tool_messages(self) -> bool:
        """Return True if the active provider accepts list-type tool content.

        Some providers (e.g. Xiaomi MiMo) support multimodal user messages
        but reject list-type tool message content with 400 errors.  This
        checks the provider profile's ``supports_vision_tool_messages`` field.
        """
        try:
            from providers import get_provider_profile
            provider = (getattr(self, "provider", "") or "").strip()
            profile = get_provider_profile(provider)
            if profile is not None:
                return getattr(profile, "supports_vision_tool_messages", True)
        except Exception:
            pass
        return True  # default: assume compatible

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

    def _tool_result_content_for_active_model(self, tool_name: str, result: Any) -> Any:
        """Return the tool message content that is safe for the active model.

        Multimodal tool results normally unwrap to OpenAI-style content parts so
        vision-capable models can inspect screenshots.  Text-only providers must
        not receive those image parts, because a rejected tool result becomes
        part of the canonical history and can make the next user turn fail before
        the agent has a chance to recover.
        """
        if not _is_multimodal_tool_result(result):
            return result

        content = result.get("content") or []
        if not self._content_has_image_parts(content):
            return content

        if self._model_supports_vision():
            # Vision-capable on paper — but if the provider rejects list-type
            # tool content (e.g. Xiaomi MiMo's 400 "text is not set"), or if
            # we've already learned this lesson in-session, short-circuit to
            # a text summary so we don't burn a round-trip relearning it.
            if not self._provider_supports_vision_tool_messages():
                logger.debug(
                    "Tool %s: provider %s does not accept list-type tool "
                    "content — sending text summary",
                    tool_name, getattr(self, "provider", ""),
                )
                return _multimodal_text_summary(result)
            key = (
                (getattr(self, "provider", "") or "").strip().lower(),
                (getattr(self, "model", "") or "").strip(),
            )
            no_list = getattr(self, "_no_list_tool_content_models", None)
            if no_list and key in no_list:
                logger.debug(
                    "Tool %s: model %s/%s known to reject list-type tool "
                    "content this session — sending text summary",
                    tool_name, key[0], key[1],
                )
                return _multimodal_text_summary(result)
            return content

        summary = _multimodal_text_summary(result)
        if tool_name == "computer_use":
            return json.dumps({
                "error": (
                    "computer_use returned screenshot/image content, but the active "
                    "model/provider does not support image input. Switch to a "
                    "vision-capable model for desktop computer use, or use browser "
                    "tools for browser tasks."
                ),
                "text_summary": summary,
            })

        logger.warning(
            "Tool %s returned image content for non-vision model %s/%s; "
            "falling back to text summary",
            tool_name,
            self.provider,
            self.model,
        )
        return summary

    def _try_shrink_image_parts_in_messages(
        self,
        api_messages: list,
        *,
        max_dimension: int = 8000,
    ) -> bool:
        """Forwarder — see ``agent.conversation_compression.try_shrink_image_parts_in_messages``."""
        from agent.conversation_compression import try_shrink_image_parts_in_messages
        return try_shrink_image_parts_in_messages(
            api_messages,
            max_dimension=max_dimension,
        )

    def _try_strip_image_parts_from_tool_messages(
        self,
        api_messages: list,
        *,
        remember_model: bool = True,
    ) -> bool:
        """Downgrade list-type tool messages to text summaries in-place.

        Recovery path for providers that reject list-type tool message content
        (e.g. Xiaomi MiMo's 400 "text is not set"; see issue #27344).  Walks
        ``api_messages`` for any ``role: "tool"`` message whose ``content`` is
        a list containing image parts, replaces the content with the existing
        text part(s) (or a minimal placeholder if none survive), and by default
        records the active (provider, model) in
        ``self._no_list_tool_content_models`` so subsequent
        ``_tool_result_content_for_active_model`` calls in this session
        preemptively downgrade screenshots without a round-trip.

        413 payload-size recovery passes ``remember_model=False`` because that
        error means this request body was too large, not that the provider/model
        rejects list-type tool content in general.

        Returns True when at least one tool message was downgraded — the
        caller (the 400 recovery branch in ``agent.conversation_loop``) uses
        this to decide whether to retry the API call with the modified
        history or surface the original error.
        """
        if not isinstance(api_messages, list):
            return False

        if remember_model:
            # Record (provider, model) so we don't relearn this lesson.
            key = (
                (getattr(self, "provider", "") or "").strip().lower(),
                (getattr(self, "model", "") or "").strip(),
            )
            if not hasattr(self, "_no_list_tool_content_models"):
                self._no_list_tool_content_models = set()
            if key[1]:  # only record when we actually have a model id
                self._no_list_tool_content_models.add(key)

        changed = False
        for msg in api_messages:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            # Salvage any text parts so the model still sees some signal.
            text_parts: List[str] = []
            had_image = False
            for part in content:
                if not isinstance(part, dict):
                    if isinstance(part, str) and part.strip():
                        text_parts.append(part.strip())
                    continue
                ptype = part.get("type")
                if ptype == "image_url" or ptype == "input_image":
                    had_image = True
                    continue
                if ptype in {"text", "input_text"}:
                    text = str(part.get("text") or "").strip()
                    if text:
                        text_parts.append(text)

            if not had_image:
                # List-type content but no image parts — leave alone (some
                # providers reject ANY list content, but stripping a
                # text-only list doesn't reduce ambiguity; let the caller
                # surface the original error if this turns out to be the
                # case).
                continue

            if text_parts:
                msg["content"] = "\n\n".join(text_parts)
            else:
                msg["content"] = (
                    "[image content removed — provider does not accept "
                    "list-type tool message content]"
                )
            changed = True

        return changed

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
            "xiaomi", "vertex",
        }:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        host = base_url_hostname(base)
        return (
            "dashscope" in host
            or base_url_host_matches(base, "aliyuncs.com")
            or "minimax" in host
            or (base_url_host_matches(base, "opencode.ai") and "/zen/" in base)
            or base_url_host_matches(base, "bigmodel.cn")
            or base_url_host_matches(base, "xiaomimimo.com")
            # Vertex AI OpenAI-compat endpoint — Gemini model ids keep dots
            # (e.g. google/gemini-3.5-flash); the hyphenated form is wrong.
            or base_url_host_matches(base, "aiplatform.googleapis.com")
            # AWS Bedrock runtime endpoints — defense-in-depth when
            # ``provider`` is unset but ``base_url`` still names Bedrock.
            or host.startswith("bedrock-runtime.")
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

    def _build_api_kwargs(self, api_messages: list, tools_for_api: Optional[list] = None) -> dict:
        """Forwarder — see ``agent.chat_completion_helpers.build_api_kwargs``."""
        from agent.chat_completion_helpers import build_api_kwargs
        return build_api_kwargs(self, api_messages, tools_for_api=tools_for_api)

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
            or base_url_host_matches(self._base_url_lower, "githubcopilot.com")
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
        # Ollama Cloud (and any Ollama-compatible server): the native
        # /api/show capabilities list is authoritative — emit reasoning_effort
        # only for models that declare the "thinking" capability. deepseek-v4
        # has it; gemma3 / qwen3-coder don't. Cached per (model, base_url).
        if base_url_host_matches(self._base_url_lower, "ollama.com"):
            return self._ollama_supports_thinking_cached()
        if not self._is_openrouter_url():
            return False
        if base_url_host_matches(self._base_url_lower, "api.mistral.ai"):
            return False

        model = (self.model or "").lower()
        # Live-catalog metadata first (ported from
        # PrimeIntellect-ai/prime-agent#1258): OpenRouter's /v1/models entries
        # advertise reasoning support via supported_parameters + a reasoning
        # object, which covers every routed vendor without a hand-maintained
        # prefix list. The static prefix allowlist below repeatedly went
        # stale one vendor at a time (nvidia/ missing → #75386; same class
        # as tencent/, xiaomi/ additions before it) — metadata makes new
        # vendors work without a code change. One catalog fetch per process,
        # cached; unknown (catalog unreachable / unlisted model) falls back
        # to the static list.
        try:
            from hermes_cli.models import (
                openrouter_model_reasoning_capabilities,
                warm_openrouter_reasoning_caps_async,
            )
            caps = openrouter_model_reasoning_capabilities(self.model)
            if caps is None:
                # Cache cold (no picker run this process) — warm it in the
                # background so subsequent turns get metadata; never block
                # this turn on HTTP.
                warm_openrouter_reasoning_caps_async()
        except Exception:
            caps = None
        if caps is not None:
            return bool(caps.get("supports_reasoning"))
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "google/gemma-4",
            "qwen/qwen3",
            "tencent/hy",
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

    def _ollama_supports_thinking_cached(self) -> bool:
        """Probe Ollama's ``/api/show`` capabilities once per (model, base_url).

        Returns True only when the model declares the ``thinking`` capability.
        Caching mirrors the LM Studio probe: a True/False result is permanent
        (capabilities don't change), while a probe failure (None) is cached
        with a 60-second TTL so a transient outage doesn't suppress reasoning
        for the rest of the session but also doesn't round-trip every turn.
        """
        import time as _time

        cache = getattr(self, "_ollama_thinking_cache", None)
        if cache is None:
            cache = self._ollama_thinking_cache = {}
        key = (self.model, self.base_url)
        cached = cache.get(key)
        if cached is not None:
            supported, ts = cached
            # Definitive True/False → permanent. Unknown (None) → 60s TTL.
            if supported is not None or (_time.monotonic() - ts) < 60:
                return bool(supported)
        try:
            from hermes_cli.models import ollama_model_supports_thinking
            supported = ollama_model_supports_thinking(
                self.model, self.base_url, getattr(self, "api_key", "")
            )
        except Exception:
            supported = None
        cache[key] = (supported, _time.monotonic())
        return bool(supported)

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

        if requested_effort == "xhigh" and "xhigh" not in supported_efforts and "high" in supported_efforts:
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
        """Forwarder — see ``agent.chat_completion_helpers.build_assistant_message``."""
        from agent.chat_completion_helpers import build_assistant_message
        return build_assistant_message(self, assistant_message, finish_reason)

    def _needs_thinking_reasoning_pad(self) -> bool:
        """Return True when the active provider enforces reasoning_content echo-back.

        DeepSeek v4 thinking and Kimi / Moonshot thinking both reject replays
        of assistant tool-call messages that omit ``reasoning_content`` (refs
        #15250, #17400). Xiaomi MiMo thinking mode has the same requirement.

        Result cached on the AIAgent instance keyed by (provider, model,
        base_url); invalidated whenever ``switch_model()`` /
        ``_try_activate_fallback()`` mutate any of those. This is hot — the
        agent loop hits ~16 invocations per turn, each of which would
        otherwise re-run ~5 ``base_url_host_matches`` (and therefore
        ``urlparse``) calls under it. Caching drops the per-turn cost from
        ~5us × 16 = ~80us to <1us.
        """
        key = (self.provider, self.model, getattr(self, "_base_url_lower", self.base_url))
        cached = getattr(self, "_thinking_pad_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = (
            self._needs_deepseek_tool_reasoning()
            or self._needs_kimi_tool_reasoning()
            or self._needs_mimo_tool_reasoning()
            or self._reasoning_echo_opt_in()
        )
        self._thinking_pad_cache = (key, result)
        return result

    def _reasoning_echo_opt_in(self) -> bool:
        """Return True when the user has opted in to ``reasoning_content``
        echo-back for the *current* provider via config.

        This covers custom providers and OpenAI-compatible gateways that
        proxy thinking-mode models (e.g. a reverse proxy fronting Kimi K3
        or GLM-5.2) but are not matched by the built-in host-based
        ``_REASONING_ECHO_RULES`` (DeepSeek / Kimi / MiMo).

        The flag is per-active-provider:

        * **Primary** — read from ``model.reasoning_echo`` in config.yaml
          at agent init and on ``switch_model()``.
        * **Fallback** — set by ``try_activate_fallback()`` from the
          fallback entry's ``reasoning_echo`` field.
        * **Restore** — ``restore_primary_runtime()`` copies the snapshot
          saved by ``switch_model()``.

        Unlike a global toggle, this flag travels with the active
        provider, so falling back to a strict provider (Mistral, Groq,
        Cerebras) correctly strips ``reasoning_content`` even when the
        primary had the flag enabled.
        """
        return bool(getattr(self, "_reasoning_echo_flag", False))

    @staticmethod
    def _read_reasoning_echo_from_config() -> bool:
        """Read ``model.reasoning_echo`` from config; False on any error."""
        try:
            from hermes_cli.config import load_config_readonly
            return bool(
                (load_config_readonly().get("model") or {}).get("reasoning_echo")
            )
        except Exception:
            return False

    def _needs_kimi_tool_reasoning(self) -> bool:
        """Return True when the current provider is Kimi / Moonshot thinking mode.

        Kimi ``/coding`` and Moonshot thinking mode both require
        ``reasoning_content`` on every assistant tool-call message; omitting
        it causes the next replay to fail with HTTP 400.

        Detection is host-driven, not model-name-driven: aggregators like
        OpenRouter that re-export Kimi/Moonshot models speak their own
        protocol and reject ``reasoning_content`` echoes. We only enable the
        kimi-reasoning replay when the request actually targets a
        kimi/moonshot endpoint or the dedicated kimi-coding provider.

        Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "kimi", self.provider, None, self.base_url
        )

    def _needs_deepseek_tool_reasoning(self) -> bool:
        """Return True when the current provider is DeepSeek thinking mode.

        DeepSeek V4 thinking mode requires ``reasoning_content`` on every
        assistant tool-call turn; omitting it causes HTTP 400 when the
        message is replayed in a subsequent API request (#15250).

        Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "deepseek", (self.provider or "").lower(), self.model, self.base_url
        )

    def _needs_mimo_tool_reasoning(self) -> bool:
        """Return True when the current provider is Xiaomi MiMo thinking mode.

        MiMo thinking mode requires ``reasoning_content`` on every assistant
        tool-call message when replaying history; omitting it causes HTTP 400.
        Refs: https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/passing-back-reasoning_content

        Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "mimo", (self.provider or "").lower(), self.model, self.base_url
        )

    def _copy_reasoning_content_for_api(self, source_msg: dict, api_msg: dict) -> None:
        """Forwarder — see ``agent.agent_runtime_helpers.copy_reasoning_content_for_api``."""
        from agent.agent_runtime_helpers import copy_reasoning_content_for_api
        return copy_reasoning_content_for_api(self, source_msg, api_msg)

    def _reapply_reasoning_echo_for_provider(self, api_messages: list) -> int:
        """Forwarder — see ``agent.agent_runtime_helpers.reapply_reasoning_echo_for_provider``."""
        from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider
        return reapply_reasoning_echo_for_provider(self, api_messages)

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict, model: "str | None" = None) -> dict:
        """Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
        validate the Chat Completions schema and reject unknown fields (call_id,
        response_item_id) with 400 or 422 errors. These fields are preserved in
        the internal message history — this method only modifies the outgoing
        API copy.

        ``extra_content`` (Gemini thought_signature) is also stripped — strict
        providers reject it with "Extra inputs are not permitted" — UNLESS the
        outgoing ``model`` is itself Gemini-family, in which case it must be
        replayed (Gemini 3 thinking models 400 without it). Defaults to
        stripping when no model is supplied.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).

        Fields stripped: call_id, response_item_id, extra_content (model-gated)
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        from agent.transports.chat_completions import _model_consumes_thought_signature
        _STRIP_KEYS = {"call_id", "response_item_id"}
        if not _model_consumes_thought_signature(model):
            _STRIP_KEYS = _STRIP_KEYS | {"extra_content"}
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
        cursor=None,
    ) -> int:
        """Forwarder — see ``agent.agent_runtime_helpers.sanitize_tool_call_arguments``."""
        from agent.agent_runtime_helpers import sanitize_tool_call_arguments
        return sanitize_tool_call_arguments(
            messages, logger=logger, session_id=session_id, cursor=cursor
        )

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

    def _compress_context(
        self,
        messages: list,
        system_message: str,
        *,
        approx_tokens: int = None,
        task_id: str = "default",
        focus_topic: str = None,
        force: bool = False,
        defer_context_engine_notification: bool = False,
        commit_fence=None,
    ) -> tuple:
        """Forwarder — see ``agent.conversation_compression.compress_context``.

        ``force=True`` is passed by the manual ``/compress`` slash command
        so users can bypass the summary-failure cooldown after an
        auto-compress abort.  Auto-compress callers use the default
        ``force=False``.
        """
        # Per-attempt signal consumed by turn-start preflight (#98424) and the
        # in-loop pre-API/overflow consumers. A stalled compression must not
        # be mistaken for a structural no-op and followed by the oversized
        # provider request it was meant to prevent. The typed helper upgrades
        # the simple attribute to thread-local state guarded by a per-agent
        # lock so overlapping automatic/manual entrypoints cannot clobber each
        # other's outcome (#98741).
        from agent.conversation_compression import (
            CompressionCommitFence,
            compress_context,
            mark_context_compression_timed_out,
            reset_context_compression_timeout_outcome,
            resolve_context_compression_timeouts,
            run_compress_context_with_progress_timeout,
        )
        reset_context_compression_timeout_outcome(self)
        from agent.portal_tags import (
            get_affinity_scope,
            get_conversation_context,
            reset_affinity_scope,
            reset_conversation_context,
            set_affinity_scope,
            set_conversation_context,
        )
        from agent.prompt_cache_scope import declared_conversation_scope_safe
        # Out-of-turn compaction entry points — ``/compact`` (cli.py), the
        # gateway ``/compress`` command and its hygiene sweep (both of which
        # build a throwaway agent), and partial head compression — call this
        # forwarder directly, outside ``run_conversation``'s ambient scope.
        # With nothing ambient the summarizer's auxiliary call carries no
        # conversation tag and no Portal sticky key, so it routes independently
        # of the conversation it belongs to. Publish the root here as a
        # fallback; in-turn callers already have it set to the same value, so
        # this is a no-op for them.
        #
        # Note this does NOT keep the compaction turn's own prompt cache warm:
        # compaction replaces the history with a summary and rebuilds the
        # system prompt, so that request is a cold write on any endpoint. What
        # it buys is the turns AFTER compaction reading the cache it wrote.
        token = None
        if get_conversation_context() is None:
            root = self._conversation_root_id()
            if root:
                token = set_conversation_context(root)
        # Same fallback for the ROUTING scope: out-of-turn compaction would
        # otherwise send the summarizer's call with no sticky key at all, or
        # (worse, on a per-response host) with a key that no longer matches
        # the conversation it is compacting. Only set when the host declared
        # one — unset keeps the pre-#96811 conversation-id fallback.
        affinity_token = None
        if get_affinity_scope() is None:
            declared = declared_conversation_scope_safe(self)
            if declared:
                affinity_token = set_affinity_scope(declared)
        # Every AIAgent compression has a fence, including ordinary in-turn and
        # manual paths. hard_interrupt() uses this exact instance to serialize
        # cancel admission against begin_commit().
        active_fence = commit_fence or CompressionCommitFence()
        # A single agent can receive overlapping automatic/manual entrypoints.
        # Serialize fence publication so a waiter cannot replace the fence of
        # the attempt currently generating/committing a summary.
        fence_registration_lock = vars(self).setdefault(
            "_compression_commit_fence_lock", threading.RLock()
        )
        with fence_registration_lock:
            missing_fence = object()
            previous_fence = vars(self).get(
                "_active_compression_commit_fence", missing_fence
            )
            self._active_compression_commit_fence = active_fence
        try:
            def _run(fence=None, target_messages=None):
                return compress_context(
                    self,
                    target_messages if target_messages is not None else messages,
                    system_message,
                    approx_tokens=approx_tokens, task_id=task_id,
                    focus_topic=focus_topic,
                    force=force,
                    defer_context_engine_notification=(
                        defer_context_engine_notification
                    ),
                    commit_fence=fence,
                )

            # Callers that already own a progress-aware wait (gateway session
            # hygiene) pass commit_fence and must not be double-wrapped.
            direct_path = commit_fence is not None
            idle_timeout = total_ceiling = None
            if not direct_path:
                idle_timeout, total_ceiling = resolve_context_compression_timeouts()
                if idle_timeout <= 0:
                    direct_path = True

            if direct_path:
                result = _run(active_fence)
            else:
                def _snapshot_worker(fence=None):
                    # #76354 review F3: the pooled worker must NEVER share the
                    # caller's live transcript. Plugin/legacy context engines are
                    # allowed to mutate their input list in place; after a host
                    # timeout the worker stays alive, so a shared list would let
                    # a late engine rewrite the live conversation (roles,
                    # ordering, persisted content) behind the caller's back.
                    # Deep-snapshot here, on the worker thread, so the caller's
                    # list object is never touched by pooled code. Results are
                    # published to caller-visible state only via the returned
                    # value of an ADMITTED commit (the host discards results on
                    # timeout/cancel); durable SessionDB mutation is already
                    # gated behind the commit fence inside compress_context.
                    snapshot = copy.deepcopy(messages)
                    result_msgs, result_prompt = _run(
                        fence, target_messages=snapshot
                    )
                    if result_msgs is snapshot:
                        # No-op/abort path returned the snapshot unchanged: hand
                        # back the caller's ORIGINAL list so identity-based
                        # semantics (len/identity no-op detection, flush dedup
                        # by id()) keep working.
                        return messages, result_prompt
                    return result_msgs, result_prompt

                # Resolve the fallback prompt lazily on timeout only. Eager
                # rebuild here would raise before compress_context runs whenever
                # _cached_system_prompt is unset and _build_system_prompt fails
                # (lock-refresher / noop-exception tests rely on that path).
                def _fallback_prompt():
                    cached = getattr(self, "_cached_system_prompt", None)
                    if cached:
                        return cached
                    try:
                        return self._build_system_prompt(system_message)
                    except Exception:
                        logger.debug(
                            "compress_context timeout fallback prompt rebuild "
                            "failed; using raw system_message",
                            exc_info=True,
                        )
                        return system_message or ""

                timeout_cause = {
                    "total_exhausted": False,
                    "progress_observed": False,
                }

                def _on_timeout_cause(total_exhausted, progress_observed):
                    timeout_cause["total_exhausted"] = total_exhausted
                    timeout_cause["progress_observed"] = progress_observed

                def _on_timeout(idle, waited, since_progress):
                    mark_context_compression_timed_out(self)
                    total_exhausted = timeout_cause["total_exhausted"]
                    progress_observed = timeout_cause["progress_observed"]
                    if total_exhausted:
                        logger.warning(
                            "Context compression reached its total ceiling "
                            "after %.1fs (progress observed=%s); continuing "
                            "without compression",
                            waited,
                            progress_observed,
                        )
                    else:
                        logger.warning(
                            "Context compression made no progress for %.1fs "
                            "(total wait %.1fs, ceiling %.1fs); continuing "
                            "without compression",
                            since_progress,
                            waited,
                            total_ceiling,
                        )
                    touch = getattr(self, "_touch_activity", None)
                    if callable(touch):
                        try:
                            touch(
                                "context compression timed out",
                                provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
                            )
                        except Exception:
                            logger.debug(
                                "compress_context timeout activity touch failed",
                                exc_info=True,
                            )
                    # Same timeout cooldown ladder as summary-LLM timeouts
                    # (#62452): avoid re-burning the full idle budget every turn.
                    compressor = getattr(self, "context_compressor", None)
                    if compressor is not None:
                        record = getattr(compressor, "record_timeout_failure", None)
                        if callable(record):
                            try:
                                reason = (
                                    "host compress_context total ceiling "
                                    "exhausted"
                                    if total_exhausted
                                    else "host compress_context timeout "
                                    "(no summary progress)"
                                )
                                record(
                                    reason,
                                    failure_kind=(
                                        "ceiling_exhausted"
                                        if total_exhausted
                                        else "stalled"
                                    ),
                                )
                            except Exception:
                                logger.debug(
                                    "failed to record compress_context timeout "
                                    "cooldown",
                                    exc_info=True,
                                )
                    emit = getattr(self, "_emit_warning", None)
                    if callable(emit):
                        if total_exhausted:
                            progress = (
                                " after summary output was observed"
                                if progress_observed
                                else ""
                            )
                            emit(
                                "⚠ Context compression reached its total ceiling "
                                f"after {waited:.1f}s{progress}. No messages were "
                                "dropped — continuing without compression. Run "
                                "/compress to retry or /new for a clean session."
                            )
                        else:
                            emit(
                                "⚠ Context compression timed out "
                                f"after {idle:.1f}s with no output from the summary "
                                "model. No messages were dropped — continuing "
                                "without compression. Run /compress to retry, /new "
                                "for a clean session, or check "
                                "auxiliary.compression."
                            )

                def _on_commit_overrun(waited, ceiling):
                    # Commit-phase ceiling breach: the SessionDB mutation is in
                    # flight and must complete (abandoning it mid-commit would
                    # diverge live messages from durable session state), so this
                    # only surfaces the overrun — it never cancels the commit.
                    emit = getattr(self, "_emit_warning", None)
                    if callable(emit):
                        emit(
                            "⚠ Context compression commit is taking unusually "
                            f"long ({waited:.0f}s, ceiling {ceiling:.0f}s). "
                            "Waiting for it to finish safely — if this persists, "
                            "check SessionDB health (disk / lock contention)."
                        )

                def _publish_new_fence():
                    # The stall-fallback retry (#78981) needs a fence the aborted
                    # attempt cannot veto. Publish it on the same serialized slot
                    # hard_interrupt() reads, so a /stop during the retry admits
                    # against the attempt that is actually running. The finally
                    # below restores whatever the caller had either way.
                    retry_fence = CompressionCommitFence()
                    with fence_registration_lock:
                        self._active_compression_commit_fence = retry_fence
                    return retry_fence

                result = run_compress_context_with_progress_timeout(
                    worker=_snapshot_worker,
                    messages=messages,
                    system_prompt_fallback=_fallback_prompt,
                    idle_timeout_seconds=idle_timeout,
                    total_ceiling_seconds=total_ceiling,
                    on_timeout=_on_timeout,
                    on_timeout_cause=_on_timeout_cause,
                    on_commit_overrun=_on_commit_overrun,
                    fence=active_fence,
                    telemetry_agent=self,
                    new_fence=_publish_new_fence,
                )
            # _DB_PERSISTED_MARKER lives at module level in
            # agent.context_compressor; conversation_compression only
            # imports it locally (cannot be imported from there). Imported
            # UNCONDITIONALLY (no fallback): both modules are already
            # hard dependencies at this point — agent.context_compressor is
            # imported at the top of this module (line ~162), and
            # agent.conversation_compression is imported at the top of this
            # very method and its compress_context is invoked below. The
            # only way these imports can fail while the wrapper is
            # functional is a renamed/removed symbol, and that must fail
            # LOUDLY: a silent fallback literal ("_db_persisted") would
            # split the stamping key from the flush's and quietly resurrect
            # the duplicate-row bug this fix removed.
            from agent.context_compressor import _DB_PERSISTED_MARKER
            from agent.conversation_compression import (
                _messages_match_scoped_identity,

            )

            def _sync_persisted_markers(target_messages, source_messages):
                if not isinstance(target_messages, list) or not isinstance(
                    source_messages, list
                ):
                    return
                # Compression runs against a deepcopy snapshot on the pooled
                # worker path, so publish stamps land on the result list first.
                # Mirror them back onto the live caller lists by scoped
                # identity after publish succeeds; timestamp-less repeated
                # content is ambiguous, so we stamp every scoped match instead
                # of stopping at the first one.
                for source_message in source_messages:
                    if not (
                        isinstance(source_message, dict)
                        and source_message.get(_DB_PERSISTED_MARKER)
                    ):
                        continue
                    source_timestamp = source_message.get("timestamp")
                    matched_exact_timestamp = False
                    if source_timestamp is not None:
                        for target_message in target_messages:
                            if not isinstance(target_message, dict):
                                continue
                            if target_message.get(_DB_PERSISTED_MARKER):
                                continue
                            if not _messages_match_scoped_identity(
                                target_message, source_message
                            ):
                                continue
                            if target_message.get("timestamp") != source_timestamp:
                                continue
                            target_message[_DB_PERSISTED_MARKER] = True
                            matched_exact_timestamp = True
                        if matched_exact_timestamp:
                            continue
                    for target_message in target_messages:
                        if not isinstance(target_message, dict):
                            continue
                        if target_message.get(_DB_PERSISTED_MARKER):
                            continue
                        if not _messages_match_scoped_identity(
                            target_message, source_message
                        ):
                            continue
                        target_message[_DB_PERSISTED_MARKER] = True

            if isinstance(result, tuple) and result:
                result_messages = result[0]
                if isinstance(result_messages, list):
                    # Direct-path callers bypass the snapshot worker, so they
                    # still need the same post-publish mirror onto the live
                    # caller list even when the returned list already points at
                    # the active transcript.
                    if direct_path or result_messages is not messages:
                        _sync_persisted_markers(messages, result_messages)
                    session_messages = getattr(self, "_session_messages", None)
                    if (
                        isinstance(session_messages, list)
                        and session_messages is not messages
                    ):
                        # Intentional: durable-parent adoption can leave
                        # `_session_messages` on the pre-adoption live list
                        # while `messages` now points at the adopted snapshot,
                        # so both lists need the post-publish marker sync.
                        _sync_persisted_markers(session_messages, result_messages)
            # compress_context ran on a daemon pool worker thread; the session
            # id rotation updated hermes_logging._session_context (a
            # threading.local) on the WORKER thread, not this one. Propagate
            # the current session_id back so subsequent log lines on this
            # thread carry the rotated id (#34089).
            try:
                from hermes_logging import set_session_context
                set_session_context(self.session_id)
            except Exception:
                pass
            # #76354 review F5: the worker thread also rebound the session
            # ContextVar inside its own (copied) context, which the caller
            # never sees — and get_session_env() prefers an already-bound
            # ContextVar over os.environ. Rebind in the CALLER's context so
            # post-compression tools/subprocesses on this thread resolve
            # HERMES_SESSION_ID to the child id after an out-of-place
            # rotation (idempotent when no rotation happened).
            try:
                from gateway.session_context import set_current_session_id
                if self.session_id:
                    set_current_session_id(self.session_id)
            except Exception:
                logger.debug(
                    "post-compression session ContextVar rebind failed",
                    exc_info=True,
                )
            return result
        finally:
            with fence_registration_lock:
                if previous_fence is missing_fence:
                    vars(self).pop("_active_compression_commit_fence", None)
                else:
                    self._active_compression_commit_fence = previous_fence
            # Restore whatever the caller had, so a compaction never leaks its
            # tag into the surrounding scope.
            if token is not None:
                reset_conversation_context(token)
            if affinity_token is not None:
                reset_affinity_scope(affinity_token)

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
        tool_call_id: str = "",
    ) -> str:
        decision = self._tool_guardrails.after_call(
            tool_name,
            function_args,
            function_result,
            failed=failed,
        )
        # Identical-call stall guards (agent.stall_guards): notice-only, no
        # blocking. Observed on the RAW result (before the loop-warning suffix
        # below, whose embedded count changes per call and would defeat
        # result-identity matching). Applied here — at result construction,
        # before the tool message is built — so it is cache-safe (tool results
        # are append-only; nothing already sent to the provider is mutated).
        stall_notice = None
        result_stub = None
        if self._stall_guards_enabled():
            try:
                observation = self._tool_guardrails.observe_call(
                    tool_name,
                    function_args,
                    function_result if isinstance(function_result, str) else None,
                    tool_call_id=tool_call_id,
                    failed=failed,
                )
                stall_notice = observation.notice
                result_stub = observation.stub
            except Exception as exc:
                logger.debug("stall-guard identical-call observation failed: %s", exc)
        # Result-reference stubbing: a 2nd+ consecutive identical call whose
        # FRESH result is byte-identical enters context as a short reference
        # stub instead of the duplicate payload. The tool still executed —
        # this is not a cache; a changed result flows through whole. Only
        # plain-string results are stubbed (multimodal content lists pass
        # through untouched), and the current message keeps its role and
        # tool_call_id — only the content is replaced.
        if result_stub and isinstance(function_result, str):
            function_result = result_stub
        if decision.action in {"warn", "halt"}:
            function_result = append_toolguard_guidance(function_result, decision)
        if decision.should_halt:
            self._set_tool_guardrail_halt(decision)
        if stall_notice:
            function_result = (function_result or "") + "\n\n" + stall_notice
        return function_result

    def _stall_guards_enabled(self) -> bool:
        """Config gate for the runtime anti-stall guards (agent.stall_guards)."""
        return bool(getattr(self, "_stall_guards", True))

    def _guardrail_block_result(self, decision: ToolGuardrailDecision) -> str:
        self._set_tool_guardrail_halt(decision)
        return toolguard_synthetic_result(decision)

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        The segment planner splits the batch into maximal contiguous runs of
        parallel-safe calls (read-only tools, non-overlapping file targets,
        opted-in MCP tools) separated by sequential barriers (interactive,
        unsafe, or unrecognized tools). Homogeneous batches keep their
        original single-path dispatch; mixed batches execute segment by
        segment in emission order so safe subsets still run concurrently
        while side-effect ordering is preserved.
        """
        tool_calls = assistant_message.tool_calls

        # Allow _vprint during tool execution even with stream consumers
        self._executing_tools = True
        try:
            if len(tool_calls) <= 1:
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            from agent.tool_dispatch_helpers import _plan_tool_batch_segments
            _active_env = get_active_env(effective_task_id)
            _exec_cwd = Path(_active_env.cwd) if _active_env is not None and _active_env.cwd else None
            segments = _plan_tool_batch_segments(tool_calls, execution_cwd=_exec_cwd)

            if len(segments) == 1:
                kind = segments[0][0]
                if kind == "parallel":
                    return self._execute_tool_calls_concurrent(
                        assistant_message, messages, effective_task_id, api_call_count
                    )
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            from agent.tool_executor import execute_tool_calls_segmented
            return execute_tool_calls_segmented(
                self, assistant_message, messages, effective_task_id, api_call_count,
                segments=segments,
            )
        finally:
            self._executing_tools = False

    def _dispatch_delegate_task(self, function_args: dict) -> str:
        """Single call site for delegate_task dispatch.

        New DELEGATE_TASK_SCHEMA fields only need to be added here to reach all
        invocation paths (concurrent, sequential, inline).
        """
        from tools.delegate_tool import (
            _strip_model_hidden_task_fields,
            delegate_task as _delegate_task,
        )
        # Delegations from the top-level MODEL always run in the background —
        # the model does not get to choose. delegate_task returns immediately
        # with a handle (one per task) and each subagent's result re-enters the
        # conversation as a new message when it finishes. This applies to BOTH
        # a single task and a fan-out batch (each task becomes its own
        # independent background subagent). The one exception:
        #   - A delegation from an ORCHESTRATOR SUBAGENT (depth > 0) stays
        #     synchronous: the orchestrator needs its workers' results within
        #     its own turn to compose a summary, and a subagent doesn't own the
        #     gateway session the async result would route back to.
        # The schema-level `background` param is intentionally ignored here.
        _is_subagent = getattr(self, "_delegate_depth", 0) > 0
        return _delegate_task(
            goal=function_args.get("goal"),
            context=function_args.get("context"),
            tasks=_strip_model_hidden_task_fields(function_args.get("tasks")),
            max_iterations=function_args.get("max_iterations"),
            role=function_args.get("role"),
            background=(not _is_subagent),
            action=function_args.get("action"),
            subagent_id=function_args.get("subagent_id"),
            message=function_args.get("message"),
            parent_agent=self,
        )

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None, messages: list = None,
                     pre_tool_block_checked: bool = False,
                     skip_tool_request_middleware: bool = False,
                     tool_request_middleware_trace: Optional[list[dict[str, Any]]] = None,
                     skip_tool_execution_middleware: bool = False) -> str:
        """Forwarder — see ``agent.agent_runtime_helpers.invoke_tool``."""
        from agent.agent_runtime_helpers import invoke_tool
        return invoke_tool(
            self,
            function_name,
            function_args,
            effective_task_id,
            tool_call_id,
            messages,
            pre_tool_block_checked,
            skip_tool_request_middleware,
            tool_request_middleware_trace,
            skip_tool_execution_middleware,
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
        """Forwarder — see ``agent.tool_executor.execute_tool_calls_concurrent``."""
        from agent.tool_executor import execute_tool_calls_concurrent
        return execute_tool_calls_concurrent(self, assistant_message, messages, effective_task_id, api_call_count)

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Forwarder — see ``agent.tool_executor.execute_tool_calls_sequential``."""
        from agent.tool_executor import execute_tool_calls_sequential
        return execute_tool_calls_sequential(self, assistant_message, messages, effective_task_id, api_call_count)

    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """Forwarder — see ``agent.chat_completion_helpers.handle_max_iterations``."""
        from agent.chat_completion_helpers import handle_max_iterations
        return handle_max_iterations(self, messages, api_call_count)

    def _conversation_root_id(self) -> Optional[str]:
        """Resolve the stable conversation id for Portal usage attribution.

        Returns the session-lineage ROOT id rather than the current segment
        id, so one user-facing conversation keeps a single ``conversation=``
        tag across context-compression rotation (`/new` starts a genuinely
        new lineage). Delegate subagents resolve through their
        ``_parent_session_id`` so an entire delegation tree tags as the
        parent conversation.

        Best-effort: falls back to the raw session id when the session DB
        is unavailable or the lineage walk fails.
        """
        sid = getattr(self, "session_id", None)
        if not sid:
            return None
        # Subagents may not have a DB row yet on their first turn; walking
        # from the parent id still lands on the right root.
        start = getattr(self, "_parent_session_id", None) or sid
        db = getattr(self, "_session_db", None)
        if db is not None:
            try:
                root = db.get_conversation_root(start)
                if root:
                    return root
            except Exception:
                logger.debug("Conversation root lineage walk failed", exc_info=True)
        return start

    def run_conversation(
        self,
        user_message: Any,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        persist_user_display_metadata: Optional[Dict[str, Any]] = None,
        persist_user_platform_id: Optional[str] = None,
        moa_config: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Forwarder — see ``agent.conversation_loop.run_conversation``."""
        # A review deliberately shares this agent's session_id for prompt-cache
        # parity. Fence review startup or interrupt an admitted request, then
        # await that request's exit before opening any live-turn Relay or task
        # instrumentation for the same session. Foreground priority is retained
        # if the review does not acknowledge within the bounded deadline (#84423).
        from agent.background_review import cancel_background_review_for_live_turn

        cancel_background_review_for_live_turn(self)

        from agent.aux_accounting import (
            reset_accounting_context,
            set_accounting_context,
        )
        from agent import relay_runtime
        from agent.conversation_loop import run_conversation
        from agent.portal_tags import (
            reset_affinity_scope,
            reset_conversation_context,
            set_affinity_scope,
            set_conversation_context,
        )
        from agent.prompt_cache_scope import declared_conversation_scope_safe
        from hermes_cli.observability.relay_shared_metrics import (
            finish_task_run,
            start_task_run,
        )
        from agent.subagent_lifecycle import bind_subagent_parent
        effective_task_id = task_id or str(uuid.uuid4())
        session_id = str(getattr(self, "session_id", None) or "")
        task_context = {
            "session_id": session_id,
            "task_id": effective_task_id,
            "platform": getattr(self, "platform", None) or "",
        }
        relay_turn_id = (
            f"{session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
        )
        self._relay_pending_turn_id = relay_turn_id
        relay_parent_session_id = (
            str(getattr(self, "_parent_session_id", None) or "")
            if task_context["platform"] == "subagent"
            else ""
        )
        relay_lease = None
        relay_turn = None
        durable_turn_lease = None
        durable_turn_lease_stop = None
        durable_turn_lease_thread = None
        durable_turn_liveness_thread = None
        durable_turn_lease_activity_lock = threading.Lock()
        durable_turn_lease_turn_active = False
        durable_turn_lease_interrupt_message = None
        token = None
        # Initialized alongside `token`: the turn-lease timeout/interrupt
        # early returns leave the try block before set_affinity_scope() runs,
        # and the finally reads this name unconditionally (UnboundLocalError
        # otherwise — the 4 red cross-process lease tests on PR #97158).
        affinity_token = None
        acct_token = None
        task_started = False
        task_finished = False
        relay_outcome = "failed"

        def _stop_durable_turn_lease_refresher() -> None:
            nonlocal durable_turn_lease_turn_active
            with durable_turn_lease_activity_lock:
                durable_turn_lease_turn_active = False
                if durable_turn_lease_stop is not None:
                    durable_turn_lease_stop.set()

        def _clear_durable_turn_lease_interrupt() -> None:
            """Clear only the interrupt admitted by this turn's refresher."""
            message = durable_turn_lease_interrupt_message
            if not message:
                return

            def _clear_if_owned() -> None:
                if getattr(self, "_interrupt_message", None) != message:
                    return
                self._interrupt_requested = False
                self._interrupt_message = None
                getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
                self._interrupt_thread_signal_pending = False
                if self._execution_thread_id is not None:
                    _set_interrupt(False, self._execution_thread_id)

            redirect_lock = getattr(self, "_pending_redirect_lock", None)
            if redirect_lock is None:
                _clear_if_owned()
            else:
                with redirect_lock:
                    _clear_if_owned()

        try:
            # Serialize the full load -> run -> flush region across Hermes
            # processes. Gateway's asyncio lease closes alias routing inside one
            # process; this durable lease covers Desktop, CLI resume, gateway,
            # and background delivery processes sharing state.db (#84234).
            _turn_db = getattr(self, "_session_db", None)
            _durable_session_exists = False
            if _turn_db is not None and session_id:
                try:
                    _durable_session_exists = _turn_db.get_session(session_id) is not None
                except Exception:
                    # A locked / non-WAL read is not proof the row is absent.
                    # Treating probe failure as "fresh session" skipped the
                    # lease this block exists to take and ran fail-open on
                    # the exact contention point (#84234). Acquire (or fail
                    # closed if acquire itself cannot) rather than start
                    # load/run/flush unsynchronized. get_session returns
                    # None — it does not raise — when the row is missing.
                    logger.warning(
                        "Could not check durable session before turn lease; "
                        "will acquire rather than run without serialization",
                        exc_info=True,
                    )
                    _durable_session_exists = True
            if (
                _turn_db is not None
                and session_id
                and not getattr(self, "_persist_disabled", False)
                # A fresh session id is process-unique and has no durable
                # transcript to race over. More importantly, subagent/new-turn
                # callers may intentionally supply an in-memory seed before the
                # row exists; reloading an absent row would erase that seed.
                and _durable_session_exists
                # Test doubles and third-party DB shims may accept arbitrary
                # MagicMock attributes without implementing the protocol. Check
                # the concrete type so only real implementations opt in.
                and callable(
                    getattr(type(_turn_db), "acquire_session_turn_lease", None)
                )
            ):
                # Resumed agents also defer their create check until the turn
                # prologue. We just proved this row exists, so suppress the
                # redundant create attempt after acquiring it.
                self._session_db_created = True
                _durable_holder = (
                    f"pid={os.getpid()}:turn={relay_turn_id}:platform="
                    f"{task_context['platform'] or 'unknown'}"
                )
                _lease_ttl = 300.0
                _lease_waited = False

                def _on_session_turn_lease_wait(elapsed: float) -> None:
                    nonlocal _lease_waited
                    _lease_waited = True
                    if elapsed < 1.0:
                        self._emit_status(
                            "⏳ Another Hermes process is using this session; "
                            "waiting for it to finish before starting your turn..."
                        )
                    else:
                        self._emit_status(
                            "⏳ Still waiting for the other Hermes process on "
                            f"this session ({int(elapsed)}s)..."
                        )

                if not _turn_db.acquire_session_turn_lease(
                    session_id,
                    _durable_holder,
                    ttl_seconds=_lease_ttl,
                    wait_seconds=1800.0,
                    on_wait=_on_session_turn_lease_wait,
                    should_abort=lambda: getattr(self, "_interrupt_requested", False),
                ):
                    if getattr(self, "_interrupt_requested", False):
                        logger.info(
                            "session turn lease wait aborted by interrupt: %s",
                            session_id,
                        )
                        relay_outcome = "cancelled"
                        interrupt_msg = (
                            "Stopped waiting for another Hermes process on "
                            "this session. Your message was not processed."
                        )
                        interrupt_result = {
                            "final_response": interrupt_msg,
                            "messages": list(conversation_history or []),
                            "api_calls": 0,
                            "completed": False,
                            "interrupted": True,
                        }
                        interrupt_message = getattr(
                            self, "_interrupt_message", None
                        )
                        if interrupt_message:
                            interrupt_result["interrupt_message"] = (
                                interrupt_message
                            )
                        # Conversation-loop finalizer never runs on this
                        # early return. Clear so a cached agent cannot
                        # fail-close the next turn as interrupted.
                        try:
                            self.clear_interrupt()
                        except Exception:
                            self._interrupt_requested = False
                            self._interrupt_message = None
                        return interrupt_result
                    # Fail closed like gateway TurnLeaseTimeoutError: do not
                    # enter load/run/flush, and surface a resend notice instead
                    # of a bare TimeoutError that looks like a hang.
                    timeout_msg = (
                        "⏳ Another Hermes process kept this session busy too "
                        "long. Your message was not processed - wait for the "
                        "other process to finish, then send it again."
                    )
                    logger.error(
                        "session turn lease wait timed out for %s",
                        session_id,
                    )
                    try:
                        self._emit_warning(timeout_msg)
                    except Exception:
                        logger.debug(
                            "Failed to emit session turn lease timeout warning",
                            exc_info=True,
                        )
                    relay_outcome = "timed_out"
                    return {
                        "final_response": timeout_msg,
                        "messages": list(conversation_history or []),
                        "api_calls": 0,
                        "completed": False,
                        "failed": True,
                        "error": f"session_turn_lease_timeout:{session_id}",
                    }

                # Assign only after admission so finally release cannot target a
                # holder string that never owned the row. Persist paths read
                # the agent attr so a late flush after reclaim is fenced in
                # the same SQLite write transaction as the transcript insert.
                durable_turn_lease = _durable_holder
                self._active_session_turn_lease_holder = _durable_holder
                self._active_session_turn_lease_ttl_seconds = _lease_ttl
                if _lease_waited:
                    self._emit_status(
                        "Session is free; loading the latest transcript..."
                    )

                # The holder may have compressed and rotated the session while
                # this process waited. Resolve and reload only AFTER admission;
                # a caller-provided in-memory snapshot is necessarily stale.
                # Skip when acquisition was immediate — no other process held
                # the lease, so the in-memory history is current and reloading
                # would only cause an unnecessary prompt cache miss.
                if _lease_waited:
                    latest_session_id = _turn_db.resolve_resume_session_id(session_id)
                    if latest_session_id:
                        self.session_id = latest_session_id
                        task_context["session_id"] = latest_session_id
                    conversation_history = _turn_db.get_messages_as_conversation(
                        self.session_id,
                        repair_alternation=True,
                        include_row_ids=True,
                    )

                # Long model/tool/compression turns outlive a fixed TTL. Refresh
                # in a daemon thread; holder-qualified UPDATE and DELETE fence a
                # late refresher/release from a successor lease.
                durable_turn_lease_stop = threading.Event()
                _lease_refresh_interval = float(
                    getattr(self, "_session_turn_lease_refresh_interval", 60.0)
                )

                # ── Turn liveness watchdog (#95548) ─────────────────────
                # The durable lease refresher keeps the lease alive for as
                # long as the turn runs, so lease renewal is NOT evidence of
                # progress. A turn that stalls silently (observed #95548: no
                # tool execution, no API call, no persisted message for 9+
                # minutes after a slow model response + desktop WS
                # disconnect) would otherwise renew its lease forever, look
                # "active", and never be force-aborted.
                #
                # The watchdog policy (config resolution, sampling state
                # machine, thread mechanics) lives in agent/turn_liveness.py;
                # this block is only the integration seam: resolve the
                # config.yaml settings, wire the commit/deactivate callbacks
                # that own turn-lease state, and start the thread.
                try:
                    from hermes_cli.config import (
                        load_config_readonly as _liveness_load_config,
                    )
                    _liveness_config = _liveness_load_config() or {}
                except Exception:
                    _liveness_config = {}
                from agent import turn_liveness

                _liveness_timeout, _liveness_poll = (
                    turn_liveness.resolve_turn_liveness_settings(_liveness_config)
                )

                def _interrupt_turn(message: str) -> None:
                    # Lease-loss interrupts fire UNCONDITIONALLY (no
                    # require_generation claim): losing the durable lease
                    # means this process no longer owns the session, so
                    # the turn must stop regardless of activity-clock
                    # progress. The generation-claim machinery is the
                    # liveness watchdog's only — its stalls can be
                    # spuriously stale, a lost lease cannot.
                    nonlocal durable_turn_lease_interrupt_message
                    with durable_turn_lease_activity_lock:
                        if (
                            durable_turn_lease_stop.is_set()
                            or not durable_turn_lease_turn_active
                        ):
                            return
                        durable_turn_lease_interrupt_message = message
                        try:
                            self.interrupt(message, hard_cancel=True)
                        except Exception:
                            self._interrupt_requested = True
                            self._interrupt_message = message

                def _commit_turn_liveness_abort(
                    snapshot: "turn_liveness.ActivitySnapshot",
                    message: str,
                ) -> bool:
                    """Commit point for the watchdog's stall observation.

                    Revalidates the observed ``(generation, timestamp)`` pair
                    under the SAME lock ``_touch_activity`` stamps the clock
                    with, so a turn that resumed while the stall was being
                    logged/emitted is never hard-cancelled (#95663 review):
                    it continues and its lease keeps renewing. Returns False
                    when the observation is stale (watchdog keeps sampling)
                    or the turn is already winding down.

                    Round-3 (#95663): the revalidated generation is carried
                    into the interrupt path as a claim
                    (``require_generation``); ``interrupt`` reserves it,
                    consumes it and publishes the first interrupt state in
                    ONE activity-lock critical section (round-6) and
                    abandons the abort when it went stale.

                    Round-4 (#95663): if ``interrupt`` raises, the abort
                    declines FAIL-CLOSED — the exceptional path must not
                    convert the inability to validate/publish the claim
                    through the normal path into unconditional interrupt
                    authority. No interrupt state is mutated here; the
                    watchdog keeps sampling while the turn (which may have
                    resumed) continues.
                    """
                    nonlocal durable_turn_lease_interrupt_message
                    with self._liveness_activity_lock():
                        current_generation = getattr(
                            self, "_turn_liveness_activity_generation", 0
                        )
                        if (
                            current_generation,
                            getattr(self, "_last_activity_ts", None),
                        ) != (snapshot.generation, snapshot.activity_ts):
                            return False
                    with durable_turn_lease_activity_lock:
                        if (
                            durable_turn_lease_stop.is_set()
                            or not durable_turn_lease_turn_active
                        ):
                            return False
                    try:
                        published = self.interrupt(
                            message,
                            hard_cancel=True,
                            require_generation=current_generation,
                        )
                    except Exception:
                        # Round-4 (#95663): fail closed. An exceptional
                        # interrupt path must not turn the inability to
                        # validate/publish the generation claim into
                        # unconditional abort authority — declining keeps
                        # the watchdog sampling while the turn (which may
                        # have resumed) continues.
                        logger.debug(
                            "Turn liveness abort interrupt raised; "
                            "declining the abort",
                            exc_info=True,
                        )
                        published = False
                    if published is False:
                        # The generation claim went stale between the
                        # revalidation above and the hammer: real progress
                        # landed in the window, so the abort abandons itself
                        # and the watchdog keeps sampling while the turn
                        # (and its lease) continue.
                        return False
                    with durable_turn_lease_activity_lock:
                        durable_turn_lease_interrupt_message = message
                    return True

                def _deactivate_turn_after_liveness_abort() -> None:
                    """Stop lease renewal after a committed liveness abort.

                    A wedge the hard interrupt cannot unwind must not keep
                    the lease alive forever (the issue's "lease keeps
                    renewing" masking); TTL expiry then lets stale-turn
                    cleanup reclaim the row.
                    """
                    nonlocal durable_turn_lease_turn_active
                    with durable_turn_lease_activity_lock:
                        durable_turn_lease_stop.set()
                        durable_turn_lease_turn_active = False

                def _turn_is_active() -> bool:
                    with durable_turn_lease_activity_lock:
                        return durable_turn_lease_turn_active

                def _refresh_durable_turn_lease() -> None:
                    while not durable_turn_lease_stop.wait(_lease_refresh_interval):
                        try:
                            if not _turn_db.refresh_session_turn_lease(
                                getattr(self, "session_id", None) or session_id,
                                durable_turn_lease,
                                ttl_seconds=_lease_ttl,
                            ):
                                # finally sets the stop event then releases.
                                # A late holder-fenced miss after that join
                                # timeout must not hard-interrupt the next turn.
                                if durable_turn_lease_stop.is_set():
                                    return
                                logger.error(
                                    "Lost session turn lease while turn is active: %s",
                                    getattr(self, "session_id", None) or session_id,
                                )
                                _interrupt_turn(
                                    "Session turn lease lost; stopping to protect "
                                    "the transcript."
                                )
                                return
                        except Exception:
                            if durable_turn_lease_stop.is_set():
                                return
                            logger.warning(
                                "Failed to refresh session turn lease: %s",
                                getattr(self, "session_id", None) or session_id,
                                exc_info=True,
                            )
                            _interrupt_turn(
                                "Session turn lease could not be refreshed; "
                                "stopping to protect the transcript."
                            )
                            return

                durable_turn_lease_thread = threading.Thread(
                    target=_refresh_durable_turn_lease,
                    name="session-turn-lease-refresh",
                    daemon=True,
                )
                if _liveness_timeout is not None:
                    durable_turn_liveness_thread = (
                        turn_liveness.TurnLivenessWatchdog(
                            self,
                            session_id=getattr(self, "session_id", None) or session_id,
                            timeout_s=_liveness_timeout,
                            poll_s=_liveness_poll,
                            stop_event=durable_turn_lease_stop,
                            activity_lock=self._liveness_activity_lock(),
                            is_turn_active=_turn_is_active,
                            commit_abort=_commit_turn_liveness_abort,
                            deactivate_turn=_deactivate_turn_after_liveness_abort,
                        ).make_thread()
                    )


            relay_lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=task_context["session_id"],
                platform=task_context["platform"],
                parent_session_id=relay_parent_session_id,
                model=str(getattr(self, "model", None) or ""),
            )
            relay_turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
                relay_lease,
                turn_id=relay_turn_id,
                task_id=effective_task_id,
            )
            # Keep existing tests and external relay-runtime shims that return
            # a minimal turn object compatible with the new opt-out flag.
            if getattr(relay_turn, "relay_enabled", True):
                start_task_run(
                    **task_context,
                    parent_session_id=getattr(self, "_parent_session_id", None) or "",
                )
                task_started = True
            # Publish the conversation id for ambient Nous Portal tagging. Every
            # LLM call made inside this turn — main loop, compression, vision,
            # web_extract, session_search, MoA slots, background-review forks
            # (which copy this Context into their thread) — inherits the
            # ``conversation=<root>`` tag with zero per-call-site plumbing.
            token = set_conversation_context(self._conversation_root_id())
            # Routing/affinity scope for the same turn — the conversation the
            # HOST declared, when it declared one. Providers fall back to the
            # attribution id above when it is unset, so this changes nothing
            # for a host that keeps one session id per conversation (#96811).
            affinity_token = set_affinity_scope(
                declared_conversation_scope_safe(self)
            )
            # Publish the session accounting handles the same way so auxiliary
            # calls record their token usage into session_model_usage (task
            # dimension) — the fix for aux spend being invisible in analytics
            # (issue #23270).
            acct_token = set_accounting_context(
                getattr(self, "_session_db", None),
                getattr(self, "session_id", None),
            )
            from agent.auxiliary_client import scoped_runtime_main

            # The outer token restores the caller's Context even though turn setup
            # replaces the value with the live runtime after fallback restoration.
            # Keep the scope local instead of storing ContextVar tokens on the agent,
            # which may be observed from another thread.
            with bind_subagent_parent(self), scoped_runtime_main({}):
                try:
                    if durable_turn_lease_thread is not None:
                        with durable_turn_lease_activity_lock:
                            durable_turn_lease_turn_active = True
                        # Stamp the activity clock at turn entry (#95663
                        # review): a real agent keeps ``_last_activity_ts``
                        # across turns (idle time between turns is normal —
                        # ``_reset_activity_labels_after_turn`` preserves
                        # it by design), so without this stamp the liveness
                        # watchdog would measure idle from the PREVIOUS
                        # turn and force-abort a just-started turn on its
                        # first poll whenever the agent had been idle longer
                        # than the watchdog bound.
                        self._touch_activity("starting new turn")
                        durable_turn_lease_thread.start()
                        if durable_turn_liveness_thread is not None:
                            durable_turn_liveness_thread.start()
                    result = run_conversation(
                        self,
                        user_message,
                        system_message,
                        conversation_history,
                        effective_task_id,
                        stream_callback,
                        persist_user_message,
                        persist_user_timestamp=persist_user_timestamp,
                        persist_user_display_kind=persist_user_display_kind,
                        persist_user_display_metadata=persist_user_display_metadata,
                        persist_user_platform_id=persist_user_platform_id,
                        moa_config=moa_config,
                    )
                finally:
                    # The lease remains held through relay/task finalization, but
                    # those post-loop steps must not receive a late refresh
                    # interrupt that poisons the next turn on a cached agent.
                    _stop_durable_turn_lease_refresher()
                    # Interrupt clear is deferred to after thread join in the
                    # outer finally: a refresher firing between stop and join
                    # would otherwise set an interrupt that survives the clear.
            terminal = result if isinstance(result, dict) else {}
            if terminal.get("interrupted") is True:
                relay_outcome = "cancelled"
            elif terminal.get("failed") is True:
                relay_outcome = "failed"
            else:
                relay_outcome = "success"
            relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                relay_turn,
                outcome=relay_outcome,
            )
            if task_started:
                task_finished = True
                finish_task_run(**task_context, result=result)
            return result
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (
                type(exc).__name__ == "CancelledError"
            ):
                relay_outcome = "cancelled"
            elif isinstance(exc, TimeoutError):
                relay_outcome = "timed_out"
            if relay_turn is not None:
                relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                    relay_turn,
                    outcome=relay_outcome,
                )
            if task_started and not task_finished:
                task_finished = True
                finish_task_run(**task_context, error=exc)
            raise
        finally:
            try:
                if relay_turn is not None:
                    relay_runtime.SESSION_COORDINATOR.end_turn(
                        relay_turn,
                        outcome=relay_outcome,
                    )
            finally:
                try:
                    if relay_lease is not None:
                        relay_runtime.SESSION_COORDINATOR.release_conversation(
                            relay_lease
                        )
                finally:
                    _stop_durable_turn_lease_refresher()
                    for _durable_thread in (
                        durable_turn_lease_thread,
                        durable_turn_liveness_thread,
                    ):
                        if (
                            _durable_thread is not None
                            and _durable_thread.is_alive()
                        ):
                            _durable_thread.join(timeout=1.0)
                    # Clear any interrupt the refresher may have fired between
                    # the inner stop and this join. Must run AFTER join so a
                    # late interrupt does not survive into the next turn.
                    _clear_durable_turn_lease_interrupt()
                    if durable_turn_lease is not None:
                        try:
                            _turn_db.release_session_turn_lease(
                                session_id, durable_turn_lease
                            )
                        except Exception:
                            logger.error(
                                "Failed to release session turn lease: %s",
                                session_id,
                                exc_info=True,
                            )
                        if (
                            getattr(self, "_active_session_turn_lease_holder", None)
                            == durable_turn_lease
                        ):
                            self._active_session_turn_lease_holder = None
                            self._active_session_turn_lease_ttl_seconds = None
                    # Always clear mid-turn labels when the turn exits — including
                    # interrupted early returns that skip finalize_turn. Keep ts.
                    try:
                        self._reset_activity_labels_after_turn()
                    except Exception:
                        pass
                    if getattr(self, "_relay_pending_turn_id", None) == relay_turn_id:
                        self._relay_pending_turn_id = None
                    if acct_token is not None:
                        reset_accounting_context(acct_token)
                    if token is not None:
                        reset_conversation_context(token)
                    if affinity_token is not None:
                        reset_affinity_scope(affinity_token)

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

    def _run_codex_app_server_turn(
        self,
        *,
        user_message: str,
        original_user_message: Any,
        messages: List[Dict[str, Any]],
        effective_task_id: str,
        should_review_memory: bool = False,
    ) -> Dict[str, Any]:
        """Forwarder — see ``agent.codex_runtime.run_codex_app_server_turn``."""
        from agent.codex_runtime import run_codex_app_server_turn
        return run_codex_app_server_turn(self, user_message=user_message, original_user_message=original_user_message, messages=messages, effective_task_id=effective_task_id, should_review_memory=should_review_memory)

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
