#!/usr/bin/env python3
"""AIAgent: the tool-calling agent runner (conversation loop, tool execution, session lifecycle).

    from run_agent import AIAgent
    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

# hermes_bootstrap must be the very first import (UTF-8 stdio on Windows; no-op on POSIX).
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass  # partial `hermes update` — only skips the Windows UTF-8 stdio setup

import json
import logging
logger = logging.getLogger(__name__)
import os
import re
import sys
import time
import threading
import uuid
import warnings
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home


def _launch_cwd_for_session(source: str) -> Optional[str]:
    """cwd to stamp on a new session row (``hermes -c`` / ``--resume``), or None.

    Only local CLI sessions record one: gateway/cron/remote backends (non-"local" ``TERMINAL_ENV``) have no
    stable host cwd for the agent's tools.
    """
    if source != "cli" or (os.environ.get("TERMINAL_ENV") or "local").strip().lower() not in ("", "local"):
        return None
    try:
        return os.getcwd()
    except OSError:  # cwd was unlinked out from under us
        return None


def _session_source_for_agent(platform: Optional[str]) -> str:
    try:
        from gateway.session_context import get_session_env

        source = get_session_env("HERMES_SESSION_SOURCE", "")
    except Exception:
        source = os.environ.get("HERMES_SESSION_SOURCE", "")
    return str(source or "").strip() or platform or "cli"


def _gateway_origin_json(agent: "AIAgent") -> Optional[str]:
    """Gateway routing ``origin_json`` for a session row; None when the agent carries no gateway identity.

    Mirrors ``SessionSource.to_dict()`` so state.db consumers see the same fields ``record_gateway_session_peer`` writes.
    """
    chat_id = getattr(agent, "_chat_id", None)
    session_key = getattr(agent, "_gateway_session_key", None)
    user_id = getattr(agent, "_user_id", None)
    if not (chat_id or session_key or user_id):
        return None
    origin: Dict[str, Any] = {
        "platform": getattr(agent, "platform", None) or "", "chat_id": chat_id,
        "chat_name": getattr(agent, "_chat_name", None), "chat_type": getattr(agent, "_chat_type", None) or "dm",
        "user_id": user_id, "user_name": getattr(agent, "_user_name", None), "thread_id": getattr(agent, "_thread_id", None),
    }
    if getattr(agent, "_user_id_alt", None):
        origin["user_id_alt"] = agent._user_id_alt
    profile = getattr(agent, "_profile_name", None)
    if not profile:
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
        except Exception:
            profile = None
        if profile == "default":
            profile = None
    if profile:
        origin["profile"] = profile
    try:
        return json.dumps(origin)
    except Exception:
        return None


from agent.iteration_budget import IterationBudget
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout

_hermes_home = get_hermes_home()  # read by agent_init via _ra()._hermes_home
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).parent / '.env')
for _env_path in _loaded_env_paths:
    logger.info("Loaded environment variables from %s", _env_path)
if not _loaded_env_paths:
    logger.info("No .env file found. Using system environment variables.")


from model_tools import get_toolset_for_tool
from tools.terminal_tool_lifecycle import cleanup_vm, get_active_env
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool_lifecycle import cleanup_browser

from agent.memory_provider import is_trivial_prompt
from agent.client_lifecycle import ClientLifecycleMixin
from agent.stream_delivery import StreamDeliveryMixin
from agent.status_output import StatusOutputMixin
from agent.api_request_hooks import ApiRequestHooksMixin
from agent.api_error_summary import ApiErrorSummaryMixin
from agent.interrupt_control import InterruptControlMixin
from agent.turn_explainers import TurnExplainersMixin
from agent.activity_tracking import ActivityTrackingMixin
from agent.rate_limit_credits import RateLimitCreditsMixin
from agent.session_persistence import SessionPersistenceMixin
from agent.compression_facade import CompressionFacadeMixin
from agent.turn_facade import TurnFacadeMixin
from agent.vision_message_prep import VisionMessagePrepMixin
from agent.reasoning_params import ReasoningParamsMixin
from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from agent.session_activity import ActivityProvenance
from agent.model_metadata import is_local_endpoint
from agent.message_sanitization import (
    coalesce_tool_call_id as _sanitize_coalesce_tool_call_id,
    deterministic_call_id as _codex_deterministic_call_id,
    uniquify_tool_call_ids as _sanitize_uniquify_tool_call_ids,
)
from agent.codex_responses_adapter import (
    _derive_responses_function_call_id as _codex_derive_responses_function_call_id,
    _split_responses_tool_id as _codex_split_responses_tool_id,
    _summarize_user_message_for_log,
)
from agent.tool_guardrails import ToolGuardrailDecision, append_toolguard_guidance, toolguard_synthetic_result
from utils import base_url_host_matches, base_url_hostname, env_float, model_forces_max_completion_tokens


_MAX_TOOL_WORKERS = 8


# Spawn the OpenRouter pre-warm thread once per process, not per AIAgent (gateway thread leak).
_openrouter_prewarm_done = threading.Event()


def _quietly(fn: Callable, *args, **kwargs) -> None:
    """Run one teardown step, swallowing any exception so sibling steps still run."""
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def _call_engine_hook(engine: Any, hook: str, *args, **kwargs) -> None:
    """Invoke an optional context-engine lifecycle hook; failures are logged, never raised."""
    if not hasattr(engine, hook):
        return
    try:
        getattr(engine, hook)(*args, **kwargs)
    except Exception as exc:
        logger.debug("context engine %s during transition: %s", hook, exc)


def _positive_int(value: Any) -> Optional[int]:
    """``value`` when it is a real positive int (bools excluded), else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _review_should_defer(agent: Any, task_cfg: Optional[Dict[str, Any]]) -> bool:
    """True when an automatic background review targets the managed local runtime under ``defer: auto``."""
    from agent.review_idle_queue import defer_mode, review_targets_managed_local
    return defer_mode(task_cfg) == "auto" and review_targets_managed_local(agent, task_cfg)


def _review_queue_key(agent: Any) -> str:
    return str(getattr(agent, "session_id", None) or id(agent))


def _notify_context_engine_session_end(agent: Any, messages: Optional[list]) -> None:
    """Tell the context engine the session ended (flush DAG, close DBs) at the same lifecycle moment as the
    memory manager, so per-session engine state never leaks into the next session."""
    engine = getattr(agent, "context_compressor", None)
    if engine:
        _quietly(lambda: engine.on_session_end(agent.session_id or "", messages or []))


def _pool_may_recover_from_rate_limit(pool) -> bool:
    """Wait for credential-pool rotation (True) or fall back to ``fallback_model`` (False) after a 429.

    Rotation only helps when the pool has somewhere to go; a single-credential pool would retry the same quota.

    See issues #11314 and #13636.
    """
    return pool is not None and pool.has_available() and len(pool.entries()) > 1


class _StreamErrorEvent(Exception):
    """Provider error synthesized from a standalone Responses ``type=error`` SSE frame (Codex-style backends).

    Gives ``_summarize_api_error`` / the entitlement detector the familiar ``.body`` / ``.status_code`` shape.
    """

    def __init__(self, message: str, *, code: Optional[str] = None, param: Optional[str] = None,
                 status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message, self.code, self.param, self.status_code = message, code, param, status_code
        # OpenAI SDK-shaped body so _extract_api_error_context / _summarize_api_error / classify_api_error pick it up.
        self.body: Dict[str, Any] = {"error": {"message": message, "code": code, "param": param, "type": "error"}}


class AIAgent(
    ClientLifecycleMixin, StreamDeliveryMixin, StatusOutputMixin, ApiRequestHooksMixin, ApiErrorSummaryMixin,
    InterruptControlMixin, TurnExplainersMixin, ActivityTrackingMixin, RateLimitCreditsMixin,
    SessionPersistenceMixin, CompressionFacadeMixin, TurnFacadeMixin, VisionMessagePrepMixin, ReasoningParamsMixin,
):
    """AI Agent with tool calling capabilities."""

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
        base_url: str = None, api_key: str = None, provider: str = None, api_mode: str = None,
        acp_command: str = None, acp_args: list[str] | None = None, command: str = None, args: list[str] | None = None,
        model: str = "",
        max_iterations: int = sys.maxsize,  # unlimited tool-calling iterations by default (shared with subagents)
        tool_delay: float = None,  # deprecated: accepted for compatibility, ignored
        enabled_toolsets: List[str] = None, disabled_toolsets: List[str] = None,
        save_trajectories: bool = False, verbose_logging: bool = False, quiet_mode: bool = False,
        tool_progress_mode: str = "all", ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100, log_prefix: str = "",
        providers_allowed: List[str] = None, providers_ignored: List[str] = None, providers_order: List[str] = None,
        provider_sort: str = None, provider_require_parameters: bool = False, provider_data_collection: str = None,
        openrouter_min_coding_score: Optional[float] = None,
        session_id: str = None,
        tool_progress_callback: callable = None, tool_start_callback: callable = None,
        tool_complete_callback: callable = None, thinking_callback: callable = None,
        reasoning_callback: callable = None, clarify_callback: callable = None,
        read_terminal_callback: callable = None, read_preview_callback: callable = None,
        drive_preview_callback: callable = None, read_window_below_callback: callable = None,
        setup_mcp_callback: callable = None, workflow_callback: callable = None,
        tour_callback: callable = None, step_callback: callable = None,
        stream_delta_callback: callable = None, interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None, status_callback: callable = None,
        notice_callback: callable = None, notice_clear_callback: callable = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
        reaction_callback: Optional[Callable[[str], None]] = None,
        max_tokens: int = None, reasoning_config: Dict[str, Any] = None, service_tier: str = None,
        request_overrides: Dict[str, Any] = None, prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None, user_id: str = None, user_id_alt: str = None, user_name: str = None,
        chat_id: str = None, chat_name: str = None, chat_type: str = None, thread_id: str = None,
        gateway_session_key: str = None,
        skip_context_files: bool = False, load_soul_identity: bool = False,
        skip_memory: bool = False, skip_background_review: bool = False,
        session_db=None, parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None, run_budget_seconds: Optional[float] = None,
        fallback_model: Dict[str, Any] = None, credential_pool=None,
        checkpoints_enabled: bool = False, checkpoint_max_snapshots: int = 20,
        checkpoint_max_total_size_mb: int = 500, checkpoint_max_file_size_mb: int = 10,
        pass_session_id: bool = False, requested_provider: str = None,
        capabilities: Dict[str, bool] | None = None,
    ):
        """Forwarder — see ``agent.agent_init.init_agent`` (same keyword parameters, minus ``tool_delay``)."""
        init_kwargs = {k: v for k, v in locals().items() if k not in ("self", "tool_delay")}
        if tool_delay is not None:
            warnings.warn("tool_delay is deprecated and ignored; sequential tool calls "
                          "no longer sleep between executions.", DeprecationWarning, stacklevel=2)
        from agent.agent_init import init_agent
        init_agent(self, **init_kwargs)

    def _get_session_db_for_recall(self):
        """SessionDB for recall, opening the default state DB when no ``session_db`` was passed so the
        advertised ``session_search`` tool stays usable."""
        # Persistence-isolated forks (background review) must not lazily open the canonical state DB —
        # that would re-arm the flush to write the fork's harness turn into the user's real session.
        if getattr(self, "_persist_disabled", False):
            return None
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state_registry import acquire

            self._session_db = acquire()
            self._owns_session_db = True  # we opened it, so close() must release it
            return self._session_db
        except Exception:
            logger.debug("SessionDB unavailable for recall", exc_info=True)
            return None

    def _session_row_model_config(self) -> Any:
        """``model_config`` for the session row: the init config plus the live YOLO bypass.

        The row is created lazily on the first turn, so this is the only chance to record a pre-first-turn
        /yolo toggle for ``hermes --resume``.
        """
        model_config = self._session_init_model_config
        try:
            from tools.approval import is_session_yolo_enabled
            if is_session_yolo_enabled(self.session_id):
                model_config = dict(model_config or {})
                model_config["yolo_mode"] = True
        except Exception:
            pass
        return model_config

    def _ensure_db_session(self) -> None:
        """Create the session DB row on first use; a transient failure leaves it to retry next turn."""
        if getattr(self, "_persist_disabled", False) or self._session_db_created or not self._session_db:
            return
        source = _session_source_for_agent(self.platform)
        try:
            # Persist the profile name explicitly, including "default": profile-keyed consumers treat NULL
            # as unowned.
            try:
                from hermes_cli.profiles import get_active_profile_name
                profile_for_session = get_active_profile_name()
            except Exception:
                # Persist the profile name EXPLICITLY, including "default". NULL used to stand in for the
                # default profile, but the #94724 legacy-owner backfill already stamps literal "default"
                # onto old rows, and profile-keyed consumers (sidebar scope matching,
                # @session:<profile>/<id> deep links) treat NULL as unowned — rows minted NULL after the
                # one-shot backfill vanished from the sidebar (#99222).
                profile_for_session = None
            # Carry the gateway routing identity: when the gateway SessionStore degraded to JSONL (corrupt
            # state.db) this lazy create is the ONLY durable write, and an identity-less row is unrecoverable.
            self._session_db.create_session(
                session_id=self.session_id, source=source, model=self.model,
                model_config=self._session_row_model_config(), system_prompt=self._cached_system_prompt,
                user_id=getattr(self, "_user_id", None), session_key=getattr(self, "_gateway_session_key", None),
                chat_id=getattr(self, "_chat_id", None), chat_type=getattr(self, "_chat_type", None),
                thread_id=getattr(self, "_thread_id", None),
                display_name=getattr(self, "_chat_name", None) or getattr(self, "_user_name", None),
                origin_json=_gateway_origin_json(self), parent_session_id=self._parent_session_id,
                cwd=_launch_cwd_for_session(source), profile_name=profile_for_session,
            )
            self._session_db_created = True
        except Exception as e:
            # Transient failure (e.g. SQLite lock): _session_db_created stays False so the next turn retries.
            logger.warning("Session DB creation failed (will retry next turn): %s", e)

    def _transition_context_engine_session(
        self, *, old_session_id: Optional[str] = None, new_session_id: Optional[str] = None,
        previous_messages: Optional[list] = None, carry_over_context: bool = False, reset_engine: bool = True,
        **extra_context,
    ) -> None:
        """Drive the context engine's session transition: on_session_end → on_session_reset → on_session_start
        → carry_over_new_session_context. Each hook is optional (the built-in compressor only resets)."""
        engine = getattr(self, "context_compressor", None)
        if not engine:
            return
        if old_session_id and previous_messages is not None:
            _call_engine_hook(engine, "on_session_end", old_session_id, previous_messages)
        if reset_engine:
            _call_engine_hook(engine, "on_session_reset")

        should_start = bool(old_session_id or previous_messages is not None or carry_over_context or extra_context)
        target_session_id = new_session_id or getattr(self, "session_id", "") or ""
        if should_start and target_session_id and hasattr(engine, "on_session_start"):
            start_context = {
                "old_session_id": old_session_id, "carry_over_context": carry_over_context,
                "platform": _session_source_for_agent(getattr(self, "platform", None)),
                "model": getattr(self, "model", ""), "context_length": getattr(engine, "context_length", None),
                "conversation_id": getattr(self, "_gateway_session_key", None), **extra_context,
            }
            start_context = {k: v for k, v in start_context.items() if v not in (None, "")}
            _call_engine_hook(engine, "on_session_start", target_session_id, **start_context)
        if carry_over_context and old_session_id and target_session_id:
            _call_engine_hook(engine, "carry_over_new_session_context", old_session_id, target_session_id)

    def reset_session_state(self, previous_messages: Optional[list] = None, old_session_id: Optional[str] = None,
                            carry_over_context: bool = False):
        """Reset session-scoped token/cost counters and compressor state for a fresh session.

        With ``previous_messages`` / ``old_session_id`` / ``carry_over_context`` the context engine gets the
        full transition lifecycle instead of a bare reset.
        """
        for counter in (
            "session_total_tokens", "session_input_tokens", "session_output_tokens", "session_prompt_tokens",
            "session_completion_tokens", "session_cache_read_tokens", "session_cache_write_tokens",
            "session_reasoning_tokens", "session_api_calls",
        ):
            setattr(self, counter, 0)
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"

        # Session boundary: the usage anchor describes the OLD transcript; fall back to full estimation.
        self._usage_anchor = None
        self._turn_base_usage_anchor = None

        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0
        # Copilot x-initiator: True for the first API call of a user turn, False for tool-loop follow-ups.
        self._is_user_initiated_turn = False

        self._transition_context_engine_session(
            old_session_id=old_session_id, new_session_id=getattr(self, "session_id", None),
            previous_messages=previous_messages, carry_over_context=carry_over_context, reset_engine=True,
        )

        # Reset-only switches (/new, /resume, /branch) change session_id before this call; rebind the
        # built-in compressor's session-keyed cooldown state when no full start hook ran.
        engine = getattr(self, "context_compressor", None)
        target_session_id = getattr(self, "session_id", "") or ""
        if (engine is not None and hasattr(engine, "bind_session_state") and target_session_id
                and target_session_id != getattr(engine, "_session_id", "")):
            try:
                engine.bind_session_state(getattr(self, "_session_db", None), target_session_id)
            except Exception as exc:
                logger.debug("context engine bind_session_state during reset: %s", exc)

    @staticmethod
    def _effective_lmstudio_context_length(config_context_length: Optional[int], runtime_context_length: Any) -> Optional[int]:
        """Return a safe context budget from explicit intent and verified runtime."""
        explicit = _positive_int(config_context_length)
        runtime = _positive_int(getattr(runtime_context_length, "context_length", runtime_context_length))
        if bool(getattr(runtime_context_length, "rejected", False)) or (
            bool(getattr(runtime_context_length, "load_attempted", False)) and runtime is None
        ):
            return None
        if runtime is not None and explicit is not None:
            return min(runtime, explicit)
        return runtime if runtime is not None else explicit

    @staticmethod
    def _lmstudio_load_was_unverified(load_result: Any) -> bool:
        """Return true when a management load was rejected or unverifiable."""
        return bool(getattr(load_result, "rejected", False)) or (
            bool(getattr(load_result, "load_attempted", False)) and getattr(load_result, "context_length", None) is None
        )

    def _ensure_lmstudio_runtime_loaded(self, config_context_length: Optional[int] = None) -> Any:
        """Preload LM Studio unless configured to rely on JIT loading."""
        if (self.provider or "").strip().lower() != "lmstudio":
            return None
        if (getattr(self, "lmstudio_load_mode", "explicit") or "explicit").strip().lower() == "jit":
            logger.debug("LM Studio explicit preload skipped: lmstudio_load_mode=jit")
            return None
        from hermes_cli.models_local import ensure_lmstudio_model_loaded

        if config_context_length is None:
            config_context_length = getattr(self, "_config_context_length", None)
        return ensure_lmstudio_model_loaded(
            self.model, self.base_url, getattr(self, "api_key", ""), config_context_length, return_load_result=True,
        )

    switch_model = _forward("agent.agent_runtime_helpers", "switch_model")

    def _disable_codex_reasoning_replay(self, messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
        """On HTTP 400 ``invalid_encrypted_content``: disable Responses reasoning replay and pop
        ``codex_reasoning_items`` from every assistant message. Returns ``{"messages", "items"}`` counts."""
        stripped_messages = stripped_items = 0
        for msg in (messages if isinstance(messages, list) else []):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            items = msg.pop("codex_reasoning_items", None)
            if isinstance(items, list) and items:
                stripped_messages += 1
                stripped_items += len(items)
        self._codex_reasoning_replay_enabled = False
        return {"messages": stripped_messages, "items": stripped_items}

    _stream_diag_init = _forward_static("agent.stream_diag", "stream_diag_init")
    _stream_diag_capture_response = _forward("agent.stream_diag", "stream_diag_capture_response")
    _flatten_exception_chain = _forward_static("agent.stream_diag", "flatten_exception_chain")

    def _is_provider_stream_parse_error(self, error: BaseException) -> bool:
        """True for a malformed Anthropic event-stream frame (surfaced by the SDK as a plain ``ValueError``);
        that is wire trouble, not local validation, so it follows the truncated-JSON retry path."""
        return (getattr(self, "api_mode", None) == "anthropic_messages" and isinstance(error, ValueError)
                and not isinstance(error, (UnicodeEncodeError, json.JSONDecodeError))
                and "expected ident at line" in str(error).strip().lower())

    _log_stream_retry = _forward("agent.stream_diag", "log_stream_retry")
    _emit_stream_drop = _forward("agent.stream_diag", "emit_stream_drop")

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
        return {key: getattr(self, key, "") or "" for key in ("model", "provider", "base_url", "api_key", "api_mode", "auth_mode")}

    _check_compression_model_feasibility = _forward("agent.conversation_compression", "check_compression_model_feasibility")
    _replay_compression_warning = _forward("agent.conversation_compression", "replay_compression_warning")

    def _hostname_for(self, base_url: Optional[str]) -> str:
        """Hostname of ``base_url``, or of the agent's own base URL when None."""
        if base_url is not None:
            return base_url_hostname(base_url)
        return getattr(self, "_base_url_hostname", "") or base_url_hostname(getattr(self, "_base_url_lower", ""))

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        return self._hostname_for(base_url) == "api.openai.com"

    def _is_azure_openai_url(self, base_url: str = None) -> bool:
        """True when a base URL targets Azure OpenAI (standard client, but NO Responses API support)."""
        url = str(base_url).lower() if base_url is not None else (getattr(self, "_base_url_lower", "") or "")
        return base_url_host_matches(url, "openai.azure.com")

    def _is_github_copilot_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets GitHub Copilot's OpenAI-compatible API."""
        hostname = self._hostname_for(base_url)
        return bool(hostname) and (hostname == "api.githubcopilot.com" or hostname.endswith(".githubcopilot.com"))

    def _resolved_api_call_timeout(self) -> float:
        """Per-call request timeout: per-model ``timeout_seconds`` > provider ``request_timeout_seconds`` >
        ``HERMES_API_TIMEOUT`` > 1800s."""
        cfg = get_provider_request_timeout(self.provider, self.model)
        return cfg if cfg is not None else env_float("HERMES_API_TIMEOUT", 1800.0)

    def _resolved_api_call_stale_timeout_base(self) -> tuple[float, bool]:
        """Base non-stream stale timeout: per-model ``stale_timeout_seconds`` > provider-wide >
        ``HERMES_API_CALL_STALE_TIMEOUT`` > reasoning floor > 90s.

        Returns ``(seconds, uses_implicit_default)``; the implicit flag lets callers auto-disable the detector
        for local endpoints only when the user configured nothing.
        """
        cfg = get_provider_stale_timeout(self.provider, self.model)
        if cfg is not None:
            return cfg, False
        env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT")
        if env_timeout is not None:
            return float(env_timeout), False
        # Reasoning-model floor (cloud gateways idle-kill mid-think); not "implicit" so the local-endpoint
        # short-circuit does not disable stale detection here.
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        reasoning_floor = get_reasoning_stale_timeout_floor(self.model)
        if reasoning_floor is not None:
            return reasoning_floor, False
        return 90.0, True

    def _compute_non_stream_stale_timeout(self, api_payload: Any) -> float:
        """Effective non-stream stale timeout for ``api_payload`` (an ``api_kwargs`` dict or legacy ``messages``
        list), scaled by estimated context size and capped by the run budget."""
        stale_base, uses_implicit_default = self._resolved_api_call_stale_timeout_base()
        base_url = getattr(self, "_base_url", None) or self.base_url or ""
        if uses_implicit_default and base_url and is_local_endpoint(base_url):
            return float("inf")

        from agent.chat_completion_helpers import estimate_request_context_tokens
        est_tokens = estimate_request_context_tokens(api_payload)
        timeout = max(stale_base, 240.0) if est_tokens > 100_000 else max(stale_base, 150.0) if est_tokens > 50_000 else stale_base
        # Run-budget cap: an implicit stale timeout is capped at half the remaining budget (>= 60s) so one
        # hung call cannot eat the run. Never raises the timeout; explicit user config still wins.
        run_budget = getattr(self, "run_budget_seconds", None)
        started = getattr(self, "_run_budget_started_at", None)
        if run_budget and started and not self._stale_timeout_is_explicit():
            remaining = float(run_budget) - (time.time() - started)
            timeout = min(timeout, max(60.0, remaining * 0.5))
        return timeout

    def _stale_timeout_is_explicit(self) -> bool:
        """True when the user explicitly configured the stale timeout (config or env var); implicit values
        (reasoning floors, the 90s default) yield to the run-budget cap, explicit ones never do."""
        return (get_provider_stale_timeout(self.provider, self.model) is not None
                or os.getenv("HERMES_API_CALL_STALE_TIMEOUT") is not None)

    def _codex_silent_hang_hint(self, model: Optional[str] = None) -> Optional[str]:
        """Actionable hint when the request matches a known Codex silent-reject shape (currently the ``gpt-5.5``
        family: connection accepted, no events, no error), else None. Makes the stale timeout actionable."""
        if self.api_mode != "codex_responses":
            return None
        from agent.codex_responses_adapter import classify_responses_route

        if not classify_responses_route(self).is_codex_backend:
            return None
        eff_model = (model if model is not None else self.model) or ""
        # Match the gpt-5.5 family at word boundaries (bare, -codex, vendor-prefixed) but not gpt-5.50.
        if not re.search(r"(?:^|[/\-_])gpt-5\.5(?:$|[\-_])", eff_model.lower()):
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
        return any(base_url_host_matches(self._base_url_lower, h) for h in ("api.githubcopilot.com", "models.github.ai"))

    def _is_copilot_provider(self) -> bool:
        """True when the active provider is GitHub Copilot under any alias (``copilot`` / ``github-copilot`` /
        ``github``) or by base URL; a bare equality check would silently skip credential recovery."""
        return (self.provider or "").strip().lower() in {"copilot", "github-copilot", "github"} or self._is_copilot_url()

    def _is_codex_backend(self) -> bool:
        """Return True for the ChatGPT OAuth Codex Responses backend."""
        return (getattr(self, "api_mode", None) == "codex_responses"
                and getattr(self, "_base_url_hostname", "") == "chatgpt.com"
                and "/backend-api/codex" in (getattr(self, "_base_url_lower", "") or ""))

    _anthropic_prompt_cache_policy = _forward("agent.agent_runtime_helpers", "anthropic_prompt_cache_policy")
    _direct_native_anthropic_tool_cache_capability = _forward("agent.agent_runtime_helpers", "_direct_native_anthropic_tool_cache_capability")

    @staticmethod
    def _model_requires_responses_api(model: str) -> bool:
        """True for GPT-5.x, which OpenAI and OpenRouter reject on /v1/chat/completions
        (``unsupported_api_for_model``)."""
        return model.lower().rsplit("/", 1)[-1].startswith("gpt-5")  # strip vendor prefix ("openai/gpt-5.4")

    @staticmethod
    def _provider_model_requires_responses_api(model: str, *, provider: Optional[str] = None) -> bool:
        """Return True when this provider/model pair should use Responses API."""
        normalized_provider = (provider or "").strip().lower()
        # Nous serves GPT-5.x via chat completions (its /v1/responses returns 404); generic custom endpoints
        # may relay GPT-5 without full Responses semantics — only direct OpenAI/xAI URLs auto-upgrade.
        if normalized_provider in ("nous", "custom"):
            return False
        if normalized_provider == "copilot":
            try:
                from hermes_cli.models import _should_use_copilot_responses_api
                return _should_use_copilot_responses_api(model)
            except Exception:
                pass  # fall back to the generic GPT-5 rule
        return AIAgent._model_requires_responses_api(model)

    def _max_tokens_param(self, value: int) -> dict:
        """``max_completion_tokens`` for newer OpenAI families (and Azure / Copilot serving them), else
        ``max_tokens``. URL-first, then model-name fallback for third-party endpoints fronting those models."""
        if (self._is_direct_openai_url() or self._is_azure_openai_url() or self._is_github_copilot_url()
                or model_forces_max_completion_tokens(self.model)):
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    @staticmethod
    def _requested_output_cap_from_api_kwargs(api_kwargs: Any) -> Optional[int]:
        """Extract the outgoing response token cap from a prepared request."""
        if not isinstance(api_kwargs, dict):
            return None
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            try:
                value = int(api_kwargs.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _has_content_after_think_block(self, content: str) -> bool:
        """True when text remains after stripping reasoning blocks (reasoning-only output is retried)."""
        return bool(content) and bool(self._strip_think_blocks(content).strip())

    _strip_think_blocks = _forward("agent.agent_runtime_helpers", "strip_think_blocks")

    @staticmethod
    def _has_natural_response_ending(content: str) -> bool:
        """Heuristic: does visible assistant text look intentionally finished?"""
        stripped = (content or "").rstrip()
        if not stripped:
            return False
        last = stripped[-1]
        # Closing punctuation/brackets, a fenced-code close, or an emoji (Misc Symbols, Dingbats, Emoticons, ...).
        return stripped.endswith("```") or last in '.!?:)"\']}。！？：）】」』》^' or ord(last) >= 0x1F300

    def _is_ollama_glm_backend(self) -> bool:
        """Ollama-hosted GLM models misreport finish_reason='stop'. Matches only explicit Ollama signatures
        (port 11434, "ollama" in URL, provider ollama), never arbitrary local proxies; excludes Ollama Cloud
        (``ollama.com`` / ``:cloud``), which reports faithfully — rewriting it would manufacture truncations.

        Crucially it does NOT match arbitrary local/private endpoints (LiteLLM/sglang/vLLM/LM Studio
        proxies, Tailscale boxes), which report finish_reason correctly and were the source of #13971's
        false-positive truncation continuations.
        Two signatures identify it: the ``ollama.com`` host (provider ``ollama-cloud``) and the ``:cloud``
        model suffix (cloud generation proxied through a local 11434 endpoint, #98406). Applying the
        stop→length rewrite to them manufactures false truncations and causes the continuation nudge to
        consume the model's output budget on the next retry, making further false-positives more likely.
        """
        model_lower = (self.model or "").lower()
        provider_lower = (self.provider or "").lower()
        if "glm" not in model_lower and provider_lower != "zai":
            return False
        base = self._base_url_lower
        # Ollama Cloud (hosted service or :cloud proxy) forwards finish_reason faithfully — do not rewrite.
        if "ollama.com" in base or ":cloud" in model_lower:
            return False
        if "ollama" in base or ":11434" in base:
            return True
        return provider_lower == "ollama"

    def _should_treat_stop_as_truncated(self, finish_reason: str, assistant_message, messages: Optional[list] = None) -> bool:
        """Detect conservative stop->length misreports for Ollama-hosted GLM models."""
        if finish_reason != "stop" or self.api_mode != "chat_completions" or not self._is_ollama_glm_backend():
            return False
        if not any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in (messages or [])):
            return False
        if assistant_message is None or getattr(assistant_message, "tool_calls", None):
            return False
        content = getattr(assistant_message, "content", None)
        if not isinstance(content, str):
            return False
        visible_text = self._strip_think_blocks(content).strip()
        if len(visible_text) < 20 or not re.search(r"\s", visible_text):
            return False
        return not self._has_natural_response_ending(visible_text)

    _looks_like_codex_intermediate_ack = _forward("agent.agent_runtime_helpers", "looks_like_codex_intermediate_ack")
    _extract_reasoning = _forward("agent.agent_runtime_helpers", "extract_reasoning")
    _cleanup_task_resources = _forward("agent.chat_completion_helpers", "cleanup_task_resources")

    # Background memory/skill review — prompts live in agent.background_review.
    from agent.background_review import _MEMORY_REVIEW_PROMPT, _SKILL_REVIEW_PROMPT, _COMBINED_REVIEW_PROMPT
    _summarize_background_review_actions = _forward_static("agent.background_review", "summarize_background_review_actions")

    def _spawn_background_review(self, messages_snapshot: List[Dict], review_memory: bool = False,
                                 review_skills: bool = False, focus: Optional[str] = None, explicit: bool = False) -> None:
        """Post-turn review entry point: decide WHEN, then spawn.

        A review whose runtime is the MANAGED LOCAL llama-server is queued for machine idle (``defer: auto``)
        instead of hitting the user's GPU mid-session; everything else spawns immediately. ``explicit``
        (/refine) is never deferred but does not touch the ``focus``-keyed delegate/enabled gates.
        """
        # Gates run at enqueue/spawn time; the idle dispatcher re-checks `enabled` at dispatch time.
        if focus is None and getattr(self, "_delegate_depth", 0) > 0:
            return
        task_cfg = None
        if focus is None:
            from agent.background_review import load_background_review_settings
            enabled, task_cfg = load_background_review_settings()
            if not enabled:
                return

        # Structural clone at the single chokepoint: the fork sanitizes in place, and a shallow copy would
        # alias the live history's nested tool_calls/content.
        # Structural clone at the single chokepoint every review path (automatic, /refine, idle-queue
        # deferral) goes through. See #100795.
        from agent.turn_finalizer import _clone_background_review_messages
        kwargs = dict(messages_snapshot=_clone_background_review_messages(messages_snapshot),
                      review_memory=review_memory, review_skills=review_skills, focus=focus, task_cfg=task_cfg)
        if focus is None and not explicit and _review_should_defer(self, task_cfg):
            from agent.review_idle_queue import QUEUE
            QUEUE.enqueue(self, _review_queue_key(self), kwargs)
            return
        self._spawn_background_review_now(**kwargs)

    def _spawn_background_review_now(self, messages_snapshot: List[Dict], review_memory: bool = False,
                                     review_skills: bool = False, focus: Optional[str] = None,
                                     task_cfg: Optional[Dict[str, Any]] = None, _requeue_attempts: int = 0) -> None:
        """Spawn the background memory/skill review thread.

        ``threading.Thread`` is constructed here so tests patching ``run_agent.threading.Thread`` keep working.
        ``focus`` is /refine steering text; ``task_cfg`` is the pre-loaded config block (None on direct calls).
        A deferred review preempted by a live turn is requeued (bounded) rather than lost.
        """
        from agent.background_review import (
            finish_background_review_run, prepare_background_review_run, spawn_background_review_thread,
        )
        from tools.thread_context import propagate_context_to_thread

        review_run = prepare_background_review_run(self)
        if review_run is None:
            return
        try:
            target, _prompt = spawn_background_review_thread(
                self, messages_snapshot, review_memory=review_memory, review_skills=review_skills,
                focus=focus, task_cfg=task_cfg, review_run=review_run,
            )

            def _target_with_requeue() -> None:
                target()
                self._maybe_requeue_preempted_review(review_run, dict(
                    messages_snapshot=messages_snapshot, review_memory=review_memory, review_skills=review_skills,
                    focus=focus, task_cfg=task_cfg, _requeue_attempts=_requeue_attempts + 1))

            # Carry the active profile into the review thread so MEMORY.md / skill review writes land in the
            # right profile.
            threading.Thread(target=propagate_context_to_thread(_target_with_requeue), daemon=True, name="bg-review").start()
        except Exception:
            finish_background_review_run(self, review_run)
            raise

    _REVIEW_REQUEUE_MAX_ATTEMPTS = 3

    def _maybe_requeue_preempted_review(self, review_run, kwargs) -> None:
        """Requeue a deferred-mode review that a live turn cancelled.

        Only for automatic reviews on the managed local runtime; bounded attempts stop a busy box cycling
        forever.
        """
        try:
            # Not cancelled == ran to completion (or was never admitted).
            if not review_run.cancel_requested.is_set() or kwargs.get("focus") is not None:
                return
            if kwargs.get("_requeue_attempts", 0) > self._REVIEW_REQUEUE_MAX_ATTEMPTS:
                logger.info("Preempted background review dropped after %d requeues", self._REVIEW_REQUEUE_MAX_ATTEMPTS)
                return
            if not _review_should_defer(self, kwargs.get("task_cfg")):
                return
            from agent.review_idle_queue import QUEUE
            # kwargs carries the incremented _requeue_attempts through the queue so the cap survives.
            QUEUE.enqueue(self, _review_queue_key(self), dict(kwargs))
        except Exception:  # noqa: BLE001 — requeue is best-effort
            logger.debug("Preempted-review requeue failed", exc_info=True)

    _build_memory_write_metadata = _forward("agent.background_review", "build_memory_write_metadata")
    _apply_pending_steer_to_tool_results = _forward("agent.agent_runtime_helpers", "apply_pending_steer_to_tool_results")

    def get_activity_summary(self) -> dict:
        """Diagnostic snapshot: ``last_activity_*`` plus the short aliases gateway and delegate readers use."""
        from agent.session_activity import build_activity_snapshot

        provenance = getattr(self, "_last_activity_provenance", None)
        return build_activity_snapshot(
            last_activity_at=getattr(self, "_last_activity_ts", None),
            last_activity_description=getattr(self, "_last_activity_desc", None) or "",
            last_activity_provenance=provenance if provenance is not None else ActivityProvenance.UNKNOWN,
            extra={
                "current_tool": self._current_tool, "api_call_count": self._api_call_count,
                "max_iterations": self.max_iterations, "budget_used": self.iteration_budget.used,
                "budget_max": self.iteration_budget.max_total,
            },
        )

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down the memory provider and context engine at session end (idempotent: gateway cleanup and
        ``close()`` may both call it)."""
        if getattr(self, "_memory_provider_shutdown", False):
            return
        self._memory_provider_shutdown = True
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception as e:
                logger.warning("Memory provider on_session_end failed during shutdown: %s", e, exc_info=True)
            _quietly(lambda: self._memory_manager.shutdown_all())
        _notify_context_engine_session_end(self, messages)

    def commit_memory_session(self, messages: list = None) -> None:
        """Flush end-of-session extraction on session_id rotation (/new, compression) without tearing providers
        down."""
        if self._memory_manager:
            _quietly(lambda: self._memory_manager.on_session_end(messages or []))
        _notify_context_engine_session_end(self, messages)

    def _sync_external_memory_for_turn(self, *, original_user_message: Any, final_response: Any, interrupted: bool,
                                       messages: list | None = None) -> None:
        """Mirror a completed turn into external memory providers (``sync_all`` + ``queue_prefetch_all``).

        Uses ``original_user_message`` (``user_message`` may carry injected skill content). Interrupted turns
        are skipped: partial output is not durable truth. Best-effort — an offline backend never blocks.

        A partial assistant output, an aborted tool chain, or a mid-stream reset is not durable
        conversational truth — mirroring it into an external memory backend pollutes future recall with
        state the user never saw completed. The prefetch is gated on the same flag: the user's next message
        is almost certainly a retry of the same intent, and a prefetch keyed on the interrupted turn would
        fire against stale context. See #15218.
        """
        if interrupted or not (self._memory_manager and final_response and original_user_message):
            return
        # Flatten multimodal parts to text (newline-joined for memory).
        user_text = _summarize_user_message_for_log(original_user_message, sep="\n")
        response_text = _summarize_user_message_for_log(final_response, sep="\n")
        if not (user_text and response_text):
            return
        try:
            sync_kwargs = {"session_id": self.session_id or "", **({"messages": messages} if messages is not None else {})}
            self._memory_manager.sync_all(user_text, response_text, **sync_kwargs)
            # Sibling of the build_turn_context() prefetch gate: don't key recall on zero-signal prompts.
            if not is_trivial_prompt(user_text):
                self._memory_manager.queue_prefetch_all(user_text, session_id=self.session_id or "")
        except Exception:
            pass

    def release_clients(self) -> None:
        """Release LLM clients and child agents WITHOUT tearing down session tool state (gateway cache
        eviction: the session may resume on the same task_id, so processes, sandbox, browser, computer-use and
        memory provider are kept). Idempotent; distinct from ``close()``."""
        self._close_active_children(soft=True)
        # Retire (don't hard-close) the shared client: eviction runs on the gateway memory-manager thread,
        # and a cross-thread close can release TLS FDs under a still-unwinding worker.
        _quietly(self._drop_shared_client, lambda c: self._retire_shared_openai_client(c, reason="cache_evict"))
        self._close_request_clients("cache_evict")

    def close(self) -> None:
        """Release every resource this agent holds (idempotent); each phase is guarded so one failure never
        blocks the rest."""
        # close() is the hard owner boundary; shutdown_memory_provider() is idempotent so gateway pre-calls
        # never double-extract.
        session_messages = getattr(self, "_session_messages", None)
        _quietly(self.shutdown_memory_provider, session_messages if isinstance(session_messages, list) else None)
        self._close_task_resources(getattr(self, "session_id", None) or "")
        self._close_active_children(soft=False)
        _quietly(self._drop_shared_client, lambda c: self._close_openai_client(c, reason="agent_close", shared=True))
        self._close_request_clients("agent_close")
        _quietly(self._close_codex_session)
        # Free conversation history proactively: callers may still hold the closed agent. The DB-flush
        # settled-prefix snapshot and the streamed-text accumulator are shadow copies of the same transcript;
        # on a closed delegate child they were the only remaining owners, pinning its history in the parent heap.
        self._session_messages = []
        self._db_flush_scan_prefix = None
        self._streamed_assistant_text_parts = []
        _quietly(self._trim_process_memory)
        _quietly(self._finalize_owned_session_row)

    # -- close()/release_clients() phases -------------------------------------------------------------

    def _close_task_resources(self, task_id: str) -> None:
        """Kill this task's background processes, then its terminal sandbox, browser daemon and computer-use
        backend (lazy imports keep the core footprint narrow)."""
        def kill_processes() -> None:
            from tools.process_registry import process_registry
            process_registry.kill_all(task_id=task_id)

        def release_computer_use() -> None:
            from tools.computer_use.tool import release_computer_use_session
            release_computer_use_session(task_id)

        for step in (kill_processes, lambda: cleanup_vm(task_id), lambda: cleanup_browser(task_id), release_computer_use):
            _quietly(step)

    def _close_active_children(self, *, soft: bool) -> None:
        """Detach and close per-turn child agents; ``soft`` releases their clients first, falling back to close()."""
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
        except Exception:
            return
        for child in children:
            if soft:
                try:
                    child.release_clients()
                    continue
                except Exception:
                    pass
            _quietly(lambda: child.close())

    def _drop_shared_client(self, close_fn: Callable[[Any], None]) -> None:
        """Hand the shared OpenAI/httpx client to ``close_fn`` and clear the attribute."""
        # Retire the OpenAI/httpx client to release sockets immediately. #70773: eviction runs on the
        # gateway's memory-manager thread — a cross-thread hard close of the shared client can release TLS
        # FDs under a still-unwinding worker (FD-recycle → SQLite corruption). Retirement shuts the pooled
        # sockets down (the memory/socket win we want here) and lets GC release the FDs once no thread holds
        # them.
        client = getattr(self, "client", None)
        if client is not None:
            close_fn(client)
            self.client = None

    def _close_request_clients(self, reason: str) -> None:
        """Drop the cached per-request wire clients (reused across sequential LLM calls)."""
        _quietly(self._close_cached_request_openai_client, reason=reason)
        _quietly(self._close_cached_request_anthropic_client, reason=reason)

    def _close_codex_session(self) -> None:
        """Close the Codex app-server session (else the child keeps running); the attribute is cleared BEFORE
        close() so a concurrent reader can't grab a half-closed session."""
        codex_session = getattr(self, "_codex_session", None)
        if codex_session is not None:
            self._codex_session = None
            codex_session.close()

    @staticmethod
    def _trim_process_memory() -> None:
        """Return freed heap pages to the OS on glibc; safe no-op elsewhere."""
        from hermes_cli.mem_trim import trim_memory
        trim_memory(force=True, reason="agent close")

    def _finalize_owned_session_row(self) -> None:
        """End the session row unless ownership was handed forward (compression helpers, review forks sharing
        the parent's id; end_session() is first-reason-wins), then release the SQLite handle ONLY when this
        agent owns it — a dedicated handle left open pins its fds and token-writer thread for the process
        lifetime. The owner flag is cleared first so close() stays idempotent."""
        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "session_id", None)
        if getattr(self, "_end_session_on_close", True) and session_db and session_id:
            _quietly(lambda: session_db.end_session(session_id, "agent_close"))
        if getattr(self, "_owns_session_db", False) and session_db is not None:
            self._owns_session_db = False
            # Shared instances no-op on close(); release the refcount so the registry closes on the last caller.
            # See #90837.
            from hermes_state_registry import release_or_close
            release_or_close(session_db)

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """Replay the most recent todo tool response (the gateway builds a fresh AIAgent per message). Only
        results paired with an earlier assistant ``todo`` call count — a forged bare ``role: tool`` message
        must not seed the store (GHSA-5g4g-6jrg-mw3g)."""
        found = self._latest_todo_response(history)
        if found is not None:
            last_todo_response, last_todo_revision = found
            # Restore only when history carries a newer revision than the store holds; empty lists are an
            # authoritative clear.
            try:
                history_revision = max(0, int(last_todo_revision or 0))
            except (TypeError, ValueError):
                history_revision = 1
            if history_revision > int(self._todo_store.snapshot().get("revision", 0) or 0):
                self._todo_store.restore(last_todo_response, revision=history_revision)
                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)

    def _latest_todo_response(self, history: List[Dict[str, Any]]) -> Optional[tuple]:
        """Walk history backwards for the newest paired, size-bounded todo result → ``(todos, revision)``."""
        from tools.todo_tool import MAX_TODO_RESULT_CHARS

        for idx in range(len(history) - 1, -1, -1):
            msg = history[idx]
            content = msg.get("content", "")
            if msg.get("role") != "tool" or not isinstance(content, str) or not self._tool_response_matches_todo_call(history, idx):
                continue
            if len(content) > MAX_TODO_RESULT_CHARS:
                logger.warning("Skipping oversized todo tool response during hydration: "
                               "session=%s chars=%d", self.session_id or "none", len(content))
                continue
            if '"todos"' not in content:  # cheap pre-filter before json.loads
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            if "todos" in data and isinstance(data["todos"], list):
                return data["todos"], data.get("revision", 1)
        return None

    @classmethod
    def _tool_response_matches_todo_call(cls, history: List[Dict[str, Any]], tool_index: int) -> bool:
        """True when the nearest prior assistant message issued a ``todo`` call with this ``tool_call_id``; a
        ``user``/``system`` boundary or missing id means unpaired → must not hydrate."""
        tool_call_id = history[tool_index].get("tool_call_id") if 0 <= tool_index < len(history) else None
        if not tool_call_id:
            return False
        for prior in reversed(history[:tool_index]):
            role = prior.get("role")
            if role == "assistant":
                return cls._assistant_has_todo_tool_call(prior, tool_call_id)
            if role in {"user", "system"}:
                return False
        return False

    @classmethod
    def _assistant_has_todo_tool_call(cls, assistant_msg: Dict[str, Any], tool_call_id: str) -> bool:
        """True when the assistant message issued a ``todo`` call with this id."""
        tool_calls = assistant_msg.get("tool_calls")
        return isinstance(tool_calls, list) and any(
            cls._get_tool_call_id_static(tc) == tool_call_id and cls._get_tool_call_name_static(tc) == "todo"
            for tc in tool_calls
        )

    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested

    _build_system_prompt = _forward("agent.system_prompt", "build_system_prompt")

    # Call ID of a tool_call entry (dict or object); policy owner: ``message_sanitization.coalesce_tool_call_id``.
    _get_tool_call_id_static = staticmethod(_sanitize_coalesce_tool_call_id)

    @staticmethod
    def _get_tool_call_name_static(tc) -> str:
        """Function name of a tool_call entry (dict or object); Gemini requires it on every ``role: tool`` message."""
        if isinstance(tc, dict):
            fn = tc.get("function")
            return (fn.get("name", "") or "") if isinstance(fn, dict) else ""
        return getattr(getattr(tc, "function", None), "name", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})
    _sanitize_api_messages = _forward_static("agent.agent_runtime_helpers", "sanitize_api_messages")

    @staticmethod
    def _is_thinking_only_assistant(msg: Dict[str, Any], *, drop_codex_reasoning_items: bool = True) -> bool:
        """True if ``msg`` is an assistant turn whose only payload is reasoning (no text, no tool_calls).

        Providers converting reasoning to thinking blocks reject it (400 "final block cannot be thinking"), so
        the turn is dropped from the API copy; the transcript keeps the reasoning block.
        """
        if not isinstance(msg, dict) or msg.get("role") != "assistant" or msg.get("tool_calls"):
            return False
        # Prefill stubs are thinking-only by construction; checked before content inspection since
        # repair_empty_non_final_messages may have healed content.
        if msg.get("_thinking_prefill"):
            return True
        if AIAgent._content_has_real_payload(msg.get("content")):
            return False
        # A native compaction checkpoint makes a carrier never thinking-only, regardless of api_mode or
        # reasoning field. Checked above every reasoning branch so no carrier shape is dropped.
        # The checkpoint is the server-side stand-in for already-pruned history and exists in exactly one
        # place; the codex_responses adapter also surfaces commentary text via msg["reasoning"], so the
        # string branch below would otherwise drop a carrier before the sidecar is ever inspected. See
        # #82108.
        from agent.native_compaction import has_compaction_checkpoint

        if has_compaction_checkpoint(msg.get("codex_reasoning_items")):
            return False
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        rd = msg.get("reasoning_details")
        if (isinstance(reasoning, str) and reasoning.strip()) or (isinstance(rd, list) and rd):
            return True
        # Codex Responses keeps encrypted reasoning under a separate key; only real items count as
        # thinking-only, empty/junk lists fall through to generic empty-turn handling.
        codex_items = msg.get("codex_reasoning_items")
        if drop_codex_reasoning_items and isinstance(codex_items, list):
            return any(isinstance(item, dict) and item.get("type") == "reasoning" for item in codex_items)
        return False

    @staticmethod
    def _content_has_real_payload(content: Any) -> bool:
        """True when assistant ``content`` carries anything beyond (redacted) thinking blocks / whitespace."""
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if block:  # non-empty non-dict string etc.
                        return True
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return True
                elif btype not in {"thinking", "redacted_thinking"}:
                    return True  # tool_use, image, document, etc. — real payload
            return False
        return content is not None and content != ""

    _drop_thinking_only_and_merge_users = _forward_static("agent.agent_runtime_helpers", "drop_thinking_only_and_merge_users")

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """Cap delegate_task calls in one turn at max_concurrent_children (non-delegate calls all kept);
        returns the original list when nothing was truncated."""
        from tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
        kept_delegates, truncated = 0, []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates >= max_children:
                    continue
                kept_delegates += 1
            truncated.append(tc)
        logger.warning("Truncated %d excess delegate_task call(s) to enforce "
                       "max_concurrent_children=%d limit", delegate_count - max_children, max_children)
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """Drop duplicate (tool_name, arguments) pairs in one turn (first wins). Valid JSON arguments are
        canonicalized so key order/whitespace can't evade dedup; returns the original list when nothing was removed."""
        seen, unique = set(), []
        for tc in tool_calls:
            arguments = tc.function.arguments
            try:
                arguments = json.dumps(json.loads(arguments), separators=(",", ":"), sort_keys=True)
            except (TypeError, ValueError):
                pass
            key = (tc.function.name, arguments)
            if key in seen:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
                continue
            seen.add(key)
            unique.append(tc)
        return unique if len(unique) < len(tool_calls) else tool_calls

    # Distinct ids per assistant turn, in place (policy owner: ``message_sanitization``). Collisions get a
    # deterministic ``<id>_d<n>`` suffix — never uuid4, for prompt-cache prefix stability.
    _uniquify_tool_call_ids = staticmethod(_sanitize_uniquify_tool_call_ids)

    _repair_tool_call = _forward("agent.agent_runtime_helpers", "repair_tool_call")
    _invalidate_system_prompt = _forward("agent.system_prompt", "invalidate_system_prompt")

    # Codex Responses id policy (agent.codex_responses_adapter): deterministic call ids when the API omits one
    # (random UUIDs would break the provider prompt cache), split stored ids, derive valid ``fc_`` ids.
    _deterministic_call_id = staticmethod(_codex_deterministic_call_id)
    _split_responses_tool_id = staticmethod(_codex_split_responses_tool_id)
    _derive_responses_function_call_id = staticmethod(_codex_derive_responses_function_call_id)

    _interruptible_api_call = _forward("agent.chat_completion_helpers", "interruptible_api_call")
    _interruptible_streaming_api_call = _forward("agent.chat_completion_helpers", "interruptible_streaming_api_call")
    _try_activate_fallback = _forward("agent.chat_completion_helpers", "try_activate_fallback")

    def _has_pending_fallback(self) -> bool:
        """Whether a fallback provider remains (mirrors ``try_activate_fallback``'s guard) — gates the
        "trying fallback..." status so we never announce one that won't be attempted.

        See #17446.
        """
        return getattr(self, "_fallback_index", 0) < len(getattr(self, "_fallback_chain", None) or [])

    _restore_primary_runtime = _forward("agent.agent_runtime_helpers", "restore_primary_runtime")
    _try_recover_primary_transport = _forward("agent.agent_runtime_helpers", "try_recover_primary_transport")
    _build_api_kwargs = _forward("agent.chat_completion_helpers", "build_api_kwargs")

    def _set_tool_guardrail_halt(self, decision: ToolGuardrailDecision) -> None:
        """Record the first guardrail decision that should stop this turn."""
        if decision.should_halt and self._tool_guardrail_halt_decision is None:
            self._tool_guardrail_halt_decision = decision

    def _toolguard_controlled_halt_response(self, decision: ToolGuardrailDecision) -> str:
        return (
            f"I stopped retrying {decision.tool_name or 'a tool'} because it hit the tool-call guardrail "
            f"({decision.code}) after {decision.count} repeated non-progressing "
            "attempts. The last tool result explains the blocker; the next step is "
            "to change strategy instead of repeating the same call."
        )

    def _append_guardrail_observation(self, tool_name: str, function_args: dict, function_result: str, *,
                                      failed: bool, tool_call_id: str = "") -> str:
        decision = self._tool_guardrails.after_call(tool_name, function_args, function_result, failed=failed)
        # Identical-call stall guards observe the RAW result (before the per-call loop suffix) and are applied
        # at result construction so tool results stay append-only / cache-safe.
        stall_notice = result_stub = None
        if self._stall_guards_enabled():
            try:
                observation = self._tool_guardrails.observe_call(
                    tool_name, function_args, function_result if isinstance(function_result, str) else None,
                    tool_call_id=tool_call_id, failed=failed,
                )
                stall_notice, result_stub = observation.notice, observation.stub
            except Exception as exc:
                logger.debug("stall-guard identical-call observation failed: %s", exc)
        # Result-reference stubbing: a 2nd+ identical call with a byte-identical FRESH result enters
        # context as a short stub. Not a cache — the tool ran; only plain-string results are stubbed.
        if result_stub and isinstance(function_result, str):
            function_result = result_stub
        if decision.action in {"warn", "halt"}:
            function_result = append_toolguard_guidance(function_result, decision)
        if decision.should_halt:
            self._set_tool_guardrail_halt(decision)
        else:
            # observe_call may have raised the identical-call streak halt (hard_stop_enabled, tool-agnostic).
            streak_halt = self._tool_guardrails.halt_decision
            if streak_halt is not None and streak_halt.code == "identical_call_streak_halt":
                function_result = append_toolguard_guidance(function_result, streak_halt)
                self._set_tool_guardrail_halt(streak_halt)
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
        """Execute the assistant's tool calls and append results to ``messages``.

        The segment planner splits the batch into runs of parallel-safe calls (read-only, non-overlapping file
        targets, opted-in MCP) separated by sequential barriers, run in emission order.
        """
        tool_calls = assistant_message.tool_calls
        args = (assistant_message, messages, effective_task_id, api_call_count)
        self._executing_tools = True  # allow _vprint during tool execution even with stream consumers
        try:
            if len(tool_calls) <= 1:
                return self._execute_tool_calls_sequential(*args)

            from agent.tool_dispatch_helpers import _plan_tool_batch_segments
            active_env = get_active_env(effective_task_id)
            exec_cwd = Path(active_env.cwd) if active_env is not None and active_env.cwd else None
            segments = _plan_tool_batch_segments(tool_calls, execution_cwd=exec_cwd)
            if len(segments) == 1:
                run = self._execute_tool_calls_concurrent if segments[0][0] == "parallel" else self._execute_tool_calls_sequential
                return run(*args)
            from agent.tool_executor import execute_tool_calls_segmented
            return execute_tool_calls_segmented(self, *args, segments=segments)
        finally:
            self._executing_tools = False

    def _dispatch_delegate_task(self, function_args: dict) -> str:
        """Single call site for delegate_task dispatch; new DELEGATE_TASK_SCHEMA fields are added only here."""
        from tools.delegate_tool import _strip_model_hidden_task_fields, delegate_task as _delegate_task
        # Top-level MODEL delegations always run in the background (handle returned, results re-enter as
        # messages). An ORCHESTRATOR SUBAGENT (depth > 0) stays synchronous — it needs results in-turn and
        # owns no gateway session. The schema-level `background` param is intentionally ignored.
        return _delegate_task(
            goal=function_args.get("goal"), context=function_args.get("context"),
            tasks=_strip_model_hidden_task_fields(function_args.get("tasks")),
            max_iterations=function_args.get("max_iterations"), role=function_args.get("role"),
            background=not (getattr(self, "_delegate_depth", 0) > 0), action=function_args.get("action"),
            subagent_id=function_args.get("subagent_id"), message=function_args.get("message"), parent_agent=self,
        )

    _invoke_tool = _forward("agent.agent_runtime_helpers", "invoke_tool")

    @staticmethod
    def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
        """Word-wrap verbose tool output to the terminal width (each existing line separately), continuation
        lines indented."""
        import shutil, textwrap
        wrap_width = max(40, shutil.get_terminal_size((120, 24)).columns - len(indent))
        out_lines: list[str] = []
        for raw_line in text.split("\n"):
            if len(raw_line) <= wrap_width:
                out_lines.append(raw_line)
            else:
                out_lines.extend(textwrap.wrap(raw_line, width=wrap_width, break_long_words=True, break_on_hyphens=False) or [raw_line])
        return f"{indent}{label}" + ("\n" + indent).join(out_lines)

    _execute_tool_calls_concurrent = _forward("agent.tool_executor", "execute_tool_calls_concurrent")
    _execute_tool_calls_sequential = _forward("agent.tool_executor", "execute_tool_calls_sequential")
    _handle_max_iterations = _forward("agent.chat_completion_helpers", "handle_max_iterations")

    def _conversation_root_id(self) -> Optional[str]:
        """Session-lineage ROOT id for Portal usage attribution, so one conversation keeps a single
        ``conversation=`` tag across compression rotation; subagents resolve via ``_parent_session_id``."""
        sid = getattr(self, "session_id", None)
        if not sid:
            return None
        # Subagents may not have a DB row yet on their first turn; walking from the parent id still lands
        # on the right root.
        start = getattr(self, "_parent_session_id", None) or sid
        db = getattr(self, "_session_db", None)
        if db is None:
            return start
        try:
            return db.get_conversation_root(start) or start
        except Exception:
            logger.debug("Conversation root lineage walk failed", exc_info=True)
            return start


_BASIC_TOOLSETS = {"web", "terminal", "vision", "creative", "reasoning"}
_COMPOSITE_TOOLSETS = {"research", "development", "analysis", "content_creation", "full_stack"}
_LIST_TOOLS_USAGE = """
💡 Usage Examples:
  # Use predefined toolsets
  python run_agent.py --enabled_toolsets=research --query='search for Python news'
  python run_agent.py --enabled_toolsets=development --query='debug this code'
  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'

  # Combine multiple toolsets
  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'

  # Disable toolsets
  python run_agent.py --disabled_toolsets=terminal --query='no command execution'

  # Run with trajectory saving enabled
  python run_agent.py --save_trajectories --query='your question here'"""


def _print_tool_listing() -> None:
    """``--list_tools``: print toolsets (basic / composite / scenario / legacy), every tool, and usage examples."""
    from model_tools import get_all_tool_names, get_available_toolsets
    from toolsets import get_all_toolsets, get_toolset_info

    print("📋 Available Tools & Toolsets:")
    print("-" * 50)
    print("\n🎯 Predefined Toolsets (New System):")
    print("-" * 40)
    basic_toolsets, composite_toolsets, scenario_toolsets = [], [], []
    for name in get_all_toolsets():
        info = get_toolset_info(name)
        if info:
            bucket = basic_toolsets if name in _BASIC_TOOLSETS else composite_toolsets if name in _COMPOSITE_TOOLSETS else scenario_toolsets
            bucket.append((name, info))
    print("\n📌 Basic Toolsets:")
    for name, info in basic_toolsets:
        print(f"  • {name:15} - {info['description']}")
        print(f"    Tools: {', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'}")
    print("\n📂 Composite Toolsets (built from other toolsets):")
    for name, info in composite_toolsets:
        print(f"  • {name:15} - {info['description']}")
        print(f"    Includes: {', '.join(info['includes']) if info['includes'] else 'none'}")
        print(f"    Total tools: {info['tool_count']}")
    print("\n🎭 Scenario-Specific Toolsets:")
    for name, info in scenario_toolsets:
        print(f"  • {name:20} - {info['description']}")
        print(f"    Total tools: {info['tool_count']}")
    print("\n📦 Legacy Toolsets (for backward compatibility):")
    for name, info in get_available_toolsets().items():
        print(f"  {'✅' if info['available'] else '❌'} {name}: {info['description']}")
        if not info["available"]:
            print(f"    Requirements: {', '.join(info['requirements'])}")
    all_tools = get_all_tool_names()
    print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
    for tool_name in sorted(all_tools):
        print(f"  📌 {tool_name} (from {get_toolset_for_tool(tool_name)})")
    print(_LIST_TOOLS_USAGE)


def _parse_toolset_arg(raw: Optional[str], label: str) -> Optional[List[str]]:
    """Comma-separated toolset CLI arg → list (echoed), or None when absent."""
    if not raw:
        return None
    names = [t.strip() for t in raw.split(",")]
    print(f"{label}: {names}")
    return names


def _save_sample_trajectory(agent: "AIAgent", result: dict, user_query: str, model: str) -> None:
    """``--save_sample``: write one trajectory (same format as batch_runner) to a UUID-named JSON file."""
    sample_filename = f"sample_{str(uuid.uuid4())[:8]}.json"
    entry = {
        "conversations": agent._convert_to_trajectory_format(result['messages'], user_query, result['completed']),
        "timestamp": datetime.now().isoformat(), "model": model, "completed": result['completed'], "query": user_query,
    }
    try:
        with open(sample_filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2))
        print(f"\n💾 Sample trajectory saved to: {sample_filename}")
    except Exception as e:
        print(f"\n⚠️ Failed to save sample: {e}")


def main(
    query: str = None, model: str = "", api_key: str = None, base_url: str = "", max_turns: int = 10,
    enabled_toolsets: str = None, disabled_toolsets: str = None, list_tools: bool = False,
    save_trajectories: bool = False, save_sample: bool = False, verbose: bool = False, log_prefix_chars: int = 20,
):
    """
    Main function for running the agent directly.

    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use (OpenRouter format: provider/model). Defaults to anthropic/claude-
        sonnet-4.6.
        api_key (str): API key for authentication. Uses OPENROUTER_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://openrouter.ai/api/v1
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_toolsets (str): Comma-separated list of toolsets to enable. Supports predefined
                              toolsets (e.g., "research", "development", "safe").
                              Multiple toolsets can be combined: "web,vision"
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files (appends to
        trajectory_samples.jsonl). Defaults to False.
        save_sample (bool): Save a single trajectory sample to a UUID-named JSONL file for inspection.
        Defaults to False.
        verbose (bool): Enable verbose logging for debugging. Defaults to False.
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses.
        Defaults to 20.

    Toolset Examples:
        - "research": Web search, extract, crawl + vision tools
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    if list_tools:
        return _print_tool_listing()

    enabled_toolsets_list = _parse_toolset_arg(enabled_toolsets, "🎯 Enabled toolsets")
    disabled_toolsets_list = _parse_toolset_arg(disabled_toolsets, "🚫 Disabled toolsets")
    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")

    try:
        agent = AIAgent(
            base_url=base_url, model=model, api_key=api_key, max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list, disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories, verbose_logging=verbose, log_prefix_chars=log_prefix_chars,
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    user_query = query if query is not None else ("Tell me about the latest developments in Python 3.13 and what new features "
                                                  "developers should know about. Please search for current information and try it out.")
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)

    result = agent.run_conversation(user_query)

    print("\n" + "=" * 50 + "\n📋 CONVERSATION SUMMARY\n" + "=" * 50)
    print(f"✅ Completed: {result['completed']}\n📞 API Calls: {result['api_calls']}\n💬 Messages: {len(result['messages'])}")
    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:\n" + "-" * 30 + "\n" + result['final_response'])
    if save_sample:
        _save_sample_trajectory(agent, result, user_query, model)
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    import fire
    fire.Fire(main)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from types import SimpleNamespace  # noqa: F401,E402
import asyncio  # noqa: F401,E402
import base64  # noqa: F401,E402
import copy  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import tempfile  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'COMPRESSED_SUMMARY_METADATA_KEY': ('agent.context_compressor', 'COMPRESSED_SUMMARY_METADATA_KEY'),
    'ContextCompressor': ('agent.context_compressor', 'ContextCompressor'),
    'DEFAULT_AGENT_IDENTITY': ('agent.prompt_builder', 'DEFAULT_AGENT_IDENTITY'),
    'FailoverReason': ('agent.error_classifier', 'FailoverReason'),
    'OpenAI': ('agent.process_bootstrap', 'OpenAI'),
    'atomic_json_write': ('utils', 'atomic_json_write'),
    'build_context_files_prompt': ('agent.prompt_builder', 'build_context_files_prompt'),
    'build_environment_hints': ('agent.prompt_builder', 'build_environment_hints'),
    'build_skills_system_prompt': ('agent.prompt_builder', 'build_skills_system_prompt'),
    'check_toolset_requirements': ('model_tools', 'check_toolset_requirements'),
    'convert_scratchpad_to_think': ('agent.trajectory', 'convert_scratchpad_to_think'),
    'estimate_request_tokens_rough': ('agent.model_metadata', 'estimate_request_tokens_rough'),
    'file_mutation_result_landed': ('agent.tool_result_classification', 'file_mutation_result_landed'),
    'flatten_message_text': ('agent.message_content', 'flatten_message_text'),
    'get_tool_definitions': ('model_tools', 'get_tool_definitions'),
    'handle_function_call': ('model_tools', 'handle_function_call'),
    'is_truthy_value': ('utils', 'is_truthy_value'),
    'jittered_backoff': ('agent.retry_utils', 'jittered_backoff'),
    'load_soul_md': ('agent.prompt_builder', 'load_soul_md'),
    'normalize_usage': ('agent.usage_pricing', 'normalize_usage'),
    'redact_sensitive_text': ('agent.redact', 'redact_sensitive_text'),
    'request_hard_interrupt': ('agent.interrupt_compat', 'request_hard_interrupt'),
    'sanitize_context': ('agent.memory_manager', 'sanitize_context'),
    'user_originated_turn_view': ('agent.context_compressor', 'user_originated_turn_view'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
