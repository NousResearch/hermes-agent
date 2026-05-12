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
from agent.error_classifier import classify_api_error, FailoverReason
from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolGuardrailDecision,
    append_toolguard_guidance,
    toolguard_synthetic_result,
)

# ── Standalone utilities (extracted to agent/utils.py) ─────────────────
from agent.utils import *  # noqa: F401,F403

# _OPENAI_CLS_CACHE lives in agent.utils now (imported via wildcard above)

# ── Imports needed by AIAgent (previously local inside extracted functions) ──
from utils import atomic_json_write, base_url_host_matches, base_url_hostname, env_var_enabled, normalize_proxy_url
from hermes_cli.config import cfg_get
from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout
from model_tools import get_tool_definitions, get_toolset_for_tool, handle_function_call, check_toolset_requirements
from tools.terminal_tool import _get_approval_callback, _get_sudo_password_callback, cleanup_vm, get_active_env, is_persistent_env
from tools.tool_result_storage import maybe_persist_tool_result, enforce_turn_budget
from tools.browser_tool import cleanup_browser
from agent.memory_manager import StreamingContextScrubber, build_memory_context_block, sanitize_context
from agent.think_scrubber import StreamingThinkScrubber
from agent.retry_utils import jittered_backoff
from agent.context_compressor import ContextCompressor
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.prompt_caching import apply_anthropic_cache_control
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS, MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE, HERMES_AGENT_HELP_GUIDANCE, KANBAN_GUIDANCE,
    build_nous_subscription_prompt, build_skills_system_prompt, build_context_files_prompt,
    build_environment_hints, load_soul_md,
    TOOL_USE_ENFORCEMENT_GUIDANCE, TOOL_USE_ENFORCEMENT_MODELS,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE, OPENAI_MODEL_EXECUTION_GUIDANCE,
)
from agent.model_metadata import (
    fetch_model_metadata, estimate_messages_tokens_rough, estimate_request_tokens_rough,
    get_next_probe_tier, parse_context_limit_from_error, parse_available_output_tokens_from_error,
    save_context_length, is_local_endpoint, query_ollama_num_ctx,
)
from agent.codex_responses_adapter import _derive_responses_function_call_id, _deterministic_call_id, _split_responses_tool_id, _summarize_user_message_for_log
from agent.display import KawaiiSpinner, _detect_tool_failure
from agent.trajectory import convert_scratchpad_to_think, has_incomplete_scratchpad
from agent.usage_pricing import estimate_usage_cost, normalize_usage



from agent.streaming import StreamingMixin
from agent.tool_execution import ToolExecutionMixin
from agent.fallback import FallbackMixin
from agent.compression import CompressionMixin
from agent.session import SessionMixin
from agent.loop_support import LoopSupportMixin
from agent.model_switch import ModelSwitchMixin
from agent.message_prep import MessagePrepMixin
from agent.network import NetworkMixin
from agent.steer import SteerMixin


class AIAgent(StreamingMixin, ToolExecutionMixin, FallbackMixin, CompressionMixin, SessionMixin, LoopSupportMixin, ModelSwitchMixin, MessagePrepMixin, NetworkMixin, SteerMixin):
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
        # Structured config objects (take precedence over individual params above)
        provider_config: "agent.config.ProviderConfig" = None,
        session_config: "agent.config.SessionConfig" = None,
        budget_config: "agent.config.BudgetConfig" = None,
        callback_config: "agent.config.CallbackConfig" = None,
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

        # -----------------------------------------------------------------
        # Config object resolution: structured configs override individual
        # params.  This preserves full backward compatibility while allowing
        # new callers to pass a single ProviderConfig / SessionConfig / etc.
        # -----------------------------------------------------------------
        if provider_config is not None:
            from agent.config import ProviderConfig as _PC
            if not isinstance(provider_config, _PC):
                raise TypeError(f"provider_config must be a ProviderConfig, got {type(provider_config).__name__}")
            base_url = provider_config.base_url or base_url
            api_key = provider_config.api_key or api_key
            provider = provider_config.provider or provider
            model = provider_config.model or model
            if provider_config.api_mode:
                api_mode = provider_config.api_mode
            max_iterations = provider_config.max_iterations
            if provider_config.fallback_model is not None:
                fallback_model = provider_config.fallback_model
            if provider_config.credential_pool is not None:
                credential_pool = provider_config.credential_pool
            if provider_config.service_tier is not None:
                service_tier = provider_config.service_tier
            if provider_config.providers_allowed is not None:
                providers_allowed = provider_config.providers_allowed
            if provider_config.providers_ignored is not None:
                providers_ignored = provider_config.providers_ignored
            if provider_config.providers_order is not None:
                providers_order = provider_config.providers_order
            if provider_config.provider_sort is not None:
                provider_sort = provider_config.provider_sort
            if provider_config.openrouter_min_coding_score is not None:
                openrouter_min_coding_score = provider_config.openrouter_min_coding_score
            if provider_config.acp_command is not None:
                acp_command = provider_config.acp_command
            if provider_config.acp_args is not None:
                acp_args = provider_config.acp_args

        if session_config is not None:
            from agent.config import SessionConfig as _SC
            if not isinstance(session_config, _SC):
                raise TypeError(f"session_config must be a SessionConfig, got {type(session_config).__name__}")
            if session_config.session_id is not None:
                session_id = session_config.session_id
            if session_config.platform is not None:
                platform = session_config.platform
            if session_config.user_id is not None:
                user_id = session_config.user_id
            if session_config.user_name is not None:
                user_name = session_config.user_name
            if session_config.chat_id is not None:
                chat_id = session_config.chat_id
            if session_config.chat_name is not None:
                chat_name = session_config.chat_name
            if session_config.chat_type is not None:
                chat_type = session_config.chat_type
            if session_config.thread_id is not None:
                thread_id = session_config.thread_id
            if session_config.gateway_session_key is not None:
                gateway_session_key = session_config.gateway_session_key
            if session_config.parent_session_id is not None:
                parent_session_id = session_config.parent_session_id
            skip_context_files = session_config.skip_context_files
            load_soul_identity = session_config.load_soul_identity
            pass_session_id = session_config.pass_session_id

        if budget_config is not None:
            from agent.config import BudgetConfig as _BC
            if not isinstance(budget_config, _BC):
                raise TypeError(f"budget_config must be a BudgetConfig, got {type(budget_config).__name__}")
            max_iterations = budget_config.max_iterations
            save_trajectories = budget_config.save_trajectories
            if budget_config.max_tokens is not None:
                max_tokens = budget_config.max_tokens
            if budget_config.reasoning_config is not None:
                reasoning_config = budget_config.reasoning_config
            if budget_config.request_overrides is not None:
                request_overrides = budget_config.request_overrides
            if budget_config.prefill_messages is not None:
                prefill_messages = budget_config.prefill_messages
            checkpoints_enabled = budget_config.checkpoints_enabled
            checkpoint_max_snapshots = budget_config.checkpoint_max_snapshots
            checkpoint_max_total_size_mb = budget_config.checkpoint_max_total_size_mb
            checkpoint_max_file_size_mb = budget_config.checkpoint_max_file_size_mb

        if callback_config is not None:
            from agent.config import CallbackConfig as _CC
            if not isinstance(callback_config, _CC):
                raise TypeError(f"callback_config must be a CallbackConfig, got {type(callback_config).__name__}")
            if callback_config.tool_progress_callback is not None:
                tool_progress_callback = callback_config.tool_progress_callback
            if callback_config.tool_start_callback is not None:
                tool_start_callback = callback_config.tool_start_callback
            if callback_config.tool_complete_callback is not None:
                tool_complete_callback = callback_config.tool_complete_callback
            if callback_config.thinking_callback is not None:
                thinking_callback = callback_config.thinking_callback
            if callback_config.reasoning_callback is not None:
                reasoning_callback = callback_config.reasoning_callback
            if callback_config.clarify_callback is not None:
                clarify_callback = callback_config.clarify_callback
            if callback_config.step_callback is not None:
                step_callback = callback_config.step_callback
            if callback_config.stream_delta_callback is not None:
                stream_delta_callback = callback_config.stream_delta_callback
            if callback_config.interim_assistant_callback is not None:
                interim_assistant_callback = callback_config.interim_assistant_callback
            if callback_config.tool_gen_callback is not None:
                tool_gen_callback = callback_config.tool_gen_callback
            if callback_config.status_callback is not None:
                status_callback = callback_config.status_callback

        # -----------------------------------------------------------------
        # Original initialization (unchanged below this line)
        # -----------------------------------------------------------------

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
            if _ttl in ("5m", "1h"):
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
        else:
            if self.quiet_mode:
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
                    if _explicit and _explicit not in ("auto", "openrouter", "custom"):
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
            if _api_retries < 1:
                _api_retries = 1  # 1 = no retry (single attempt)
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
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in ("true", "1", "yes")
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

    def _current_main_runtime(self) -> Dict[str, str]:
        """Return the live main runtime for session-scoped auxiliary routing."""
        return {
            "model": getattr(self, "model", "") or "",
            "provider": getattr(self, "provider", "") or "",
            "base_url": getattr(self, "base_url", "") or "",
            "api_key": getattr(self, "api_key", "") or "",
            "api_mode": getattr(self, "api_mode", "") or "",
        }

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
        return {k: v for k, v in metadata.items() if v not in (None, "")}

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite on any exit path.

        Ensures conversations are never lost, even on errors or early returns.
        """
        self._drop_trailing_empty_response_scaffolding(messages)
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)

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
                if value not in (None, ""):
                    context["reset_at"] = value
                    break
            retry_after = payload.get("retry_after")
            if retry_after not in (None, "") and "reset_at" not in context:
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

    def get_rate_limit_state(self):
        """Return the last captured RateLimitState, or None."""
        return self._rate_limit_state

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
    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        Assemble the full system prompt from all layers.
        
        Called once per session (cached on self._cached_system_prompt) and only
        rebuilt after context compression events. This ensures the system prompt
        is stable across all turns in a session, maximizing prefix cache hits.
        """
        # Layers (in order):
        #   1. Agent identity — SOUL.md when available, else DEFAULT_AGENT_IDENTITY
        #   2. User / gateway system prompt (if provided)
        #   3. Persistent memory (frozen snapshot)
        #   4. Skills guidance (if skills tools are loaded)
        #   5. Context files (AGENTS.md, .cursorrules — SOUL.md excluded here when used as identity)
        #   6. Current date & time (frozen at build time)
        #   7. Platform-specific formatting hint

        # Try SOUL.md as primary identity unless the caller explicitly skipped it.
        # Some execution modes (cron) still want HERMES_HOME persona while keeping
        # cwd project instructions disabled.
        _soul_loaded = False
        if self.load_soul_identity or not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            # Fallback to hardcoded identity
            prompt_parts = [DEFAULT_AGENT_IDENTITY]

        # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
        prompt_parts.append(HERMES_AGENT_HELP_GUIDANCE)

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
            prompt_parts.append(" ".join(tool_guidance))

        # Computer-use (macOS) — goes in as its own block rather than being
        # merged into tool_guidance because the content is multi-paragraph.
        if "computer_use" in self.valid_tool_names:
            from agent.prompt_builder import COMPUTER_USE_GUIDANCE
            prompt_parts.append(COMPUTER_USE_GUIDANCE)

        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            prompt_parts.append(nous_subscription_prompt)
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
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in ("true", "always", "yes", "on")):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in ("false", "never", "no", "off")):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (self.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                # "auto" or any unrecognised value — use hardcoded defaults
                model_lower = (self.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                prompt_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (self.model or "").lower()
                # Google model operational guidance (conciseness, absolute
                # paths, parallel tool calls, verify-before-edit, etc.)
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    prompt_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex execution discipline (tool persistence,
                # prerequisite checks, verification, anti-hallucination).
                if "gpt" in _model_lower or "codex" in _model_lower:
                    prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        # so it can refer the user to them rather than reinventing answers.

        # Note: ephemeral_system_prompt is NOT included here. It's injected at
        # API-call time only so it stays out of the cached/stored system prompt.
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md is always included when enabled.
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # External memory provider system prompt block (additive to built-in)
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    prompt_parts.append(_ext_mem_block)
            except Exception:
                pass

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
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            # Use TERMINAL_CWD for context file discovery when set (gateway
            # mode).  The gateway process runs from the hermes-agent install
            # dir, so os.getcwd() would pick up the repo's AGENTS.md and
            # other dev files — inflating token usage by ~10k for no benefit.
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
        # of the requested model. Inject explicit model identity into the system prompt
        # so the agent can correctly report which model it is (workaround for API bug).
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        # Environment hints (WSL, Termux, etc.) — tell the agent about the
        # execution environment so it can translate paths and adapt behavior.
        _env_hints = build_environment_hints()
        if _env_hints:
            prompt_parts.append(_env_hints)

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])
        elif platform_key:
            # Check plugin registry for platform-specific LLM guidance
            try:
                from gateway.platform_registry import platform_registry
                _entry = platform_registry.get(platform_key)
                if _entry and _entry.platform_hint:
                    prompt_parts.append(_entry.platform_hint)
            except Exception:
                pass

        return "\n\n".join(p.strip() for p in prompt_parts if p.strip())

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

    def _anthropic_messages_create(self, api_kwargs: dict):
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        return self._anthropic_client.messages.create(**api_kwargs)

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
        with tmp:
            tmp.write(base64.b64decode(data))
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
                tools=self.tools,
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
                tools=self.tools,
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
                tools=self.tools,
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
                tools=self.tools,
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
            tools=self.tools,
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

        # ── AgentLoop integration (Phase 2) ─────────────────────────────
        # Build a LoopContext that syncs with the existing IterationBudget
        # and interrupt signal, then drive the loop through AgentLoop.
        # The body is unchanged — only the control flow is delegated.
        from agent.loop import AgentLoop as _AgentLoop, LoopContext as _LoopContext
        from agent.middleware import (
            SteerDrainMiddleware as _SteerDrainMiddleware,
            StepCallbackMiddleware as _StepCallbackMiddleware,
            SkillNudgeMiddleware as _SkillNudgeMiddleware,
            MessageSanitizationMiddleware as _MessageSanitizationMiddleware,
            ApiMessageBuilder as _ApiMessageBuilder,
            ApiMessageFinalizer as _ApiMessageFinalizer,
        )
        _loop_ctx = _LoopContext(max_iterations=self.max_iterations)
        # Sync already-consumed iterations from the budget object.
        _loop_ctx._consumed = self.iteration_budget.used

        # Build middleware instances for loop-body concerns
        _mw_steer = _SteerDrainMiddleware(self)
        _mw_step = _StepCallbackMiddleware(self.step_callback)
        _mw_skill = _SkillNudgeMiddleware(
            nudge_interval=getattr(self, "_skill_nudge_interval", 0),
            has_skill_manage="skill_manage" in self.valid_tool_names,
        )
        _mw_sanitize = _MessageSanitizationMiddleware(self)
        _mw_api_builder = _ApiMessageBuilder(
            self,
            current_turn_user_idx=current_turn_user_idx,
            memory_prefetch=_ext_prefetch_cache,
            plugin_context=_plugin_user_context,
            memory_fence_fn=build_memory_context_block if _ext_prefetch_cache else None,
        )
        _mw_finalizer = _ApiMessageFinalizer(
            self,
            active_system_prompt=active_system_prompt,
            prompt_cache_fn=apply_anthropic_cache_control if getattr(self, "_use_prompt_caching", False) else None,
            surrogate_sanitize_fn=_sanitize_messages_surrogates,
        )
        _agent_loop = _AgentLoop(_loop_ctx, middlewares=[
            _mw_sanitize, _mw_steer, _mw_step, _mw_skill, _mw_api_builder, _mw_finalizer,
        ])

        while _agent_loop.context.should_continue() or self._budget_grace_call:
            # Reset per-turn checkpoint dedup so each iteration can take one snapshot
            self._checkpoint_mgr.new_turn()

            # Check for interrupt request (e.g., user sent new message)
            if self._interrupt_requested:
                _agent_loop.context.request_interrupt()
                interrupted = True
                _turn_exit_reason = "interrupted_by_user"
                if not self.quiet_mode:
                    self._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1
            self._api_call_count = api_call_count
            _agent_loop.iteration = api_call_count
            self._touch_activity(f"starting API call #{api_call_count}")

            # Grace call: the budget is exhausted but we gave the model one
            # more chance.  Consume the grace flag so the loop exits after
            # this iteration regardless of outcome.
            if self._budget_grace_call:
                self._budget_grace_call = False
                _agent_loop.context.enable_grace_call()
            elif not self.iteration_budget.consume():
                _loop_ctx._consumed = self.iteration_budget.used
                _turn_exit_reason = "budget_exhausted"
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
                break
            _loop_ctx._consumed = self.iteration_budget.used

            # Fire step_callback for gateway hooks (agent:step event)
            # Delegated to StepCallbackMiddleware
            _mw_step.before_iteration(_loop_ctx, api_call_count, messages=messages)

            # Track tool-calling iterations for skill nudge.
            # Delegated to SkillNudgeMiddleware
            _mw_skill.before_iteration(_loop_ctx, api_call_count)
            
            # ── Pre-API-call /steer drain ──────────────────────────────────
            # Delegated to SteerDrainMiddleware
            _mw_steer.before_iteration(_loop_ctx, api_call_count, messages=messages)

            # Message sanitization (tool-call args + role-alternation repair)
            # Delegated to MessageSanitizationMiddleware
            _mw_sanitize.before_iteration(_loop_ctx, api_call_count, messages=messages)

            # Build API messages (strip internals, inject context, copy reasoning)
            # Delegated to ApiMessageBuilder
            _mw_api_builder.before_iteration(_loop_ctx, api_call_count, messages=messages)
            api_messages = _mw_api_builder.last_api_messages

            # Finalize API messages (system prompt, prefill, caching, sanitize,
            # drop thinking-only, normalize, strip surrogates).
            # Delegated to ApiMessageFinalizer
            _mw_finalizer.before_iteration(_loop_ctx, api_call_count, messages=api_messages)
            api_messages = _mw_finalizer.last_finalized_messages

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
                        elif _resp_error_code in (500, 502):
                            _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                        elif _resp_error_code in (503, 529):
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

                        if self.api_mode in ("chat_completions", "bedrock_converse", "anthropic_messages"):
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

                        if self.api_mode in ("chat_completions", "bedrock_converse", "anthropic_messages"):
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
                    is_rate_limited = classified.reason in (
                        FailoverReason.rate_limit,
                        FailoverReason.billing,
                    )
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
                            and classified.reason not in (
                                FailoverReason.rate_limit,
                                FailoverReason.billing,
                                FailoverReason.overloaded,
                                FailoverReason.context_overflow,
                                FailoverReason.payload_too_large,
                                FailoverReason.long_context_tier,
                                FailoverReason.thinking_signature,
                            )
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
                                    _retry_after = min(int(_ra_raw), 120)  # Cap at 2 minutes
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
            "tool_turns=%d last_msg_role=%s response_len=%d session=%s "
            "agent_loop(iter=%d,interrupted=%s)"
        )
        _diag_args = (
            _turn_exit_reason, self.model, api_call_count, self.max_iterations,
            _budget_used, _budget_max,
            _turn_tool_count, _last_msg_role, _resp_len,
            self.session_id or "none",
            _agent_loop.iteration, _agent_loop.interrupted,
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
                and _mw_skill.iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            _should_review_skills = True
            _mw_skill.reset()

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
                if name in ["web", "terminal", "vision", "creative", "reasoning"]:
                    basic_toolsets.append(entry)
                elif name in ["research", "development", "analysis", "content_creation", "full_stack"]:
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