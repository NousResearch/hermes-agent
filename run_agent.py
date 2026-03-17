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

import atexit
import asyncio
import base64
import concurrent.futures
import copy
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import random
import re
import sys
import tempfile
import time
import threading
import weakref
from types import SimpleNamespace
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime
from pathlib import Path

# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_project_env = Path(__file__).parent / '.env'
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")

# Point mini-swe-agent at ~/.hermes/ so it shares our config
os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(_hermes_home))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

# Import our tool system
from model_tools import get_tool_definitions, handle_function_call, check_toolset_requirements
from tools.terminal_tool import cleanup_vm
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser

import requests

from hermes_constants import OPENROUTER_BASE_URL, OPENROUTER_MODELS_URL

# Agent internals extracted to agent/ package for modularity
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS,
    MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE,
)
from agent.model_metadata import (
    fetch_model_metadata, get_model_context_length,
    estimate_tokens_rough, estimate_messages_tokens_rough,
    get_next_probe_tier, parse_context_limit_from_error,
    save_context_length,
)
from agent.context_compressor import ContextCompressor
from agent.prompt_caching import apply_anthropic_cache_control
from agent.prompt_builder import build_skills_system_prompt, build_context_files_prompt
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from agent.display import (
    KawaiiSpinner, build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
    _detect_tool_failure,
    get_tool_emoji as _get_tool_emoji,
)
from agent.trajectory import (
    convert_scratchpad_to_think, has_incomplete_scratchpad,
    save_trajectory as _save_trajectory_to_file,
)
from utils import atomic_json_write

HONCHO_TOOL_NAMES = {
    "honcho_context",
    "honcho_profile",
    "honcho_search",
    "honcho_conclude",
}


class _SafeWriter:
    """Transparent stdio wrapper that catches OSError from broken pipes.

    When hermes-agent runs as a systemd service, Docker container, or headless
    daemon, the stdout/stderr pipe can become unavailable (idle timeout, buffer
    exhaustion, socket reset). Any print() call then raises
    ``OSError: [Errno 5] Input/output error``, which can crash agent setup or
    run_conversation() — especially via double-fault when an except handler
    also tries to print.

    This wrapper delegates all writes to the underlying stream and silently
    catches OSError. It is transparent when the wrapped stream is healthy.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except OSError:
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except OSError:
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except OSError:
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


class IterationBudget:
    """Thread-safe shared iteration counter for parent and child agents.

    Tracks total LLM-call iterations consumed across a parent agent and all
    its subagents.  A single ``IterationBudget`` is created by the parent
    and passed to every child so they share the same cap.

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
        return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


# Tools that must never run concurrently (interactive / user-facing).
# When any of these appear in a batch, we fall back to sequential execution.
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# Maximum number of concurrent worker threads for parallel tool execution.
_MAX_TOOL_WORKERS = 8

# Patterns that indicate a terminal command may modify/delete files.
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
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


def _inject_honcho_turn_context(content, turn_context: str):
    """Append Honcho recall to the current-turn user message without mutating history.

    The returned content is sent to the API for this turn only. Keeping Honcho
    recall out of the system prompt preserves the stable cache prefix while
    still giving the model continuity context.
    """
    if not turn_context:
        return content

    note = (
        "[System note: The following Honcho memory was retrieved from prior "
        "sessions. It is continuity context for this turn only, not new user "
        "input.]\n\n"
        f"{turn_context}"
    )

    if isinstance(content, list):
        return list(content) + [{"type": "text", "text": note}]

    text = "" if content is None else str(content)
    if not text.strip():
        return note
    return f"{text}\n\n{note}"


class AIAgent:
    """
    AI Agent with tool calling capabilities.
    
    This class manages the conversation flow, tool execution, and response handling
    for AI models that support function calling.
    """
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        model: str = "anthropic/claude-opus-4.6",  # OpenRouter format
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
        session_id: str = None,
        tool_progress_callback: callable = None,
        thinking_callback: callable = None,
        reasoning_callback: callable = None,
        clarify_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db=None,
        honcho_session_key: str = None,
        honcho_manager=None,
        honcho_config=None,
        iteration_budget: "IterationBudget" = None,
        fallback_model: Dict[str, Any] = None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 50,
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
            platform (str): The interface platform the user is on (e.g. "cli", "telegram", "discord", "whatsapp").
                Used to inject platform-specific formatting hints into the system prompt.
            skip_context_files (bool): If True, skip auto-injection of SOUL.md, AGENTS.md, and .cursorrules
                into the system prompt. Use this for batch processing and data generation to avoid
                polluting trajectories with user-specific persona or project instructions.
            honcho_session_key (str): Session key for Honcho integration (e.g., "telegram:123456" or CLI session_id).
                When provided and Honcho is enabled in config, enables persistent cross-session user modeling.
            honcho_manager: Optional shared HonchoSessionManager owned by the caller.
            honcho_config: Optional HonchoClientConfig corresponding to honcho_manager.
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
        self.skip_context_files = skip_context_files
        self.pass_session_id = pass_session_id
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        # Store effective base URL for feature detection (prompt caching, reasoning, etc.)
        # When no base_url is provided, the client defaults to OpenRouter, so reflect that here.
        self.base_url = base_url or OPENROUTER_BASE_URL
        provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        self.provider = provider_name or "openrouter"
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif (provider_name is None) and "chatgpt.com/backend-api/codex" in self.base_url.lower():
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif self.provider == "anthropic" or (provider_name is None and "api.anthropic.com" in self.base_url.lower()):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        else:
            self.api_mode = "chat_completions"

        # Pre-warm OpenRouter model metadata cache in a background thread.
        # fetch_model_metadata() is cached for 1 hour; this avoids a blocking
        # HTTP request on the first API response when pricing is estimated.
        if self.provider == "openrouter" or "openrouter" in self.base_url.lower():
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self._last_reported_tool = None  # Track for "new tool" mode
        
        # Interrupt mechanism for breaking out of tool loops
        self._interrupt_requested = False
        self._interrupt_message = None  # Optional message that triggered interrupt
        self._client_lock = threading.RLock()
        
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

        # Store toolset filtering options
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # Model response configuration
        self.max_tokens = max_tokens  # None = use model default
        self.reasoning_config = reasoning_config  # None = use default (medium for OpenRouter)
        self.prefill_messages = prefill_messages or []  # Prefilled conversation turns
        
        # Anthropic prompt caching: auto-enabled for Claude models via OpenRouter.
        # Reduces input costs by ~75% on multi-turn conversations by caching the
        # conversation prefix. Uses system_and_3 strategy (4 breakpoints).
        is_openrouter = "openrouter" in self.base_url.lower()
        is_claude = "claude" in self.model.lower()
        is_native_anthropic = self.api_mode == "anthropic_messages"
        self._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic
        self._cache_ttl = "5m"  # Default 5-minute TTL (1.25x write cost)
        
        # Iteration budget pressure: warn the LLM as it approaches max_iterations.
        # Warnings are injected into the last tool result JSON (not as separate
        # messages) so they don't break message structure or invalidate caching.
        self._budget_caution_threshold = 0.7   # 70% — nudge to start wrapping up
        self._budget_warning_threshold = 0.9   # 90% — urgent, respond now
        self._budget_pressure_enabled = True

        # Persistent error log -- always writes WARNING+ to ~/.hermes/logs/errors.log
        # so tool failures, API errors, etc. are inspectable after the fact.
        # In gateway mode, each incoming message creates a new AIAgent instance,
        # while the root logger is process-global. Re-adding the same errors.log
        # handler would cause each warning/error line to be written multiple times.
        from logging.handlers import RotatingFileHandler
        root_logger = logging.getLogger()
        error_log_dir = _hermes_home / "logs"
        error_log_path = error_log_dir / "errors.log"
        resolved_error_log_path = error_log_path.resolve()
        has_errors_log_handler = any(
            isinstance(handler, RotatingFileHandler)
            and Path(getattr(handler, "baseFilename", "")).resolve() == resolved_error_log_path
            for handler in root_logger.handlers
        )
        from agent.redact import RedactingFormatter
        if not has_errors_log_handler:
            error_log_dir.mkdir(parents=True, exist_ok=True)
            error_file_handler = RotatingFileHandler(
                error_log_path, maxBytes=2 * 1024 * 1024, backupCount=2,
            )
            error_file_handler.setLevel(logging.WARNING)
            error_file_handler.setFormatter(RedactingFormatter(
                '%(asctime)s %(levelname)s %(name)s: %(message)s',
            ))
            root_logger.addHandler(error_file_handler)

        if self.verbose_logging:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            for handler in logging.getLogger().handlers:
                handler.setFormatter(RedactingFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                ))
            # Keep third-party libraries at WARNING level to reduce noise
            # We have our own retry and error logging that's more informative
            logging.getLogger('openai').setLevel(logging.WARNING)
            logging.getLogger('openai._base_client').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('httpcore').setLevel(logging.WARNING)
            logging.getLogger('asyncio').setLevel(logging.WARNING)
            # Suppress Modal/gRPC related debug spam
            logging.getLogger('hpack').setLevel(logging.WARNING)
            logging.getLogger('hpack.hpack').setLevel(logging.WARNING)
            logging.getLogger('grpc').setLevel(logging.WARNING)
            logging.getLogger('modal').setLevel(logging.WARNING)
            logging.getLogger('rex-deploy').setLevel(logging.INFO)  # Keep INFO for sandbox status
            logger.info("Verbose logging enabled (third-party library logs suppressed)")
        else:
            # Set logging to INFO level for important messages only
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            # Suppress noisy library logging
            logging.getLogger('openai').setLevel(logging.ERROR)
            logging.getLogger('openai._base_client').setLevel(logging.ERROR)
            logging.getLogger('httpx').setLevel(logging.ERROR)
            logging.getLogger('httpcore').setLevel(logging.ERROR)
            if self.quiet_mode:
                # In quiet mode (CLI default), suppress all tool/infra log
                # noise. The TUI has its own rich display for status; logger
                # INFO/WARNING messages just clutter it.
                for quiet_logger in [
                    'tools',               # all tools.* (terminal, browser, web, file, etc.)
                    'minisweagent',         # mini-swe-agent execution backend
                    'run_agent',            # agent runner internals
                    'trajectory_compressor',
                    'cron',                 # scheduler (only relevant in daemon mode)
                    'hermes_cli',           # CLI helpers
                ]:
                    logging.getLogger(quiet_logger).setLevel(logging.ERROR)
        
        # Internal stream callback (set during streaming TTS).
        # Initialized here so _vprint can reference it before run_conversation.
        self._stream_callback = None

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

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            effective_key = api_key or resolve_anthropic_token() or ""
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url
            from agent.anthropic_adapter import _is_oauth_token as _is_oat
            self._is_anthropic_oauth = _is_oat(effective_key)
            self._anthropic_client = build_anthropic_client(effective_key, base_url)
            # No OpenAI client needed for Anthropic mode
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)")
                if effective_key and len(effective_key) > 12:
                    print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        else:
            if api_key and base_url:
                # Explicit credentials from CLI/gateway — construct directly.
                # The runtime provider resolver already handled auth for us.
                client_kwargs = {"api_key": api_key, "base_url": base_url}
                effective_base = base_url
                if "openrouter" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-OpenRouter-Title": "Hermes Agent",
                        "X-OpenRouter-Categories": "productivity,cli-agent",
                    }
                elif "api.kimi.com" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "User-Agent": "KimiCLI/1.3",
                    }
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
                    # Preserve any default_headers the router set
                    if hasattr(_routed_client, '_default_headers') and _routed_client._default_headers:
                        client_kwargs["default_headers"] = dict(_routed_client._default_headers)
                else:
                    # Final fallback: try raw OpenRouter key
                    client_kwargs = {
                        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
                        "base_url": OPENROUTER_BASE_URL,
                        "default_headers": {
                            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                            "X-OpenRouter-Title": "Hermes Agent",
                            "X-OpenRouter-Categories": "productivity,cli-agent",
                        },
                    }
            
            self._client_kwargs = client_kwargs  # stored for rebuilding after interrupt
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
        
        # Provider fallback — a single backup model/provider tried when the
        # primary is exhausted (rate-limit, overload, connection failure).
        # Config shape: {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
        self._fallback_model = fallback_model if isinstance(fallback_model, dict) else None
        self._fallback_activated = False
        if self._fallback_model:
            fb_p = self._fallback_model.get("provider", "")
            fb_m = self._fallback_model.get("model", "")
            if fb_p and fb_m and not self.quiet_mode:
                print(f"🔄 Fallback model: {fb_m} ({fb_p})")

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
            source = "native Anthropic" if is_native_anthropic else "Claude via OpenRouter"
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
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
        
        # Track conversation messages for session logging
        self._session_messages: List[Dict[str, Any]] = []
        
        # Cached system prompt -- built once per session, only rebuilt on compression
        self._cached_system_prompt: Optional[str] = None
        
        # Filesystem checkpoint manager (transparent — not a tool)
        from tools.checkpoint_manager import CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
        )
        
        # SQLite session store (optional -- provided by CLI or gateway)
        self._session_db = session_db
        self._last_flushed_db_idx = 0  # tracks DB-write cursor to prevent duplicate writes
        if self._session_db:
            try:
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or "cli",
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_iterations,
                        "reasoning_config": reasoning_config,
                        "max_tokens": max_tokens,
                    },
                    user_id=None,
                )
            except Exception as e:
                logger.debug("Session DB create_session failed: %s", e)
        
        # In-memory todo list for task planning (one per agent/session)
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()
        
        # Persistent memory (MEMORY.md + USER.md) -- loaded from disk
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 10
        self._memory_flush_min_turns = 6
        if not skip_memory:
            try:
                from hermes_cli.config import load_config as _load_mem_config
                mem_config = _load_mem_config().get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                self._memory_flush_min_turns = int(mem_config.get("flush_min_turns", 6))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass  # Memory is optional -- don't break agent init
        
        # Honcho AI-native memory (cross-session user modeling)
        # Reads ~/.honcho/config.json as the single source of truth.
        self._honcho = None  # HonchoSessionManager | None
        self._honcho_session_key = honcho_session_key
        self._honcho_config = None  # HonchoClientConfig | None
        self._honcho_exit_hook_registered = False
        if not skip_memory:
            try:
                if honcho_manager is not None:
                    hcfg = honcho_config or getattr(honcho_manager, "_config", None)
                    self._honcho_config = hcfg
                    if hcfg and self._honcho_should_activate(hcfg):
                        self._honcho = honcho_manager
                        self._activate_honcho(
                            hcfg,
                            enabled_toolsets=enabled_toolsets,
                            disabled_toolsets=disabled_toolsets,
                            session_db=session_db,
                        )
                else:
                    from honcho_integration.client import HonchoClientConfig, get_honcho_client
                    hcfg = HonchoClientConfig.from_global_config()
                    self._honcho_config = hcfg
                    if self._honcho_should_activate(hcfg):
                        from honcho_integration.session import HonchoSessionManager
                        client = get_honcho_client(hcfg)
                        self._honcho = HonchoSessionManager(
                            honcho=client,
                            config=hcfg,
                            context_tokens=hcfg.context_tokens,
                        )
                        self._activate_honcho(
                            hcfg,
                            enabled_toolsets=enabled_toolsets,
                            disabled_toolsets=disabled_toolsets,
                            session_db=session_db,
                        )
                    else:
                        if not hcfg.enabled:
                            logger.debug("Honcho disabled in global config")
                        elif not hcfg.api_key:
                            logger.debug("Honcho enabled but no API key configured")
                        else:
                            logger.debug("Honcho enabled but missing API key or disabled in config")
            except Exception as e:
                logger.warning("Honcho init failed — memory disabled: %s", e)
                print(f"  Honcho init failed: {e}")
                print("  Run 'hermes honcho setup' to reconfigure.")
                self._honcho = None

        # Tools are initially discovered before Honcho activation. If Honcho
        # stays inactive, remove any stale honcho_* tools from prior process state.
        if not self._honcho:
            self._strip_honcho_tools_from_surface()

        # Gate local memory writes based on per-peer memory modes.
        # AI peer governs MEMORY.md; user peer governs USER.md.
        # "honcho" = Honcho only, disable local writes.
        if self._honcho_config and self._honcho:
            _hcfg = self._honcho_config
            _agent_mode = _hcfg.peer_memory_mode(_hcfg.ai_peer)
            _user_mode = _hcfg.peer_memory_mode(_hcfg.peer_name or "user")
            if _agent_mode == "honcho":
                self._memory_flush_min_turns = 0
                self._memory_enabled = False
                logger.debug("peer %s memory_mode=honcho: local MEMORY.md writes disabled", _hcfg.ai_peer)
            if _user_mode == "honcho":
                self._user_profile_enabled = False
                logger.debug("peer %s memory_mode=honcho: local USER.md writes disabled", _hcfg.peer_name or "user")

        # Skills config: nudge interval for skill creation reminders
        self._skill_nudge_interval = 10
        try:
            from hermes_cli.config import load_config as _load_skills_config
            skills_config = _load_skills_config().get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 15))
        except Exception:
            pass
        
        # Initialize context compressor for automatic context management
        # Compresses conversation when approaching model's context limit
        # Configuration via config.yaml (compression section) or environment variables
        compression_threshold = float(os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "0.50"))
        compression_enabled = os.getenv("CONTEXT_COMPRESSION_ENABLED", "true").lower() in ("true", "1", "yes")
        compression_summary_model = os.getenv("CONTEXT_COMPRESSION_MODEL") or None
        
        self.context_compressor = ContextCompressor(
            model=self.model,
            threshold_percent=compression_threshold,
            protect_first_n=3,
            protect_last_n=4,
            summary_target_tokens=500,
            summary_model_override=compression_summary_model,
            quiet_mode=self.quiet_mode,
            base_url=self.base_url,
        )
        self.compression_enabled = compression_enabled
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
        
        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")
    
    @staticmethod
    def _safe_print(*args, **kwargs):
        """Print that silently handles broken pipes / closed stdout.

        In headless environments (systemd, Docker, nohup) stdout may become
        unavailable mid-session.  A raw ``print()`` raises ``OSError`` which
        can crash cron jobs and lose completed work.
        """
        try:
            print(*args, **kwargs)
        except OSError:
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """Verbose print — suppressed when streaming TTS is active.

        Pass ``force=True`` for error/warning messages that should always be
        shown even during streaming playback (TTS or display).
        """
        if not force and self._has_stream_consumers():
            return
        self._safe_print(*args, **kwargs)

    def _max_tokens_param(self, value: int) -> dict:
        """Return the correct max tokens kwarg for the current provider.
        
        OpenAI's newer models (gpt-4o, o-series, gpt-5+) require
        'max_completion_tokens'. OpenRouter, local models, and older
        OpenAI models use 'max_tokens'.
        """
        _is_direct_openai = (
            "api.openai.com" in self.base_url.lower()
            and "openrouter" not in self.base_url.lower()
        )
        if _is_direct_openai:
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    def _has_content_after_think_block(self, content: str) -> bool:
        """
        Check if content has actual text after any <think></think> blocks.
        
        This detects cases where the model only outputs reasoning but no actual
        response, which indicates an incomplete generation that should be retried.
        
        Args:
            content: The assistant message content to check
            
        Returns:
            True if there's meaningful content after think blocks, False otherwise
        """
        if not content:
            return False
        
        # Remove all <think>...</think> blocks (including nested ones, non-greedy)
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())
    
    def _strip_think_blocks(self, content: str) -> str:
        """Remove <think>...</think> blocks from content, returning only visible text."""
        if not content:
            return ""
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

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
                    summary = detail.get('summary') or detail.get('content') or detail.get('text')
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)
        
        # Combine all reasoning parts
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None
    
    def _cleanup_task_resources(self, task_id: str) -> None:
        """Clean up VM and browser resources for a given task."""
        try:
            cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

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
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)

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
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
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
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.debug("Session DB append_message failed: %s", e)

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
                        tool_response = f"<tool_response>\n"
                        
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
    
    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

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

            if os.getenv("HERMES_DUMP_REQUEST_STDOUT", "").strip().lower() in {"1", "true", "yes", "on"}:
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
        # Signal all tools to abort any in-flight operations immediately
        _set_interrupt(True)
        # Propagate interrupt to any running child agents (subagent delegation)
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print(f"\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
    
    def clear_interrupt(self) -> None:
        """Clear any pending interrupt request and the global tool interrupt signal."""
        self._interrupt_requested = False
        self._interrupt_message = None
        _set_interrupt(False)
    
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

    # ── Honcho integration helpers ──

    def _honcho_should_activate(self, hcfg) -> bool:
        """Return True when remote Honcho should be active."""
        if not hcfg or not hcfg.enabled or not hcfg.api_key:
            return False
        return True

    def _strip_honcho_tools_from_surface(self) -> None:
        """Remove Honcho tools from the active tool surface."""
        if not self.tools:
            self.valid_tool_names = set()
            return

        self.tools = [
            tool for tool in self.tools
            if tool.get("function", {}).get("name") not in HONCHO_TOOL_NAMES
        ]
        self.valid_tool_names = {
            tool["function"]["name"] for tool in self.tools
        } if self.tools else set()

    def _activate_honcho(
        self,
        hcfg,
        *,
        enabled_toolsets: Optional[List[str]],
        disabled_toolsets: Optional[List[str]],
        session_db,
    ) -> None:
        """Finish Honcho setup once a session manager is available."""
        if not self._honcho:
            return

        if not self._honcho_session_key:
            session_title = None
            if session_db is not None:
                try:
                    session_title = session_db.get_session_title(self.session_id or "")
                except Exception:
                    pass
            self._honcho_session_key = (
                hcfg.resolve_session_name(
                    session_title=session_title,
                    session_id=self.session_id,
                )
                or "hermes-default"
            )

        honcho_sess = self._honcho.get_or_create(self._honcho_session_key)
        if not honcho_sess.messages:
            try:
                from hermes_cli.config import get_hermes_home

                mem_dir = str(get_hermes_home() / "memories")
                self._honcho.migrate_memory_files(
                    self._honcho_session_key,
                    mem_dir,
                )
            except Exception as exc:
                logger.debug("Memory files migration failed (non-fatal): %s", exc)

        from tools.honcho_tools import set_session_context

        set_session_context(self._honcho, self._honcho_session_key)

        # Rebuild tool surface after Honcho context injection. Tool availability
        # is check_fn-gated and may change once session context is attached.
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=True,
        )
        self.valid_tool_names = {
            tool["function"]["name"] for tool in self.tools
        } if self.tools else set()

        if hcfg.recall_mode == "context":
            self._strip_honcho_tools_from_surface()
            if not self.quiet_mode:
                print("  Honcho active — recall_mode: context (Honcho tools hidden)")
        else:
            if not self.quiet_mode:
                print(f"  Honcho active — recall_mode: {hcfg.recall_mode}")

        logger.info(
            "Honcho active (session: %s, user: %s, workspace: %s, "
            "write_frequency: %s, memory_mode: %s)",
            self._honcho_session_key,
            hcfg.peer_name,
            hcfg.workspace_id,
            hcfg.write_frequency,
            hcfg.memory_mode,
        )

        recall_mode = hcfg.recall_mode
        if recall_mode != "tools":
            try:
                ctx = self._honcho.get_prefetch_context(self._honcho_session_key)
                if ctx:
                    self._honcho.set_context_result(self._honcho_session_key, ctx)
                    logger.debug("Honcho context pre-warmed for first turn")
            except Exception as exc:
                logger.debug("Honcho context prefetch failed (non-fatal): %s", exc)

        self._register_honcho_exit_hook()

    def _register_honcho_exit_hook(self) -> None:
        """Register a process-exit flush hook without clobbering signal handlers."""
        if self._honcho_exit_hook_registered or not self._honcho:
            return

        honcho_ref = weakref.ref(self._honcho)

        def _flush_honcho_on_exit():
            manager = honcho_ref()
            if manager is None:
                return
            try:
                manager.flush_all()
            except Exception as exc:
                logger.debug("Honcho flush on exit failed (non-fatal): %s", exc)

        atexit.register(_flush_honcho_on_exit)
        self._honcho_exit_hook_registered = True

    def _queue_honcho_prefetch(self, user_message: str) -> None:
        """Queue turn-end Honcho prefetch so the next turn can consume cached results."""
        if not self._honcho or not self._honcho_session_key:
            return

        recall_mode = (self._honcho_config.recall_mode if self._honcho_config else "hybrid")
        if recall_mode == "tools":
            return

        try:
            self._honcho.prefetch_context(self._honcho_session_key, user_message)
            self._honcho.prefetch_dialectic(self._honcho_session_key, user_message or "What were we working on?")
        except Exception as exc:
            logger.debug("Honcho background prefetch failed (non-fatal): %s", exc)

    def _honcho_prefetch(self, user_message: str) -> str:
        """Assemble the first-turn Honcho context from the pre-warmed cache."""
        if not self._honcho or not self._honcho_session_key:
            return ""
        try:
            parts = []

            ctx = self._honcho.pop_context_result(self._honcho_session_key)
            if ctx:
                rep = ctx.get("representation", "")
                card = ctx.get("card", "")
                if rep:
                    parts.append(f"## User representation\n{rep}")
                if card:
                    parts.append(card)
                ai_rep = ctx.get("ai_representation", "")
                ai_card = ctx.get("ai_card", "")
                if ai_rep:
                    parts.append(f"## AI peer representation\n{ai_rep}")
                if ai_card:
                    parts.append(ai_card)

            dialectic = self._honcho.pop_dialectic_result(self._honcho_session_key)
            if dialectic:
                parts.append(f"## Continuity synthesis\n{dialectic}")

            if not parts:
                return ""
            header = (
                "# Honcho Memory (persistent cross-session context)\n"
                "Use this to answer questions about the user, prior sessions, "
                "and what you were working on together. Do not call tools to "
                "look up information that is already present here.\n"
            )
            return header + "\n\n".join(parts)
        except Exception as e:
            logger.debug("Honcho prefetch failed (non-fatal): %s", e)
            return ""

    def _honcho_save_user_observation(self, content: str) -> str:
        """Route a memory tool target=user add to Honcho.

        Sends the content as a user peer message so Honcho's reasoning
        model can incorporate it into the user representation.
        """
        if not content or not content.strip():
            return json.dumps({"success": False, "error": "Content cannot be empty."})
        try:
            session = self._honcho.get_or_create(self._honcho_session_key)
            session.add_message("user", f"[observation] {content.strip()}")
            self._honcho.save(session)
            return json.dumps({
                "success": True,
                "target": "user",
                "message": "Saved to Honcho user model.",
            })
        except Exception as e:
            logger.debug("Honcho user observation failed: %s", e)
            return json.dumps({"success": False, "error": f"Honcho save failed: {e}"})

    def _honcho_sync(self, user_content: str, assistant_content: str) -> None:
        """Sync the user/assistant message pair to Honcho."""
        if not self._honcho or not self._honcho_session_key:
            return
        try:
            session = self._honcho.get_or_create(self._honcho_session_key)
            session.add_message("user", user_content)
            session.add_message("assistant", assistant_content)
            self._honcho.save(session)
            logger.info("Honcho sync queued for session %s (%d messages)",
                        self._honcho_session_key, len(session.messages))
        except Exception as e:
            logger.warning("Honcho sync failed: %s", e)
            if not self.quiet_mode:
                print(f"  Honcho write failed: {e}")

    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        Assemble the full system prompt from all layers.
        
        Called once per session (cached on self._cached_system_prompt) and only
        rebuilt after context compression events. This ensures the system prompt
        is stable across all turns in a session, maximizing prefix cache hits.
        """
        # Layers (in order):
        #   1. Default agent identity (always present)
        #   2. User / gateway system prompt (if provided)
        #   3. Persistent memory (frozen snapshot)
        #   4. Skills guidance (if skills tools are loaded)
        #   5. Context files (SOUL.md, AGENTS.md, .cursorrules)
        #   6. Current date & time (frozen at build time)
        #   7. Platform-specific formatting hint
        # If an AI peer name is configured in Honcho, personalise the identity line.
        _ai_peer_name = (
            self._honcho_config.ai_peer
            if self._honcho_config and self._honcho_config.ai_peer != "hermes"
            else None
        )
        if _ai_peer_name:
            _identity = DEFAULT_AGENT_IDENTITY.replace(
                "You are Hermes Agent",
                f"You are {_ai_peer_name}",
                1,
            )
        else:
            _identity = DEFAULT_AGENT_IDENTITY
        prompt_parts = [_identity]

        # Tool-aware behavioral guidance: only inject when the tools are loaded
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        # Honcho CLI awareness: tell Hermes about its own management commands
        # so it can refer the user to them rather than reinventing answers.
        if self._honcho and self._honcho_session_key:
            hcfg = self._honcho_config
            mode = hcfg.memory_mode if hcfg else "hybrid"
            freq = hcfg.write_frequency if hcfg else "async"
            recall_mode = hcfg.recall_mode if hcfg else "hybrid"
            honcho_block = (
                "# Honcho memory integration\n"
                f"Active. Session: {self._honcho_session_key}. "
                f"Mode: {mode}. Write frequency: {freq}. Recall: {recall_mode}.\n"
            )
            if recall_mode == "context":
                honcho_block += (
                    "Honcho context is injected into this system prompt below. "
                    "All memory retrieval comes from this context — no Honcho tools "
                    "are available. Answer questions about the user, prior sessions, "
                    "and recent work directly from the Honcho Memory section.\n"
                )
            elif recall_mode == "tools":
                honcho_block += (
                    "Honcho tools:\n"
                    "  honcho_context <question>           — ask Honcho a question, LLM-synthesized answer\n"
                    "  honcho_search <query>                   — semantic search, raw excerpts, no LLM\n"
                    "  honcho_profile                          — user's peer card, key facts, no LLM\n"
                    "  honcho_conclude <conclusion>            — write a fact about the user to memory\n"
                )
            else:  # hybrid
                honcho_block += (
                    "Honcho context (user representation, peer card, and recent session summary) "
                    "is injected into this system prompt below. Use it to answer continuity "
                    "questions ('where were we?', 'what were we working on?') WITHOUT calling "
                    "any tools. Only call Honcho tools when you need information beyond what is "
                    "already present in the Honcho Memory section.\n"
                    "Honcho tools:\n"
                    "  honcho_context <question>           — ask Honcho a question, LLM-synthesized answer\n"
                    "  honcho_search <query>                   — semantic search, raw excerpts, no LLM\n"
                    "  honcho_profile                          — user's peer card, key facts, no LLM\n"
                    "  honcho_conclude <conclusion>            — write a fact about the user to memory\n"
                )
            honcho_block += (
                "Management commands (refer users here instead of explaining manually):\n"
                "  hermes honcho status                    — show full config + connection\n"
                "  hermes honcho mode [hybrid|honcho]       — show or set memory mode\n"
                "  hermes honcho tokens [--context N] [--dialectic N] — show or set token budgets\n"
                "  hermes honcho peer [--user NAME] [--ai NAME] [--reasoning LEVEL]\n"
                "  hermes honcho sessions                  — list directory→session mappings\n"
                "  hermes honcho map <name>                — map cwd to a session name\n"
                "  hermes honcho identity [<file>] [--show] — seed or show AI peer identity\n"
                "  hermes honcho migrate                   — migration guide from openclaw-honcho\n"
                "  hermes honcho setup                     — full interactive wizard"
            )
            prompt_parts.append(honcho_block)

        # Note: ephemeral_system_prompt is NOT included here. It's injected at
        # API-call time only so it stays out of the cached/stored system prompt.
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md is always included when enabled -- Honcho prefetch is additive.
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {ts for ts, avail in check_toolset_requirements().items() if avail}
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            context_files_prompt = build_context_files_prompt()
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        prompt_parts.append(timestamp_line)

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(prompt_parts)
    
    def _repair_tool_call(self, tool_name: str) -> str | None:
        """Attempt to repair a mismatched tool name before aborting.

        1. Try lowercase
        2. Try normalized (lowercase + hyphens/spaces -> underscores)
        3. Try fuzzy match (difflib, cutoff=0.7)

        Returns the repaired name if found in valid_tool_names, else None.
        """
        from difflib import get_close_matches

        # 1. Lowercase
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered

        # 2. Normalize
        normalized = lowered.replace("-", "_").replace(" ", "_")
        if normalized in self.valid_tool_names:
            return normalized

        # 3. Fuzzy match
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

    def _responses_tools(self, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """Convert chat-completions tool schemas to Responses function-tool schemas."""
        source_tools = tools if tools is not None else self.tools
        if not source_tools:
            return None

        converted: List[Dict[str, Any]] = []
        for item in source_tools:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append({
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted or None

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """Split a stored tool id into (call_id, response_item_id)."""
        if not isinstance(raw_id, str):
            return None, None
        value = raw_id.strip()
        if not value:
            return None, None
        if "|" in value:
            call_id, response_item_id = value.split("|", 1)
            call_id = call_id.strip() or None
            response_item_id = response_item_id.strip() or None
            return call_id, response_item_id
        if value.startswith("fc_"):
            return None, value
        return value, None

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """Build a valid Responses `function_call.id` (must start with `fc_`)."""
        if isinstance(response_item_id, str):
            candidate = response_item_id.strip()
            if candidate.startswith("fc_"):
                return candidate

        source = (call_id or "").strip()
        if source.startswith("fc_"):
            return source
        if source.startswith("call_") and len(source) > len("call_"):
            return f"fc_{source[len('call_'):]}"

        sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
        if sanitized.startswith("fc_"):
            return sanitized
        if sanitized.startswith("call_") and len(sanitized) > len("call_"):
            return f"fc_{sanitized[len('call_'):]}"
        if sanitized:
            return f"fc_{sanitized[:48]}"

        seed = source or str(response_item_id or "") or uuid.uuid4().hex
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
        return f"fc_{digest}"

    def _chat_messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal chat-style messages to Responses input items."""
        items: List[Dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                continue

            if role in {"user", "assistant"}:
                content = msg.get("content", "")
                content_text = str(content) if content is not None else ""

                if role == "assistant":
                    # Replay encrypted reasoning items from previous turns
                    # so the API can maintain coherent reasoning chains.
                    codex_reasoning = msg.get("codex_reasoning_items")
                    if isinstance(codex_reasoning, list):
                        for ri in codex_reasoning:
                            if isinstance(ri, dict) and ri.get("encrypted_content"):
                                items.append(ri)

                    if content_text.strip():
                        items.append({"role": "assistant", "content": content_text})

                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function", {})
                            fn_name = fn.get("name")
                            if not isinstance(fn_name, str) or not fn_name.strip():
                                continue

                            embedded_call_id, embedded_response_item_id = self._split_responses_tool_id(
                                tc.get("id")
                            )
                            call_id = tc.get("call_id")
                            if not isinstance(call_id, str) or not call_id.strip():
                                call_id = embedded_call_id
                            if not isinstance(call_id, str) or not call_id.strip():
                                if (
                                    isinstance(embedded_response_item_id, str)
                                    and embedded_response_item_id.startswith("fc_")
                                    and len(embedded_response_item_id) > len("fc_")
                                ):
                                    call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                                else:
                                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                            call_id = call_id.strip()

                            arguments = fn.get("arguments", "{}")
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            elif not isinstance(arguments, str):
                                arguments = str(arguments)
                            arguments = arguments.strip() or "{}"

                            items.append({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": fn_name,
                                "arguments": arguments,
                            })
                    continue

                items.append({"role": role, "content": content_text})
                continue

            if role == "tool":
                raw_tool_call_id = msg.get("tool_call_id")
                call_id, _ = self._split_responses_tool_id(raw_tool_call_id)
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                        call_id = raw_tool_call_id.strip()
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(msg.get("content", "") or ""),
                })

        return items

    def _preflight_codex_input_items(self, raw_items: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            raise ValueError("Codex Responses input must be a list of input items.")

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Codex Responses input[{idx}] must be an object.")

            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                arguments = arguments.strip() or "{}"

                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": call_id.strip(),
                        "name": name.strip(),
                        "arguments": arguments,
                    }
                )
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
                output = item.get("output", "")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    output = str(output)

                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": output,
                    }
                )
                continue

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    reasoning_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id:
                        reasoning_item["id"] = item_id
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        reasoning_item["summary"] = summary
                    else:
                        reasoning_item["summary"] = []
                    normalized.append(reasoning_item)
                continue

            role = item.get("role")
            if role in {"user", "assistant"}:
                content = item.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)

                normalized.append({"role": role, "content": content})
                continue

            raise ValueError(
                f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
            )

        return normalized

    def _preflight_codex_api_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(api_kwargs, dict):
            raise ValueError("Codex Responses request must be a dict.")

        required = {"model", "instructions", "input"}
        missing = [key for key in required if key not in api_kwargs]
        if missing:
            raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")

        model = api_kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Codex Responses request 'model' must be a non-empty string.")
        model = model.strip()

        instructions = api_kwargs.get("instructions")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            instructions = str(instructions)
        instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY

        normalized_input = self._preflight_codex_input_items(api_kwargs.get("input"))

        tools = api_kwargs.get("tools")
        normalized_tools = None
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("Codex Responses request 'tools' must be a list when provided.")
            normalized_tools = []
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] must be an object.")
                if tool.get("type") != "function":
                    raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")

                name = tool.get("name")
                parameters = tool.get("parameters")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses tools[{idx}] is missing a valid name.")
                if not isinstance(parameters, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] is missing valid parameters.")

                description = tool.get("description", "")
                if description is None:
                    description = ""
                if not isinstance(description, str):
                    description = str(description)

                strict = tool.get("strict", False)
                if not isinstance(strict, bool):
                    strict = bool(strict)

                normalized_tools.append(
                    {
                        "type": "function",
                        "name": name.strip(),
                        "description": description,
                        "strict": strict,
                        "parameters": parameters,
                    }
                )

        store = api_kwargs.get("store", False)
        if store is not False:
            raise ValueError("Codex Responses contract requires 'store' to be false.")

        allowed_keys = {
            "model", "instructions", "input", "tools", "store",
            "reasoning", "include", "max_output_tokens", "temperature",
            "tool_choice", "parallel_tool_calls", "prompt_cache_key",
        }
        normalized: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": normalized_input,
            "tools": normalized_tools,
            "store": False,
        }

        # Pass through reasoning config
        reasoning = api_kwargs.get("reasoning")
        if isinstance(reasoning, dict):
            normalized["reasoning"] = reasoning
        include = api_kwargs.get("include")
        if isinstance(include, list):
            normalized["include"] = include

        # Pass through max_output_tokens and temperature
        max_output_tokens = api_kwargs.get("max_output_tokens")
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            normalized["max_output_tokens"] = int(max_output_tokens)
        temperature = api_kwargs.get("temperature")
        if isinstance(temperature, (int, float)):
            normalized["temperature"] = float(temperature)

        # Pass through tool_choice, parallel_tool_calls, prompt_cache_key
        for passthrough_key in ("tool_choice", "parallel_tool_calls", "prompt_cache_key"):
            val = api_kwargs.get(passthrough_key)
            if val is not None:
                normalized[passthrough_key] = val

        if allow_stream:
            stream = api_kwargs.get("stream")
            if stream is not None and stream is not True:
                raise ValueError("Codex Responses 'stream' must be true when set.")
            if stream is True:
                normalized["stream"] = True
            allowed_keys.add("stream")
        elif "stream" in api_kwargs:
            raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")

        unexpected = sorted(key for key in api_kwargs.keys() if key not in allowed_keys)
        if unexpected:
            raise ValueError(
                f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
            )

        return normalized

    def _extract_responses_message_text(self, item: Any) -> str:
        """Extract assistant text from a Responses message output item."""
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            return ""

        chunks: List[str] = []
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype not in {"output_text", "text"}:
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks).strip()

    def _extract_responses_reasoning_text(self, item: Any) -> str:
        """Extract a compact reasoning text from a Responses reasoning item."""
        summary = getattr(item, "summary", None)
        if isinstance(summary, list):
            chunks: List[str] = []
            for part in summary:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return text.strip()
        return ""

    def _normalize_codex_response(self, response: Any) -> tuple[Any, str]:
        """Normalize a Responses API object to an assistant_message-like object."""
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            raise RuntimeError("Responses API returned no output items")

        response_status = getattr(response, "status", None)
        if isinstance(response_status, str):
            response_status = response_status.strip().lower()
        else:
            response_status = None

        if response_status in {"failed", "cancelled"}:
            error_obj = getattr(response, "error", None)
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = str(error_obj) if error_obj else f"Responses API returned status '{response_status}'"
            raise RuntimeError(error_msg)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items_raw: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
        saw_commentary_phase = False
        saw_final_answer_phase = False

        for item in output:
            item_type = getattr(item, "type", None)
            item_status = getattr(item, "status", None)
            if isinstance(item_status, str):
                item_status = item_status.strip().lower()
            else:
                item_status = None

            if item_status in {"queued", "in_progress", "incomplete"}:
                has_incomplete_items = True

            if item_type == "message":
                item_phase = getattr(item, "phase", None)
                if isinstance(item_phase, str):
                    normalized_phase = item_phase.strip().lower()
                    if normalized_phase in {"commentary", "analysis"}:
                        saw_commentary_phase = True
                    elif normalized_phase in {"final_answer", "final"}:
                        saw_final_answer_phase = True
                message_text = self._extract_responses_message_text(item)
                if message_text:
                    content_parts.append(message_text)
            elif item_type == "reasoning":
                reasoning_text = self._extract_responses_reasoning_text(item)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                # Capture the full reasoning item for multi-turn continuity.
                # encrypted_content is an opaque blob the API needs back on
                # subsequent turns to maintain coherent reasoning chains.
                encrypted = getattr(item, "encrypted_content", None)
                if isinstance(encrypted, str) and encrypted:
                    raw_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = getattr(item, "id", None)
                    if isinstance(item_id, str) and item_id:
                        raw_item["id"] = item_id
                    # Capture summary — required by the API when replaying reasoning items
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        raw_summary = []
                        for part in summary:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                raw_summary.append({"type": "summary_text", "text": text})
                        raw_item["summary"] = raw_summary
                    reasoning_items_raw.append(raw_item)
            elif item_type == "function_call":
                if item_status in {"queued", "in_progress", "incomplete"}:
                    continue
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))
            elif item_type == "custom_tool_call":
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "input", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))

        final_text = "\n".join([p for p in content_parts if p]).strip()
        if not final_text and hasattr(response, "output_text"):
            out_text = getattr(response, "output_text", "")
            if isinstance(out_text, str):
                final_text = out_text.strip()

        assistant_message = SimpleNamespace(
            content=final_text,
            tool_calls=tool_calls,
            reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=reasoning_items_raw or None,
        )

        if tool_calls:
            finish_reason = "tool_calls"
        elif has_incomplete_items or (saw_commentary_phase and not saw_final_answer_phase):
            finish_reason = "incomplete"
        else:
            finish_reason = "stop"
        return assistant_message, finish_reason

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
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False
        http_client = getattr(client, "_client", None)
        return bool(getattr(http_client, "is_closed", False))

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s) %s",
                reason,
                shared,
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

    def _create_request_openai_client(self, *, reason: str) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        self._close_openai_client(client, reason=reason, shared=False)

    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """Execute one streaming Responses API request and return the final response."""
        active_client = client or self._ensure_primary_openai_client(reason="codex_stream_direct")
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        for attempt in range(max_stream_retries + 1):
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # Fire callbacks on text content deltas (suppress during tool calls)
                        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", "")
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
                    return stream.get_final_response()
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
        fallback_kwargs = self._preflight_codex_api_kwargs(fallback_kwargs, allow_stream=True)
        stream_or_response = active_client.responses.create(**fallback_kwargs)

        # Compatibility shim for mocks or providers that still return a concrete response.
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        try:
            for event in stream_or_response:
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")
                if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                    continue

                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
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

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
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
            self._anthropic_client = build_anthropic_client(new_token, getattr(self, "_anthropic_base_url", None))
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        return True

    def _anthropic_messages_create(self, api_kwargs: dict):
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        return self._anthropic_client.messages.create(**api_kwargs)

    def _interruptible_api_call(self, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.

        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = self._create_request_openai_client(reason="codex_stream_request")
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                else:
                    request_client_holder["client"] = self._create_request_openai_client(reason="chat_completion_request")
                    result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="request_complete")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if self._interrupt_requested:
                # Force-close the in-flight worker-local HTTP connection to stop
                # token generation without poisoning the shared client used to
                # seed future retries.
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
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

    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        for cb in (self.stream_delta_callback, self._stream_callback):
            if cb is not None:
                try:
                    cb(text)
                except Exception:
                    pass

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered."""
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
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

        result = {"response": None, "error": None}
        request_client_holder = {"client": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """Stream a chat completions response."""
            stream_kwargs = {**api_kwargs, "stream": True, "stream_options": {"include_usage": True}}
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request"
            )
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            content_parts: list = []
            tool_calls_acc: dict = {}
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None

            for chunk in stream:
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
                    self._fire_reasoning_delta(reasoning_text)

                # Accumulate text content — fire callback only when no tool calls
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True

                # Accumulate tool call deltas (silently, no callback)
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index if tc_delta.index is not None else 0
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage in the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # Build mock response matching non-streaming shape
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    ))

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
                finish_reason=finish_reason or "stop",
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

            # Use the Anthropic SDK's streaming context manager
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    self._fire_reasoning_delta(thinking_text)

                # Return the native Anthropic Message for downstream processing
                return stream.get_final_message()

        def _call():
            try:
                if self.api_mode == "anthropic_messages":
                    self._try_refresh_anthropic_client_credentials()
                    result["response"] = _call_anthropic()
                else:
                    result["response"] = _call_chat_completions()
            except Exception as e:
                if deltas_were_sent["yes"]:
                    # Streaming failed AFTER some tokens were already delivered
                    # to consumers. Don't fall back — that would cause
                    # double-delivery (partial streamed + full non-streamed).
                    # Let the error propagate; the partial content already
                    # reached the user via the stream.
                    logger.warning("Streaming failed after partial delivery, not falling back: %s", e)
                    result["error"] = e
                else:
                    # Streaming failed before any tokens reached consumers.
                    # Safe to fall back to the standard non-streaming path.
                    logger.info("Streaming failed before delivery, falling back to non-streaming: %s", e)
                    try:
                        result["response"] = self._interruptible_api_call(api_kwargs)
                    except Exception as fallback_err:
                        result["error"] = fallback_err
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if self._interrupt_requested:
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during streaming API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Provider fallback ──────────────────────────────────────────────────

    def _try_activate_fallback(self) -> bool:
        """Switch to the configured fallback model/provider.

        Called when the primary model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  One-shot: returns False if
        already activated or not configured.

        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.
        """
        if self._fallback_activated or not self._fallback_model:
            return False

        fb = self._fallback_model
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return False

        # Use centralized router for client construction.
        # raw_codex=True because the main agent needs direct responses.stream()
        # access for Codex providers.
        try:
            from agent.auxiliary_client import resolve_provider_client
            fb_client, _ = resolve_provider_client(
                fb_provider, model=fb_model, raw_codex=True)
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured",
                    fb_provider)
                return False

            # Determine api_mode from provider
            fb_api_mode = "chat_completions"
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic":
                fb_api_mode = "anthropic_messages"
            fb_base_url = str(fb_client.base_url)

            old_model = self.model
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            self._fallback_activated = True

            if fb_api_mode == "anthropic_messages":
                # Build native Anthropic client instead of using OpenAI client
                from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
                effective_key = fb_client.api_key or resolve_anthropic_token() or ""
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = getattr(fb_client, "base_url", None)
                self._anthropic_client = build_anthropic_client(effective_key, self._anthropic_base_url)
                self.client = None
                self._client_kwargs = {}
            else:
                # Swap OpenAI client and config in-place
                self.client = fb_client
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                }

            # Re-evaluate prompt caching for the new provider/model
            is_native_anthropic = fb_api_mode == "anthropic_messages"
            self._use_prompt_caching = (
                ("openrouter" in fb_base_url.lower() and "claude" in fb_model.lower())
                or is_native_anthropic
            )

            print(
                f"{self.log_prefix}🔄 Primary model failed — switching to fallback: "
                f"{fb_model} via {fb_provider}"
            )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model, fb_model, fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback model: %s", e)
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

    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the active API mode."""
        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_kwargs
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            return build_anthropic_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=getattr(self, "_is_anthropic_oauth", False),
            )

        if self.api_mode == "codex_responses":
            instructions = ""
            payload_messages = api_messages
            if api_messages and api_messages[0].get("role") == "system":
                instructions = str(api_messages[0].get("content") or "").strip()
                payload_messages = api_messages[1:]
            if not instructions:
                instructions = DEFAULT_AGENT_IDENTITY

            # Resolve reasoning effort: config > default (medium)
            reasoning_effort = "medium"
            reasoning_enabled = True
            if self.reasoning_config and isinstance(self.reasoning_config, dict):
                if self.reasoning_config.get("enabled") is False:
                    reasoning_enabled = False
                elif self.reasoning_config.get("effort"):
                    reasoning_effort = self.reasoning_config["effort"]

            kwargs = {
                "model": self.model,
                "instructions": instructions,
                "input": self._chat_messages_to_responses_input(payload_messages),
                "tools": self._responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "store": False,
                "prompt_cache_key": self.session_id,
            }

            if reasoning_enabled:
                kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                kwargs["include"] = ["reasoning.encrypted_content"]
            else:
                kwargs["include"] = []

            if self.max_tokens is not None:
                kwargs["max_output_tokens"] = self.max_tokens

            return kwargs

        sanitized_messages = api_messages
        needs_sanitization = False
        for msg in api_messages:
            if not isinstance(msg, dict):
                continue
            if "codex_reasoning_items" in msg:
                needs_sanitization = True
                break

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    if "call_id" in tool_call or "response_item_id" in tool_call:
                        needs_sanitization = True
                        break
                if needs_sanitization:
                    break

        if needs_sanitization:
            sanitized_messages = copy.deepcopy(api_messages)
            for msg in sanitized_messages:
                if not isinstance(msg, dict):
                    continue

                # Codex-only replay state must not leak into strict chat-completions APIs.
                msg.pop("codex_reasoning_items", None)

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call.pop("call_id", None)
                            tool_call.pop("response_item_id", None)

        provider_preferences = {}
        if self.providers_allowed:
            provider_preferences["only"] = self.providers_allowed
        if self.providers_ignored:
            provider_preferences["ignore"] = self.providers_ignored
        if self.providers_order:
            provider_preferences["order"] = self.providers_order
        if self.provider_sort:
            provider_preferences["sort"] = self.provider_sort
        if self.provider_require_parameters:
            provider_preferences["require_parameters"] = True
        if self.provider_data_collection:
            provider_preferences["data_collection"] = self.provider_data_collection

        api_kwargs = {
            "model": self.model,
            "messages": sanitized_messages,
            "tools": self.tools if self.tools else None,
            "timeout": float(os.getenv("HERMES_API_TIMEOUT", 900.0)),
        }

        if self.max_tokens is not None:
            api_kwargs.update(self._max_tokens_param(self.max_tokens))

        extra_body = {}

        _is_openrouter = "openrouter" in self.base_url.lower()

        # Provider preferences (only, ignore, order, sort) are OpenRouter-
        # specific.  Only send to OpenRouter-compatible endpoints.
        # TODO: Nous Portal will add transparent proxy support — re-enable
        # for _is_nous when their backend is updated.
        if provider_preferences and _is_openrouter:
            extra_body["provider"] = provider_preferences
        _is_nous = "nousresearch" in self.base_url.lower()

        if self._supports_reasoning_extra_body():
            if self.reasoning_config is not None:
                rc = dict(self.reasoning_config)
                # Nous Portal requires reasoning enabled — don't send
                # enabled=false to it (would cause 400).
                if _is_nous and rc.get("enabled") is False:
                    pass  # omit reasoning entirely for Nous when disabled
                else:
                    extra_body["reasoning"] = rc
            else:
                extra_body["reasoning"] = {
                    "enabled": True,
                    "effort": "medium"
                }

        # Nous Portal product attribution
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        return api_kwargs

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body fields to upstream providers.
        Some providers/routes reject `reasoning` with 400s, so gate it to
        known reasoning-capable model families and direct Nous Portal.
        """
        base_url = (self.base_url or "").lower()
        if "nousresearch" in base_url:
            return True
        if "ai-gateway.vercel.sh" in base_url:
            return True
        if "openrouter" not in base_url:
            return False
        if "api.mistral.ai" in base_url:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        reasoning_text = self._extract_reasoning(assistant_message)

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
            try:
                self.reasoning_callback(reasoning_text)
            except Exception:
                pass

        msg = {
            "role": "assistant",
            "content": assistant_message.content or "",
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

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

        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        call_id = f"call_{uuid.uuid4().hex[:12]}"
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

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral strictly validate the Chat Completions schema
        and reject unknown fields (call_id, response_item_id) with 422.
        These fields are preserved in the internal message history — this
        method only modifies the outgoing API copy.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).
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

    def flush_memories(self, messages: list = None, min_turns: int = None):
        """Give the model one turn to persist memories before context is lost.

        Called before compression, session reset, or CLI exit. Injects a flush
        message, makes one API call, executes any memory tool calls, then
        strips all flush artifacts from the message list.

        Args:
            messages: The current conversation messages. If None, uses
                      self._session_messages (last run_conversation state).
            min_turns: Minimum user turns required to trigger the flush.
                       None = use config value (flush_min_turns).
                       0 = always flush (used for compression).
        """
        if self._memory_flush_min_turns == 0 and min_turns is None:
            return
        if "memory" not in self.valid_tool_names or not self._memory_store:
            return
        # honcho-only agent mode: skip local MEMORY.md flush
        _hcfg = getattr(self, '_honcho_config', None)
        if _hcfg and _hcfg.peer_memory_mode(_hcfg.ai_peer) == "honcho":
            return
        effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, '_session_messages', None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Save anything worth remembering — prioritize user preferences, "
            "corrections, and recurring patterns over task-specific details.]"
        )
        _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
        flush_msg = {"role": "user", "content": flush_content, "_flush_sentinel": _sentinel}
        messages.append(flush_msg)

        try:
            # Build API messages for the flush call
            _is_strict_api = "api.mistral.ai" in self.base_url.lower()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                if msg.get("role") == "assistant":
                    reasoning = msg.get("reasoning")
                    if reasoning:
                        api_msg["reasoning_content"] = reasoning
                api_msg.pop("reasoning", None)
                api_msg.pop("finish_reason", None)
                api_msg.pop("_flush_sentinel", None)
                if _is_strict_api:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            if self._cached_system_prompt:
                api_messages = [{"role": "system", "content": self._cached_system_prompt}] + api_messages

            # Make one API call with only the memory tool available
            memory_tool_def = None
            for t in (self.tools or []):
                if t.get("function", {}).get("name") == "memory":
                    memory_tool_def = t
                    break

            if not memory_tool_def:
                messages.pop()  # remove flush msg
                return

            # Use auxiliary client for the flush call when available --
            # it's cheaper and avoids Codex Responses API incompatibility.
            from agent.auxiliary_client import call_llm as _call_llm
            _aux_available = True
            try:
                response = _call_llm(
                    task="flush_memories",
                    messages=api_messages,
                    tools=[memory_tool_def],
                    temperature=0.3,
                    max_tokens=5120,
                    timeout=30.0,
                )
            except RuntimeError:
                _aux_available = False
                response = None

            if not _aux_available and self.api_mode == "codex_responses":
                # No auxiliary client -- use the Codex Responses path directly
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs["tools"] = self._responses_tools([memory_tool_def])
                codex_kwargs["temperature"] = 0.3
                if "max_output_tokens" in codex_kwargs:
                    codex_kwargs["max_output_tokens"] = 5120
                response = self._run_codex_stream(codex_kwargs)
            elif not _aux_available and self.api_mode == "anthropic_messages":
                # Native Anthropic — use the Anthropic client directly
                from agent.anthropic_adapter import build_anthropic_kwargs as _build_ant_kwargs
                ant_kwargs = _build_ant_kwargs(
                    model=self.model, messages=api_messages,
                    tools=[memory_tool_def], max_tokens=5120,
                    reasoning_config=None,
                )
                response = self._anthropic_messages_create(ant_kwargs)
            elif not _aux_available:
                api_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": [memory_tool_def],
                    "temperature": 0.3,
                    **self._max_tokens_param(5120),
                }
                response = self._ensure_primary_openai_client(reason="flush_memories").chat.completions.create(**api_kwargs, timeout=30.0)

            # Extract tool calls from the response, handling all API formats
            tool_calls = []
            if self.api_mode == "codex_responses" and not _aux_available:
                assistant_msg, _ = self._normalize_codex_response(response)
                if assistant_msg and assistant_msg.tool_calls:
                    tool_calls = assistant_msg.tool_calls
            elif self.api_mode == "anthropic_messages" and not _aux_available:
                from agent.anthropic_adapter import normalize_anthropic_response as _nar_flush
                _flush_msg, _ = _nar_flush(response, strip_tool_prefix=getattr(self, '_is_anthropic_oauth', False))
                if _flush_msg and _flush_msg.tool_calls:
                    tool_calls = _flush_msg.tool_calls
            elif hasattr(response, "choices") and response.choices:
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    tool_calls = assistant_message.tool_calls

            for tc in tool_calls:
                if tc.function.name == "memory":
                    try:
                        args = json.loads(tc.function.arguments)
                        flush_target = args.get("target", "memory")
                        from tools.memory_tool import memory_tool as _memory_tool
                        result = _memory_tool(
                            action=args.get("action"),
                            target=flush_target,
                            content=args.get("content"),
                            old_text=args.get("old_text"),
                            store=self._memory_store,
                        )
                        if self._honcho and flush_target == "user" and args.get("action") == "add":
                            self._honcho_save_user_observation(args.get("content", ""))
                        if not self.quiet_mode:
                            print(f"  🧠 Memory flush: saved to {args.get('target', 'memory')}")
                    except Exception as e:
                        logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # Strip flush artifacts: remove everything from the flush message onward.
            # Use sentinel marker instead of identity check for robustness.
            while messages and messages[-1].get("_flush_sentinel") != _sentinel:
                messages.pop()
                if not messages:
                    break
            if messages and messages[-1].get("_flush_sentinel") == _sentinel:
                messages.pop()

    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default") -> tuple:
        """Compress conversation context and split the session in SQLite.

        Returns:
            (compressed_messages, new_system_prompt) tuple
        """
        # Pre-compression memory flush: let the model save memories before they're lost
        self.flush_memories(messages, min_turns=0)

        compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens)

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        # Preserve file-read history so the model doesn't re-read files
        # it already examined before compression.
        try:
            from tools.file_tools import get_read_files_summary
            read_files = get_read_files_summary(task_id)
            if read_files:
                file_list = "\n".join(
                    f"  - {f['path']} ({', '.join(f['regions'])})"
                    for f in read_files
                )
                compressed.append({"role": "user", "content": (
                    "[Files already read in this session — do NOT re-read these]\n"
                    f"{file_list}\n"
                    "Use the information from the context summary above. "
                    "Proceed with writing, editing, or responding."
                )})
        except Exception:
            pass  # Don't break compression if file tracking fails

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # Propagate title to the new session with auto-numbering
                old_title = self._session_db.get_session_title(self.session_id)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or "cli",
                    model=self.model,
                    parent_session_id=old_session_id,
                )
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
                logger.debug("Session DB compression split failed: %s", e)

        return compressed, new_system_prompt

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        Dispatches to concurrent execution when multiple independent tool calls
        are present, falling back to sequential execution for single calls or
        when interactive tools (e.g. clarify) are in the batch.
        """
        tool_calls = assistant_message.tool_calls

        # Single tool call or interactive tool present → sequential
        if (len(tool_calls) <= 1
                or any(tc.function.name in _NEVER_PARALLEL_TOOLS for tc in tool_calls)):
            return self._execute_tool_calls_sequential(
                assistant_message, messages, effective_task_id, api_call_count
            )

        # Multiple non-interactive tools → concurrent
        return self._execute_tool_calls_concurrent(
            assistant_message, messages, effective_task_id, api_call_count
        )

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str) -> str:
        """Invoke a single tool and return the result string. No display logic.

        Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
        tools. Used by the concurrent execution path; the sequential path retains
        its own inline invocation for backward-compatible display handling.
        """
        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            if not self._session_db:
                return json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=self._session_db,
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
            # Also send user observations to Honcho when active
            if self._honcho and target == "user" and function_args.get("action") == "add":
                self._honcho_save_user_observation(function_args.get("content", ""))
            return result
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            from tools.delegate_tool import delegate_task as _delegate_task
            return _delegate_task(
                goal=function_args.get("goal"),
                context=function_args.get("context"),
                toolsets=function_args.get("toolsets"),
                tasks=function_args.get("tasks"),
                max_iterations=function_args.get("max_iterations"),
                parent_agent=self,
            )
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                honcho_manager=self._honcho,
                honcho_session_key=self._honcho_session_key,
            )

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
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
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

            parsed_calls.append((tool_call, function_name, function_args))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for _, name, args in parsed_calls:
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback(name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # Each slot holds (function_name, function_args, function_result, duration, error_flag)
        results = [None] * num_tools

        def _run_tool(index, tool_call, function_name, function_args):
            """Worker function executed in a thread."""
            start = time.time()
            try:
                result = self._invoke_tool(function_name, function_args, effective_task_id)
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            results[index] = (function_name, function_args, result, duration, is_error)

        # Start spinner for CLI mode
        spinner = None
        if self.quiet_mode:
            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots')
            spinner.start()

        try:
            max_workers = min(num_tools, _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, (tc, name, args) in enumerate(parsed_calls):
                    f = executor.submit(_run_tool, i, tc, name, args)
                    futures.append(f)

                # Wait for all to complete (exceptions are captured inside _run_tool)
                concurrent.futures.wait(futures)
        finally:
            if spinner:
                # Build a summary message for the spinner stop
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args) in enumerate(parsed_calls):
            r = results[i]
            if r is None:
                # Shouldn't happen, but safety fallback
                function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error = r

                if is_error:
                    result_preview = function_result[:200] if len(function_result) > 200 else function_result
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # Print cute message per tool
            if self.quiet_mode:
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                print(f"  {cute_msg}")
            elif not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            # Truncate oversized results
            MAX_TOOL_RESULT_CHARS = 100_000
            if len(function_result) > MAX_TOOL_RESULT_CHARS:
                original_len = len(function_result)
                function_result = (
                    function_result[:MAX_TOOL_RESULT_CHARS]
                    + f"\n\n[Truncated: tool response was {original_len:,} chars, "
                    f"exceeding the {MAX_TOOL_RESULT_CHARS:,} char limit]"
                )

            # Append tool result message in order
            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

        # ── Budget pressure injection ────────────────────────────────────
        budget_warning = self._get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            if not self.quiet_mode:
                remaining = self.max_iterations - api_call_count
                tier = "⚠️  WARNING" if remaining <= self.max_iterations * 0.1 else "💡 CAUTION"
                print(f"{self.log_prefix}{tier}: {remaining} iterations remaining")

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
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            # Reset nudge counters when the relevant tool is actually used
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback(function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            # Checkpoint: snapshot working dir before file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
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
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
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

            if function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                if not self._session_db:
                    function_result = json.dumps({"success": False, "error": "Session database not available."})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=self._session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
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
                # Also send user observations to Honcho when active
                if self._honcho and target == "user" and function_args.get("action") == "add":
                    self._honcho_save_user_observation(function_args.get("content", ""))
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                from tools.delegate_tool import delegate_task as _delegate_task
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self.quiet_mode:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots')
                    spinner.start()
                self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = _delegate_task(
                        goal=function_args.get("goal"),
                        context=function_args.get("context"),
                        toolsets=function_args.get("toolsets"),
                        tasks=tasks_arg,
                        max_iterations=function_args.get("max_iterations"),
                        parent_agent=self,
                    )
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self.quiet_mode:
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode and not self._has_stream_consumers():
                face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                emoji = _get_tool_emoji(function_name)
                preview = _build_tool_preview(function_name, function_args) or function_name
                if len(preview) > 30:
                    preview = preview[:27] + "..."
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots')
                spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        honcho_manager=self._honcho,
                        honcho_session_key=self._honcho_session_key,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    spinner.stop(cute_msg)
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        honcho_manager=self._honcho,
                        honcho_session_key=self._honcho_session_key,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            result_preview = function_result if self.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )

            # Log tool errors to the persistent error log so [error] tags
            # in the UI always have a corresponding detailed entry on disk.
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # Guard against tools returning absurdly large content that would
            # blow up the context window. 100K chars ≈ 25K tokens — generous
            # enough for any reasonable tool output but prevents catastrophic
            # context explosions (e.g. accidental base64 image dumps).
            MAX_TOOL_RESULT_CHARS = 100_000
            if len(function_result) > MAX_TOOL_RESULT_CHARS:
                original_len = len(function_result)
                function_result = (
                    function_result[:MAX_TOOL_RESULT_CHARS]
                    + f"\n\n[Truncated: tool response was {original_len:,} chars, "
                    f"exceeding the {MAX_TOOL_RESULT_CHARS:,} char limit]"
                )

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── Budget pressure injection ─────────────────────────────────
        # After all tool calls in this turn are processed, check if we're
        # approaching max_iterations. If so, inject a warning into the LAST
        # tool result's JSON so the LLM sees it naturally when reading results.
        budget_warning = self._get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            if not self.quiet_mode:
                remaining = self.max_iterations - api_call_count
                tier = "⚠️  WARNING" if remaining <= self.max_iterations * 0.1 else "💡 CAUTION"
                print(f"{self.log_prefix}{tier}: {remaining} iterations remaining")

    def _get_budget_warning(self, api_call_count: int) -> Optional[str]:
        """Return a budget pressure string, or None if not yet needed.

        Two-tier system:
          - Caution (70%): nudge to consolidate work
          - Warning (90%): urgent, must respond now
        """
        if not self._budget_pressure_enabled or self.max_iterations <= 0:
            return None
        progress = api_call_count / self.max_iterations
        remaining = self.max_iterations - api_call_count
        if progress >= self._budget_warning_threshold:
            return (
                f"[BUDGET WARNING: Iteration {api_call_count}/{self.max_iterations}. "
                f"Only {remaining} iteration(s) left. "
                "Provide your final response NOW. No more tool calls unless absolutely critical.]"
            )
        if progress >= self._budget_caution_threshold:
            return (
                f"[BUDGET: Iteration {api_call_count}/{self.max_iterations}. "
                f"{remaining} iterations left. Start consolidating your work.]"
            )
        return None

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
            _is_strict_api = "api.mistral.ai" in self.base_url.lower()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                for internal_field in ("reasoning", "finish_reason"):
                    api_msg.pop(internal_field, None)
                if _is_strict_api:
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

            summary_extra_body = {}
            _is_nous = "nousresearch" in self.base_url.lower()
            if self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                assistant_message, _ = self._normalize_codex_response(summary_response)
                final_response = (assistant_message.content or "").strip() if assistant_message else ""
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))

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
                if provider_preferences:
                    summary_extra_body["provider"] = provider_preferences

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak, normalize_anthropic_response as _nar
                    _ant_kw = _bak(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=getattr(self, '_is_anthropic_oauth', False))
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _msg, _ = _nar(summary_response, strip_tool_prefix=getattr(self, '_is_anthropic_oauth', False))
                    final_response = (_msg.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

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
                    retry_msg, _ = self._normalize_codex_response(retry_response)
                    final_response = (retry_msg.content or "").strip() if retry_msg else ""
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak2, normalize_anthropic_response as _nar2
                    _ant_kw2 = _bak2(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=getattr(self, '_is_anthropic_oauth', False),
                                     max_tokens=self.max_tokens, reasoning_config=self.reasoning_config)
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_msg, _ = _nar2(retry_response, strip_tool_prefix=getattr(self, '_is_anthropic_oauth', False))
                    final_response = (_retry_msg.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

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
        sync_honcho: bool = True,
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
            sync_honcho: When False, skip writing the final synthetic turn back
                to Honcho or queuing follow-up prefetch work.

        Returns:
            Dict: Complete conversation result with final response and message history
        """
        # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
        # Installed once, transparent when streams are healthy, prevents crash on write.
        _install_safe_stdio()

        # Store stream callback for _interruptible_api_call to pick up
        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message
        # Generate unique task_id if not provided to isolate VMs between concurrent tasks
        effective_task_id = task_id or str(uuid.uuid4())
        
        # Reset retry counters and iteration budget at the start of each turn
        # so subagent usage from a previous turn doesn't eat into the next one.
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._last_content_with_tools = None
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        self.iteration_budget = IterationBudget(self.max_iterations)
        
        # Initialize conversation (copy to avoid mutating the caller's list)
        messages = list(conversation_history) if conversation_history else []
        
        # Hydrate todo store from conversation history (gateway creates a fresh
        # AIAgent per message, so the in-memory store is empty -- we need to
        # recover the todo state from the most recent todo tool response in history)
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)
        
        # Prefill messages (few-shot priming) are injected at API-call time only,
        # never stored in the messages list. This keeps them ephemeral: they won't
        # be saved to session DB, session logs, or batch trajectories, but they're
        # automatically re-applied on every API call (including session continuations).
        
        # Track user turns for memory flush and periodic nudge logic
        self._user_turn_count += 1

        # Preserve the original user message before nudge injection.
        # Honcho should receive the actual user input, not system nudges.
        original_user_message = persist_user_message if persist_user_message is not None else user_message

        # Periodic memory nudge: remind the model to consider saving memories.
        # Counter resets whenever the memory tool is actually used.
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                user_message += (
                    "\n\n[System: You've had several exchanges. Consider: "
                    "has the user shared preferences, corrected you, or revealed "
                    "something about their workflow worth remembering for future sessions?]"
                )
                self._turns_since_memory = 0

        # Skill creation nudge: fires on the first user message after a long tool loop.
        # The counter increments per API iteration in the tool loop and is checked here.
        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            user_message += (
                "\n\n[System: The previous task involved many tool calls. "
                "Save the approach as a skill if it's reusable, or update "
                "any existing skill you used if it was wrong or incomplete.]"
            )
            self._iters_since_skill = 0

        # Honcho prefetch consumption:
        # - First turn: bake into cached system prompt (stable for the session).
        # - Later turns: attach recall to the current-turn user message at
        #   API-call time only (never persisted to history / session DB).
        #
        # This keeps the system-prefix cache stable while still allowing turn N
        # to consume background prefetch results from turn N-1.
        self._honcho_context = ""
        self._honcho_turn_context = ""
        _recall_mode = (self._honcho_config.recall_mode if self._honcho_config else "hybrid")
        if self._honcho and self._honcho_session_key and _recall_mode != "tools":
            try:
                prefetched_context = self._honcho_prefetch(original_user_message)
                if prefetched_context:
                    if not conversation_history:
                        self._honcho_context = prefetched_context
                    else:
                        self._honcho_turn_context = prefetched_context
            except Exception as e:
                logger.debug("Honcho prefetch failed (non-fatal): %s", e)

        # Add user message
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx
        
        if not self.quiet_mode:
            self._safe_print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
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
                # Bake Honcho context into the prompt so it's stable for
                # the entire session (not re-fetched per turn).
                if self._honcho_context:
                    self._cached_system_prompt = (
                        self._cached_system_prompt + "\n\n" + self._honcho_context
                    ).strip()
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
            _sys_tok_est = estimate_tokens_rough(active_system_prompt or "")
            _msg_tok_est = estimate_messages_tokens_rough(messages)
            _preflight_tokens = _sys_tok_est + _msg_tok_est

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                if not self.quiet_mode:
                    self._safe_print(
                        f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                        f">= {self.context_compressor.threshold_tokens:,} threshold"
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
                    # Re-estimate after compression
                    _sys_tok_est = estimate_tokens_rough(active_system_prompt or "")
                    _msg_tok_est = estimate_messages_tokens_rough(messages)
                    _preflight_tokens = _sys_tok_est + _msg_tok_est
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # Under threshold

        # Main conversation loop
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_response_prefix = ""
        
        # Clear any stale interrupt state at start
        self.clear_interrupt()
        
        while api_call_count < self.max_iterations and self.iteration_budget.remaining > 0:
            # Reset per-turn checkpoint dedup so each iteration can take one snapshot
            self._checkpoint_mgr.new_turn()

            # Check for interrupt request (e.g., user sent new message)
            if self._interrupt_requested:
                interrupted = True
                if not self.quiet_mode:
                    self._safe_print(f"\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1
            if not self.iteration_budget.consume():
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  Session iteration budget exhausted ({self.iteration_budget.max_total} total across agent + subagents)")
                break

            # Fire step_callback for gateway hooks (agent:step event)
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _m in reversed(messages):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            prev_tools = [
                                tc["function"]["name"]
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
            
            # Prepare messages for API call
            # If we have an ephemeral system prompt, prepend it to the messages
            # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
            # However, providers like Moonshot AI require a separate 'reasoning_content' field
            # on assistant messages with tool_calls. We handle both cases here.
            api_messages = []
            for idx, msg in enumerate(messages):
                api_msg = msg.copy()

                if idx == current_turn_user_idx and msg.get("role") == "user" and self._honcho_turn_context:
                    api_msg["content"] = _inject_honcho_turn_context(
                        api_msg.get("content", ""), self._honcho_turn_context
                    )

                # For ALL assistant messages, pass reasoning back to the API
                # This ensures multi-turn reasoning context is preserved
                if msg.get("role") == "assistant":
                    reasoning_text = msg.get("reasoning")
                    if reasoning_text:
                        # Add reasoning_content for API compatibility (Moonshot AI, Novita, OpenRouter)
                        api_msg["reasoning_content"] = reasoning_text

                # Remove 'reasoning' field - it's for trajectory storage only
                # We've copied it to 'reasoning_content' for the API above
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
                if "finish_reason" in api_msg:
                    api_msg.pop("finish_reason")
                # Strip Codex Responses API fields (call_id, response_item_id) for
                # strict providers like Mistral that reject unknown fields with 422.
                # Uses new dicts so the internal messages list retains the fields
                # for Codex Responses compatibility.
                if "api.mistral.ai" in self.base_url.lower():
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
                # The signature field helps maintain reasoning continuity
                api_messages.append(api_msg)

            # Build the final system message: cached prompt + ephemeral system prompt.
            # Ephemeral additions are API-call-time only (not persisted to session DB).
            # Honcho later-turn recall is intentionally kept OUT of the system prompt
            # so the stable cache prefix remains unchanged.
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages

            # Inject ephemeral prefill messages right after the system prompt
            # but before conversation history. Same API-call-time-only pattern.
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # Apply Anthropic prompt caching for Claude models via OpenRouter.
            # Auto-detected: if model name contains "claude" and base_url is OpenRouter,
            # inject cache_control breakpoints (system + last 3 messages) to reduce
            # input token costs by ~75% on multi-turn conversations.
            if self._use_prompt_caching:
                api_messages = apply_anthropic_cache_control(api_messages, cache_ttl=self._cache_ttl)

            # Safety net: strip orphaned tool results / add stubs for missing
            # results before sending to the API.  The compressor handles this
            # during compression, but orphans can also sneak in from session
            # loading or manual message manipulation.
            if hasattr(self, 'context_compressor') and self.context_compressor:
                api_messages = self.context_compressor._sanitize_tool_pairs(api_messages)

            # Calculate approximate request size for logging
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = total_chars // 4  # Rough estimate: 4 chars per token
            
            # Thinking spinner for quiet mode (animated during API call)
            thinking_spinner = None
            
            if not self.quiet_mode:
                self._vprint(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
                self._vprint(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
                self._vprint(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            elif not self._has_stream_consumers():
                # Animated thinking spinner in quiet mode (skip during streaming)
                face = random.choice(KawaiiSpinner.KAWAII_THINKING)
                verb = random.choice(KawaiiSpinner.THINKING_VERBS)
                if self.thinking_callback:
                    # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                    self.thinking_callback(f"{face} {verb}...")
                else:
                    spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                    thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type)
                    thinking_spinner.start()
            
            # Log request details if verbose
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            compression_attempts = 0
            max_compression_attempts = 3
            codex_auth_retry_attempted = False
            anthropic_auth_retry_attempted = False
            nous_auth_retry_attempted = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # Guard against UnboundLocalError if all retries fail

            while retry_count < max_retries:
                try:
                    api_kwargs = self._build_api_kwargs(api_messages)
                    if self.api_mode == "codex_responses":
                        api_kwargs = self._preflight_codex_api_kwargs(api_kwargs, allow_stream=False)

                    if os.getenv("HERMES_DUMP_REQUESTS", "").strip().lower() in {"1", "true", "yes", "on"}:
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    if self._has_stream_consumers():
                        # Streaming path: fire delta callbacks for real-time
                        # token delivery to CLI display, gateway, or TTS.
                        def _stop_spinner():
                            nonlocal thinking_spinner
                            if thinking_spinner:
                                thinking_spinner.stop("")
                                thinking_spinner = None
                            if self.thinking_callback:
                                self.thinking_callback("")

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
                        output_items = getattr(response, "output", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(output_items, list):
                            response_invalid = True
                            error_details.append("response.output is not a list")
                        elif len(output_items) == 0:
                            response_invalid = True
                            error_details.append("response.output is empty")
                    elif self.api_mode == "anthropic_messages":
                        content_blocks = getattr(response, "content", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(content_blocks, list):
                            response_invalid = True
                            error_details.append("response.content is not a list")
                        elif len(content_blocks) == 0:
                            response_invalid = True
                            error_details.append("response.content is empty")
                    else:
                        if response is None or not hasattr(response, 'choices') or response.choices is None or len(response.choices) == 0:
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
                            thinking_spinner.stop(f"(´;ω;`) oops, retrying...")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")
                        
                        # This is often rate limiting or provider returning malformed response
                        retry_count += 1
                        
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
                        
                        self._vprint(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                        self._vprint(f"{self.log_prefix}   🏢 Provider: {provider_name}", force=True)
                        self._vprint(f"{self.log_prefix}   📝 Provider message: {error_msg[:200]}", force=True)
                        self._vprint(f"{self.log_prefix}   ⏱️  Response time: {api_duration:.2f}s (fast response often indicates rate limiting)", force=True)
                        
                        if retry_count >= max_retries:
                            # Try fallback before giving up
                            if self._try_activate_fallback():
                                retry_count = 0
                                continue
                            self._vprint(f"{self.log_prefix}❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.", force=True)
                            logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Invalid API response shape. Likely rate limited or malformed provider response.",
                                "failed": True  # Mark as failure for filtering
                            }
                        
                        # Longer backoff for rate limiting (likely cause of None choices)
                        wait_time = min(5 * (2 ** (retry_count - 1)), 120)  # 5s, 10s, 20s, 40s, 80s, 120s
                        self._vprint(f"{self.log_prefix}⏳ Retrying in {wait_time}s (extended backoff for possible rate limit)...", force=True)
                        logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                        
                        # Sleep in small increments to stay responsive to interrupts
                        sleep_end = time.time() + wait_time
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                                self._persist_session(messages, conversation_history)
                                self.clear_interrupt()
                                return {
                                    "final_response": f"Operation interrupted: retrying API call after rate limit (retry {retry_count}/{max_retries}).",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
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
                        stop_reason_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}
                        finish_reason = stop_reason_map.get(response.stop_reason, "stop")
                    else:
                        finish_reason = response.choices[0].finish_reason

                    if finish_reason == "length":
                        self._vprint(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

                        if self.api_mode == "chat_completions":
                            assistant_message = response.choices[0].message
                            if not assistant_message.tool_calls:
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

                        # Cache discovered context length after successful call
                        if self.context_compressor._context_probed:
                            ctx = self.context_compressor.context_length
                            save_context_length(self.model, self.base_url, ctx)
                            self._safe_print(f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}")
                            self.context_compressor._context_probed = False

                        self.session_prompt_tokens += prompt_tokens
                        self.session_completion_tokens += completion_tokens
                        self.session_total_tokens += total_tokens
                        self.session_api_calls += 1
                        self.session_input_tokens += canonical_usage.input_tokens
                        self.session_output_tokens += canonical_usage.output_tokens
                        self.session_cache_read_tokens += canonical_usage.cache_read_tokens
                        self.session_cache_write_tokens += canonical_usage.cache_write_tokens
                        self.session_reasoning_tokens += canonical_usage.reasoning_tokens

                        cost_result = estimate_usage_cost(
                            self.model,
                            canonical_usage,
                            provider=self.provider,
                            base_url=self.base_url,
                        )
                        if cost_result.amount_usd is not None:
                            self.session_estimated_cost_usd += float(cost_result.amount_usd)
                        self.session_cost_status = cost_result.status
                        self.session_cost_source = cost_result.source

                        # Persist token counts to session DB for /insights.
                        # Gateway sessions persist via session_store.update_session()
                        # after run_conversation returns, so only persist here for
                        # CLI (and other non-gateway) platforms to avoid double-counting.
                        if (self._session_db and self.session_id
                                and getattr(self, 'platform', None) == 'cli'):
                            try:
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
                                )
                            except Exception:
                                pass  # never block the agent loop
                        
                        if self.verbose_logging:
                            logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                        
                        # Log cache hit stats when prompt caching is active
                        if self._use_prompt_caching:
                            if self.api_mode == "anthropic_messages":
                                # Anthropic uses cache_read_input_tokens / cache_creation_input_tokens
                                cached = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
                                written = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                            else:
                                # OpenRouter uses prompt_tokens_details.cached_tokens
                                details = getattr(response.usage, 'prompt_tokens_details', None)
                                cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
                                written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
                            prompt = usage_dict["prompt_tokens"]
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            if not self.quiet_mode:
                                self._vprint(f"{self.log_prefix}   💾 Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)")
                    
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
                        thinking_spinner.stop(f"(╥_╥) error, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    status_code = getattr(api_error, "status_code", None)
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
                        print(f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in ~/.hermes/.env for Hermes-managed OAuth/setup tokens")
                        print(f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in ~/.hermes/.env for API keys or legacy token values")
                        print(f"{self.log_prefix}     • For API keys: verify at https://console.anthropic.com/settings/keys")
                        print(f"{self.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                        print(f"{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_TOKEN \"\"")
                        print(f"{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_API_KEY \"\"")

                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    
                    # Enhanced error logging
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    logger.warning(
                        "API call failed (attempt %s/%s) error_type=%s %s error=%s",
                        retry_count,
                        max_retries,
                        error_type,
                        self._client_log_context(),
                        api_error,
                    )

                    self._vprint(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  Time elapsed before failure: {elapsed_time:.2f}s")
                    self._vprint(f"{self.log_prefix}   📝 Error: {str(api_error)[:200]}", force=True)
                    self._vprint(f"{self.log_prefix}   📊 Request context: {len(api_messages)} messages, ~{approx_tokens:,} tokens, {len(self.tools) if self.tools else 0} tools")
                    
                    # Check for interrupt before deciding to retry
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: handling API error ({error_type}: {str(api_error)[:80]}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    
                    # Check for 413 payload-too-large BEFORE generic 4xx handler.
                    # A 413 is a payload-size error — the correct response is to
                    # compress history and retry, not abort immediately.
                    status_code = getattr(api_error, "status_code", None)
                    is_payload_too_large = (
                        status_code == 413
                        or 'request entity too large' in error_msg
                        or 'payload too large' in error_msg
                        or 'error code: 413' in error_msg
                    )

                    if is_payload_too_large:
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                            logging.error(f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True
                            }
                        self._vprint(f"{self.log_prefix}⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )

                        if len(messages) < original_len:
                            self._vprint(f"{self.log_prefix}   🗜️  Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            self._vprint(f"{self.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                            logging.error(f"{self.log_prefix}413 payload too large. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Request payload too large (413). Cannot compress further.",
                                "partial": True
                            }

                    # Check for context-length errors BEFORE generic 4xx handler.
                    # Local backends (LM Studio, Ollama, llama.cpp) often return
                    # HTTP 400 with messages like "Context size has been exceeded"
                    # which must trigger compression, not an immediate abort.
                    is_context_length_error = any(phrase in error_msg for phrase in [
                        'context length', 'context size', 'maximum context',
                        'token limit', 'too many tokens', 'reduce the length',
                        'exceeds the limit', 'context window',
                        'request entity too large',  # OpenRouter/Nous 413 safety net
                        'prompt is too long',  # Anthropic: "prompt is too long: N tokens > M maximum"
                    ])

                    # Fallback heuristic: Anthropic sometimes returns a generic
                    # 400 invalid_request_error with just "Error" as the message
                    # when the context is too large.  If the error message is very
                    # short/generic AND the session is large, treat it as a
                    # probable context-length error and attempt compression rather
                    # than aborting.  This prevents an infinite failure loop where
                    # each failed message gets persisted, making the session even
                    # larger. (#1630)
                    if not is_context_length_error and status_code == 400:
                        ctx_len = getattr(getattr(self, 'context_compressor', None), 'context_length', 200000)
                        is_large_session = approx_tokens > ctx_len * 0.4 or len(api_messages) > 80
                        is_generic_error = len(error_msg.strip()) < 30  # e.g. just "error"
                        if is_large_session and is_generic_error:
                            is_context_length_error = True
                            self._vprint(
                                f"{self.log_prefix}⚠️  Generic 400 with large session "
                                f"(~{approx_tokens:,} tokens, {len(api_messages)} msgs) — "
                                f"treating as probable context overflow.",
                                force=True,
                            )
                    
                    if is_context_length_error:
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length

                        # Try to parse the actual limit from the error message
                        parsed_limit = parse_context_limit_from_error(error_msg)
                        if parsed_limit and parsed_limit < old_ctx:
                            new_ctx = parsed_limit
                            self._vprint(f"{self.log_prefix}⚠️  Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
                        else:
                            # Step down to the next probe tier
                            new_ctx = get_next_probe_tier(old_ctx)

                        if new_ctx and new_ctx < old_ctx:
                            compressor.context_length = new_ctx
                            compressor.threshold_tokens = int(new_ctx * compressor.threshold_percent)
                            compressor._context_probed = True
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True
                            }
                        self._vprint(f"{self.log_prefix}   🗜️  Context compression attempt {compression_attempts}/{max_compression_attempts}...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )

                        if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                            if len(messages) < original_len:
                                self._vprint(f"{self.log_prefix}   🗜️  Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            # Can't compress further and already at minimum tier
                            self._vprint(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 The conversation has accumulated too much content.", force=True)
                            logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                                "partial": True
                            }

                    # Check for non-retryable client errors (4xx HTTP status codes).
                    # These indicate a problem with the request itself (bad model ID,
                    # invalid API key, forbidden, etc.) and will never succeed on retry.
                    # Note: 413 and context-length errors are excluded — handled above.
                    # 429 (rate limit) is transient and MUST be retried with backoff.
                    # 529 (Anthropic overloaded) is also transient.
                    # Also catch local validation errors (ValueError, TypeError) — these
                    # are programming bugs, not transient failures.
                    _RETRYABLE_STATUS_CODES = {413, 429, 529}
                    is_local_validation_error = isinstance(api_error, (ValueError, TypeError))
                    # Detect generic 400s from Anthropic OAuth (transient server-side failures).
                    # Real invalid_request_error responses include a descriptive message;
                    # transient ones contain only "Error" or are empty. (ref: issue #1608)
                    _err_body = getattr(api_error, "body", None) or {}
                    _err_message = (_err_body.get("error", {}).get("message", "") if isinstance(_err_body, dict) else "")
                    _is_generic_400 = (status_code == 400 and _err_message.strip().lower() in ("error", ""))
                    is_client_status_error = isinstance(status_code, int) and 400 <= status_code < 500 and status_code not in _RETRYABLE_STATUS_CODES and not _is_generic_400
                    is_client_error = (is_local_validation_error or is_client_status_error or any(phrase in error_msg for phrase in [
                        'error code: 401', 'error code: 403',
                        'error code: 404', 'error code: 422',
                        'is not a valid model', 'invalid model', 'model not found',
                        'invalid api key', 'invalid_api_key', 'authentication',
                        'unauthorized', 'forbidden', 'not found',
                    ])) and not is_context_length_error

                    if is_client_error:
                        # Try fallback before aborting — a different provider
                        # may not have the same issue (rate limit, auth, etc.)
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue
                        self._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                        self._vprint(f"{self.log_prefix}❌ Non-retryable client error detected. Aborting immediately.", force=True)
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
                        # Try fallback before giving up entirely
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue
                        self._vprint(f"{self.log_prefix}❌ Max retries ({max_retries}) exceeded. Giving up.", force=True)
                        logging.error(f"{self.log_prefix}API call failed after {max_retries} retries. Last error: {api_error}")
                        logging.error(f"{self.log_prefix}Request details - Messages: {len(api_messages)}, Approx tokens: {approx_tokens:,}")
                        raise api_error

                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s, 60s
                    logger.warning(
                        "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                        wait_time,
                        retry_count,
                        max_retries,
                        self._client_log_context(),
                        api_error,
                    )
                    if retry_count >= max_retries:
                        self._vprint(f"{self.log_prefix}⚠️  API call failed after {retry_count} attempts: {str(api_error)[:100]}")
                        self._vprint(f"{self.log_prefix}⏳ Final retry in {wait_time}s...")
                    
                    # Sleep in small increments so we can respond to interrupts quickly
                    # instead of blocking the entire wait_time in one sleep() call
                    sleep_end = time.time() + wait_time
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
            
            # If the API call was interrupted, skip response processing
            if interrupted:
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                continue

            if restart_with_length_continuation:
                continue

            # Guard: if all retries exhausted without a successful response
            # (e.g. repeated context-length errors that exhausted retry_count),
            # the `response` variable is still None. Break out cleanly.
            if response is None:
                print(f"{self.log_prefix}❌ All API retries exhausted with no successful response.")
                self._persist_session(messages, conversation_history)
                break

            try:
                if self.api_mode == "codex_responses":
                    assistant_message, finish_reason = self._normalize_codex_response(response)
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import normalize_anthropic_response
                    assistant_message, finish_reason = normalize_anthropic_response(
                        response, strip_tool_prefix=getattr(self, "_is_anthropic_oauth", False)
                    )
                else:
                    assistant_message = response.choices[0].message
                
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

                # Handle assistant response
                if assistant_message.content and not self.quiet_mode:
                    if self.verbose_logging:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
                    else:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

                # Notify progress callback of model's thinking (used by subagent
                # delegation to relay the child's reasoning to the parent display).
                # Guard: only fire for subagents (_delegate_depth >= 1) to avoid
                # spamming gateway platforms with the main agent's every thought.
                if (assistant_message.content and self.tool_progress_callback
                        and getattr(self, '_delegate_depth', 0) > 0):
                    _think_text = assistant_message.content.strip()
                    # Strip reasoning XML tags that shouldn't leak to parent display
                    _think_text = re.sub(
                        r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                    ).strip()
                    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                    if first_line:
                        try:
                            self.tool_progress_callback("_thinking", first_line)
                        except Exception:
                            pass
                
                # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
                # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    if not hasattr(self, '_incomplete_scratchpad_retries'):
                        self._incomplete_scratchpad_retries = 0
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
                if hasattr(self, '_incomplete_scratchpad_retries'):
                    self._incomplete_scratchpad_retries = 0

                if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                    if not hasattr(self, "_codex_incomplete_retries"):
                        self._codex_incomplete_retries = 0
                    self._codex_incomplete_retries += 1

                    interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                    interim_has_content = bool((interim_msg.get("content") or "").strip())
                    interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False

                    if interim_has_content or interim_has_reasoning:
                        last_msg = messages[-1] if messages else None
                        duplicate_interim = (
                            isinstance(last_msg, dict)
                            and last_msg.get("role") == "assistant"
                            and last_msg.get("finish_reason") == "incomplete"
                            and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                            and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                        )
                        if not duplicate_interim:
                            messages.append(interim_msg)

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
                        if not hasattr(self, '_invalid_tool_retries'):
                            self._invalid_tool_retries = 0
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
                                content = f"Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": content,
                            })
                        continue
                    # Reset retry counter on successful tool call validation
                    if hasattr(self, '_invalid_tool_retries'):
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
                        # Track retries for invalid JSON arguments
                        self._invalid_json_retries += 1
                        
                        tool_name, error_msg = invalid_json_args[0]
                        self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")
                        
                        if self._invalid_json_retries < 3:
                            self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            # Instead of returning partial, inject a helpful message and let model recover
                            self._vprint(f"{self.log_prefix}⚠️  Injecting recovery message for invalid JSON...")
                            self._invalid_json_retries = 0  # Reset for next attempt
                            
                            # Add a user message explaining the issue
                            recovery_msg = (
                                f"Your tool call to '{tool_name}' had invalid JSON arguments. "
                                f"Error: {error_msg}. "
                                f"For tools with no required parameters, use an empty object: {{}}. "
                                f"Please either retry the tool call with valid JSON, or respond without using that tool."
                            )
                            recovery_dict = {"role": "user", "content": recovery_msg}
                            messages.append(recovery_dict)
                            continue
                    
                    # Reset retry counter on successful JSON validation
                    self._invalid_json_retries = 0
                    
                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    # If this turn has both content AND tool_calls, capture the content
                    # as a fallback final response. Common pattern: model delivers its
                    # answer and calls memory/skill tools as a side-effect in the same
                    # turn. If the follow-up turn after tools is empty, we use this.
                    turn_content = assistant_message.content or ""
                    if turn_content and self._has_content_after_think_block(turn_content):
                        self._last_content_with_tools = turn_content
                        # Show intermediate commentary so the user can follow along
                        if self.quiet_mode:
                            clean = self._strip_think_blocks(turn_content).strip()
                            if clean:
                                self._vprint(f"  ┊ 💬 {clean}")
                    
                    messages.append(assistant_msg)
                    
                    _msg_count_before_tools = len(messages)
                    self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                    # Refund the iteration if the ONLY tool(s) called were
                    # execute_code (programmatic tool calling).  These are
                    # cheap RPC-style calls that shouldn't eat the budget.
                    _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                    if _tc_names == {"execute_code"}:
                        self.iteration_budget.refund()
                    
                    # Estimate next prompt size using real token counts from the
                    # last API response + rough estimate of newly appended tool
                    # results.  This catches cases where tool results push the
                    # context past the limit that last_prompt_tokens alone misses
                    # (e.g. large file reads, web extractions).
                    _compressor = self.context_compressor
                    _new_tool_msgs = messages[_msg_count_before_tools:]
                    _new_chars = sum(len(str(m.get("content", "") or "")) for m in _new_tool_msgs)
                    _estimated_next_prompt = (
                        _compressor.last_prompt_tokens
                        + _compressor.last_completion_tokens
                        + _new_chars // 3  # conservative: JSON-heavy tool results ≈ 3 chars/token
                    )
                    if self.compression_enabled and _compressor.should_compress(_estimated_next_prompt):
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=self.context_compressor.last_prompt_tokens,
                            task_id=effective_task_id,
                        )
                    
                    # Save session log incrementally (so progress is visible even if interrupted)
                    self._session_messages = messages
                    self._save_session_log(messages)
                    
                    # Continue loop for next response
                    continue
                
                else:
                    # No tool calls - this is the final response
                    final_response = assistant_message.content or ""
                    
                    # Check if response only has think block with no actual content after it
                    if not self._has_content_after_think_block(final_response):
                        # If the previous turn already delivered real content alongside
                        # tool calls (e.g. "You're welcome!" + memory save), the model
                        # has nothing more to say. Use the earlier content immediately
                        # instead of wasting API calls on retries that won't help.
                        fallback = getattr(self, '_last_content_with_tools', None)
                        if fallback:
                            logger.debug("Empty follow-up after tool calls — using prior turn content as final response")
                            self._last_content_with_tools = None
                            self._empty_content_retries = 0
                            for i in range(len(messages) - 1, -1, -1):
                                msg = messages[i]
                                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                    tool_names = []
                                    for tc in msg["tool_calls"]:
                                        fn = tc.get("function", {})
                                        tool_names.append(fn.get("name", "unknown"))
                                    msg["content"] = f"Calling the {', '.join(tool_names)} tool{'s' if len(tool_names) > 1 else ''}..."
                                    break
                            final_response = self._strip_think_blocks(fallback).strip()
                            self._response_was_previewed = True
                            break

                        # No fallback available — this is a genuine empty response.
                        # Retry in case the model just had a bad generation.
                        if not hasattr(self, '_empty_content_retries'):
                            self._empty_content_retries = 0
                        self._empty_content_retries += 1
                        
                        reasoning_text = self._extract_reasoning(assistant_message)
                        self._vprint(f"{self.log_prefix}⚠️  Response only contains think block with no content after it")
                        if reasoning_text:
                            reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                            self._vprint(f"{self.log_prefix}   Reasoning: {reasoning_preview}")
                        else:
                            content_preview = final_response[:80] + "..." if len(final_response) > 80 else final_response
                            self._vprint(f"{self.log_prefix}   Content: '{content_preview}'")
                        
                        if self._empty_content_retries < 3:
                            self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._empty_content_retries}/3)...")
                            continue
                        else:
                            self._vprint(f"{self.log_prefix}❌ Max retries (3) for empty content exceeded.", force=True)
                            self._empty_content_retries = 0
                            
                            # If a prior tool_calls turn had real content, salvage it:
                            # rewrite that turn's content to a brief tool description,
                            # and use the original content as the final response here.
                            fallback = getattr(self, '_last_content_with_tools', None)
                            if fallback:
                                self._last_content_with_tools = None
                                # Find the last assistant message with tool_calls and rewrite it
                                for i in range(len(messages) - 1, -1, -1):
                                    msg = messages[i]
                                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                        tool_names = []
                                        for tc in msg["tool_calls"]:
                                            fn = tc.get("function", {})
                                            tool_names.append(fn.get("name", "unknown"))
                                        msg["content"] = f"Calling the {', '.join(tool_names)} tool{'s' if len(tool_names) > 1 else ''}..."
                                        break
                                # Strip <think> blocks from fallback content for user display
                                final_response = self._strip_think_blocks(fallback).strip()
                                self._response_was_previewed = True
                                break
                            
                            # No fallback -- append the empty message as-is
                            empty_msg = {
                                "role": "assistant",
                                "content": final_response,
                                "reasoning": reasoning_text,
                                "finish_reason": finish_reason,
                            }
                            messages.append(empty_msg)
                            
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            
                            return {
                                "final_response": final_response or None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Model generated only think blocks with no actual response after 3 retries"
                            }
                    
                    # Reset retry counter on successful content
                    if hasattr(self, '_empty_content_retries'):
                        self._empty_content_retries = 0

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
                    
                    # Strip <think> blocks from user-facing response (keep raw in messages for trajectory)
                    final_response = self._strip_think_blocks(final_response).strip()
                    
                    final_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    messages.append(final_msg)
                    
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except OSError:
                    logger.error(error_msg)
                
                if self.verbose_logging:
                    logging.exception("Detailed error information:")
                
                # If an assistant message with tool_calls was already appended,
                # the API expects a role="tool" result for every tool_call_id.
                # Fill in error results for any that weren't answered yet.
                pending_handled = False
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
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                messages.append(err_msg)
                        pending_handled = True
                    break
                
                if not pending_handled:
                    # Error happened before tool processing (e.g. response parsing).
                    # Use a user-role message so the model can see what went wrong
                    # without confusing the API with a fabricated assistant turn.
                    sys_err_msg = {
                        "role": "user",
                        "content": f"[System error during processing: {error_msg}]",
                    }
                    messages.append(sys_err_msg)
                
                # If we're near the limit, break to avoid infinite loops
                if api_call_count >= self.max_iterations - 1:
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    break
        
        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ):
            if self.iteration_budget.remaining <= 0 and not self.quiet_mode:
                print(f"\n⚠️  Session iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} used, including subagents)")
            final_response = self._handle_max_iterations(messages, api_call_count)
        
        # Determine if conversation completed successfully
        completed = final_response is not None and api_call_count < self.max_iterations

        # Save trajectory if enabled
        self._save_trajectory(messages, user_message, completed)

        # Clean up VM and browser for this task after conversation completes
        self._cleanup_task_resources(effective_task_id)

        # Persist session to both JSON log and SQLite
        self._persist_session(messages, conversation_history)

        # Sync conversation to Honcho for user modeling
        if final_response and not interrupted and sync_honcho:
            self._honcho_sync(original_user_message, final_response)
            self._queue_honcho_prefetch(original_user_message)

        # Extract reasoning from the last assistant message (if any)
        last_reasoning = None
        for msg in reversed(messages):
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
        self._response_was_previewed = False
        
        # Include interrupt message if one triggered the interrupt
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        # Clear interrupt state after handling
        self.clear_interrupt()

        # Clear stream callback so it doesn't leak into future calls
        self._stream_callback = None

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
    model: str = "anthropic/claude-opus-4.6",
    api_key: str = None,
    base_url: str = "https://openrouter.ai/api/v1",
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
        from model_tools import get_all_tool_names, get_toolset_for_tool, get_available_toolsets
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
        
        print(f"\n💡 Usage Examples:")
        print(f"  # Use predefined toolsets")
        print(f"  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print(f"  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print(f"  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print(f"  ")
        print(f"  # Combine multiple toolsets")
        print(f"  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print(f"  ")
        print(f"  # Disable toolsets")
        print(f"  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print(f"  ")
        print(f"  # Run with trajectory saving enabled")
        print(f"  python run_agent.py --save_trajectories --query='your question here'")
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
        print(f"💾 Trajectory saving: ENABLED")
        print(f"   - Successful conversations → trajectory_samples.jsonl")
        print(f"   - Failed conversations → failed_trajectories.jsonl")
    
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
        print(f"\n🎯 FINAL RESPONSE:")
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
    fire.Fire(main)
