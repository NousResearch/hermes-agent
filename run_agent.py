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

import copy
import json
import logging
logger = logging.getLogger(__name__)
import os
import random
import re
import sys
import time
import threading
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime
from pathlib import Path

# Load .env from ~/.hermes/.env first, then project root as dev fallback
from dotenv import load_dotenv

_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_user_env = _hermes_home / ".env"
_project_env = Path(__file__).parent / '.env'
if _user_env.exists():
    try:
        load_dotenv(dotenv_path=_user_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_user_env, encoding="latin-1")
    logger.info("Loaded environment variables from %s", _user_env)
elif _project_env.exists():
    try:
        load_dotenv(dotenv_path=_project_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_project_env, encoding="latin-1")
    logger.info("Loaded environment variables from %s", _project_env)
else:
    logger.info("No .env file found. Using system environment variables.")

# Point mini-swe-agent at ~/.hermes/ so it shares our config
os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(_hermes_home))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

# Import our tool system
from model_tools import get_tool_definitions, check_toolset_requirements
from tools.terminal_tool import cleanup_vm
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser

import requests

from hermes_constants import OPENROUTER_BASE_URL

# Agent internals extracted to agent/ package for modularity
from agent.context_compressor import ContextCompressor
from agent.prompt_caching import apply_anthropic_cache_control
from agent.display import KawaiiSpinner
from agent.trajectory import has_incomplete_scratchpad
from agent.prompt_assembler import PromptAssembler
from agent.session_persister import SessionPersister
from agent.tool_executor import ToolExecConfig, execute_tool_calls


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
        model: str = "anthropic/claude-opus-4.6",  # OpenRouter format
        max_iterations: int = 60,  # Default tool-calling iterations
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
        session_id: str = None,
        tool_progress_callback: callable = None,
        clarify_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db=None,
    ):
        """
        Initialize the AI Agent.

        Args:
            base_url (str): Base URL for the model API (optional)
            api_key (str): API key for authentication (optional, uses env var if not provided)
            model (str): Model name to use (default: "anthropic/claude-opus-4.6")
            max_iterations (int): Maximum number of tool calling iterations (default: 60)
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
                If None, defaults to {"enabled": True, "effort": "xhigh"} for OpenRouter. Set to disable/customize reasoning.
            prefill_messages (List[Dict]): Messages to prepend to conversation history as prefilled context.
                Useful for injecting a few-shot example or priming the model's response style.
                Example: [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
            platform (str): The interface platform the user is on (e.g. "cli", "telegram", "discord", "whatsapp").
                Used to inject platform-specific formatting hints into the system prompt.
            skip_context_files (bool): If True, skip auto-injection of SOUL.md, AGENTS.md, and .cursorrules
                into the system prompt. Use this for batch processing and data generation to avoid
                polluting trajectories with user-specific persona or project instructions.
        """
        self.model = model
        self.max_iterations = max_iterations
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform  # "cli", "telegram", "discord", "whatsapp", etc.
        self.skip_context_files = skip_context_files
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        # Store effective base URL for feature detection (prompt caching, reasoning, etc.)
        # When no base_url is provided, the client defaults to OpenRouter, so reflect that here.
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.tool_progress_callback = tool_progress_callback
        self.clarify_callback = clarify_callback
        self._last_reported_tool = None  # Track for "new tool" mode
        
        # Interrupt mechanism for breaking out of tool loops
        self._interrupt_requested = False
        self._interrupt_message = None  # Optional message that triggered interrupt
        
        # Subagent delegation state
        self._delegate_depth = 0        # 0 = top-level agent, incremented for children
        self._active_children = []      # Running child AIAgents (for interrupt propagation)
        
        # Store OpenRouter provider preferences
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort

        # Store toolset filtering options
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # Model response configuration
        self.max_tokens = max_tokens  # None = use model default
        self.reasoning_config = reasoning_config  # None = use default (xhigh for OpenRouter)
        self.prefill_messages = prefill_messages or []  # Prefilled conversation turns
        
        # Anthropic prompt caching: auto-enabled for Claude models via OpenRouter.
        # Reduces input costs by ~75% on multi-turn conversations by caching the
        # conversation prefix. Uses system_and_3 strategy (4 breakpoints).
        is_openrouter = "openrouter" in self.base_url.lower()
        is_claude = "claude" in self.model.lower()
        self._use_prompt_caching = is_openrouter and is_claude
        self._cache_ttl = "5m"  # Default 5-minute TTL (1.25x write cost)
        
        # Configure logging
        if self.verbose_logging:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
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
        
        # Initialize OpenAI client - defaults to OpenRouter
        client_kwargs = {}
        
        # Default to OpenRouter if no base_url provided
        if base_url:
            client_kwargs["base_url"] = base_url
        else:
            client_kwargs["base_url"] = OPENROUTER_BASE_URL
        
        # Handle API key - OpenRouter is the primary provider
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            # Primary: OPENROUTER_API_KEY, fallback to direct provider keys
            client_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY", "")
        
        # OpenRouter app attribution — shows hermes-agent in rankings/analytics
        effective_base = client_kwargs.get("base_url", "")
        if "openrouter" in effective_base.lower():
            client_kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
                "X-OpenRouter-Title": "Hermes Agent",
                "X-OpenRouter-Categories": "cli-agent",
            }
        
        self._client_kwargs = client_kwargs  # stored for rebuilding after interrupt
        try:
            self.client = OpenAI(**client_kwargs)
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
            print(f"💾 Prompt caching: ENABLED (Claude via OpenRouter, {self._cache_ttl} TTL)")
        
        # Session logging setup - auto-save conversation trajectories for debugging
        self.session_start = datetime.now()
        if session_id:
            _initial_session_id = session_id
        else:
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            _initial_session_id = f"{timestamp_str}_{short_uuid}"

        # Session logs go into ~/.hermes/sessions/ alongside gateway sessions
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{_initial_session_id}.json"
        
        # Track conversation messages for session logging
        self._session_messages: List[Dict[str, Any]] = []
        
        # Cached system prompt -- built once per session, only rebuilt on compression
        self._cached_system_prompt: Optional[str] = None
        
        # SQLite session store (optional -- provided by CLI or gateway)
        self._session_db = session_db
        if self._session_db:
            try:
                self._session_db.create_session(
                    session_id=_initial_session_id,
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
        
        # Skills config: nudge interval for skill creation reminders
        self._skill_nudge_interval = 15
        try:
            from hermes_cli.config import load_config as _load_skills_config
            skills_config = _load_skills_config().get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 15))
        except Exception:
            pass

        # Composed components (extracted in Phases 3-5)
        self._prompt_assembler = PromptAssembler(
            platform=self.platform,
            skip_context_files=self.skip_context_files,
        )

        self._persister = SessionPersister(
            session_id=_initial_session_id,
            session_db=self._session_db,
            logs_dir=self.logs_dir,
            model=self.model,
            base_url=self.base_url,
            platform=self.platform,
            session_start=self.session_start,
            save_trajectories=self.save_trajectories,
            tools=self.tools,
            verbose_logging=self.verbose_logging,
            quiet_mode=self.quiet_mode,
        )

        self._tool_exec_config = ToolExecConfig(
            todo_store=self._todo_store,
            memory_store=self._memory_store,
            session_db=self._session_db,
            clarify_callback=self.clarify_callback,
            tool_delay=self.tool_delay,
            tool_progress_callback=self.tool_progress_callback,
            quiet_mode=self.quiet_mode,
            verbose_logging=self.verbose_logging,
            log_prefix=self.log_prefix,
            log_prefix_chars=self.log_prefix_chars,
        )

        # Initialize context compressor for automatic context management
        # Compresses conversation when approaching model's context limit
        # Configuration via environment variables (can be set in .env or cli-config.yaml)
        compression_threshold = float(os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "0.85"))
        compression_enabled = os.getenv("CONTEXT_COMPRESSION_ENABLED", "true").lower() in ("true", "1", "yes")
        
        self.context_compressor = ContextCompressor(
            model=self.model,
            threshold_percent=compression_threshold,
            protect_first_n=3,
            protect_last_n=4,
            summary_target_tokens=500,
            quiet_mode=self.quiet_mode,
        )
        self.compression_enabled = compression_enabled
        self._user_turn_count = 0
        
        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")

    @property
    def session_id(self):
        return self._persister.session_id

    @session_id.setter
    def session_id(self, value):
        self._persister.session_id = value

    def _on_tool_executed(self, tool_name: str):
        """Reset nudge counters when relevant tools are used."""
        if tool_name == "memory":
            self._turns_since_memory = 0
        elif tool_name == "skill_manage":
            self._iters_since_skill = 0

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
        Dump a debug-friendly HTTP request record for chat.completions.create().

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
                    "url": f"{self.base_url.rstrip('/')}/chat/completions",
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

            print(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if os.getenv("HERMES_DUMP_REQUEST_STDOUT", "").strip().lower() in {"1", "true", "yes", "on"}:
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None

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
        for child in self._active_children:
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
                print(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)
    
    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested
    
    def _interruptible_api_call(self, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.
        
        On interrupt, closes the HTTP client to cancel the in-flight request
        (stops token generation and avoids wasting money), then rebuilds the
        client for future calls.
        """
        result = {"response": None, "error": None}

        def _call():
            try:
                result["response"] = self.client.chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if self._interrupt_requested:
                # Force-close the HTTP connection to stop token generation
                try:
                    self.client.close()
                except Exception:
                    pass
                # Rebuild the client for future calls (cheap, no network)
                try:
                    self.client = OpenAI(**self._client_kwargs)
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the chat completions API call."""
        provider_preferences = {}
        if self.providers_allowed:
            provider_preferences["only"] = self.providers_allowed
        if self.providers_ignored:
            provider_preferences["ignore"] = self.providers_ignored
        if self.providers_order:
            provider_preferences["order"] = self.providers_order
        if self.provider_sort:
            provider_preferences["sort"] = self.provider_sort

        api_kwargs = {
            "model": self.model,
            "messages": api_messages,
            "tools": self.tools if self.tools else None,
            "timeout": 600.0,
        }

        if self.max_tokens is not None:
            api_kwargs["max_tokens"] = self.max_tokens

        extra_body = {}

        if provider_preferences:
            extra_body["provider"] = provider_preferences

        _is_openrouter = "openrouter" in self.base_url.lower()
        _is_nous = "nousresearch" in self.base_url.lower()

        if _is_openrouter or _is_nous:
            if self.reasoning_config is not None:
                extra_body["reasoning"] = self.reasoning_config
            else:
                extra_body["reasoning"] = {
                    "enabled": True,
                    "effort": "xhigh"
                }

        # Nous Portal product attribution
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        return api_kwargs

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        reasoning_text = self._extract_reasoning(assistant_message)

        if reasoning_text and self.verbose_logging:
            preview = reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {preview}")

        msg = {
            "role": "assistant",
            "content": assistant_message.content or "",
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            msg["reasoning_details"] = [
                {"type": d.get("type"), "text": d.get("text"), "signature": d.get("signature")}
                for d in assistant_message.reasoning_details
                if isinstance(d, dict)
            ]

        if assistant_message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in assistant_message.tool_calls
            ]

        return msg

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
        effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, '_session_messages', None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Please save anything worth remembering to your memories.]"
        )
        flush_msg = {"role": "user", "content": flush_content}
        messages.append(flush_msg)

        try:
            # Build API messages for the flush call
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                if msg.get("role") == "assistant":
                    reasoning = msg.get("reasoning")
                    if reasoning:
                        api_msg["reasoning_content"] = reasoning
                api_msg.pop("reasoning", None)
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

            api_kwargs = {
                "model": self.model,
                "messages": api_messages,
                "tools": [memory_tool_def],
                "temperature": 0.3,
                "max_tokens": 1024,
            }

            response = self.client.chat.completions.create(**api_kwargs, timeout=30.0)

            if response.choices:
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    # Execute only memory tool calls
                    for tc in assistant_message.tool_calls:
                        if tc.function.name == "memory":
                            try:
                                args = json.loads(tc.function.arguments)
                                from tools.memory_tool import memory_tool as _memory_tool
                                result = _memory_tool(
                                    action=args.get("action"),
                                    target=args.get("target", "memory"),
                                    content=args.get("content"),
                                    old_text=args.get("old_text"),
                                    store=self._memory_store,
                                )
                                if not self.quiet_mode:
                                    print(f"  🧠 Memory flush: saved to {args.get('target', 'memory')}")
                            except Exception as e:
                                logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # Strip flush artifacts: remove everything from the flush message onward
            while messages and messages[-1] is not flush_msg and len(messages) > 0:
                messages.pop()
            if messages and messages[-1] is flush_msg:
                messages.pop()

    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None) -> tuple:
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

        self._prompt_assembler.invalidate(memory_store=self._memory_store)
        new_system_prompt = self._prompt_assembler.build(
            valid_tool_names=self.valid_tool_names,
            system_message=system_message,
            memory_store=self._memory_store,
            memory_enabled=self._memory_enabled,
            user_profile_enabled=self._user_profile_enabled,
        )
        self._cached_system_prompt = new_system_prompt

        self._persister.create_compression_session(
            platform=self.platform,
            model=self.model,
            parent_session_id=self.session_id,
        )
        self._persister.update_system_prompt(new_system_prompt)

        return compressed, new_system_prompt

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
            api_messages = messages.copy()
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
            _is_openrouter = "openrouter" in self.base_url.lower()
            _is_nous = "nousresearch" in self.base_url.lower()
            if _is_openrouter or _is_nous:
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "xhigh"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            summary_kwargs = {
                "model": self.model,
                "messages": api_messages,
            }
            if self.max_tokens is not None:
                summary_kwargs["max_tokens"] = self.max_tokens
            if summary_extra_body:
                summary_kwargs["extra_body"] = summary_extra_body

            summary_response = self.client.chat.completions.create(**summary_kwargs)

            if summary_response.choices and summary_response.choices[0].message.content:
                final_response = summary_response.choices[0].message.content
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                messages.append({"role": "assistant", "content": final_response})
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
        task_id: str = None
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling until completion.

        Args:
            user_message (str): The user's message/question
            system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
            conversation_history (List[Dict]): Previous conversation messages (optional)
            task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)

        Returns:
            Dict: Complete conversation result with final response and message history
        """
        # Generate unique task_id if not provided to isolate VMs between concurrent tasks
        effective_task_id = task_id or str(uuid.uuid4())
        
        # Reset retry counters at the start of each conversation to prevent state leakage
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._last_content_with_tools = None
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        
        # Initialize conversation
        messages = conversation_history or []
        
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

        # Periodic memory nudge: remind the model to consider saving memories.
        # Counter resets whenever the memory tool is actually used.
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                user_message += (
                    "\n\n[System: You've had several exchanges in this session. "
                    "Consider whether there's anything worth saving to your memories.]"
                )
                self._turns_since_memory = 0

        # Skill creation nudge: fires on the first user message after a long tool loop.
        # The counter increments per API iteration in the tool loop and is checked here.
        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            user_message += (
                "\n\n[System: The previous task involved many steps. "
                "If you discovered a reusable workflow, consider saving it as a skill.]"
            )
            self._iters_since_skill = 0

        # Add user message
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        self._persister.log_message(user_msg)
        
        if not self.quiet_mode:
            print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
        # ── System prompt (cached per session for prefix caching) ──
        # Built once on first call, reused for all subsequent calls.
        # Only rebuilt after context compression events (which invalidate
        # the cache and reload memory from disk).
        if self._cached_system_prompt is None:
            self._cached_system_prompt = self._prompt_assembler.build(
                valid_tool_names=self.valid_tool_names,
                system_message=system_message,
                memory_store=self._memory_store,
                memory_enabled=self._memory_enabled,
                user_profile_enabled=self._user_profile_enabled,
            )
            self._persister.update_system_prompt(self._cached_system_prompt)

        active_system_prompt = self._cached_system_prompt

        # Main conversation loop
        api_call_count = 0
        final_response = None
        interrupted = False
        
        # Clear any stale interrupt state at start
        self.clear_interrupt()
        
        while api_call_count < self.max_iterations:
            # Check for interrupt request (e.g., user sent new message)
            if self._interrupt_requested:
                interrupted = True
                if not self.quiet_mode:
                    print(f"\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1

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
            for msg in messages:
                api_msg = msg.copy()
                
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
                # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
                # The signature field helps maintain reasoning continuity
                api_messages.append(api_msg)
            
            # Build the final system message: cached prompt + ephemeral system prompt.
            # The ephemeral part is appended here (not baked into the cached prompt)
            # so it stays out of the session DB and logs.
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
            
            # Calculate approximate request size for logging
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = total_chars // 4  # Rough estimate: 4 chars per token
            
            # Thinking spinner for quiet mode (animated during API call)
            thinking_spinner = None
            
            if not self.quiet_mode:
                print(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
                print(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
                print(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            else:
                # Animated thinking spinner in quiet mode
                face = random.choice(KawaiiSpinner.KAWAII_THINKING)
                verb = random.choice(KawaiiSpinner.THINKING_VERBS)
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
            max_retries = 6  # Increased to allow longer backoff periods

            while retry_count <= max_retries:
                try:
                    api_kwargs = self._build_api_kwargs(api_messages)

                    if os.getenv("HERMES_DUMP_REQUESTS", "").strip().lower() in {"1", "true", "yes", "on"}:
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    response = self._interruptible_api_call(api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    
                    # Stop thinking spinner silently -- the response box or tool
                    # execution messages that follow are more informative.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    
                    if not self.quiet_mode:
                        print(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                    
                    if self.verbose_logging:
                        # Log response with provider info if available
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    # Validate response has valid choices before proceeding
                    if response is None or not hasattr(response, 'choices') or response.choices is None or len(response.choices) == 0:
                        # Stop spinner before printing error messages
                        if thinking_spinner:
                            thinking_spinner.stop(f"(´;ω;`) oops, retrying...")
                            thinking_spinner = None
                        
                        # This is often rate limiting or provider returning malformed response
                        retry_count += 1
                        error_details = []
                        if response is None:
                            error_details.append("response is None")
                        elif not hasattr(response, 'choices'):
                            error_details.append("response has no 'choices' attribute")
                        elif response.choices is None:
                            error_details.append("response.choices is None")
                        else:
                            error_details.append("response.choices is empty")
                        
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
                        
                        print(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}")
                        print(f"{self.log_prefix}   🏢 Provider: {provider_name}")
                        print(f"{self.log_prefix}   📝 Provider message: {error_msg[:200]}")
                        print(f"{self.log_prefix}   ⏱️  Response time: {api_duration:.2f}s (fast response often indicates rate limiting)")
                        
                        if retry_count > max_retries:
                            print(f"{self.log_prefix}❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                            logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                            self._persister.persist(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Invalid API response (choices is None/empty). Likely rate limited by provider.",
                                "failed": True  # Mark as failure for filtering
                            }
                        
                        # Longer backoff for rate limiting (likely cause of None choices)
                        wait_time = min(5 * (2 ** (retry_count - 1)), 120)  # 5s, 10s, 20s, 40s, 80s, 120s
                        print(f"{self.log_prefix}⏳ Retrying in {wait_time}s (extended backoff for possible rate limit)...")
                        logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                        
                        # Sleep in small increments to stay responsive to interrupts
                        sleep_end = time.time() + wait_time
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                print(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.")
                                self._persister.persist(messages, conversation_history)
                                return {
                                    "final_response": "Operation interrupted.",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
                        continue  # Retry the API call

                    # Check finish_reason before proceeding
                    finish_reason = response.choices[0].finish_reason
                    
                    # Handle "length" finish_reason - response was truncated
                    if finish_reason == "length":
                        print(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens")
                        
                        # If we have prior messages, roll back to last complete state
                        if len(messages) > 1:
                            print(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                            
                            self._cleanup_task_resources(effective_task_id)
                            self._persister.persist(messages, conversation_history)
                            
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
                            print(f"{self.log_prefix}❌ First response truncated - cannot recover")
                            self._persister.persist(messages, conversation_history)
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
                        usage_dict = {
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                            "total_tokens": getattr(response.usage, 'total_tokens', 0),
                        }
                        self.context_compressor.update_from_response(usage_dict)
                        
                        if self.verbose_logging:
                            logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                        
                        # Log cache hit stats when prompt caching is active
                        if self._use_prompt_caching:
                            details = getattr(response.usage, 'prompt_tokens_details', None)
                            cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
                            written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
                            prompt = usage_dict["prompt_tokens"]
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            if not self.quiet_mode:
                                print(f"{self.log_prefix}   💾 Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)")
                    
                    break  # Success, exit retry loop

                except InterruptedError:
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    print(f"{self.log_prefix}⚡ Interrupted during API call.")
                    self._persister.persist(messages, conversation_history)
                    interrupted = True
                    final_response = "Operation interrupted."
                    break

                except Exception as api_error:
                    # Stop spinner before printing error messages
                    if thinking_spinner:
                        thinking_spinner.stop(f"(╥_╥) error, retrying...")
                        thinking_spinner = None
                    
                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    
                    # Enhanced error logging
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    
                    print(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}")
                    print(f"{self.log_prefix}   ⏱️  Time elapsed before failure: {elapsed_time:.2f}s")
                    print(f"{self.log_prefix}   📝 Error: {str(api_error)[:200]}")
                    print(f"{self.log_prefix}   📊 Request context: {len(api_messages)} messages, ~{approx_tokens:,} tokens, {len(self.tools) if self.tools else 0} tools")
                    
                    # Check for interrupt before deciding to retry
                    if self._interrupt_requested:
                        print(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.")
                        self._persister.persist(messages, conversation_history)
                        return {
                            "final_response": "Operation interrupted.",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    
                    # Check for non-retryable client errors (4xx HTTP status codes).
                    # These indicate a problem with the request itself (bad model ID,
                    # invalid API key, forbidden, etc.) and will never succeed on retry.
                    status_code = getattr(api_error, "status_code", None)
                    is_client_status_error = isinstance(status_code, int) and 400 <= status_code < 500
                    is_client_error = is_client_status_error or any(phrase in error_msg for phrase in [
                        'error code: 400', 'error code: 401', 'error code: 403',
                        'error code: 404', 'error code: 422',
                        'is not a valid model', 'invalid model', 'model not found',
                        'invalid api key', 'invalid_api_key', 'authentication',
                        'unauthorized', 'forbidden', 'not found',
                    ])
                    
                    if is_client_error:
                        self._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                        print(f"{self.log_prefix}❌ Non-retryable client error detected. Aborting immediately.")
                        print(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.")
                        logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")
                        self._persister.persist(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": str(api_error),
                        }
                    
                    # Check for non-retryable errors (context length exceeded)
                    is_context_length_error = any(phrase in error_msg for phrase in [
                        'context length', 'maximum context', 'token limit', 
                        'too many tokens', 'reduce the length', 'exceeds the limit'
                    ])
                    
                    if is_context_length_error:
                        print(f"{self.log_prefix}⚠️  Context length exceeded - attempting compression...")
                        
                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens
                        )
                        
                        if len(messages) < original_len:
                            print(f"{self.log_prefix}   🗜️  Compressed {original_len} → {len(messages)} messages, retrying...")
                            continue  # Retry with compressed messages
                        else:
                            # Can't compress further
                            print(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.")
                            print(f"{self.log_prefix}   💡 The conversation has accumulated too much content.")
                            logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                            self._persister.persist(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                                "partial": True
                            }
                    
                    if retry_count > max_retries:
                        print(f"{self.log_prefix}❌ Max retries ({max_retries}) exceeded. Giving up.")
                        logging.error(f"{self.log_prefix}API call failed after {max_retries} retries. Last error: {api_error}")
                        logging.error(f"{self.log_prefix}Request details - Messages: {len(api_messages)}, Approx tokens: {approx_tokens:,}")
                        raise api_error

                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s, 60s
                    print(f"⚠️  OpenAI-compatible API call failed (attempt {retry_count}/{max_retries}): {str(api_error)[:100]}")
                    print(f"⏳ Retrying in {wait_time}s...")
                    logging.warning(f"API retry {retry_count}/{max_retries} after error: {api_error}")
                    
                    # Sleep in small increments so we can respond to interrupts quickly
                    # instead of blocking the entire wait_time in one sleep() call
                    sleep_end = time.time() + wait_time
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            print(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.")
                            self._persister.persist(messages, conversation_history)
                            return {
                                "final_response": "Operation interrupted.",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)  # Check interrupt every 200ms
            
            # If the API call was interrupted, skip response processing
            if interrupted:
                break

            try:
                assistant_message = response.choices[0].message
                
                # Handle assistant response
                if assistant_message.content and not self.quiet_mode:
                    print(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")
                
                # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
                # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    if not hasattr(self, '_incomplete_scratchpad_retries'):
                        self._incomplete_scratchpad_retries = 0
                    self._incomplete_scratchpad_retries += 1
                    
                    print(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                    
                    if self._incomplete_scratchpad_retries <= 2:
                        print(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                        # Don't add the broken message, just retry
                        continue
                    else:
                        # Max retries - discard this turn and save as partial
                        print(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.")
                        self._incomplete_scratchpad_retries = 0
                        
                        rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                        self._cleanup_task_resources(effective_task_id)
                        self._persister.persist(messages, conversation_history)
                        
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
                
                # Check for tool calls
                if assistant_message.tool_calls:
                    if not self.quiet_mode:
                        print(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
                    # Validate tool call names - detect model hallucinations
                    invalid_tool_calls = [
                        tc.function.name for tc in assistant_message.tool_calls 
                        if tc.function.name not in self.valid_tool_names
                    ]
                    
                    if invalid_tool_calls:
                        # Track retries for invalid tool calls
                        if not hasattr(self, '_invalid_tool_retries'):
                            self._invalid_tool_retries = 0
                        self._invalid_tool_retries += 1
                        
                        invalid_preview = invalid_tool_calls[0][:80] + "..." if len(invalid_tool_calls[0]) > 80 else invalid_tool_calls[0]
                        print(f"{self.log_prefix}⚠️  Invalid tool call detected: '{invalid_preview}'")
                        print(f"{self.log_prefix}   Valid tools: {sorted(self.valid_tool_names)}")
                        
                        if self._invalid_tool_retries < 3:
                            print(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_tool_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            print(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.")
                            # Return partial result - don't include the bad tool call in messages
                            self._invalid_tool_retries = 0
                            self._persister.persist(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}"
                            }
                    
                    # Reset retry counter on successful tool call validation
                    if hasattr(self, '_invalid_tool_retries'):
                        self._invalid_tool_retries = 0
                    
                    # Validate tool call arguments are valid JSON
                    # Handle empty strings as empty objects (common model quirk)
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
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
                        print(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")
                        
                        if self._invalid_json_retries < 3:
                            print(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            # Instead of returning partial, inject a helpful message and let model recover
                            print(f"{self.log_prefix}⚠️  Injecting recovery message for invalid JSON...")
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
                            self._persister.log_message(recovery_dict)
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
                                preview = clean[:120] + "..." if len(clean) > 120 else clean
                                print(f"  ┊ 💬 {preview}")
                    
                    messages.append(assistant_msg)
                    self._persister.log_message(assistant_msg)
                    
                    execute_tool_calls(
                        self._tool_exec_config, assistant_message, messages, effective_task_id,
                        is_interrupted=lambda: self._interrupt_requested,
                        log_msg_to_db=self._persister.log_message,
                        on_tool_executed=self._on_tool_executed,
                        parent_agent=self,
                    )
                    
                    if self.compression_enabled and self.context_compressor.should_compress():
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=self.context_compressor.last_prompt_tokens
                        )
                    
                    # Save session log incrementally (so progress is visible even if interrupted)
                    self._persister.maybe_save_session_log(messages)
                    
                    # Continue loop for next response
                    continue
                
                else:
                    # No tool calls - this is the final response
                    final_response = assistant_message.content or ""
                    
                    # Check if response only has think block with no actual content after it
                    if not self._has_content_after_think_block(final_response):
                        # Track retries for empty-after-think responses
                        if not hasattr(self, '_empty_content_retries'):
                            self._empty_content_retries = 0
                        self._empty_content_retries += 1
                        
                        # Show the reasoning/thinking content so the user can see
                        # what the model was thinking even though content is empty
                        reasoning_text = self._extract_reasoning(assistant_message)
                        print(f"{self.log_prefix}⚠️  Response only contains think block with no content after it")
                        if reasoning_text:
                            reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                            print(f"{self.log_prefix}   Reasoning: {reasoning_preview}")
                        else:
                            content_preview = final_response[:80] + "..." if len(final_response) > 80 else final_response
                            print(f"{self.log_prefix}   Content: '{content_preview}'")
                        
                        if self._empty_content_retries < 3:
                            print(f"{self.log_prefix}🔄 Retrying API call ({self._empty_content_retries}/3)...")
                            continue
                        else:
                            print(f"{self.log_prefix}❌ Max retries (3) for empty content exceeded.")
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
                                final_response = fallback
                                break
                            
                            # No fallback -- append the empty message as-is
                            empty_msg = {
                                "role": "assistant",
                                "content": final_response,
                                "reasoning": reasoning_text,
                                "finish_reason": finish_reason,
                            }
                            messages.append(empty_msg)
                            self._persister.log_message(empty_msg)
                            
                            self._cleanup_task_resources(effective_task_id)
                            self._persister.persist(messages, conversation_history)
                            
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
                    
                    final_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    messages.append(final_msg)
                    self._persister.log_message(final_msg)
                    
                    if not self.quiet_mode:
                        print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                print(f"❌ {error_msg}")
                
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
                                self._persister.log_message(err_msg)
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
                    self._persister.log_message(sys_err_msg)
                
                # If we're near the limit, break to avoid infinite loops
                if api_call_count >= self.max_iterations - 1:
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    break
        
        if api_call_count >= self.max_iterations and final_response is None:
            final_response = self._handle_max_iterations(messages, api_call_count)
        
        # Determine if conversation completed successfully
        completed = final_response is not None and api_call_count < self.max_iterations

        # Save trajectory if enabled
        self._persister.save_trajectory(messages, user_message, completed)

        # Clean up VM and browser for this task after conversation completes
        self._cleanup_task_resources(effective_task_id)

        # Persist session to both JSON log and SQLite
        self._persister.persist(messages, conversation_history)
        
        # Build result with interrupt info if applicable
        result = {
            "final_response": final_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "partial": False,  # True only when stopped due to invalid tool calls
            "interrupted": interrupted,
        }
        
        # Include interrupt message if one triggered the interrupt
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        # Clear interrupt state after handling
        self.clear_interrupt()
        
        return result
    
    def chat(self, message: str) -> str:
        """
        Simple chat interface that returns just the final response.
        
        Args:
            message (str): User message
            
        Returns:
            str: Final assistant response
        """
        result = self.run_conversation(message)
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
        model (str): Model name to use (OpenRouter format: provider/model). Defaults to anthropic/claude-sonnet-4-20250514.
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
