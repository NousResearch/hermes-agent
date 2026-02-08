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

import json
import logging
import os
import random
import sys
import time
import threading
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    if not os.getenv("HERMES_QUIET"):
        print(f"✅ Loaded environment variables from {env_path}")
elif not os.getenv("HERMES_QUIET"):
    print(f"ℹ️  No .env file found at {env_path}. Using system environment variables.")

# Import our tool system
from model_tools import get_tool_definitions, handle_function_call, check_toolset_requirements
from tools.terminal_tool import cleanup_vm
from tools.browser_tool import cleanup_browser

import requests

# =============================================================================
# Model Context Management
# =============================================================================

# Cache for model metadata from OpenRouter
_model_metadata_cache: Dict[str, Dict[str, Any]] = {}
_model_metadata_cache_time: float = 0
_MODEL_CACHE_TTL = 3600  # 1 hour cache TTL

# Default context lengths for common models (fallback if API fails)
DEFAULT_CONTEXT_LENGTHS = {
    "anthropic/claude-opus-4": 200000,
    "anthropic/claude-opus-4.5": 200000,
    "anthropic/claude-opus-4.6": 200000,
    "anthropic/claude-sonnet-4": 200000,
    "anthropic/claude-sonnet-4-20250514": 200000,
    "anthropic/claude-haiku-4.5": 200000,
    "openai/gpt-4o": 128000,
    "openai/gpt-4-turbo": 128000,
    "openai/gpt-4o-mini": 128000,
    "google/gemini-2.0-flash": 1048576,
    "google/gemini-2.5-pro": 1048576,
    "meta-llama/llama-3.3-70b-instruct": 131072,
    "deepseek/deepseek-chat-v3": 65536,
    "qwen/qwen-2.5-72b-instruct": 32768,
}


def fetch_model_metadata(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Fetch model metadata from OpenRouter's /api/v1/models endpoint.
    Results are cached for 1 hour to minimize API calls.
    
    Returns:
        Dict mapping model_id to metadata (context_length, max_completion_tokens, etc.)
    """
    global _model_metadata_cache, _model_metadata_cache_time
    
    # Return cached data if fresh
    if not force_refresh and _model_metadata_cache and (time.time() - _model_metadata_cache_time) < _MODEL_CACHE_TTL:
        return _model_metadata_cache
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Build cache mapping model_id to relevant metadata
        cache = {}
        for model in data.get("data", []):
            model_id = model.get("id", "")
            cache[model_id] = {
                "context_length": model.get("context_length", 128000),
                "max_completion_tokens": model.get("top_provider", {}).get("max_completion_tokens", 4096),
                "name": model.get("name", model_id),
                "pricing": model.get("pricing", {}),
            }
            # Also cache by canonical slug if different
            canonical = model.get("canonical_slug", "")
            if canonical and canonical != model_id:
                cache[canonical] = cache[model_id]
        
        _model_metadata_cache = cache
        _model_metadata_cache_time = time.time()
        
        if not os.getenv("HERMES_QUIET"):
            logging.debug(f"Fetched metadata for {len(cache)} models from OpenRouter")
        
        return cache
        
    except Exception as e:
        logging.warning(f"Failed to fetch model metadata from OpenRouter: {e}")
        # Return cached data even if stale, or empty dict
        return _model_metadata_cache or {}


def get_model_context_length(model: str) -> int:
    """
    Get the context length for a specific model.
    
    Args:
        model: Model identifier (e.g., "anthropic/claude-sonnet-4")
        
    Returns:
        Context length in tokens (defaults to 128000 if unknown)
    """
    # Try to get from OpenRouter API
    metadata = fetch_model_metadata()
    if model in metadata:
        return metadata[model].get("context_length", 128000)
    
    # Check default fallbacks (handles partial matches)
    for default_model, length in DEFAULT_CONTEXT_LENGTHS.items():
        if default_model in model or model in default_model:
            return length
    
    # Conservative default
    return 128000


def estimate_tokens_rough(text: str) -> int:
    """
    Rough token estimate for pre-flight checks (before API call).
    Uses ~4 chars per token heuristic.
    
    For accurate counts, use the `usage.prompt_tokens` from API responses.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Rough estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


def estimate_messages_tokens_rough(messages: List[Dict[str, Any]]) -> int:
    """
    Rough token estimate for messages (pre-flight check only).
    
    For accurate counts, use the `usage.prompt_tokens` from API responses.
    
    Args:
        messages: List of message dicts
        
    Returns:
        Rough estimated token count
    """
    total_chars = sum(len(str(msg)) for msg in messages)
    return total_chars // 4


class ContextCompressor:
    """
    Compresses conversation context when approaching model's context limit.
    
    Uses similar logic to trajectory_compressor but operates in real-time:
    1. Protects first few turns (system, initial user, first assistant response)
    2. Protects last N turns (recent context is most relevant)
    3. Summarizes middle turns when threshold is reached
    
    Token tracking uses actual counts from API responses (usage.prompt_tokens)
    rather than estimates for accuracy.
    """
    
    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.85,
        summary_model: str = "google/gemini-3-flash-preview",
        protect_first_n: int = 3,
        protect_last_n: int = 4,
        summary_target_tokens: int = 500,
        quiet_mode: bool = False,
    ):
        """
        Initialize the context compressor.
        
        Args:
            model: The main model being used (to determine context limit)
            threshold_percent: Trigger compression at this % of context (default 85%)
            summary_model: Model to use for generating summaries (cheap/fast)
            protect_first_n: Number of initial turns to always keep
            protect_last_n: Number of recent turns to always keep
            summary_target_tokens: Target token count for summaries
            quiet_mode: Suppress compression notifications
        """
        self.model = model
        self.threshold_percent = threshold_percent
        self.summary_model = summary_model
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_tokens = summary_target_tokens
        self.quiet_mode = quiet_mode
        
        self.context_length = get_model_context_length(model)
        self.threshold_tokens = int(self.context_length * threshold_percent)
        self.compression_count = 0
        
        # Track actual token usage from API responses
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        
        # Initialize OpenRouter client for summarization
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        ) if api_key else None
    
    def update_from_response(self, usage: Dict[str, Any]):
        """
        Update tracked token usage from API response.
        
        Args:
            usage: The usage dict from response (contains prompt_tokens, completion_tokens, total_tokens)
        """
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)
    
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """
        Check if context exceeds the compression threshold.
        
        Uses actual token count from API response for accuracy.
        
        Args:
            prompt_tokens: Actual prompt tokens from last API response.
                          If None, uses last tracked value.
            
        Returns:
            True if compression should be triggered
        """
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens
    
    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Quick pre-flight check using rough estimate (before API call).
        
        Use this to avoid making an API call that would fail due to context overflow.
        For post-response compression decisions, use should_compress() with actual tokens.
        
        Args:
            messages: Current conversation messages
            
        Returns:
            True if compression is likely needed
        """
        rough_estimate = estimate_messages_tokens_rough(messages)
        return rough_estimate >= self.threshold_tokens
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current compression status for display/logging.
        
        Returns:
            Dict with token usage and threshold info
        """
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (self.last_prompt_tokens / self.context_length * 100) if self.context_length else 0,
            "compression_count": self.compression_count,
        }
    
    def _generate_summary(self, turns_to_summarize: List[Dict[str, Any]]) -> str:
        """
        Generate a concise summary of conversation turns using a fast model.
        
        Args:
            turns_to_summarize: List of message dicts to summarize
            
        Returns:
            Summary string
        """
        if not self.client:
            # Fallback if no API key
            return "[CONTEXT SUMMARY]: Previous conversation turns have been compressed to save space. The assistant performed various actions and received responses."
        
        # Format turns for summarization
        parts = []
        for i, msg in enumerate(turns_to_summarize):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Truncate very long content
            if len(content) > 2000:
                content = content[:1000] + "\n...[truncated]...\n" + content[-500:]
            
            # Include tool call info if present
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls if isinstance(tc, dict)]
                content += f"\n[Tool calls: {', '.join(tool_names)}]"
            
            parts.append(f"[{role.upper()}]: {content}")
        
        content_to_summarize = "\n\n".join(parts)
        
        prompt = f"""Summarize these conversation turns concisely. This summary will replace these turns in the conversation history.

Write from a neutral perspective describing:
1. What actions were taken (tool calls, searches, file operations)
2. Key information or results obtained
3. Important decisions or findings
4. Relevant data, file names, or outputs

Keep factual and informative. Target ~{self.summary_target_tokens} tokens.

---
TURNS TO SUMMARIZE:
{content_to_summarize}
---

Write only the summary, starting with "[CONTEXT SUMMARY]:" prefix."""

        try:
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=self.summary_target_tokens * 2,
                timeout=30.0,
            )
            
            summary = response.choices[0].message.content.strip()
            if not summary.startswith("[CONTEXT SUMMARY]:"):
                summary = "[CONTEXT SUMMARY]: " + summary
            
            return summary
            
        except Exception as e:
            logging.warning(f"Failed to generate context summary: {e}")
            return "[CONTEXT SUMMARY]: Previous conversation turns have been compressed. The assistant performed tool calls and received responses."
    
    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Compress conversation messages by summarizing middle turns.
        
        Algorithm:
        1. Keep first N turns (system prompt, initial context)
        2. Keep last N turns (recent/relevant context)
        3. Summarize everything in between
        4. Insert summary as a user message
        
        Args:
            messages: Current conversation messages
            current_tokens: Actual token count from API (for logging). If None, uses estimate.
            
        Returns:
            Compressed message list
        """
        n_messages = len(messages)
        
        # Not enough messages to compress
        if n_messages <= self.protect_first_n + self.protect_last_n + 1:
            if not self.quiet_mode:
                print(f"⚠️  Cannot compress: only {n_messages} messages (need > {self.protect_first_n + self.protect_last_n + 1})")
            return messages
        
        # Determine compression boundaries
        compress_start = self.protect_first_n
        compress_end = n_messages - self.protect_last_n
        
        # Nothing to compress
        if compress_start >= compress_end:
            return messages
        
        # Extract turns to summarize
        turns_to_summarize = messages[compress_start:compress_end]
        
        # Use actual token count if provided, otherwise estimate
        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)
        
        if not self.quiet_mode:
            print(f"\n📦 Context compression triggered ({display_tokens:,} tokens ≥ {self.threshold_tokens:,} threshold)")
            print(f"   📊 Model context limit: {self.context_length:,} tokens ({self.threshold_percent*100:.0f}% = {self.threshold_tokens:,})")
            print(f"   🗜️  Summarizing turns {compress_start+1}-{compress_end} ({len(turns_to_summarize)} turns)")
        
        # Generate summary
        summary = self._generate_summary(turns_to_summarize)
        
        # Build compressed messages
        compressed = []
        
        # Keep protected head turns
        for i in range(compress_start):
            msg = messages[i].copy()
            # Add notice to system message on first compression
            if i == 0 and msg.get("role") == "system" and self.compression_count == 0:
                msg["content"] = msg.get("content", "") + "\n\n[Note: Some earlier conversation turns may be summarized to preserve context space.]"
            compressed.append(msg)
        
        # Add summary as user message
        compressed.append({
            "role": "user",
            "content": summary
        })
        
        # Keep protected tail turns
        for i in range(compress_end, n_messages):
            compressed.append(messages[i].copy())
        
        self.compression_count += 1
        
        if not self.quiet_mode:
            # Estimate new size (actual will be known after next API call)
            new_estimate = estimate_messages_tokens_rough(compressed)
            saved_estimate = display_tokens - new_estimate
            print(f"   ✅ Compressed: {n_messages} → {len(compressed)} messages (~{saved_estimate:,} tokens saved)")
            print(f"   💡 Compression #{self.compression_count} complete")
        
        return compressed


# =============================================================================
# Default System Prompt Components
# =============================================================================

# Skills guidance - instructs the model to check skills before technical tasks
SKILLS_SYSTEM_PROMPT = """## Skills
Before answering technical questions about tools, frameworks, or workflows:
1. Check skills_categories to see if a relevant category exists
2. If a category matches your task, use skills_list with that category
3. If a skill matches, load it with skill_view and follow its instructions

Skills contain vetted, up-to-date instructions for specific tools and workflows."""


class KawaiiSpinner:
    """
    Animated spinner with kawaii faces for CLI feedback during tool execution.
    Runs in a background thread and can be stopped when the operation completes.
    
    Uses stdout with carriage return to animate in place.
    """
    
    # Different spinner animation sets
    SPINNERS = {
        'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        'bounce': ['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈'],
        'grow': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂'],
        'arrows': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        'star': ['✶', '✷', '✸', '✹', '✺', '✹', '✸', '✷'],
        'moon': ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'],
        'pulse': ['◜', '◠', '◝', '◞', '◡', '◟'],
        'brain': ['🧠', '💭', '💡', '✨', '💫', '🌟', '💡', '💭'],
        'sparkle': ['⁺', '˚', '*', '✧', '✦', '✧', '*', '˚'],
    }
    
    # General waiting faces
    KAWAII_WAITING = [
        "(｡◕‿◕｡)", "(◕‿◕✿)", "٩(◕‿◕｡)۶", "(✿◠‿◠)", "( ˘▽˘)っ",
        "♪(´ε` )", "(◕ᴗ◕✿)", "ヾ(＾∇＾)", "(≧◡≦)", "(★ω★)",
    ]
    
    # Thinking-specific faces and messages
    KAWAII_THINKING = [
        "(｡•́︿•̀｡)", "(◔_◔)", "(¬‿¬)", "( •_•)>⌐■-■", "(⌐■_■)",
        "(´･_･`)", "◉_◉", "(°ロ°)", "( ˘⌣˘)♡", "ヽ(>∀<☆)☆",
        "٩(๑❛ᴗ❛๑)۶", "(⊙_⊙)", "(¬_¬)", "( ͡° ͜ʖ ͡°)", "ಠ_ಠ",
    ]
    
    THINKING_VERBS = [
        "pondering", "contemplating", "musing", "cogitating", "ruminating",
        "deliberating", "mulling", "reflecting", "processing", "reasoning",
        "analyzing", "computing", "synthesizing", "formulating", "brainstorming",
    ]
    
    def __init__(self, message: str = "", spinner_type: str = 'dots'):
        self.message = message
        self.spinner_frames = self.SPINNERS.get(spinner_type, self.SPINNERS['dots'])
        self.running = False
        self.thread = None
        self.frame_idx = 0
        self.start_time = None
        self.last_line_len = 0
        
    def _animate(self):
        """Animation loop that runs in background thread."""
        while self.running:
            # Check for pause signal (e.g., during sudo password prompt)
            if os.getenv("HERMES_SPINNER_PAUSE"):
                time.sleep(0.1)
                continue
            
            frame = self.spinner_frames[self.frame_idx % len(self.spinner_frames)]
            elapsed = time.time() - self.start_time
            
            # Build the spinner line
            line = f"  {frame} {self.message} ({elapsed:.1f}s)"
            
            # Clear previous line and write new one
            clear = '\r' + ' ' * self.last_line_len + '\r'
            print(clear + line, end='', flush=True)
            self.last_line_len = len(line)
            
            self.frame_idx += 1
            time.sleep(0.12)  # ~8 FPS animation
    
    def start(self):
        """Start the spinner animation."""
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self, final_message: str = None):
        """Stop the spinner and optionally print a final message."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        
        # Clear the spinner line
        print('\r' + ' ' * (self.last_line_len + 5) + '\r', end='', flush=True)
        
        # Print final message if provided
        if final_message:
            print(f"  {final_message}", flush=True)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


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
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
    ):
        """
        Initialize the AI Agent.

        Args:
            base_url (str): Base URL for the model API (optional)
            api_key (str): API key for authentication (optional, uses env var if not provided)
            model (str): Model name to use (default: "gpt-4")
            max_iterations (int): Maximum number of tool calling iterations (default: 10)
            tool_delay (float): Delay between tool calls in seconds (default: 1.0)
            enabled_toolsets (List[str]): Only enable tools from these toolsets (optional)
            disabled_toolsets (List[str]): Disable tools from these toolsets (optional)
            save_trajectories (bool): Whether to save conversation trajectories to JSONL files (default: False)
            verbose_logging (bool): Enable verbose logging for debugging (default: False)
            quiet_mode (bool): Suppress progress output for clean CLI experience (default: False)
            ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
            log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 20)
            log_prefix (str): Prefix to add to all log messages for identification in parallel processing (default: "")
            providers_allowed (List[str]): OpenRouter providers to allow (optional)
            providers_ignored (List[str]): OpenRouter providers to ignore (optional)
            providers_order (List[str]): OpenRouter providers to try in order (optional)
            provider_sort (str): Sort providers by price/throughput/latency (optional)
            session_id (str): Pre-generated session ID for logging (optional, auto-generated if not provided)
            tool_progress_callback (callable): Callback function(tool_name, args_preview) for progress notifications
            max_tokens (int): Maximum tokens for model responses (optional, uses model default if not set)
            reasoning_config (Dict): OpenRouter reasoning configuration override (e.g. {"effort": "none"} to disable thinking).
                If None, defaults to {"enabled": True, "effort": "xhigh"} for OpenRouter. Set to disable/customize reasoning.
            prefill_messages (List[Dict]): Messages to prepend to conversation history as prefilled context.
                Useful for injecting a few-shot example or priming the model's response style.
                Example: [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
        """
        self.model = model
        self.max_iterations = max_iterations
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        self.base_url = base_url or ""  # Store for OpenRouter detection
        self.tool_progress_callback = tool_progress_callback
        self._last_reported_tool = None  # Track for "new tool" mode
        
        # Interrupt mechanism for breaking out of tool loops
        self._interrupt_requested = False
        self._interrupt_message = None  # Optional message that triggered interrupt
        
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
            if not self.quiet_mode:
                print("🔍 Verbose logging enabled (third-party library logs suppressed)")
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
        
        # Initialize OpenAI client - defaults to OpenRouter
        client_kwargs = {}
        
        # Default to OpenRouter if no base_url provided
        if base_url:
            client_kwargs["base_url"] = base_url
        else:
            client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
        
        # Handle API key - OpenRouter is the primary provider
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            # Primary: OPENROUTER_API_KEY, fallback to direct provider keys
            client_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY", "")
        
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
        
        # Setup logs directory
        self.logs_dir = Path(__file__).parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
        
        # Track conversation messages for session logging
        self._session_messages: List[Dict[str, Any]] = []
        
        # Initialize context compressor for automatic context management
        # Compresses conversation when approaching model's context limit
        # Configuration via environment variables (can be set in .env or cli-config.yaml)
        compression_threshold = float(os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "0.85"))
        compression_model = os.getenv("CONTEXT_COMPRESSION_MODEL", "google/gemini-3-flash-preview")
        compression_enabled = os.getenv("CONTEXT_COMPRESSION_ENABLED", "true").lower() in ("true", "1", "yes")
        
        self.context_compressor = ContextCompressor(
            model=self.model,
            threshold_percent=compression_threshold,
            summary_model=compression_model,
            protect_first_n=3,  # Keep system, first user, first assistant
            protect_last_n=4,   # Keep recent context
            summary_target_tokens=500,
            quiet_mode=self.quiet_mode,
        )
        self.compression_enabled = compression_enabled
        
        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")
    
    # Pools of kawaii faces for random selection
    KAWAII_SEARCH = [
        "♪(´ε` )", "(｡◕‿◕｡)", "ヾ(＾∇＾)", "(◕ᴗ◕✿)", "( ˘▽˘)っ",
        "٩(◕‿◕｡)۶", "(✿◠‿◠)", "♪～(´ε｀ )", "(ノ´ヮ`)ノ*:・゚✧", "＼(◎o◎)／",
    ]
    KAWAII_READ = [
        "φ(゜▽゜*)♪", "( ˘▽˘)っ", "(⌐■_■)", "٩(｡•́‿•̀｡)۶", "(◕‿◕✿)",
        "ヾ(＠⌒ー⌒＠)ノ", "(✧ω✧)", "♪(๑ᴖ◡ᴖ๑)♪", "(≧◡≦)", "( ´ ▽ ` )ノ",
    ]
    KAWAII_TERMINAL = [
        "ヽ(>∀<☆)ノ", "(ノ°∀°)ノ", "٩(^ᴗ^)۶", "ヾ(⌐■_■)ノ♪", "(•̀ᴗ•́)و",
        "┗(＾0＾)┓", "(｀・ω・´)", "＼(￣▽￣)／", "(ง •̀_•́)ง", "ヽ(´▽`)/",
    ]
    KAWAII_BROWSER = [
        "(ノ°∀°)ノ", "(☞゚ヮ゚)☞", "( ͡° ͜ʖ ͡°)", "┌( ಠ_ಠ)┘", "(⊙_⊙)？",
        "ヾ(•ω•`)o", "(￣ω￣)", "( ˇωˇ )", "(ᵔᴥᵔ)", "＼(◎o◎)／",
    ]
    KAWAII_CREATE = [
        "✧*。٩(ˊᗜˋ*)و✧", "(ﾉ◕ヮ◕)ﾉ*:・ﾟ✧", "ヽ(>∀<☆)ノ", "٩(♡ε♡)۶", "(◕‿◕)♡",
        "✿◕ ‿ ◕✿", "(*≧▽≦)", "ヾ(＾-＾)ノ", "(☆▽☆)", "°˖✧◝(⁰▿⁰)◜✧˖°",
    ]
    KAWAII_SKILL = [
        "ヾ(＠⌒ー⌒＠)ノ", "(๑˃ᴗ˂)ﻭ", "٩(◕‿◕｡)۶", "(✿╹◡╹)", "ヽ(・∀・)ノ",
        "(ノ´ヮ`)ノ*:・ﾟ✧", "♪(๑ᴖ◡ᴖ๑)♪", "(◠‿◠)", "٩(ˊᗜˋ*)و", "(＾▽＾)",
        "ヾ(＾∇＾)", "(★ω★)/", "٩(｡•́‿•̀｡)۶", "(◕ᴗ◕✿)", "＼(◎o◎)／",
        "(✧ω✧)", "ヽ(>∀<☆)ノ", "( ˘▽˘)っ", "(≧◡≦) ♡", "ヾ(￣▽￣)",
    ]
    KAWAII_THINK = [
        "(っ°Д°;)っ", "(；′⌒`)", "(・_・ヾ", "( ´_ゝ`)", "(￣ヘ￣)",
        "(。-`ω´-)", "( ˘︹˘ )", "(¬_¬)", "ヽ(ー_ー )ノ", "(；一_一)",
    ]
    KAWAII_GENERIC = [
        "♪(´ε` )", "(◕‿◕✿)", "ヾ(＾∇＾)", "٩(◕‿◕｡)۶", "(✿◠‿◠)",
        "(ノ´ヮ`)ノ*:・ﾟ✧", "ヽ(>∀<☆)ノ", "(☆▽☆)", "( ˘▽˘)っ", "(≧◡≦)",
    ]
    
    def _get_cute_tool_message(self, tool_name: str, args: dict, duration: float) -> str:
        """
        Generate a kawaii ASCII/unicode art message for tool execution in CLI mode.
        
        Args:
            tool_name: Name of the tool being called
            args: Arguments passed to the tool
            duration: How long the tool took to execute
        
        Returns:
            A cute ASCII art message about what the tool did
        """
        time_str = f"⏱ {duration:.1f}s"
        
        # Web tools - show what we're searching/reading
        if tool_name == "web_search":
            query = args.get("query", "the web")
            if len(query) > 40:
                query = query[:37] + "..."
            face = random.choice(self.KAWAII_SEARCH)
            return f"{face} 🔍 Searching for '{query}'... {time_str}"
        
        elif tool_name == "web_extract":
            urls = args.get("urls", [])
            face = random.choice(self.KAWAII_READ)
            if urls:
                url = urls[0] if isinstance(urls, list) else str(urls)
                domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                if len(domain) > 25:
                    domain = domain[:22] + "..."
                if len(urls) > 1:
                    return f"{face} 📖 Reading {domain} +{len(urls)-1} more... {time_str}"
                return f"{face} 📖 Reading {domain}... {time_str}"
            return f"{face} 📖 Reading pages... {time_str}"
        
        elif tool_name == "web_crawl":
            url = args.get("url", "website")
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            if len(domain) > 25:
                domain = domain[:22] + "..."
            face = random.choice(self.KAWAII_READ)
            return f"{face} 🕸️ Crawling {domain}... {time_str}"
        
        # Terminal tool
        elif tool_name == "terminal":
            command = args.get("command", "")
            if len(command) > 30:
                command = command[:27] + "..."
            face = random.choice(self.KAWAII_TERMINAL)
            return f"{face} 💻 $ {command} {time_str}"
        
        # Browser tools
        elif tool_name == "browser_navigate":
            url = args.get("url", "page")
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            if len(domain) > 25:
                domain = domain[:22] + "..."
            face = random.choice(self.KAWAII_BROWSER)
            return f"{face} 🌐 → {domain} {time_str}"
        
        elif tool_name == "browser_snapshot":
            face = random.choice(self.KAWAII_BROWSER)
            return f"{face} 📸 *snap* {time_str}"
        
        elif tool_name == "browser_click":
            element = args.get("ref", "element")
            face = random.choice(self.KAWAII_BROWSER)
            return f"{face} 👆 *click* {element} {time_str}"
        
        elif tool_name == "browser_type":
            text = args.get("text", "")
            if len(text) > 15:
                text = text[:12] + "..."
            face = random.choice(self.KAWAII_BROWSER)
            return f"{face} ⌨️ typing '{text}' {time_str}"
        
        elif tool_name == "browser_scroll":
            direction = args.get("direction", "down")
            arrow = "↓" if direction == "down" else "↑"
            face = random.choice(self.KAWAII_BROWSER)
            return f"{face} {arrow} scrolling {direction}... {time_str}"
        
        elif tool_name == "browser_back":
            face = random.choice(self.KAWAII_BROWSER)
            return f"{face} ← going back... {time_str}"
        
        elif tool_name == "browser_vision":
            face = random.choice(self.KAWAII_BROWSER)
            return f"{face} 👁️ analyzing visually... {time_str}"
        
        # Image generation
        elif tool_name == "image_generate":
            prompt = args.get("prompt", "image")
            if len(prompt) > 20:
                prompt = prompt[:17] + "..."
            face = random.choice(self.KAWAII_CREATE)
            return f"{face} 🎨 creating '{prompt}'... {time_str}"
        
        # Skills - use large pool for variety
        elif tool_name == "skills_categories":
            face = random.choice(self.KAWAII_SKILL)
            return f"{face} 📚 listing categories... {time_str}"
        
        elif tool_name == "skills_list":
            category = args.get("category", "skills")
            face = random.choice(self.KAWAII_SKILL)
            return f"{face} 📋 listing {category} skills... {time_str}"
        
        elif tool_name == "skill_view":
            name = args.get("name", "skill")
            face = random.choice(self.KAWAII_SKILL)
            return f"{face} 📖 loading {name}... {time_str}"
        
        # Vision tools
        elif tool_name == "vision_analyze":
            face = random.choice(self.KAWAII_BROWSER)
            return f"{face} 👁️✨ analyzing image... {time_str}"
        
        # Mixture of agents
        elif tool_name == "mixture_of_agents":
            face = random.choice(self.KAWAII_THINK)
            return f"{face} 🧠💭 thinking REALLY hard... {time_str}"
        
        # Default fallback - random generic kawaii
        else:
            face = random.choice(self.KAWAII_GENERIC)
            return f"{face} ⚡ {tool_name}... {time_str}"
    
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
        
        import re
        # Remove all <think>...</think> blocks (including nested ones, non-greedy)
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())
    
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
    def _convert_scratchpad_to_think(content: str) -> str:
        """
        Convert <REASONING_SCRATCHPAD> tags to <think> tags in content.
        
        When native thinking/reasoning is disabled and the model is prompted to
        reason inside <REASONING_SCRATCHPAD> XML tags instead, this converts those
        to the standard <think> format used in our trajectory storage.
        
        Args:
            content: Assistant message content that may contain scratchpad tags
            
        Returns:
            Content with scratchpad tags replaced by think tags
        """
        if not content or "<REASONING_SCRATCHPAD>" not in content:
            return content
        return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")
    
    @staticmethod
    def _has_incomplete_scratchpad(content: str) -> bool:
        """
        Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag.
        
        This indicates the model ran out of output tokens mid-reasoning, producing
        a broken turn that shouldn't be saved. The caller should retry or discard.
        
        Args:
            content: Assistant message content to check
            
        Returns:
            True if there's an unclosed scratchpad tag
        """
        if not content:
            return False
        return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content
    
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
        
        # Calculate where agent responses start in the messages list.
        # Prefill messages are ephemeral (only used to prime model response style)
        # so we skip them entirely in the saved trajectory.
        # Layout: [*prefill_msgs, actual_user_msg, ...agent_responses...]
        num_prefill = len(self.prefill_messages) if self.prefill_messages else 0
        i = num_prefill + 1  # Skip prefill messages + the actual user message (already added above)
        
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
                        content += self._convert_scratchpad_to_think(msg["content"]) + "\n"
                    
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
                        
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": msg["tool_calls"][len(tool_responses)]["function"]["name"] if len(tool_responses) < len(msg["tool_calls"]) else "unknown",
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
                    content += self._convert_scratchpad_to_think(raw_content)
                    
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
        
        # Convert messages to trajectory format
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        
        # Determine which file to save to
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
        
        # Create trajectory entry
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "completed": completed
        }
        
        # Append to JSONL file
        try:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"💾 Trajectory saved to {filename}")
        except Exception as e:
            print(f"⚠️ Failed to save trajectory: {e}")
    
    def _log_api_payload(self, turn_number: int, api_kwargs: Dict[str, Any], response=None):
        """
        [TEMPORARY DEBUG] Log the full API payload and response token metrics
        for each agent turn to a per-session JSONL file for inspection.
        
        Writes one JSON line per turn to logs/payload_<session_id>.jsonl.
        Tool schemas are summarized (just names) to keep logs readable.
        
        Args:
            turn_number: Which API call this is (1-indexed)
            api_kwargs: The full kwargs dict being passed to chat.completions.create
            response: The API response object (optional, added after the call completes)
        """
        try:
            payload_log_file = self.logs_dir / f"payload_{self.session_id}.jsonl"
            
            # Build a serializable copy of the request payload
            payload = {
                "turn": turn_number,
                "timestamp": datetime.now().isoformat(),
                "model": api_kwargs.get("model"),
                "max_tokens": api_kwargs.get("max_tokens"),
                "extra_body": api_kwargs.get("extra_body"),
                "num_tools": len(api_kwargs.get("tools") or []),
                "tool_names": [t["function"]["name"] for t in (api_kwargs.get("tools") or [])],
                "messages": api_kwargs.get("messages", []),
            }
            
            # Add response token metrics if available
            if response is not None:
                try:
                    usage_raw = response.usage.model_dump() if hasattr(response.usage, 'model_dump') else {}
                    payload["response"] = {
                        # Core token counts
                        "prompt_tokens": usage_raw.get("prompt_tokens"),
                        "completion_tokens": usage_raw.get("completion_tokens"),
                        "total_tokens": usage_raw.get("total_tokens"),
                        # Completion breakdown (reasoning tokens, etc.)
                        "completion_tokens_details": usage_raw.get("completion_tokens_details"),
                        # Prompt breakdown (cached tokens, etc.)
                        "prompt_tokens_details": usage_raw.get("prompt_tokens_details"),
                        # Cost tracking
                        "cost": usage_raw.get("cost"),
                        "is_byok": usage_raw.get("is_byok"),
                        "cost_details": usage_raw.get("cost_details"),
                        # Provider info (top-level field from OpenRouter)
                        "provider": getattr(response, 'provider', None),
                        "response_model": getattr(response, 'model', None),
                    }
                except Exception:
                    payload["response"] = {"error": "failed to extract usage"}
            
            with open(payload_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                
        except Exception as e:
            # Silent fail - don't interrupt the agent for debug logging
            if self.verbose_logging:
                logging.warning(f"Failed to log API payload: {e}")
    
    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        Save the current session trajectory to the logs directory.
        
        Automatically called after each conversation turn to maintain
        a complete log of the session for debugging and inspection.
        
        Args:
            messages: Message history to save (uses self._session_messages if not provided)
        """
        messages = messages or self._session_messages
        if not messages:
            return
        
        try:
            # Extract the actual user query for the trajectory format.
            # Skip prefill messages (they're ephemeral and shouldn't appear in trajectories)
            # so the first user message we find is the real task prompt.
            first_user_query = ""
            start_idx = len(self.prefill_messages) if self.prefill_messages else 0
            for msg in messages[start_idx:]:
                if msg.get("role") == "user":
                    first_user_query = msg.get("content", "")
                    break
            
            # Convert to trajectory format
            trajectory = self._convert_to_trajectory_format(messages, first_user_query, True)
            
            # Build the session log entry
            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "message_count": len(messages),
                "conversations": trajectory,
            }
            
            # Write to session log file (overwrite with latest state)
            with open(self.session_log_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            # Silent fail - don't interrupt the user experience for logging issues
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
    
    def interrupt(self, message: str = None) -> None:
        """
        Request the agent to interrupt its current tool-calling loop.
        
        Call this from another thread (e.g., input handler, message receiver)
        to gracefully stop the agent and process a new message.
        
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
        if not self.quiet_mode:
            print(f"\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
    
    def clear_interrupt(self) -> None:
        """Clear any pending interrupt request."""
        self._interrupt_requested = False
        self._interrupt_message = None
    
    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested
    
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
        import uuid
        effective_task_id = task_id or str(uuid.uuid4())
        
        # Reset retry counters at the start of each conversation to prevent state leakage
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        
        # Initialize conversation
        messages = conversation_history or []
        
        # Inject prefill messages at the start of conversation (before user's actual prompt)
        # This is used for few-shot priming, e.g., a greeting exchange to set response style
        if self.prefill_messages and not conversation_history:
            for prefill_msg in self.prefill_messages:
                messages.append(prefill_msg.copy())
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        if not self.quiet_mode:
            print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
        # Determine which system prompt to use for API calls (ephemeral)
        # Priority: explicit system_message > ephemeral_system_prompt > None
        base_system_prompt = system_message if system_message is not None else self.ephemeral_system_prompt
        
        # Auto-include skills guidance if skills tools are available
        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skills_categories', 'skill_view'])
        if has_skills_tools:
            if base_system_prompt:
                active_system_prompt = f"{base_system_prompt}\n\n{SKILLS_SYSTEM_PROMPT}"
            else:
                active_system_prompt = SKILLS_SYSTEM_PROMPT
        else:
            active_system_prompt = base_system_prompt
        
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
            
            if active_system_prompt:
                # Insert system message at the beginning
                api_messages = [{"role": "system", "content": active_system_prompt}] + api_messages
            
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
                    # Build OpenRouter provider preferences if specified
                    provider_preferences = {}
                    if self.providers_allowed:
                        provider_preferences["only"] = self.providers_allowed
                    if self.providers_ignored:
                        provider_preferences["ignore"] = self.providers_ignored
                    if self.providers_order:
                        provider_preferences["order"] = self.providers_order
                    if self.provider_sort:
                        provider_preferences["sort"] = self.provider_sort
                    
                    # Make API call with tools - increased timeout for long responses
                    api_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                        "tools": self.tools if self.tools else None,
                        "timeout": 600.0  # 10 minute timeout for very long responses
                    }
                    
                    # Add max_tokens if configured (overrides model default)
                    if self.max_tokens is not None:
                        api_kwargs["max_tokens"] = self.max_tokens
                    
                    # Add extra_body for OpenRouter (provider preferences + reasoning)
                    extra_body = {}
                    
                    # Add provider preferences if specified
                    if provider_preferences:
                        extra_body["provider"] = provider_preferences
                    
                    # Configure reasoning for OpenRouter
                    # If reasoning_config is explicitly provided, use it (allows disabling/customizing)
                    # Otherwise, default to xhigh effort for OpenRouter models
                    if "openrouter" in self.base_url.lower():
                        if self.reasoning_config is not None:
                            extra_body["reasoning"] = self.reasoning_config
                        else:
                            extra_body["reasoning"] = {
                                "enabled": True,
                                "effort": "xhigh"
                            }
                    
                    if extra_body:
                        api_kwargs["extra_body"] = extra_body
                    
                    response = self.client.chat.completions.create(**api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    
                    # Stop thinking spinner with cute completion message
                    if thinking_spinner:
                        face = random.choice(["(◕‿◕✿)", "ヾ(＾∇＾)", "(≧◡≦)", "✧٩(ˊᗜˋ*)و✧", "(*^▽^*)"])
                        thinking_spinner.stop(f"{face} got it! ({api_duration:.1f}s)")
                        thinking_spinner = None
                    
                    if not self.quiet_mode:
                        print(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                    
                    if self.verbose_logging:
                        # Log response with provider info if available
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    # [DEBUG] Log the full API payload + response token metrics
                    self._log_api_payload(api_call_count, api_kwargs, response=response)

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
                            
                            # Clean up VM and browser
                            try:
                                cleanup_vm(effective_task_id)
                            except Exception as e:
                                if self.verbose_logging:
                                    logging.warning(f"Failed to cleanup VM for task {effective_task_id}: {e}")
                            try:
                                cleanup_browser(effective_task_id)
                            except Exception as e:
                                if self.verbose_logging:
                                    logging.warning(f"Failed to cleanup browser for task {effective_task_id}: {e}")
                            
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
                    
                    break  # Success, exit retry loop

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
                    is_client_error = any(phrase in error_msg for phrase in [
                        'error code: 400', 'error code: 401', 'error code: 403',
                        'error code: 404', 'error code: 422',
                        'is not a valid model', 'invalid model', 'model not found',
                        'invalid api key', 'invalid_api_key', 'authentication',
                        'unauthorized', 'forbidden', 'not found',
                    ])
                    
                    if is_client_error:
                        print(f"{self.log_prefix}❌ Non-retryable client error detected. Aborting immediately.")
                        print(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.")
                        logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")
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
                        
                        # Try to compress and retry
                        original_len = len(messages)
                        messages = self.context_compressor.compress(messages, current_tokens=approx_tokens)
                        
                        if len(messages) < original_len:
                            # Compression was possible, retry
                            print(f"{self.log_prefix}   🗜️  Compressed {original_len} → {len(messages)} messages, retrying...")
                            continue  # Retry with compressed messages
                        else:
                            # Can't compress further
                            print(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.")
                            print(f"{self.log_prefix}   💡 The conversation has accumulated too much content.")
                            logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
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
                            return {
                                "final_response": "Operation interrupted.",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)  # Check interrupt every 200ms
            
            try:
                assistant_message = response.choices[0].message
                
                # Handle assistant response
                if assistant_message.content and not self.quiet_mode:
                    print(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")
                
                # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
                # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
                if self._has_incomplete_scratchpad(assistant_message.content or ""):
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
                        
                        try:
                            cleanup_vm(effective_task_id)
                        except Exception:
                            pass
                        try:
                            cleanup_browser(effective_task_id)
                        except Exception:
                            pass
                        
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
                            self._invalid_tool_retries = 0  # Reset for next conversation
                            return {
                                "final_response": None,
                                "messages": messages,  # Messages up to last valid point
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
                            messages.append({"role": "user", "content": recovery_msg})
                            # Continue the loop - model will see this message and can recover
                            continue
                    
                    # Reset retry counter on successful JSON validation
                    self._invalid_json_retries = 0
                    
                    # Extract reasoning from response if available
                    # OpenRouter can return reasoning in multiple formats:
                    # 1. message.reasoning - direct reasoning field
                    # 2. message.reasoning_content - alternative field (some providers)
                    # 3. message.reasoning_details - array with {summary: "..."} objects
                    reasoning_text = self._extract_reasoning(assistant_message)
                    
                    if reasoning_text and self.verbose_logging:
                        preview = reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text
                        logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {preview}")
                    
                    # Build assistant message with tool calls
                    # Content stays as-is; reasoning is stored separately and will be passed
                    # to the API via reasoning_content field when preparing api_messages
                    assistant_msg = {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "reasoning": reasoning_text,  # Stored for trajectory extraction & API calls
                        "tool_calls": [
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
                    }
                    
                    # Store reasoning_details for multi-turn reasoning context (OpenRouter)
                    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
                        assistant_msg["reasoning_details"] = [
                            {"type": d.get("type"), "text": d.get("text"), "signature": d.get("signature")}
                            for d in assistant_message.reasoning_details
                            if isinstance(d, dict)
                        ]
                    
                    messages.append(assistant_msg)
                    
                    # Execute each tool call
                    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
                        function_name = tool_call.function.name
                        
                        # Parse arguments - should always succeed since we validated above
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            # This shouldn't happen since we validate and retry above
                            logging.warning(f"Unexpected JSON error after validation: {e}")
                            function_args = {}
                        
                        # Preview tool call - cleaner format for quiet mode
                        if not self.quiet_mode:
                            args_str = json.dumps(function_args, ensure_ascii=False)
                            args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                            print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")
                        
                        # Fire progress callback if registered (for messaging platforms)
                        if self.tool_progress_callback:
                            try:
                                # Build preview for terminal commands
                                if function_name == "terminal":
                                    cmd = function_args.get("command", "")
                                    preview = cmd[:50] + "..." if len(cmd) > 50 else cmd
                                else:
                                    preview = None
                                self.tool_progress_callback(function_name, preview)
                            except Exception as cb_err:
                                logging.debug(f"Tool progress callback error: {cb_err}")

                        tool_start_time = time.time()

                        # Execute the tool - with animated spinner in quiet mode
                        if self.quiet_mode:
                            # Tool-specific spinner animations
                            tool_spinners = {
                                'web_search': ('arrows', ['🔍', '🌐', '📡', '🔎']),
                                'web_extract': ('grow', ['📄', '📖', '📑', '🗒️']),
                                'web_crawl': ('arrows', ['🕷️', '🕸️', '🔗', '🌐']),
                                'terminal': ('dots', ['💻', '⌨️', '🖥️', '📟']),
                                'browser_navigate': ('moon', ['🌐', '🧭', '🔗', '🚀']),
                                'browser_click': ('bounce', ['👆', '🖱️', '👇', '✨']),
                                'browser_type': ('dots', ['⌨️', '✍️', '📝', '💬']),
                                'browser_screenshot': ('star', ['📸', '🖼️', '📷', '✨']),
                                'image_generate': ('sparkle', ['🎨', '✨', '🖼️', '🌟']),
                                'skill_view': ('star', ['📚', '📖', '🎓', '✨']),
                                'skills_list': ('pulse', ['📋', '📝', '📑', '📜']),
                                'skills_categories': ('pulse', ['📂', '🗂️', '📁', '🏷️']),
                                'moa_query': ('brain', ['🧠', '💭', '🤔', '💡']),
                                'analyze_image': ('sparkle', ['👁️', '🔍', '📷', '✨']),
                            }
                            
                            spinner_type, tool_emojis = tool_spinners.get(function_name, ('dots', ['⚙️', '🔧', '⚡', '✨']))
                            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                            tool_emoji = random.choice(tool_emojis)
                            spinner = KawaiiSpinner(f"{face} {tool_emoji} {function_name}...", spinner_type=spinner_type)
                            spinner.start()
                            try:
                                function_result = handle_function_call(function_name, function_args, effective_task_id)
                            finally:
                                tool_duration = time.time() - tool_start_time
                                cute_msg = self._get_cute_tool_message(function_name, function_args, tool_duration)
                                spinner.stop(cute_msg)
                        else:
                            function_result = handle_function_call(function_name, function_args, effective_task_id)
                            tool_duration = time.time() - tool_start_time

                        result_preview = function_result[:200] if len(function_result) > 200 else function_result

                        if self.verbose_logging:
                            logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                            logging.debug(f"Tool result preview: {result_preview}...")

                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "content": function_result,
                            "tool_call_id": tool_call.id
                        })

                        # Preview tool response (only in non-quiet mode)
                        if not self.quiet_mode:
                            response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                            print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")
                        
                        # Delay between tool calls
                        if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                            time.sleep(self.tool_delay)
                    
                    # Check if context compression is needed before next API call
                    # Uses actual token count from last API response
                    if self.compression_enabled and self.context_compressor.should_compress():
                        messages = self.context_compressor.compress(
                            messages, 
                            current_tokens=self.context_compressor.last_prompt_tokens
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
                        # Track retries for empty-after-think responses
                        if not hasattr(self, '_empty_content_retries'):
                            self._empty_content_retries = 0
                        self._empty_content_retries += 1
                        
                        content_preview = final_response[:80] + "..." if len(final_response) > 80 else final_response
                        print(f"{self.log_prefix}⚠️  Response only contains think block with no content after it")
                        print(f"{self.log_prefix}   Content: '{content_preview}'")
                        
                        if self._empty_content_retries < 3:
                            print(f"{self.log_prefix}🔄 Retrying API call ({self._empty_content_retries}/3)...")
                            # Don't add the incomplete message, just retry
                            continue
                        else:
                            # Max retries exceeded - roll back to last complete assistant turn
                            print(f"{self.log_prefix}❌ Max retries (3) for empty content exceeded. Rolling back to last complete turn.")
                            self._empty_content_retries = 0  # Reset for next conversation
                            
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                            
                            # Clean up VM and browser
                            try:
                                cleanup_vm(effective_task_id)
                            except Exception as e:
                                if self.verbose_logging:
                                    logging.warning(f"Failed to cleanup VM for task {effective_task_id}: {e}")
                            try:
                                cleanup_browser(effective_task_id)
                            except Exception as e:
                                if self.verbose_logging:
                                    logging.warning(f"Failed to cleanup browser for task {effective_task_id}: {e}")
                            
                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Model generated only think blocks with no actual response after 3 retries"
                            }
                    
                    # Reset retry counter on successful content
                    if hasattr(self, '_empty_content_retries'):
                        self._empty_content_retries = 0
                    
                    # Extract reasoning from response if available
                    reasoning_text = self._extract_reasoning(assistant_message)
                    
                    if reasoning_text and self.verbose_logging:
                        preview = reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text
                        logging.debug(f"Captured final reasoning ({len(reasoning_text)} chars): {preview}")
                    
                    # Build final assistant message
                    # Content stays as-is; reasoning stored separately for trajectory extraction
                    final_msg = {
                        "role": "assistant", 
                        "content": final_response,
                        "reasoning": reasoning_text  # Stored for trajectory extraction
                    }
                    
                    # Store reasoning_details for multi-turn reasoning context (OpenRouter)
                    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
                        final_msg["reasoning_details"] = [
                            {"type": d.get("type"), "text": d.get("text"), "signature": d.get("signature")}
                            for d in assistant_message.reasoning_details
                            if isinstance(d, dict)
                        ]
                    
                    messages.append(final_msg)
                    
                    if not self.quiet_mode:
                        print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                print(f"❌ {error_msg}")
                
                if self.verbose_logging:
                    logging.exception("Detailed error information:")
                
                # Add error to conversation and try to continue
                messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {error_msg}. Let me try a different approach."
                })
                
                # If we're near the limit, break to avoid infinite loops
                if api_call_count >= self.max_iterations - 1:
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    break
        
        # Handle max iterations reached - ask model to summarize what it found
        if api_call_count >= self.max_iterations and final_response is None:
            print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")
            
            # Inject a user message asking for a summary
            summary_request = (
                "You've reached the maximum number of tool-calling iterations allowed. "
                "Please provide a final response summarizing what you've found and accomplished so far, "
                "without calling any more tools."
            )
            messages.append({"role": "user", "content": summary_request})
            
            # Make one final API call WITHOUT tools to force a text response
            try:
                api_messages = messages.copy()
                if self.ephemeral_system_prompt:
                    api_messages = [{"role": "system", "content": self.ephemeral_system_prompt}] + api_messages
                
                # Build extra_body for summary call (same reasoning config as main loop)
                summary_extra_body = {}
                if "openrouter" in self.base_url.lower():
                    if self.reasoning_config is not None:
                        summary_extra_body["reasoning"] = self.reasoning_config
                    else:
                        summary_extra_body["reasoning"] = {
                            "enabled": True,
                            "effort": "xhigh"
                        }
                
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    # No tools parameter - forces text response
                }
                if self.max_tokens is not None:
                    summary_kwargs["max_tokens"] = self.max_tokens
                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body
                
                summary_response = self.client.chat.completions.create(**summary_kwargs)
                
                if summary_response.choices and summary_response.choices[0].message.content:
                    final_response = summary_response.choices[0].message.content
                    # Strip think blocks from final response
                    if "<think>" in final_response:
                        import re
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    
                    # Add to messages for session continuity
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
                    
            except Exception as e:
                logging.warning(f"Failed to get summary response: {e}")
                final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"
        
        # Determine if conversation completed successfully
        completed = final_response is not None and api_call_count < self.max_iterations

        # Save trajectory if enabled
        self._save_trajectory(messages, user_message, completed)

        # Clean up VM and browser for this task after conversation completes
        try:
            cleanup_vm(effective_task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {effective_task_id}: {e}")
        
        try:
            cleanup_browser(effective_task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {effective_task_id}: {e}")

        # Update session messages and save session log
        self._session_messages = messages
        self._save_session_log(messages)
        
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
        import uuid
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
