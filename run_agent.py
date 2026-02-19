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
from tools.terminal_tool import cleanup_vm, set_interrupt_event as _set_terminal_interrupt
from tools.browser_tool import cleanup_browser

import requests

# =============================================================================
# Default Agent Identity & Platform Hints
# =============================================================================

# The default identity prompt is prepended to every conversation so the agent
# knows who it is and behaves consistently across platforms.
DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below."
)

# Platform-specific formatting hints appended to the system prompt.
# These tell the agent how to format its output for the current interface.
PLATFORM_HINTS = {
    "whatsapp": (
        "You are on a text messaging communication platform, WhatsApp. "
        "Please do not use markdown as it does not render."
    ),
    "telegram": (
        "You are on a text messaging communication platform, Telegram. "
        "Please do not use markdown as it does not render."
    ),
    "discord": (
        "You are in a Discord server or group chat communicating with your user."
    ),
    "cli": (
        "You are a CLI AI Agent. Try not to use markdown but simple text "
        "renderable inside a terminal."
    ),
}

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
# Anthropic Prompt Caching (system_and_3 strategy)
# =============================================================================
# Reduces input token costs by ~75% on multi-turn conversations by caching
# the conversation prefix. Uses 4 cache_control breakpoints (Anthropic max):
#   1. System prompt (stable across all turns)
#   2-4. Last 3 non-system messages (rolling window)
#
# Cached tokens are read at 0.1x input price. Cache writes cost 1.25x (5m TTL)
# or 2x (1h TTL). Only applied to Claude models via OpenRouter.

def _apply_cache_marker(msg: dict, cache_marker: dict) -> None:
    """
    Add cache_control to a single message, handling all format variations.

    - tool messages: cache_control at message level (Anthropic API quirk)
    - string content: converted to multipart content array
    - list content: marker added to last item
    - None content (assistant with tool_calls): message level
    """
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        msg["cache_control"] = cache_marker
        return

    if content is None:
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, "cache_control": cache_marker}]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
) -> List[Dict[str, Any]]:
    """
    Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints:
      1. System prompt (index 0, stable across all turns)
      2-4. Last 3 non-system messages (rolling cache frontier)

    Each breakpoint tells Anthropic "cache everything from the start up to here."
    Multiple breakpoints create a ladder of cached prefixes at different depths,
    which provides robust cache hits even when the most recent cache entry hasn't
    propagated yet.

    Args:
        api_messages: Fully assembled message list (system prompt first).
        cache_ttl: "5m" (default, 1.25x write cost) or "1h" (2x write cost).

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    breakpoints_used = 0

    # Breakpoint 1: System prompt (always stable, gives a guaranteed minimum hit)
    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker)
        breakpoints_used += 1

    # Breakpoints 2-4: Last 3 non-system messages (rolling window)
    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker)

    return messages


# =============================================================================
# Default System Prompt Components
# =============================================================================

# Skills guidance - embeds a compact skill index in the system prompt so
# the model can match skills at a glance without extra tool calls.
def build_skills_system_prompt() -> str:
    """
    Build a dynamic skills system prompt by scanning the skills/ directory.
    
    Returns a prompt section that lists all skill categories (with descriptions
    from DESCRIPTION.md) and their skill names inline, so the model can
    immediately see if a relevant skill exists and load it with a single
    skill_view(name) call -- no discovery tool calls needed.
    
    Returns:
        str: The skills system prompt section, or empty string if no skills found.
    """
    import re
    from pathlib import Path
    
    skills_dir = Path(__file__).parent / "skills"
    if not skills_dir.exists():
        return ""
    
    # Scan for SKILL.md files grouped by category
    skills_by_category = {}
    for skill_file in skills_dir.rglob("SKILL.md"):
        rel_path = skill_file.relative_to(skills_dir)
        parts = rel_path.parts
        if len(parts) >= 2:
            category = parts[0]
            skill_name = parts[-2]  # Folder containing SKILL.md
        else:
            category = "general"
            skill_name = skill_file.parent.name
        skills_by_category.setdefault(category, []).append(skill_name)
    
    if not skills_by_category:
        return ""
    
    # Load category descriptions from DESCRIPTION.md files (YAML frontmatter)
    category_descriptions = {}
    for category in skills_by_category:
        desc_file = skills_dir / category / "DESCRIPTION.md"
        if desc_file.exists():
            try:
                content = desc_file.read_text(encoding="utf-8")
                # Parse description from YAML frontmatter: ---\ndescription: ...\n---
                match = re.search(r"^---\s*\n.*?description:\s*(.+?)\s*\n.*?^---", content, re.MULTILINE | re.DOTALL)
                if match:
                    category_descriptions[category] = match.group(1).strip()
            except Exception:
                pass
    
    # Build compact index: category with description + skill names
    index_lines = []
    for category in sorted(skills_by_category.keys()):
        desc = category_descriptions.get(category, "")
        names = ", ".join(sorted(skills_by_category[category]))
        if desc:
            index_lines.append(f"  {category}: {desc}")
        else:
            index_lines.append(f"  {category}:")
        index_lines.append(f"    skills: {names}")
    
    return (
        "## Skills (mandatory)\n"
        "Before replying, scan the skills below. If one clearly matches your task, "
        "load it with skill_view(name) and follow its instructions.\n"
        "\n"
        "<available_skills>\n"
        + "\n".join(index_lines) + "\n"
        "</available_skills>\n"
        "\n"
        "If none match, proceed normally without loading a skill."
    )


# =============================================================================
# Context File Injection (SOUL.md, AGENTS.md, .cursorrules)
# =============================================================================

# Maximum characters per context file before truncation
CONTEXT_FILE_MAX_CHARS = 20_000
# Truncation strategy: keep 70% from the head, 20% from the tail
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2


def _truncate_content(content: str, filename: str, max_chars: int = CONTEXT_FILE_MAX_CHARS) -> str:
    """
    Truncate content if it exceeds max_chars using a head/tail strategy.
    
    Keeps 70% from the start and 20% from the end, with a truncation
    marker in the middle so the model knows content was cut.
    """
    if len(content) <= max_chars:
        return content
    
    head_chars = int(max_chars * CONTEXT_TRUNCATE_HEAD_RATIO)
    tail_chars = int(max_chars * CONTEXT_TRUNCATE_TAIL_RATIO)
    head = content[:head_chars]
    tail = content[-tail_chars:]
    
    marker = f"\n\n[...truncated {filename}: kept {head_chars}+{tail_chars} of {len(content)} chars. Use file tools to read the full file.]\n\n"
    return head + marker + tail


def build_context_files_prompt(cwd: str = None) -> str:
    """
    Discover and load context files (SOUL.md, AGENTS.md, .cursorrules)
    for injection into the system prompt.
    
    Discovery rules:
    - AGENTS.md: Recursively search from cwd (only if top-level exists).
                 Each file becomes a ## section with its relative path.
    - .cursorrules: Check cwd for .cursorrules file and .cursor/rules/*.mdc
    - SOUL.md: Check cwd first, then ~/.hermes/SOUL.md as global fallback
    
    Args:
        cwd: Working directory to search from. Defaults to os.getcwd().
    
    Returns:
        str: The context files prompt section, or empty string if none found.
    """
    import os
    import glob as glob_mod
    from pathlib import Path
    
    if cwd is None:
        cwd = os.getcwd()
    
    cwd_path = Path(cwd).resolve()
    sections = []
    
    # ----- AGENTS.md (hierarchical, recursive) -----
    top_level_agents = None
    for name in ["AGENTS.md", "agents.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            top_level_agents = candidate
            break
    
    if top_level_agents:
        # Recursively find all AGENTS.md files (case-insensitive)
        agents_files = []
        for root, dirs, files in os.walk(cwd_path):
            # Skip hidden directories and common non-project dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.venv')]
            for f in files:
                if f.lower() == "agents.md":
                    agents_files.append(Path(root) / f)
        
        # Sort by path depth (top-level first, then deeper)
        agents_files.sort(key=lambda p: len(p.parts))
        
        total_agents_content = ""
        for agents_path in agents_files:
            try:
                content = agents_path.read_text(encoding="utf-8").strip()
                if content:
                    rel_path = agents_path.relative_to(cwd_path)
                    total_agents_content += f"## {rel_path}\n\n{content}\n\n"
            except Exception:
                pass
        
        if total_agents_content:
            total_agents_content = _truncate_content(total_agents_content, "AGENTS.md")
            sections.append(total_agents_content)
    
    # ----- .cursorrules -----
    cursorrules_content = ""
    
    # Check for .cursorrules file
    cursorrules_file = cwd_path / ".cursorrules"
    if cursorrules_file.exists():
        try:
            content = cursorrules_file.read_text(encoding="utf-8").strip()
            if content:
                cursorrules_content += f"## .cursorrules\n\n{content}\n\n"
        except Exception:
            pass
    
    # Check for .cursor/rules/*.mdc files
    cursor_rules_dir = cwd_path / ".cursor" / "rules"
    if cursor_rules_dir.exists() and cursor_rules_dir.is_dir():
        mdc_files = sorted(cursor_rules_dir.glob("*.mdc"))
        for mdc_file in mdc_files:
            try:
                content = mdc_file.read_text(encoding="utf-8").strip()
                if content:
                    cursorrules_content += f"## .cursor/rules/{mdc_file.name}\n\n{content}\n\n"
            except Exception:
                pass
    
    if cursorrules_content:
        cursorrules_content = _truncate_content(cursorrules_content, ".cursorrules")
        sections.append(cursorrules_content)
    
    # ----- SOUL.md (cwd first, then ~/.hermes/ fallback) -----
    soul_content = ""
    soul_path = None
    
    for name in ["SOUL.md", "soul.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            soul_path = candidate
            break
    
    if not soul_path:
        # Global fallback
        global_soul = Path.home() / ".hermes" / "SOUL.md"
        if global_soul.exists():
            soul_path = global_soul
    
    if soul_path:
        try:
            content = soul_path.read_text(encoding="utf-8").strip()
            if content:
                content = _truncate_content(content, "SOUL.md")
                soul_content = f"## SOUL.md\n\nIf SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it.\n\n{content}"
                sections.append(soul_content)
        except Exception:
            pass
    
    # ----- Assemble -----
    if not sections:
        return ""
    
    return "# Project Context\n\nThe following project context files have been loaded and should be followed:\n\n" + "\n".join(sections)


def _build_tool_preview(tool_name: str, args: dict, max_len: int = 40) -> str:
    """
    Build a short preview of a tool call's primary argument for display.
    
    Returns a truncated string showing the most informative argument,
    or None if no meaningful preview is available.
    
    Args:
        tool_name: Name of the tool being called
        args: The tool call arguments dict
        max_len: Maximum preview length before truncation
    
    Returns:
        str or None: Short preview string, or None
    """
    # Map tool names to their primary argument key(s)
    primary_args = {
        "terminal": "command",
        "web_search": "query",
        "web_extract": "urls",
        "read_file": "path",
        "write_file": "path",
        "patch": "path",
        "search": "pattern",
        "browser_navigate": "url",
        "browser_click": "ref",
        "browser_type": "text",
        "image_generate": "prompt",
        "text_to_speech": "text",
        "vision_analyze": "question",
        "mixture_of_agents": "user_prompt",
        "skill_view": "name",
        "skills_list": "category",
        "schedule_cronjob": "name",
    }
    
    # Special handling for tools with composite previews
    if tool_name == "process":
        action = args.get("action", "")
        session_id = args.get("session_id", "")
        data = args.get("data", "")
        timeout = args.get("timeout")
        parts = [action]
        if session_id:
            parts.append(session_id[:16])
        if data:
            parts.append(f'"{data[:20]}"')
        if timeout and action == "wait":
            parts.append(f"{timeout}s")
        return " ".join(parts) if parts else None
    
    if tool_name == "todo":
        todos_arg = args.get("todos")
        merge = args.get("merge", False)
        if todos_arg is None:
            return "reading task list"
        elif merge:
            return f"updating {len(todos_arg)} task(s)"
        else:
            return f"planning {len(todos_arg)} task(s)"
    
    if tool_name == "session_search":
        query = args.get("query", "")
        return f"recall: \"{query[:25]}{'...' if len(query) > 25 else ''}\""

    if tool_name == "memory":
        action = args.get("action", "")
        target = args.get("target", "")
        if action == "add":
            content = args.get("content", "")
            return f"+{target}: \"{content[:25]}{'...' if len(content) > 25 else ''}\""
        elif action == "replace":
            return f"~{target}: \"{args.get('old_text', '')[:20]}\""
        elif action == "remove":
            return f"-{target}: \"{args.get('old_text', '')[:20]}\""
        return action
    
    if tool_name == "send_message":
        target = args.get("target", "?")
        msg = args.get("message", "")
        if len(msg) > 20:
            msg = msg[:17] + "..."
        return f"to {target}: \"{msg}\""
    
    if tool_name.startswith("rl_"):
        rl_previews = {
            "rl_list_environments": "listing envs",
            "rl_select_environment": args.get("name", ""),
            "rl_get_current_config": "reading config",
            "rl_edit_config": f"{args.get('field', '')}={args.get('value', '')}",
            "rl_start_training": "starting",
            "rl_check_status": args.get("run_id", "")[:16],
            "rl_stop_training": f"stopping {args.get('run_id', '')[:16]}",
            "rl_get_results": args.get("run_id", "")[:16],
            "rl_list_runs": "listing runs",
            "rl_test_inference": f"{args.get('num_steps', 3)} steps",
        }
        return rl_previews.get(tool_name)

    key = primary_args.get(tool_name)
    if not key:
        # Try common arg names as fallback
        for fallback_key in ("query", "text", "command", "path", "name", "prompt"):
            if fallback_key in args:
                key = fallback_key
                break
    
    if not key or key not in args:
        return None
    
    value = args[key]
    
    # Handle list values (e.g., urls)
    if isinstance(value, list):
        value = value[0] if value else ""
    
    preview = str(value).strip()
    if not preview:
        return None
    
    # Truncate
    if len(preview) > max_len:
        preview = preview[:max_len - 3] + "..."
    
    return preview


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
        self.base_url = base_url or "https://openrouter.ai/api/v1"
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
        
        # Show prompt caching status
        if self._use_prompt_caching and not self.quiet_mode:
            print(f"💾 Prompt caching: ENABLED (Claude via OpenRouter, {self._cache_ttl} TTL)")
        
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
        
        # Cached system prompt -- built once per session, only rebuilt on compression
        self._cached_system_prompt: Optional[str] = None
        
        # SQLite session store (optional -- provided by CLI or gateway)
        self._session_db = session_db
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
            except Exception:
                pass
        
        # In-memory todo list for task planning (one per agent/session)
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()
        
        # Persistent memory (MEMORY.md + USER.md) -- loaded from disk
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        if not skip_memory:
            try:
                from hermes_cli.config import load_config as _load_mem_config
                mem_config = _load_mem_config().get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass  # Memory is optional -- don't break agent init
        
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
        Generate a clean, aligned tool activity line for CLI quiet mode.

        Format: ┊ {emoji} {verb:9} {detail}  {duration}

        Kawaii faces live in the animated spinner (while the tool runs).
        This completion message replaces the spinner with a permanent log line.
        """
        dur = f"{duration:.1f}s"

        def _trunc(s, n=40):
            s = str(s)
            return (s[:n-3] + "...") if len(s) > n else s

        def _path(p, n=35):
            p = str(p)
            return ("..." + p[-(n-3):]) if len(p) > n else p

        # ── Web ──
        if tool_name == "web_search":
            q = _trunc(args.get("query", ""), 42)
            return f"┊ 🔍 search    {q}  {dur}"

        if tool_name == "web_extract":
            urls = args.get("urls", [])
            if urls:
                url = urls[0] if isinstance(urls, list) else str(urls)
                domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                extra = f" +{len(urls)-1}" if len(urls) > 1 else ""
                return f"┊ 📄 fetch     {_trunc(domain, 35)}{extra}  {dur}"
            return f"┊ 📄 fetch     pages  {dur}"

        if tool_name == "web_crawl":
            url = args.get("url", "")
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            return f"┊ 🕸️  crawl     {_trunc(domain, 35)}  {dur}"

        # ── Terminal & Process ──
        if tool_name == "terminal":
            cmd = _trunc(args.get("command", ""), 42)
            return f"┊ 💻 $         {cmd}  {dur}"

        if tool_name == "process":
            action = args.get("action", "?")
            sid = args.get("session_id", "")[:12]
            labels = {
                "list": "ls processes", "poll": f"poll {sid}",
                "log": f"log {sid}", "wait": f"wait {sid}",
                "kill": f"kill {sid}", "write": f"write {sid}",
                "submit": f"submit {sid}",
            }
            detail = labels.get(action, f"{action} {sid}")
            return f"┊ ⚙️  proc      {detail}  {dur}"

        # ── Files ──
        if tool_name == "read_file":
            return f"┊ 📖 read      {_path(args.get('path', ''))}  {dur}"

        if tool_name == "write_file":
            return f"┊ ✍️  write     {_path(args.get('path', ''))}  {dur}"

        if tool_name == "patch":
            return f"┊ 🔧 patch     {_path(args.get('path', ''))}  {dur}"

        if tool_name == "search":
            pattern = _trunc(args.get("pattern", ""), 35)
            target = args.get("target", "content")
            verb = "find" if target == "files" else "grep"
            return f"┊ 🔎 {verb:9} {pattern}  {dur}"

        # ── Browser ──
        if tool_name == "browser_navigate":
            url = args.get("url", "")
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            return f"┊ 🌐 navigate  {_trunc(domain, 35)}  {dur}"

        if tool_name == "browser_snapshot":
            mode = "full" if args.get("full") else "compact"
            return f"┊ 📸 snapshot  {mode}  {dur}"

        if tool_name == "browser_click":
            return f"┊ 👆 click     {args.get('ref', '?')}  {dur}"

        if tool_name == "browser_type":
            text = _trunc(args.get("text", ""), 30)
            return f"┊ ⌨️  type      \"{text}\"  {dur}"

        if tool_name == "browser_scroll":
            d = args.get("direction", "down")
            arrow = {"down": "↓", "up": "↑", "right": "→", "left": "←"}.get(d, "↓")
            return f"┊ {arrow}  scroll    {d}  {dur}"

        if tool_name == "browser_back":
            return f"┊ ◀️  back      {dur}"

        if tool_name == "browser_press":
            return f"┊ ⌨️  press     {args.get('key', '?')}  {dur}"

        if tool_name == "browser_close":
            return f"┊ 🚪 close     browser  {dur}"

        if tool_name == "browser_get_images":
            return f"┊ 🖼️  images    extracting  {dur}"

        if tool_name == "browser_vision":
            return f"┊ 👁️  vision    analyzing page  {dur}"

        # ── Planning ──
        if tool_name == "todo":
            todos_arg = args.get("todos")
            merge = args.get("merge", False)
            if todos_arg is None:
                return f"┊ 📋 plan      reading tasks  {dur}"
            elif merge:
                return f"┊ 📋 plan      update {len(todos_arg)} task(s)  {dur}"
            else:
                return f"┊ 📋 plan      {len(todos_arg)} task(s)  {dur}"

        # ── Session Search ──
        if tool_name == "session_search":
            query = _trunc(args.get("query", ""), 35)
            return f"┊ 🔍 recall    \"{query}\"  {dur}"

        # ── Memory ──
        if tool_name == "memory":
            action = args.get("action", "?")
            target = args.get("target", "")
            if action == "add":
                preview = _trunc(args.get("content", ""), 30)
                return f"┊ 🧠 memory    +{target}: \"{preview}\"  {dur}"
            elif action == "replace":
                snippet = _trunc(args.get("old_text", ""), 20)
                return f"┊ 🧠 memory    ~{target}: \"{snippet}\"  {dur}"
            elif action == "remove":
                snippet = _trunc(args.get("old_text", ""), 20)
                return f"┊ 🧠 memory    -{target}: \"{snippet}\"  {dur}"
            elif action == "search_sessions":
                query = _trunc(args.get("content", ""), 30)
                return f"┊ 🧠 recall    \"{query}\"  {dur}"
            else:
                return f"┊ 🧠 memory    {action}  {dur}"

        # ── Skills ──
        if tool_name == "skills_list":
            return f"┊ 📚 skills    list {args.get('category', 'all')}  {dur}"

        if tool_name == "skill_view":
            return f"┊ 📚 skill     {_trunc(args.get('name', ''), 30)}  {dur}"

        # ── Generation & Media ──
        if tool_name == "image_generate":
            return f"┊ 🎨 create    {_trunc(args.get('prompt', ''), 35)}  {dur}"

        if tool_name == "text_to_speech":
            return f"┊ 🔊 speak     {_trunc(args.get('text', ''), 30)}  {dur}"

        if tool_name == "vision_analyze":
            return f"┊ 👁️  vision    {_trunc(args.get('question', ''), 30)}  {dur}"

        if tool_name == "mixture_of_agents":
            return f"┊ 🧠 reason    {_trunc(args.get('user_prompt', ''), 30)}  {dur}"

        # ── Messaging & Scheduling ──
        if tool_name == "send_message":
            target = args.get("target", "?")
            msg = _trunc(args.get("message", ""), 25)
            return f"┊ 📨 send      {target}: \"{msg}\"  {dur}"

        if tool_name == "schedule_cronjob":
            name = _trunc(args.get("name", args.get("prompt", "task")), 30)
            return f"┊ ⏰ schedule  {name}  {dur}"

        if tool_name == "list_cronjobs":
            return f"┊ ⏰ jobs      listing  {dur}"

        if tool_name == "remove_cronjob":
            return f"┊ ⏰ remove    job {args.get('job_id', '?')}  {dur}"

        # ── RL Training ──
        if tool_name.startswith("rl_"):
            rl = {
                "rl_list_environments": "list envs",
                "rl_select_environment": f"select {args.get('name', '')}",
                "rl_get_current_config": "get config",
                "rl_edit_config": f"set {args.get('field', '?')}",
                "rl_start_training": "start training",
                "rl_check_status": f"status {args.get('run_id', '?')[:12]}",
                "rl_stop_training": f"stop {args.get('run_id', '?')[:12]}",
                "rl_get_results": f"results {args.get('run_id', '?')[:12]}",
                "rl_list_runs": "list runs",
                "rl_test_inference": "test inference",
            }
            detail = rl.get(tool_name, tool_name.replace("rl_", ""))
            return f"┊ 🧪 rl        {detail}  {dur}"

        # ── Fallback ──
        preview = _build_tool_preview(tool_name, args) or ""
        return f"┊ ⚡ {tool_name[:9]:9} {_trunc(preview, 35)}  {dur}"
    
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
        # Signal the terminal tool to kill any running subprocess immediately
        _set_terminal_interrupt(True)
        if not self.quiet_mode:
            print(f"\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
    
    def clear_interrupt(self) -> None:
        """Clear any pending interrupt request."""
        self._interrupt_requested = False
        self._interrupt_message = None
    
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
        _set_terminal_interrupt(False)
    
    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested
    
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
        prompt_parts = [DEFAULT_AGENT_IDENTITY]

        caller_prompt = system_message if system_message is not None else self.ephemeral_system_prompt
        if caller_prompt:
            prompt_parts.append(caller_prompt)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view'])
        skills_prompt = build_skills_system_prompt() if has_skills_tools else ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            context_files_prompt = build_context_files_prompt()
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        now = datetime.now()
        prompt_parts.append(
            f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        )

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(prompt_parts)
    
    def _invalidate_system_prompt(self):
        """
        Invalidate the cached system prompt, forcing a rebuild on the next turn.
        
        Called after context compression events. Also reloads memory from disk
        so the rebuilt prompt captures any writes from this session.
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()
    
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
        
        # Hydrate todo store from conversation history (gateway creates a fresh
        # AIAgent per message, so the in-memory store is empty -- we need to
        # recover the todo state from the most recent todo tool response in history)
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)
        
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
        
        # ── System prompt (cached per session for prefix caching) ──
        # Built once on first call, reused for all subsequent calls.
        # Only rebuilt after context compression events (which invalidate
        # the cache and reload memory from disk).
        if self._cached_system_prompt is None:
            self._cached_system_prompt = self._build_system_prompt(system_message)
            # Store the system prompt snapshot in SQLite
            if self._session_db:
                try:
                    self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                except Exception:
                    pass

        active_system_prompt = self._cached_system_prompt

        # Log user message to SQLite
        if self._session_db:
            try:
                self._session_db.append_message(self.session_id, "user", user_message)
            except Exception:
                pass

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
                    
                    # Stop thinking spinner -- defer "got it!" message for final text responses.
                    # For tool calls, stop silently since the tool execution messages are more informative.
                    if thinking_spinner:
                        has_tool_calls = (
                            hasattr(response.choices[0].message, 'tool_calls')
                            and response.choices[0].message.tool_calls
                        )
                        if has_tool_calls:
                            thinking_spinner.stop("")
                        else:
                            face = random.choice(["(◕‿◕✿)", "ヾ(＾∇＾)", "(≧◡≦)", "✧٩(ˊᗜˋ*)و✧", "(*^▽^*)"])
                            thinking_spinner.stop(f"{face} ⚕ ready ({api_duration:.1f}s)")
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
                            # Compression was possible -- re-inject todo state
                            todo_snapshot = self._todo_store.format_for_injection()
                            if todo_snapshot:
                                messages.append({"role": "user", "content": todo_snapshot})
                            # Rebuild system prompt with fresh date/time + memory
                            self._invalidate_system_prompt()
                            active_system_prompt = self._build_system_prompt(system_message)
                            self._cached_system_prompt = active_system_prompt
                            # Split session in SQLite (close old, open new with parent link)
                            if self._session_db:
                                try:
                                    self._session_db.end_session(self.session_id, "compression")
                                    old_session_id = self.session_id
                                    self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                                    self._session_db.create_session(
                                        session_id=self.session_id,
                                        source=self.platform or "cli",
                                        model=self.model,
                                        parent_session_id=old_session_id,
                                    )
                                    self._session_db.update_system_prompt(self.session_id, active_system_prompt)
                                except Exception:
                                    pass
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
                                # Build a short preview of the primary argument
                                preview = _build_tool_preview(function_name, function_args)
                                self.tool_progress_callback(function_name, preview)
                            except Exception as cb_err:
                                logging.debug(f"Tool progress callback error: {cb_err}")

                        tool_start_time = time.time()

                        # Todo tool -- handle directly (needs agent's TodoStore instance)
                        if function_name == "todo":
                            from tools.todo_tool import todo_tool as _todo_tool
                            function_result = _todo_tool(
                                todos=function_args.get("todos"),
                                merge=function_args.get("merge", False),
                                store=self._todo_store,
                            )
                            tool_duration = time.time() - tool_start_time
                            if self.quiet_mode:
                                print(f"  {self._get_cute_tool_message('todo', function_args, tool_duration)}")
                        # Session search -- handle directly (needs SessionDB instance)
                        elif function_name == "session_search" and self._session_db:
                            from tools.session_search_tool import session_search as _session_search
                            function_result = _session_search(
                                query=function_args.get("query", ""),
                                role_filter=function_args.get("role_filter"),
                                limit=function_args.get("limit", 3),
                                db=self._session_db,
                            )
                            tool_duration = time.time() - tool_start_time
                            if self.quiet_mode:
                                print(f"  {self._get_cute_tool_message('session_search', function_args, tool_duration)}")
                        # Memory tool -- handle directly (needs agent's MemoryStore instance)
                        elif function_name == "memory":
                            from tools.memory_tool import memory_tool as _memory_tool
                            function_result = _memory_tool(
                                action=function_args.get("action"),
                                target=function_args.get("target", "memory"),
                                content=function_args.get("content"),
                                old_text=function_args.get("old_text"),
                                store=self._memory_store,
                            )
                            tool_duration = time.time() - tool_start_time
                            if self.quiet_mode:
                                print(f"  {self._get_cute_tool_message('memory', function_args, tool_duration)}")
                        # Execute other tools - with animated kawaii spinner in quiet mode
                        # The face is "alive" while the tool works, then vanishes
                        # and is replaced by the clean result line.
                        elif self.quiet_mode:
                            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                            tool_emoji_map = {
                                'web_search': '🔍', 'web_extract': '📄', 'web_crawl': '🕸️',
                                'terminal': '💻', 'process': '⚙️',
                                'read_file': '📖', 'write_file': '✍️', 'patch': '🔧', 'search': '🔎',
                                'browser_navigate': '🌐', 'browser_snapshot': '📸',
                                'browser_click': '👆', 'browser_type': '⌨️',
                                'browser_scroll': '📜', 'browser_back': '◀️',
                                'browser_press': '⌨️', 'browser_close': '🚪',
                                'browser_get_images': '🖼️', 'browser_vision': '👁️',
                                'image_generate': '🎨', 'text_to_speech': '🔊',
                                'vision_analyze': '👁️', 'mixture_of_agents': '🧠',
                                'skills_list': '📚', 'skill_view': '📚',
                                'schedule_cronjob': '⏰', 'list_cronjobs': '⏰', 'remove_cronjob': '⏰',
                                'send_message': '📨', 'todo': '📋', 'memory': '🧠', 'session_search': '🔍',
                            }
                            emoji = tool_emoji_map.get(function_name, '⚡')
                            preview = _build_tool_preview(function_name, function_args) or function_name
                            if len(preview) > 30:
                                preview = preview[:27] + "..."
                            spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots')
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
                        
                        # Check for interrupt between tool calls - skip remaining
                        # tools so the agent can respond to the user immediately
                        if self._interrupt_requested and i < len(assistant_message.tool_calls):
                            remaining = len(assistant_message.tool_calls) - i
                            print(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)")
                            # Add placeholder results for skipped tool calls so the
                            # message sequence stays valid (assistant tool_calls need matching tool results)
                            for skipped_tc in assistant_message.tool_calls[i:]:
                                messages.append({
                                    "role": "tool",
                                    "content": "[Tool execution skipped - user sent a new message]",
                                    "tool_call_id": skipped_tc.id
                                })
                            break
                        
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
                        # Re-inject todo state after compression
                        todo_snapshot = self._todo_store.format_for_injection()
                        if todo_snapshot:
                            messages.append({"role": "user", "content": todo_snapshot})
                        # Rebuild system prompt with fresh date/time + memory
                        self._invalidate_system_prompt()
                        active_system_prompt = self._build_system_prompt(system_message)
                        self._cached_system_prompt = active_system_prompt
                        # Split session in SQLite (close old, open new with parent link)
                        if self._session_db:
                            try:
                                self._session_db.end_session(self.session_id, "compression")
                                old_session_id = self.session_id
                                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                                self._session_db.create_session(
                                    session_id=self.session_id,
                                    source=self.platform or "cli",
                                    model=self.model,
                                    parent_session_id=old_session_id,
                                )
                                self._session_db.update_system_prompt(self.session_id, active_system_prompt)
                            except Exception:
                                pass
                    
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
        
        # Log new messages to SQLite session store (everything after the user message we already logged)
        if self._session_db:
            try:
                # Skip messages that were in the conversation history before this call
                # (the user message was already logged at the start of run_conversation)
                start_idx = (len(conversation_history) if conversation_history else 0) + 1  # +1 for the user msg
                for msg in messages[start_idx:]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content")
                    # Extract tool call info from assistant messages
                    tool_calls_data = None
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_calls_data = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in msg.tool_calls]
                    elif isinstance(msg.get("tool_calls"), list):
                        tool_calls_data = msg["tool_calls"]
                    self._session_db.append_message(
                        session_id=self.session_id,
                        role=role,
                        content=content,
                        tool_name=msg.get("tool_name"),
                        tool_calls=tool_calls_data,
                        tool_call_id=msg.get("tool_call_id"),
                    )
            except Exception:
                pass
        
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
