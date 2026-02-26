"""System prompt assembly with caching.

Extracted from AIAgent._build_system_prompt. Owns the prompt-building logic
and the single-invalidation caching contract.

Caching contract:
    - build() returns cached value on subsequent calls
    - invalidate() clears cache and optionally reloads memory from disk
    - After invalidate(), the next build() call creates a fresh prompt
"""

from datetime import datetime
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    MEMORY_GUIDANCE,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    PLATFORM_HINTS,
    build_skills_system_prompt,
    build_context_files_prompt,
)


class PromptAssembler:
    """Assembles the full system prompt from layered components.

    Args:
        platform: The interface platform (e.g. "cli", "telegram", "discord").
        skip_context_files: Whether to skip loading context files (SOUL.md, etc.).
    """

    def __init__(self, *, platform=None, skip_context_files=False):
        self._platform = platform
        self._skip_context_files = skip_context_files
        self._cached_prompt = None

    def build(
        self,
        *,
        valid_tool_names,
        system_message=None,
        memory_store=None,
        memory_enabled=False,
        user_profile_enabled=False,
    ) -> str:
        """Assemble the full system prompt from all layers.

        CACHING CONTRACT: Returns cached value on subsequent calls.
        Call invalidate() before build() to force a rebuild.

        Args:
            valid_tool_names: List of enabled tool names for this session.
            system_message: Optional custom system message to include.
            memory_store: Optional memory store with format_for_system_prompt().
            memory_enabled: Whether to include memory blocks in the prompt.
            user_profile_enabled: Whether to include user profile blocks.

        Returns:
            The fully assembled system prompt string.
        """
        if self._cached_prompt is not None:
            return self._cached_prompt

        prompt_parts = [DEFAULT_AGENT_IDENTITY]

        # Tool-aware behavioral guidance
        tool_guidance = []
        if "memory" in valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        # Custom system message (NOT ephemeral — ephemeral is injected separately)
        if system_message is not None:
            prompt_parts.append(system_message)

        # Memory and user profile blocks
        if memory_store:
            if memory_enabled:
                mem_block = memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            if user_profile_enabled:
                user_block = memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # Skills system prompt
        has_skills_tools = any(
            name in valid_tool_names
            for name in ["skills_list", "skill_view", "skill_manage"]
        )
        skills_prompt = build_skills_system_prompt() if has_skills_tools else ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        # Context files (SOUL.md, etc.)
        if not self._skip_context_files:
            context_files_prompt = build_context_files_prompt()
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        # Timestamp
        now = datetime.now()
        prompt_parts.append(
            f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        )

        # Platform-specific hints (always last when present)
        platform_key = (self._platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        result = "\n\n".join(prompt_parts)
        self._cached_prompt = result
        return result

    @property
    def cached(self):
        """Return the cached prompt, or None if not yet built/invalidated."""
        return self._cached_prompt

    def invalidate(self, memory_store=None):
        """Invalidate the cached prompt, forcing rebuild on next build() call.

        If memory_store is provided, also reloads memory from disk so the
        rebuilt prompt captures any writes from this session.

        Args:
            memory_store: Optional memory store to reload from disk.
        """
        self._cached_prompt = None
        if memory_store is not None:
            memory_store.load_from_disk()
