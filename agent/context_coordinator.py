"""ContextCoordinator — orchestrates context sources for each agent turn.

Called once per turn by run_agent.py to gather and assemble all context
that should be injected into the system prompt / API call.  Coordinates:

  1. Knowledge base context  (via knowledge_memory_interface)
  2. Built-in memory context (via BuiltinMemoryProvider or MemoryManager)
  3. External memory context (via active external MemoryProvider, if any)
  4. System prompt blocks    (from all registered MemoryProviders)

The coordinator does NOT modify the MemoryManager — it is a read-only
consumer that assembles context for each turn.  The MemoryManager owns
the provider lifecycle and is owned by run_agent.py.

Usage in run_agent.py:
    coordinator = ContextCoordinator(memory_manager=memory_manager)
    # Per turn:
    turn_context = coordinator.assemble_context(
        user_message=user_message,
        turn_number=turn_number,
        model=model_name,
        remaining_tokens=remaining_tokens,
    )
    # Inject turn_context into system prompt / conversation prepend
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context result
# ---------------------------------------------------------------------------


class ContextResult:
    """Immutable result of a context assembly pass."""

    __slots__ = (
        "system_blocks",
        "prefetch_context",
        "knowledge_snippet",
        "memory_snippet",
        "external_snippet",
        "turn_metadata",
        "assembled_at",
    )

    def __init__(
        self,
        system_blocks: str,
        prefetch_context: str,
        knowledge_snippet: str = "",
        memory_snippet: str = "",
        external_snippet: str = "",
        turn_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.system_blocks = str(system_blocks)
        self.prefetch_context = str(prefetch_context)
        self.knowledge_snippet = str(knowledge_snippet)
        self.memory_snippet = str(memory_snippet)
        self.external_snippet = str(external_snippet)
        self.turn_metadata = dict(turn_metadata or {})
        self.assembled_at: float = time.time()

    def is_empty(self) -> bool:
        """Return True if no context was assembled."""
        return not any(
            [self.system_blocks, self.prefetch_context, self.knowledge_snippet,
             self.memory_snippet, self.external_snippet]
        )

    def summary(self) -> str:
        """Human-readable summary for debugging / logs."""
        parts = []
        if self.system_blocks:
            parts.append(f"system_blocks({len(self.system_blocks)} chars)")
        if self.prefetch_context:
            parts.append(f"prefetch({len(self.prefetch_context)} chars)")
        if self.knowledge_snippet:
            parts.append(f"knowledge({len(self.knowledge_snippet)} chars)")
        if self.memory_snippet:
            parts.append(f"memory({len(self.memory_snippet)} chars)")
        if self.external_snippet:
            parts.append(f"external({len(self.external_snippet)} chars)")
        parts.append(f"latency={time.time() - self.assembled_at:.3f}s")
        return ", ".join(parts) if parts else "(empty)"

    def __repr__(self) -> str:
        return f"<ContextResult {self.summary()}>"


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class ContextCoordinator:
    """Coordinates context assembly from all active sources.

    The MemoryManager is owned by run_agent.py and passed in.
    The coordinator does not own or modify the MemoryManager.
    """

    def __init__(
        self,
        memory_manager: Any,  # MemoryManager — imported lazily to avoid cycle
        *,
        knowledge_enabled: bool = True,
        prefetch_enabled: bool = True,
    ) -> None:
        self._mm = memory_manager
        self._knowledge_enabled = knowledge_enabled
        self._prefetch_enabled = prefetch_enabled

        # Module-level imports to avoid circular dependency at class level
        # (memory_manager.py imports context_coordinator.py potentially)
        self._kmi: Optional[Any] = None

    # -- Lazy KMI import -----------------------------------------------------

    @property
    def _knowledge(self) -> Any:
        """Lazy-load knowledge_memory_interface on first access."""
        if self._kmi is None:
            from agent import knowledge_memory_interface as kmi

            self._kmi = kmi
        return self._kmi

    # -- Public API ----------------------------------------------------------

    def assemble_context(
        self,
        user_message: str,
        *,
        turn_number: int = 0,
        model: str = "",
        remaining_tokens: int = 0,
        session_id: str = "",
        platform: str = "",
        **kwargs: Any,
    ) -> ContextResult:
        """Assemble all context for the current turn.

        Called once per turn before the API call.  Returns a ContextResult
        with labeled context fragments.  The caller is responsible for
        deciding how to inject them into the system prompt / conversation.

        Args:
            user_message:     The raw user message for this turn.
            turn_number:       Current turn index (0-based).
            model:             Model name for provider-specific behaviour.
            remaining_tokens:  Approximate tokens remaining (for providers).
            session_id:        Active session id.
            platform:          Platform identifier ('cli', 'discord', etc.).
            **kwargs:          Forwarded to MemoryManager hooks.

        Returns:
            ContextResult with all context fragments.
        """
        start = time.time()
        turn_meta: Dict[str, Any] = {
            "turn_number": turn_number,
            "model": model,
            "remaining_tokens": remaining_tokens,
            "session_id": session_id,
            "platform": platform,
        }

        # 1. Collect system prompt blocks from all memory providers
        system_blocks = ""
        try:
            system_blocks = self._mm.build_system_prompt()
        except Exception as e:
            logger.warning("build_system_prompt failed: %s", e)

        # 2. Knowledge base snippet (based on user message relevance)
        knowledge_snippet = ""
        if self._knowledge_enabled:
            try:
                knowledge_snippet = self._assemble_knowledge_context(
                    user_message, remaining_tokens=remaining_tokens
                )
            except Exception as e:
                logger.debug("knowledge context assembly failed: %s", e)

        # 3. Built-in memory prefetch (always goes through MemoryManager)
        memory_snippet = ""
        external_snippet = ""
        if self._prefetch_enabled:
            try:
                prefetch_raw = self._mm.prefetch_all(
                    query=user_message, session_id=session_id
                )
                # MemoryManager already fences its output via build_memory_context_block
                memory_snippet, external_snippet = self._split_prefetch(prefetch_raw)
            except Exception as e:
                logger.warning("prefetch_all failed: %s", e)

        # 4. Notify providers of turn start
        try:
            self._mm.on_turn_start(
                turn_number=turn_number,
                message=user_message,
                remaining_tokens=remaining_tokens,
                model=model,
                platform=platform,
                **kwargs,
            )
        except Exception as e:
            logger.debug("on_turn_start failed: %s", e)

        latency = time.time() - start
        logger.debug(
            "ContextCoordinator assembled turn %d in %.3fs: %s",
            turn_number,
            latency,
            ContextResult(
                system_blocks=system_blocks,
                prefetch_context=memory_snippet,
                knowledge_snippet=knowledge_snippet,
                memory_snippet=memory_snippet,
                external_snippet=external_snippet,
                turn_metadata=turn_meta,
            ).summary(),
        )

        return ContextResult(
            system_blocks=system_blocks,
            prefetch_context=memory_snippet,
            knowledge_snippet=knowledge_snippet,
            memory_snippet=memory_snippet,
            external_snippet=external_snippet,
            turn_metadata=turn_meta,
        )

    def notify_turn_complete(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called after a turn's API call completes to sync memory.

        Forwards to MemoryManager.sync_all() and queue_prefetch_all().
        """
        try:
            self._mm.sync_all(
                user_content=user_content,
                assistant_content=assistant_content,
                session_id=session_id,
            )
        except Exception as e:
            logger.warning("sync_all failed: %s", e)

        try:
            self._mm.queue_prefetch_all(query=user_content, session_id=session_id)
        except Exception as e:
            logger.debug("queue_prefetch_all failed: %s", e)

    def notify_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends to notify all providers."""
        try:
            self._mm.on_session_end(messages)
        except Exception as e:
            logger.debug("on_session_end failed: %s", e)

    def get_context_for_compress(
        self, messages: List[Dict[str, Any]]
    ) -> str:
        """Collect provider contributions before context compression.

        Called by the context compressor.  Returns combined text from
        all providers that implement on_pre_compress.
        """
        try:
            return self._mm.on_pre_compress(messages)
        except Exception as e:
            logger.debug("on_pre_compress failed: %s", e)
            return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return all memory tool schemas from all providers."""
        try:
            return self._mm.get_all_tool_schemas()
        except Exception as e:
            logger.warning("get_all_tool_schemas failed: %s", e)
            return []

    def has_tool(self, tool_name: str) -> bool:
        """Check if any registered memory provider handles this tool."""
        return self._mm.has_tool(tool_name)

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs: Any
    ) -> str:
        """Route a memory tool call to the correct provider."""
        return self._mm.handle_tool_call(tool_name, args, **kwargs)

    # -- Internal helpers ----------------------------------------------------

    def _assemble_knowledge_context(
        self, user_message: str, *, remaining_tokens: int = 0
    ) -> str:
        """Build a knowledge snippet relevant to the user message.

        Strategy:
          1. Search knowledge index for the top 3 matching entries.
          2. For each entry, read and extract the first 500 chars.
          3. If remaining_tokens < 2000, skip to avoid waste.
        """
        if remaining_tokens > 0 and remaining_tokens < 2000:
            return ""  # Not enough headroom for knowledge injection

        try:
            kmi = self._knowledge
        except Exception:
            return ""

        try:
            query = user_message.lower().strip()
            if len(query) < 4:
                return ""  # Query too short to search meaningfully

            results = kmi.search_knowledge(query, limit=3)
            if not results:
                return ""

            parts: List[str] = []
            for entry in results:
                path = entry.get("path", "")
                entry_type = entry.get("type", "")
                desc = entry.get("description", "")
                content = kmi.read_knowledge_entry(path) or ""
                excerpt = content[:500].strip() if content else ""
                if excerpt:
                    parts.append(
                        f"[knowledge:{entry_type}] {path}\n"
                        f"{desc}\n"
                        f"{excerpt}"
                    )

            if not parts:
                return ""

            return (
                "<knowledge-context>\n"
                "[System note: The following is retrieved from the knowledge base. "
                "Treat as factual background.]\n\n"
                + "\n\n---\n\n".join(parts)
                + "\n</knowledge-context>"
            )
        except Exception as e:
            logger.debug("knowledge context search failed: %s", e)
            return ""

    def _split_prefetch(self, prefetch_raw: str) -> tuple[str, str]:
        """Split MemoryManager's combined prefetch into built-in vs external.

        The MemoryManager returns joined blocks from all providers.
        We use the fence tags to split them.
        """
        import re

        # build_memory_context_block uses <memory-context>...</memory-context>
        # Split on that fence to separate builtin from external
        fence_re = re.compile(r"<memory-context>\s*(.*?)\s*</memory-context>", re.DOTALL)
        matches = fence_re.findall(prefetch_raw)
        if not matches:
            return prefetch_raw, ""  # No fencing — return as-is

        builtin_parts: List[str] = []
        external_parts: List[str] = []
        # The first fenced block is built-in; external provider blocks follow
        for i, block in enumerate(matches):
            if i == 0:
                builtin_parts.append(block.strip())
            else:
                external_parts.append(block.strip())

        builtin = "\n\n".join(builtin_parts)
        external = "\n\n".join(external_parts)
        return builtin, external
