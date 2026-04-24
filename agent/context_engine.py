"""Abstract base class for pluggable context engines.

A context engine controls how conversation context is managed when
approaching the model's token limit. The built-in ContextCompressor
is the default implementation. Third-party engines (e.g. LCM) can
replace it via the plugin system or by being placed in the
``plugins/context_engine/<name>/`` directory.

Selection is config-driven: ``context.engine`` in config.yaml.
Default is ``"compressor"`` (the built-in). Only one engine is active.

The engine is responsible for:
  - Deciding when compaction should fire
  - Performing compaction (summarization, DAG construction, etc.)
  - Optionally exposing tools the agent can call (e.g. lcm_grep)
  - Tracking token usage from API responses

Lifecycle:
  1. Engine is instantiated and registered (plugin register() or default)
  2. on_session_start() called when a conversation begins
  3. update_from_response() called after each API response with usage data
  4. should_compress() checked after each turn
  5. compress() called when should_compress() returns True
  6. on_session_end() called at real session boundaries (CLI exit, /reset,
     gateway session expiry) — NOT per-turn
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ContextEngine(ABC):
    """Base class all context engines must implement."""

    # -- Identity ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'compressor', 'lcm')."""

    # -- Token state (read by run_agent.py for display/logging) ------------
    #
    # Engines MUST maintain these. run_agent.py reads them directly.

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # -- Compaction parameters (read by run_agent.py for preflight) --------
    #
    # These control the preflight compression check.  Subclasses may
    # override via __init__ or property; defaults are sensible for most
    # engines.

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # -- Core interface ----------------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response.

        Called after every LLM call with the usage dict from the response.
        """

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Compact the message list and return the new message list.

        This is the main entry point. The engine receives the full message
        list and returns a (possibly shorter) list that fits within the
        context budget. The implementation is free to summarize, build a
        DAG, or do anything else — as long as the returned list is a valid
        OpenAI-format message sequence.

        Args:
            focus_topic: Optional topic string from manual ``/compress <focus>``.
                Engines that support guided compression should prioritise
                preserving information related to this topic.  Engines that
                don't support it may simply ignore this argument.
        """

    # -- Optional: pre-flight check ----------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick rough check before the API call (no real token count yet).

        Default returns False (skip pre-flight). Override if your engine
        can do a cheap estimate.
        """
        return False

    # -- Optional: manual /compress preflight ------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick check: is there anything in ``messages`` that can be compacted?

        Used by the gateway ``/compress`` command as a preflight guard —
        returning False lets the gateway report "nothing to compress yet"
        without making an LLM call.

        Default returns True (always attempt).  Engines with a cheap way
        to introspect their own head/tail boundaries should override this
        to return False when the transcript is still entirely protected.
        """
        return True

    # -- Optional: session lifecycle ---------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins.

        Use this to load persisted state (DAG, store) for the session.
        kwargs may include hermes_home, platform, model, etc.
        """

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Called at real session boundaries (CLI exit, /reset, gateway expiry).

        Use this to flush state, close DB connections, etc.
        NOT called per-turn — only when the session truly ends.
        """

    def on_session_reset(self) -> None:
        """Called on /new or /reset. Reset per-session state.

        Default resets compression_count and token tracking.
        """
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    # -- Optional: tools ---------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this engine provides to the agent.

        Default returns empty list (no tools). LCM would return schemas
        for lcm_grep, lcm_describe, lcm_expand here.
        """
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent.

        Only called for tool names returned by get_tool_schemas().
        Must return a JSON string.

        kwargs may include:
          messages: the current in-memory message list (for live ingestion)
        """
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- Optional: status / display ----------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return status dict for display/logging.

        Default returns the standard fields run_agent.py expects.
        """
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length else 0
            ),
            "compression_count": self.compression_count,
        }

    # -- Optional: model switch support ------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        """Called when the user switches models or on fallback activation.

        Default updates context_length and recalculates threshold_tokens
        from threshold_percent. Override if your engine needs more
        (e.g. recalculate DAG budgets, switch summary models).
        """
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
