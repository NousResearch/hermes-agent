"""Recursive Language Model context engine plugin.

Implements the RLM paradigm (arXiv:2512.24601) as a ContextEngine plugin.
The RLM treats the prompt as an external Python variable in a REPL
environment, allowing the LLM to programmatically peek, grep, chunk,
and recursively call sub-LLMs over the context.

Usage:
  - As a standalone engine: set context.engine: rlm in config.yaml
  - As a companion to LCM: set context.engine: lcm + context.rlm: true

When used with LCM, RLM provides peek/grep/partition tools while LCM
handles message compression. This is the recommended configuration.

Requires an external LLM provider (OpenAI, custom, etc.) for sub-LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)


# ── RLM REPL Environment ──────────────────────────────────────────────────


class RLMAgentEnvironment:
    """Python REPL environment for RLM context management.

    Stores the context as a variable. The root LLM can:
    - Peek at context snippets
    - Grep/search with regex
    - Chunk and map over context
    - Call sub-LLMs via env.llm_batch() or env.llm_call()

    The LLM provides its final answer via the answer dict:
    answer = {"content": "...", "ready": True}
    """

    def __init__(
        self,
        context: str,
        max_iterations: int = 20,
        output_limit: int = 8192,
    ):
        self.context = context
        self.max_iterations = max_iterations
        self.output_limit = output_limit

        # REPL namespace
        self.namespace: dict = {
            "__context__": context,
            "__context_length__": len(context),
            "__context_token_length__": self._estimate_tokens(context),
            "answer": {"content": "", "ready": False},
            "llm_batch": self._llm_batch_handler,
            "llm_call": self._llm_call_handler,
        }

        self.iteration_count = 0

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    def execute(self, code: str) -> str:
        """Execute Python code in the REPL environment.

        Returns truncated output visible to the LLM.
        """
        try:
            compiled = compile(code, "<rlm>", "exec")
            exec(compiled, self.namespace)

            result = self.namespace.get("answer", "")
            if isinstance(result, dict):
                output = result.get("content", "")
            else:
                output = str(result)

            if len(output) > self.output_limit:
                output = output[: self.output_limit] + "...[truncated]"

            return output

        except Exception as e:
            return f"REPL Error: {type(e).__name__}: {e}"

    def execute_expression(self, expression: str) -> str:
        """Execute a Python expression and return the result."""
        try:
            result = eval(expression, self.namespace)
            output = str(result)

            if len(output) > self.output_limit:
                output = output[: self.output_limit] + "...[truncated]"

            return output

        except Exception as e:
            return f"Expression Error: {type(e).__name__}: {e}"

    def _llm_batch_handler(self, prompts: list) -> list:
        """Batch sub-LLM calls. Parallel execution.

        Each prompt is a question about a subset of __context__.
        Returns list of answers from sub-LLMs.

        Note: This is a stub — the real implementation delegates to
        the connected LLM client.
        """
        return []

    def _llm_call_handler(self, prompt: str) -> str:
        """Recursive sub-LLM call stub."""
        return ""

    def set_answer(self, content: str, ready: bool = False):
        """Set the final answer."""
        self.namespace["answer"]["content"] = content
        self.namespace["answer"]["ready"] = ready

    def check_iteration_limit(self) -> bool:
        """Whether we've hit max iterations."""
        self.iteration_count += 1
        return self.iteration_count >= self.max_iterations


# ── RLM ContextEngine Plugin ─────────────────────────────────────────────


class RlmContextEngine(ContextEngine):
    """RLM adapter that implements the ContextEngine ABC.

    Unlike LCM which compresses messages, the RLM treats the context as
    an external Python variable and allows the root LLM to programmatically
    interact with it via a REPL environment.

    When used as a companion to LCM, RLM provides tools (rlm_peek,
    rlm_grep, rlm_partition) while LCM handles compression.

    Key design decisions:
    - Never compresses messages (delegates to companion engine)
    - Exposes rlm_peek, rlm_grep, rlm_partition tools to the agent
    - Sub-LLMs get tools, root LLM does not (keeps root context clean)
    - Parallel batch sub-LLM calls via llm_batch()
    """

    def __init__(self, **kwargs: Any) -> None:
        self._config: dict = kwargs.get("rlm_config", {})
        self._max_iterations = self._config.get("max_iterations", 20)
        self._output_limit = self._config.get("output_limit", 8192)

        # Token tracking (required by ABC)
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.threshold_tokens: int = 0
        self.context_length: int = 0
        self.compression_count: int = 0
        self.threshold_percent: float = 1.0  # Never triggers compression
        self.protect_last_n: int = 6

        # Store context for this session
        self._session_context: Optional[str] = None
        self._rlm_env: Optional[RLMAgentEnvironment] = None

    # ── Identity ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "rlm"

    # ── Core ABC interface ──────────────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from API response."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = prompt_tokens + completion_tokens

        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens
        self.last_total_tokens = total

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """RLM never compresses on its own — delegates to companion."""
        return False

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """RLM preflight check — always False."""
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
    ) -> List[Dict[str, Any]]:
        """Pass messages through unchanged.

        When used as a companion to LCM, LCM handles compression.
        When used standalone, RLM doesn't compress — it delegates
        context management to the REPL environment.
        """
        return list(messages)

    # ── Session lifecycle ───────────────────────────────────────────────

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Initialize RLM environment for the session."""
        self.context_length = kwargs.get("context_length", 128_000)
        self.threshold_tokens = self.context_length

        # Extract context from kwargs — passed by the agent
        context = kwargs.get("rlm_context")
        if context:
            self._session_context = context
            self._init_rlm_env(context)

        logger.info(
            "RLM session started (session=%s, max_iter=%d, output_limit=%d)",
            session_id,
            self._max_iterations,
            self._output_limit,
        )

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Session end — context is ephemeral (RLM stores in memory)."""
        pass

    def on_session_reset(self) -> None:
        """Reset per-session state."""
        super().on_session_reset()
        self._session_context = None
        self._rlm_env = None

    def build_context_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Build a flat context string from LLM messages for the REPL.

        This is called by the agent so the RLM engine can populate
        its context with the actual conversation.

        Handles both Anthropic-style content blocks
        (``content: [{type: tool_use, ...}]``) and OpenAI-style
        ``tool_calls: [{function: {name, arguments}}]`` on assistant messages.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"[{role}]: {content}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and item.get("text"):
                            parts.append(f"[{role}]: {item['text']}")
                        elif item.get("type") == "tool_use":
                            parts.append(f"[{role}]: {json.dumps(item.get('input', {}))}")
            # OpenAI-style: tool_calls=[{function: {name, arguments}}]
            # These appear alongside empty/null content on assistant messages.
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    if fn:
                        parts.append(
                            f"[{role}]: tool_call {fn.get('name', '?')} "
                            f"{fn.get('arguments', '')}"
                        )
        return "\n\n".join(parts)

    def refresh_context(self, messages: List[Dict[str, Any]]) -> None:
        """Rebuild the REPL context from the latest messages.

        Call this before each API call to keep the context fresh.
        If _rlm_env has not been created yet (no rlm_context was passed to
        on_session_start), initialize it now from the current messages.
        """
        context = self.build_context_from_messages(messages)
        if context:
            self._session_context = context
            if self._rlm_env is None:
                # First call after a session start without rlm_context — bootstrap.
                self._init_rlm_env(context)
            else:
                self._rlm_env.context = context
                self._rlm_env.namespace["__context__"] = context
                self._rlm_env.namespace["__context_length__"] = len(context)
                self._rlm_env.namespace["__context_token_length__"] = self._rlm_env._estimate_tokens(
                    context
                )

    def _init_rlm_env(self, context: str) -> None:
        """Create the REPL environment for the given context."""
        self._session_context = context
        self._rlm_env = RLMAgentEnvironment(
            context=context,
            max_iterations=self._max_iterations,
            output_limit=self._output_limit,
        )

    # ── Tools ───────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return RLM tool schemas (rlm_peek, rlm_grep, rlm_partition)."""
        return list(RLM_TOOL_SCHEMAS.values())

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch RLM tool calls."""
        handlers = {
            "rlm_peek": self._handle_peek,
            "rlm_grep": self._handle_grep,
            "rlm_partition": self._handle_partition,
        }

        handler = handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown RLM tool: {name}"})

        return handler(args)

    def _handle_peek(self, args: Dict[str, Any]) -> str:
        """Peek at a section of the context.

        Args:
            start: Start character index
            length: Number of characters to read

        Returns the context snippet.
        """
        if not self._rlm_env or not self._session_context:
            return json.dumps({"error": "No context loaded"})

        start = args.get("start", 0)
        length = args.get("length", 2000)

        try:
            context = self._session_context
            snippet = context[start : start + length]
            return json.dumps({"snippet": snippet, "start": start, "length": len(snippet)})
        except Exception as e:
            return json.dumps({"error": f"Peek failed: {e}"})

    def _handle_grep(self, args: Dict[str, Any]) -> str:
        """Grep the context for regex patterns.

        Args:
            pattern: Regex pattern to search for
            max_matches: Maximum number of matches to return (default: 20)

        Returns list of matching lines.
        """
        if not self._rlm_env or not self._session_context:
            return json.dumps({"error": "No context loaded"})

        pattern = args.get("pattern", "")
        max_matches = args.get("max_matches", 20)

        try:
            context = self._session_context
            lines = context.split("\n")
            matches = []

            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    matches.append({"line_num": i, "content": line})
                    if len(matches) >= max_matches:
                        break

            return json.dumps({
                "pattern": pattern,
                "matches": matches,
                "total_found": len([l for l in lines if re.search(pattern, l)]),
            })
        except Exception as e:
            return json.dumps({"error": f"Grep failed: {e}"})

    def _handle_partition(self, args: Dict[str, Any]) -> str:
        """Partition context into chunks and return chunk boundaries.

        Args:
            chunk_size: Characters per chunk (default: 8000)
            overlap: Characters of overlap between chunks (default: 500)

        Returns list of chunk metadata (index, start, end, length).
        """
        if not self._rlm_env or not self._session_context:
            return json.dumps({"error": "No context loaded"})

        chunk_size = args.get("chunk_size", 8000)
        overlap = args.get("overlap", 500)

        # Reject malformed bounds before entering the loop.  overlap >= chunk_size
        # causes start to not advance (or move backwards), producing an infinite loop.
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            return json.dumps({"error": f"chunk_size must be a positive integer, got {chunk_size!r}"})
        if not isinstance(overlap, int) or overlap < 0:
            return json.dumps({"error": f"overlap must be a non-negative integer, got {overlap!r}"})
        if overlap >= chunk_size:
            return json.dumps({"error": f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"})

        try:
            context = self._session_context
            chunks = []
            start = 0

            while start < len(context):
                end = min(start + chunk_size, len(context))
                chunk_text = context[start:end]

                chunks.append({
                    "index": len(chunks),
                    "start": start,
                    "end": end,
                    "length": len(chunk_text),
                })

                start = end - overlap if end < len(context) else end

            return json.dumps({
                "total_chunks": len(chunks),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunks": chunks,
            })
        except Exception as e:
            return json.dumps({"error": f"Partition failed: {e}"})


# ── Composite Engine (LCM + RLM) ────────────────────────────────────────


class CompositeContextEngine(ContextEngine):
    """Combines LCM (compression) and RLM (tools/REPL) into one engine.

    When LCM is enabled for compression, this wrapper adds RLM's
    peek/grep/partition tools alongside LCM's 7 tools. The agent gets
    all 10 tools and LCM handles all compression automatically.

    Usage in run_agent.py:
        lcm_engine = load_context_engine("lcm", config=lcm_cfg)
        rlm_engine = RlmContextEngine(config=rlm_cfg)
        engine = CompositeContextEngine(lcm_engine, rlm_engine)
    """

    def __init__(
        self,
        compression_engine: ContextEngine,
        rlm_engine: Optional[RlmContextEngine] = None,
        **kwargs: Any,
    ) -> None:
        self._compression_engine = compression_engine
        self._rlm_engine = rlm_engine

        # Name reflects which engines are active
        if rlm_engine is not None:
            self._name = f"{compression_engine.name}+rlm"
        else:
            self._name = compression_engine.name

        # Copy token tracking from compression engine
        self.last_prompt_tokens = getattr(compression_engine, "last_prompt_tokens", 0)
        self.last_completion_tokens = getattr(compression_engine, "last_completion_tokens", 0)
        self.last_total_tokens = getattr(compression_engine, "last_total_tokens", 0)
        self.context_length = getattr(compression_engine, "context_length", 0)
        self.threshold_tokens = getattr(compression_engine, "threshold_tokens", 0)
        self.compression_count = getattr(compression_engine, "compression_count", 0)
        self.threshold_percent = getattr(compression_engine, "threshold_percent", 1.0)
        self.protect_last_n = getattr(compression_engine, "protect_last_n", 3)

    @property
    def name(self) -> str:
        return self._name

    # ── Core ABC interface (delegate to compression engine) ────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self._compression_engine.update_from_response(usage)
        # Re-sync wrapper-mirrored token attributes so callers reading
        # composite.last_prompt_tokens etc. see up-to-date values.
        self.last_prompt_tokens = getattr(self._compression_engine, "last_prompt_tokens", 0)
        self.last_completion_tokens = getattr(self._compression_engine, "last_completion_tokens", 0)
        self.last_total_tokens = getattr(self._compression_engine, "last_total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return self._compression_engine.should_compress(prompt_tokens)

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        return self._compression_engine.should_compress_preflight(messages)

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        return self._compression_engine.compress(
            messages, current_tokens, focus_topic=focus_topic
        )

    # ── Session lifecycle (delegate to both) ────────────────────────────

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._compression_engine.on_session_start(session_id, **kwargs)
        if self._rlm_engine:
            # Pass the session messages as context for RLM
            self._rlm_engine.on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._compression_engine.on_session_end(session_id, messages)
        if self._rlm_engine:
            self._rlm_engine.on_session_end(session_id, messages)

    def on_session_reset(self) -> None:
        self._compression_engine.on_session_reset()
        if self._rlm_engine:
            self._rlm_engine.on_session_reset()
        # Re-sync wrapper-mirrored token and counter attributes from the inner
        # compression engine so the next compression decision uses fresh values.
        self.last_prompt_tokens = getattr(self._compression_engine, "last_prompt_tokens", 0)
        self.last_completion_tokens = getattr(self._compression_engine, "last_completion_tokens", 0)
        self.last_total_tokens = getattr(self._compression_engine, "last_total_tokens", 0)
        self.compression_count = getattr(self._compression_engine, "compression_count", 0)

    def update_model(self, **kwargs) -> None:
        """Update model info on both engines and re-sync wrapper-mirrored fields."""
        self._compression_engine.update_model(**kwargs)
        if self._rlm_engine:
            self._rlm_engine.update_model(**kwargs)
        # Re-sync wrapper-mirrored attributes from the compression engine so
        # callers that read composite.context_length / .threshold_tokens etc.
        # get up-to-date values rather than the stale constructor-time copies.
        self.context_length = getattr(self._compression_engine, "context_length", self.context_length)
        self.threshold_tokens = getattr(self._compression_engine, "threshold_tokens", self.threshold_tokens)
        self.threshold_percent = getattr(self._compression_engine, "threshold_percent", self.threshold_percent)
        self.protect_last_n = getattr(self._compression_engine, "protect_last_n", self.protect_last_n)

    # ── Tools (combine from both engines) ───────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return combined tool schemas from LCM and RLM."""
        schemas = list(self._compression_engine.get_tool_schemas())
        if self._rlm_engine:
            schemas.extend(self._rlm_engine.get_tool_schemas())
        return schemas

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch to the appropriate engine based on tool name."""
        if self._rlm_engine and name.startswith("rlm_"):
            try:
                return self._rlm_engine.handle_tool_call(name, args, **kwargs)
            except Exception as e:
                return json.dumps({"error": f"RLM tool '{name}' failed: {e}"})
        else:
            return self._compression_engine.handle_tool_call(name, args, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """Return combined status from both engines."""
        status = self._compression_engine.get_status()
        if self._rlm_engine:
            rlm_status = self._rlm_engine.get_status()
            status["rlm_enabled"] = True
            status["rlm_max_iterations"] = self._rlm_engine._max_iterations
            status["rlm_output_limit"] = self._rlm_engine._output_limit
            status["rlm_context_loaded"] = self._rlm_engine._session_context is not None
        return status

    def refresh_context(self, messages: List[Dict[str, Any]]) -> None:
        """Refresh the RLM context from the latest messages."""
        if self._rlm_engine:
            self._rlm_engine.refresh_context(messages)


# ── Tool Schemas ────────────────────────────────────────────────────────


RLM_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "rlm_peek": {
        "name": "rlm_peek",
        "description": "Peek at a section of the context stored in the RLM REPL environment. Returns a snippet of characters from the context. Use this to understand the structure of the context before deciding how to process it.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "integer",
                    "description": "Start character index (0-based). Default: 0",
                },
                "length": {
                    "type": "integer",
                    "description": "Number of characters to return. Default: 2000",
                },
            },
            "required": [],
        },
    },
    "rlm_grep": {
        "name": "rlm_grep",
        "description": "Grep the context for regex pattern matches. Returns up to max_matches matching lines with line numbers. Use this to find specific IDs, keywords, or patterns in the context without loading the entire context into the model's window.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum matches to return. Default: 20",
                },
            },
            "required": ["pattern"],
        },
    },
    "rlm_partition": {
        "name": "rlm_partition",
        "description": "Partition the context into fixed-size chunks with optional overlap. Returns metadata about each chunk (index, start, end, length). Use this to prepare chunks for parallel processing.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_size": {
                    "type": "integer",
                    "description": "Characters per chunk. Default: 8000",
                    "minimum": 1,
                },
                "overlap": {
                    "type": "integer",
                    "description": "Characters of overlap between chunks. Must be less than chunk_size. Default: 500",
                    "minimum": 0,
                },
            },
            "required": [],
        },
    },
}


# ── Plugin Registration ─────────────────────────────────────────────────


def register(ctx, **kwargs) -> None:
    """Plugin entry point — register RLM as a context engine.

    kwargs are forwarded to the engine constructor (e.g. rlm_config).
    """
    engine = RlmContextEngine(**kwargs)
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)
    else:
        logger.warning("RLM plugin: context engine registration not supported")


# ── Export ──────────────────────────────────────────────────────────────

__all__ = [
    "RlmContextEngine",
    "RLMAgentEnvironment",
    "CompositeContextEngine",
    "RLM_TOOL_SCHEMAS",
    "register",
]
