"""
HermesAgentLoop -- Reusable Multi-Turn Agent Engine

Runs the hermes-agent tool-calling loop using standard OpenAI-spec tool calling.
Works with any server that returns ChatCompletion objects with tool_calls:
    - Phase 1: OpenAI server type (VLLM, SGLang, OpenRouter, OpenAI API)
    - Phase 2: ManagedServer with client-side tool call parser

The loop passes tools= and checks response.choices[0].message.tool_calls,
identical to hermes-agent's run_agent.py. Tool execution is dispatched via
handle_function_call() from model_tools.py.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from model_tools import handle_function_call

# Thread pool for running sync tool calls that internally use asyncio.run()
# (e.g., mini-swe-agent's modal/docker backends). Running them in a separate
# thread gives them a clean event loop so they don't deadlock inside Atropos's loop.
# Size must be large enough for concurrent eval tasks (e.g., 89 TB2 tasks all
# making tool calls). Too small = thread pool starvation, tasks queue for minutes.
# Resized at runtime by HermesAgentBaseEnv.__init__ via resize_tool_pool().
_tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=128)


def resize_tool_pool(max_workers: int):
    """
    Replace the global tool executor with a new one of the given size.

    Called by HermesAgentBaseEnv.__init__ based on config.tool_pool_size.
    Safe to call before any tasks are submitted.
    """
    global _tool_executor
    _tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    logger.info("Tool thread pool resized to %d workers", max_workers)

logger = logging.getLogger(__name__)


@dataclass
class ToolError:
    """Record of a tool execution error during the agent loop."""

    turn: int                  # Which turn the error occurred on
    tool_name: str             # Which tool was called
    arguments: str             # The arguments passed (truncated)
    error: str                 # The error message
    tool_result: str           # The raw result returned to the model


@dataclass
class AgentResult:
    """Result of running the agent loop."""

    # Full conversation history in OpenAI message format
    messages: List[Dict[str, Any]]
    # ManagedServer.get_state() if available (Phase 2), None otherwise
    managed_state: Optional[Dict[str, Any]] = None
    # How many LLM calls were made
    turns_used: int = 0
    # True if model stopped calling tools naturally (vs hitting max_turns)
    finished_naturally: bool = False
    # Extracted reasoning content per turn (from PR #297 helpers)
    reasoning_per_turn: List[Optional[str]] = field(default_factory=list)
    # Tool errors encountered during the loop
    tool_errors: List[ToolError] = field(default_factory=list)

    # Tool-call metrics (debugging / optional reward shaping)
    tool_calls_attempted: int = 0
    tool_calls_schema_valid: int = 0
    tool_calls_executed_ok: int = 0
    tool_calls_exec_error: int = 0


def _extract_reasoning_from_message(message) -> Optional[str]:
    """
    Extract reasoning content from a ChatCompletion message.

    Handles multiple provider formats:
    1. message.reasoning_content field (some providers)
    2. message.reasoning field (some providers)
    3. message.reasoning_details[].text (OpenRouter style)

    Note: <think> block extraction from content is NOT done here -- that's
    handled by the response already in Phase 1 (server does it) or by
    ManagedServer's patch in Phase 2.

    Args:
        message: The assistant message from ChatCompletion response

    Returns:
        Extracted reasoning text, or None if not found
    """
    # Check reasoning_content field (common across providers)
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content

    # Check reasoning field
    if hasattr(message, "reasoning") and message.reasoning:
        return message.reasoning

    # Check reasoning_details (OpenRouter style)
    if hasattr(message, "reasoning_details") and message.reasoning_details:
        for detail in message.reasoning_details:
            if hasattr(detail, "text") and detail.text:
                return detail.text
            if isinstance(detail, dict) and detail.get("text"):
                return detail["text"]

    return None


class HermesAgentLoop:
    """
    Runs hermes-agent's tool-calling loop using standard OpenAI-spec tool calling.

    Same pattern as run_agent.py:
    - Pass tools= to the API
    - Check response.choices[0].message.tool_calls
    - Dispatch via handle_function_call()

    Works identically with any server type -- OpenAI, VLLM, SGLang, OpenRouter,
    or ManagedServer with a parser. The server determines how tool_calls get
    populated on the response.
    """

    def __init__(
        self,
        server,
        tool_schemas: List[Dict[str, Any]],
        valid_tool_names: Set[str],
        max_turns: int = 30,
        task_id: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        tool_handler=None,
        max_context_tokens: Optional[int] = None,
    ):
        """
        Initialize the agent loop.

        Args:
            server: Server object with chat_completion() method (OpenAIServer,
                    ManagedServer, ServerManager, etc.)
            tool_schemas: OpenAI-format tool definitions from get_tool_definitions()
            valid_tool_names: Set of tool names the model is allowed to call
            max_turns: Maximum number of LLM calls before stopping
            task_id: Unique ID for terminal/browser session isolation
            temperature: Sampling temperature for generation
            max_tokens: Max tokens per generation (None for server default)
            extra_body: Extra parameters passed to the OpenAI client's create() call.
                        Used for OpenRouter provider preferences, transforms, etc.
                        e.g. {"provider": {"ignore": ["DeepInfra"]}}
            tool_handler: Optional async callable(tool_name, args, task_id) -> str.
                         When provided, used INSTEAD of handle_function_call() for
                         tool dispatch. This allows sandbox backends (Modal, Nomad)
                         to route tool calls through their slot-based execution.
            max_context_tokens: Maximum prompt tokens before truncation.
                               If None, no truncation is applied.
                               Recommended: set to max_model_len - max_tokens - 512 (safety margin).
        """
        self.server = server
        self.tool_schemas = tool_schemas
        self.valid_tool_names = valid_tool_names
        self.max_turns = max_turns
        self.task_id = task_id or str(uuid.uuid4())
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.tool_handler = tool_handler
        self.max_context_tokens = max_context_tokens

    def _truncate_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Truncate conversation history to fit within max_context_tokens.

        Strategy:
        - Keep system message (index 0) and initial user message (index 1) always
        - Keep last 6 messages (recent context) always
        - For everything in between, progressively truncate tool result content
        - If still too long, drop oldest middle messages entirely

        Uses rough char/4 token estimate (fast, no tokenizer needed).

        NOTE: This function mutates the provided list (it may pop/replace entries).
        Call it on a copy when you want to preserve the full trajectory.
        """
        if self.max_context_tokens is None:
            return messages

        def estimate_tokens(msgs):
            total = 0
            for m in msgs:
                content = m.get("content", "") or ""
                total += len(content) // 4 + 10  # ~4 chars per token + overhead
                if "tool_calls" in m:
                    total += 50 * len(m["tool_calls"])  # tool call overhead
            return total

        if estimate_tokens(messages) <= self.max_context_tokens:
            return messages

        protect_head = 2
        protect_tail = max(0, min(6, len(messages) - protect_head))
        middle_start = protect_head
        middle_end = len(messages) - protect_tail

        # Phase 1: truncate tool outputs in the middle
        if middle_start < middle_end:
            for i in range(middle_start, middle_end):
                if messages[i].get("role") == "tool":
                    content = messages[i].get("content", "") or ""
                    if len(content) > 200:
                        messages[i] = dict(messages[i])
                        messages[i]["content"] = content[:100] + "\n...[truncated]...\n" + content[-50:]

            if estimate_tokens(messages) <= self.max_context_tokens:
                return messages

        # Phase 2: drop oldest middle messages (try to keep assistant+tool pairs)
        while middle_start < middle_end and estimate_tokens(messages) > self.max_context_tokens:
            msg = messages[middle_start]
            messages.pop(middle_start)
            middle_end -= 1

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_ids = {
                    tc.get("id") or tc.get("tool_call_id", "")
                    for tc in msg.get("tool_calls", [])
                    if isinstance(tc, dict)
                }
                i = middle_start
                while i < middle_end:
                    if messages[i].get("role") == "tool" and messages[i].get("tool_call_id", "") in tool_ids:
                        messages.pop(i)
                        middle_end -= 1
                    else:
                        i += 1

        return messages

    def _normalize_tool_args(self, tool_name: str, tool_args_raw: str) -> (Dict[str, Any], bool):
        """Normalize tool arguments into a dict.

        Returns: (args_dict, schema_valid)

        schema_valid is True only when arguments decode directly into a dict
        (no double-decoding and no coercion/wrapping required).

        Goal: keep environments robust (never crash on args format drift) while
        still allowing reward functions to penalize malformed formats if desired.
        """
        try:
            decoded = json.loads(tool_args_raw)
        except json.JSONDecodeError:
            # Not JSON at all — treat as a plain string
            if tool_name == "terminal":
                return {"command": tool_args_raw}, False
            return {"input": tool_args_raw}, False

        if isinstance(decoded, dict):
            if tool_name == "terminal":
                cmd = decoded.get("command")
                if isinstance(cmd, str) and cmd.strip():
                    return decoded, True
                if isinstance(decoded.get("input"), str):
                    return {"command": decoded.get("input")}, False
                return decoded, False
            return decoded, True

        if isinstance(decoded, str):
            s = decoded.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    decoded2 = json.loads(s)
                except json.JSONDecodeError:
                    decoded2 = None
                if isinstance(decoded2, dict):
                    return decoded2, False

            if tool_name == "terminal":
                return {"command": decoded}, False
            return {"input": decoded}, False

        if tool_name == "terminal":
            return {"command": str(decoded)}, False
        return {"input": decoded}, False

    async def run(self, messages: List[Dict[str, Any]]) -> AgentResult:
        """
        Execute the full agent loop using standard OpenAI tool calling.

        Args:
            messages: Initial conversation messages (system + user).
                      Modified in-place as the conversation progresses.

        Returns:
            AgentResult with full conversation history, managed state, and metadata
        """
        reasoning_per_turn = []
        tool_errors: List[ToolError] = []

        tool_calls_attempted = 0
        tool_calls_schema_valid = 0
        tool_calls_executed_ok = 0
        tool_calls_exec_error = 0

        import time as _time

        for turn in range(self.max_turns):
            turn_start = _time.monotonic()

            # Truncate prompt view on a copy (preserve full trajectory in `messages`)
            prompt_messages = self._truncate_context(list(messages))

            # Build the chat_completion kwargs
            chat_kwargs = {
                "messages": prompt_messages,
                "n": 1,
                "temperature": self.temperature,
            }

            # Only pass tools if we have them
            if self.tool_schemas:
                chat_kwargs["tools"] = self.tool_schemas

            # Only pass max_tokens if explicitly set
            if self.max_tokens is not None:
                chat_kwargs["max_tokens"] = self.max_tokens

            # Inject extra_body for provider-specific params (e.g., OpenRouter
            # provider preferences like banned/preferred providers, transforms)
            if self.extra_body:
                chat_kwargs["extra_body"] = self.extra_body

            # Make the API call -- standard OpenAI spec
            api_start = _time.monotonic()
            try:
                response = await self.server.chat_completion(**chat_kwargs)
            except Exception as e:
                api_elapsed = _time.monotonic() - api_start
                logger.error("API call failed on turn %d (%.1fs): %s", turn + 1, api_elapsed, e)
                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=False,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                    tool_calls_attempted=tool_calls_attempted,
                    tool_calls_schema_valid=tool_calls_schema_valid,
                    tool_calls_executed_ok=tool_calls_executed_ok,
                    tool_calls_exec_error=tool_calls_exec_error,
                )

            api_elapsed = _time.monotonic() - api_start

            if not response or not response.choices:
                logger.warning("Empty response on turn %d (api=%.1fs)", turn + 1, api_elapsed)
                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=False,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                    tool_calls_attempted=tool_calls_attempted,
                    tool_calls_schema_valid=tool_calls_schema_valid,
                    tool_calls_executed_ok=tool_calls_executed_ok,
                    tool_calls_exec_error=tool_calls_exec_error,
                )

            assistant_msg = response.choices[0].message

            # Extract reasoning content from the response (all provider formats)
            reasoning = _extract_reasoning_from_message(assistant_msg)
            reasoning_per_turn.append(reasoning)

            # Check for tool calls -- standard OpenAI spec
            if assistant_msg.tool_calls:
                # Build the assistant message dict for conversation history
                msg_dict: Dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ],
                }

                # Preserve reasoning_content for multi-turn chat template handling
                # (e.g., Kimi-K2's template renders <think> blocks differently
                # for history vs. the latest turn based on this field)
                if reasoning:
                    msg_dict["reasoning_content"] = reasoning

                messages.append(msg_dict)

                # Execute each tool call via hermes-agent's dispatch
                for tc in assistant_msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args_raw = tc.function.arguments

                    # Validate tool name
                    if tool_name not in self.valid_tool_names:
                        tool_calls_exec_error += 1
                        tool_result = json.dumps(
                            {
                                "error": f"Unknown tool '{tool_name}'. "
                                f"Available tools: {sorted(self.valid_tool_names)}"
                            }
                        )
                        tool_errors.append(ToolError(
                            turn=turn + 1, tool_name=tool_name,
                            arguments=tool_args_raw[:200],
                            error=f"Unknown tool '{tool_name}'",
                            tool_result=tool_result,
                        ))
                        logger.warning(
                            "Model called unknown tool '%s' on turn %d",
                            tool_name, turn + 1,
                        )
                    else:
                        tool_calls_attempted += 1
                        args, schema_valid = self._normalize_tool_args(tool_name, tool_args_raw)
                        if schema_valid:
                            tool_calls_schema_valid += 1

                        try:
                            if tool_name == "terminal":
                                backend = os.getenv("TERMINAL_ENV", "local")
                                cmd_preview = str(args.get("command", ""))[:80]
                                logger.info(
                                    "[%s] $ %s", self.task_id[:8], cmd_preview,
                                )

                            tool_submit_time = _time.monotonic()

                            if self.tool_handler:
                                tool_result = await self.tool_handler(tool_name, args, self.task_id)
                            else:
                                # Run tool calls in a thread pool so backends that use
                                # asyncio.run() internally (modal, docker) get a clean
                                # event loop instead of deadlocking inside Atropos's loop.
                                loop = asyncio.get_event_loop()
                                tool_result = await loop.run_in_executor(
                                    _tool_executor,
                                    lambda: handle_function_call(
                                        tool_name, args, task_id=self.task_id
                                    ),
                                )

                            tool_elapsed = _time.monotonic() - tool_submit_time

                            # Log slow tools and thread pool stats for debugging
                            pool_active = _tool_executor._work_queue.qsize()
                            if tool_elapsed > 30:
                                logger.warning(
                                    "[%s] turn %d: %s took %.1fs (pool queue=%d)",
                                    self.task_id[:8], turn + 1, tool_name,
                                    tool_elapsed, pool_active,
                                )
                        except Exception as e:
                            tool_calls_exec_error += 1
                            tool_result = json.dumps(
                                {"error": f"Tool execution failed: {type(e).__name__}: {str(e)}"}
                            )
                            tool_errors.append(ToolError(
                                turn=turn + 1, tool_name=tool_name,
                                arguments=tool_args_raw[:200],
                                error=f"{type(e).__name__}: {str(e)}",
                                tool_result=tool_result,
                            ))
                            logger.error(
                                "Tool '%s' execution failed on turn %d: %s",
                                tool_name, turn + 1, e,
                            )
                        else:
                            tool_err = False
                            try:
                                result_data = json.loads(tool_result)
                                if isinstance(result_data, dict):
                                    err = result_data.get("error")
                                    if err:
                                        tool_err = True

                                    exit_code = result_data.get("exit_code")
                                    if exit_code is not None and isinstance(exit_code, int) and exit_code < 0:
                                        tool_err = True
                                        tool_errors.append(ToolError(
                                            turn=turn + 1, tool_name=tool_name,
                                            arguments=tool_args_raw[:200],
                                            error=str(err) if err else "nonzero exit_code",
                                            tool_result=tool_result[:500],
                                        ))
                            except (json.JSONDecodeError, TypeError):
                                pass

                            if tool_err:
                                tool_calls_exec_error += 1
                            else:
                                tool_calls_executed_ok += 1

                    # Add tool response to conversation
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )

                turn_elapsed = _time.monotonic() - turn_start
                logger.info(
                    "[%s] turn %d: api=%.1fs, %d tools, turn_total=%.1fs",
                    self.task_id[:8], turn + 1, api_elapsed,
                    len(assistant_msg.tool_calls), turn_elapsed,
                )

            else:
                # No tool calls -- model is done
                msg_dict = {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                }
                if reasoning:
                    msg_dict["reasoning_content"] = reasoning
                messages.append(msg_dict)

                turn_elapsed = _time.monotonic() - turn_start
                logger.info(
                    "[%s] turn %d: api=%.1fs, no tools (finished), turn_total=%.1fs",
                    self.task_id[:8], turn + 1, api_elapsed, turn_elapsed,
                )

                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=True,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                    tool_calls_attempted=tool_calls_attempted,
                    tool_calls_schema_valid=tool_calls_schema_valid,
                    tool_calls_executed_ok=tool_calls_executed_ok,
                    tool_calls_exec_error=tool_calls_exec_error,
                )

        # Hit max turns without the model stopping
        logger.info("Agent hit max_turns (%d) without finishing", self.max_turns)
        return AgentResult(
            messages=messages,
            managed_state=self._get_managed_state(),
            turns_used=self.max_turns,
            finished_naturally=False,
            reasoning_per_turn=reasoning_per_turn,
            tool_errors=tool_errors,
            tool_calls_attempted=tool_calls_attempted,
            tool_calls_schema_valid=tool_calls_schema_valid,
            tool_calls_executed_ok=tool_calls_executed_ok,
            tool_calls_exec_error=tool_calls_exec_error,
        )

    def _get_managed_state(self) -> Optional[Dict[str, Any]]:
        """
        Get ManagedServer state if the server supports it.

        Returns state dict with SequenceNodes containing tokens/logprobs/masks,
        or None if the server doesn't support get_state() (e.g., regular OpenAI server).
        """
        if hasattr(self.server, "get_state"):
            return self.server.get_state()
        return None
