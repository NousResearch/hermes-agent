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
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from model_tools import handle_function_call

# Thread pool for running sync tool calls that internally use asyncio.run()
# (e.g., mini-swe-agent's modal/docker backends). Running them in a separate
# thread gives them a clean event loop so they don't deadlock inside Atropos's loop.
_tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

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
        self.tool_handler = tool_handler
        self.max_context_tokens = max_context_tokens


    def _truncate_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Truncate conversation history to fit within max_context_tokens.

        Strategy:
        - Keep system message (index 0) and initial user message (index 1) always
        - Keep last 6 messages (recent context) always
        - For everything in between, progressively truncate tool result content
        - If still too long, drop oldest middle messages entirely

        Uses rough char/4 token estimate (fast, no tokenizer needed).
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

        est = estimate_tokens(messages)
        if est <= self.max_context_tokens:
            return messages

        # Phase 1: Truncate tool result content in middle messages
        # Keep first 2 and last 6 messages untouched
        protect_head = 2
        protect_tail = min(6, len(messages) - protect_head)
        middle_start = protect_head
        middle_end = len(messages) - protect_tail

        if middle_start < middle_end:
            # Truncate tool results from oldest first
            for i in range(middle_start, middle_end):
                if messages[i].get("role") == "tool":
                    content = messages[i].get("content", "") or ""
                    if len(content) > 200:
                        messages[i] = dict(messages[i])  # copy
                        messages[i]["content"] = content[:100] + "\n...[truncated]...\n" + content[-50:]

            est = estimate_tokens(messages)
            if est <= self.max_context_tokens:
                logger.debug("Context truncated (phase 1: tool results): %d tokens", est)
                return messages

        # Phase 2: Drop oldest middle messages entirely
        while middle_start < middle_end and estimate_tokens(messages) > self.max_context_tokens:
            # Remove the oldest middle message
            # But keep assistant+tool pairs together
            msg = messages[middle_start]
            messages.pop(middle_start)
            middle_end -= 1
            # If we removed an assistant with tool_calls, also remove matching tool responses
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_ids = {tc.get("id") or tc.get("tool_call_id", "") for tc in msg.get("tool_calls", []) if isinstance(tc, dict)}
                # Remove tool responses for those IDs
                i = middle_start
                while i < middle_end:
                    if messages[i].get("role") == "tool" and messages[i].get("tool_call_id", "") in tool_ids:
                        messages.pop(i)
                        middle_end -= 1
                    else:
                        i += 1

        est = estimate_tokens(messages)
        logger.info("Context truncated (phase 2: dropped messages): %d estimated tokens, %d messages remaining", est, len(messages))
        return messages

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

        for turn in range(self.max_turns):
            # Truncate context if approaching limit
            messages = self._truncate_context(messages)

            # Build the chat_completion kwargs
            chat_kwargs = {
                "messages": messages,
                "n": 1,
                "temperature": self.temperature,
            }

            # Only pass tools if we have them
            if self.tool_schemas:
                chat_kwargs["tools"] = self.tool_schemas

            # Only pass max_tokens if explicitly set
            if self.max_tokens is not None:
                chat_kwargs["max_tokens"] = self.max_tokens

            # Make the API call -- standard OpenAI spec
            try:
                response = await self.server.chat_completion(**chat_kwargs)
            except Exception as e:
                logger.error("API call failed on turn %d: %s", turn + 1, e)
                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=False,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                )

            if not response or not response.choices:
                logger.warning("Empty response on turn %d", turn + 1)
                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=False,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
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
                        # Parse arguments and dispatch
                        try:
                            args = json.loads(tool_args_raw)
                            # Guard against double-encoded JSON strings
                            # Model sometimes outputs '{"command": "ls"}' as a JSON string
                            # so json.loads produces the string '{"command": "ls"}' not a dict
                            if isinstance(args, str):
                                try:
                                    args2 = json.loads(args)
                                    if isinstance(args2, dict):
                                        args = args2
                                    elif isinstance(args2, str):
                                        # Triple-encoded... just wrap it
                                        if tool_name == "terminal":
                                            args = {"command": args2}
                                        else:
                                            args = {"input": args2}
                                    else:
                                        args = {"input": args2}
                                except (json.JSONDecodeError, TypeError):
                                    # Plain string, not JSON - wrap it
                                    if tool_name == "terminal":
                                        args = {"command": args}
                                    else:
                                        args = {"input": args}
                                logger.debug(
                                    "Tool args for '%s' decoded from string: %s",
                                    tool_name, tool_args_raw[:200],
                                )
                        except json.JSONDecodeError:
                            args = {}
                            logger.warning(
                                "Invalid JSON in tool call arguments for '%s': %s",
                                tool_name, tool_args_raw[:200],
                            )

                        try:
                            if tool_name == "terminal":
                                import os
                                backend = os.getenv("TERMINAL_ENV", "local")
                                if self.tool_handler:
                                    backend = "sandbox"
                                cmd_preview = args.get("command", "")[:80]
                                print(f"  🖥️  [{backend}] $ {cmd_preview}")

                            if self.tool_handler:
                                # Use custom tool handler (sandbox backend routing)
                                tool_result = await self.tool_handler(
                                    tool_name, args, self.task_id
                                )
                            else:
                                # Default: run via hermes-agent's handle_function_call
                                # in a thread pool so backends that use asyncio.run()
                                # internally (modal, docker) get a clean event loop
                                # instead of deadlocking inside Atropos's loop.
                                loop = asyncio.get_event_loop()
                                tool_result = await loop.run_in_executor(
                                    _tool_executor,
                                    lambda: handle_function_call(
                                        tool_name, args, task_id=self.task_id
                                    ),
                                )
                        except Exception as e:
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

                        # Also check if the tool returned an error in its JSON result
                        try:
                            result_data = json.loads(tool_result)
                            if isinstance(result_data, dict):
                                err = result_data.get("error")
                                exit_code = result_data.get("exit_code")
                                if err and exit_code and exit_code < 0:
                                    tool_errors.append(ToolError(
                                        turn=turn + 1, tool_name=tool_name,
                                        arguments=tool_args_raw[:200],
                                        error=str(err),
                                        tool_result=tool_result[:500],
                                    ))
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Add tool response to conversation
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )

                logger.debug(
                    "Turn %d: %d tool calls executed",
                    turn + 1,
                    len(assistant_msg.tool_calls),
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

                logger.debug(
                    "Turn %d: model finished naturally (no tool calls)", turn + 1
                )

                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=True,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
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
