"""
HermesAgent for tau-bench evaluation.

Implements the tau-bench Agent interface using atroposlib's OpenAIServer for
inference — the same server HermesAgentLoop uses in training environments.
Tool execution goes through tau-bench's env.step() so the environment's
state machine, user simulator, and reward computation work correctly.

Usage:
    python environments/benchmarks/taubench/run_eval.py \\
        --model anthropic/claude-sonnet-4-5 --base-url openrouter --env retail
"""

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure hermes-agent repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

logger = logging.getLogger(__name__)


def _message_to_action(message: Dict[str, Any]) -> Action:
    """Convert an OpenAI-format assistant message to a tau-bench Action."""
    tool_calls = message.get("tool_calls")
    if tool_calls and len(tool_calls) > 0 and tool_calls[0].get("function") is not None:
        tc = tool_calls[0]
        return Action(
            name=tc["function"]["name"],
            kwargs=json.loads(tc["function"]["arguments"]),
        )
    return Action(
        name=RESPOND_ACTION_NAME,
        kwargs={"content": message.get("content") or ""},
    )


def _normalize_tool_calls(tool_calls) -> List[Dict[str, Any]]:
    """Normalize tool_calls from OpenAI SDK objects or dicts to plain dicts."""
    if not tool_calls:
        return []
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            result.append({
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                },
            })
        else:
            result.append({
                "id": getattr(tc, "id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
    return result

class HermesAgent(Agent):
    """
    tau-bench Agent using atroposlib's OpenAIServer for inference.

    Uses the same OpenAIServer abstraction as HermesAgentLoop so inference
    goes through the same OpenAI-compatible client used in training.
    Tool execution goes through tau-bench's env.step() — tau-bench tools are
    retail/airline state-machine operations, not hermes tools.

    Args:
        server: atroposlib OpenAIServer instance (from build_server() in run_eval.py).
        system_prompt: Optional system prompt override. Defaults to env.wiki.
        temperature: Sampling temperature.
        max_tokens: Optional token limit per generation step.
        extra_body: Extra params forwarded to chat_completion (e.g. OR preferences).
    """

    def __init__(
        self,
        server,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.server = server
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        """Synchronous entry point required by tau-bench. Runs the async loop."""
        return asyncio.run(self._solve_async(env, task_index, max_num_steps))

    async def _solve_async(
        self,
        env: Env,
        task_index: Optional[int],
        max_num_steps: int,
    ) -> SolveResult:
        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation
        info = env_reset.info.model_dump()
        reward = 0.0

        # Patch the user sim's generate_next_message to guard against None returns.
        # tau-bench's user sim crashes env.step() when the LLM returns None content
        # (e.g. emits a tool call instead of text). This wraps it to return "" instead.
        _orig_gen = env.user.generate_next_message
        def _safe_gen(messages):
            result = _orig_gen(messages)
            return result if result is not None else ""
        env.user.generate_next_message = _safe_gen

        system_content = self.system_prompt if self.system_prompt is not None else env.wiki
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": obs},
        ]

        for step in range(max_num_steps):
            chat_kwargs: Dict[str, Any] = {
                "messages": messages,
                "n": 1,
                "temperature": self.temperature,
                "tools": env.tools_info,
            }
            if self.max_tokens is not None:
                chat_kwargs["max_tokens"] = self.max_tokens
            if self.extra_body:
                chat_kwargs["extra_body"] = self.extra_body

            try:
                response = await self.server.chat_completion(**chat_kwargs)
            except Exception as e:
                logger.error("chat_completion failed on step %d: %s", step, e)
                break

            if not response or not response.choices:
                logger.warning("Empty response on step %d", step)
                break

            assistant_msg_raw = response.choices[0].message

            if hasattr(assistant_msg_raw, "model_dump"):
                assistant_msg = assistant_msg_raw.model_dump()
                # model_dump() returns content=None when absent; normalize to ""
                if assistant_msg.get("content") is None:
                    assistant_msg["content"] = ""
            else:
                assistant_msg = {
                    "role": "assistant",
                    "content": assistant_msg_raw.content or "",
                    "tool_calls": _normalize_tool_calls(
                        getattr(assistant_msg_raw, "tool_calls", None)
                    ),
                }

            action = _message_to_action(assistant_msg)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                tool_calls = assistant_msg.get("tool_calls") or []
                messages.append({
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": tool_calls[:1],
                })
                if tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_calls[0].get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": action.name,
                        "content": env_response.observation,
                    })
                else:
                    messages.append({"role": "user", "content": env_response.observation})
            else:
                messages.append({
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                })
                messages.append({"role": "user", "content": env_response.observation})

            if env_response.done:
                logger.info("Task complete at step %d, reward=%.3f", step + 1, reward)
                break

        # Close the underlying httpx client before the event loop shuts down.
        # Without this, asyncio.run() closes the loop while httpx still has open
        # connections, producing noisy "Event loop is closed" errors on cleanup.
        if hasattr(self.server, "openai") and hasattr(self.server.openai, "close"):
            await self.server.openai.close()

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=None,
        )
