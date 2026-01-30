#!/usr/bin/env python3
"""
Atropos-compatible Hermes agent runner.

This is a minimal subclass of Hermes-Agent's `AIAgent` that swaps the OpenAI
function-calling backend for Atroposlib's `ManagedServer`/`ServerManager` backend
and uses Hermes-style XML tool tags:

- <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- <tool_response>{...}</tool_response>

Tool observations are appended as `role="user"` messages containing one or more
`<tool_response>` blocks so they survive common chat templates during tokenization.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from model_tools import cleanup_vm, handle_function_call
from run_agent import AIAgent

_TOOL_CALL_RE = re.compile(r"<tool_call>\\s*(.*?)\\s*</tool_call>", re.DOTALL)


ATROPOS_TOOL_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

## Available Tools
<tools>
{tool_descriptions}
</tools>

## How to Use Tools
To call a tool, output:
<tool_call>{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}</tool_call>

You may include optional reasoning in <think>...</think> before tool calls.

After each tool call, you will receive tool results as:
<tool_response>{{...}}</tool_response>

Continue until finished, then provide a final response with no <tool_call> blocks.
"""


class AtroposAIAgent(AIAgent):
    """
    Hermes `AIAgent` variant that uses Atroposlib ServerManager/ManagedServer.

    Notes:
    - The default Hermes `AIAgent` remains unchanged; this class is opt-in.
    - The underlying server must expose `managed_server(tokenizer=...)` OR be a single
      APIServer-compatible object usable by Atroposlib's `ManagedServer`.
    """

    def __init__(
        self,
        *,
        server: Any,
        tokenizer: Any = None,
        model: str = "local",
        max_iterations: int = 10,
        tool_delay: float = 0.0,
        enabled_toolsets: Optional[List[str]] = None,
        disabled_toolsets: Optional[List[str]] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        ephemeral_system_prompt: Optional[str] = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        # Call parent init mainly to reuse tool selection + trajectory saving utilities.
        super().__init__(
            base_url="http://unused",
            api_key="dummy-key",
            model=model,
            max_iterations=max_iterations,
            tool_delay=tool_delay,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            save_trajectories=save_trajectories,
            verbose_logging=verbose_logging,
            ephemeral_system_prompt=ephemeral_system_prompt,
            log_prefix_chars=log_prefix_chars,
            log_prefix=log_prefix,
        )

        self.server = server
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens

    @asynccontextmanager
    async def _managed(self) -> AsyncGenerator[Any, None]:
        if hasattr(self.server, "managed_server"):
            async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
                yield managed
            return

        # Fall back to directly wrapping a single server object.
        from atroposlib.envs.server_handling.managed_server import ManagedServer

        managed = ManagedServer(server=self.server, tokenizer=self.tokenizer)
        try:
            yield managed
        finally:
            managed.reset()

    def _tool_descriptions_text(self) -> str:
        if not self.tools:
            return "(no tools available)"

        parts: List[str] = []
        for tool in self.tools:
            fn = (tool or {}).get("function", {})
            name = fn.get("name", "")
            desc = (fn.get("description") or "").strip()
            if not name:
                continue
            if desc:
                parts.append(f"- {name}: {desc}")
            else:
                parts.append(f"- {name}")
        return "\n".join(parts) if parts else "(no tools available)"

    def _build_system_prompt(self, system_message: Optional[str]) -> Optional[str]:
        if system_message is not None:
            return system_message
        if self.ephemeral_system_prompt:
            return self.ephemeral_system_prompt
        return ATROPOS_TOOL_SYSTEM_PROMPT.format(
            tool_descriptions=self._tool_descriptions_text()
        )

    def _parse_tool_calls(self, content: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[str]]:
        """
        Returns:
          (calls, errors)
        """
        calls: List[Tuple[str, Dict[str, Any]]] = []
        errors: List[str] = []

        for raw in _TOOL_CALL_RE.findall(content or ""):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON inside <tool_call>: {exc}")
                continue

            name = payload.get("name")
            args = payload.get("arguments", {})
            if not isinstance(name, str) or not name:
                errors.append("Tool call missing 'name' string")
                continue
            if not isinstance(args, dict):
                errors.append("Tool call 'arguments' must be an object")
                continue

            calls.append((name, args))

        return calls, errors

    async def run_conversation_async(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        import uuid

        effective_task_id = task_id or str(uuid.uuid4())

        messages: List[Dict[str, Any]] = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": user_message})

        active_system_prompt = self._build_system_prompt(system_message)

        api_call_count = 0
        final_response: Optional[str] = None
        managed_state: Optional[Dict[str, Any]] = None
        completed = False

        try:
            async with self._managed() as managed:
                while api_call_count < self.max_iterations:
                    api_call_count += 1

                    api_messages = messages.copy()
                    if active_system_prompt:
                        api_messages = [{"role": "system", "content": active_system_prompt}] + api_messages

                    response = await managed.chat_completion(
                        messages=api_messages,
                        n=1,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )

                    if hasattr(managed, "get_state"):
                        managed_state = managed.get_state()

                    assistant_content = response.choices[0].message.content or ""
                    messages.append({"role": "assistant", "content": assistant_content})

                    tool_calls, parse_errors = self._parse_tool_calls(assistant_content)

                    if parse_errors and not tool_calls:
                        # Ask the model to retry with valid tool JSON.
                        err_text = "; ".join(parse_errors[:3])
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"<tool_response>{json.dumps({'error': err_text}, ensure_ascii=False)}</tool_response>\n"
                                    "The previous <tool_call> blocks were invalid. Please output valid JSON inside <tool_call>."
                                ),
                            }
                        )
                        continue

                    if not tool_calls:
                        # No tool calls: treat as final answer.
                        final_response = assistant_content
                        completed = True
                        break

                    tool_responses: List[str] = []
                    for tool_name, tool_args in tool_calls:
                        tool_start = time.time()
                        tool_result = handle_function_call(tool_name, tool_args, effective_task_id)
                        tool_duration = time.time() - tool_start

                        try:
                            parsed = json.loads(tool_result)
                            payload: Any = parsed
                        except Exception:
                            payload = tool_result

                        tool_payload = {
                            "name": tool_name,
                            "duration_s": round(tool_duration, 3),
                            "result": payload,
                        }
                        tool_responses.append(
                            f"<tool_response>{json.dumps(tool_payload, ensure_ascii=False)}</tool_response>"
                        )

                        if self.tool_delay and self.tool_delay > 0:
                            await asyncio.sleep(self.tool_delay)

                    messages.append({"role": "user", "content": "\n".join(tool_responses)})

                if final_response is None:
                    final_response = "I've reached the maximum number of iterations."

        finally:
            try:
                cleanup_vm(effective_task_id)
            except Exception:
                pass

        # Save trajectory using Hermes formatting (optional).
        self._save_trajectory(messages, user_message, completed=completed)

        return {
            "final_response": final_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "managed_state": managed_state,
            "system_prompt": active_system_prompt,
            "task_id": effective_task_id,
        }

    def run_conversation(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Sync wrapper for convenience.

        If already inside an event loop, call `await run_conversation_async(...)` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_conversation_async(*args, **kwargs))
        raise RuntimeError("AtroposAIAgent.run_conversation() cannot be called from a running event loop; use await run_conversation_async().")
