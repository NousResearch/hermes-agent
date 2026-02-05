"""
ReACT-style agent implementation for atropos-agent.

This module provides the core AtroposAgent class that implements a basic
Reason-Act-Observe loop with tool calling capabilities.

Uses ManagedServer from atroposlib for automatic token/logprob tracking,
making trajectories ready for RL training.

The agent uses Hermes-style XML tags for tool calls:
- <think>...</think> for reasoning
- <tool_call>{"name": "...", "arguments": {...}}</tool_call> for actions
- <tool_response>...</tool_response> for observations
"""

import asyncio
import os
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv
import httpx

from ..tools import ToolCall, ToolRegistry, ToolResult
from atroposlib.envs.server_handling.managed_server import ManagedServer

load_dotenv()


# Default system prompt with tool calling instructions.
#
# IMPORTANT: In training-mode environments we want "raw text in -> raw text out" and we
# parse tool calls from completion text. Do not rely on server-specific `tool_calls` fields.
AGENT_SYSTEM_PROMPT = """You are a deep thinking AI. You MUST enclose your internal reasoning inside <think>...</think> tags.

You are a function calling AI model.

You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.

After calling & executing a function, you will be provided with function results within <tool_response></tool_response> XML tags.

Here are the available tools:
<tools>
{tools_json}
</tools>

Use the following JSON schema for each tool call you will make:
{"title": "FunctionCall", "type": "object", "properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, "required": ["name", "arguments"]}

## REQUIRED TOOL FORMAT

When you decide to call a tool, your assistant message MUST be:
1) exactly one <think>...</think> block, followed by
2) one or more <tool_call>...</tool_call> blocks,
and NOTHING else in that message.

If you need to explain anything, put it inside <think>. Do NOT write natural language outside <think> or <tool_call>.

For each function call return a JSON object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": "<function-name>", "arguments": {"arg1": "value1"}}
</tool_call>

Each <tool_call> must be on its own and contain ONLY the JSON object (no extra text).
The JSON inside <tool_call> MUST be valid JSON with double quotes.

Do NOT output <tool_response> in an assistant message.

After you receive tool results, you may either call more tools (same required format) or provide the final answer.
When providing the final answer, do NOT include any <tool_call> blocks.

## TERMINAL TOOL NOTES

- Commands execute under POSIX `/bin/sh` (not bash).
- Each tool call runs in a fresh shell: environment changes (like `cd` or venv activation) do not persist across tool calls.
- Avoid bash-only features like `source`, `[[ ... ]]`, or process substitution.
- Prefer explicit venv usage:
  - `python -m venv .venv && . .venv/bin/activate && python -m pip install -e .` (POSIX `.` activation), or
  - `.venv/bin/python -m pip install -e .` (no activation required).

## ICL (examples)

User: Show the current directory.
Assistant:
<think>I should run pwd.</think>
<tool_call>
{"name": "terminal", "arguments": {"command": "pwd"}}
</tool_call>
User: <tool_response>{"success": true, "output": "/tmp\\n"}</tool_response>
Assistant: /tmp

User: List files, then count them.
Assistant:
<think>I should count files.</think>
<tool_call>
{"name": "terminal", "arguments": {"command": "ls -1 | wc -l"}}
</tool_call>
User: <tool_response>{"success": true, "output": "3\\n"}</tool_response>
Assistant: 3

User: Run pwd, then print ok (two tool calls).
Assistant:
<think>I should run two commands.</think>
<tool_call>
{"name": "terminal", "arguments": {"command": "pwd"}}
</tool_call>
<tool_call>
{"name": "terminal", "arguments": {"command": "echo ok"}}
</tool_call>
User: <tool_response>{"success": true, "output": "/tmp\\n"}</tool_response>
User: <tool_response>{"success": true, "output": "ok\\n"}</tool_response>
Assistant: ok
"""


@dataclass
class AgentConfig:
    """Configuration for the AtroposAgent."""
    
    # Generation parameters
    temperature: Optional[float] = 0.7
    # Default to "let the backend decide" (important for tool-tag completions that may be longer).
    max_tokens: Optional[int] = None
    
    # Agent behavior
    max_steps: int = 50
    system_prompt: Optional[str] = None
    tool_delay_s: float = 0.0
    
    # Working directory for tools
    working_dir: Optional[str] = None


@dataclass
class SequenceData:
    """Token/logprob data from a single completion."""
    
    full_text: str
    tokens: List[int]
    masked_tokens: List[int]  # -100 for prompt, actual IDs for completion
    logprobs: List[float]  # 1.0 for prompt, actual values for completion
    
    @classmethod
    def from_sequence_node(cls, node) -> "SequenceData":
        """Create from a ManagedServer SequenceNode."""
        return cls(
            full_text=node.full_text,
            tokens=node.tokens,
            masked_tokens=node.masked_tokens,
            logprobs=node.logprobs,
        )


@dataclass
class AgentStep:
    """A single step in the agent's trajectory."""
    
    step_number: int
    assistant_message: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    sequence_data: Optional[SequenceData] = None  # Token data from this step
    
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class AgentResult:
    """Result of running an agent trajectory."""
    
    success: bool
    final_response: str
    steps: List[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Full trajectory token data for RL training
    trajectory_data: Optional[SequenceData] = None
    
    @property
    def num_steps(self) -> int:
        return len(self.steps)
    
    @property
    def total_tool_calls(self) -> int:
        return sum(len(step.tool_calls) for step in self.steps)
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert trajectory to messages format for logging."""
        messages = []
        for step in self.steps:
            messages.append({"role": "assistant", "content": step.assistant_message})
            if step.tool_results:
                # Combine all tool responses
                responses = "\n".join(r.to_xml() for r in step.tool_results)
                messages.append({"role": "user", "content": responses})
        return messages
    
    def to_scored_data(self, score: float) -> Optional[Dict[str, Any]]:
        """
        Convert to format suitable for ScoredDataGroup.
        
        Args:
            score: The score for this trajectory
            
        Returns:
            Dict with tokens, masks, scores suitable for training, or None if no data
        """
        if self.trajectory_data is None:
            return None
        
        return {
            "tokens": self.trajectory_data.tokens,
            "masks": self.trajectory_data.masked_tokens,
            "scores": score,
            "logprobs": self.trajectory_data.logprobs,
        }


class AtroposAgent:
    """
    A ReACT-style agent that uses LLMs with tool calling.
    
    This implementation wraps ManagedServer for automatic token/logprob tracking,
    making trajectories ready for RL training.
    
    Example:
        # `server` may be an Atropos `ServerManager` (recommended) or a single `APIServer`.
        # In practice, environments usually construct this via `BaseEnv`.
        server = ...
        tools = ToolRegistry()
        tools.register(BashTool())
        
        agent = AtroposAgent(server=server, tools=tools)
        result = await agent.run("List the files in the current directory")
        
        # Access token data for training
        if result.trajectory_data:
            print(f"Tokens: {result.trajectory_data.tokens}")
            print(f"Masked: {result.trajectory_data.masked_tokens}")
    """
    
    def __init__(
        self,
        server,  # ServerManager or APIServer
        tools: Optional[ToolRegistry] = None,
        config: Optional[AgentConfig] = None,
        tokenizer: Optional[Any] = None,
        execute_tool: Optional[Callable[[ToolCall], Awaitable[ToolResult]]] = None,
    ):
        self.server = server
        self.tools = tools or ToolRegistry()
        self.config = config or AgentConfig()
        self.tokenizer = tokenizer or getattr(server, "tokenizer", None)
        self.execute_tool = execute_tool or self.tools.execute

    @asynccontextmanager
    async def _managed(self) -> AsyncGenerator[Any, None]:
        """
        Yield a ManagedServer-like object.

        - If `self.server` is a ServerManager, use its `managed_server()` context manager.
        - If `self.server` is a single APIServer, wrap it in `ManagedServer` directly.
        """
        if os.getenv("ATROPOS_BYPASS_MANAGED_SERVER") == "1":
            yield _DirectChatCompletionClient(server=self.server)
            return
        if hasattr(self.server, "managed_server"):
            async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
                yield managed
        else:
            managed = ManagedServer(server=self.server, tokenizer=self.tokenizer)
            try:
                yield managed
            finally:
                managed.reset()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        if self.config.system_prompt:
            return self.config.system_prompt

        tools_json = self.tools.get_prompt_tool_definitions_json()
        # Avoid `str.format()` here because the prompt contains many literal `{}` braces
        # in JSON examples; we only want to substitute the single `{tools_json}` token.
        return AGENT_SYSTEM_PROMPT.replace("{tools_json}", tools_json)

    def _infer_server_model_for_debug(self) -> Optional[str]:
        """
        Best-effort inference of the configured model name for debug payload saving.

        ManagedServer/server_manager typically injects `model` internally, so `chat_kwargs`
        may not contain it. For replaying saved payloads via curl, it's useful to persist it.
        """
        servers = getattr(self.server, "servers", None)
        if isinstance(servers, list) and servers:
            s0 = servers[0]
            cfg = getattr(s0, "config", None)
            model = getattr(cfg, "model_name", None) or getattr(s0, "model_name", None)
            if isinstance(model, str) and model:
                return model
        model = getattr(self.server, "model_name", None) or getattr(self.server, "model", None)
        if isinstance(model, str) and model:
            return model
        return None

    def _debug_dump_request(self, *, step_num: int, chat_kwargs: Dict[str, Any]) -> None:
        if os.getenv("ATROPOS_DEBUG_AGENT_REQUEST") != "1":
            return
        try:
            # Avoid dumping megabytes by default; messages can be huge.
            meta = {
                "step": step_num,
                "chat_kwargs_keys": sorted(list(chat_kwargs.keys())),
                "n": chat_kwargs.get("n"),
                "max_tokens": chat_kwargs.get("max_tokens"),
                "temperature": chat_kwargs.get("temperature"),
                "num_messages": len(chat_kwargs.get("messages") or []),
            }
            print("\n=== ATROPOS_DEBUG_AGENT_REQUEST ===", flush=True)
            print(meta, flush=True)

            if os.getenv("ATROPOS_DEBUG_AGENT_REQUEST_FULL") == "1":
                payload = dict(chat_kwargs)
                # Make the payload more legible and less huge.
                try:
                    dumped = json.dumps(payload, ensure_ascii=False, indent=2)
                except Exception:
                    dumped = repr(payload)
                print("\n=== ATROPOS_DEBUG_AGENT_REQUEST_FULL ===", flush=True)
                print(dumped[:200_000], flush=True)

            # Optional: save the FULL request payload to disk (no truncation).
            save_dir = os.getenv("ATROPOS_DEBUG_AGENT_REQUEST_SAVE_DIR")
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                payload: Dict[str, Any] = dict(chat_kwargs)
                if "model" not in payload:
                    model = self._infer_server_model_for_debug()
                    if model:
                        payload["model"] = model
                # Use a unique filename so parallel trajectories don't clobber each other.
                fname = os.path.join(
                    save_dir,
                    f"atropos_agent_request_step{step_num}_{int(time.time()*1000)}_{os.getpid()}_{uuid4().hex}.json",
                )
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"[AtroposAgent] saved request payload: {fname}", flush=True)
        except Exception:
            return

    def _debug_dump_response(self, *, step_num: int, response: Any) -> None:
        if os.getenv("ATROPOS_DEBUG_AGENT_RESPONSE") != "1":
            return
        print("\n=== ATROPOS_DEBUG_AGENT_RESPONSE ===", flush=True)
        print({"step": step_num, "type": type(response).__name__}, flush=True)
        try:
            dumped = response.model_dump()  # openai pydantic model
        except Exception:
            dumped = getattr(response, "__dict__", {"repr": repr(response)})
        # Keep the dump bounded; we only need enough to see the assistant message content.
        text = str(dumped)
        print(text[:200_000], flush=True)

    async def _chat_completion_with_debug(
        self, *, managed: Any, step_num: int, chat_kwargs: Dict[str, Any]
    ) -> Any:
        """
        Call `managed.chat_completion()` with optional timeout + richer failure logging.

        Debug env vars:
        - `ATROPOS_AGENT_CHAT_TIMEOUT_S`: if set, wraps the await in `asyncio.wait_for`.
        - `ATROPOS_DEBUG_AGENT_WAIT_EVERY_S`: if set, prints a heartbeat while waiting.
        """
        timeout_s_raw = os.getenv("ATROPOS_AGENT_CHAT_TIMEOUT_S")
        timeout_s = float(timeout_s_raw) if timeout_s_raw else None

        wait_every_raw = os.getenv("ATROPOS_DEBUG_AGENT_WAIT_EVERY_S")
        wait_every_s = float(wait_every_raw) if wait_every_raw else None

        async def _await_call() -> Any:
            if not wait_every_s or wait_every_s <= 0:
                return await managed.chat_completion(**chat_kwargs)

            # Heartbeat mode: wait in chunks without cancelling the underlying request.
            # NOTE: do NOT use `asyncio.wait_for(task, timeout=...)` here, because a timeout
            # will cancel the task and surface as `CancelledError` on the next loop.
            task = asyncio.create_task(managed.chat_completion(**chat_kwargs))
            t0 = time.perf_counter()
            try:
                while True:
                    done, _pending = await asyncio.wait({task}, timeout=wait_every_s)
                    if task in done:
                        return task.result()

                    waited = time.perf_counter() - t0
                    print(
                        f"[AtroposAgent] step={step_num} still waiting for chat_completion... ({waited:.1f}s)",
                        flush=True,
                    )
            except asyncio.CancelledError:
                task.cancel()
                raise

        try:
            if timeout_s and timeout_s > 0:
                return await asyncio.wait_for(_await_call(), timeout=timeout_s)
            return await _await_call()
        except Exception as e:
            detail: Dict[str, Any] = {
                "step": step_num,
                "exc_type": type(e).__name__,
                "exc_str": str(e),
            }
            if isinstance(e, httpx.HTTPStatusError):
                try:
                    detail["status_code"] = e.response.status_code
                    detail["response_text"] = e.response.text[:20_000]
                except Exception:
                    pass
            elif isinstance(e, httpx.RequestError):
                detail["request"] = repr(getattr(e, "request", None))

            print("\n=== ATROPOS_DEBUG_AGENT_CHAT_FAILURE ===", flush=True)
            print(detail, flush=True)
            raise

    async def run(
        self,
        task: str,
        initial_messages: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResult:
        """
        Run the agent on a task using ManagedServer for token tracking.
        
        Args:
            task: The task/prompt for the agent
            initial_messages: Optional additional context messages
            
        Returns:
            AgentResult with the trajectory, final response, and token data
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        
        if initial_messages:
            messages.extend(initial_messages)
        
        messages.append({"role": "user", "content": task})
        
        steps = []
        final_response = ""
        final_node = None
        final_prompt_messages: Optional[List[Dict[str, str]]] = None
        last_node = None
        last_prompt_messages: Optional[List[Dict[str, str]]] = None
        last_response_text: str = ""
        
        # Use ManagedServer for automatic token tracking
        async with self._managed() as managed:
            for step_num in range(self.config.max_steps):
                # ReACT loop iteration here, just call -> tools -> observe until done (no tools called)
                try:
                    # Keep a copy of the prompt messages used for this completion.
                    # Useful for reconstructing tokens/masks when state tracking is unavailable.
                    prompt_messages = list(messages)
                    chat_kwargs: Dict[str, Any] = {"messages": messages, "n": 1}
                    if self.config.max_tokens is not None:
                        chat_kwargs["max_tokens"] = self.config.max_tokens
                    if self.config.temperature is not None:
                        chat_kwargs["temperature"] = self.config.temperature

                    t_req = time.perf_counter()
                    print(
                        f"[AtroposAgent] step={step_num+1} chat_completion start "
                        f"(messages={len(messages)}, max_tokens={self.config.max_tokens}, temp={self.config.temperature})",
                        flush=True,
                    )
                    self._debug_dump_request(step_num=step_num + 1, chat_kwargs=chat_kwargs)
                    response = await self._chat_completion_with_debug(
                        managed=managed, step_num=step_num + 1, chat_kwargs=chat_kwargs
                    )
                    self._debug_dump_response(step_num=step_num + 1, response=response)
                    print(
                        f"[AtroposAgent] step={step_num+1} chat_completion done in {time.perf_counter() - t_req:.2f}s",
                        flush=True,
                    )
                    
                    current_node = None
                    if hasattr(managed, "get_state"):
                        state = managed.get_state()
                        nodes = state.get("nodes", [])
                        current_node = nodes[-1] if nodes else None
                    
                except Exception as e:
                    return AgentResult(
                        success=False,
                        final_response="",
                        steps=steps,
                        error=f"Generation error: {str(e)}",
                    )
                
                msg = response.choices[0].message
                # Some OpenAI-compatible servers populate `message.reasoning` and leave `content=""`.
                response_text = (msg.content or "") or (getattr(msg, "reasoning", None) or "")
                tool_calls = ToolCall.parse_from_text(response_text)
                last_node = current_node
                last_prompt_messages = prompt_messages
                last_response_text = response_text
                
                step = AgentStep(
                    step_number=step_num + 1,
                    assistant_message=response_text,
                    tool_calls=tool_calls,
                    sequence_data=SequenceData.from_sequence_node(current_node) if current_node else None,
                )
                
                if not tool_calls:
                    steps.append(step)
                    final_response = response_text
                    final_node = current_node
                    final_prompt_messages = prompt_messages
                    break
                
                messages.append({"role": "assistant", "content": response_text})
                
                tool_responses = []
                for call in tool_calls:
                    result = await self.execute_tool(call)
                    step.tool_results.append(result)
                    tool_responses.append(result.to_xml())
                    if self.config.tool_delay_s > 0:
                        await asyncio.sleep(self.config.tool_delay_s)
                
                steps.append(step)
            
                responses_text = "\n".join(tool_responses)
                # Tool observations are represented as user content with Hermes-style tags.
                # This is compatible with most OpenAI-compatible chat APIs and ensures
                # tokenizers/chat templates include tool outputs during training.
                messages.append({"role": "user", "content": responses_text})
            
            else:
                # Reached max steps without completing
                # Return a failure result but include the last observed completion so callers can
                # record the trajectory (score=0) without triggering retries.
                final_response = last_response_text or final_response
                final_node = last_node
                final_prompt_messages = last_prompt_messages
                trajectory_data = None
                if final_node:
                    trajectory_data = SequenceData.from_sequence_node(final_node)
                elif final_prompt_messages is not None and self.tokenizer is not None:
                    if hasattr(self.tokenizer, "apply_chat_template"):
                        prompt_text = self.tokenizer.apply_chat_template(
                            final_prompt_messages, tokenize=False, add_generation_prompt=True
                        )
                        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                    else:
                        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in final_prompt_messages])
                        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
                    output_tokens = self.tokenizer.encode(final_response, add_special_tokens=False)
                    tokens = prompt_tokens + output_tokens
                    masked_tokens = ([-100] * len(prompt_tokens)) + output_tokens
                    logprobs = ([1.0] * len(prompt_tokens)) + ([0.0] * len(output_tokens))
                    trajectory_data = SequenceData(
                        full_text=f"{prompt_text}{final_response}",
                        tokens=tokens,
                        masked_tokens=masked_tokens,
                        logprobs=logprobs,
                    )
                return AgentResult(
                    success=False,
                    final_response=final_response,
                    steps=steps,
                    error=f"Reached maximum steps ({self.config.max_steps})",
                    trajectory_data=trajectory_data,
                )
        
        # Build result with trajectory data
        trajectory_data = None
        if final_node:
            trajectory_data = SequenceData.from_sequence_node(final_node)
        elif final_prompt_messages is not None and self.tokenizer is not None:
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt_text = self.tokenizer.apply_chat_template(
                    final_prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            else:
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in final_prompt_messages])
                prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            output_tokens = self.tokenizer.encode(final_response, add_special_tokens=False)
            tokens = prompt_tokens + output_tokens
            masked_tokens = ([-100] * len(prompt_tokens)) + output_tokens
            logprobs = ([1.0] * len(prompt_tokens)) + ([0.0] * len(output_tokens))
            trajectory_data = SequenceData(
                full_text=f"{prompt_text}{final_response}",
                tokens=tokens,
                masked_tokens=masked_tokens,
                logprobs=logprobs,
            )
        
        return AgentResult(
            success=True,
            final_response=final_response,
            steps=steps,
            trajectory_data=trajectory_data,
        )
    
    async def run_single_turn(
        self,
        messages: List[Dict[str, str]],
        execute_tools: bool = True,
    ) -> tuple[str, List[ToolResult], Optional[SequenceData]]:
        """
        Run a single turn of the agent (one LLM call + tool execution).
        
        This is useful for integration with BaseEnv where you want more
        control over the loop.
        
        Args:
            messages: The conversation history
            execute_tools: Whether to execute parsed tool calls
            
        Returns:
            Tuple of (response_text, tool_results, sequence_data)
        """
        async with self._managed() as managed:
            chat_kwargs: Dict[str, Any] = {"messages": messages, "n": 1}
            if self.config.max_tokens is not None:
                chat_kwargs["max_tokens"] = self.config.max_tokens
            if self.config.temperature is not None:
                chat_kwargs["temperature"] = self.config.temperature

            self._debug_dump_request(step_num=1, chat_kwargs=chat_kwargs)
            response = await self._chat_completion_with_debug(managed=managed, step_num=1, chat_kwargs=chat_kwargs)
            self._debug_dump_response(step_num=1, response=response)
            
            current_node = None
            if hasattr(managed, "get_state"):
                state = managed.get_state()
                nodes = state.get("nodes", [])
                current_node = nodes[-1] if nodes else None
        
        msg = response.choices[0].message
        response_text = (msg.content or "") or (getattr(msg, "reasoning", None) or "")
        tool_results = []
        
        if execute_tools:
            tool_calls = ToolCall.parse_from_text(response_text)
            for call in tool_calls:
                result = await self.execute_tool(call)
                tool_results.append(result)
        
        sequence_data = SequenceData.from_sequence_node(current_node) if current_node else None
        
        return response_text, tool_results, sequence_data


class _DirectChatCompletionClient:
    """
    Minimal stand-in for ManagedServer that calls the OpenAI-compatible endpoint directly.

    This is for isolating issues where `ManagedServer.chat_completion()` hangs or misbehaves.
    It intentionally does NOT do token/logprob tracking.
    """

    def __init__(self, server: Any):
        self._server = server

    def _server_config(self) -> tuple[str, str, str]:
        # ServerManager case: first configured server.
        servers = getattr(self._server, "servers", None)
        if isinstance(servers, list) and servers:
            s0 = servers[0]
            cfg = getattr(s0, "config", None)
            base_url = getattr(cfg, "base_url", None) or getattr(s0, "base_url", None)
            api_key = getattr(cfg, "api_key", None) or getattr(s0, "api_key", None)
            model = getattr(cfg, "model_name", None) or getattr(s0, "model_name", None)
            if isinstance(base_url, str) and isinstance(api_key, str) and isinstance(model, str):
                return base_url.rstrip("/"), api_key, model

        # APIServer-like fallback.
        base_url = getattr(self._server, "base_url", None)
        api_key = getattr(self._server, "api_key", None)
        model = getattr(self._server, "model_name", None) or getattr(self._server, "model", None)
        if isinstance(base_url, str) and isinstance(api_key, str) and isinstance(model, str):
            return base_url.rstrip("/"), api_key, model

        raise RuntimeError("Unable to resolve server base_url/api_key/model for direct chat completion")

    async def chat_completion(self, *, messages: List[Dict[str, str]], n: int = 1, **kwargs: Any) -> Any:
        base_url, api_key, model = self._server_config()
        url = f"{base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "n": n,
        }
        # Pass through common generation kwargs.
        for k in ("max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty", "stop"):
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        timeout_s = float(os.getenv("ATROPOS_DIRECT_REQUEST_TIMEOUT_S") or "120")
        print(f"[AtroposAgent] DIRECT chat_completion POST {url} (timeout={timeout_s}s)", flush=True)
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Return a very small object compatible with the code paths that read
        # `response.choices[0].message.content`.
        class _Msg:
            def __init__(self, d: Dict[str, Any]):
                self.content = d.get("content")
                self.reasoning = d.get("reasoning")

        class _Choice:
            def __init__(self, d: Dict[str, Any]):
                self.message = _Msg(d.get("message") or {})

        class _Resp:
            def __init__(self, d: Dict[str, Any]):
                self._d = d
                self.choices = [_Choice(c) for c in (d.get("choices") or [])]

            def model_dump(self) -> Dict[str, Any]:
                return self._d

        return _Resp(data)
