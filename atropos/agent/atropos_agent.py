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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv

from ..tools import ToolCall, ToolRegistry, ToolResult
from atroposlib.envs.server_handling.managed_server import ManagedServer

load_dotenv()


# Default system prompt with tool calling instructions
AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools. You can use tools to accomplish tasks.

## Available Tools
<tools>
{tool_descriptions}
</tools>

## How to Use Tools
To use a tool, output a tool call in the following format:
<tool_call>{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}</tool_call>

You may reason about what to do before calling a tool:
<think>I need to check what files are in the current directory...</think>
<tool_call>{{"name": "bash", "arguments": {{"command": "ls -la"}}}}</tool_call>

After a tool is executed, you will receive the result:
<tool_response>{{"success": true, "output": "..."}}</tool_response>

Continue using tools as needed until you have completed the task.
When you have finished, provide your final response without any tool calls.

## Important Guidelines
- Think step by step about what you need to do
- Use tools to gather information and perform actions
- If a tool call fails, analyze the error and try a different approach
- Provide clear, concise responses when the task is complete
"""


@dataclass
class AgentConfig:
    """Configuration for the AtroposAgent."""
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    
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
        
        tool_descriptions = self.tools.get_prompt_description()
        if not tool_descriptions:
            tool_descriptions = "(No tools available)"
        
        return AGENT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
    
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
        
        # Use ManagedServer for automatic token tracking
        async with self._managed() as managed:
            for step_num in range(self.config.max_steps):
                try:
                    # Keep a copy of the prompt messages used for this completion.
                    # Useful for reconstructing tokens/masks when state tracking is unavailable.
                    prompt_messages = list(messages)
                    response = await managed.chat_completion(
                        messages=messages,
                        n=1,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
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
                
                response_text = response.choices[0].message.content or ""
                tool_calls = ToolCall.parse_from_text(response_text)
                
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
                return AgentResult(
                    success=False,
                    final_response=final_response,
                    steps=steps,
                    error=f"Reached maximum steps ({self.config.max_steps})",
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
            response = await managed.chat_completion(
                messages=messages,
                n=1,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            current_node = None
            if hasattr(managed, "get_state"):
                state = managed.get_state()
                nodes = state.get("nodes", [])
                current_node = nodes[-1] if nodes else None
        
        response_text = response.choices[0].message.content or ""
        tool_results = []
        
        if execute_tools:
            tool_calls = ToolCall.parse_from_text(response_text)
            for call in tool_calls:
                result = await self.execute_tool(call)
                tool_results.append(result)
        
        sequence_data = SequenceData.from_sequence_node(current_node) if current_node else None
        
        return response_text, tool_results, sequence_data
