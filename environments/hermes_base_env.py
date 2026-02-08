"""
HermesAgentBaseEnv -- Abstract Base Environment for Hermes-Agent + Atropos

Provides the Atropos integration plumbing that all hermes-agent environments share:
- Two-mode operation (OpenAI server for Phase 1, VLLM ManagedServer for Phase 2)
- Per-group toolset/distribution resolution
- Agent loop orchestration via HermesAgentLoop
- ToolContext creation for reward functions
- ScoredDataGroup construction from ManagedServer state

Subclasses only need to implement:
    setup()           -- Load dataset, initialize state
    get_next_item()   -- Return the next item from the dataset
    format_prompt()   -- Convert a dataset item into the user message
    compute_reward()  -- Score the rollout (has full ToolContext access)
    evaluate()        -- Periodic evaluation
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure the hermes-agent repo root is on sys.path so that imports like
# `from model_tools import ...` and `from environments.X import ...` work
# regardless of where the script is invoked from.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv
from pydantic import Field

# Load API keys from hermes-agent/.env so all environments can access them
_env_path = _repo_root / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)

# Apply monkey patches for async-safe tool operation inside Atropos's event loop.
# This patches SwerexModalEnvironment to use a background thread instead of
# asyncio.run(), which would deadlock inside Atropos. Safe for normal CLI too.
from environments.patches import apply_patches
apply_patches()

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.envs.server_handling.server_manager import (
    APIServerConfig,
    ServerBaseline,
    ServerManager,
)
from atroposlib.type_definitions import Item

from environments.agent_loop import AgentResult, HermesAgentLoop
from environments.tool_context import ToolContext

# Import hermes-agent toolset infrastructure
from model_tools import get_tool_definitions
from toolset_distributions import sample_toolsets_from_distribution

logger = logging.getLogger(__name__)


class HermesAgentEnvConfig(BaseEnvConfig):
    """
    Configuration for hermes-agent Atropos environments.

    Extends BaseEnvConfig with agent-specific settings for toolsets,
    terminal backend, dataset loading, and tool call parsing.
    """

    # --- Toolset configuration ---
    # Mutually exclusive: use either enabled_toolsets OR distribution
    enabled_toolsets: Optional[List[str]] = Field(
        default=None,
        description="Explicit list of hermes toolsets to enable (e.g., ['terminal', 'file', 'web']). "
        "If None and distribution is also None, all available toolsets are enabled.",
    )
    disabled_toolsets: Optional[List[str]] = Field(
        default=None,
        description="Toolsets to disable. Applied as a filter on top of enabled_toolsets or distribution.",
    )
    distribution: Optional[str] = Field(
        default=None,
        description="Name of a toolset distribution from toolset_distributions.py "
        "(e.g., 'development', 'terminal_tasks'). Sampled once per group. "
        "Mutually exclusive with enabled_toolsets.",
    )

    # --- Agent loop configuration ---
    max_agent_turns: int = Field(
        default=30,
        description="Maximum number of LLM calls (tool-calling iterations) per rollout.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for the agent. Tools are handled via the tools= parameter, "
        "not embedded in the prompt text.",
    )
    agent_temperature: float = Field(
        default=1.0,
        description="Sampling temperature for agent generation during rollouts.",
    )

    # --- Terminal backend ---
    terminal_backend: str = Field(
        default="local",
        description="Terminal backend: 'local', 'docker', 'modal', 'ssh', 'singularity'. "
        "Modal recommended for production RL (cloud isolation per rollout).",
    )

    # --- Dataset ---
    dataset_name: Optional[str] = Field(
        default=None,
        description="HuggingFace dataset name. Optional if tasks are defined inline.",
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split to use.",
    )
    prompt_field: str = Field(
        default="prompt",
        description="Which field in the dataset contains the prompt.",
    )

    # --- Phase 2: Tool call parsing ---
    tool_call_parser: str = Field(
        default="hermes",
        description="Tool call parser name for Phase 2 (VLLM server type). "
        "Ignored in Phase 1 (OpenAI server type where VLLM parses natively). "
        "Options: hermes, mistral, llama3_json, qwen, deepseek_v3, etc.",
    )


class HermesAgentBaseEnv(BaseEnv):
    """
    Abstract base environment for hermes-agent Atropos integration.

    Handles two modes of operation:
    - Phase 1 (OpenAI server type): Uses server.chat_completion() directly.
      The server (VLLM, SGLang, OpenRouter, OpenAI) handles tool call parsing
      and reasoning extraction natively. DummyManagedServer provides placeholder
      tokens. Good for SFT data gen, verifier testing, evaluation.

    - Phase 2 (VLLM server type): Uses ManagedServer for exact token IDs + logprobs
      via /generate. Client-side tool call parser reconstructs structured tool_calls
      from raw output. Full RL training capability.

    Subclasses must implement:
        setup()           -- Load dataset, initialize state
        get_next_item()   -- Return the next item to roll out
        format_prompt()   -- Convert a dataset item into the user message string
        compute_reward()  -- Score the rollout using ToolContext
        evaluate()        -- Periodic evaluation
    """

    name: Optional[str] = "hermes-agent"
    env_config_cls = HermesAgentEnvConfig

    def __init__(
        self,
        config: HermesAgentEnvConfig,
        server_configs: Union[ServerBaseline, List[APIServerConfig]],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        # Set terminal backend environment variable so hermes tools pick it up
        if config.terminal_backend:
            os.environ["TERMINAL_ENV"] = config.terminal_backend
            print(f"🖥️  Terminal backend: {config.terminal_backend}")

        # Current group's resolved tools (set in collect_trajectories)
        self._current_group_tools: Optional[Tuple[List[Dict], Set[str]]] = None

        # Tool error tracking for wandb logging
        self._tool_error_buffer: List[Dict[str, Any]] = []

    # =========================================================================
    # Toolset resolution (per-group)
    # =========================================================================

    def _resolve_tools_for_group(self) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """
        Resolve toolsets for a group. Called once in collect_trajectories(),
        then shared by all collect_trajectory() calls in the group.

        If distribution is set, samples probabilistically.
        If enabled_toolsets is set, uses that explicit list.
        disabled_toolsets is applied as a filter on top.

        Returns:
            (tool_schemas, valid_tool_names) tuple
        """
        config = self.config

        if config.distribution:
            group_toolsets = sample_toolsets_from_distribution(config.distribution)
            logger.info("Sampled toolsets from '%s': %s", config.distribution, group_toolsets)
        else:
            group_toolsets = config.enabled_toolsets  # None means "all available"

        tools = get_tool_definitions(
            enabled_toolsets=group_toolsets,
            disabled_toolsets=config.disabled_toolsets,
            quiet_mode=True,
        )

        valid_names = {t["function"]["name"] for t in tools} if tools else set()
        logger.info("Resolved %d tools for group: %s", len(valid_names), sorted(valid_names))
        return tools, valid_names

    # =========================================================================
    # Server mode detection
    # =========================================================================

    def _use_managed_server(self) -> bool:
        """
        Determine if we should use ManagedServer (Phase 2) or direct server (Phase 1).

        Phase 2 (ManagedServer) is used when the server type is 'vllm' or 'sglang',
        which go through the /generate endpoint for exact token tracking.

        Phase 1 (direct server) is used for 'openai' server type, which uses
        /v1/chat/completions with native tool call parsing.
        """
        if not self.server.servers:
            return False

        server = self.server.servers[0]
        # If the server is an OpenAI server (not VLLM/SGLang), use direct mode
        from atroposlib.envs.server_handling.openai_server import OpenAIServer
        return not isinstance(server, OpenAIServer)

    # =========================================================================
    # Core Atropos integration
    # =========================================================================

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[
        Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]],
        List[Item],
    ]:
        """
        Override collect_trajectories to resolve toolsets once per group,
        then delegate to the standard group-level collection.

        The default BaseEnv.collect_trajectories() calls collect_trajectory()
        group_size times in parallel. We resolve tools once here and store
        them for all those calls to use.
        """
        # Resolve toolsets for this group (shared by all rollouts in the group)
        self._current_group_tools = self._resolve_tools_for_group()

        # Delegate to the default implementation which calls collect_trajectory()
        # group_size times via asyncio.gather
        return await super().collect_trajectories(item)

    # =========================================================================
    # Wandb rollout display -- format trajectories nicely
    # =========================================================================

    @staticmethod
    def _format_trajectory_for_display(messages: List[Dict[str, Any]]) -> str:
        """
        Format a conversation's messages into a readable trajectory string
        for wandb rollout tables. Shows tool calls, tool results, and reasoning
        in a structured way instead of raw token decoding.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"[SYSTEM]\n{content}")

            elif role == "user":
                parts.append(f"[USER]\n{content}")

            elif role == "assistant":
                # Show reasoning if present
                reasoning = msg.get("reasoning_content", "")
                if reasoning:
                    # Truncate long reasoning for display
                    if len(reasoning) > 300:
                        reasoning = reasoning[:300] + "..."
                    parts.append(f"[ASSISTANT thinking]\n{reasoning}")

                # Show content
                if content:
                    parts.append(f"[ASSISTANT]\n{content}")

                # Show tool calls
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")
                    # Truncate long arguments for display
                    if len(args) > 200:
                        args = args[:200] + "..."
                    parts.append(f"[TOOL CALL] {name}({args})")

            elif role == "tool":
                tool_id = msg.get("tool_call_id", "")
                result = content
                # Truncate long tool results for display
                if len(result) > 500:
                    result = result[:500] + "..."
                parts.append(f"[TOOL RESULT] {result}")

        return "\n\n".join(parts)

    async def add_rollouts_for_wandb(
        self,
        scored_data,
        item=None,
    ):
        """
        Override to show formatted trajectories with tool calls visible,
        instead of raw token decoding which loses all structure.
        """
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        group = []
        for i in range(min(num_keep, len(scored_data.get("scores", [])))):
            score = scored_data["scores"][i]

            # Use messages if available for rich display
            messages = None
            if scored_data.get("messages") and i < len(scored_data["messages"]):
                messages = scored_data["messages"][i]

            if messages:
                text = self._format_trajectory_for_display(messages)
            elif scored_data.get("tokens") and i < len(scored_data["tokens"]):
                text = self.tokenizer.decode(scored_data["tokens"][i])
            else:
                text = "(no data)"

            group.append((text, score))

        self.rollouts_for_wandb.append(group)
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log base metrics including tool errors to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log tool error stats
        if self._tool_error_buffer:
            wandb_metrics["train/tool_errors_count"] = len(self._tool_error_buffer)

            # Log error details as a summary string (tables can crash wandb on tmp cleanup)
            error_summaries = []
            for err in self._tool_error_buffer:
                error_summaries.append(
                    f"[turn {err['turn']}] {err['tool']}({err['args'][:80]}) -> {err['error'][:150]}"
                )
            wandb_metrics["train/tool_error_details"] = "\n".join(error_summaries)

            # Also print to stdout for immediate visibility
            for summary in error_summaries:
                print(f"  Tool Error: {summary}")

            self._tool_error_buffer = []
        else:
            wandb_metrics["train/tool_errors_count"] = 0

        await super().wandb_log(wandb_metrics)

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[Union[ScoredDataItem, Any]], List[Item]]:
        """
        Run a single rollout: agent loop + reward computation.

        This is called group_size times in parallel by collect_trajectories().
        Each call gets its own task_id for terminal/browser session isolation.
        """
        task_id = str(uuid.uuid4())

        # Get group-level tools (resolved once in collect_trajectories)
        if self._current_group_tools is None:
            # Fallback: resolve per-trajectory if called outside collect_trajectories
            tools, valid_names = self._resolve_tools_for_group()
        else:
            tools, valid_names = self._current_group_tools

        # Build initial messages
        messages: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": self.format_prompt(item)})

        # Run the agent loop
        result: AgentResult
        if self._use_managed_server():
            # Phase 2: ManagedServer with parser -- exact tokens + logprobs
            # Load the tool call parser from registry based on config
            from environments.tool_call_parsers import get_parser
            try:
                tc_parser = get_parser(self.config.tool_call_parser)
            except KeyError:
                logger.warning(
                    "Tool call parser '%s' not found, falling back to 'hermes'",
                    self.config.tool_call_parser,
                )
                tc_parser = get_parser("hermes")

            try:
                async with self.server.managed_server(
                    tokenizer=self.tokenizer,
                    tool_call_parser=tc_parser,
                ) as managed:
                    agent = HermesAgentLoop(
                        server=managed,
                        tool_schemas=tools,
                        valid_tool_names=valid_names,
                        max_turns=self.config.max_agent_turns,
                        task_id=task_id,
                        temperature=self.config.agent_temperature,
                        max_tokens=self.config.max_token_length,
                    )
                    result = await agent.run(messages)
            except NotImplementedError:
                # DummyManagedServer not allowed -- fall back to Phase 1
                logger.warning(
                    "ManagedServer not available (OpenAI server?). "
                    "Falling back to direct server mode."
                )
                agent = HermesAgentLoop(
                    server=self.server,
                    tool_schemas=tools,
                    valid_tool_names=valid_names,
                    max_turns=self.config.max_agent_turns,
                    task_id=task_id,
                    temperature=self.config.agent_temperature,
                    max_tokens=self.config.max_token_length,
                )
                result = await agent.run(messages)
        else:
            # Phase 1: OpenAI server -- native tool_calls, placeholder tokens
            agent = HermesAgentLoop(
                server=self.server,
                tool_schemas=tools,
                valid_tool_names=valid_names,
                max_turns=self.config.max_agent_turns,
                task_id=task_id,
                temperature=self.config.agent_temperature,
                max_tokens=self.config.max_token_length,
            )
            result = await agent.run(messages)

        # Skip reward computation if the agent loop produced no meaningful work
        # (e.g., API call failed on turn 1). No point spinning up a Modal sandbox
        # just to verify files that were never created.
        only_system_and_user = all(
            msg.get("role") in ("system", "user") for msg in result.messages
        )
        if result.turns_used == 0 or only_system_and_user:
            logger.warning(
                "Agent loop produced no output (turns=%d, msgs=%d). Skipping reward.",
                result.turns_used, len(result.messages),
            )
            reward = 0.0
        else:
            # Compute reward using ToolContext (gives verifier full tool access)
            ctx = ToolContext(task_id)
            try:
                reward = await self.compute_reward(item, result, ctx)
            except Exception as e:
                logger.error("compute_reward failed: %s", e)
                reward = 0.0
            finally:
                ctx.cleanup()

        # Track tool errors for wandb logging
        if result.tool_errors:
            for err in result.tool_errors:
                self._tool_error_buffer.append({
                    "turn": err.turn,
                    "tool": err.tool_name,
                    "args": err.arguments[:150],
                    "error": err.error[:300],
                    "result": err.tool_result[:300],
                })

        # Build ScoredDataItem from ManagedServer state
        # Phase 2: real tokens/masks/logprobs from SequenceNodes
        # Phase 1: placeholder tokens (still need a valid ScoredDataItem for the pipeline)
        nodes = (result.managed_state or {}).get("nodes", [])

        if nodes:
            # Phase 2 (or DummyManagedServer): use actual node data
            node = nodes[-1]  # Final sequence node = full trajectory
            scored_item: Dict[str, Any] = {
                "tokens": node.tokens,
                "masks": node.masked_tokens,
                "scores": reward,
            }

            # Include logprobs if available (Phase 2)
            if hasattr(node, "logprobs") and node.logprobs:
                scored_item["advantages"] = None  # Computed by trainer
                scored_item["ref_logprobs"] = None
        else:
            # Phase 1 with no managed state: create placeholder tokens
            # so the data pipeline doesn't break. These are NOT suitable
            # for training but allow process mode (SFT data gen) to work.
            # Tokenize the full conversation to get approximate tokens.
            full_text = "\n".join(
                msg.get("content", "") for msg in result.messages if msg.get("content")
            )
            if self.tokenizer:
                tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            else:
                tokens = list(range(min(len(full_text) // 4, 128)))

            scored_item = {
                "tokens": tokens,
                "masks": [-100] + tokens[1:],  # Mask first token as prompt
                "scores": reward,
            }

        # Always include messages for wandb rollout display and data logging
        scored_item["messages"] = result.messages

        return scored_item, []

    # =========================================================================
    # Abstract methods -- subclasses must implement
    # =========================================================================

    @abstractmethod
    async def setup(self):
        """
        Load dataset, initialize state.

        Called once when the environment starts. Typical implementation:
            self.dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
            self.iter = 0
        """
        raise NotImplementedError

    @abstractmethod
    async def get_next_item(self) -> Item:
        """
        Return the next item from the dataset for rollout.

        Called by the base env's main loop to get items for workers.
        Should cycle through the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def format_prompt(self, item: Item) -> str:
        """
        Convert a dataset item into the user message for the agent.

        Args:
            item: Dataset item (dict, tuple, etc.)

        Returns:
            The prompt string to send to the agent
        """
        raise NotImplementedError

    @abstractmethod
    async def compute_reward(
        self, item: Item, result: AgentResult, ctx: ToolContext
    ) -> float:
        """
        Score the rollout. Has full access to:
        - item: the original dataset item (ground truth, test commands, etc.)
        - result: AgentResult with full messages, turn count, reasoning, etc.
        - ctx: ToolContext -- call ANY hermes-agent tool (terminal, file, web,
               browser, vision...) scoped to this rollout's sandbox. Nothing
               is off-limits.

        Args:
            item: The dataset item that was rolled out
            result: The agent's rollout result
            ctx: ToolContext with full tool access for verification

        Returns:
            Reward float (typically 0.0 to 1.0, but any float is valid)
        """
        raise NotImplementedError

    @abstractmethod
    async def evaluate(self, *args, **kwargs):
        """
        Periodic evaluation. Called every steps_per_eval steps.

        Typical implementation runs the agent on a held-out eval set
        and logs metrics via wandb/evaluate_log.
        """
        raise NotImplementedError
