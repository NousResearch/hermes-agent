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
# apply_patches()  # DISABLED: sglang patch breaks native vLLM /generate

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
from model_tools import get_tool_definitions, handle_function_call
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

    # --- Sandbox pool mode (optional, for scaled environments) ---
    tool_pool_mode: str = Field(
        default="default",
        description="Tool execution mode: 'default' (terminal tool per task_id), "
        "'nomad' (slot pool via Nomad/Docker/Singularity), or 'modal' (Modal sandbox pool).",
    )

    # Sandbox pool: shared settings
    allow_network: bool = Field(default=True, description="Whether sandbox bash commands may access the network.")
    require_sandbox: bool = Field(default=False, description="Fail closed if bubblewrap is unavailable.")
    purge_job_on_start: bool = Field(default=False, description="Purge existing sandbox job on startup.")
    purge_job_on_shutdown: bool = Field(default=True, description="Purge sandbox job on shutdown.")
    acquire_timeout_s: float = Field(default=30.0, description="Slot acquisition timeout (seconds).")

    # Sandbox pool: Nomad settings
    nomad_address: str = Field(default="http://localhost:4646", description="Nomad API address.")
    sandbox_job_id: str = Field(default="atropos-sandbox", description="Nomad job id for sandbox containers.")
    sandbox_image: str = Field(default="atropos-sandbox:local", description="Docker image for sandbox containers.")
    slots_per_container: int = Field(default=10, description="Nomad: slots per container.")
    min_containers: int = Field(default=1, description="Nomad: minimum containers.")
    max_containers: int = Field(default=10, description="Nomad: maximum containers.")
    privileged: bool = Field(default=False, description="Nomad: run container privileged.")
    driver: str = Field(default="docker", description="Nomad task driver: 'docker' or 'singularity'.")
    singularity_image: Optional[str] = Field(default=None, description="Path to .sif file for Singularity driver.")

    # Sandbox pool: Modal settings
    modal_app_name: str = Field(default="atropos-sandbox", description="Modal app name prefix.")
    modal_image: str = Field(default="python:3.11", description="Modal: container image.")
    modal_gpu: Optional[str] = Field(default=None, description="Modal: GPU type (None, 'T4', 'A10G', 'A100', 'H100').")
    modal_cpu: float = Field(default=1.0, description="Modal: CPU cores.")
    modal_memory: int = Field(default=2048, description="Modal: memory in MB.")
    modal_slots_per_sandbox: int = Field(default=10, description="Modal: slots per sandbox.")
    modal_min_sandboxes: int = Field(default=1, description="Modal: minimum sandboxes.")
    modal_max_sandboxes: int = Field(default=5, description="Modal: maximum sandboxes.")
    modal_idle_timeout: int = Field(default=120, description="Modal: idle timeout (seconds).")
    modal_max_lifetime: int = Field(default=3600, description="Modal: max sandbox lifetime (seconds).")
    modal_acquire_timeout: float = Field(default=60.0, description="Modal: slot acquisition timeout (seconds).")
    modal_execution_timeout: float = Field(default=30.0, description="Modal: command execution timeout (seconds).")
    modal_secrets: str = Field(default="", description="Modal: comma-separated Modal Secret names.")
    modal_env_vars: str = Field(default="", description="Modal: semicolon-separated KEY=VALUE pairs.")
    modal_workspace_base: str = Field(default="/data", description="Modal: workspace base directory.")


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

        # Sandbox pool backend (only used when tool_pool_mode != "default")
        self._sandbox_backend = None

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
        import sys
        result = self._use_managed_server_inner()
        print(f"HERMES_DEBUG _use_managed_server={result}, servers={len(self.server.servers) if hasattr(self.server, 'servers') else 'N/A'}, type={type(self.server.servers[0]).__name__ if hasattr(self.server, 'servers') and self.server.servers else 'N/A'}", file=sys.stderr, flush=True)
        return result

    def _use_managed_server_inner(self) -> bool:
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
    # Sandbox pool backend (tool_pool_mode != "default")
    # =========================================================================

    async def _start_sandbox_backend(self) -> None:
        """
        Configure the slot pool backend if tool_pool_mode is not 'default'.

        Sets TERMINAL_ENV=slot_pool and configures env vars so that ALL hermes
        tools (terminal, file, etc.) automatically route through the sandbox
        pool via _SlotPoolEnvironment in terminal_tool.py.
        """
        if self.config.tool_pool_mode == "default":
            return

        mode = self.config.tool_pool_mode
        logger.info("Configuring slot pool backend (mode=%s)", mode)

        # Set TERMINAL_ENV=slot_pool so terminal_tool.py uses _SlotPoolEnvironment
        os.environ["TERMINAL_ENV"] = "slot_pool"

        # Set the backend type (modal or nomad)
        if mode == "modal":
            os.environ["TERMINAL_SLOT_BACKEND"] = "modal"
            # Forward modal config from env config to slot pool env vars
            os.environ.setdefault("TERMINAL_MODAL_IMAGE", self.config.modal_image)
            os.environ.setdefault("TERMINAL_MODAL_SLOTS", str(self.config.modal_slots_per_sandbox))
            os.environ.setdefault("TERMINAL_MODAL_MIN", str(self.config.modal_min_sandboxes))
            os.environ.setdefault("TERMINAL_MODAL_MAX", str(self.config.modal_max_sandboxes))
            os.environ.setdefault("TERMINAL_MODAL_IDLE_TIMEOUT", str(self.config.modal_idle_timeout))
            os.environ.setdefault("TERMINAL_MODAL_MAX_LIFETIME", str(self.config.modal_max_lifetime))
            os.environ.setdefault("TERMINAL_MODAL_ACQUIRE_TIMEOUT", str(self.config.modal_acquire_timeout))
            os.environ.setdefault("TERMINAL_MODAL_EXEC_TIMEOUT", str(self.config.modal_execution_timeout))
            os.environ.setdefault("TERMINAL_MODAL_WORKSPACE", self.config.modal_workspace_base)
            if self.config.modal_gpu:
                os.environ.setdefault("TERMINAL_MODAL_GPU", self.config.modal_gpu)
        elif mode == "nomad":
            os.environ["TERMINAL_SLOT_BACKEND"] = "nomad"
            os.environ.setdefault("TERMINAL_NOMAD_ADDRESS", self.config.nomad_address)
            os.environ.setdefault("TERMINAL_NOMAD_IMAGE", self.config.sandbox_image)
            os.environ.setdefault("TERMINAL_NOMAD_DRIVER", self.config.driver)
            os.environ.setdefault("TERMINAL_NOMAD_SLOTS", str(self.config.slots_per_container))
            os.environ.setdefault("TERMINAL_NOMAD_MIN", str(self.config.min_containers))
            os.environ.setdefault("TERMINAL_NOMAD_MAX", str(self.config.max_containers))

        # Eagerly start the _SlotPoolManager so the backend is ready
        # before any trajectories try to use it
        from tools.terminal_tool import _SlotPoolManager
        _SlotPoolManager.get_instance()  # Triggers _start() which creates sandboxes

        self._sandbox_backend = True  # Flag that sandbox mode is active
        print(f"🔧 Slot pool started: TERMINAL_ENV=slot_pool, backend={mode}")

    async def _stop_sandbox_backend(self) -> None:
        """Stop the slot pool backend."""
        if self._sandbox_backend:
            logger.info("Stopping slot pool backend")
            try:
                from tools.terminal_tool import _SlotPoolManager
                _SlotPoolManager.reset_instance()
            except Exception as e:
                logger.warning("Slot pool shutdown: %s", e)
            self._sandbox_backend = None

    # =========================================================================
    # Optional hooks for sandbox environments
    # =========================================================================

    async def setup_trajectory_workspace(
        self,
        item: Item,
        *,
        trajectory_id: str,
        exec_tool,
    ) -> Dict[str, Any]:
        """
        Optional hook: prepare the sandbox workspace before the agent starts.

        Override in subclasses for environments that need workspace setup
        (e.g., git clone, worktree creation, dependency installation).

        Args:
            item: The dataset item being rolled out
            trajectory_id: Unique ID for this trajectory
            exec_tool: Callable to execute tool calls in the sandbox

        Returns:
            Dict of workspace metadata (passed to verify_and_score_trajectory)
        """
        return {}

    async def verify_and_score_trajectory(
        self,
        item: Item,
        result: AgentResult,
        *,
        trajectory_id: str,
        exec_tool,
        workspace_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Optional hook: run in-sandbox verification before scoring.

        Override in subclasses for environments that need to verify results
        inside the sandbox (e.g., run pytest, check file contents).

        Default: calls compute_reward() with ToolContext.

        Args:
            item: The dataset item
            result: The agent's rollout result
            trajectory_id: Unique ID for this trajectory
            exec_tool: Callable to execute tool calls in the sandbox
            workspace_meta: Metadata from setup_trajectory_workspace

        Returns:
            Tuple of (reward, metadata_dict)
        """
        ctx = ToolContext(trajectory_id)
        try:
            reward = await self.compute_reward(item, result, ctx)
        except Exception as e:
            logger.error("compute_reward failed: %s", e)
            reward = 0.0
        finally:
            ctx.cleanup()
        return reward, {}

    # =========================================================================
    # Lifecycle hooks for env_manager/process_manager cleanup
    # =========================================================================

    async def env_manager(self):
        """Start sandbox backend, run env, then clean up."""
        await self._start_sandbox_backend()
        try:
            return await super().env_manager()
        finally:
            await self._stop_sandbox_backend()

    async def process_manager(self):
        """Start sandbox backend, run process, then clean up."""
        await self._start_sandbox_backend()
        try:
            return await super().process_manager()
        finally:
            await self._stop_sandbox_backend()

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

    def _use_sandbox_backend(self) -> bool:
        """Check if we should route tool execution through a sandbox backend."""
        return (
            self.config.tool_pool_mode != "default"
            and self._sandbox_backend is not None
        )

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[Union[ScoredDataItem, Any]], List[Item]]:
        """
        Run a single rollout: agent loop + reward computation.

        This is called group_size times in parallel by collect_trajectories().
        Each call gets its own task_id for terminal/browser session isolation.

        When tool_pool_mode != "default", routes tool execution through the
        sandbox backend (Modal, Nomad) with slot-based multiplexing:
        1. Acquire a slot from the sandbox pool
        2. Setup workspace via subclass hook (e.g., git clone + worktree)
        3. Run agent loop with terminal calls routed through sandbox
        4. Verify and score in-sandbox via subclass hook (e.g., pytest)
        5. Release the slot
        """
        task_id = str(uuid.uuid4())

        # Get group-level tools (resolved once in collect_trajectories)
        if self._current_group_tools is None:
            tools, valid_names = self._resolve_tools_for_group()
        else:
            tools, valid_names = self._current_group_tools

        # Build initial messages
        messages: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": self.format_prompt(item)})

        # Dispatch to the appropriate path
        if self._use_sandbox_backend():
            return await self._collect_trajectory_sandbox(
                item, task_id, tools, valid_names, messages
            )
        else:
            return await self._collect_trajectory_local(
                item, task_id, tools, valid_names, messages
            )

    async def _collect_trajectory_local(
        self,
        item: Item,
        task_id: str,
        tools: List[Dict[str, Any]],
        valid_names: Set[str],
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[Union[ScoredDataItem, Any]], List[Item]]:
        """
        Default (local) trajectory collection path.

        Uses hermes-agent's handle_function_call() for tool execution.
        Reward computed via compute_reward() with ToolContext.
        """
        result = await self._run_agent_loop(
            task_id, tools, valid_names, messages, tool_handler=None
        )

        # Skip reward if the agent loop produced no meaningful work
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
            ctx = ToolContext(task_id)
            try:
                reward = await self.compute_reward(item, result, ctx)
            except Exception as e:
                logger.error("compute_reward failed: %s", e)
                reward = 0.0
            finally:
                ctx.cleanup()

        return self._build_scored_item(item, result, reward)

    async def _collect_trajectory_sandbox(
        self,
        item: Item,
        task_id: str,
        tools: List[Dict[str, Any]],
        valid_names: Set[str],
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[Union[ScoredDataItem, Any]], List[Item]]:
        """
        Sandbox trajectory collection path (Modal, Nomad).

        Uses TERMINAL_ENV=slot_pool so ALL hermes tools (terminal, file, web)
        automatically route through the sandbox pool via _SlotPoolEnvironment.
        No per-tool routing needed — the slot pool is the terminal backend.

        Flow:
        1. Pre-warm terminal env (acquires a slot in the pool)
        2. Setup workspace via subclass hook (e.g., git clone + worktree)
        3. Run agent loop with tool_handler=None (all tools use handle_function_call)
        4. Verify and score in-sandbox via subclass hook (e.g., pytest)
        5. Release the slot via cleanup_vm()
        """
        from tools.terminal_tool import _SlotPoolManager, cleanup_vm
        from dataclasses import dataclass

        @dataclass
        class _ExecResult:
            """Lightweight result for exec_tool compatibility with env hooks."""
            success: bool
            output: str = ""
            error: str = ""
            metadata: Dict[str, Any] = None
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}

        try:
            # 1. Pre-warm: trigger terminal env creation → acquires slot
            logger.info("Pre-warming sandbox slot for task %s", task_id)
            loop = asyncio.get_event_loop()
            warmup = await loop.run_in_executor(
                None,
                lambda: handle_function_call(
                    "terminal", {"command": "echo slot_ready"}, task_id=task_id
                ),
            )
            logger.info("Sandbox slot acquired for task %s", task_id)

            # 2. Create exec_tool for setup/verify hooks
            #    Routes through handle_function_call → terminal_tool → same _SlotPoolEnvironment
            async def exec_tool(tool_name: str, args: Dict[str, Any], timeout: float = 300) -> _ExecResult:
                command = args.get("command", "")
                result_json = await loop.run_in_executor(
                    None,
                    lambda: handle_function_call(
                        "terminal",
                        {"command": command, "timeout": int(timeout)},
                        task_id=task_id,
                    ),
                )
                try:
                    result_dict = json.loads(result_json)
                except (json.JSONDecodeError, TypeError):
                    result_dict = {"output": str(result_json), "exit_code": 1}
                returncode = result_dict.get("exit_code", result_dict.get("returncode", 1))
                output = result_dict.get("output", "")
                return _ExecResult(
                    success=(returncode == 0),
                    output=output,
                    error=result_dict.get("error", "") if returncode != 0 else "",
                    metadata={"returncode": returncode},
                )

            # 3. Setup workspace (subclass hook: git clone, worktree, etc.)
            workspace_meta = await self.setup_trajectory_workspace(
                item, trajectory_id=task_id, exec_tool=exec_tool
            )

            # 4. Run agent loop — tool_handler=None means ALL tools go through
            #    handle_function_call() → terminal_tool() → _SlotPoolEnvironment
            #    → same sandbox slot. File tools also route through same env.
            result = await self._run_agent_loop(
                task_id, tools, valid_names, messages,
                tool_handler=None,
            )

            # 5. Skip verification if no meaningful work
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
                # 6. Verify and score in-sandbox (subclass hook: pytest, etc.)
                reward, score_meta = await self.verify_and_score_trajectory(
                    item, result,
                    trajectory_id=task_id,
                    exec_tool=exec_tool,
                    workspace_meta=workspace_meta,
                )
                logger.info("Sandbox reward for task %s: %.2f", task_id, reward)

            return self._build_scored_item(item, result, reward)

        except Exception as e:
            logger.error("Sandbox trajectory failed for task %s: %s", task_id, e, exc_info=True)
            dummy_result = AgentResult(
                messages=messages, turns_used=0, finished_naturally=False
            )
            return self._build_scored_item(item, dummy_result, 0.0)

        finally:
            # Release the slot back to the pool
            try:
                cleanup_vm(task_id)
                logger.info("Released sandbox slot for task %s", task_id)
            except Exception as e:
                logger.error("Failed to release slot for task %s: %s", task_id, e)

    async def _run_agent_loop(
        self,
        task_id: str,
        tools: List[Dict[str, Any]],
        valid_names: Set[str],
        messages: List[Dict[str, Any]],
        tool_handler=None,
    ) -> AgentResult:
        """
        Run the agent loop in either Phase 1 or Phase 2 mode.

        Shared between local and sandbox paths -- the only difference is
        the tool_handler parameter (None for local, sandbox callable for sandbox).
        """
        if self._use_managed_server():
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
                    # Calculate max prompt tokens
                    # Context budget = max_token_length (prompt can be as long as generation budget)
                    # This ensures prompt + generation stays under typical model context limits
                    # E.g., max_token_length=16384 → 16384 prompt + 16384 gen = 32K < 40960 model limit
                    _max_ctx = None
                    if self.config.max_token_length and self.config.max_token_length > 0:
                        _max_ctx = self.config.max_token_length
                    agent = HermesAgentLoop(
                        server=managed,
                        tool_schemas=tools,
                        valid_tool_names=valid_names,
                        max_turns=self.config.max_agent_turns,
                        task_id=task_id,
                        temperature=self.config.agent_temperature,
                        max_tokens=self.config.max_token_length,
                        tool_handler=tool_handler,
                        max_context_tokens=_max_ctx,
                    )
                    return await agent.run(messages)
            except NotImplementedError:
                logger.warning(
                    "ManagedServer not available (OpenAI server?). "
                    "Falling back to direct server mode."
                )
                _max_ctx = None
                if self.config.max_token_length and self.config.max_token_length > 0:
                    _max_ctx = self.config.max_token_length
                agent = HermesAgentLoop(
                    server=self.server,
                    tool_schemas=tools,
                    valid_tool_names=valid_names,
                    max_turns=self.config.max_agent_turns,
                    task_id=task_id,
                    temperature=self.config.agent_temperature,
                    max_tokens=self.config.max_token_length,
                    tool_handler=tool_handler,
                    max_context_tokens=_max_ctx,
                )
                return await agent.run(messages)
        else:
            _max_ctx = None
            if self.config.max_token_length and self.config.max_token_length > 0:
                _max_ctx = self.config.max_token_length
            agent = HermesAgentLoop(
                server=self.server,
                tool_schemas=tools,
                valid_tool_names=valid_names,
                max_turns=self.config.max_agent_turns,
                task_id=task_id,
                temperature=self.config.agent_temperature,
                max_tokens=self.config.max_token_length,
                tool_handler=tool_handler,
                max_context_tokens=_max_ctx,
            )
            return await agent.run(messages)

    def _build_scored_item(
        self,
        item: Item,
        result: AgentResult,
        reward: float,
    ) -> Tuple[Optional[Union[ScoredDataItem, Any]], List[Item]]:
        """
        Build a ScoredDataItem from an AgentResult and reward.

        Shared between local and sandbox paths.
        """
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
        nodes = (result.managed_state or {}).get("nodes", [])

        if nodes:
            node = nodes[-1]
            scored_item: Dict[str, Any] = {
                "tokens": node.tokens,
                "masks": node.masked_tokens,
                "scores": reward,
            }
            if hasattr(node, "logprobs") and node.logprobs:
                scored_item["advantages"] = None
                scored_item["ref_logprobs"] = None
        else:
            full_text = "\n".join(
                msg.get("content", "") for msg in result.messages if msg.get("content")
            )
            if self.tokenizer:
                tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            else:
                tokens = list(range(min(len(full_text) // 4, 128)))

            scored_item = {
                "tokens": tokens,
                "masks": [-100] + tokens[1:],
                "scores": reward,
            }

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
