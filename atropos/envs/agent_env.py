"""
AgentEnv - Atropos BaseEnv extension for agent/tool-call workloads.

AgentEnv is responsible for starting the sandbox tool execution backend and
providing helpers for running agent trajectories with queued/batched tool calls.

For Phase 4 we support a Nomad-backed `SlotPool` for true container sandboxing.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from pydantic import Field

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, Item, ScoredDataGroup, ScoredDataItem

from ..agent import AgentConfig, AtroposAgent
from ..slots import SlotPool, SlotPoolConfig
from ..tools import BashTool, ReadFileTool, ToolRegistry, WriteFileTool
from ..tools.image_generation_tool import ImageGenerateTool
from ..tools.mixture_of_agents_tool import MixtureOfAgentsTool
from ..tools.terminal_tool import TerminalTool
from ..tools.terminal_stateful_tool import TerminalStatefulTool
from ..tools.tmux_tool import TmuxTool
from ..tools.toolsets import resolve_multiple_toolsets
from ..tools.vision_tools import VisionAnalyzeTool
from ..tools.web_tools import WebCrawlTool, WebExtractTool, WebSearchTool
from ..tools.tool_executor import ToolExecutor, ToolExecutorConfig


class AgentEnvConfig(BaseEnvConfig):
    tool_pool_mode: str = Field(default="nomad", description="Tool execution backend (only 'nomad' is supported)")

    allow_network: bool = Field(
        default=True,
        description="Whether sandbox bash commands may access the network (env policy).",
    )
    require_sandbox: bool = Field(
        default=False,
        description="Fail closed if bubblewrap sandboxing is unavailable/unusable for stateless sandbox tools.",
    )
    require_stateful_sandbox: bool = Field(
        default=False,
        description="Fail closed if bubblewrap/PID isolation is unavailable for stateful terminal tools (tmux).",
    )
    tool_batch_window_ms: int = Field(default=20, description="ToolExecutor batching window (ms)")
    tool_max_batch_size: int = Field(default=200, description="ToolExecutor maximum batch size")

    # nomad mode settings
    nomad_address: str = Field(default="http://localhost:4646", description="Nomad API address")
    sandbox_job_id: str = Field(default="atropos-sandbox-agent-env", description="Nomad job id for sandbox containers")
    sandbox_image: str = Field(default="atropos-sandbox:local", description="Docker image for sandbox containers")
    slots_per_container: int = Field(default=10, description="Nomad mode: slots per container")
    min_containers: int = Field(default=1, description="Nomad mode: minimum containers")
    max_containers: int = Field(default=10, description="Nomad mode: maximum containers")
    privileged: bool = Field(default=False, description="Nomad mode: run container privileged")
    acquire_timeout_s: float = Field(default=30.0, description="Slot acquisition timeout (seconds)")
    purge_job_on_shutdown: bool = Field(default=True, description="Nomad mode: stop/purge job on shutdown")

    # basic agent defaults
    agent_max_steps: int = Field(default=50, description="Max ReACT steps per trajectory")
    agent_temperature: float = Field(default=0.7, description="Sampling temperature")
    agent_max_tokens: int = Field(default=4096, description="Max tokens per model response")
    agent_tool_delay_s: float = Field(default=0.0, description="Delay between tool calls (seconds)")

    # tool selection
    enabled_toolsets: List[str] = Field(
        default_factory=lambda: ["default"],
        description="Toolsets to enable (Hermes-style grouping).",
    )
    disabled_toolsets: List[str] = Field(
        default_factory=list,
        description="Toolsets to disable (applied after enabled_toolsets).",
    )

    # external ToolServer routing (Phase 4.5+)
    tool_server_url: Optional[str] = Field(
        default=None,
        description="Base URL for external ToolServer (enables external tools).",
    )
    tool_server_token: Optional[str] = Field(
        default=None,
        description="Bearer token for ToolServer auth (optional in dev).",
    )

AgentEnvConfigT = TypeVar("AgentEnvConfigT", bound="AgentEnvConfig")


class AgentEnv(BaseEnv, ABC, Generic[AgentEnvConfigT]):
    env_config_cls = AgentEnvConfig

    def __init__(
        self,
        config: AgentEnvConfigT,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: AgentEnvConfigT = config

        self.tools: ToolRegistry = self.build_tools()

        self._pool: Optional[Any] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._tool_server_inprocess: bool = False

    def build_tools(self) -> ToolRegistry:
        available_tools = [
            BashTool(),
            TerminalTool(),
            TerminalStatefulTool(),
            TmuxTool(),
            ReadFileTool(),
            WriteFileTool(),
            ImageGenerateTool(),
            WebSearchTool(),
            WebExtractTool(),
            WebCrawlTool(),
            VisionAnalyzeTool(),
            MixtureOfAgentsTool(),
        ]

        tool_by_name = {t.name: t for t in available_tools}

        enabled_toolsets = self.config.enabled_toolsets or ["default"]
        selected = set(resolve_multiple_toolsets(enabled_toolsets))
        if self.config.disabled_toolsets:
            selected -= set(resolve_multiple_toolsets(self.config.disabled_toolsets))

        tools = ToolRegistry()
        for name in sorted(selected):
            tool = tool_by_name.get(name)
            if tool is None:
                continue
            # External tools require a ToolServer URL; avoid advertising broken tools.
            if tool.schema.external and not self.config.tool_server_url:
                continue
            ok, _reason = tool.is_available()
            if not ok:
                continue
            tools.register(tool)

        return tools

    @abstractmethod
    def build_task(self, item: Item) -> str:
        """Return the user-facing task string for the agent."""

    @abstractmethod
    async def score_trajectory(self, item: Item, final_response: str) -> float:
        """Return a scalar score for this trajectory."""

    async def setup_trajectory_workspace(
        self,
        item: Item,
        *,
        trajectory_id: str,
        exec_tool: Callable[["ToolCall"], Awaitable["ToolResult"]],
    ) -> Dict[str, Any]:
        """
        Optional hook: prepare the sandbox workspace before the agent starts.

        Examples:
        - clone a repo and checkout a commit
        - write fixture files (e.g. images) for external-tool demos
        - pre-install dependencies

        Default: no-op.
        """
        _ = (item, trajectory_id, exec_tool)
        return {}

    async def verify_and_score_trajectory(
        self,
        item: Item,
        final_response: str,
        *,
        trajectory_id: str,
        exec_tool: Callable[["ToolCall"], Awaitable["ToolResult"]],
    ) -> tuple[float, Dict[str, Any]]:
        """
        Optional hook: run in-sandbox verification before scoring.

        Many agent envs need to execute verification inside the same trajectory
        workspace (e.g. pytest) before releasing/resetting the slot.

        Default: calls `score_trajectory()` and returns empty metadata.
        """
        _ = (trajectory_id, exec_tool)  # default ignores in-workspace verification
        score = await self.score_trajectory(item, final_response)
        return score, {}

    def build_agent_config(self, item: Item) -> AgentConfig:  # noqa: ARG002
        return AgentConfig(
            max_steps=self.config.agent_max_steps,
            temperature=self.config.agent_temperature,
            max_tokens=self.config.agent_max_tokens,
            tool_delay_s=self.config.agent_tool_delay_s,
        )

    async def setup(self) -> None:
        await self._start_tool_backend()
        await self.setup_agent_env()

    @abstractmethod
    async def setup_agent_env(self) -> None:
        """Subclass hook for env-specific setup."""

    async def evaluate(self, *args, **kwargs):  # noqa: ARG002
        """
        Default eval hook (no-op).

        Atropos BaseEnv requires an `evaluate()` implementation. Many agent envs
        won't have a meaningful evaluation path during early PoC work; they can
        override this when needed.
        """
        return {}

    async def env_manager(self):
        try:
            return await super().env_manager()
        finally:
            await self.shutdown_tool_backend()

    async def process_manager(self):
        try:
            return await super().process_manager()
        finally:
            await self.shutdown_tool_backend()

    async def _start_tool_backend(self) -> None:
        if self._tool_executor is not None:
            return

        tool_server_url = self.config.tool_server_url
        tool_server_client = None
        if tool_server_url == "inprocess":
            import httpx
            from ..api.tool_server import app as tool_server_app

            await tool_server_app.router.startup()
            tool_server_client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=tool_server_app),
                base_url="http://toolserver",
            )
            tool_server_url = "http://toolserver"
            self._tool_server_inprocess = True

        if self.config.tool_pool_mode != "nomad":
            raise RuntimeError("tool_pool_mode must be 'nomad' (local/in-process pools are not supported)")

        pool = SlotPool(
            SlotPoolConfig(
                nomad_address=self.config.nomad_address,
                job_id=self.config.sandbox_job_id,
                image=self.config.sandbox_image,
                slots_per_container=self.config.slots_per_container,
                min_containers=self.config.min_containers,
                max_containers=self.config.max_containers,
                privileged=self.config.privileged,
                acquire_timeout=self.config.acquire_timeout_s,
            )
        )
        await pool.start()

        executor = ToolExecutor(
            pool=pool,
            tools=self.tools,
            config=ToolExecutorConfig(
                batch_window_ms=self.config.tool_batch_window_ms,
                max_batch_size=self.config.tool_max_batch_size,
                allow_network=self.config.allow_network,
                require_sandbox=self.config.require_sandbox,
                require_stateful_sandbox=self.config.require_stateful_sandbox,
                tool_server_url=tool_server_url,
                tool_server_token=self.config.tool_server_token,
            ),
        )
        await executor.start()
        if tool_server_client is not None:
            executor._tool_server_client = tool_server_client  # type: ignore[attr-defined]

        self._pool = pool
        self._tool_executor = executor

    async def shutdown_tool_backend(self) -> None:
        executor = self._tool_executor
        pool = self._pool
        inprocess_tool_server = self._tool_server_inprocess
        self._tool_executor = None
        self._pool = None
        self._tool_server_inprocess = False

        if executor is not None:
            await executor.close()
        if pool is not None:
            await pool.stop(purge_job=bool(self.config.purge_job_on_shutdown))
        if inprocess_tool_server:
            from ..api.tool_server import app as tool_server_app

            await tool_server_app.router.shutdown()

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        if self._tool_executor is None:
            raise RuntimeError("Tool backend not started")

        trajectory_id = str(uuid.uuid4())
        task = self.build_task(item)
        agent_config = self.build_agent_config(item)

        async def _exec(call):
            return await self._tool_executor.execute(trajectory_id, call)

        agent = AtroposAgent(
            server=self.server,
            tokenizer=self.tokenizer,
            tools=self.tools,
            config=agent_config,
            execute_tool=_exec,
        )

        try:
            await self.setup_trajectory_workspace(item, trajectory_id=trajectory_id, exec_tool=_exec)

            result = await agent.run(task)
            if not result.success or result.trajectory_data is None:
                return None, []

            score, _score_metadata = await self.verify_and_score_trajectory(
                item,
                result.final_response,
                trajectory_id=trajectory_id,
                exec_tool=_exec,
            )

            messages = [{"role": "system", "content": agent._build_system_prompt()}]  # noqa: SLF001
            messages.append({"role": "user", "content": task})
            for step in result.steps:
                messages.append({"role": "assistant", "content": step.assistant_message})
                if step.tool_results:
                    tool_text = "\n".join(r.to_xml() for r in step.tool_results)
                    messages.append({"role": "user", "content": tool_text})

            scored: ScoredDataItem = {
                "tokens": result.trajectory_data.tokens,
                "masks": result.trajectory_data.masked_tokens,
                "scores": score,
            }
            if self.config.include_messages:
                scored["messages"] = messages

            return scored, []
        finally:
            await self._tool_executor.release_trajectory(trajectory_id, reset_workspace=True)

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        tasks = [self.collect_trajectory(item) for _ in range(self.config.group_size)]
        results = await asyncio.gather(*tasks)

        backlog: List[Item] = []
        items: List[ScoredDataItem] = []
        for scored, b in results:
            backlog.extend(b)
            if scored is not None:
                items.append(scored)

        if len(items) != self.config.group_size:
            return None, backlog

        group: ScoredDataGroup = ScoredDataGroup(
            tokens=[],
            masks=[],
            scores=[],
            advantages=[],
            ref_logprobs=[],
            messages=[] if self.config.include_messages else None,
            group_overrides={},
            overrides=[],
            images=[],
        )

        for it in items:
            group["tokens"].append(it["tokens"])
            group["masks"].append(it["masks"])
            group["scores"].append(it["scores"])
            if group.get("messages") is not None and it.get("messages") is not None:
                group["messages"].append(it["messages"])

        return group, backlog

    async def run_agent(self, task: str, *, trajectory_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Run the AtroposAgent on a single task and return (final_response, debug).

        This is a helper intended for simple environments and tests.
        """
        if self._tool_executor is None:
            raise RuntimeError("Tool backend not started")

        tid = trajectory_id or str(uuid.uuid4())

        async def _exec(call):
            return await self._tool_executor.execute(tid, call)

        agent = AtroposAgent(
            server=self.server,
            tokenizer=self.tokenizer,
            tools=self.tools,
            config=AgentConfig(
                max_steps=self.config.agent_max_steps,
                temperature=self.config.agent_temperature,
                max_tokens=self.config.agent_max_tokens,
            ),
            execute_tool=_exec,
        )
        result = await agent.run(task)
        await self._tool_executor.release_trajectory(tid, reset_workspace=True)
        return result.final_response, {"success": result.success, "error": result.error, "tool_calls": result.total_tool_calls}
