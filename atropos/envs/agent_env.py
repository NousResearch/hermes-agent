"""
AgentEnv - Atropos BaseEnv extension for agent/tool-call workloads.

AgentEnv is responsible for starting the sandbox tool execution backend and
providing helpers for running agent trajectories with queued/batched tool calls.
"""

from __future__ import annotations
import os
import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from pydantic import Field

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, Item, ScoredDataGroup, ScoredDataItem
from atroposlib.envs.server_handling.server_baseline import AsyncSemWithAdaptiveWeight

from ..agent import AgentConfig, AgentResult, AtroposAgent
from ..backends import ToolBackend, create_tool_backend
from ..tools import ToolRegistry, build_tool_registry
from ..tools.tool_executor import ToolExecutor, ToolExecutorConfig

# Main BaseEnv child classes. Child class THESE to get agent+tooling functionality easily.

class AgentEnvConfig(BaseEnvConfig):
    tool_pool_mode: str = Field(default="nomad", description="Tool execution backend ('nomad' or 'modal')")

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

    # nomad mode settings. TODO: Add Modal support, split this into own config
    nomad_address: str = Field(default="http://localhost:4646", description="Nomad API address")
    sandbox_job_id: str = Field(default="atropos-sandbox-agent-env", description="Nomad job id for sandbox containers")
    sandbox_image: str = Field(default="atropos-sandbox:local", description="Docker image for sandbox containers")
    slots_per_container: int = Field(default=10, description="Nomad mode: slots per container")
    min_containers: int = Field(default=1, description="Nomad mode: minimum containers")
    max_containers: int = Field(default=10, description="Nomad mode: maximum containers")
    privileged: bool = Field(default=False, description="Nomad mode: run container privileged")
    acquire_timeout_s: float = Field(default=30.0, description="Slot acquisition timeout (seconds)")
    purge_job_on_start: bool = Field(
        default=False,
        description=(
            "Nomad mode: stop/purge the sandbox job on startup. This is helpful in local dev and training runs "
            "to recover from previous crashes that leave the job in a restart backoff state."
        ),
    )
    purge_job_on_shutdown: bool = Field(default=True, description="Nomad mode: stop/purge job on shutdown")

    # modal mode settings (stub; implementation pending)
    modal_app_name: str = Field(default="atropos-sandbox", description="Modal app name (stub)")
    modal_function_name: str = Field(default="sandbox_server", description="Modal function/actor name (stub)")
    modal_volume_name: Optional[str] = Field(default=None, description="Modal Volume name for persistent storage (stub)")
    modal_volume_mount_path: str = Field(default="/data", description="Modal Volume mount path (stub)")

    # basic agent defaults
    agent_max_steps: int = Field(default=50, description="Max ReACT steps per trajectory")
    agent_temperature: float = Field(default=0.7, description="Sampling temperature")
    agent_max_tokens: Optional[int] = Field(
        default=None,
        description="Max tokens per model response (default: let backend decide)",
    )
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

        self._backend: Optional[ToolBackend] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._tool_server_inprocess: bool = False
        self._trajectory_workspace_meta: Dict[str, Dict[str, Any]] = {}

    def build_tools(self) -> ToolRegistry:
        """Wraps original Hermes-Agent ToolRegistry for atropos AgentEnv use.
        See Hermes-Agent docs for toolsets and available tools etc.
        """
        return build_tool_registry(
            enabled_toolsets=self.config.enabled_toolsets or ["default"],
            disabled_toolsets=self.config.disabled_toolsets or None,
            tool_server_url=self.config.tool_server_url,
        )

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
        agent_result: Optional[AgentResult] = None,
        workspace_meta: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, Dict[str, Any]]:
        """
        Optional hook: run in-sandbox verification before scoring.

        Many agent envs need to execute verification inside the same trajectory
        workspace (e.g. pytest) before releasing/resetting the slot.

        Default: calls `score_trajectory()` and returns empty metadata.
        """
        _ = (trajectory_id, exec_tool, agent_result, workspace_meta)  # default ignores in-workspace verification
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
        print(f"[AgentEnv] setup(): starting tool backend ({self.config.tool_pool_mode})", flush=True)
        await self._start_tool_backend()
        print("[AgentEnv] setup(): configuring server concurrency", flush=True)
        self._configure_server_concurrency()
        print("[AgentEnv] setup(): running env-specific setup_agent_env()", flush=True)
        await self.setup_agent_env()
        print("[AgentEnv] setup(): done", flush=True)

    def _configure_server_concurrency(self) -> None:
        """
        Ensure the LLM server concurrency isn't accidentally capped below `group_size`.

        In `BaseEnv process` mode, groups are collected concurrently and if the underlying
        ServerManager/OpenAIServer semaphore is left at 1, we serialize inference even
        when `--env.group_size` is > 1.
        """
        desired = int(getattr(self.config, "group_size", 1) or 1)
        if desired <= 1:
            return

        servers = getattr(self.server, "servers", None)
        if not isinstance(servers, list) or not servers:
            return

        for s in servers:
            sem = getattr(s, "sem", None)
            eval_sem = getattr(s, "eval_sem", None)
            # Only increase; never shrink.
            if sem is not None and getattr(sem, "max_val", 0) < desired:
                s.sem = AsyncSemWithAdaptiveWeight(desired)
                if hasattr(s, "config") and hasattr(s.config, "num_max_requests_at_once"):
                    s.config.num_max_requests_at_once = desired
            if eval_sem is not None and getattr(eval_sem, "max_val", 0) < desired:
                s.eval_sem = AsyncSemWithAdaptiveWeight(desired)
                if hasattr(s, "config") and hasattr(s.config, "num_requests_for_eval"):
                    s.config.num_requests_for_eval = desired

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

        backend = create_tool_backend(self.config)
        await backend.start()

        executor = ToolExecutor(
            backend=backend,
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

        self._backend = backend
        self._tool_executor = executor

    async def shutdown_tool_backend(self) -> None:
        executor = self._tool_executor
        backend = self._backend
        inprocess_tool_server = self._tool_server_inprocess
        self._tool_executor = None
        self._backend = None
        self._tool_server_inprocess = False

        if executor is not None:
            await executor.close()
        if backend is not None:
            await backend.stop(purge=bool(self.config.purge_job_on_shutdown))
        if inprocess_tool_server:
            from ..api.tool_server import app as tool_server_app

            await tool_server_app.router.shutdown()

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        if self._tool_executor is None:
            raise RuntimeError("Tool backend not started")

        trajectory_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        print(f"[AgentEnv] collect_trajectory(): tid={trajectory_id} start", flush=True)
        task = self.build_task(item)
        agent_config = self.build_agent_config(item)
        if os.getenv("ATROPOS_DEBUG_PRINT_TASK") == "1":
            print(f"Starting trajectory {trajectory_id} with task: {task}", flush=True)
        else:
            # Avoid printing the full task prompt by default (can be huge/noisy).
            one_line = " ".join(str(task).splitlines()).strip()
            preview = one_line[:240] + ("…" if len(one_line) > 240 else "")
            print(f"Starting trajectory {trajectory_id} (task preview): {preview}", flush=True)

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
            print(f"[AgentEnv] tid={trajectory_id} setup_trajectory_workspace() start", flush=True)
            workspace_meta = await self.setup_trajectory_workspace(item, trajectory_id=trajectory_id, exec_tool=_exec)
            if not isinstance(workspace_meta, dict):
                workspace_meta = {}
            self._trajectory_workspace_meta[trajectory_id] = workspace_meta
            print(
                f"[AgentEnv] tid={trajectory_id} setup_trajectory_workspace() done in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

            print(f"[AgentEnv] tid={trajectory_id} agent.run() start", flush=True)
            result = await agent.run(task)
            print(
                f"[AgentEnv] tid={trajectory_id} agent.run() done in {time.perf_counter() - t0:.2f}s "
                f"success={result.success} tool_calls={result.total_tool_calls}",
                flush=True,
            )
            if not result.success or result.trajectory_data is None:
                # Do not trigger BaseEnv retries for agent failures.
                # Record the trajectory with score 0.0 so training/eval can see the failure mode.
                messages = [{"role": "system", "content": agent._build_system_prompt()}]  # noqa: SLF001
                messages.append({"role": "user", "content": task})
                for step in result.steps:
                    messages.append({"role": "assistant", "content": step.assistant_message})
                    if step.tool_results:
                        tool_text = "\n".join(r.to_xml() for r in step.tool_results)
                        messages.append({"role": "user", "content": tool_text})

                scored: ScoredDataItem = {
                    "tokens": (result.trajectory_data.tokens if result.trajectory_data else []),
                    "masks": (result.trajectory_data.masked_tokens if result.trajectory_data else []),
                    "scores": 0.0,
                }
                if self.config.include_messages:
                    # Record a final failure marker as a user-side tool_response-like block so it survives templates.
                    import json

                    err = result.error or "agent_failed"
                    messages.append(
                        {
                            "role": "user",
                            "content": f"<tool_response>{json.dumps({'success': False, 'error': err})}</tool_response>",
                        }
                    )
                    scored["messages"] = messages
                return scored, []

            print(f"[AgentEnv] tid={trajectory_id} verify_and_score_trajectory() start", flush=True)
            score, score_metadata = await self.verify_and_score_trajectory(
                item,
                result.final_response,
                trajectory_id=trajectory_id,
                exec_tool=_exec,
                agent_result=result,
                workspace_meta=workspace_meta,
            )
            print(
                f"[AgentEnv] tid={trajectory_id} verify_and_score_trajectory() done in {time.perf_counter() - t0:.2f}s "
                f"score={score}",
                flush=True,
            )

            messages = [{"role": "system", "content": agent._build_system_prompt()}]  # noqa: SLF001
            messages.append({"role": "user", "content": task})
            for step in result.steps:
                messages.append({"role": "assistant", "content": step.assistant_message})
                if step.tool_results:
                    tool_text = "\n".join(r.to_xml() for r in step.tool_results)
                    messages.append({"role": "user", "content": tool_text})

            # Optional: allow env verification to attach additional messages (e.g. install logs).
            if self.config.include_messages and isinstance(score_metadata, dict):
                extra = score_metadata.get("verification_messages")
                if isinstance(extra, list):
                    for m in extra:
                        if isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str):
                            messages.append({"role": m["role"], "content": m["content"]})

            scored: ScoredDataItem = {
                "tokens": result.trajectory_data.tokens,
                "masks": result.trajectory_data.masked_tokens,
                "scores": score,
            }
            if self.config.include_messages:
                scored["messages"] = messages

            return scored, []
        finally:
            self._trajectory_workspace_meta.pop(trajectory_id, None)
            print(f"[AgentEnv] tid={trajectory_id} release_trajectory(reset_workspace=True)", flush=True)
            await self._tool_executor.release_trajectory(trajectory_id, reset_workspace=True)
            print(f"[AgentEnv] collect_trajectory(): tid={trajectory_id} done in {time.perf_counter() - t0:.2f}s", flush=True)

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
        
        # TODO: Mack sure logprobs included

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
