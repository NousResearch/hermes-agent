"""
Multi-Agent Orchestration Layer — 基于 #344 的架构设计

核心组件：
1. MultiAgentCoordinator — 负责任务分解+DAG编排+结果综合
2. AgentRole — 预定义角色系统（Coordinator/Researcher/Developer/Reviewer/Synthesizer）
3. SharedMemory — 跨Agent共享上下文池
4. DAGWorkflow — 有向无环图工作流，支持依赖等待

设计原则（来自#344）：
- 保留现有delegate_task的隔离安全性
- 新增角色分工+依赖感知+结果聚合
- Coordinator做任务分解，Worker执行，Synthesizer聚合
"""

from __future__ import annotations

import json
import logging
import threading
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, AsyncIterator
from collections import defaultdict

from agent.flow import Flow, StatefulFlow, AgentFlow, entry, after

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """预定义Agent角色，对应特定工具集+系统提示"""
    COORDINATOR = "coordinator"     # 分解任务、分配工作、监控进度
    RESEARCHER = "researcher"      # web搜索、文档分析、信息收集
    DEVELOPER = "developer"        # 写代码、terminal操作、git
    REVIEWER = "reviewer"          # 代码审查、质量评估、验收标准
    SYNTHESIZER = "synthesizer"      # 聚合多Agent结果、生成最终输出
    BROWSER = "browser"            # 浏览器操作、表单填写、视觉验证


# 每个角色默认使用的工具集
ROLE_TOOLSETS: dict[AgentRole, list[str]] = {
    AgentRole.COORDINATOR: ["terminal", "file", "skills"],
    AgentRole.RESEARCHER: ["search", "web", "file"],
    AgentRole.DEVELOPER: ["terminal", "file", "code_execution"],
    AgentRole.REVIEWER: ["terminal", "file"],
    AgentRole.SYNTHESIZER: ["terminal", "file"],
    AgentRole.BROWSER: ["browser", "terminal"],
}

# 每个角色的系统提示前缀
ROLE_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.COORDINATOR: (
        "You are the Coordinator for a multi-agent workflow. "
        "Your job is to decompose a complex task into subtasks, assign each to the right role, "
        "monitor progress, handle failures, and synthesize the final result."
    ),
    AgentRole.RESEARCHER: (
        "You are a Research Agent. Your job is to gather, analyze, and summarize information "
        "from web searches, documents, and code analysis. Be thorough and cite sources."
    ),
    AgentRole.DEVELOPER: (
        "You are a Developer Agent. Your job is to write, modify, and test code. "
        "Follow best practices, write clean code, and verify your changes work."
    ),
    AgentRole.REVIEWER: (
        "You are a Reviewer Agent. Your job is to evaluate code quality, "
        "check against acceptance criteria, and provide actionable feedback."
    ),
    AgentRole.SYNTHESIZER: (
        "You are a Synthesizer Agent. Your job is to take results from multiple agents "
        "and produce a unified, coherent final output."
    ),
    AgentRole.BROWSER: (
        "You are a Browser Agent. Your job is to interact with web pages, "
        "fill forms, verify visual elements, and extract structured data from websites."
    ),
}


@dataclass
class TaskNode:
    """DAG中的一个任务节点"""
    task_id: str
    goal: str
    role: AgentRole
    context: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)  # 依赖的task_id列表
    result: Optional[Any] = None
    status: str = "pending"  # pending/running/completed/failed
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    agent_id: Optional[str] = None  # 执行这个任务的子agent标识
    _interim_results: list[Any] = field(default_factory=list)  # 流式中间结果（Phase3）

    def publish_stream_result(self, partial: Any) -> None:
        """记录一个流式中间结果，供下游任务立即消费"""
        self._interim_results.append(partial)

    def get_stream_results(self) -> list[Any]:
        return list(self._interim_results)


@dataclass
class SharedMemory:
    """
    跨Agent共享内存池 — 解决#344中"children can't talk to each other"的问题

    使用 asyncio.Queue 消息总线 + 发布-订阅模式：
    - 每个Agent可以向池中写入（publish）
    - 每个Agent可以订阅特定类型的更新（subscribe）
    - Coordinator可以读取所有内容
    - 完全异步，非阻塞

    设计原则（参考 autogen 的 asyncio.Queue 消息总线）：
    - 用 asyncio.Lock 替代 threading.Lock（非阻塞）
    - asyncio.Queue 用于异步消息传递
    - 同步方法保留以兼容现有调用方
    """
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _data: dict[str, Any] = field(default_factory=dict)
    _subscribers: dict[str, list[Callable]] = field(default_factory=dict)
    _events: list[dict] = field(default_factory=list)
    _max_events: int = 1000
    _queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1000))

    def publish(self, key: str, value: Any, event_type: str = "update") -> None:
        """写入共享数据（同步兼容）"""
        self._data[key] = value
        event = {
            "ts": time.time(),
            "type": event_type,
            "key": key,
            "value": repr(value)[:200] if not isinstance(value, str) else value[:200],
        }
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        # 通知订阅者（同步回调，用于兼容）
        for callback in self._subscribers.get(key, []) + self._subscribers.get("*", []):
            try:
                callback(key, value)
            except Exception as e:
                logger.debug("Subscriber callback error: %s", e)

    async def publish_async(self, key: str, value: Any, event_type: str = "update") -> None:
        """异步发布（推荐）"""
        async with self._lock:
            self._data[key] = value
            event = {
                "ts": time.time(),
                "type": event_type,
                "key": key,
                "value": repr(value)[:200] if not isinstance(value, str) else value[:200],
            }
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
        # 非阻塞地放入队列（不持有锁）
        try:
            self._queue.put_nowait({"key": key, "value": value, "type": event_type})
        except asyncio.QueueFull:
            logger.debug("SharedMemory queue full, dropping event")
        # 通知订阅者
        for callback in self._subscribers.get(key, []) + self._subscribers.get("*", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(key, value)
                else:
                    callback(key, value)
            except Exception as e:
                logger.debug("Subscriber callback error: %s", e)

    async def read_async(self, key: str, default: Any = None) -> Any:
        """异步读取"""
        async with self._lock:
            return self._data.get(key, default)

    async def read_all_async(self) -> dict[str, Any]:
        """异步读取所有"""
        async with self._lock:
            return dict(self._data)

    def read(self, key: str, default: Any = None) -> Any:
        """同步读取（兼容现有调用方）"""
        return self._data.get(key, default)

    def read_all(self) -> dict[str, Any]:
        """同步读取所有（快照）"""
        return dict(self._data)

    def subscribe(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """订阅特定key的更新（同步/异步均可）"""
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)

    async def subscribe_async(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """异步订阅"""
        async with self._lock:
            if key not in self._subscribers:
                self._subscribers[key] = []
            self._subscribers[key].append(callback)

    async def wait_for(self, key: str, timeout: float = 30.0) -> Any:
        """等待特定key有值（类似 asyncio.Queue.get）"""
        async with self._lock:
            if key in self._data:
                return self._data[key]
            # 临时订阅
            event = asyncio.Event()

            def _check(key_: str, value: Any) -> None:
                event.set()
            self._subscribers.setdefault(key, []).append(_check)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return self._data.get(key)
        finally:
            async with self._lock:
                if key in self._subscribers and _check in self._subscribers[key]:
                    self._subscribers[key].remove(_check)
        return self._data.get(key)

    def get_events(self, since_ts: float = 0) -> list[dict]:
        """获取事件日志（用于调试和trace）"""

    # ----------------------------------------------------------------------
    # Phase3: 流式中间结果发布/订阅（agent间实时通信）
    # ----------------------------------------------------------------------

    def publish_stream(self, task_id: str, partial: Any) -> None:
        """发布任务的流式中间结果（同步兼容）"""
        stream_key = f"stream:{task_id}"
        if stream_key not in self._data:
            self._data[stream_key] = []
        self._data[stream_key].append({
            "ts": time.time(),
            "value": partial,
        })
        event = {
            "ts": time.time(),
            "type": "stream",
            "key": stream_key,
            "value": repr(partial)[:200] if not isinstance(partial, str) else partial[:200],
        }
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        for callback in self._subscribers.get(stream_key, []) + self._subscribers.get("stream:*", []):
            try:
                callback(task_id, partial)
            except Exception as e:
                logger.debug("Stream subscriber error: %s", e)

    async def subscribe_stream_async(self, task_id: str) -> AsyncIterator[Any]:
        """异步迭代获取任务的流式结果（供下游消费者用）"""
        stream_key = f"stream:{task_id}"
        last_len = 0
        while True:
            async with self._lock:
                stream_list = self._data.get(stream_key, [])
                if len(stream_list) > last_len:
                    yield stream_list[last_len]["value"]
                    last_len = len(stream_list)
                    continue
            event = asyncio.Event()
            def _notify(tid: str, val: Any) -> None:
                del tid
                event.set()
            async with self._lock:
                self._subscribers.setdefault(stream_key, []).append(_notify)
            try:
                await asyncio.wait_for(event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                break
            finally:
                async with self._lock:
                    if _notify in self._subscribers.get(stream_key, []):
                        self._subscribers[stream_key].remove(_notify)

    def get_stream_results(self, task_id: str) -> list[Any]:
        """同步读取任务的所有流式中间结果"""
        stream_key = f"stream:{task_id}"
        return [e["value"] for e in self._data.get(stream_key, [])]
        return [e for e in self._events if e["ts"] > since_ts]


class DAGWorkflow:
    """
    有向无环图工作流 — 支持依赖感知的任务调度

    与简单parallel dispatch的区别：
    - 任务按依赖关系排序，不是所有任务同时开始
    - 一个任务的输出可以作为另一个任务的输入
    - 支持条件分支（role-based routing）
    """

    def __init__(self, nodes: list[TaskNode]):
        self.nodes: dict[str, TaskNode] = {n.task_id: n for n in nodes}
        self._topo_order: list[str] = []
        self._ready_queue: list[str] = []
        self._completed: set[str] = set()
        self._failed: set[str] = set()

    def get_ready_tasks(self, stream_enabled: bool = False) -> list[TaskNode]:
        """返回所有依赖已满足且未执行的任务

        Args:
            stream_enabled: 若为True，下游任务可在上游有流式中间结果时提前启动
        """
        ready = []
        for task_id, node in self.nodes.items():
            if node.status != "pending":
                continue
            deps_done = all(
                self.nodes[d].status == "completed"
                for d in node.depends_on
            )
            if deps_done:
                ready.append(node)
            elif stream_enabled:
                # 部分依赖已有流式结果，提前启动下游
                has_stream = any(
                    len(self.nodes[d].get_stream_results()) > 0
                    for d in node.depends_on
                    if self.nodes[d].status in ("running", "completed")
                )
                if has_stream:
                    ready.append(node)
        return ready


    def mark_running(self, task_id: str) -> None:
        self.nodes[task_id].status = "running"
        self.nodes[task_id].started_at = time.time()

    def mark_completed(self, task_id: str, result: Any) -> None:
        self.nodes[task_id].status = "completed"
        self.nodes[task_id].result = result
        self.nodes[task_id].completed_at = time.time()
        self._completed.add(task_id)

    def mark_failed(self, task_id: str, error: str) -> None:
        self.nodes[task_id].status = "failed"
        self.nodes[task_id].error = error
        self.nodes[task_id].completed_at = time.time()
        self._failed.add(task_id)

    def is_done(self) -> bool:
        """所有任务都执行完毕（成功或失败）"""
        return len(self._completed) + len(self._failed) == len(self.nodes)

    def get_result_for(self, task_id: str) -> Optional[Any]:
        """获取任务结果，如果任务失败返回None"""
        node = self.nodes.get(task_id)
        if node and node.status == "completed":
            return node.result
        return None

    def get_summary(self) -> dict:
        """返回工作流执行摘要"""
        return {
            "total": len(self.nodes),
            "completed": len(self._completed),
            "failed": len(self._failed),
            "pending": sum(1 for n in self.nodes.values() if n.status == "pending"),
            "running": sum(1 for n in self.nodes.values() if n.status == "running"),
            "nodes": {
                tid: {
                    "role": node.role.value,
                    "status": node.status,
                    "error": node.error,
                    "duration": (
                        round(node.completed_at - node.started_at, 1)
                        if node.completed_at and node.started_at else None
                    ),
                }
                for tid, node in self.nodes.items()
            },
        }


class MultiAgentCoordinator:
    """
    多Agent编排器 — 核心协调逻辑

    使用流程：
    1. 初始化Coordinator，传入复杂目标
    2. 调用 decompose() 分解任务为DAG
    3. 调用 execute() 并行/按序执行
    4. 调用 synthesize() 聚合结果

    与现有delegate_task的关系：
    - 内部复用delegate_task的能力（隔离子Agent、安全工具限制）
    - 新增：角色分配、依赖感知、结果聚合、共享内存
    """

    def __init__(
        self,
        parent_agent,  # 用于创建子Agent的父级引用
        goal: str,
        context: Optional[str] = None,
        max_concurrent: int = 3,
        workflow_mode: str = "auto",  # "auto"=自动分解, "manual"=手动指定任务
        tasks: Optional[list[dict]] = None,  # 手动指定的任务列表（含task_id/goal/role/depends_on）
    ):
        self.parent_agent = parent_agent
        self.goal = goal
        self.context = context or ""
        self.max_concurrent = max_concurrent
        self.workflow_mode = workflow_mode
        self._manual_tasks = tasks  # store for decompose()

        self.shared_memory = SharedMemory()
        self.workflow: Optional[DAGWorkflow] = None
        self._active_children: list = []
        self._lock = threading.Lock()

        # 执行统计
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def decompose(self, role_hint: Optional[str] = None) -> list[TaskNode]:
        """
        将复杂目标分解为任务DAG。

        策略：
        1. 如果self._manual_tasks有值，使用预定义任务列表（manual DAG模式）
        2. 如果workflow_mode="manual"且无预定义，用LLM分解
        3. 否则，让LLM分析目标并生成任务列表+依赖关系

        返回：TaskNode列表
        """
        # 优先使用预定义任务（manual DAG模式）
        if self._manual_tasks is not None:
            nodes = self._build_nodes_from_manual_tasks(self._manual_tasks)
            logger.info(
                "Using %d manual tasks: %s",
                len(nodes),
                [n.task_id for n in nodes],
            )
        else:
            nodes = self._llm_decompose(self._build_decomposition_prompt())
            logger.info(
                "Decomposed goal into %d tasks: %s",
                len(nodes),
                [n.task_id for n in nodes],
            )
        self.workflow = DAGWorkflow(nodes)
        return nodes

    def _build_nodes_from_manual_tasks(self, tasks: list[dict]) -> list[TaskNode]:
        """从预定义任务字典列表构建TaskNode列表。"""
        nodes = []
        for t in tasks:
            role_str = t.get("role", "developer")
            try:
                role = AgentRole(role_str)
            except ValueError:
                role = AgentRole.DEVELOPER
            nodes.append(TaskNode(
                task_id=t["task_id"],
                goal=t["goal"],
                role=role,
                context=t.get("context"),
                depends_on=t.get("depends_on", []),
            ))
        return nodes

    def _build_decomposition_prompt(self) -> str:
        return f"""Analyze this goal and decompose it into tasks for a multi-agent team:

GOAL: {self.goal}
CONTEXT: {self.context}

Available roles:
- coordinator: task decomposition and workflow management
- researcher: web search, document analysis, information gathering
- developer: code writing, terminal operations, git
- reviewer: code review, acceptance criteria evaluation
- synthesizer: result aggregation and final output generation
- browser: web interaction, form filling, visual verification

Rules:
1. Each task should be small enough to complete in one agent session
2. Tasks can depend on others — use depends_on for ordering
3. Assign the most appropriate role to each task
4. Include a "synthesize" task at the end that depends on all others
5. Return a JSON array of tasks, each with:
   - task_id: unique string
   - goal: what this task should accomplish
   - role: one of the available roles
   - context: additional context for this specific task
   - depends_on: array of task_ids this depends on

Respond ONLY with valid JSON, no markdown, no explanation."""

    def _llm_decompose(self, prompt: str) -> list[TaskNode]:
        """通过父Agent的LLM进行任务分解"""
        import json as _json
        import re
        from tools.delegate_tool import delegate_task

        result = delegate_task(
            goal=prompt,
            context=(
                "You are a task decomposition specialist. "
                "Analyze the goal and produce a JSON task list. "
                "Return ONLY valid JSON array, no markdown formatting."
            ),
            toolsets=["terminal", "file"],
            max_iterations=10,
            parent_agent=self.parent_agent,
        )

        try:
            data = json.loads(result)
            if isinstance(data, dict) and "result" in data:
                text = data["result"]
            elif isinstance(data, str):
                text = data
            else:
                text = str(data)

            # 提取JSON（可能在markdown代码块里）
            match = re.search(r'\[[\s\S]*\]', text)
            tasks_data = _json.loads(match.group()) if match else _json.loads(text)

            nodes = []
            for t in tasks_data:
                role_str = t.get("role", "developer")
                try:
                    role = AgentRole(role_str)
                except ValueError:
                    role = AgentRole.DEVELOPER

                nodes.append(TaskNode(
                    task_id=t["task_id"],
                    goal=t["goal"],
                    role=role,
                    context=t.get("context"),
                    depends_on=t.get("depends_on", []),
                ))
            return nodes

        except Exception as e:
            logger.warning("Task decomposition failed, falling back to single task: %s", e)
            return [
                TaskNode(
                    task_id="main",
                    goal=self.goal,
                    role=AgentRole.DEVELOPER,
                    context=self.context,
                    depends_on=[],
                )
            ]

    async def execute_async(
        self,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        cancel_token=None,
    ) -> dict:
        """
        执行工作流（异步版本）。

        并行策略：
        - 维护一个待执行队列
        - 同时运行的任务不超过max_concurrent
        - 每当一个任务完成，检查是否有新任务可以开始
        - 支持依赖等待
        - 集成CancellationToken，支持优雅取消
        """
        if not self.workflow:
            raise RuntimeError("Must call decompose() before execute_async()")

        self.started_at = time.time()
        results: dict[str, Any] = {}
        execution_log = []
        running_tasks: dict[str, asyncio.Task] = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_task(task: TaskNode) -> None:
            """在信号量限制下运行单个任务，结果存入workflow.nodes[task_id].result"""
            async with semaphore:
                # 进入临界区前检查取消
                if cancel_token and cancel_token.is_cancelled():
                    self.workflow.mark_failed(task.task_id, "Cancelled")
                    return

                self.workflow.mark_running(task.task_id)
                log_entry = {
                    "task_id": task.task_id,
                    "role": task.role.value,
                    "started": time.time(),
                }

                # 在线程池中运行同步的delegate_task，等待结果写入workflow
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._execute_task_sync, task, cancel_token
                )
                # 结果已在 self.workflow.nodes[task_id].result/.error 中

                log_entry["completed"] = time.time()
                execution_log.append(log_entry)

                if progress_callback:
                    try:
                        progress_callback(task.task_id, f"Completed {task.role.value}")
                    except Exception:
                        pass

        # 主调度循环
        while not self.workflow.is_done():
            if cancel_token and cancel_token.is_cancelled():
                for t in running_tasks.values():
                    t.cancel()
                break

            # ------------------------------------------------------------------
            # Phase3: 轮询运行中任务的流式中间结果，传播给下游任务
            # ------------------------------------------------------------------
            for tid, node in self.workflow.nodes.items():
                if node.status == "running":
                    stream_results = self.shared_memory.get_stream_results(tid)
                    new_results = stream_results[len(node._interim_results):]
                    for partial in new_results:
                        node.publish_stream_result(partial)
                        for downstream in self.workflow.nodes.values():
                            if tid in downstream.depends_on:
                                downstream.publish_stream_result(partial)

            ready_tasks = self.workflow.get_ready_tasks(stream_enabled=True)
            if not ready_tasks:
                running = [n for n in self.workflow.nodes.values() if n.status == "running"]
                if running:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    logger.error("Workflow deadlock detected")
                    break

            # 启动就绪的任务（受限于max_concurrent）
            for task in ready_tasks[:self.max_concurrent]:
                t = asyncio.create_task(run_task(task))
                running_tasks[task.task_id] = t

            # 等待任意一个任务完成
            if running_tasks:
                done, _pending = await asyncio.wait(
                    running_tasks.values(),
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for d in done:
                    for tid, t_ref in list(running_tasks.items()):
                        if t_ref is d:
                            del running_tasks[tid]
                            break

                # 清理已完成的任务
                for t in done:
                    try:
                        t.result()
                    except Exception as e:
                        logger.error("Task error: %s", e)

        # 等待剩余任务
        if running_tasks:
            await asyncio.wait(running_tasks.values(), timeout=5.0)

        self.completed_at = time.time()
        duration = self.completed_at - self.started_at

        for task_id, node in self.workflow.nodes.items():
            results[task_id] = {
                "status": node.status,
                "result": node.result,
                "error": node.error,
                "role": node.role.value,
                "duration": (
                    round(node.completed_at - node.started_at, 1)
                    if node.completed_at and node.started_at else None
                ),
            }

        return {
            "workflow_summary": self.workflow.get_summary(),
            "results": results,
            "total_duration": round(duration, 1),
            "execution_log": execution_log,
        }

    def _execute_task_sync(self, task: TaskNode, cancel_token=None) -> None:
        """同步版本的任务执行（在线程池中运行）"""
        self._execute_task(task, None, cancel_token)

    def _execute_task(
        self,
        task: TaskNode,
        progress_callback: Optional[Callable],
        cancel_token=None,
    ) -> None:
        """在子线程中执行单个任务（含三层失败恢复）"""
        from tools.delegate_tool import delegate_task

        # 检查取消
        if cancel_token and cancel_token.is_cancelled():
            self.workflow.mark_failed(task.task_id, "Cancelled")
            return

        # --- 内部辅助：收集依赖context ---
        def _build_context() -> str:
            dep_context_parts = []
            for dep_id in task.depends_on:
                dep_result = self.workflow.get_result_for(dep_id)
                if dep_result is not None:
                    dep_context_parts.append(f"[Input from {dep_id}]:\n{dep_result}\n")
            ctx = "\n".join(dep_context_parts)
            if task.context:
                ctx = f"{ctx}\n{task.context}" if ctx else task.context
            return ctx

        role_prompt = ROLE_SYSTEM_PROMPTS.get(task.role, "")
        full_goal = f"{role_prompt}\n\nTASK:\n{task.goal}"

        if progress_callback:
            progress_callback(task.task_id, f"Starting {task.role.value} task")

        # --- 三层失败恢复 (Retry → Replan → Decompose) ---
        attempts = 0
        max_retries = 2
        last_error = None

        while attempts <= max_retries:
            attempts += 1
            if attempts > 1:
                logger.info("Task %s attempt %d/%d", task.task_id, attempts, max_retries + 1)
                if progress_callback:
                    progress_callback(task.task_id, f"Retry {attempts}/{max_retries + 1}")

            try:
                result = delegate_task(
                    goal=full_goal,
                    context=_build_context(),
                    toolsets=ROLE_TOOLSETS.get(task.role, ["terminal", "file"]),
                    max_iterations=30,
                    parent_agent=self.parent_agent,
                )

                try:
                    result_data = json.loads(result)
                    if isinstance(result_data, dict) and "error" in result_data:
                        raise ValueError(result_data["error"])
                    result_text = result_data.get("result", result) if isinstance(result_data, dict) else result
                except (json.JSONDecodeError, TypeError):
                    result_text = str(result)

                # 成功
                self.shared_memory.publish(f"task:{task.task_id}", result_text)
                self.workflow.mark_completed(task.task_id, result_text)
                if progress_callback:
                    progress_callback(task.task_id, f"Completed {task.role.value} task")
                return

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning("Task %s attempt %d failed: %s", task.task_id, attempts, last_error)

                if attempts <= max_retries:
                    # Layer 1: Retry
                    continue

                # Layer 2: Replan
                replan_prompt = (
                    "The following task failed: " + task.goal + "\n"
                    "Error: " + last_error + "\n\n"
                    "Role: " + task.role.value + "\n"
                    "Context: " + _build_context() + "\n\n"
                    "Try a different approach to accomplish the same goal. "
                    "Break it down differently, use different tools, or simplify the scope."
                )
                try:
                    result = delegate_task(
                        goal=replan_prompt,
                        context=_build_context(),
                        toolsets=ROLE_TOOLSETS.get(task.role, ["terminal", "file"]),
                        max_iterations=30,
                        parent_agent=self.parent_agent,
                    )
                    try:
                        result_data = json.loads(result)
                        result_text = result_data.get("result", result) if isinstance(result_data, dict) else result
                    except (json.JSONDecodeError, TypeError):
                        result_text = str(result)
                    self.shared_memory.publish(f"task:{task.task_id}", result_text)
                    self.workflow.mark_completed(task.task_id, result_text)
                    if progress_callback:
                        progress_callback(task.task_id, f"Completed after replan ({task.role.value})")
                    return
                except Exception as replan_error:
                    last_error = f"Replan failed: {type(replan_error).__name__}: {replan_error}"

                # Layer 3: Decompose
                logger.error("Task %s failed permanently after Retry+Replan: %s", task.task_id, last_error)
                self.workflow.mark_failed(task.task_id, last_error)
                self.shared_memory.publish(f"task:{task.task_id}:error", last_error)
                if progress_callback:
                    progress_callback(task.task_id, f"Failed: {last_error}")

    def execute_sync(self) -> dict:
        """同步执行入口（内部用线程池运行execute_async）"""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.execute_async())
        finally:
            loop.close()

    def synthesize(self, result: dict) -> str:
        """综合所有Agent的结果生成最终输出"""
        summary = self.workflow.get_summary() if self.workflow else {}
        all_results = result.get("results", {})

        ordered_results = []
        for task_id, data in all_results.items():
            ordered_results.append({
                "task_id": task_id,
                "role": data.get("role"),
                "status": data.get("status"),
                "result": data.get("result"),
            })

        synthesis_prompt = f"""You are a Synthesizer. Aggregate results from multiple agents into a final report.

ORIGINAL GOAL: {self.goal}

AGENT RESULTS:
{json.dumps(ordered_results, indent=2, ensure_ascii=False)}

Your job:
1. Review all agent results in dependency order
2. Identify key findings, decisions, and artifacts
3. Produce a coherent final report addressing the original goal
4. Highlight any failures or partial results
5. Provide next steps if applicable

Be thorough and specific. Do not just summarize — synthesize into actionable insights."""

        from tools.delegate_tool import delegate_task

        try:
            synthesis_result = delegate_task(
                goal=synthesis_prompt,
                context=f"Original goal: {self.goal}\nResults: {json.dumps(all_results, indent=2)[:5000]}",
                toolsets=["terminal", "file"],
                max_iterations=20,
                parent_agent=self.parent_agent,
            )
            return synthesis_result
        except Exception as e:
            return f"Synthesis failed: {e}\n\nRaw results: {json.dumps(all_results, indent=2)}"


# ---------------------------------------------------------------------------
# FlowBasedMultiAgentCoordinator — @entry/@after 装饰器风格定义多Agent工作流
# ---------------------------------------------------------------------------

def agent_step(role: AgentRole, goal_template: str = ""):
    """
    标记一个方法作为Agent步骤，配合 @entry / @after 使用。

    用法：
        class MyFlow(FlowBasedMultiAgentCoordinator):
            @entry
            @agent_step(Role.DEVELOPER, "Write code for: {input}")
            def start(self, input):
                return input

            @after("start")
            @agent_step(Role.REVIEWER, "Review the code: {input}")
            def review(self, input):
                return input
    """
    def decorator(func: Callable) -> Callable:
        func.__agent_role__ = role
        func.__agent_goal_template__ = goal_template or func.__name__
        return func
    return decorator


class FlowBasedMultiAgentCoordinator(AgentFlow):
    """
    用 @entry / @after 装饰器定义多Agent工作流。

    每个步骤用 @agent_step(role, goal_template) 标记，
    coordinator 自动把步骤转换为 TaskNode DAG 并执行。

    用法：
        class CodeReviewFlow(FlowBasedMultiAgentCoordinator):
            @entry
            @agent_step(AgentRole.DEVELOPER, "实现功能：{input}")
            def develop(self, goal):
                return goal

            @after("develop")
            @agent_step(AgentRole.REVIEWER, "审查代码：{input}")
            def review(self, context):
                return context

            @after("develop")
            @agent_step(AgentRole.RESEARCHER, "查找相关资料：{input}")
            def research(self, context):
                return context

            @after("review")
            @after("research")
            @agent_step(AgentRole.SYNTHESIZER, "综合结果")
            def synthesize(self, review_result, research_result):
                return {"review": review_result, "research": research_result}

        flow = CodeReviewFlow(
            parent_agent=self,
            initial_goal="某个复杂任务",
            max_concurrent=3,
        )
        result = flow.run()
    """

    def __init__(
        self,
        parent_agent,
        initial_goal: str = "",
        max_concurrent: int = 3,
    ):
        self.parent_agent = parent_agent
        self.initial_goal = initial_goal
        self.max_concurrent = max_concurrent
        self._results: dict[str, Any] = {}

    def run_agent(self, role: AgentRole, goal: str, context: str) -> str:
        """执行真实agent（同步，阻塞）"""
        from tools.delegate_tool import delegate_task
        role_prompt = ROLE_SYSTEM_PROMPTS.get(role, "")
        full_goal = f"{role_prompt}\n\nTASK:\n{goal}"
        try:
            result = delegate_task(
                goal=full_goal,
                context=context,
                toolsets=ROLE_TOOLSETS.get(role, ["terminal", "file"]),
                max_iterations=30,
                parent_agent=self.parent_agent,
            )
            try:
                data = json.loads(result)
                if isinstance(data, dict) and "error" in data:
                    raise ValueError(data["error"])
                return data.get("result", result) if isinstance(data, dict) else result
            except (json.JSONDecodeError, TypeError):
                return str(result)
        except Exception as e:
            return f"[{type(e).__name__}] {e}"

    async def _run_layer_async(
        self, layer: list[str], results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        重写 Flow._run_layer_async：
        每层的方法用信号量限制并发，
        每个方法调用 run_agent(role, goal, context)。
        """
        meta = self._meta()

        async def run_one(name: str):
            method = meta["nodes"][name]
            srcs = self._sources_for(name)
            args = [results[s] for s in srcs if s in results]

            # 收集上游结果作为context
            context_parts = []
            for i, src in enumerate(srcs):
                context_parts.append(f"[From {src}]:\n{args[i] if i < len(args) else ''}")
            context = "\n\n".join(context_parts)

            # 获取role和goal
            role: AgentRole = getattr(method, "__agent_role__", AgentRole.DEVELOPER)
            goal_tpl: str = getattr(method, "__agent_goal_template__", name)

            # 渲染goal：{input} 或 {0}, {1} 等
            if args:
                try:
                    goal = goal_tpl.format(input=args[0], **dict(zip([f"{{{i}}}" for i in range(len(args))], args)))
                except (IndexError, KeyError):
                    goal = goal_tpl + "\n\n" + context
            else:
                goal = goal_tpl

            # 在线程池中运行同步的run_agent
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.run_agent(role, goal, context)
            )
            return name, result

        # 每层并发，受 max_concurrent 限制
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_run_one(name: str):
            async with semaphore:
                return await run_one(name)

        tasks = [bounded_run_one(n) for n in layer if n in meta["nodes"]]
        if not tasks:
            return results

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        new_results = dict(results)
        for o in outcomes:
            if isinstance(o, Exception):
                raise RuntimeError(f"FlowBasedMultiAgentCoordinator error: {o}") from o
            name, val = o
            new_results[name] = val
        self._results = new_results
        return new_results

    def get_results(self) -> dict[str, Any]:
        """返回所有步骤的结果"""
        return dict(self._results)

    def execute_sync(self) -> dict:
        """同步入口"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()

    async def run_async(self) -> dict[str, Any]:
        """执行装饰器定义的工作流"""
        self._results = await super().run_async()
        return self._results


def coordinated_delegate(
    goal: str,
    context: Optional[str] = None,
    tasks: Optional[list[dict]] = None,
    role_hint: Optional[str] = None,
    max_concurrent: int = 3,
    synthesize: bool = True,
    parent_agent=None,
) -> str:
    """
    高级多Agent委托 — 替代简单的delegate_task

    使用方式：
        result = coordinated_delegate(
            goal="Build a web app with auth and real-time updates",
            context="Using React + FastAPI + WebSockets",
            max_concurrent=3,
            synthesize=True,
            parent_agent=self,
        )

    或者手动指定DAG任务：
        result = coordinated_delegate(
            goal="Build and test a web app",
            tasks=[
                {"task_id": "research", "goal": "Research stack options", "depends_on": []},
                {"task_id": "build", "goal": "Build the app", "depends_on": ["research"]},
                {"task_id": "test", "goal": "Write tests", "depends_on": ["build"]},
            ],
            parent_agent=self,
        )

    内部流程：
    1. Coordinator分解任务为DAG（若tasks为空）
    2. 按依赖并行执行多个角色Agent
    3. 结果写入SharedMemory
    4. Synthesizer聚合为最终输出
    """
    if parent_agent is None:
        return json.dumps({"error": "coordinated_delegate requires parent_agent"})

    coordinator = MultiAgentCoordinator(
        parent_agent=parent_agent,
        goal=goal,
        context=context,
        tasks=tasks,
        max_concurrent=max_concurrent,
    )

    coordinator.decompose(role_hint=role_hint)

    # 使用同步执行（内部用线程池）
    exec_result = coordinator.execute_sync()

    if synthesize:
        final_output = coordinator.synthesize(exec_result)
        return json.dumps({
            "success": True,
            "execution": exec_result,
            "synthesis": final_output,
        }, indent=2, ensure_ascii=False)
    else:
        return json.dumps({
            "success": True,
            "execution": exec_result,
        }, indent=2, ensure_ascii=False)
