# agents/supervisor.py
"""Supervisor agent — manages the research task queue and decision loop."""

import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from agents.base import BaseAgent
from agents.llm_client import LLMClient
from server.config import load_config


SUPERVISOR_SYSTEM = """你是 AI2050-OpenOne 研究项目的调度监督者。

## 职责
1. 管理研究任务队列
2. 分配任务到 #tasks 频道
3. 监听 #general 频道接收共识和总结
4. 基于结果决定：继续、修订、或跳转下一任务

## 决策规则
- 收到 ACCEPTED 共识 → 推进到下一任务
- 收到 REVISION_NEEDED → 将任务重新入队（附带修订说明）
- 收到 INCONCLUSIVE → 记录并推进（不阻塞）
- DeepSeek 总结到达后 → 更新进度追踪

## 输出格式
- 任务分配: msg_type="task", metadata={"task_id": "T00N", "title": "...", "iteration": "N"}
- 进度公告: msg_type="system", channel="#general"
"""


class Supervisor(BaseAgent):
    """Manages the multi-agent research workflow."""

    def __init__(self, name: str, task_registry_path: str = None,
                 server_url: str = "ws://localhost:8765",
                 max_iterations: int = 50, review_timeout: int = 180):
        super().__init__(name, server_url)
        self.task_registry_path = Path(task_registry_path) if task_registry_path else \
            Path(__file__).parent.parent / "tasks" / "task_registry.yaml"
        self.tasks = []
        self.current_task_idx = 0
        self.iteration = 0
        self.max_iterations = max_iterations
        self.review_timeout = review_timeout
        self._consensus_count = 0
        self._load_tasks()

    def _load_tasks(self):
        """Load tasks from YAML registry."""
        if self.task_registry_path.exists():
            with open(self.task_registry_path) as f:
                data = yaml.safe_load(f)
                self.tasks = data.get("tasks", [])
        else:
            # Default tasks if no registry
            self.tasks = [
                {"id": "T001", "title": "验证频谱分工振荡定理",
                 "description": "验证 Band1-5频谱 ↔ θ/α/β/γ神经振荡 对应关系"},
                {"id": "T002", "title": "验证正交编码最优性定理",
                 "description": "在能量约束下验证正交编码是否使信息传输率最大化"},
                {"id": "T003", "title": "brain_grounding 弱轴分析",
                 "description": "定位 field_observability 瓶颈，分析 brain→falsification 传播路径"},
            ]
        print(f"[{self.name}] Loaded {len(self.tasks)} tasks")

    async def handle_message(self, message: dict):
        """Process messages from #general channel."""
        channel = message.get("channel", "")
        msg_type = message.get("msg_type", "")

        if channel == "#general":
            if msg_type == "consensus":
                await self._handle_consensus(message)
            elif msg_type == "researcher_summary":
                await self._handle_summary(message)

    async def _handle_consensus(self, message: dict):
        """Process consensus from GPT-5.4."""
        content = message.get("content", "")
        self._consensus_count += 1
        print(f"\n[{self.name}] 📊 Received consensus #{self._consensus_count}")

        if "ACCEPTED" in content:
            print(f"[{self.name}] ✅ Task accepted, advancing to next")
            self.current_task_idx += 1
            self.iteration += 1
        elif "REVISION_NEEDED" in content:
            print(f"[{self.name}] 🔄 Task needs revision, re-queuing")
            # Re-assign with feedback
            if self.current_task_idx < len(self.tasks):
                self.tasks[self.current_task_idx]["description"] += \
                    f"\n\n[Revision #{self.iteration}] Feedback: {content[:300]}"
        else:
            print(f"[{self.name}] ⏸️ Inconclusive, advancing with note")
            self.current_task_idx += 1
            self.iteration += 1

    async def _handle_summary(self, message: dict):
        """Process DeepSeek's summary and decide next step."""
        content = message.get("content", "")
        print(f"[{self.name}] 📝 Received researcher summary")

        # Check if we should continue
        if self.current_task_idx >= len(self.tasks):
            await self.publish(
                channel="#general",
                content="🏁 所有研究任务已完成！",
                msg_type="system"
            )
            return

        if self.iteration >= self.max_iterations:
            await self.publish(
                channel="#general",
                content=f"⚠️ 达到最大迭代次数 ({self.max_iterations})，停止。",
                msg_type="system"
            )
            return

        # Wait briefly then assign next task
        await asyncio.sleep(2)

    async def assign_task(self):
        """Assign the next task from the queue."""
        if self.current_task_idx >= len(self.tasks):
            print(f"[{self.name}] 🏁 All tasks assigned")
            return

        task = self.tasks[self.current_task_idx]
        task_id = task["id"]
        title = task["title"]
        description = task.get("description", "")

        print(f"\n[{self.name}] 📋 Assigning task {task_id}: {title}")

        await self.publish(
            channel="#tasks",
            content=f"## 研究任务: {task_id} - {title}\n\n{description}",
            msg_type="task",
            metadata={
                "task_id": task_id,
                "title": title,
                "iteration": str(self.iteration),
                "assigned_at": datetime.now().isoformat(),
            }
        )

    async def run(self):
        """Start the supervisor workflow."""
        self.on_message(self.handle_message)

        await self.connect()
        await self.subscribe("#general")
        await self.subscribe("#consensus")

        # Read config to show active models
        config = load_config()
        coding_cfg = config.get("coding", {})
        reviewers_cfg = config.get("reviewers", [])

        reviewer_summary = ", ".join([
            f"{r['name']}({r['provider']}/{r['model']})" for r in reviewers_cfg
        ])

        # Announce start with config summary
        await self.publish(
            channel="#general",
            content=(
                f"🚀 AI2050-OpenOne 多模型协作启动\n"
                f"- 编程: {coding_cfg.get('name', 'coder')}({coding_cfg.get('provider','?')}/{coding_cfg.get('model','?')})\n"
                f"- 审核: {reviewer_summary}\n"
                f"- 任务: {len(self.tasks)} 个就绪\n"
            ),
            msg_type="system"
        )

        # Start by assigning the first task
        await asyncio.sleep(2)
        await self.assign_task()

        # Keep listening
        await self.listen()
