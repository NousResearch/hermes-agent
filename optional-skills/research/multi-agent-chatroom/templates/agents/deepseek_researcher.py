# agents/deepseek_researcher.py
"""DeepSeek-V4-Pro researcher agent — executes research tasks and reports results."""

import os
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

from agents.base import BaseAgent
from agents.llm_client import LLMClient
from server.config import load_config, get_agent_config

DEEPSEEK_SYSTEM_PROMPT = """你是 AI2050-OpenOne 项目的核心研究者。
你的任务是执行科学分析、验证假设、编写代码并产出研究报告。

## 项目目标
逆向破解深度神经网络的数学原理，从语言能力中反推大脑编码机制和智能数学原理。

## 统一框架
X(t) = (a, r, f, g, q, b, p, h, m, c)
代表激活、回返一致性、纤维复用、门控路由、上下文条件化、偏置、可塑性、稳态偏差、拥塞负载、传送成本。

## 工作模式
1. 接收任务后，先阅读项目相关代码和数据文件
2. 编写 Python 分析脚本进行实验
3. 将结果整理为研究报告
4. 保存到 research/deepseek/{task_id}/ 目录
5. 汇报关键发现

## 汇报格式（严格遵循）
```
## 任务: {task_id} - {task_title}
### 执行摘要
{3-5句话核心发现}

### 关键数据
| 指标 | 值 | 解读 |
|------|-----|------|
| ... | ... | ... |

### 方法论
{分析方法简述}

### 完整报告
research/deepseek/{task_id}/report.md

### 置信度评估
- 内部置信度: {0.0-1.0}
- 主要不确定性来源: {描述}
```
"""

REPORT_TEMPLATE = """# {task_id}: {task_title}

> 执行时间: {timestamp}
> 研究者: deepseek-v4-pro
> 状态: {status}

## 1. 任务描述
{task_description}

## 2. 方法论
{methodology}

## 3. 实验过程

### 3.1 数据准备
{data_preparation}

### 3.2 分析脚本
{analysis_script}

### 3.3 结果
{results}

## 4. 关键发现
{key_findings}

## 5. 与统一框架的关系
{framework_relation}

## 6. 已知规律兼容性检查
{law_compatibility}

## 7. 置信度评估
- 内部置信度: {confidence}
- 不确定性来源: {uncertainty}

## 8. 建议的下一步
{next_steps}
"""


class DeepSeekResearcher(BaseAgent):
    """DeepSeek-V4-Pro researcher that analyzes AI2050 project data and reports results."""

    def __init__(self, name: str, provider: str = "deepseek", model: str = "deepseek-v4-pro",
                 workdir: str = None, server_url: str = "ws://localhost:8765",
                 temperature: float = 0.3, max_tokens: int = 16384):
        super().__init__(name, server_url)
        self.llm = LLMClient(provider=provider, model=model,
                             temperature=temperature, max_tokens=max_tokens)
        self.workdir = Path(workdir) if workdir else Path.cwd()
        self.current_task = None

    async def handle_message(self, message: dict):
        """Route incoming messages by channel."""
        channel = message.get("channel", "")
        msg_type = message.get("msg_type", "")

        if channel == "#tasks" and msg_type == "task":
            await self._handle_task(message)
        elif channel == "#general" and msg_type == "consensus":
            await self._handle_consensus(message)

    async def _handle_task(self, message: dict):
        """Execute a research task."""
        task_content = message.get("content", "")
        metadata = message.get("metadata", {})
        task_id = metadata.get("task_id", f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        task_title = metadata.get("title", "Untitled Task")

        self.current_task = {"id": task_id, "title": task_title, "content": task_content}
        print(f"\n[{self.name}] 📋 Received task: {task_id} - {task_title}")

        # Step 1: Analyze the task and read relevant files
        analysis_prompt = f"""任务: {task_title}
描述: {task_content}

请执行以下步骤：
1. 分析任务涉及哪些项目文件（data/, nfb_data/, models/, research/ 等）
2. 设计分析方案
3. 编写 Python 分析脚本
4. 在你本地执行脚本（模拟执行，给出预期输出和分析结果）
5. 产出完整研究报告

工作目录: {self.workdir}

请给出完整的研究报告，严格按照 REPORT_TEMPLATE 格式。"""
        
        print(f"[{self.name}] 🔬 Analyzing task...")
        try:
            report = await self.llm.chat(
                system_prompt=DEEPSEEK_SYSTEM_PROMPT,
                user_message=analysis_prompt,
            )
        except Exception as e:
            report = f"⚠️ LLM 调用失败: {e}\n请检查 API Key 配置。"

        # Step 2: Save report
        report_dir = self.workdir / "research" / "deepseek" / task_id
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")

        # Step 3: Extract summary (~300 chars)
        summary = report[:500] + ("..." if len(report) > 500 else "")

        # Step 4: Report to #review channel
        review_content = f"""## 任务: {task_id} - {task_title}
### 执行摘要
{summary}

### 关键数据
详见完整报告

### 完整报告
research/deepseek/{task_id}/report.md

### 置信度评估
- 内部置信度: 见报告
"""
        await self.publish(
            channel="#review",
            content=review_content,
            msg_type="review",
            metadata={"task_id": task_id, "title": task_title, "report_path": str(report_path)}
        )
        print(f"[{self.name}] ✅ Results published to #review")

    async def _handle_consensus(self, message: dict):
        """Read consensus and produce a summary for the next iteration."""
        consensus = message.get("content", "")
        print(f"[{self.name}] 📊 Received consensus, producing summary...")

        summary_prompt = f"""以下是评审委员会的综合共识：

{consensus}

请基于共识产出以下内容：
1. 当前任务结论（ACCEPTED/REVISION_NEEDED/INCONCLUSIVE）
2. 对整体研究框架的影响
3. 建议的下一任务方向
4. 拼图格子更新建议

格式：
```
## DeepSeek 总结
### 当前任务结论: {结论}
### 框架影响
{影响分析}
### 下一任务建议
{建议}
### 拼图更新
{格子更新}
```
"""
        try:
            summary = await self.llm.chat(
                system_prompt="你是 AI2050-OpenOne 研究者，负责吸收评审共识并规划下一步。",
                user_message=summary_prompt,
                max_tokens=2048,
            )
        except Exception as e:
            summary = f"总结生成失败: {e}"

        await self.publish(
            channel="#general",
            content=summary,
            msg_type="researcher_summary",
            metadata={"task_id": self.current_task.get("id", "") if self.current_task else ""}
        )
        print(f"[{self.name}] 📝 Summary published to #general")

    async def run(self):
        self.on_message(self.handle_message)
        await super().run(["#tasks", "#general"])
