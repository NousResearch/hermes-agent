# agents/gpt_reviewer.py
"""GPT-5.4 reviewer + synthesizer agent — double role in the research pipeline."""

import asyncio
from datetime import datetime
from collections import defaultdict

from agents.base import BaseAgent
from agents.llm_client import LLMClient

GPT_REVIEWER_SYSTEM = """你是 GPT-5.4，AI2050-OpenOne 项目的首席评审员。你有双重职责：

## 角色A: 独立评审
当 DeepSeek 研究者向 #review 频道汇报成果后，从以下维度评审：
1. **科学严谨性**: 方法论是否合理？统计检验是否充分？
2. **理论一致性**: 发现是否与统一状态系统 X(t) = (a,r,f,g,q,b,p,h,m,c) 兼容？
3. **数据充分性**: 样本量、边界情况、统计功效
4. **可复现性**: 实验是否可被独立复现？

## 角色B: 综合共识
当收集到所有评审（自己的 + Claude的）后：
1. 识别共同结论和分歧点
2. 判断当前研究置信度变化
3. 综合成统一共识 → 发布到 #general

## 对 DeepSeek 汇报的评审格式
```
## GPT-5.4 评审
### 科学严谨性
{分析}
### 理论一致性 (与 X(t) 框架)
{兼容性检查}
### 数据充分性
{分析}
### 可复现性
{评估}
### 建议
{具体建议}
### 评分
| 维度 | 分数 (0-1) |
|------|-----------|
| 科学严谨性 | ... |
| 理论一致性 | ... |
| 数据充分性 | ... |
| 可复现性 | ... |
| **总评** | ... |
```

## 综合共识格式
```
## 综合共识 #{n}
### 任务: {task_id}
### 研究结论: {ACCEPTED|REVISION_NEEDED|INCONCLUSIVE}

### 置信度变化
- 当前任务置信度: {0-1}
- 对整体框架影响: {+/-值}

### 一致意见
{共同结论}

### 分歧与解决
{分歧点处理}

### 建议的下一步
1. {具体建议}
2. ...

### 拼图状态更新建议
{格子变更}
```
"""


class GPTReviewer(BaseAgent):
    """GPT-5.4: reviews research and synthesizes consensus."""

    def __init__(self, name: str, provider: str = "openai", model: str = "gpt-5.4",
                 server_url: str = "ws://localhost:8765",
                 temperature: float = 0.5, max_tokens: int = 8192):
        super().__init__(name, server_url)
        self.llm = LLMClient(provider=provider, model=model,
                             temperature=temperature, max_tokens=max_tokens)
        # Track reviews per task_id
        self._pending_reviews: dict[str, list[dict]] = defaultdict(list)
        self._reviewed_tasks: set = set()
        self._synthesis_count = 0

    async def handle_message(self, message: dict):
        """Route messages by channel and type."""
        channel = message.get("channel", "")
        msg_type = message.get("msg_type", "")

        if channel == "#review" and msg_type == "review":
            await self._review(message)
        elif channel == "#consensus" and msg_type == "review_analysis":
            await self._collect_for_synthesis(message)
        elif channel == "#general" and msg_type == "consensus":
            # Already synthesized, skip
            pass

    async def _review(self, message: dict):
        """Review DeepSeek's research output."""
        content = message.get("content", "")
        metadata = message.get("metadata", {})
        task_id = metadata.get("task_id", "unknown")
        sender = message.get("sender", "")

        # Don't review our own messages
        if sender == self.name:
            return

        # Don't re-review the same task
        if task_id in self._reviewed_tasks:
            return

        print(f"\n[{self.name}] 🔍 Reviewing task: {task_id}")

        review_prompt = f"""请评审以下研究成果：

研究内容:
{content}

请从科学严谨性、理论一致性、数据充分性、可复现性四个维度进行评审。"""
        
        try:
            review_text = await self.llm.chat(
                system_prompt=GPT_REVIEWER_SYSTEM,
                user_message=review_prompt,
            )
        except Exception as e:
            review_text = f"评审失败: {e}"

        await self.publish(
            channel="#consensus",
            content=review_text,
            msg_type="review_analysis",
            metadata={"reviewer": self.name, "reviewer_role": "reviewer+synthesizer",
                      "task_id": task_id, "provider": "openai", "model": "gpt-5.4"}
        )
        self._reviewed_tasks.add(task_id)
        # Also track for synthesis
        self._pending_reviews[task_id].append({
            "sender": self.name, "content": review_text, "role": "gpt"
        })
        print(f"[{self.name}] ✅ Review posted to #consensus")

    async def _collect_for_synthesis(self, message: dict):
        """Collect reviews for synthesis when both are in."""
        metadata = message.get("metadata", {})
        task_id = metadata.get("task_id", "unknown")
        sender = message.get("sender", "")

        # Don't collect if already synthesized
        if task_id in self._synthesized_tasks:
            return

        self._pending_reviews[task_id].append({
            "sender": sender, "content": message.get("content", ""),
            "role": metadata.get("reviewer_role", "")
        })

        # Wait for both reviews (GPT + Claude)
        reviewers_seen = {r["sender"] for r in self._pending_reviews[task_id]}
        if self.name in reviewers_seen and len(reviewers_seen) >= 2:
            await self._synthesize(task_id)

    async def _synthesize(self, task_id: str):
        """Synthesize all reviews into a unified consensus."""
        self._synthesis_count += 1
        reviews = self._pending_reviews.get(task_id, [])
        if len(reviews) < 2:
            return

        print(f"\n[{self.name}] 🧠 Synthesizing consensus #{self._synthesis_count} for {task_id}")

        reviews_text = "\n\n---\n\n".join([
            f"### {r['sender']} ({r.get('role', 'unknown')})\n{r['content']}"
            for r in reviews
        ])

        synthesis_prompt = f"""请综合以下两份评审，产出一份统一共识：

{reviews_text}

请识别共同结论、分歧点，给出明确的决策（ACCEPTED/REVISION_NEEDED/INCONCLUSIVE），
以及具体可执行的下一步建议。"""
        
        try:
            consensus = await self.llm.chat(
                system_prompt=GPT_REVIEWER_SYSTEM,
                user_message=synthesis_prompt,
                max_tokens=4096,
            )
        except Exception as e:
            consensus = f"综合失败: {e}"

        # Publish to #general
        await self.publish(
            channel="#general",
            content=consensus,
            msg_type="consensus",
            metadata={"task_id": task_id, "synthesis_count": str(self._synthesis_count),
                      "synthesizer": self.name}
        )
        # Track synthesized
        self._synthesized_tasks.add(task_id)
        # Clean up
        del self._pending_reviews[task_id]
        print(f"[{self.name}] 📢 Consensus #{self._synthesis_count} published to #general")

    _synthesized_tasks: set = set()

    async def run(self):
        self.on_message(self.handle_message)
        await super().run(["#review", "#consensus"])
