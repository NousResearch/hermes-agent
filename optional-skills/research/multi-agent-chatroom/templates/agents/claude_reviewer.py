# agents/claude_reviewer.py
"""Claude Opus 4.6 reviewer agent — focuses on mathematical rigor and framework compatibility."""

from agents.base import BaseAgent
from agents.llm_client import LLMClient

CLAUDE_REVIEWER_SYSTEM = """你是 Claude Opus 4.6，AI2050-OpenOne 项目的数学严谨性评审员。

## 评审职责
对 DeepSeek 研究者的成果进行独立评审，重点：

### 1. 数学严谨性
- 推导是否严密？
- 假设是否明确？
- 统计检验是否恰当？
- 是否存在未检查的边界条件？

### 2. 统一框架兼容性
检查发现与统一状态系统 X(t) = (a, r, f, g, q, b, p, h, m, c) 的一致性：
- a/r: 局部激活与回返一致性
- f/g: 纤维复用与门控路由
- q/b: 上下文条件化与偏置
- p/h/m/c: 可塑性、稳态偏差、拥塞负载、传送成本

### 3. 已知规律冲突检测
对照已确认的20条规律（四层结构）：
- R1-R5: 第零层 DNN组件编码机制
- R6-R10: 第一层 几何动力学
- R11-R15: 第二层 概念编码与传播
- R16-R18: 第三层 语法-语义解耦
- R19-R20: 第四层 属性编码机制

### 4. 反例/漏洞识别
- 哪些边界情况可能推翻当前结论？
- brain_plane → falsification_plane 传播路径是否被考虑？
- evidence_isolation_clause 是否被满足？

## 评审格式
```
## Claude Opus 4.6 评审

### 数学严谨性
{详细分析}

### 统一框架兼容性
| 分量 | 兼容性 | 说明 |
|------|--------|------|
| a (激活) | ✅/⚠️/❌ | ... |
| r (回返) | ✅/⚠️/❌ | ... |
| ... (共10个分量) | ... | ... |

### 已知规律冲突检查
- R1-R5 (组件编码): ...
- R6-R10 (几何动力学): ...
- R11-R15 (概念编码): ...
- R16-R18 (语法-语义): ...
- R19-R20 (属性编码): ...

### 潜在反例与漏洞
{identified issues}

### 改进建议
1. {具体建议}
2. ...

### 评分
| 维度 | 分数 (0-1) |
|------|-----------|
| 数学严谨性 | ... |
| 框架兼容性 | ... |
| 反例鲁棒性 | ... |
| **总评** | ... |
```
"""


class ClaudeReviewer(BaseAgent):
    """Claude Opus 4.6 — mathematical rigor reviewer."""

    def __init__(self, name: str, provider: str = "anthropic", model: str = "claude-opus-4-6",
                 server_url: str = "ws://localhost:8765",
                 temperature: float = 0.4, max_tokens: int = 8192):
        super().__init__(name, server_url)
        self.llm = LLMClient(provider=provider, model=model,
                             temperature=temperature, max_tokens=max_tokens)
        self._reviewed_tasks: set = set()

    async def handle_message(self, message: dict):
        """Handle incoming #review messages."""
        channel = message.get("channel", "")
        msg_type = message.get("msg_type", "")

        if channel != "#review" or msg_type != "review":
            return

        sender = message.get("sender", "")
        if sender == self.name:
            return

        metadata = message.get("metadata", {})
        task_id = metadata.get("task_id", "unknown")

        if task_id in self._reviewed_tasks:
            return

        content = message.get("content", "")
        task_title = metadata.get("title", "Unknown")

        print(f"\n[{self.name}] 🔬 Evaluating task: {task_id}")

        review_prompt = f"""请评审以下研究：

任务: {task_title}
研究结果:
{content}

请从数学严谨性、统一框架兼容性、已知规律冲突、潜在反例四个维度进行全面评审。"""

        try:
            review_text = await self.llm.chat(
                system_prompt=CLAUDE_REVIEWER_SYSTEM,
                user_message=review_prompt,
            )
        except Exception as e:
            review_text = f"⚠️ Claude 评审失败: {e}"

        await self.publish(
            channel="#consensus",
            content=review_text,
            msg_type="review_analysis",
            metadata={"reviewer": self.name, "reviewer_role": "mathematical-rigor",
                      "task_id": task_id, "provider": "anthropic", "model": "claude-opus-4-6"}
        )
        self._reviewed_tasks.add(task_id)
        print(f"[{self.name}] ✅ Mathematical review posted to #consensus")

    async def run(self):
        self.on_message(self.handle_message)
        await super().run(["#review"])
