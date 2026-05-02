# Sprint 02 — 当前功能

## 新增能力

### 1. Review Gate（质量门禁）

每个任务交付给用户前，自动经过 Review Gate 审查。两层检查：

**Rule-based（确定性）**：
- `has_task_card`：Task Card 是否存在
- `has_compiled_intent`：compiled_intent.real_task 是否非空
- `has_result_summary`：result_summary 是否非空
- `success_criteria_addressed`：是否覆盖所有 success_criteria

**LLM-based（语义检查模板）**：
- 按任务类别预定义检查问题（品牌类 5 个、架构类 5 个、通用 4 个）
- 当前返回 pending 状态，由主 Agent 评估后填写

### 2. 静态审查模板

按 `task_category` 硬编码三套模板：

| 模板 | 检查数 | 适用场景 |
|------|--------|---------|
| brand_strategy | 5 条 LLM | 品牌方案审查 |
| architecture_review | 5 条 LLM | 架构设计审查 |
| universal | 4 条 rule + 4 条 llm | 所有任务通用 |

### 3. 阻断与降级交付规则

**阻断交付**（任一成立）：
1. 无 Task Card
2. 无 compiled_intent
3. 无 result_summary
4. 未回应 success_criteria
5. 任一 rule_check 未通过
6. quality_score < 70 且 revision_count < 1
7. needs_revision = true 且 revision_count < 1

**允许降级交付**（标注 risk）：
- LLM 检查不确定
- 信息不足但已标注 limitation
- 子 Agent 失败但已说明风险和替代方案

**Revision 硬上限**：最多返工 1 次，超限后降级交付并标注 `review_exhausted: true`。

## 已有能力（继承 Sprint 0-1）

- Sprint 0：9 大类已有能力
- Sprint 1：Task Card + Minimal Event Log
