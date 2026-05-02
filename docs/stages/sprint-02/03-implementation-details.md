# Sprint 02 — 实现细节

## 1. agent/review_templates.py

### 模板结构

```python
REVIEW_TEMPLATES = {
    "brand_strategy": {
        "name": "品牌方案审查",
        "checks": [
            {"id": "brand_core", "question": "...", "type": "llm"},
            ...
        ],
    },
    "architecture_review": { ... },
    "universal": { ... },
}
```

### 关键函数

- `get_checks_for_category(task_category)`：合并 category-specific + universal 检查
- `get_template_name(task_category)`：返回模板的可读名称

### 设计约束

- 硬编码，不读 Memory，不读数据库
- 每个 check 有稳定的 `id`，用于 ReviewResult 中引用
- `type: "rule"` = 确定性自动执行，`type: "llm"` = 语义判断

## 2. agent/review_gate.py

### ReviewGate 类

```python
class ReviewGate:
    def check(task_card, result_summary, llm_check_results, previous_review) -> ReviewResult
```

- `llm_check_results`：可选，预计算的 LLM 检查结果。不传则所有 LLM 检查标记为 pending
- `previous_review`：可选，用于 revision 计数和 exhaustion 判断

### Rule Check 实现

```python
def _run_rule_check(check, task_card, result_summary):
    - has_task_card:          bool(task_card and task_card.task_id)
    - has_compiled_intent:    bool(intent and intent.real_task)
    - has_result_summary:     bool(result_summary.strip())
    - success_criteria_addressed: 关键词匹配 heuristic
        - 无 criteria → 跳过（pass=True）
        - 有 criteria → 检查 result 是否包含 criteria 的关键词
```

### 评分逻辑

```python
def _compute_score(rule_checks, llm_checks):
    score = (passed_checks / evaluated_checks) * 100
    - rule checks: 全部计入（pass=True → +1, pass=False → +0）
    - llm checks: pass=True → 计入（+1），pass=False → 计入（+0），pass=None → 不计入
    - 无 evaluated checks → 返回 100
```

Pending LLM checks **不降低分数**，避免误伤未启用 LLM 审查的任务。

### 阻断判断

```python
def _evaluate_blocking(rule_checks, llm_checks, quality_score, revision_count):
    reasons = []
    - 4 个结构性检查任一不通过 → 阻断
    - 任一 rule check 失败 → 阻断
    - quality_score < 70 且 revision_count < 1 → 阻断
    return (needs_revision, instruction_string)
```

### 静态方法

- `ReviewGate.is_blocked(review_result)`：硬阻断（非降级交付）
- `ReviewGate.allows_degraded_delivery(review_result)`：是否允许降级交付

## 3. run_agent.py 修改

### 改动位置 1：Import + __init__

```python
from agent.review_gate import ReviewGate
# ...
self._review_gate = ReviewGate()
```

### 改动位置 2：Review Gate 调用（line 13364 后）

```python
tc = getattr(self, "_current_task_card", None)
if tc is not None and completed and final_response:
    review_result = self._review_gate.check(
        task_card=tc,
        result_summary=(final_response or "")[:500],
    )
    tc.review_result = review_result.to_dict()
    save_task_card(tc)
    # ... 阻断/降级处理 ...
```

- 仅在 `completed=True` 且有 `final_response` 时运行
- 阻断时 → `tc.status = "reviewing"` + 写 `status_changed` 事件
- 降级时 → 收集 risks 到 `_review_risks`

### 改动位置 3：Result 注入

```python
result = {
    ...
    "review_blocked": _review_blocked,
    "review_risks": _review_risks,
}
```

`review_blocked` 和 `review_risks` 传递给调用方（CLI/Gateway），用于向用户展示审查结果。

## 4. 错误处理

| 场景 | 行为 |
|------|------|
| Review Gate 异常 | logger.warning，不阻断主流程 |
| save_task_card 失败 | 异常穿透，review result 未持久化 |
| LLM 检查未提供 | 标记为 pending，不影响评分 |
| 两次 revision 后仍不通过 | review_exhausted=true，降级交付 |
