# Sprint 02 — 已知问题

## 1. 本 Sprint 限制

### 1.1 LLM 检查尚未自动执行

- **现状**：LLM-based 检查定义为模板问题，`ReviewGate.check()` 返回 `pass=None`（pending），不自动调用模型评估。
- **影响**：语义质量检查需要主 Agent 自行评估后填写 `llm_check_results`。
- **对策**：后续 Sprint 可集成辅助模型自动执行 LLM 检查。

### 1.2 success_criteria_addressed 使用简单启发式

- **现状**：通过关键词匹配判断 result 是否覆盖 success_criteria。取 criteria 前 3 个词在 result_summary 中搜索。
- **影响**：可能漏判语义相关但关键词不匹配的情况，或误判关键词出现但不实质性回应的情况。
- **对策**：后续可用 LLM 替代关键词匹配，但当前 heuristic 对 Sprint 2 够用。

### 1.3 品牌/架构审查模板仅含 LLM 检查

- **现状**：brand_strategy 和 architecture_review 模板全部为 LLM 检查，无 rule 检查。
- **影响**：这两类任务在当前自动审查中不会有任何通过的检查项（全部 pending），但也不会被阻断（pending 不计分）。
- **对策**：Sprint 3 接入 Memory 后可让 LLM 检查生效。

### 1.4 Review Gate 在子 Agent 任务中不运行

- **现状**：子 Agent 通过 `delegate_tool` 执行时，不经过 Review Gate。
- **影响**：子 Agent 输出未经审查。
- **对策**：Sprint 4（Agent Router/Pipeline）可考虑向子 Agent 传播 Review Gate。

## 2. 与工程方案的差异

### 2.1 方案中的 LLM 调用未实现

工程方案提到 "LLM-based：语义质量检查，由主 Agent 调用辅助模型判断"。当前实现仅提供模板定义，不自动调用辅助模型。主 Agent 可根据模板问题自行评估。

### 2.2 阻断规则与方案完全一致

7 条阻断规则、降级交付规则、revison 硬上限（1 次）均按方案实现。

## 3. 对 Sprint 3 的影响

### 前置依赖确认

Sprint 3（Lightweight Memory）需要：
- Review Gate 的 `matches_user_preferences` 和 `matches_project_context` 检查 → 已在 templates 中定义
- Memory 系统提供偏好和项目规则 → Sprint 3 实现

### 风险提示

- `review_templates.py` 中的 universal 检查已包含 `matches_gu_style`（符合 Gu 表达偏好），Sprint 3 可接入 Memory 使其生效
- 评分中的 LLM 检查当前不计分，Sprint 3 接入 Memory 后 LLM 检查可自动化

## 4. 遗留问题

- run_agent.py 超过 14,000 行（Sprint 0-2 持续新增）
- LLM 检查自动化依赖辅助模型调用（未排入当前 Sprint）
- 子 Agent 审查未覆盖
