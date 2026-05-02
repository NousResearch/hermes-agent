# Sprint 02 — 当前架构

## 架构变更

### 新增模块

```
hermes-agent/
  agent/
    review_gate.py        ← ReviewGate 类 + ReviewResult + 阻断/降级逻辑
    review_templates.py   ← 3 套硬编码审查模板 + get_checks_for_category()
```

### 修改模块

```
hermes-agent/
  run_agent.py            ← 集成 ReviewGate.check() + 阻断/降级结果注入
```

## 架构图（更新后）

```
User Input
  → TaskCard.create()                    [Sprint 1]
  → [Agent 执行循环...]
  → result_summary 生成
  → ReviewGate.check()                   [Sprint 2 NEW]
      ├── rule_checks (自动)
      │   ├── has_task_card
      │   ├── has_compiled_intent
      │   ├── has_result_summary
      │   └── success_criteria_addressed
      ├── llm_checks (模板定义)
      │   ├── brand_strategy: 5 checks
      │   ├── architecture_review: 5 checks
      │   └── universal: 4 checks
      ├── _compute_score()
      ├── _evaluate_blocking()
      │   ├── 阻断: needs_revision=true + instruction
      │   └── 降级: review_exhausted=true + risks
      └── _collect_risks()
  → 阻断? → status=reviewing
  → 降级? → result 标注 risks
  → EventLog [Sprint 1]
  → _persist_session [Sprint 1 safety net]
  → return result (含 review_blocked / review_risks)
```

## ReviewResult 数据模型

```
ReviewResult
├── task_id: str
├── checked_at: ISO timestamp
├── rule_checks: {check_id: {pass, question, detail}}
├── llm_checks: {check_id: {pass, question, evidence}}
├── quality_score: int (0-100)
├── risks: [str]
├── needs_revision: bool
├── revision_instruction: str
├── revision_count: int
└── review_exhausted: bool
```

TaskCard 新增字段：
```
TaskCard.review_result = ReviewResult.to_dict()
```

## 状态流转（更新后）

```
pending → running → {reviewing} → completed   [NEW: reviewing]
                 ↘ failed
                 ↘ partial
                 ↘ {reviewing} (blocked) → completed (after revision)
                 ↘ {reviewing} → completed (degraded, review_exhausted)
```

## 存储架构

```
~/.hermes/
  events.db             ← Sprint 1
  task_cards/           ← Sprint 1 (review_result 字段新增)
    {task_id}.json
  state.db              ← 已有
  sessions/             ← 已有
```
