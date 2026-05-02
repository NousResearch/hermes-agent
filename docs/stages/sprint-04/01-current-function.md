# Sprint 04 — 当前功能

## 新增能力

### 1. Agent Router（智能路由）

按 `task_category` 自动决定执行模式和目标 Agent：

| task_category | 默认 mode | 目标 Agent |
|---|---|---|
| architecture_review | self_execute | — |
| code_analysis | pipeline | Kimi → Hermes |
| brand_strategy | self_execute | — |
| visual_design | single_agent | visual |
| research | single_agent | Kimi |
| document | pipeline | Kimi → Hermes |
| prompt_design | self_execute | — |
| other | self_execute | — |

### 2. 路由覆盖规则

- **用户显式指定**：`user_agent_override` → 强制 single_agent
- **能力缺口**：默认 Agent 缺少 required_capabilities → 自动追加
- **风险升级**：risk_level=high → 强制 pipeline + Review Gate

### 3. Pipeline 模式

```
Kimi 查资料 / 读代码
      ↓
Hermes 分析判断 / 撰写整合
      ↓
Claude Code 执行明确修改（仅当需要改文件）
      ↓
Hermes 验收（Review Gate）
```

### 4. Fallback 机制

- 子 Agent 失败 → 主 Agent 接管，标记信息不足
- Pipeline 任一步失败 → status=partial 或 blocked
- Claude Code 执行失败 → 回 Hermes 生成修复建议

### 5. 子 Agent 结果审查

新增 `agent_result_accepted` rule check：
- 无子 Agent → 跳过
- 有子 Agent 但 result 未体现审查 → 阻断
- 有子 Agent 且 result 体现审查 → 通过

## 已有能力（继承 Sprint 0-3）

完整继承。
