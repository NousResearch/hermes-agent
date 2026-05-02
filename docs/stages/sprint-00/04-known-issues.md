# Sprint 00 — 已知问题

## 1. 架构层面

### 1.1 无结构化任务卡片
- **现状**：Todo 工具只在内存中，无持久化，无 schema，无状态机。
- **影响**：无法追踪任务从创建到完成的完整生命周期。
- **2.8 对策**：Sprint 1 引入 Task Card schema + JSON 文件存储。

### 1.2 无事件日志系统
- **现状**：`agent/session_event_log.py` 不存在。会话日志只有 JSONL 对话记录和 state.db 中的 messages 表。
- **影响**：无法回答"为什么任务状态改变"、"为什么选择了某个 Agent"、"为什么 Review 通过/不通过"。
- **2.8 对策**：Sprint 1 实现 Minimal Event Log（events.db + SQLite）。

### 1.3 无 Review Gate
- **现状**：任务从 running 直接完成，没有任何质量审查环节。
- **影响**：交付质量完全依赖主 Agent 当前判断力，无系统化质量门禁。
- **2.8 对策**：Sprint 2 实现 Review Gate + 静态审查模板。

### 1.4 无 Agent Router
- **现状**：主 Agent 默认自己完成所有任务。Delegate 工具可用但需要 Agent 手动判断何时使用。
- **影响**：缺少"按任务类型自动路由到正确 Agent"的能力。
- **2.8 对策**：Sprint 4 实现 Agent Router / Pipeline。

### 1.5 记忆系统无结构化元数据
- **现状**：MEMORY.md 用简单的 `类型: xxx / 范围: xxx` 注释，但不是真正的 YAML frontmatter。记忆类型、scope、confidence 无 schema 约束。
- **影响**：无法按 scope 过滤注入，无法做 confidence 排序。
- **2.8 对策**：Sprint 3 实现 Lightweight Memory（type/scope/confidence/source/last_verified_at）。

### 1.6 Skill 无权限边界
- **现状**：SKILL.md 的 frontmatter 有 `permissions` 字段预留，但不做运行时校验。
- **影响**：第三方 skill 可以执行任意操作。
- **2.8 对策**：Sprint 6 实现 Skill Permission MVP。

## 2. 工程层面

### 2.1 run_agent.py 过于庞大
- **问题**：13,737 行的单文件包含了对话循环、工具分发、模型适配、会话管理、资源清理等所有逻辑。
- **风险**：Sprint 1-2 需要修改 run_agent.py，在如此大的文件中定位修改点需要谨慎。
- **建议**：2.8 只做必要修改，不做文件拆分重构。

### 2.2 无 task_cards/ 和 events.db
- **状态**：当前 Hermes 数据目录下不存在 `task_cards/` 目录和 `events.db` 数据库。
- **对策**：Sprint 1 创建这些存储设施，独立于现有 state.db。

### 2.3 docs/stages/ 目录不存在
- **状态**：需要在 `~/.hermes/docs/stages/` 下创建 sprint-xx 子目录。
- **对策**：每个 Sprint 开始前创建对应目录。

### 2.4 firecrawl MCP 和 context MCP 均未接入
- **现状**：config.yaml 中只有 minimax 和 openchronicle 两个 MCP server。
- **影响**：任务书要求接入的 firecrawl MCP（网页抓取）和 context MCP（历史上下文）均不可用。
- **对策**：Sprint 2（按任务书原计划）或后续 Sprint 接入。

## 3. 兼容性风险

### 3.1 两套方案体系
- **问题**：存在两份升级方案文档：
  - `hermes-2.8-harness-engineering-plan.md`（工程方案，6 个 Sprint，以 Task Card + Event Log 为核心）
  - `hermes_2_8_harness_engineering_task_brief.md`（任务书，6 个 Sprint，以 Intent Card + Harness Foundation 为起点）
- **差异**：工程方案更偏实际工程边界和代码落点，任务书更偏概念架构和目录规划。
- **对策**：以工程方案为执行依据，任务书作为方向参考。冲突时优先适配当前仓库实际结构。

### 3.2 新旧记忆格式兼容
- **问题**：当前 Memory 条目格式不统一，有的有 frontmatter，有的没有。
- **对策**：Sprint 3 实现时，旧条目自动获得默认元数据（`type=memory, scope=global, confidence=medium`）。

### 3.3 Schema 版本向前兼容
- **约束**：工程方案要求 Sprint 间字段变更是向前兼容的（新增字段有默认值，不删已有字段）。
- **对策**：所有 schema 使用 `schema_version` 字段，每次变更递增版本号。

## 4. 测试覆盖

### 4.1 现有测试
- `tests/` 目录下有 40+ 测试文件
- 覆盖：CLI、Yuanbao、Trajectory Compressor、Session、Memory、Timezone、MySQL 等
- 测试框架：pytest + pytest-asyncio + pytest-xdist
- 默认跳过 integration 测试（`-m "not integration"`）

### 4.2 2.8 新增测试需求
- Sprint 1：TaskCard 序列化测试、EventLog 写入/读取测试
- Sprint 2：ReviewGate 阻断规则测试
- Sprint 3：Memory scope 过滤、旧格式兼容测试
- Sprint 4：Agent Router 路由规则测试
- Sprint 5：Event Log 复盘摘要测试
- Sprint 6：Skill Permission 校验测试
