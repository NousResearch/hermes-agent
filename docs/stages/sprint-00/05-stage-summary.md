# Sprint 00 阶段总结

## 1. 本阶段目标

对 Hermes 当前代码仓库进行全面审计，理解技术栈、已有架构、现有能力和系统缺口，为 2.8 Harness Engineering 升级确定准确的工程落点。

## 2. 本阶段完成内容

- ✅ 阅读三份关键文档（工程方案、任务书、CLAUDE.md）
- ✅ 审查 hermes-agent/ 完整代码结构（300+ Python 文件）
- ✅ 分析 AIAgent 核心编排器架构（13,737 行 run_agent.py）
- ✅ 理解工具注册机制、Skill 系统、Memory 系统、Delegate 机制
- ✅ 审查 SQLite 状态存储 schema（sessions + messages + FTS5）
- ✅ 审查 MCP 配置现状（minimax + openchronicle，缺少 firecrawl + context）
- ✅ 确认 config.yaml 完整配置（模型、工具集、安全、终端、浏览器等）
- ✅ 审查会话存储格式（JSONL 文件 + state.db 双存储）
- ✅ 确认测试基础设施（pytest, 40+ 测试文件）

## 3. 当前功能总结

Hermes v0.11.0 是一个功能完善的多 Agent 系统，具备：
- 多模型适配（Anthropic、OpenAI、Gemini、Bedrock、DeepSeek、Kimi、MiniMax 等）
- 多平台接入（CLI TUI + 12+ 消息平台 gateway）
- 工具系统（60+ 内置工具 + MCP 集成）
- Skill 系统（24 个分类，YAML frontmatter）
- 记忆系统（文件记忆 + 插件 provider）
- 会话搜索（SQLite FTS5）
- 子 Agent 委托（隔离上下文 + 受限工具集）
- 定时任务（cron 调度器）
- 上下文压缩（自动压缩长对话）

## 4. 当前架构

```
CLI / Gateway → AIAgent (run_agent.py) → Model Adapters
                    ↕
            agent/ + tools/ 模块
                    ↕
    state.db + JSONL sessions + MEMORY.md
```

核心特点：
- **集中式编排**：AIAgent 是单一核心编排器
- **同步工具执行**：工具在线程池中执行，主循环同步等待
- **双存储**：会话数据同时写入 state.db（结构化查询）和 JSONL 文件（原始记录）
- **模块化工具**：通过 registry.py 自注册，松耦合

## 5. 关键实现细节

- **AIAgent**：13,737 行，__init__ 130+ 参数，run_conversation 是主循环
- **Tool Registry**：模块级 `register()` 调用，AST 解析发现
- **Memory**：文件级 § 分隔，支持 YAML frontmatter（基础），安全扫描
- **Skill**：SKILL.md + YAML frontmatter + 平台过滤 + 模板变量
- **Session DB**：SQLite WAL 模式，sessions + messages + FTS5，schema v11
- **Delegate**：ThreadPoolExecutor + 隔离上下文 + 摘要返回

## 6. 2.8 改造落点

### 新增文件（6 个 Sprint）

| Sprint | 新增文件 | 位置 |
|--------|---------|------|
| 1 | `agent/task_card.py` | hermes-agent/agent/ |
| 1 | `agent/session_event_log.py` | hermes-agent/agent/ |
| 2 | `agent/review_gate.py` | hermes-agent/agent/ |
| 2 | `agent/review_templates.py` | hermes-agent/agent/ |
| S1-S6 | `docs/stages/sprint-xx/*.md` | ~/.hermes/docs/stages/ |

### 修改文件

| Sprint | 文件 | 改动范围 |
|--------|------|---------|
| 1 | `run_agent.py` | 用户输入后构建 TaskCard，写入 task_created 事件 |
| 2 | `run_agent.py` | 任务交付前调用 ReviewGate.check() |
| 3 | `tools/memory_tool.py` | 重构：type/scope/元数据 + YAML frontmatter |
| 3 | `agent/review_gate.py` | 接入 Memory |
| 4 | `agent/task_card.py` | routing_basis + fallback 扩展 |
| 4 | `agent/review_gate.py` | agent_result_accepted/rejected |
| 5 | `agent/session_event_log.py` | 扩展事件类型 + 复盘摘要 |
| 5 | `agent/task_card.py` | 联动复盘摘要 |
| 5 | `tools/session_search_tool.py` | search_by_task_id |
| 6 | `agent/skill_utils.py` | manifest 解析 + trust 校验 |
| 6 | `agent/skill_commands.py` | manifest 加载 |
| 6 | `tools/skills_tool.py` | 权限展示 |

### 存储新增

```
~/.hermes/
  events.db              ← 新增：Event Log (SQLite)
  task_cards/            ← 新增：Task Card JSON 文件
```

## 7. 已知问题

1. **无 Task Card** — 缺少结构化任务定义和能力
2. **无 Event Log** — 无法追溯"为什么"
3. **无 Review Gate** — 缺少质量门禁
4. **无 Agent Router** — 路由逻辑靠 Agent 临时判断
5. **Memory 元数据不全** — 无 schema 化的 type/scope/confidence
6. **Skill 无权限校验** — 字段预留但不生效
7. **run_agent.py 过大** — 13,737 行单文件，修改需谨慎
8. **firecrawl/context MCP 未接入** — 任务书要求但未配置
9. **两套方案文档** — 工程方案 vs 任务书有差异，以工程方案为准

## 8. 下一阶段 (Sprint 1) 建议

**目标**：Task Card + Minimal Event Log

**执行计划**：
1. 新增 `agent/task_card.py`：TaskCard dataclass + JSON 序列化
2. 新增 `agent/session_event_log.py`：EventLog + SQLite events.db + append-only 写入
3. 修改 `run_agent.py`：用户输入后构建 TaskCard，写入 task_created 事件
4. 验证：Task Card JSON 落盘 + events.db 有 task_created
5. 生成 `docs/stages/sprint-01/` 阶段文档
6. Git commit

**风险提示**：
- `run_agent.py` 13,737 行，需要在正确的位置插入 TaskCard 创建逻辑
- 确保 events.db 路径管理正确（使用 hermes_constants.get_hermes_home()）
- Task Card 写入失败必须抛出异常，符合方案中的 fail-fast 设计

## 9. 验收结果

- ✅ 已理解当前仓库结构
- ✅ 已说明当前系统已有能力（9 大类能力）
- ✅ 已说明当前系统缺口（7 个主要缺口）
- ✅ 已提出 2.8 改造落点（新增/修改文件 + 存储新增）
- ✅ Sprint 00 文档已生成（5 个文件）

## 10. Git 提交说明

待执行：
```bash
git add .
git commit -m "docs: add sprint 00 repository audit"
```
