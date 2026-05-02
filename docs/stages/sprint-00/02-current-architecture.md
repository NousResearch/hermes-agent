# Sprint 00 — 当前架构

## 1. 整体架构概览

```
                    ┌─────────────────────────────────┐
                    │        Gateway / CLI             │
                    │  (cli.py, gateway/run.py)         │
                    │  多平台消息入口 / TUI 交互层        │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │         AIAgent (run_agent.py)    │
                    │  核心编排器，13,737 行              │
                    │  - 对话循环                        │
                    │  - 工具调用分发                     │
                    │  - 模型适配器选择                   │
                    │  - 上下文管理                       │
                    │  - 会话生命周期                     │
                    └──────┬───────────────┬───────────┘
                           │               │
          ┌────────────────▼──┐   ┌────────▼──────────┐
          │   agent/ 模块      │   │   tools/ 模块       │
          │  - 模型适配器       │   │  - 60+ 工具         │
          │  - Memory Manager  │   │  - Tool Registry    │
          │  - Skill Utils     │   │  - MCP Tool         │
          │  - Context Engine  │   │  - Delegate Tool    │
          │  - Prompt Builder  │   │  - Session Search   │
          └───────────────────┘   └────────────────────┘
                           │               │
          ┌────────────────▼───────────────▼──────────┐
          │            数据与存储层                      │
          │  - state.db (SQLite + FTS5)                 │
          │  - sessions/*.jsonl                         │
          │  - memories/MEMORY.md, USER.md              │
          │  - skills/*.md                              │
          │  - config.yaml                              │
          └────────────────────────────────────────────┘
```

## 2. 核心类关系

```
AIAgent (run_agent.py)
  ├── _memory_manager: MemoryManager
  │     └── BuiltinMemoryProvider / PluginProvider
  ├── _todo_store: TodoStore (in-memory)
  ├── session_id: str
  ├── session_log_file: Path (JSONL)
  ├── _session_db: HermesState (SQLite)
  └── 工具调用 → registry.get_handler(name)
```

## 3. 关键模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| **CLI 入口** | `cli.py` (11,483行) | TUI 交互、/ 命令、消息渲染 |
| **Gateway** | `gateway/run.py` | 多平台消息网关、会话路由 |
| **核心编排** | `run_agent.py` (13,737行) | AIAgent 类，对话循环，工具分发 |
| **状态存储** | `hermes_state.py` | SQLite sessions/messages 表，FTS5 搜索 |
| **记忆系统** | `tools/memory_tool.py` (586行) | MEMORY.md / USER.md 读写 |
| **记忆管理** | `agent/memory_manager.py` | 多 provider 记忆编排 |
| **Skill 工具** | `agent/skill_utils.py` | Frontmatter 解析、平台过滤 |
| **Skill 命令** | `agent/skill_commands.py` | /skill-name 命令处理 |
| **Skill 工具** | `tools/skills_tool.py` | skills_list / skill_view |
| **会话搜索** | `tools/session_search_tool.py` (591行) | FTS5 + LLM 摘要 |
| **委托工具** | `tools/delegate_tool.py` | 子 Agent 生成 |
| **Todo 工具** | `tools/todo_tool.py` | 内存任务列表 |
| **MCP 工具** | `tools/mcp_tool.py` | MCP 协议客户端 |
| **工具注册** | `tools/registry.py` | 中央注册表 |

## 4. 数据流（当前）

```
用户输入 (CLI/Gateway)
  → AIAgent.run_conversation()
    → 构建 system prompt (Memory + Skills + Context)
    → 循环：
      → 调用 LLM (通过模型适配器)
      → 解析 tool_calls
      → 执行工具 (registry.get_handler)
      → 将结果追加到消息历史
    → 将完整对话写入 session JSONL
    → 写入 state.db (sessions + messages 表)
```

## 5. 2.8 升级前缺失的架构层

```
当前架构：
  User → AIAgent → 直接执行 → 返回结果

2.8 目标架构：
  User → Intent Harness → Task Card → Capability Harness
       → Agent Router → Execution → Review Gate
       → Documentation Harness → Git Commit
       （全程 Event Log 记录）
```

当前缺失：
- **Intent Harness**：没有结构化的意图理解层
- **Task Card**：没有持久化的任务卡片
- **Event Log**：没有专用的事件日志系统
- **Review Gate**：没有交付前的质量审查
- **Agent Router**：没有基于任务类型的路由策略（只有手动 delegate）
- **Documentation Harness**：没有自动阶段文档生成
- **Git Workflow**：没有自动 Git 提交

## 6. 存储架构

```
~/.hermes/
  state.db               ← SQLite: sessions + messages + FTS5
  sessions/*.jsonl       ← 每会话独立 JSONL 文件 (522个)
  memories/
    MEMORY.md            ← Agent 笔记 (§ 分隔)
    USER.md              ← 用户偏好
  skills/                ← 33 个 skill 目录
  config.yaml            ← 全局配置
  models_dev_cache.json  ← 模型目录缓存
  events.db              ← 不存在（2.8 新增）
  task_cards/            ← 不存在（2.8 新增）
```

## 7. 技术栈总结

| 维度 | 选择 |
|------|------|
| **语言** | Python 3.11+ |
| **包管理** | setuptools + pyproject.toml |
| **异步** | asyncio + concurrent.futures |
| **CLI** | prompt_toolkit + rich |
| **数据验证** | pydantic |
| **数据库** | SQLite 3 + FTS5 |
| **模型 SDK** | anthropic, openai, google-genai |
| **配置** | YAML (config.yaml) |
| **测试** | pytest + pytest-asyncio + pytest-xdist |
| **Lint** | ruff |
| **类型检查** | ty (tachyon) |
| **容器** | Docker / Modal / Daytona / Vercel Sandbox |
