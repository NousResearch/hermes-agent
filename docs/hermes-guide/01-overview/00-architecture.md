# 系统架构总览

## 1.1 Hermes Agent 简介

Hermes Agent 是由 [Nous Research](https://nousresearch.com) 构建的**自改进 AI Agent 框架**。其核心特点是内置学习循环——从经验中创建技能、在使用中改进技能。

### 核心特性

| 特性 | 说明 |
|------|------|
| **多平台** | CLI、Terminal UI、Telegram、Discord、Slack、VS Code、Zed 等 |
| **模型无关** | 支持 OpenAI、Anthropic、OpenRouter、MiniMax 等 200+ 模型 |
| **自改进** | 从经验中自动创建和优化技能 |
| **持久记忆** | SQLite + FTS5 全文搜索、用户建模 |
| **研究就绪** | Batch 处理、RL 训练环境、轨迹压缩 |

---

## 1.2 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Hermes Agent                                  │
│                        (hermes-agent package)                           │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│    hermes_cli/      │  │    run_agent.py     │  │      gateway/       │
│  Interactive CLI    │  │   AIAgent (核心)     │  │   Messaging Gateway │
│                    │  │                      │  │                     │
│  - main.py (~200k) │  │  - run_conversation()│  │  - run.py (~287k)  │
│  - setup.py        │  │  - chat()           │  │  - platforms/      │
│  - commands.py     │  │  - Tool dispatch    │  │    - telegram      │
│  - config.py       │  │                      │  │    - discord       │
│  - auth.py         │  └──────────┬──────────┘  │    - slack          │
│  - skin_engine.py  │             │              │    - ...            │
└─────────────────────┘             ▼              └─────────────────────┘
                          ┌─────────────────────┐
                          │   model_tools.py    │
                          │  Tool orchestration │
                          │ handle_function_call│
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  tools/registry.py   │
                          │ Central Tool Dispatch│
                          └──────────┬──────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
 ┌────────▼─────┐           ┌────────▼─────┐          ┌────────▼─────┐
 │ file_tools  │           │  web_tools   │          │   mcp_tool   │
 │terminal_tool│           │ browser_tool │          │ delegate_tool│
 │    ...      │           │     ...      │          │     ...      │
 └─────────────┘           └──────────────┘          └──────────────┘

                    ┌─────────────────┐
                    │  hermes_state.py │
                    │   SessionDB      │
                    │  (SQLite + FTS5) │
                    └─────────────────┘
```

---

## 1.3 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           入口层                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  hermes        → hermes_cli/main.py:main()      [CLI 主入口]           │
│  hermes-agent  → run_agent.py:main()           [Agent 直接调用]        │
│  hermes gateway→ gateway/run.py                 [消息网关]              │
│  hermes-acp    → acp_adapter/entry.py           [编辑器集成]           │
│  hermes cron   → cron/scheduler.py              [定时任务]              │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          核心层                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  run_agent.py          AIAgent 核心类，对话循环                          │
│  cli.py                CLI 编排器                                        │
│  model_tools.py        工具编排与函数调用处理                             │
│  hermes_state.py       SessionDB (SQLite FTS5)                         │
│  toolsets.py           工具集定义                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          工具层                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  tools/registry.py      中心化工具注册表                                 │
│  tools/file_tools.py    文件操作工具                                    │
│  tools/terminal_tool.py 终端控制工具                                    │
│  tools/web_tools.py     Web 搜索/提取                                   │
│  tools/browser_tool.py 浏览器自动化                                     │
│  tools/mcp_tool.py      MCP 客户端                                      │
│  tools/delegate_tool.py 子 Agent 委托                                   │
│  tools/skills_tool.py   技能管理                                        │
│  ... (30+ 工具)                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         平台层                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  gateway/base.py        平台基类                                        │
│  gateway/run.py         网关主循环                                       │
│  gateway/session.py     会话管理                                         │
│  gateway/platforms/                                                    │
│    ├── telegram.py      Telegram 适配器                                 │
│    ├── discord.py       Discord 适配器                                   │
│    ├── slack.py         Slack 适配器                                     │
│    ├── feishu.py        飞书适配器                                       │
│    ├── wecom.py         企业微信                                         │
│    └── ...              其他平台                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1.4 入口点详解

### 1.4.1 CLI 入口 (`hermes`)

**文件**: `hermes_cli/main.py`

```python
# pyproject.toml 中的入口点定义
[project.scripts]
hermes = "hermes_cli.main:main"
```

**命令体系**:
```
hermes                    # 交互式 CLI
hermes setup             # 首次配置向导
hermes config            # 配置管理
hermes model             # 模型切换
hermes skills            # 技能管理
hermes tools             # 工具管理
hermes gateway           # 消息网关
hermes cron              # 定时任务
```

### 1.4.2 Agent 入口 (`hermes-agent`)

**文件**: `run_agent.py`

```python
# pyproject.toml 中的入口点定义
[project.scripts]
hermes-agent = "run_agent:main"
```

### 1.4.3 Gateway 入口 (`hermes gateway`)

**文件**: `gateway/run.py`

```python
# 子命令入口
hermes gateway           # 启动网关
hermes gateway telegram  # 仅启动 Telegram
hermes gateway discord   # 仅启动 Discord
```

### 1.4.4 ACP 入口 (`hermes-acp`)

**文件**: `acp_adapter/entry.py`

用于 VS Code、Zed、JetBrains 编辑器集成。

---

## 1.5 核心文件索引

| 文件路径 | 行数(估) | 功能 | 入口依赖 |
|---------|---------|------|---------|
| `run_agent.py` | ~500 | AIAgent 核心类 | 被 cli.py, batch_runner 调用 |
| `cli.py` | ~300 | CLI 编排器 | 导入 run_agent |
| `model_tools.py` | ~400 | 工具编排 | 被 run_agent 导入 |
| `hermes_state.py` | ~300 | SessionDB | 被多处导入 |
| `hermes_cli/main.py` | ~200k | CLI 主程序 | 入口点 |
| `gateway/run.py` | ~287k | 网关主程序 | 入口点 |
| `tools/registry.py` | ~300 | 工具注册表 | 被所有工具导入 |
| `toolsets.py` | ~200 | 工具集定义 | 被 model_tools 导入 |

---

## 1.6 关键设计模式

### 1.6.1 工具注册模式

每个工具文件通过 `@registry.register()` 装饰器在导入时注册自己：

```python
# tools/file_tools.py (示例结构)
def register_handlers(registry):
    registry.register("Read", read_handler, schema, ...)

# tools/registry.py
class Registry:
    def register(self, name, handler, schema, ...):
        self._handlers[name] = handler
        self._schemas[name] = schema
```

**触发时机**: `model_tools.py` 的 `_discover_tools()` 在首次使用时扫描导入。

### 1.6.2 Slash 命令模式

所有 CLI 命令定义在 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY`：

```python
# 自动生成：CLI 帮助、Telegram 菜单、Slack 路由、自动补全
COMMAND_REGISTRY = {
    "/help": CommandDef(...),
    "/model": CommandDef(...),
    "/skills": CommandDef(...),
    ...
}
```

### 1.6.3 平台适配器模式

```python
# gateway/base.py
class Platform(ABC):
    @abstractmethod
    async def send_message(self, message: Message) -> None: ...
    @abstractmethod
    async def get_updates(self) -> List[Update]: ...

# gateway/platforms/telegram.py
class Telegram(Platform):
    async def send_message(self, message: Message) -> None: ...
```

---

## 1.7 配置体系

### 配置文件位置

```
~/.hermes/
├── config.yaml          # 用户配置
├── .env                 # API 密钥
├── sessions.db          # SQLite 会话数据库
└── skills/              # 用户技能
```

### 配置优先级（从高到低）

1. **环境变量** (`HERMES_*`)
2. **CLI 参数** (`--model gpt-4`)
3. **用户配置** (`~/.hermes/config.yaml`)
4. **默认配置** (`hermes_cli/config.py::DEFAULT_CONFIG`)

---

## 1.8 下一步

- [核心概念详解](01-core-concepts.md) - 深入理解 Agent 对话循环、Tool 调用
- [AIAgent 核心模块](../02-core/01-aiagent.md) - run_agent.py 详细分析
- [CLI 系统详解](../04-cli-system/01-main.md) - 命令行接口架构

---

[← 返回总目录](../SUMMARY.md) | [核心概念 →](01-core-concepts.md)
