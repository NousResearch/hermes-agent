# Hermes Agent - AI 上下文文档

> 更新时间：2026-04-16 | 当前版本：v2026.4.13 | 上游版本：v2026.4.13

## 项目愿景

Hermes Agent 是一个**自我改进的 AI 代理**，由 [Nous Research](https://nousresearch.com) 构建。它是唯一具有内置学习循环的代理——能够从经验中创建技能、在使用过程中改进技能、自动持久化知识、搜索过去的对话，并跨会话建立深入的用户模型。

核心特性：
- **真正的终端界面**：完整的 TUI，支持多行编辑、斜杠命令自动补全、对话历史、中断重定向和流式工具输出
- **Web UI Dashboard**：基于浏览器的管理界面，支持配置管理、会话监控、技能管理、环境变量配置和日志查看
- **多平台集成**：Telegram、Discord、Slack、WhatsApp、Signal、CLI —— 单一网关进程支持所有平台
- **封闭学习循环**：代理策展记忆、定期推送、自主技能创建、技能自我改进、FTS5 会话搜索
- **定时自动化**：内置 cron 调度器，支持向任何平台传递
- **委派和并行化**：生成隔离的子代理用于并行工作流
- **随处运行**：六种终端后端（本地、Docker、SSH、Daytona、Singularity、Modal）
- **备份和恢复**：完整备份和快速快照功能，支持安全的 SQLite 复制
- **OAuth 认证**：Web Dashboard 中的 OAuth 提供商管理

## 架构总览

### 技术栈
- **语言**：Python 3.11+、TypeScript
- **核心框架**：OpenAI SDK、Anthropic SDK
- **CLI 框架**：prompt_toolkit、Rich
- **Web UI**：FastAPI、React 19、Vite、Tailwind CSS v4
- **消息平台**：python-telegram-bot、discord.py、slack-bolt、aiohttp
- **备份系统**：zipfile、sqlite3.backup() API
- **测试框架**：pytest、pytest-asyncio、pytest-xdist（~3000 测试用例）
- **依赖管理**：pyproject.toml、setuptools、npm

### 架构模式
- **同步代理循环**：核心对话循环完全同步，简化状态管理
- **工具注册表**：中央化的工具注册系统，支持动态发现和调度
- **配置系统**：YAML 配置文件 + 环境变量，支持多实例配置
- **插件系统**：技能系统、MCP 集成、记忆提供者插件
- **备份系统**：完整备份 + 快速快照，支持 SQLite 安全复制
- **OAuth 认证**：Web Dashboard 中的 OAuth 提供商管理

### 设计原则
1. **提示缓存不能破坏**：整个对话过程中保持缓存有效性
2. **配置文件优先**：用户管理的配置文件优于过时的 shell 导出
3. **多实例隔离**：每个配置实例有完全独立的 HERMES_HOME 目录
4. **跨平台一致性**：CLI 和消息平台共享相同的命令注册表和工具系统

## 模块结构图

```mermaid
graph TD
    A["(根) Hermes Agent"] --> B["agent"];
    A --> C["hermes_cli"];
    A --> D["tools"];
    A --> E["gateway"];
    A --> F["environments"];
    A --> G["cron"];
    A --> H["acp_adapter"];
    A --> I["plugins"];
    A --> J["web"];
    A --> K["tests"];

    B --> B1["prompt_builder.py"];
    B --> B2["context_compressor.py"];
    B --> B3["memory_manager.py"];
    B --> B4["display.py"];
    B --> B5["skill_commands.py"];
    B --> B6["credential_pool.py"];
    B --> B7["context_engine.py"];

    C --> C1["main.py"];
    C --> C2["config.py"];
    C --> C3["commands.py"];
    C --> C4["setup.py"];
    C --> C5["skin_engine.py"];
    C --> C6["web_server.py"];
    C --> C7["backup.py"];
    C --> C8["auth_commands.py"];

    D --> D1["registry.py"];
    D --> D2["terminal_tool.py"];
    D --> D3["file_tools.py"];
    D --> D4["web_tools.py"];
    D --> D5["browser_tool.py"];
    D --> D6["mcp_tool.py"];

    E --> E1["run.py"];
    E --> E2["session.py"];
    E --> E3["platforms/"];

    F --> F1["agent_loop.py"];
    F --> F2["benchmarks/"];

    J --> J1["src/pages/"];
    J --> J2["src/components/"];
    J --> J3["src/lib/api.ts"];

    click B "./agent/CLAUDE.md" "查看 agent 模块文档"
    click C "./hermes_cli/CLAUDE.md" "查看 hermes_cli 模块文档"
    click D "./tools/CLAUDE.md" "查看 tools 模块文档"
    click E "./gateway/CLAUDE.md" "查看 gateway 模块文档"
    click F "./environments/CLAUDE.md" "查看 environments 模块文档"
    click G "./cron/CLAUDE.md" "查看 cron 模块文档"
    click H "./acp_adapter/CLAUDE.md" "查看 acp_adapter 模块文档"
    click I "./plugins/CLAUDE.md" "查看 plugins 模块文档"
    click J "./web/README.md" "查看 Web UI 文档"
```

## 模块交互流程

### 工具调用流程

以下流程图展示了当 AI 模型请求调用工具时的完整交互流程：

```mermaid
sequenceDiagram
    participant User as 用户/平台
    participant Gateway as Gateway
    participant Agent as AIAgent
    participant Builder as PromptBuilder
    participant LLM as LLM API
    participant Tools as ToolRegistry
    participant Tool as Tool Handler

    User->>Gateway: 发送消息
    Gateway->>Agent: 传递用户消息
    Agent->>Builder: 构建提示（系统提示 + 技能索引 + 上下文）
    Builder-->>Agent: 返回完整提示
    Agent->>LLM: 发送提示
    LLM-->>Agent: 返回响应（包含工具调用请求）
    Agent->>Tools: 查询工具注册表
    Tools-->>Agent: 返回工具处理器
    Agent->>Tool: 执行工具调用
    Tool-->>Agent: 返回工具结果
    Agent->>Builder: 构建后续提示（包含工具结果）
    Builder-->>Agent: 返回更新后的提示
    Agent->>LLM: 发送后续提示
    LLM-->>Agent: 返回最终响应
    Agent-->>Gateway: 返回最终响应
    Gateway-->>User: 显示响应

    Note over Agent,Tools: 工具调用循环可能重复多次
    Note over Agent,LLM: 保持提示缓存有效性
```

**关键交互点**：
- **PromptBuilder** 负责组装系统提示、技能索引和用户消息
- **ToolRegistry** 提供工具的动态发现和调度
- **AIAgent** 协调整个调用流程，维护对话状态
- **工具调用** 可以是终端命令、文件操作、Web 浏览等

### 消息调度流程

以下流程图展示了从消息平台到 AI 代理的消息调度流程：

```mermaid
sequenceDiagram
    participant Platform as 消息平台
    participant Adapter as 平台适配器
    participant Gateway as Gateway
    participant Session as SessionManager
    participant Agent as AIAgent
    participant CLI as HermesCLI

    Platform->>Adapter: 接收用户消息
    Adapter->>Gateway: 转发消息
    Gateway->>Session: 查找/创建会话
    Session-->>Gateway: 返回会话上下文
    Gateway->>Gateway: 检测斜杠命令
    alt 是斜杠命令
        Gateway->>CLI: 调度命令
        CLI-->>Gateway: 返回命令结果
        Gateway-->>Adapter: 返回结果
    else 是危险命令
        Gateway->>Gateway: 请求用户确认
        Adapter-->>Gateway: 用户确认/拒绝
        alt 用户确认
            Gateway->>CLI: 执行危险命令
            CLI-->>Gateway: 返回结果
            Gateway-->>Adapter: 返回结果
        end
    else 普通消息
        Gateway->>Agent: 传递消息
        Agent-->>Gateway: 返回 AI 响应
        Gateway-->>Adapter: 返回响应
    end
    Adapter-->>Platform: 发送响应

    Note over Gateway,Session: 会话持久化到磁盘
    Note over Gateway,CLI: 危险命令需要审批
    Note over Gateway,Agent: 保持对话状态
```

**关键交互点**：
- **平台适配器** 处理不同消息平台的协议差异
- **SessionManager** 管理会话状态和持久化
- **Gateway** 负责命令检测和调度
- **危险命令**（如删除文件）需要用户确认

### 配置加载流程

以下流程图展示了 Hermes Agent 的配置加载和迁移流程：

```mermaid
flowchart TD
    Start([启动 Hermes]) --> LoadConfig[加载配置文件]
    LoadConfig --> ConfigExists{配置文件存在?}

    ConfigExists -->|否| CreateDefaults[创建默认配置]
    CreateDefaults --> SaveConfig[保存配置文件]
    SaveDefaults --> LoadDefaults[加载 DEFAULT_CONFIG]

    ConfigExists -->|是| ReadConfig[读取配置文件]
    ReadConfig --> CheckVersion{配置版本匹配?}

    CheckVersion -->|否| RunMigration[执行配置迁移]
    RunMigration --> UpdateVersion[更新配置版本]
    UpdateVersion --> SaveMigrated[保存迁移后的配置]
    SaveMigrated --> LoadMerged[加载合并后的配置]

    CheckVersion -->|是| LoadUser[加载用户配置]
    LoadUser --> MergeDefaults[合并默认配置]
    MergeDefaults --> ValidateConfig{配置有效?}

    LoadDefaults --> ValidateConfig
    LoadMerged --> ValidateConfig

    ValidateConfig -->|否| ShowError[显示配置错误]
    ShowError --> Exit([退出])

    ValidateConfig -->|是| LoadEnv[加载 .env 文件]
    LoadEnv --> ApplyEnv[应用环境变量]
    ApplyEnv --> Complete([配置加载完成])

    Complete --> StartCLI[启动 CLI]
    Complete --> StartGateway[启动网关]
```

**关键交互点**：
- **配置版本控制** 确保配置结构的平滑升级
- **默认配置合并** 保证新选项的可用性
- **环境变量** 提供额外的配置灵活性
- **多实例支持** 通过不同的配置文件实现隔离

### 记忆管理流程

以下流程图展示了 Hermes Agent 的记忆管理和知识持久化流程：

```mermaid
sequenceDiagram
    participant Agent as AIAgent
    participant MemMgr as MemoryManager
    participant Provider as MemoryProvider
    participant Storage as 文件系统
    participant Search as FTS5 搜索

    Agent->>MemMgr: 请求加载记忆
    MemMgr->>Provider: 查询记忆提供者
    Provider-->>MemMgr: 返回提供者列表
    alt 启用记忆提供者
        MemMgr->>Provider: 获取持久化记忆
        Provider->>Storage: 读取记忆文件
        Storage-->>Provider: 返回记忆内容
        Provider-->>MemMgr: 返回格式化的记忆
    end
    MemMgr->>MemMgr: 构建记忆提示
    MemMgr-->>Agent: 返回记忆上下文

    Agent->>Agent: 生成响应
    Agent->>MemMgr: 提交新记忆
    MemMgr->>MemMgr: 策展记忆内容
    MemMgr->>Provider: 保存记忆
    Provider->>Storage: 写入记忆文件
    Storage-->>Provider: 确认保存

    Note over MemMgr,Search: 记忆搜索功能
    Agent->>MemMgr: 搜索历史对话
    MemMgr->>Search: 执行 FTS5 查询
    Search-->>MemMgr: 返回搜索结果
    MemMgr-->>Agent: 返回相关对话

    Note over Agent,Provider: 支持多种记忆提供者<br/>（内置、Honcho、Mem0、Holographic）
```

**关键交互点**：
- **MemoryManager** 协调不同记忆提供者的交互
- **记忆策展** 智能选择和格式化记忆内容
- **FTS5 搜索** 提供快速的全文搜索能力
- **插件系统** 支持多种第三方记忆提供者

## 模块索引

| 模块名称 | 路径 | 主要职责 | 入口文件 | 测试覆盖 | 文档状态 |
|---------|------|---------|---------|---------|---------|
| **Agent Core** | `agent/` | 代理核心逻辑 | `__init__.py` | ✅ 高 | ✅ 已完成 |
| **CLI Interface** | `hermes_cli/` | 命令行界面 | `main.py` | ✅ 高 | ✅ 已完成 |
| **Tools System** | `tools/` | 工具注册与执行 | `registry.py` | ✅ 高 | ✅ 已完成 |
| **Gateway** | `gateway/` | 消息平台网关 | `run.py` | ✅ 高 | ✅ 已完成 |
| **Web UI** | `web/` | Web 管理界面 | `src/main.tsx` | ✅ 高 | ✅ 已完成 |
| **Environments** | `environments/` | RL 训练环境 | `agent_loop.py` | ✅ 中 | ✅ 已完成 |
| **Cron Scheduler** | `cron/` | 定时任务调度 | `scheduler.py` | ✅ 高 | ✅ 已完成 |
| **ACP Adapter** | `acp_adapter/` | ACP 协议适配器 | `entry.py` | ✅ 高 | ✅ 已完成 |
| **Plugins** | `plugins/` | 记忆提供者插件 | `__init__.py` | ✅ 中 | ✅ 已完成 |
| **Tests** | `tests/` | 测试套件 | `conftest.py` | N/A | ✅ 已完成 |

## 架构决策记录（ADR）

Hermes Agent 使用架构决策记录（ADR）来记录重要的架构决策和设计选择。这些文档提供了架构决策的可追溯性，帮助理解设计理念和权衡。

### ADR 导航图

以下思维导图展示了 Hermes Agent 的所有架构决策及其关系：

```mermaid
mindmap
  root((Hermes ADR))
    核心架构
      ADR-001 同步代理循环
      ADR-002 工具注册表
      ADR-004 提示缓存保护
    系统设计
      ADR-003 多实例配置
      ADR-006 命令系统
      ADR-007 会话管理
    扩展系统
      ADR-005 记忆插件
      ADR-008 技能系统
```

### ADR 索引

| ID | 决策标题 | 状态 | 核心内容 | 图表 |
|----|---------|------|---------|------|
| [ADR-001](docs/adr/001-sync-agent-loop.md) | 同步代理循环 | ✅ 接受 | 同步 vs 异步的核心循环设计 | ✅ 架构图 |
| [ADR-002](docs/adr/002-tool-registry.md) | 工具注册表 | ✅ 接受 | 中央化工具管理系统 | ✅ 架构图 |
| [ADR-003](docs/adr/003-multi-instance.md) | 多实例配置隔离 | ✅ 接受 | 支持多个隔离的 Hermes 实例 | ✅ 结构图 |
| [ADR-004](docs/adr/004-prompt-cache.md) | 提示缓存保护 | ✅ 接受 | 保护 API 提示缓存的设计原则 | ❌ |
| [ADR-005](docs/adr/005-plugin-memory.md) | 记忆插件系统 | ✅ 接受 | 可扩展的记忆存储架构 | ✅ 架构图 |
| [ADR-006](docs/adr/006-command-system.md) | 命令系统设计 | ✅ 接受 | 跨平台命令注册和调度 | ✅ 流程图 x2 |
| [ADR-007](docs/adr/007-session-management.md) | 会话管理系统 | ✅ 接受 | 跨平台会话持久化 | ✅ 状态图 x2 |
| [ADR-008](docs/adr/008-skill-system.md) | 技能系统架构 | ✅ 接受 | 自动创建和改进技能 | ✅ 状态图 x2 |

### ADR 反馈机制

我们重视社区对架构决策的反馈。您可以通过以下方式参与：

#### 投票和评分

对 ADR 的反馈：
- 👍 **赞成**：同意此架构决策
- 👎 **反对**：不同意此架构决策
- 💡 **建议**：有改进建议

**如何反馈**：
1. 在 [GitHub Discussions](https://github.com/NousResearch/hermes-agent/discussions) 中创建讨论
2. 使用 ADR 编号作为标题前缀（如：`[ADR-001] 关于同步循环的讨论`）
3. 说明您的观点、理由或建议

#### 提交新 ADR

如果您认为需要新的架构决策记录：
1. 使用 `docs/adr/README.md` 中的模板创建新 ADR
2. 编号递增（如：ADR-009）
3. 提交 PR 并说明决策背景
4. 等待团队审查和批准

#### 修改现有 ADR

如果您认为某个 ADR 需要更新：
1. 在 GitHub 中创建 Issue 说明更新原因
2. 团队讨论并达成共识
3. 提交 PR 更新 ADR
4. 在 `docs/adr/CHANGELOG.md` 中记录变更

### ADR 资源

- 📋 [ADR 索引](docs/adr/README.md) - 完整的 ADR 列表和模板
- 📜 [ADR 变更日志](docs/adr/CHANGELOG.md) - 架构决策的演进历史
- 🔗 [ADR 规范](https://adr.github.io/) - ADR 标准规范

## 运行与开发

### 快速开始
```bash
# 安装
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc

# 启动交互式 CLI
hermes

# 配置模型
hermes model

# 启动消息网关
hermes gateway

# 启动 Web UI Dashboard
hermes web

# 创建备份
hermes backup

# 快速快照
hermes backup --quick
```

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# 安装依赖
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"

# 运行测试
python -m pytest tests/ -q
```

### 核心命令
- `hermes` - 启动交互式 CLI
- `hermes model` - 选择 LLM 提供商和模型
- `hermes tools` - 配置启用的工具
- `hermes config set` - 设置配置值
- `hermes gateway` - 启动消息网关
- `hermes web` - 启动 Web UI Dashboard（默认 http://127.0.0.1:9119）
- `hermes backup` - 创建完整备份或快速快照
- `hermes import` - 从备份恢复
- `hermes debug share` - 上传调试报告到 pastebin
- `hermes setup` - 运行完整设置向导
- `hermes doctor` - 诊断问题

### 斜杠命令
- `/model` - 原生模型选择器模态框
- `/snapshot` 或 `/snap` - 快速状态快照（create/list/restore/prune）
- `/debug` - 调试命令
- `/compress <focus>` - 引导式压缩，支持聚焦主题

## 测试策略

### 测试组织
- **单元测试**：每个模块都有对应的测试目录
- **集成测试**：`tests/integration/` - 需要外部服务的测试
- **E2E 测试**：`tests/e2e/` - 端到端场景测试
- **性能测试**：`tests/benchmarks/` - 基准测试

### 测试标记
- `integration` - 需要外部服务（API 密钥、Modal 等）
- 默认运行：`pytest -m 'not integration' -n auto`
- 完整测试：`pytest tests/ -q`（约 3000 测试，约 3 分钟）

### 关键测试领域
- **工具系统**：工具注册、调度、可用性检查
- **CLI 配置**：配置加载、迁移、多实例
- **网关**：平台适配器、会话持久化、命令调度
- **代理循环**：工具调用循环、错误处理、状态管理

## 编码规范

### 代码风格
- **PEP 8**：Python 代码风格指南
- **类型提示**：所有公共 API 必须有类型注解
- **文档字符串**：Google 风格的 docstrings
- **命名约定**：
  - 模块：`snake_case`
  - 类：`PascalCase`
  - 函数/变量：`snake_case`
  - 常量：`UPPER_SNAKE_CASE`

### 关键约定
1. **多实例安全**：使用 `get_hermes_home()` 而不是硬编码 `~/.hermes`
2. **用户友好路径**：使用 `display_hermes_home()` 显示路径
3. **工具注册**：所有工具必须在模块导入时注册到中央注册表
4. **命令注册**：所有斜杠命令在 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY` 中定义
5. **配置版本**：更改配置结构时必须增加 `_config_version`

### 禁止模式
- ❌ 不要硬编码 `~/.hermes` 路径
- ❌ 不要在工具模式描述中硬编码跨工具引用
- ❌ 不要在 spinner/display 代码中使用 `\033[K`
- ❌ 不要在测试中写入 `~/.hermes/`
- ❌ 不要在对话过程中更改工具集
- ❌ 不要在对话过程中重新加载记忆

## AI 使用指引

### 架构理解
1. **代理循环**：`run_agent.py` 中的 `AIAgent.run_conversation()` 是核心同步循环
2. **工具系统**：`tools/registry.py` 是中央工具注册表，所有工具在导入时注册
3. **命令系统**：`hermes_cli/commands.py` 中的 `COMMAND_REGISTRY` 是所有斜杠命令的真实来源
4. **配置系统**：`hermes_cli/config.py` 中的 `DEFAULT_CONFIG` 定义所有配置选项

### 关键文件依赖链
```
tools/registry.py (无依赖)
    ↑
tools/*.py (每个在导入时调用 registry.register())
    ↑
model_tools.py (导入 tools/registry + 触发工具发现)
    ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

### 添加新功能的步骤
**添加工具**：
1. 创建 `tools/your_tool.py`
2. 在 `model_tools.py` 的 `_discover_tools()` 中添加导入
3. 在 `toolsets.py` 中添加到相应的工具集

**添加斜杠命令**：
1. 在 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY` 中添加 `CommandDef`
2. 在 `cli.py` 的 `HermesCLI.process_command()` 中添加处理器
3. 如果网关支持，在 `gateway/run.py` 中添加处理器

**添加配置选项**：
1. 在 `hermes_cli/config.py` 的 `DEFAULT_CONFIG` 中添加
2. 增加 `_config_version` 以触发迁移

### 常见任务
- **理解工具调用**：从 `model_tools.py:handle_function_call()` 开始
- **理解提示构建**：从 `agent/prompt_builder.py` 开始
- **理解 CLI 命令**：从 `hermes_cli/commands.py:COMMAND_REGISTRY` 开始
- **理解网关消息流**：从 `gateway/run.py:dispatch_message()` 开始
- **理解配置加载**：从 `hermes_cli/config.py:load_cli_config()` 开始

### 调试技巧
- **启用详细日志**：设置 `HERMES_DEBUG=1`
- **检查工具可用性**：`hermes tools` 命令
- **查看配置**：`hermes config show`
- **测试工具**：`pytest tests/tools/test_your_tool.py`
- **测试命令**：`pytest tests/cli/test_your_command.py`

## 相关资源

### 官方文档
- **主文档**：https://hermes-agent.nousresearch.com/docs/
- **快速开始**：https://hermes-agent.nousresearch.com/docs/getting-started/quickstart
- **架构指南**：https://hermes-agent.nousresearch.com/docs/developer-guide/architecture
- **贡献指南**：https://hermes-agent.nousresearch.com/docs/developer-guide/contributing

### 社区
- **Discord**：https://discord.gg/NousResearch
- **GitHub Issues**：https://github.com/NousResearch/hermes-agent/issues
- **Discussions**：https://github.com/NousResearch/hermes-agent/discussions
- **Skills Hub**：https://agentskills.io

### 依赖项目
- **OpenClaw**：Hermes 的前身，支持迁移导入
- **Honcho**：用于用户建模的对话式记忆系统
- **Atropos**：RL 训练环境（可选）
- **Tinker**：RL 工具集成（可选）

## 变更记录 (Changelog)

### 2026-04-16 - 同步上游 v2026.4.13 版本变更 🚀
- ✅ **新增备份和导入系统**：完整的 `hermes backup` 和 `hermes import` 命令
- 📸 **新增快照功能**：`hermes backup --quick` 和 `/snapshot` 斜杠命令
- 🔐 **新增 OAuth 提供商管理**：Web Dashboard 中的 OAuth 登录功能
- 🇨🇳 **新增 Kimi/Moonshot 提供商**：支持中国地区的模型提供商
- 🐛 **新增调试命令**：`/debug` 斜杠命令和 `hermes debug share`
- 💡 **新增提示功能**：会话开始时显示随机提示（279 条提示）
- 🎯 **新增模型选择器**：原生 `/model` 选择器模态框
- 🌐 **新增网络配置**：`network.force_ipv4` 配置选项
- 📝 **新增组件化日志**：带会话上下文和过滤功能的日志系统
- 🔧 **修复 SQLite 备份安全**：使用 `sqlite3.Connection.backup()` API
- 🛠️ **修复 78 个 CI 测试**：移除死代码和修复测试失败
- 📚 **修复 30+ 文档错误**：全面的文档更新

### 2026-04-13 16:35:00 - 代码变更同步更新 🔄
- ✅ **新增 Web UI Dashboard**：基于 React 19 + Vite + Tailwind CSS v4 的浏览器管理界面
- 📊 **新增 9 个页面**：Status、Config、Env、Sessions、Skills、Cron、Logs、Analytics 页面
- 🔧 **Gateway 改进**：重启通知机制、服务重启支持（systemd）
- 🛡️ **安全修复**：Home Assistant 工具的路径遍历验证
- 🔤 **模型名称修复**：保留 OpenCode Zen 和 ZAI 提供商模型名称中的点
- 📱 **WhatsApp UX 改进**：分块、格式化、流式输出改进
- 📨 **Telegram 修复**：使用 UTF-16 代码单元进行消息长度分割
- 🖥️ **Web 服务器**：FastAPI 后端 + React 前端的完整 Web 界面
- 📦 **新命令**：`hermes web` 启动 Web UI Dashboard
- 📖 **文档更新**：更新技术栈、模块结构图和核心命令

### 2026-04-08 18:33:30 - 完善 ADR 体系 📋
- ✅ **新增 3 个 ADR**：命令系统、会话管理、技能系统
- 📊 **添加视觉化图表**：为所有 ADR 添加 10 个 Mermaid 图表
- 🗺️ **创建 ADR 导航图**：在根 CLAUDE.md 中添加 ADR 索引和思维导图
- 🔄 **建立反馈机制**：添加 ADR 投票、评分和贡献指南
- 📜 **创建变更日志**：记录 ADR 的演进历史和未来计划

### 2026-04-08 18:29:30 - 添加模块交互流程图 🎨

### 2026-04-08 18:28:35 - 模块文档完成 🎉
- ✅ **创建所有模块文档**：为 8 个核心模块创建完整的 CLAUDE.md
- 📊 **文档覆盖率 100%**：所有核心模块都有详细文档
- 🔧 **面包屑导航**：每个模块文档都包含面包屑导航
- 📖 **接口文档**：详细的接口说明和使用示例
- 🎯 **索引更新**：更新索引文件反映当前文档状态
- 📈 **下一步建议**：创建测试套件覆盖报告和性能优化指南

### 2026-04-08 - 初始化 AI 上下文文档 🚀
- ✅ **创建根级文档**：生成项目级 CLAUDE.md
- 📊 **项目分析**：识别 9 个核心模块
- 🔧 **架构文档**：详细说明技术栈、架构模式和设计原则
- 📖 **开发指南**：提供运行、测试、编码规范和 AI 使用指引
- 🗺️ **模块索引**：创建模块结构图和索引表
- 🎯 **下一步**：需要为每个模块创建详细的 CLAUDE.md 文档

---

*提示：点击上方模块名称或 Mermaid 图表中的节点可快速跳转到对应模块的详细文档。*
