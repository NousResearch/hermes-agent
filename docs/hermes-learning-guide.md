# Hermes 项目源码学习指南

学习 Hermes 不建议从第一行开始通读整个仓库，而应按 **“先跑起来 → 跟踪一次对话 → 阅读核心模块 → 做一个小改动”** 的路径进行。Hermes 功能很多，先掌握 Agent 主链路，再看 Gateway、TUI、Desktop 等外围部分。

## 1. 先建立整体认识

先读这几份文档：

1. `README.md`：了解项目能做什么。
2. `website/docs/developer-guide/architecture.md`：了解目录、模块和请求链路。
3. `website/docs/developer-guide/agent-loop.md`：了解一次对话如何执行。
4. 根目录 `AGENTS.md`：如果之后准备改代码或提交 PR，务必阅读。

先记住两个核心设计原则：

- **Prompt caching 很重要**：对话进行过程中，不能随意修改历史消息、工具集合或系统提示词，否则可能破坏缓存并增加调用成本。
- **核心保持精简**：新能力优先做成 Skill、CLI 命令或插件，而不是直接增加核心工具。

可以先把它理解成：

```text
CLI / Gateway / TUI / Cron
            │
            ▼
        AIAgent
            │
     构造 Prompt、选择模型
            │
            ▼
        调用 LLM
            │
       是否调用工具？
       │           │
       ▼           ▼
     执行工具      返回答案
       │
       ▼
  把工具结果加入上下文，继续调用 LLM
```

## 2. 在 Windows 上运行源码版本

在仓库根目录的 PowerShell 中执行：

```powershell
$env:HERMES_HOME = Join-Path $HOME 'hermes-dev-data'
$env:HERMES_RUNTIME_DIR = Join-Path $env:HERMES_HOME 'tools'

. .\activate.ps1
hermes --version
```

然后完成模型配置并启动：

```powershell
hermes setup
hermes
```

也可以只配置模型：

```powershell
hermes model
hermes
```

建议使用独立的 `HERMES_HOME`，避免实验配置影响现有的 Hermes 数据。依赖环境由项目的 PM 管理，不要直接用 `pip` 或 `uv` 修改它。

启动后，在一个临时测试目录中让它完成一个简单任务，例如：

- 读取几个文件并总结；
- 修改一个小文件；
- 执行一个简单命令并解释结果。

同时观察：

- 模型什么时候调用工具；
- 工具结果如何回到对话中；
- `/usage` 显示的上下文使用情况；
- `/skills` 和 `/compress` 的行为。

## 3. 跟踪一次 CLI 对话的完整链路

这是最值得花时间的一步，建议按下面顺序读，并用编辑器断点或调试器辅助观察。

### 入口和 Agent 外观层

- `cli.py`：`HermesCLI`，CLI 入口和交互编排。
- `hermes_cli/cli_*_mixin.py`：CLI 的具体功能按主题拆分在这些文件中。
- `run_agent.py`：`AIAgent` 外观层，不要误以为所有循环逻辑都在这里。

### 对话主循环

- `agent/conversation_loop.py`：`run_conversation()`。
- `agent/turn_*.py`：一次循环的不同阶段，例如请求准备、API 调用、错误处理、工具执行、上下文溢出和恢复。
- `agent/tool_executor.py`：工具调用执行，以及部分 Agent 内部工具的拦截。

重点看懂这个过程：

```text
用户消息
  → 加入 conversation history
  → 构造或复用 system prompt
  → 调用模型
  → 模型返回 tool_calls
  → 执行工具
  → 工具结果加入 history
  → 再次调用模型
  → 返回最终文本
  → 保存会话
```

还要留意消息格式和角色交替规则：`system → user → assistant → tool → assistant`，不要产生非法的连续消息。

### Prompt、模型和工具

接下来阅读：

- `agent/prompt_builder.py`：系统提示词如何组装，包含身份、技能、上下文文件、记忆等。
- `website/docs/developer-guide/prompt-assembly.md`
- `website/docs/developer-guide/provider-runtime.md`：Provider、模型、API 模式和凭证如何解析。
- `model_tools.py`：工具调用的统一分发入口。
- `tools/registry.py`：工具注册、Schema 收集、可用性检查和分发。
- `toolsets.py`：不同平台启用哪些工具组合。
- `tools/file_tools.py`、`tools/terminal_tool.py`：选两个具体工具深入阅读即可。

不要一开始试图读完所有工具，先选一个文件工具和一个终端工具，弄清楚：

```text
工具定义
  → 注册到 registry
  → 生成给模型的 Schema
  → 模型发起调用
  → 参数解析与安全检查
  → 执行
  → 结果写回消息历史
```

### 会话和持久化

最后看：

- `hermes_state.py` 及 `hermes_state_*.py`：SessionDB 外观层和具体实现。
- `website/docs/developer-guide/session-storage.md`：SQLite、FTS5 搜索、会话恢复和压缩后的会话关系。

## 4. 再按兴趣选择支线

主链路理解后，再选择一条深入学习：

| 目标 | 建议阅读 |
|---|---|
| 理解工具和终端环境 | `tools/`、`tools/environments/`、`website/docs/developer-guide/tools-runtime.md` |
| 学习记忆和技能系统 | `agent/memory_manager.py`、`agent/prompt_builder.py`、`skills/` |
| 学习 Telegram/Discord 等接入 | `gateway/run.py`、`gateway/session.py`、`gateway/platforms/` |
| 学习插件开发 | `plugins/`、`website/docs/developer-guide/plugins/index.md` |
| 学习定时任务 | `cron/`、`website/docs/developer-guide/cron-internals.md` |
| 学习 TUI/Desktop | 最后再进入 `ui-tui/`、`tui_gateway/`、`apps/desktop/` |

Gateway 还要特别注意 **Profile 隔离**：一个进程可能服务多个 Profile，不能随意依赖进程级环境变量或全局状态。

## 5. 用一个小练习检验是否真正理解

推荐按这个顺序做：

1. 画出一次 CLI 对话的时序图，标出模型调用、工具调用和会话保存的位置。
2. 用调试器观察一次工具调用前后的 `messages` 内容。
3. 阅读一个已有的 Skill，并尝试在隔离环境中写一个简单的独立 Skill。
4. 找一个很小的 bug，先写能复现问题的测试，再修复。
5. 运行相关测试：

```bash
scripts/run_tests.sh tests/agent/ -v
```

Windows 上通过 Bash 运行测试脚本。准备提交前再运行：

```bash
python scripts/check
```

**不建议**一开始就从 Gateway、Desktop 或全部测试入手，也不建议把“新增核心工具”作为第一个练习；从主循环、一个具体工具或一个独立 Skill 开始，学习曲线会平缓很多。
