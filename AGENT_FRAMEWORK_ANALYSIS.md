# Hermes Agent 框架分析

本文档整理对 `hermes-agent` 工程中 Agent 框架的分析，重点覆盖整体架构、核心优势、生命周期管理、多 Agent 协作机制、与 LangGraph / Harness / Swarm 类框架的区别，以及闭环自动优化方式。

## 1. 总体定位

`hermes-agent` 更像一个完整的 Agent 运行平台，而不是单一的聊天机器人、评测 harness 或工作流编排库。

它的核心设计是：

```text
统一 AIAgent 内核
  ↓
CLI / TUI / Gateway / Dashboard 多入口复用
  ↓
工具、记忆、技能、Provider、插件横向扩展
  ↓
Cron / Kanban / Delegation 支撑长期任务和多 Agent 协作
```

核心文件包括：

- `run_agent.py`：`AIAgent` 主体和对话循环。
- `model_tools.py`：工具发现、schema 生成、工具调用分发。
- `tools/registry.py`：工具注册表。
- `toolsets.py`：控制不同平台和场景暴露哪些工具。
- `hermes_state.py`：会话存储和检索。
- `agent/memory_manager.py`：记忆系统编排。
- `agent/context_compressor.py`：上下文压缩。
- `tools/delegate_tool.py`：子代理委托。
- `gateway/run.py`、`cli.py`、`tui_gateway/server.py`：不同入口对 `AIAgent` 的复用。

## 2. 架构特点与优势

### 2.1 单一 Agent 内核，多端复用

CLI、TUI、Gateway、Dashboard 都复用同一个 `AIAgent` 运行内核。不同入口主要差异在 UI、回调、平台上下文和 session 管理，而不是各自实现一套 Agent loop。

这种设计的优势是：

- 多端行为一致。
- 核心能力只需维护一份。
- 工具调用、记忆、压缩、会话等能力天然跨端复用。
- 新增入口时，不需要重新实现 Agent 主循环。

### 2.2 工具系统插件化、可组合

工具不是硬编码在主循环里，而是通过 `tools/registry.py` 注册，再通过 `toolsets.py` 控制是否暴露给模型。

这带来几个好处：

- 不同平台可以启用不同工具集。
- 工具可以按能力分组。
- 插件可以动态注册工具。
- 模型只看到当前场景允许使用的工具。
- 可以避免工具 schema 过大或跨平台能力泄漏。

### 2.3 Provider 抽象完整

模型 provider 通过 `plugins/model-providers/*` 插件化注册。主流程使用相对统一的 OpenAI 兼容调用方式，而把不同模型厂商的 base URL、能力、上下文长度等差异下沉到 provider profile。

这使得系统可以接入多个模型后端，而不是绑定某一家。

### 2.4 记忆、压缩、会话是系统级能力

工程不是把 memory、session、compression 当作附加功能，而是纳入主循环：

- `SessionDB` 持久化会话。
- `MemoryManager` 负责记忆预取和同步。
- `ContextCompressor` 负责上下文过长时的压缩。
- prompt caching 是明确的设计约束。

这说明它面向的是长期运行的 Agent 平台，而不是短任务 demo。

### 2.5 扩展点丰富

主要扩展面包括：

- 核心工具：`tools/*.py` + `tools/registry.py` + `toolsets.py`
- 通用插件：`hermes_cli/plugins.py`
- 模型 provider 插件：`plugins/model-providers/*`
- 记忆 provider 插件：`plugins/memory/*`
- 技能系统：`skills/`、`optional-skills/`
- Gateway 平台适配器：`gateway/platforms/*`
- 生命周期 hook：pre/post LLM、pre/post tool、session start/end 等

## 3. Agent 生命周期管理

可以把生命周期分成四层：进程生命周期、会话生命周期、单轮对话生命周期、工具/子代理生命周期。

### 3.1 创建阶段

入口通常由 CLI、TUI 或 Gateway 构造 `AIAgent`。

创建时会注入：

- provider / model / base_url / api_key
- session_id
- enabled / disabled toolsets
- callbacks，例如流式输出、工具进度、clarify、approval
- memory manager
- session db
- credential pool
- platform 信息，例如 `cli`、`telegram`、`tui`

`AIAgent` 本身不绑定某个 UI，而是一个可复用的运行内核。

### 3.2 会话开始

会话开始阶段主要负责：

- 建立或恢复 `session_id`
- 初始化 `SessionDB`
- 加载历史消息
- 构建 system prompt
- 初始化 memory provider
- 发现工具 schema
- 应用技能、配置、平台上下文

插件也可以通过 session start、agent start 等生命周期 hook 介入。

### 3.3 单轮对话运行

核心在 `AIAgent.run_conversation()`。

简化流程：

```text
接收用户消息
  ↓
合并历史消息 / system prompt / memory
  ↓
检查上下文长度，必要时压缩
  ↓
调用模型
  ↓
如果模型返回 tool_calls
  ↓
执行工具
  ↓
工具结果回填 messages
  ↓
继续调用模型
  ↓
直到得到最终回复
```

关键控制点：

- 工具调用经过 `model_tools.handle_function_call()`。
- 特殊工具可能在 `run_agent.py` 内部先被拦截。
- 主循环有 `max_iterations` 和 iteration budget，避免无限工具调用。
- 中断信号会被检查，Gateway `/stop`、CLI 退出等可以打断 Agent。

### 3.4 单轮结束

一轮结束后通常会：

- 持久化 transcript。
- 更新 session metadata。
- 同步 memory。
- 触发 post hook。
- 返回 final response 给调用方。

注意，memory provider 的彻底 shutdown 不是每轮都做，而是在真实 session 边界做，例如 CLI 退出、Gateway session 过期、`/new` 重置等。

### 3.5 会话结束与资源清理

`AIAgent.shutdown_memory_provider()` 会处理：

- memory provider 的 `on_session_end`
- memory provider 的 `shutdown_all`
- context engine 的 `on_session_end`
- 工具资源清理

Gateway 还会通过 `_cleanup_agent_resources()` 关闭：

- memory provider
- terminal / browser / background process 资源
- stale async auxiliary clients

这个设计明确区分了“单轮完成”和“会话真正结束”。

## 4. 多 Agent 协作机制

`hermes-agent` 不是只有一种多 Agent 协作模式，而是按任务性质拆成不同机制。

### 4.1 `delegate_task`：同步子代理协作

`delegate_task` 是最接近 Swarm 的机制。

父 Agent 可以创建一个或多个子 `AIAgent`：

```text
父 Agent
  ↓ delegate_task
子 Agent A / B / C
  ↓
返回结构化摘要
  ↓
父 Agent 继续推理
```

特点：

- 子 Agent 有独立上下文。
- 子 Agent 不自动继承父对话历史。
- 父 Agent 必须通过 `goal` 和 `context` 显式传递任务信息。
- 子 Agent 有自己的 terminal session。
- batch 模式可以并行多个子任务。
- 默认并发上限由 `delegation.max_concurrent_children` 控制。
- 如果父 Agent 被中断，子 Agent 会取消。
- 这是同步机制，不适合长期后台任务。

角色控制：

- `leaf`：默认子代理，不能继续委托。
- `orchestrator`：可以继续创建子代理，但受 `delegation.max_spawn_depth` 限制。

### 4.2 Cron：持久化定时任务

Cron 用于不应该依附当前对话生命周期的任务，例如：

- 定时检查 CI。
- 每天总结日志。
- 定期扫描 issue。
- 周期性生成报告。

它和 `delegate_task` 的区别是：

- `delegate_task` 跟随当前父 Agent 生命周期。
- `cronjob` 是 durable 的，适合跨会话、跨时间运行。

Cron session 默认 `skip_memory=True`，避免定时任务污染普通聊天记忆。

### 4.3 Kanban：多 Agent 工作队列

Kanban 是更重的协作模型。它用 SQLite-backed board 管理任务，由 dispatcher 分配给不同 profile / worker。

流程：

```text
Board 中有任务
  ↓
Dispatcher 定期扫描
  ↓
原子 claim 任务
  ↓
启动对应 profile / worker
  ↓
worker 使用 kanban_* 工具更新状态
  ↓
任务完成 / 阻塞 / 重新分配
```

特点：

- 任务持久化，不依附某一次对话。
- 多个 Agent / profile 可以协作。
- 支持任务 claim、heartbeat、stale 回收。
- 支持任务依赖、阻塞、评论。
- worker 通过专门的 `kanban_*` 工具更新任务。
- board 是硬隔离边界。
- 连续 spawn 失败会自动 block，避免死循环。

### 4.4 Gateway：多用户、多平台会话管理

Gateway 负责 Telegram、Discord、Slack 等平台。

它管理：

- 平台消息映射到哪个 session。
- Agent 是否正在运行。
- 消息排队。
- `/stop`、`/new`、`/queue`、`/status` 等控制命令。
- approval / deny 等特殊命令绕过普通消息队列。
- session end/reset 时清理 Agent 资源。

这可以避免 Agent 正在运行时，新消息破坏 message role 顺序。

## 5. 与 LangGraph / Harness / Swarm 的区别

### 5.1 与 LangGraph

LangGraph 的核心抽象是 graph：

```text
State
  ↓
Node
  ↓
Edge / Conditional Edge
  ↓
Checkpoint
```

它强在：

- 显式定义节点和边。
- 状态流转可控。
- 多 Agent 工作流容易建模。
- checkpoint / resume / replay 能力强。
- 适合复杂业务流程。

Hermes-agent 的核心抽象是长期运行的 `AIAgent` loop：

```text
message history
  ↓
system prompt + memory + tools
  ↓
LLM call
  ↓
tool calls
  ↓
tool results
  ↓
继续 loop
```

区别：

| 维度 | Hermes-agent | LangGraph |
| --- | --- | --- |
| 核心模型 | Agent 主循环 | 状态图 / 工作流图 |
| 使用方式 | 直接运行 Agent 平台 | 编程定义 graph |
| 强项 | 工具、记忆、多端、插件、长期会话 | 可控流程、多节点编排、checkpoint |
| 多 Agent | `delegate_task`、Kanban、Cron | graph 中显式多节点 / 多 Agent |
| 流程可视性 | 隐式 loop + hooks | 显式 graph 更清晰 |

一句话：LangGraph 是用来搭 Agent 工作流的框架，Hermes-agent 是一个已经搭好的 Agent 运行平台。

### 5.2 与 Harness

如果 harness 指 agent harness / eval harness / task runner，它通常关注：

```text
给任务
  ↓
启动 Agent
  ↓
提供工具环境
  ↓
收集轨迹
  ↓
评测结果
```

Hermes-agent 不只是 harness。它包含任务运行和轨迹能力，但目标更宽：

- 用户交互。
- 长期会话。
- 记忆。
- 多平台消息入口。
- 插件。
- 技能。
- 定时任务。
- 多 Agent 工作队列。
- profile 隔离。

区别：

| 维度 | Hermes-agent | Harness |
| --- | --- | --- |
| 定位 | Agent runtime / platform | 任务运行 / 评测外壳 |
| 生命周期 | 长会话、长期运行 | 通常是单任务或批任务 |
| 用户入口 | CLI、TUI、Gateway、Dashboard | 通常是 API / 脚本 / runner |
| 记忆/会话 | 核心能力 | 不一定有 |
| 评测能力 | 可支持，但不是唯一目标 | 通常是核心目标之一 |

### 5.3 与 Swarm

Swarm 类框架通常强调 agent handoff：

```text
Agent A
  ↓ handoff
Agent B
  ↓ handoff
Agent C
```

Hermes-agent 更偏：

```text
主 Agent
  ↓ delegate_task
临时子 Agent 群

持久任务池
  ↓ dispatcher
长期 worker 群

定时任务
  ↓ scheduler
后台 Agent 任务
```

对比：

| 机制 | 类 Swarm 程度 | 生命周期 | 适合场景 |
| --- | ---: | --- | --- |
| `delegate_task` | 高 | 当前对话内 | 并行子任务、研究、调试、分析 |
| Kanban | 中高 | 持久化 | 多 Agent 长期协作、任务队列 |
| Cron | 中 | 持久化 / 定时 | 定时 Agent 工作、后台自动化 |
| Gateway 多 session | 中 | 长会话 | 多用户、多平台并发 |

Hermes-agent 的核心不是 handoff，而是任务委托、工作队列、长期调度和隔离。

## 6. 闭环自动优化

这个工程的闭环自动优化不是模型权重层面的训练，而是运行时经验沉淀、技能维护、任务反馈和可观测性驱动的优化。

整体链路：

```text
执行任务
  ↓
记录轨迹 / 工具调用 / 结果 / 技能使用
  ↓
沉淀到 memory、session、skill usage、observability
  ↓
后台 Agent / curator / cron / kanban worker 分析
  ↓
更新 skill、记忆、任务状态或配置建议
  ↓
后续会话加载改进后的上下文
```

### 6.1 Memory + Session

普通对话结束后，系统会把消息、工具结果、最终回复持久化到 `SessionDB`，同时 memory provider 会做同步。

这类闭环解决的是：

- 用户偏好。
- 项目上下文。
- 决策记录。
- 历史任务结果。
- 常见问题处理方式。

后续会话通过 `MemoryManager.prefetch` 把相关记忆重新注入 prompt，形成“越用越懂上下文”的运行时闭环。

### 6.2 Skill 自改进 + Curator

Agent 解决一个新问题后，可以把流程固化成 skill。之后 skill 会被跟踪使用情况：

- `use_count`
- `view_count`
- `patch_count`
- `last_activity_at`
- `state`
- `pinned`

Curator 会做后台维护：

```text
agent 创建 skill
  ↓
使用过程中记录 usage
  ↓
长期不用 → active → stale → archived
  ↓
有价值但漂移 → 辅助模型 review
  ↓
提出合并、修补、归档建议
```

安全边界：

- 只处理 `created_by: agent` 的 skill。
- 不碰内置 skill 和 hub 安装的 skill。
- 不自动删除，最多 archive。
- pinned skill 跳过自动迁移和 LLM review。
- 支持 dry-run、backup、rollback。

### 6.3 Cron 的监控闭环

Cron 适合“监控 → 判断 → 汇报/行动”的闭环：

```text
定时采集外部状态
  ↓
Agent 分析变化
  ↓
推送结果或写入状态
  ↓
下一次继续检查
```

常见场景：

- 定期检查 CI。
- 定时总结日志。
- 周期性扫描 issue。
- 每日生成报告。

### 6.4 Kanban 的任务反馈闭环

Kanban 优化的是任务流转质量：

- 哪些任务卡住了。
- 哪些 worker 失联了。
- 哪些任务需要阻塞。
- 哪些任务可以继续调度。
- 连续失败时自动停止重试。

它不是优化单次回答，而是优化长期多 Agent 工作流。

### 6.5 Hooks + Observability

hooks 和 observability 插件可以记录：

- LLM call。
- tool call。
- session lifecycle。
- agent start / end / step。
- pre / post tool call。
- pre / post LLM call。

这些数据可以用于：

- 失败分析。
- 工具调用统计。
- 成本 / 延迟分析。
- 高频错误归因。
- 后续人工或自动优化 prompt、skill、tool。

## 7. 一个更优雅的目标架构设计

当前架构的方向是合理的：统一 `AIAgent` 内核，多端复用，工具、记忆、技能、Provider、插件横向扩展。主要问题在于 `run_agent.py` 代码过长，`AIAgent` 承担了过多职责，逐渐形成 God object。

更优雅的改进方向不是推倒重写，而是保留 `AIAgent` 作为稳定门面，把内部能力按生命周期和职责边界拆成更小的 runtime 组件。

### 7.1 设计目标

目标不是单纯减少文件行数，而是降低耦合、提升可测试性和演进安全性：

- 保持 `AIAgent` 对外 API 稳定，避免 CLI / TUI / Gateway / Cron / Delegation 大面积改动。
- 将主循环中的横切职责拆到独立组件。
- 明确“单轮对话”“会话生命周期”“进程资源生命周期”的边界。
- 保护 prompt caching、toolset 稳定性、memory shutdown 时机等现有关键约束。
- 让新增能力优先接入明确扩展点，而不是继续堆进 `run_agent.py`。

### 7.2 目标分层

推荐目标结构如下：

```text
CLI / TUI / Gateway / Cron / Kanban
  ↓
AIAgent Facade
  ↓
AgentRuntime
  ├─ TurnRuntime
  ├─ ModelRuntime
  ├─ ToolRuntime
  ├─ SessionRuntime
  ├─ MemoryRuntime
  ├─ ContextRuntime
  ├─ CallbackRuntime
  └─ ResourceRuntime
```

其中 `AIAgent` 仍然是外部入口，继续提供：

```python
agent = AIAgent(...)
agent.chat(...)
agent.run_conversation(...)
agent.shutdown_memory_provider(...)
```

但它内部不再直接承载所有细节，而是把工作委托给 runtime 组件。

### 7.3 模块职责设计

建议逐步形成以下模块：

| 模块 | 职责 |
| --- | --- |
| `agent/runtime.py` | 高层对话运行器，承载主流程编排，但不直接实现所有细节。 |
| `agent/turn.py` | 表示一次用户输入到最终回复的 turn，包括 messages、tools、budget、metadata。 |
| `agent/model_runtime.py` | 模型调用、provider profile、fallback、api mode、reasoning config、service tier。 |
| `agent/tool_runtime.py` | tool schema 获取、特殊工具拦截、工具调用分发、tool result message 构造。 |
| `agent/session_lifecycle.py` | session 创建、恢复、消息持久化、metadata 更新、session end。 |
| `agent/memory_lifecycle.py` | memory prefetch、sync、commit、shutdown，封装 memory provider 生命周期。 |
| `agent/context_runtime.py` | token 预算、上下文压缩、prompt caching 约束、context engine 调用。 |
| `agent/callback_runtime.py` | streaming、tool progress、approval、clarify、interrupt 等回调适配。 |
| `agent/resource_runtime.py` | terminal、browser、background process、auxiliary client 等资源清理。 |
| `agent/events.py` | 统一生命周期事件模型，桥接 plugin hooks、gateway hooks、observability。 |

这样 `run_agent.py` 可以逐渐收敛成：

- 参数兼容层。
- `AIAgent` 门面。
- runtime 组件装配。
- 少量向后兼容方法。

### 7.4 理想主循环形态

重构完成后的主循环应该更接近下面的形态：

```python
def run_conversation(self, user_message, system_message=None, conversation_history=None, task_id=None):
    turn = self.session_runtime.prepare_turn(
        user_message=user_message,
        system_message=system_message,
        conversation_history=conversation_history,
        task_id=task_id,
    )

    self.memory_runtime.prefetch(turn)
    self.context_runtime.ensure_budget(turn)

    while turn.budget.can_continue():
        response = self.model_runtime.complete(turn)

        if response.tool_calls:
            tool_messages = self.tool_runtime.execute(response.tool_calls, turn)
            turn.messages.extend(tool_messages)
            continue

        return self.session_runtime.finish_turn(turn, response)
```

这个示例的重点不是具体 API，而是让主循环只表达核心控制流：

```text
准备 turn
  ↓
补充 memory / context
  ↓
调用模型
  ↓
执行工具
  ↓
收尾持久化
```

具体的 provider、tool、session、memory、callback、cleanup 细节都由专门模块处理。

### 7.5 事件总线与 Hook 收敛

当前系统里存在 plugin hooks、gateway hooks、observability hooks、shell hooks 等多套机制。长期看可以抽象一个内部事件层：

```text
AgentEvent
  ├─ session.started
  ├─ session.ended
  ├─ turn.started
  ├─ turn.completed
  ├─ model.requested
  ├─ model.completed
  ├─ tool.started
  ├─ tool.completed
  ├─ memory.synced
  └─ resource.cleaned
```

外部 hooks 不一定要统一暴露成同一套 API，但内部可以先统一事件对象，再由 adapter 分发给不同系统：

```text
AgentEventBus
  ├─ PluginHookAdapter
  ├─ GatewayHookAdapter
  ├─ ObservabilityAdapter
  └─ ShellHookAdapter
```

这样可以减少主循环里散落的 hook 调用，也方便做 tracing、debug、测试断言。

### 7.6 协作机制的优雅化

多 Agent 协作也可以进一步统一抽象。当前有 `delegate_task`、Cron、Kanban、Gateway background process，它们都在做“创建任务、运行 Agent、收集结果、更新状态”，只是生命周期不同。

可以抽象成统一的任务运行模型：

```text
AgentTask
  ├─ kind: interactive | delegated | cron | kanban | background
  ├─ session_id
  ├─ parent_task_id
  ├─ toolsets
  ├─ context
  ├─ isolation
  └─ lifecycle policy
```

不同机制只是在 lifecycle policy 上不同：

| 机制 | 生命周期策略 |
| --- | --- |
| `delegate_task` | 父 turn 内同步完成，父中断则取消。 |
| Cron | 独立 session，定时触发，可跨会话存在。 |
| Kanban | 由 dispatcher claim，worker heartbeat，失败可 block。 |
| Gateway background | 进程完成后触发新 agent turn 或通知。 |

这样能让协作机制共享更多底层能力，例如 task id、取消、timeout、trace、资源清理、结果归档。

### 7.7 推荐迁移路径

重构应分阶段进行，避免直接改动主循环核心。

第一阶段：抽外围生命周期。

- 抽 `agent/session_lifecycle.py`。
- 抽 `agent/memory_lifecycle.py`。
- 抽 `agent/resource_runtime.py`。
- 保持 `run_conversation()` 主控制流基本不变。

第二阶段：抽工具层。

- 新增 `agent/tool_runtime.py`。
- 把特殊工具拦截、`handle_function_call` 包装、tool result message 构造集中进去。
- 主循环只保留“如果有 tool_calls，则交给 tool runtime 执行”。

第三阶段：抽模型层。

- 新增 `agent/model_runtime.py`。
- 收拢 provider、fallback、api mode、reasoning config、streaming 细节。
- 主循环只调用 `model_runtime.complete(turn)`。

第四阶段：抽 turn 和 context。

- 新增 `agent/turn.py`。
- 新增或强化 `agent/context_runtime.py`。
- 把 messages、budget、compression、system prompt 状态显式建模。

第五阶段：事件层收敛。

- 新增 `agent/events.py`。
- 先内部使用，再逐步把 plugin / gateway / observability hook 适配过去。

### 7.8 必须保留的工程约束

重构时要特别保护以下约束：

- 不在对话中途随意改变 toolset。
- 不在对话中途随意重建 system prompt 或重新加载 memory。
- memory shutdown 只发生在真实 session 边界，而不是每轮结束。
- `AIAgent` 构造参数和 `run_conversation()` 返回结构保持兼容。
- Gateway 的 `/stop`、`/approve`、`/deny` 等控制命令仍能绕过普通消息队列。
- `delegate_task` 仍保持子 Agent 上下文隔离、并发限制、深度限制和父中断取消。
- Cron 仍使用 fresh session，默认 `skip_memory=True`。
- Kanban worker 的 `kanban_*` toolset 仍由环境 gating 控制。

### 7.9 测试策略

每次抽取都应该优先补行为测试，而不是只做文件移动。

重点测试：

- 无工具调用的一轮普通回复。
- 单工具调用、多工具调用、工具错误返回。
- session persist / resume。
- memory prefetch / sync / shutdown 时机。
- context compression 触发和不触发。
- prompt caching 相关不变量。
- Gateway `/stop`、`/new`、approval / deny。
- `delegate_task` 子代理隔离、并发限制、深度限制。
- Cron fresh session 和 `skip_memory=True`。
- Kanban worker toolset gating。

测试应按项目规范通过 `scripts/run_tests.sh` 运行，避免本地环境和 CI 行为漂移。

### 7.10 最终形态

最终目标不是让 `AIAgent` 消失，而是让它成为稳定门面：

```text
AIAgent = public facade
AgentRuntime = orchestration
Runtime components = cohesive responsibilities
Hooks / plugins / tools = explicit extension points
```

这种架构既保留当前平台化优势，也能解决 `run_agent.py` 过长带来的维护风险。

## 8. 总结

`hermes-agent` 的核心优势不是某个单点功能，而是把 Agent 做成了一个长期运行的工程化平台：

- `AIAgent` 提供统一内核。
- CLI / TUI / Gateway / Dashboard 多端复用。
- 工具、Provider、Memory、Skill、Plugin 体系支持横向扩展。
- `delegate_task` 支持短期子代理协作。
- Cron 支持持久后台任务。
- Kanban 支持多 Agent 长期工作队列。
- Curator、Memory、Hooks、Observability 形成运行时闭环优化。

如果用一句话概括：

```text
Hermes-agent 不是单纯的 Agent 编排框架，而是一个面向长期使用、多端接入、多工具扩展、多 Agent 协作和运行时自优化的 Agent Runtime。
```
