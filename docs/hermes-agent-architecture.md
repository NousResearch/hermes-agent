# Hermes Agent 完整架构分析

> 基于 Nous Research 开源的 Hermes Agent v0.11.0 源码分析，包含二次开发视角的解读。
> 分析日期：2026-05-08

---

## 目录

- [一、项目概述](#一项目概述)
- [二、整体架构](#二整体架构)
- [三、核心流程详解](#三核心流程详解)
  - [3.1 启动流程](#31-启动流程)
  - [3.2 对话核心循环](#32-对话核心循环)
  - [3.3 系统提示词构建](#33-系统提示词构建)
  - [3.4 工具调用分发](#34-工具调用分发)
  - [3.5 模型适配器层](#35-模型适配器层)
- [四、关键子系统](#四关键子系统)
  - [4.1 会话持久化](#41-会话持久化)
  - [4.2 消息网关](#42-消息网关)
  - [4.3 记忆系统](#43-记忆系统)
  - [4.4 技能系统](#44-技能系统)
  - [4.5 定时任务](#45-定时任务)
  - [4.6 上下文压缩](#46-上下文压缩)
  - [4.7 工具注册与发现](#47-工具注册与发现)
  - [4.8 插件系统](#48-插件系统)
- [五、ReAct 模式扩展增强](#五react-模式扩展增强)
  - [5.1 经典 ReAct 回顾](#51-经典-react-回顾)
  - [5.2 扩展全景](#52-扩展全景)
  - [5.3 扩展1：持久记忆系统](#53-扩展1持久记忆系统)
  - [5.4 扩展2：自我进化技能系统](#54-扩展2自我进化技能系统)
  - [5.5 扩展3：层级委派与并行执行](#55-扩展3层级委派与并行执行)
  - [5.6 扩展4：人机协同中断机制](#56-扩展4人机协同中断机制)
  - [5.7 扩展5：多层容错与故障转移](#57-扩展5多层容错与故障转移)
  - [5.8 扩展6：上下文压缩](#58-扩展6上下文压缩)
  - [5.9 扩展7：插件生命周期钩子](#59-扩展7插件生命周期钩子)
  - [5.10 扩展8：Prompt 缓存优化](#510-扩展8prompt-缓存优化)
  - [5.11 扩展9：后台审查代理](#511-扩展9后台审查代理)
  - [5.12 扩展10：文件系统检查点](#512-扩展10文件系统检查点)
  - [5.13 对比总结](#513-对比总结)
- [六、数据流总结](#六数据流总结)
- [七、二次开发指南](#七二次开发指南)

---

## 一、项目概述

### 1.1 项目简介

**Hermes Agent** 是 [Nous Research](https://nousresearch.com) 开源的自我进化型 AI Agent。其核心定位是一个可自我学习的、多平台的 AI 编程助手，对标 Claude Code。

- **许可协议**：MIT
- **语言**：Python 3.13+
- **代码规模**：约 400+ 源文件，核心文件 `run_agent.py` 约 12,000 行、`cli.py` 约 11,000 行
- **版本**：v0.11.0

### 1.2 核心能力

- **真实终端界面（TUI）**：多行编辑、斜杠命令自动补全、对话历史、流式工具输出
- **多平台消息网关**：Telegram、Discord、Slack、WhatsApp、Signal、钉钉、飞书、企业微信 等 15+ 平台
- **闭合学习循环**：Agent 自主管理记忆、自动创建技能、自我优化、FTS5 全文搜索历史会话
- **定时自动化**：内置 cron 调度器，结果投递到任意平台
- **委派与并行化**：spawn 隔离子 Agent 并行工作流
- **六大终端后端**：本地、Docker、SSH、Daytona、Singularity、Modal（无服务器）
- **200+ 模型**：通过 OpenRouter 及原生支持 OpenAI、Anthropic、Nous Portal、NVIDIA NIM、AWS Bedrock、Gemini 等

---

## 二、整体架构

### 2.1 分层架构图

```
┌──────────────────────────────────────────────────────┐
│                    入口层                            │
│  hermes CLI  │  hermes-agent API  │  Gateway 守护进程│
│  (hermes_cli/)│  (run_agent.py)   │  (gateway/run.py)│
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                  编排/会话层                          │
│  HermesCLI (cli.py:1801)  │  AIAgent (run_agent.py:810)│
│  交互式 REPL + TUI         │  核心对话+工具调用循环    │
│  斜杠命令 + 自动补全       │  多 provider 适配        │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                    能力层                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ 工具系统  │  │ 技能系统  │  │ 记忆系统         │   │
│  │ 70+工具   │  │ 26类技能  │  │ Honcho/Mem0/内置 │   │
│  │ 自注册模式│  │ 自动创建  │  │ 三级记忆架构     │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                   基础设施层                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Agent/  │  │ Gateway/ │  │ Cron/  │ Plugins/ │    │
│  │ 7种模型 │  │ 15+平台  │  │ 定时器  │ 8个钩子  │    │
│  │ 适配器  │  │ 适配器   │  │ 调度器  │ 生命周期 │    │
│  └─────────┘  └──────────┘  └──────────────────┘    │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                   持久化层                            │
│  SQLite + WAL (hermes_state.py)  │  文件系统         │
│  FTS5 全文搜索  │  会话管理      │  skills/memory    │
│  会话链 parent_session_id        │  轨迹 JSONL 存储   │
└──────────────────────────────────────────────────────┘
```

### 2.2 关键文件清单

| 文件 | 大小 | 用途 |
|------|------|------|
| `run_agent.py` | ~659 KB | **AIAgent 类**——核心对话循环，约 12,000 行 |
| `cli.py` | ~502 KB | **HermesCLI 类**——交互式 CLI 编排器，约 11,000 行 |
| `model_tools.py` | ~28 KB | 工具编排——`discover_builtin_tools()`、`handle_function_call()` |
| `toolsets.py` | ~25 KB | 工具集定义和解析 |
| `hermes_state.py` | ~76 KB | SessionDB——SQLite 会话存储 + FTS5 搜索 |
| `hermes_constants.py` | ~10 KB | 路径常量——`get_hermes_home()`、profile 感知路径 |
| `hermes_logging.py` | ~14 KB | 日志系统——agent.log / errors.log / gateway.log |
| `agent/prompt_builder.py` | - | 系统提示词组装——身份、记忆、技能、平台提示 |
| `agent/memory_manager.py` | - | 外部记忆 provider 管理 |
| `agent/context_compressor.py` | - | 上下文压缩引擎 |
| `agent/error_classifier.py` | - | API 错误分类 → FailoverReason |
| `tools/registry.py` | - | 工具自注册 + 发现 + 分发 |
| `gateway/run.py` | ~539 KB | 消息网关运行器 |
| `cron/scheduler.py` | - | Cron 调度器 |
| `batch_runner.py` | ~55 KB | 并行批处理（轨迹生成） |
| `trajectory_compressor.py` | ~65 KB | 轨迹压缩（训练用） |

### 2.3 主要入口点

| 入口 | 定义位置 | 用途 |
|------|----------|------|
| `hermes` | `pyproject.toml` → `hermes_cli/main.py:main()` | CLI 主命令 |
| `hermes-agent` | `pyproject.toml` → `run_agent.py:main()` | 直接调用 AIAgent |
| `hermes-acp` | `pyproject.toml` → `acp_adapter/entry.py:main()` | ACP 编辑器集成 |
| Gateway | `gateway/run.py` | 多平台消息网关守护进程 |
| Docker | `Dockerfile` + `docker/entrypoint.sh` | 容器化部署 |

---

## 三、核心流程详解

### 3.1 启动流程

```
用户执行 hermes chat
  │
  ▼
hermes_cli/main.py:main()
  ├─ 预解析 --profile / -p 参数，设置 HERMES_HOME
  ├─ 加载 ~/.hermes/.env 环境变量（hermes_cli/env_loader.py）
  └─ 路由到 chat 子命令
      │
      ▼
HermesCLI.__init__() (cli.py:1801)
  ├─ 读取 config.yaml 配置
  │   ├─ 模型选择（model.default / model.provider / model.base_url）
  │   ├─ 显示配置（streaming / compact / tool_progress / bell_on_complete）
  │   ├─ Agent 配置（max_turns / system_prompt / prefill_messages）
  │   ├─ 记忆配置（memory_enabled / provider / nudge_interval）
  │   └─ 检查点配置（enabled / max_snapshots）
  ├─ 解析 Provider（"auto" → 从 base_url / env 自动检测）
  ├─ 创建 FileHistory（prompt_toolkit 历史持久化）
  ├─ 构建 TUI（prompt_toolkit Application + Layout）
  └─ 初始化 AIAgent 实例
      │
      ▼
进入交互式 REPL 循环
  用户输入 → HermesCLI.chat() → AIAgent.run_conversation()
```

### 3.2 对话核心循环

`run_agent.py:9212` 的 `AIAgent.run_conversation()` 是整个系统的核心引擎。

```text
1. 预处理阶段
   ├─ 清理用户输入（代理字符、泄露的 memory 标签）
   ├─ 重置重试计数器、迭代预算
   ├─ 检查中断信号
   ├─ 恢复 TODO 状态（从历史消息中反序列化）
   ├─ 自动记忆/技能 nudge 检查
   └─ 构建/加载系统提示词（缓存复用，见 3.3 节）

2. 预检阶段
   ├─ Preflight 上下文压缩（切换小窗口模型时主动压缩，见 4.6 节）
   ├─ Plugin hook: pre_llm_call（注入额外上下文到用户消息）
   ├─ 外部记忆 provider 预取（Honcho/Mem0 召回，见 4.3 节）
   └─ Memory nudge 检查（每 N 轮提醒 Agent 回顾记忆）

3. 工具调用循环（while api_call_count < max_iterations）
   │
   ├─ [每次迭代]
   │   ├─ 检查中断信号（见 5.6 节）
   │   ├─ 消耗共享迭代预算（父 + 所有子 Agent）
   │   ├─ 触发 step_callback（gateway 事件钩子）
   │   ├─ 检查 /steer 队列（用户中途注入指令，见 5.6 节）
   │   ├─ 构建 API 消息列表
   │   │   ├─ 注入外部记忆到当前用户消息（API-call-time only）
   │   │   ├─ 复制 reasoning_content（多轮推理上下文保持）
   │   │   ├─ 清理内部字段（finish_reason、_thinking_prefill 等）
   │   │   ├─ 追加 ephemeral system prompt
   │   │   ├─ 注入 prefill messages（few-shot 示例）
   │   │   ├─ 应用 Anthropic prompt caching（见 5.10 节）
   │   │   ├─ 安全网：修复孤立 tool_call/tool_result
   │   │   └─ 标准化消息格式（JSON key 排序，KV 缓存友好）
   │   │
   │   ├─ 调用 LLM API（带重试循环，见 5.7 节）
   │   │   ├─ 速率限制预检（Nous Portal rate guard）
   │   │   ├─ 选择适配器（Chat Completions / Anthropic Messages / Bedrock / Codex）
   │   │   ├─ 优先 streaming 路径（健康检查：90s 陈旧流检测）
   │   │   ├─ 错误分类 → FailoverReason
   │   │   │   ├─ context_too_long     → 触发上下文压缩
   │   │   │   ├─ rate_limited         → fallback 模型
   │   │   │   ├─ auth_error           → 重试换 key / OAuth 刷新
   │   │   │   ├─ empty_content        → thinking prefill + 重试
   │   │   │   └─ billing              → fallback 模型
   │   │   └─ 成功 → 返回 assistant_message
   │   │
   │   ├─ 解析响应
   │   │   ├─ 提取 reasoning/thinking 内容（<think> 标签处理）
   │   │   ├─ 检查 tool_calls
   │   │   └─ 处理 text content（流式回调 → TTS/显示）
   │   │
   │   ├─ [有 tool_calls] → 执行工具
   │   │   ├─ 判断并行/串行（读操作可并行，写操作检查路径冲突）
   │   │   ├─ [并行] _execute_tool_calls_concurrent()
   │   │   │   └─ ThreadPoolExecutor 多线程并发执行
   │   │   ├─ [串行] _execute_tool_calls_sequential()
   │   │   ├─ 每个工具执行：
   │   │   │   ├─ 检查点快照（写文件/破坏性命令前自动 git commit）
   │   │   │   ├─ Plugin hook: pre_tool_call（可阻止执行）
   │   │   │   ├─ _invoke_tool() → registry.dispatch() → handler()
   │   │   │   ├─ Plugin hook: post_tool_call（观察结果）
   │   │   │   └─ Plugin hook: transform_tool_result（可修改结果）
   │   │   ├─ 重置对应 nudge 计数器（memory/skill_manage）
   │   │   ├─ 追加 tool_result 到 messages
   │   │   └─ 继续循环
   │   │
   │   └─ [无 tool_calls] → 最终响应
   │       ├─ 执行 checkpoints（最终文件快照）
   │       ├─ Skill nudge 检查（自动创建/优化技能，见 5.4 节）
   │       └─ 跳出循环

4. 后处理阶段
   ├─ 持久化会话到 SQLite（hermes_state.py）
   ├─ 后台审查 Agent（_spawn_background_review，见 5.11 节）
   │   ├─ fork 子 AIAgent，注入对话历史
   │   ├─ 审查 prompt：提取记忆 + 创建技能
   │   └─ 输出：💾 已保存 1 条记忆 · 已创建技能 "xxx"
   ├─ 外部记忆同步（on_turn_end → 下次 prefetch）
   ├─ Plugin hook: on_session_end
   └─ 返回最终结果
```

### 3.3 系统提示词构建

`run_agent.py:4474` 的 `_build_system_prompt()` 组装系统提示词，构建**一次**后缓存整个会话复用（最大化 prompt cache 命中率）。

```text
Layer 1:  Agent 身份
          ├─ SOUL.md（优先，用户自定义人格）
          └─ DEFAULT_AGENT_IDENTITY（agent/prompt_builder.py，默认身份）
Layer 2:  工具感知行为指导
          ├─ memory 工具加载 → MEMORY_GUIDANCE（如何使用记忆）
          ├─ session_search 工具加载 → SESSION_SEARCH_GUIDANCE
          └─ skill_manage 工具加载 → SKILLS_GUIDANCE
Layer 3:  强制工具使用指导（针对弱模型的反幻觉机制）
          └─ TOOL_USE_ENFORCEMENT_GUIDANCE
              ├─ Google 模型 → GOOGLE_MODEL_OPERATIONAL_GUIDANCE
              └─ OpenAI GPT/Codex → OPENAI_MODEL_EXECUTION_GUIDANCE
Layer 4:  用户/网关自定义系统提示词
          └─ config.yaml agent.system_prompt 或 Gateway 注入
Layer 5:  持久化记忆块
          ├─ MemoryStore.format_for_system_prompt("memory")
          └─ MemoryStore.format_for_system_prompt("user")  # USER.md
Layer 6:  外部记忆 provider 系统提示词
          └─ MemoryManager.build_system_prompt()  # Honcho 等
Layer 7:  技能索引
          └─ build_skills_system_prompt() → 列出所有可用技能及触发条件
Layer 8:  上下文文件
          └─ AGENTS.md / .cursorrules / HERMES.md（项目级指令）
Layer 9:  时间戳 + 会话元数据
          └─ "Conversation started: Friday, May 08, 2026 03:00 PM"
          └─ Session ID / Model / Provider
Layer 10: 平台格式化提示
          └─ PLATFORM_HINTS[platform] → WhatsApp(不要Markdown) / Telegram / Discord...
Layer 11: 环境提示
          └─ WSL / Termux 路径翻译提示
```

**关键设计决策**：动态内容（外部记忆召回 + plugin 上下文）注入到**用户消息**而非系统提示词，确保系统提示词在整个会话中保持 bit-perfect 一致，最大化 Anthropic prompt cache 命中率。

### 3.4 工具调用分发

`model_tools.py` 是工具系统的薄编排层：

```text
handle_function_call(name, args, task_id, ...)
  │
  ├─ 参数类型强制转换（coerce_tool_args）
  ├─ Plugin hook: pre_tool_call（可阻塞执行）
  │
  ├─ [特殊工具] AIAgent 内联处理
  │   ├─ todo / memory / session_search / clarify / delegate_task
  │   └─ 外部记忆 provider 工具
  │
  ├─ [通用工具] registry.dispatch(name, args)
  │   └─ 查找 ToolEntry → 调用 handler（同步/异步桥接）
  │
  ├─ Plugin hook: post_tool_call（观察结果 + 耗时）
  └─ Plugin hook: transform_tool_result（可修改/替换结果）
```

**异步桥接**（`model_tools.py:40-100`）：工具 handler 可以是 async 函数，通过持久化 event loop 从同步上下文调用，避免 "Event loop is closed" 错误。

### 3.5 模型适配器层

`agent/` 目录下的适配器支持 7 种 API 协议：

| 协议 | 适配器 | 适用场景 |
|------|--------|----------|
| Chat Completions | OpenAI SDK 通用 | OpenRouter、大多数第三方 provider |
| Anthropic Messages | `anthropic_adapter.py` | Claude 原生 API、Anthropic 兼容端点 |
| Bedrock Converse | `bedrock_adapter.py` | AWS Bedrock |
| Codex Responses | `codex_responses_adapter.py` | OpenAI Codex、xAI Grok |
| Gemini Native | `gemini_native_adapter.py` | Google Gemini API |
| Gemini Cloud Code | `gemini_cloudcode_adapter.py` | Google Cloud Code |
| Copilot ACP | `copilot_acp_client.py` | GitHub Copilot（通过 ACP） |

**自动检测逻辑**（`run_agent.py:977-1008`）：

```python
if provider == "anthropic" or base_url == "api.anthropic.com":
    api_mode = "anthropic_messages"
elif provider == "openai-codex" or provider == "xai":
    api_mode = "codex_responses"
elif provider == "bedrock" or base_url.startswith("bedrock-runtime."):
    api_mode = "bedrock_converse"
elif base_url.endswith("/anthropic"):  # 第三方兼容端点
    api_mode = "anthropic_messages"
else:
    api_mode = "chat_completions"  # 默认
```

---

## 四、关键子系统

### 4.1 会话持久化

**文件**：`hermes_state.py`

**核心设计**：
- **SQLite + WAL 模式**：支持并发读 + 单写者（gateway 多平台场景）
- **FTS5 全文搜索**：虚拟表索引所有会话消息，`session_search` 工具直接搜索
- **会话链**：上下文压缩后通过 `parent_session_id` 链接父子会话
- **Schema 版本管理**：当前 v9，自动迁移

**表结构**：

```sql
sessions (
    id TEXT PRIMARY KEY,          -- UUID 会话标识
    source TEXT NOT NULL,         -- 'cli' | 'telegram' | 'discord' | ...
    user_id TEXT,                 -- 平台用户 ID
    model TEXT,                   -- 模型名称
    system_prompt TEXT,           -- 系统提示词快照
    parent_session_id TEXT,       -- 压缩链
    started_at / ended_at REAL,   -- 时间戳
    message_count / tool_call_count INTEGER,
    input_tokens / output_tokens / cache_read_tokens / cache_write_tokens INTEGER,
    estimated_cost_usd / actual_cost_usd REAL,
    title TEXT                    -- 自动生成的会话标题
)

messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    role TEXT NOT NULL,           -- 'user' | 'assistant' | 'tool' | 'system'
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,              -- JSON 序列化的 tool_calls
    tool_name TEXT,               -- 工具名称（方便查询）
    timestamp REAL
)
```

### 4.2 消息网关

**文件**：`gateway/run.py`（~539 KB）

**架构**：单进程运行所有平台适配器（asyncio 并发）

**核心流程**：

```text
GatewayRunner.start()
  ├─ 加载 gateway.yaml 配置
  ├─ 初始化每个平台适配器（继承 BasePlatformAdapter）
  │   ├─ telegram.py / discord.py / slack.py / whatsapp.py
  │   ├─ dingtalk.py / feishu.py / wecom.py / weixin.py
  │   └─ signal.py / matrix.py / mattermost.py / email.py / sms.py ...
  ├─ 启动后台 cron ticker（每 60 秒检查定时任务）
  └─ 进入事件循环

消息处理:
  收到消息 → BasePlatformAdapter.on_message()
    ├─ 解析用户身份（user_id / chat_id / thread_id）
    ├─ 从 LRU 缓存获取/创建 AIAgent 实例（128 上限，1h 空闲 TTL）
    ├─ 构建会话上下文（platform / user_id / gateway_session_key）
    ├─ AIAgent.run_conversation(user_message)
    │   └─ streaming callback → 实时推送到聊天平台
    ├─ 投递回复（支持文本 + 文件附件 + 图片/视频）
    └─ 更新会话状态
```

**平台适配器模式**（`gateway/platforms/base.py`）：

```python
class BasePlatformAdapter:
    async def start(self): ...
    async def stop(self): ...
    async def send_message(self, chat_id, text, attachments): ...
    async def on_message(self, message): ...
```

### 4.3 记忆系统

**三层记忆架构**：

```text
┌─────────────────────────────────────────────┐
│ Layer 1: 系统提示词记忆（冻结快照）          │
│ 在整个会话中保持不变，构建时从磁盘加载       │
│ ├─ MEMORY.md → 关键事实、偏好、决策          │
│ └─ USER.md  → 用户画像、角色、知识背景       │
│ 格式：Markdown + YAML frontmatter            │
│ 存储：~/.hermes/memory/                      │
├─────────────────────────────────────────────┤
│ Layer 2: 外部记忆召回（每轮预取）             │
│ 每次 API 调用前从外部 provider 检索          │
│ ├─ Honcho：辩证用户建模（dialectic model）    │
│ ├─ Mem0：语义记忆检索                        │
│ └─ Supermemory：向量化记忆存储               │
│ 注入位置：用户消息（不破坏 prompt cache）     │
├─────────────────────────────────────────────┤
│ Layer 3: 后台自动存储（对话结束后）           │
│ _spawn_background_review()                   │
│ ├─ fork 子 AIAgent 审查对话                   │
│ ├─ 自动调用 memory 工具保存关键信息           │
│ └─ 用户无感知，不阻塞主回复                   │
└─────────────────────────────────────────────┘
```

**Memory Nudge 机制**：
- 每 N 轮对话（默认 10 轮），如果 Agent 没有主动使用 `memory` 工具
- 系统提示词中会建议 Agent "检查是否有值得保存的信息"
- 如果 Agent 调用了 `memory` 工具，计数器重置

**关键代码路径**：
- 初始化：`run_agent.py:1596-1682`
- 系统提示词注入：`run_agent.py:4557-4575`
- API 调用时注入：`run_agent.py:9721-9732`
- 后台审查：`run_agent.py:3220-3328`

### 4.4 技能系统

**文件**：`skills/`（26 类） + `optional-skills/`

**技能文件格式**：

```markdown
---
name: git-workflow
description: 标准 Git 工作流操作
conditions: "用户提到 git、分支、提交、PR 时触发"
platforms: ["cli", "telegram"]
---

## Git Workflow Skill

当用户需要 Git 操作时，按以下流程执行：
1. 先检查当前分支状态
2. ...
```

**技能索引注入**：`agent/prompt_builder.py` 的 `build_skills_system_prompt()` 扫描所有技能文件的 frontmatter，生成如下格式的索引：

```
Available Skills:
- git-workflow: 标准 Git 工作流操作 [触发: git、分支、提交、PR]
- code-review: 代码审查流程 [触发: review、检查代码]
- ...
```

**自动技能创建**：
- Skill nudge：每 N 轮工具调用（默认 10 轮），如果 Agent 没有使用 `skill_manage`
- 后台审查 Agent 检测可复用模式 → 自动调用 `skill_manage` 创建技能
- 技能存入 `~/.hermes/skills/`，下次对话自动加载

### 4.5 定时任务

**文件**：`cron/scheduler.py`

**架构**：

```text
Gateway 后台线程（每 60 秒）
  │
  ▼
tick() → 检查到期任务
  ├─ 文件锁（~/.hermes/cron/.tick.lock）防止并发
  ├─ 遍历 cron_jobs.json 中的任务
  ├─ 对每个到期任务：
  │   ├─ 创建独立 AIAgent 实例
  │   ├─ 注入 cron 配置（工具集、模型、系统提示词）
  │   ├─ AIAgent.run_conversation(cron_prompt)
  │   └─ 投递结果到指定平台
  └─ 写执行日志
```

**任务管理工具**：`cronjob` 工具让 Agent 可以自主创建/修改/删除定时任务。

### 4.6 上下文压缩

**文件**：`agent/context_compressor.py`

**触发机制**：

```text
主动预检（Preflight）:
  进入主循环前 → estimate_request_tokens_rough()
  → token 估算 >= threshold_tokens
  → 直接触发压缩，不等 API 报错
  → 最多 3 轮压缩迭代

被动触发（Reactive）:
  API 返回 context_too_long 错误
  → classify_api_error() → FailoverReason.context_too_long
  → _compress_context()
```

**压缩策略**：

```text
保留:
  ├─ protect_first_n（前 4 条消息）—— 任务定义和初始上下文
  └─ protect_last_n（后 6 条消息）—— 最近的对话状态

压缩:
  └─ 中间消息 → 发送给模型做摘要 → 替换为摘要消息

分裂:
  └─ 创建新的 child session（通过 parent_session_id 链接原始会话）
  └─ 重建系统提示词（重新加载记忆和上下文文件）
```

### 4.7 工具注册与发现

**文件**：`tools/registry.py`

**自注册模式**：每个工具文件在模块级调用 `registry.register()`：

```python
# 例如 tools/terminal_tool.py 末尾
registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,      # OpenAI function schema
    handler=_handle_terminal,    # 实际执行函数
    check_fn=check_terminal_requirements,  # 可用性检查
    emoji="💻",
    max_result_size_chars=100_000,
)
```

**发现流程**（`discover_builtin_tools()` at `registry.py:56`）：

```text
1. 扫描 tools/*.py 文件
2. 静态分析检测是否包含 "registry.register(" 调用
3. importlib.import_module() 触发自注册
4. 注册检查：
   ├─ 同名工具 → 仅允许 MCP→MCP 覆盖
   ├─ 不同 toolset 同名 → 拒绝注册（防止 shadowing）
   └─ 允许 MCP 动态刷新（nuke-and-repave）
5. MCP 工具也可动态注册（通过 mcp_tool.py）
```

**工具注册表**：`ToolRegistry`（`registry.py:100`）是线程安全的单例，使用 `RLock` 保护并发读写。

### 4.8 插件系统

**文件**：`plugins/` + `hermes_cli/plugins.py`

**8 个生命周期钩子**：

```text
1. on_session_start   → 会话创建时（预热缓存、初始化状态）
2. pre_llm_call       → API 调用前（注入上下文，可返回 context/string）
3. pre_api_request    → API 请求发送前（观察请求参数，Langfuse tracing）
4. pre_tool_call      → 工具调用前（可返回 block_message 阻止执行）
5. post_tool_call     → 工具调用后（观察结果+耗时，审计日志）
6. transform_tool_result → 工具结果转换（可返回替换字符串）
7. post_llm_call      → LLM 响应后（观察响应）
8. on_session_end     → 会话结束时（清理、flush 缓冲区）
```

**钩子返回值语义**：

| 钩子 | 返回值含义 |
|------|-----------|
| `pre_llm_call` | `str` 或 `{"context": "..."}` → 注入到用户消息 |
| `pre_tool_call` | `{"block": True, "message": "..."}` → 阻止工具执行 |
| `transform_tool_result` | `str` → 替换原结果 |
| 其他钩子 | 返回值被忽略（仅用于观察/记录） |

**插件示例**：
- `plugins/memory/` — Honcho、Mem0、Supermemory 记忆 provider
- `plugins/langfuse-tracing/` — Langfuse 可观测性集成
- `plugins/spotify/` — Spotify 音乐控制
- `plugins/context_engine/` — 上下文引擎扩展

---

## 五、ReAct 模式扩展增强

### 5.1 经典 ReAct 回顾

经典 ReAct（Reasoning + Acting）模式的核心循环：

```text
┌──────────────────────────────────────────┐
│                                          │
│   Thought ──→ Action ──→ Observation     │
│      ↑                          │        │
│      └──────────────────────────┘        │
│                                          │
│  1. Thought:  "我需要查看端口状态"        │
│  2. Action:   terminal("netstat -tlnp")  │
│  3. Observe:  "端口 443 被占用"           │
│  4. Thought:  "需要检查哪个进程..."       │
│  5. Action:   terminal("lsof -i :443")   │
│  6. Observe:  "nginx (pid 1234)"         │
│  7. Thought:  "最终回复用户..."           │
│                                          │
└──────────────────────────────────────────┘
```

**经典 ReAct 的核心局限**：

| 问题 | 表现 |
|------|------|
| **无记忆** | 每次对话从零开始，不知道用户是谁 |
| **无学习** | 不会从经验中改进，同样的问题重复推理 |
| **串行单线程** | 无法并行执行独立操作 |
| **不可中断** | 一旦开始就必须等它完成 |
| **零容错** | API/工具调用失败 → 崩溃 |
| **上下文无界** | 长对话超出窗口 → 截断丢失信息 |
| **封闭循环** | 无法在循环中注入外部逻辑 |

Hermes Agent 在这 7 个维度上做了系统性的扩展。

### 5.2 扩展全景

```text
                    ┌─────────────────────────┐
                    │  扩展5: 多层容错         │
                    │  (错误分类→定向恢复      │
                    │   →fallback链→降级)      │
                    └───────────┬─────────────┘
                                │
┌─────────────────────────┐    │    ┌─────────────────────────┐
│  扩展1: 持久记忆系统     │    │    │  扩展2: 自我进化技能    │
│  (三层记忆架构)          │    │    │  (跨对话技能积累)       │
└───────────┬─────────────┘    │    └───────────┬─────────────┘
            │                  │                │
            ▼                  ▼                ▼
┌──────────────────────────────────────────────────────┐
│                 Hermes Enhanced ReAct Loop            │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ [记忆召回] → [技能匹配] → [上下文注入]        │    │
│  └──────────────────┬───────────────────────────┘    │
│                     ▼                                 │
│  ┌──────────────────────────────────────────────┐    │
│  │ Thought → Action(可并行/可委派) → Observation │    │
│  │    ↑                          │              │    │
│  │    └──────────────────────────┘              │    │
│  │         [中断/Steer]  [检查点]                │    │
│  └──────────────────────────────────────────────┘    │
│                     │                                 │
│  ┌──────────────────▼───────────────────────────┐    │
│  │ [后台审查] → [自动记忆] → [自动技能]          │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
            │                  │                │
            ▼                  ▼                ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│  扩展3: 层级委派+并行    │  │  扩展4: 中断+Steer      │
│  (树形ReAct+智能并行)    │  │  (人机协同)              │
└─────────────────────────┘  └─────────────────────────┘

            ┌─────────────────────────┐
            │  扩展6: 上下文压缩       │
            │  (主动+被动, 会话分裂)   │
            └─────────────────────────┘
            ┌─────────────────────────┐
            │  扩展7: 插件钩子系统     │
            │  (8个生命周期钩子)       │
            └─────────────────────────┘
            ┌─────────────────────────┐
            │  扩展8: Prompt缓存      │
            │  (~75%输入成本节省)      │
            └─────────────────────────┘
            ┌─────────────────────────┐
            │  扩展9: 后台审查代理     │
            │  (元认知反射层)          │
            └─────────────────────────┘
            ┌─────────────────────────┐
            │  扩展10: 文件检查点      │
            │  (操作前自动git快照)     │
            └─────────────────────────┘
```

### 5.3 扩展1：持久记忆系统

**解决的问题**：经典 ReAct 每次对话从零开始，不知道用户是谁、历史做了什么。

**三层记忆架构**：

```text
Layer 1: 系统提示词记忆（冻结快照，全对话可见）
  └─ MemoryStore → "你是 Renyh，资深 Python 开发者..."
  └─ 构建时从磁盘加载，整个会话不变

Layer 2: 外部记忆召回（每轮动态检索，API-call-time only）
  └─ MemoryManager.prefetch_all(query)
  └─ Honcho → "你上次讨论过这个 bug，根因是..."
  └─ 注入到用户消息，不破坏 prompt cache

Layer 3: 后台自动存储（对话结束后自动提取）
  └─ _spawn_background_review()
  └─ fork 子 Agent 审查对话 → 自动调用 memory 工具
  └─ "用户偏好使用 async/await 而非回调"
```

**关键实现**：
- 初始化：`run_agent.py:1596-1682` — MemoryStore（内置）+ MemoryManager（外部）
- 系统提示词注入：`run_agent.py:4557-4575`
- API 调用注入：`run_agent.py:9721-9732` — `_ext_prefetch_cache` 注入用户消息
- 后台审查：`run_agent.py:3220-3328` — `_spawn_background_review()` 创建子 Agent

**与 ReAct 的区别**：ReAct 的 Observation 只是当前工具调用的结果。Hermes 的 Observation 包含**长期记忆召回的上下文**——模型看到的不只是 "netstat 输出"，还包括 "这个用户上次遇到过类似问题"。

### 5.4 扩展2：自我进化技能系统

**解决的问题**：每次遇到新问题都要重新推理，不会从经验中学习。

**技能生命周期**：

```text
创建阶段：
  后台审查 Agent 检测可复用模式
  → 调用 skill_manage("create", name="git-troubleshoot", content="...")
  → 存入 ~/.hermes/skills/git-troubleshoot.md

加载阶段：
  下次对话 → build_skills_system_prompt()
  → 扫描所有技能 frontmatter
  → 生成索引注入系统提示词

优化阶段：
  使用过程中 → Agent 发现技能不够完善
  → 调用 skill_manage("update", ...)
  → 更新技能内容
```

**Skill Nudge 机制**（`run_agent.py:12580-12586`）：

```python
# 每 N 轮工具调用后，如果没有使用 skill_manage
if _iters_since_skill >= _skill_nudge_interval:  # 默认 10
    _should_review_skills = True
    # 后台审查 Agent 会检查是否有值得提取为技能的模式
```

**与 ReAct 的区别**：这是**元学习层**。ReAct 在单次对话内优化，Hermes 在**跨对话**维度积累能力。"技能"本质是经过验证的 prompt/工作流模板——Agent 自己创建、自己使用、自己优化。

### 5.5 扩展3：层级委派与并行执行

**解决的问题**：单线程串行执行，面对复杂任务效率低。

**两层并行机制**：

```text
同层并行（工具级别）：
  _execute_tool_calls_concurrent()  (run_agent.py:8351)
  ├─ ThreadPoolExecutor 多线程并发执行
  ├─ 智能判断：
  │   ├─ read_file + web_search → 始终可并行
  │   ├─ write_file + write_file → 检查路径是否重叠
  │   └─ terminal + read_file → 始终可并行
  └─ 结果按原始顺序追加到 messages

跨层委派（Agent 级别）：
  delegate_task(goal, context, toolsets, tasks)
  → 创建新 AIAgent 子实例
  → 继承父 Agent 的：
      ├─ 迭代预算（共享，防止无限分叉）
      ├─ credential_pool
      ├─ memory_store / session_db
      └─ checkpoints / interrupt 信号
  → 子 Agent 独立运行完整 ReAct 循环
  → 结果汇总返回父 Agent
```

**关键代码**：
- 并行执行：`run_agent.py:8351-8450`
- 委派调用：`run_agent.py:8224-8241` — `_dispatch_delegate_task()`
- 智能并行判断：`_should_parallelize_tool_batch()`

**与 ReAct 的区别**：从**线性 ReAct** 变为**树形 ReAct**。父 Agent 产生子 Agent，每个子 Agent 内部又是完整循环。迭代预算在所有 Agent 间共享。

### 5.6 扩展4：人机协同中断机制

**解决的问题**：一旦开始执行就停不下来。

**三种忙时输入模式**（`cli.py:1853-1862`）：

```text
"interrupt" → Enter 键中断当前循环
  → _interrupt_requested = True
  → 检查点1: 每次循环迭代开始 (run_agent.py:9588)
  → 检查点2: 工具执行前 (run_agent.py:8361)
  → 清理: 跳过未执行工具，返回已执行结果
  → clear_interrupt() 恢复

"steer" → 输入 "/steer 别忘了检查端口安全性"
  → _pending_steer 存储
  → _drain_pending_steer() 在下个 API 调用前注入
  → 追加到最近一条 tool result 消息末尾
  → 模型在下一轮推理时看到用户指导
  → 如果中途没有工具调用 → 留在队列等待作为下一个 user turn

"queue" → Enter 排队下一条消息
  → 等价于普通的多轮对话流程
```

**Steer 注入路径**（`run_agent.py:9656-9692`）：

```text
1. pre-API-call drain:
   如果在 API 调用期间收到 steer → 在下一次 API 调用前注入
   → 向后扫描 messages 找最近的 tool 消息
   → 追加 "\n\nUser guidance: <steer内容>"

2. post-tool-execution drain:
   如果在工具执行期间收到 steer → 在执行后注入
   → 追加到刚产生的 tool result 消息

3. leftovers:
   如果整个循环结束还有未投递的 steer
   → 作为 "pending_steer" 返回给调用方
   → 作为下一条 user turn 发送
```

**与 ReAct 的区别**：ReAct 是**自动驾驶**，Hermes 是**带方向盘和刹车的自动驾驶**。

### 5.7 扩展5：多层容错与故障转移

**解决的问题**：API 调用失败 → 崩溃，工具调用失败 → 崩溃。

**五层恢复机制**：

```text
Layer 1: API 重试（智能退避）
  → jittered_backoff() 带抖动的指数退避
  → 最多 max_retries 次（默认 3）

Layer 2: 错误分类 → 定向恢复策略
  → classify_api_error() 映射到 FailoverReason:
  ┌──────────────────────┬──────────────────────────────┐
  │ FailoverReason       │ 恢复策略                      │
  ├──────────────────────┼──────────────────────────────┤
  │ context_too_long     │ 触发上下文压缩                │
  │ rate_limited         │ 切换到 fallback provider      │
  │ auth_error           │ 重试换 key / 刷新 OAuth token │
  │ empty_content        │ thinking prefill + 重试       │
  │ billing              │ 切换到 fallback               │
  └──────────────────────┴──────────────────────────────┘

Layer 3: Fallback 链（多级故障转移）
  → config.yaml 配置 fallback_models 列表
  → _try_activate_fallback() 线性遍历
  → 自动切换 provider + model + api_mode + base_url
  → 支持跨 provider 切换:
      例: Nous Portal → OpenRouter → Anthropic 直连

Layer 4: 响应修复
  → 损坏的 tool_call JSON → _sanitize_tool_call_arguments()
  → 缺失的 tool_call_id → 自动补全
  → 孤立 tool result（无对应 tool_call）→ 自动删除

Layer 5: 空响应恢复
  → 检测空响应（无 content 且无 tool_calls）
  → thinking prefill: 注入 "Continue."
  → 空响应 3 次 → 自动切换 fallback
  → 所有手段耗尽 → 优雅降级返回 "(empty)"
```

**关键代码**：
- 错误分类：`agent/error_classifier.py`
- Fallback 激活：`run_agent.py:6846-6945`
- 空响应处理：`run_agent.py:12240-12320`
- 响应修复：`run_agent.py:9700-9710`

**与 ReAct 的区别**：ReAct 假设每一步成功。Hermes 不假设任何事会成功——每个环节都有后备方案。这不是增强 ReAct 行为，而是让 ReAct 在真实生产环境中**能够存活**。

### 5.8 扩展6：上下文压缩

**解决的问题**：长对话超出上下文窗口 → 截断 → 丢失关键信息。

**双模式触发**：

```text
主动预检（Preflight）：
  run_agent.py:9435-9494
  → 进入主循环前 estimate_request_tokens_rough()
  → token 估算 >= threshold_tokens
  → 直接触发压缩，不等 API 报错
  → 最多 3 轮压缩迭代

被动触发（Reactive）：
  API 返回 context_too_long
  → classify_api_error() → FailoverReason.context_too_long
  → _compress_context()
```

**压缩策略**：

```text
保留窗口:
  ├─ protect_first_n（默认 4 条）—— 任务定义和初始上下文
  └─ protect_last_n（默认 6 条）—— 最近的对话状态

压缩中间:
  └─ 发送给模型 → "请总结以下对话的关键信息"
  └─ 返回摘要 → 替换被压缩的消息

会话分裂:
  └─ 创建新 child session（parent_session_id 链接）
  └─ 重建系统提示词（重新加载记忆和文件）
```

**与 ReAct 的区别**：ReAct 是无界的（理论上无限增长）。Hermes 实现了**有界 ReAct**——长对话自动折叠，信息密度随对话增长。

### 5.9 扩展7：插件生命周期钩子

**解决的问题**：循环逻辑完全固定，无法扩展。

**8 个钩子的完整生命周期**：

```text
on_session_start
  │
  ▼
┌─────────────────────────────────────┐
│  pre_llm_call                       │  ← 注入上下文（如 Langfuse trace）
│  (可返回 context 或 string)         │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  pre_api_request                    │  ← 观察请求参数
│  (仅观察，返回值忽略)               │
└──────────────┬──────────────────────┘
               ▼
         [LLM API 调用]
               │
               ▼
┌─────────────────────────────────────┐
│  pre_tool_call                      │  ← 可阻止执行
│  (返回 block_message 阻止工具)       │
└──────────────┬──────────────────────┘
               ▼
         [工具执行]
               │
               ▼
┌─────────────────────────────────────┐
│  post_tool_call                     │  ← 审计日志+耗时
│  (仅观察，返回值忽略)               │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  transform_tool_result              │  ← 可修改/替换结果
│  (返回 str 替换原结果)              │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  post_llm_call                      │  ← 观察 LLM 响应
│  (仅观察，返回值忽略)               │
└──────────────┬──────────────────────┘
               ▼
        [循环回到 pre_llm_call]
               │
               ▼
on_session_end  ← 清理、flush 缓冲区
```

**关键实现**：
- 钩子管理：`hermes_cli/plugins.py`
- 钩子调用点：分散在 `run_agent.py` 和 `model_tools.py` 的各处

**与 ReAct 的区别**：ReAct 是封闭循环。Hermes 把循环**打开**了——类似 Web 框架的中间件模式，允许在任何阶段插入自定义逻辑。

### 5.10 扩展8：Prompt 缓存优化

**解决的问题**：每轮 API 调用系统提示词相同，但按全价收费。

**设计原则**：

```text
固定部分（享受缓存，放在系统提示词）:
  ├─ Agent 身份
  ├─ 工具使用指导
  ├─ 记忆块（冻结快照）
  ├─ 技能索引
  ├─ 上下文文件
  └─ 平台提示

动态部分（注入用户消息，不破坏缓存）:
  ├─ 外部记忆召回（prefetch_all）
  └─ Plugin 上下文（pre_llm_call 返回）

为什么？因为 Anthropic prompt cache 按前缀匹配。
系统提示词不变 → 前缀不变 → 缓存命中。
```

**API 层实现**（`run_agent.py:9785-9790`）：

```python
# 为 system + 最后 3 条消息添加 cache_control breakpoint
if self._use_prompt_caching:
    api_messages = apply_anthropic_cache_control(
        api_messages,
        cache_ttl=self._cache_ttl,
        native_anthropic=self._use_native_cache_layout,
    )
```

**KV 缓存标准化**（`run_agent.py:9798-9829`）：

```python
# JSON key 排序 + 消息去空白 → bit-perfect 前缀匹配
for tc in tool_calls:
    args_obj = json.loads(tc["function"]["arguments"])
    tc["function"]["arguments"] = json.dumps(
        args_obj, separators=(",", ":"), sort_keys=True
    )
```

**与 ReAct 的区别**：不影响 Agent 行为，但显著降低运行成本（约 75% 输入 token 节省）。

### 5.11 扩展9：后台审查代理

**解决的问题**：对话结束就是结束，没有后续加工。

**实现**（`run_agent.py:3220-3328`）：

```text
用户收到回复
  │
  └─ daemon 线程: _spawn_background_review()
      │
      ├─ 选择审查 prompt:
      │   ├─ _should_review_memory → _MEMORY_REVIEW_PROMPT
      │   ├─ _should_review_skills → _SKILL_REVIEW_PROMPT
      │   └─ 两者都需要 → _COMBINED_REVIEW_PROMPT
      │
      ├─ fork 子 AIAgent:
      │   ├─ 相同 model / provider / credentials
      │   ├─ max_iterations=8（审查不需要太多轮）
      │   ├─ quiet_mode=True（不显示给用户）
      │   ├─ 共享 memory_store / skill_store
      │   └─ 禁用 nudge（防止无限递归）
      │
      ├─ run_conversation(审查_prompt, 对话历史)
      │   └─ 子 Agent 调用 memory/skill_manage 工具
      │
      └─ 结果汇总：
          └─ _summarize_background_review_actions()
          └─ 输出: "💾 已保存 1 条记忆" 或 "💾 已创建技能 git-workflow"
```

**与 ReAct 的区别**：这是**元认知反射层**。ReAct 的每一步都是工具调用的直接观察。Hermes 在对话结束后运行了**元认知 Agent**——不直接服务用户，而是审视整个对话过程，提取可复用的模式和知识。

### 5.12 扩展10：文件系统检查点

**解决的问题**：Agent 写了错误文件 → 无法撤销。

**实现**（`run_agent.py:8390-8410`）：

```text
写操作前自动快照:
  write_file / patch:
    → _checkpoint_mgr.ensure_checkpoint(work_dir, "before write_file")
    
  破坏性终端命令（rm / mv / dd / mkfs / ...）:
    → _checkpoint_mgr.ensure_checkpoint(work_dir, "before terminal: rm -rf ...")

快照实现:
  → 目标目录 git init（如果尚未）
  → git add -A
  → git commit --allow-empty -m "checkpoint: before write_file"
  → 限制快照数量（max_snapshots 配置项）

恢复:
  → git checkout <snapshot_commit>
```

**与 ReAct 的区别**：给 Agent 的每个破坏性操作加上安全网，让用户可以安全地让 Agent 自由探索。

### 5.13 对比总结

| 维度 | 经典 ReAct | Hermes Agent |
|------|-----------|-------------|
| **循环结构** | 线性 Reason→Act→Observe | 树形（父 Agent + 子 Agent 层级委派） |
| **执行模式** | 完全串行 | 串行 + 并行（独立工具并发执行） |
| **记忆** | 无，仅当前对话历史 | 三层：系统提示词快照 + 每轮动态召回 + 后台自动存储 |
| **学习** | 无 | 跨对话技能积累，Agent 自己创建/优化技能 |
| **人机协同** | 无 | Interrupt（终止）+ Steer（中途修正） |
| **容错** | 崩溃即结束 | 5 层恢复：重试→分类恢复→fallback 链→响应修复→降级 |
| **上下文管理** | 单次对话，超限截断 | 主动预检压缩 + 被动触发压缩 + 会话分裂 |
| **扩展性** | 完全封闭 | 8 个生命周期钩子（类中间件模式） |
| **成本优化** | 每轮全量 token 计费 | Prompt 缓存（~75% 输入成本节省）+ KV 标准化 |
| **后处理** | 无 | 后台审查 Agent（记忆提取 + 技能创建） |
| **安全回滚** | 无 | 文件系统检查点（操作前自动 git 快照） |

**核心设计哲学**：

Hermes 不是简单的 "ReAct + 更多工具"，而是在 ReAct 循环的**每个环节**都添加了增强层：

- **循环前**：记忆注入、技能匹配、上下文注入
- **循环中**：并行委派、中断协同、多层容错、检查点保护、prompt 缓存
- **循环后**：元认知反思、自动记忆存储、技能创建

把"单次问答 Agent"变成了"持续学习、自我进化的 Agent 系统"。

---

## 六、数据流总结

### 6.1 完整数据流

```text
用户输入（CLI / Telegram / Discord / ...）
  │
  ▼
HermesCLI.chat() / BasePlatformAdapter.on_message()
  │
  ▼
AIAgent.run_conversation(user_message, conversation_history)
  │
  ├─ 预处理
  │   ├─ 清理输入（代理字符 / 泄露标签）
  │   └─ 恢复 TODO 状态（从历史消息反序列化）
  │
  ├─ 系统提示词（缓存复用）
  │   ├─ SOUL.md → 人格身份
  │   ├─ MemoryStore → 持久记忆块
  │   ├─ Skills Index → 可用技能列表
  │   ├─ AGENTS.md → 项目级指令
  │   ├─ Platform Hints → 格式化指导
  │   └─ Timestamp + Session ID
  │
  ├─ 用户消息增强（API-call-time only）
  │   ├─ MemoryManager.prefetch_all() → 外部记忆召回
  │   └─ Plugin pre_llm_call → 额外上下文
  │
  ▼
┌─────────────────────────────────────────┐
│          LLM API 调用                    │
│  ├─ 适配器路由                           │
│  ├─ Prompt Cache 注入                    │
│  ├─ 错误分类 + 重试 + fallback           │
│  └─ 流式响应 → callback → 显示/推送      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
            响应解析
                   │
        ┌──────────┴──────────┐
        │                     │
   有 tool_calls          无 tool_calls
        │                     │
        ▼                     ▼
  工具执行循环            最终响应
  ├─ 并行/串行判断        ├─ 后处理
  ├─ 检查点快照            │   ├─ 后台审查 Agent
  ├─ Plugin hooks          │   │   ├─ 提取记忆
  │   ├─ pre_tool_call     │   │   └─ 创建技能
  │   ├─ post_tool_call    │   └─ 会话持久化
  │   └─ transform_result  │       └─ SQLite + FTS5
  ├─ 追加 tool_result      │
  └─ 继续循环 ─────────────┘
        │
        ▼
  返回给调用方
  ├─ CLI: 格式化输出
  └─ Gateway: 推送到聊天平台
```

### 6.2 会话持久化数据流

```text
run_conversation() 执行中
  │
  ├─ 每轮工具调用后 → _save_session_log() → JSONL 文件
  │
  └─ 对话结束后 → _persist_session() → SQLite
      ├─ sessions 表: UPSERT 会话元数据
      └─ messages 表: INSERT 全部消息
```

---

## 七、二次开发指南

### 7.1 关键切入文件

| 开发目标 | 切入点 | 说明 |
|----------|--------|------|
| 添加新工具 | `tools/` 目录 | 创建 `.py` 文件，文件末尾调用 `registry.register()` |
| 修改 Agent 行为 | `agent/prompt_builder.py` | 修改系统提示词组装逻辑 |
| 修改对话循环 | `run_agent.py:9583` | `while` 循环核心逻辑 |
| 添加新模型适配器 | `agent/` 目录 | 参考 `anthropic_adapter.py` 实现 |
| 添加新平台适配器 | `gateway/platforms/` | 继承 `BasePlatformAdapter` |
| 自定义记忆逻辑 | `plugins/memory/` | 实现 MemoryProvider 接口 |
| 自定义配置项 | `hermes_cli/config.py` | 扩展 config.yaml schema |
| 添加 CLI 子命令 | `hermes_cli/main.py` | 扩展 argparse subparser |
| 定时任务 | `cron/scheduler.py` | 修改调度逻辑或添加任务类型 |

### 7.2 配置文件结构

```text
~/.hermes/
  ├─ config.yaml          # 主配置（模型、工具集、显示、记忆等）
  ├─ gateway.yaml          # 网关配置（各平台 token、通道映射）
  ├─ .env                  # 环境变量（API keys 等）
  ├─ state.db              # SQLite 会话数据库
  ├─ memory/               # 内置记忆存储
  │   ├─ MEMORY.md         # 关键记忆条目
  │   └─ USER.md           # 用户画像
  ├─ skills/               # 用户技能存储
  ├─ cron/                 # 定时任务配置
  │   └─ cron_jobs.json
  └─ sessions/             # 会话日志（JSONL）
```

### 7.3 常见开发场景

**场景1：添加自定义工具**

```python
# tools/my_custom_tool.py
from tools.registry import registry

MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_custom_tool",
        "description": "我的自定义工具",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "参数1"}
            },
            "required": ["param1"]
        }
    }
}

def _handle_my_custom_tool(param1: str) -> str:
    return json.dumps({"result": f"处理了: {param1}"})

registry.register(
    name="my_custom_tool",
    toolset="custom",
    schema=MY_TOOL_SCHEMA,
    handler=_handle_my_custom_tool,
    emoji="🔧",
)
```

**场景2：修改系统提示词**

在 `agent/prompt_builder.py` 中修改或添加对应的 prompt 常量：

```python
# 例如修改 DEFAULT_AGENT_IDENTITY 或添加新的 PLATFORM_HINTS
PLATFORM_HINTS["my_platform"] = "自定义平台格式化提示..."
```

**场景3：添加 fallback provider**

在 `config.yaml` 中配置：

```yaml
agent:
  fallback_models:
    - provider: openrouter
      model: anthropic/claude-sonnet-4.6
    - provider: anthropic
      model: claude-sonnet-4-20250514
      base_url: https://api.anthropic.com
```

**场景4：开发插件**

在 `plugins/` 目录下创建包，实现对应的钩子函数：

```python
# plugins/my_plugin/__init__.py
def register_plugin():
    return {
        "hooks": {
            "pre_tool_call": my_pre_tool_hook,
            "post_tool_call": my_post_tool_hook,
        }
    }

def my_pre_tool_hook(tool_name, args, **kwargs):
    if tool_name == "terminal":
        # 检查命令是否在白名单中
        pass
```

### 7.4 注意事项

1. **不要修改系统提示词的缓存部分**：动态内容应注入到用户消息而非系统提示词，否则 prompt cache 失效
2. **工具注册的线程安全**：`ToolRegistry` 使用 `RLock`，读写操作自动加锁
3. **异步工具处理**：使用 `registry.dispatch()` 自动处理同步/异步桥接
4. **子 Agent 的迭代预算**：所有 Agent（父 + 子）共享 `IterationBudget`，子 Agent 也要消耗预算
5. **会话数据库的 WAL 模式**：允许并发读，但写操作需要排队
