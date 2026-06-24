# Hermes Agent 架构分析

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Hermes Agent 架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   CLI/TUI   │    │   Gateway   │    │   Cronjob   │    │    API      │  │
│  │  (入口层)   │    │  (消息层)   │    │  (任务层)   │    │   Server    │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        AIAgent (run_agent.py)                         │  │
│  │                          ─── 核心引擎 ───                             │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    会话循环 (run_conversation)                    │  │  │
│  │  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │  │  │
│  │  │  │ 用户输入  │───▶│ 构建Prompt│───▶│ 调用LLM  │───▶│ 解析响应  │  │  │  │
│  │  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │  │  │
│  │  │         │                    │                    │            │  │  │
│  │  │         │                    ▼                    │            │  │  │
│  │  │         │            ┌───────────────┐            │            │  │  │
│  │  │         │            │ 工具调用执行   │◀───────────┘            │  │  │
│  │  │         │            │ (handle_call) │                          │  │  │
│  │  │         │            └───────┬───────┘                          │  │  │
│  │  │         │                    │                                  │  │  │
│  │  │         └────────────────────┴───────────────────────────────────┘  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         ▼                          ▼                          ▼            │
│  ┌───────────────┐      ┌───────────────┐      ┌───────────────┐          │
│  │   Memory      │      │    Skills     │      │    Prompt     │          │
│  │   Manager     │      │     Hub       │      │    Builder    │          │
│  │   (记忆层)    │      │   (技能层)    │      │   (提示层)    │          │
│  └───────┬───────┘      └───────┬───────┘      └───────┬───────┘          │
│          │                      │                      │                   │
│          ▼                      ▼                      ▼                   │
│  ┌───────────────┐      ┌───────────────┐      ┌───────────────┐          │
│  │  Providers    │      │  Curator      │      │  Compressor   │          │
│  │ (Honcho/Mem0) │      │ (技能管理)    │      │ (上下文压缩)   │          │
│  └───────────────┘      └───────────────┘      └───────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Agent 核心工作机制

### 2.1 AIAgent 类

核心入口是 [run_agent.py](file:///D:/hermes-agent/run_agent.py#L1028) 中的 `AIAgent` 类，约 **12,000 行**代码。

**关键组件：**

| 组件 | 职责 |
|------|------|
| `run_conversation()` | 核心会话循环（约第 11410 行） |
| `_build_system_prompt()` | 构建系统提示词 |
| `_create_openai_client()` | 创建 LLM 客户端 |
| `_execute_tool_calls()` | 执行工具调用 |
| `IterationBudget` | 迭代预算管理（默认 90 次） |

### 2.2 会话循环流程

```python
while (api_call_count < max_iterations and budget.remaining > 0):
    if interrupt_requested: break
    
    # 1. 构建系统提示词
    system_prompt = self._build_system_prompt()
    
    # 2. 调用 LLM
    response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        tools=tool_schemas
    )
    
    # 3. 处理工具调用
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args)
            messages.append(tool_result_message(result))
    else:
        return response.content  # 直接返回
```

### 2.3 工具调用机制

工具调用通过 `model_tools.py` 管理：

- **工具注册**：`registry.register()` 在 `tools/*.py` 中定义
- **工具发现**：自动发现 `tools/*.py` 中的工具
- **工具集**：在 `toolsets.py` 中分组管理（17个核心工具集）

### 2.4 错误处理与恢复

- **重试机制**：`agent/retry_utils.py` 的抖动退避
- **错误分类**：`agent/error_classifier.py` 识别可重试/不可重试错误
- **降级策略**：自动尝试备用模型

---

## 三、记忆系统

### 3.1 架构设计

记忆系统采用**插件化设计**，通过 `MemoryManager` 统一管理：

```
MemoryManager (agent/memory_manager.py)
    │
    ├── MemoryProvider (ABC) — 抽象基类
    │       │
    │       ├── is_available()      — 检查可用性
    │       ├── initialize()        — 初始化
    │       ├── system_prompt_block() — 系统提示词片段
    │       ├── prefetch(query)     — 预取记忆
    │       ├── sync_turn()         — 同步对话
    │       └── shutdown()          — 关闭
    │
    └── 外部 Providers (plugins/memory/)
            ├── honcho
            ├── mem0
            ├── supermemory
            ├── byterover
            ├── hindsight
            ├── holographic
            ├── openviking
            └── retaindb
```

### 3.2 工作流程

**每轮对话的记忆交互：**

1. **Prefetch（预取）**：在用户消息到来前，根据查询词召回相关记忆
   ```python
   context = self._memory_manager.prefetch_all(user_message)
   ```

2. **System Prompt 注入**：记忆内容作为上下文注入系统提示词
   ```python
   prompt_parts.append(self._memory_manager.build_system_prompt())
   ```

3. **Sync（同步）**：对话结束后，异步写入本轮对话内容
   ```python
   self._memory_manager.sync_all(user_msg, assistant_response)
   ```

### 3.3 关键特性

- **单提供者限制**：同一时刻只允许一个外部记忆提供者运行，防止工具 schema 膨胀
- **上下文清理**：`StreamingContextScrubber` 清理流式响应中的记忆标签
- **会话隔离**：每个会话有独立的记忆上下文

---

## 四、技能系统

### 4.1 技能架构

```
skills/                    # 内置技能（默认启用）
    ├── github/
    ├── mlops/
    └── ...

optional-skills/           # 可选技能（需手动安装）
    ├── autonomous-ai-agents/
    ├── blockchain/
    └── ...

plugins/
    └── skills/            # 插件技能

~/.hermes/skills/          # 用户技能目录
```

### 4.2 技能管理

**技能加载流程**：

1. **发现**：扫描 `skills/` 和 `~/.hermes/skills/` 目录
2. **解析**：读取 `SKILL.md` frontmatter（name, description, tags, category）
3. **注入**：将技能描述和索引注入系统提示词

**技能元数据示例**：
```yaml
name: Web Search
description: Search the web for information
version: 1.0.0
author: Hermes Team
metadata:
  hermes:
    tags: ["search", "information"]
    category: "Tools & Skills"
```

### 4.3 Curator（技能策展人）

[agent/curator.py](file:///D:/hermes-agent/agent/curator.py) 负责后台技能维护：

**职责：**
- **生命周期管理**：根据活跃度自动转换状态（active → stale → archived）
- **智能审查**：定期启动辅助 LLM 审查和优化技能
- **归档保护**：从不删除，只归档（可恢复）

**配置参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `interval_hours` | 168（7天） | 审查间隔 |
| `min_idle_hours` | 2 | 最小空闲时间 |
| `stale_after_days` | 30 | 多少天未使用标记为 stale |
| `archive_after_days` | 90 | 多少天未使用自动归档 |

### 4.4 Skills Hub

[tools/skills_hub.py](file:///D:/hermes-agent/tools/skills_hub.py) 提供技能安装源：

- **GitHubSource**：从 GitHub 仓库获取技能
- **OptionalSkillSource**：官方可选技能
- **安全扫描**：安装前扫描恶意代码（[tools/skills_guard.py](file:///D:/hermes-agent/tools/skills_guard.py)）

---

## 五、Prompt 系统

### 5.1 系统提示词构建

[agent/prompt_builder.py](file:///D:/hermes-agent/agent/prompt_builder.py) 负责组装系统提示词：

**构建组件：**

| 组件 | 内容 |
|------|------|
| `DEFAULT_AGENT_IDENTITY` | 代理身份定义 |
| `PLATFORM_HINTS` | 平台特定提示（CLI/Telegram/Discord） |
| `MEMORY_GUIDANCE` | 记忆使用指南 |
| `SKILLS_GUIDANCE` | 技能使用指南 |
| `TOOL_USE_ENFORCEMENT_GUIDANCE` | 工具调用强制指南 |
| Context Files | `AGENTS.md`, `SOUL.md`, `.hermes.md` |

### 5.2 上下文安全扫描

**Prompt 注入防护**：

```python
_CONTEXT_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    # ... 更多模式
]
```

扫描隐形 Unicode 字符和威胁模式，阻止危险内容注入。

### 5.3 上下文压缩

[agent/context_compressor.py](file:///D:/hermes-agent/agent/context_compressor.py) 自动压缩长对话：

**工作原理：**
1. **检测溢出**：当上下文超过模型窗口限制时触发
2. **保护头尾**：保留最近的消息和重要的早期消息
3. **LLM 摘要**：使用辅助模型（廉价快速）总结中间轮次
4. **迭代更新**：保留摘要信息，支持多次压缩

**压缩策略：**
```
┌─────────────────────────────────────────────────────────────────┐
│  [头部保护] — 早期关键消息                              │
│                                                         │
│  [CONTEXT COMPACTION] — LLM 生成的摘要                   │
│  - Resolved Questions: ...                             │
│  - Pending Questions: ...                              │
│  - Remaining Work: ...                                 │
│                                                         │
│  [尾部保护] — 最近的对话（按 token 预算保护）            │
└─────────────────────────────────────────────────────────┘
```

---

## 六、关键设计特点

### 6.1 模块化设计

- **松耦合**：各组件通过接口通信，易于替换
- **插件化**：记忆提供者、模型提供者、技能均支持插件
- **关注点分离**：核心引擎、记忆、技能、提示各司其职

### 6.2 性能优化

- **延迟加载**：`OpenAI` SDK 延迟导入（约 240ms 节省）
- **Prompt 缓存**：避免重复构建系统提示词
- **上下文压缩**：智能管理 token 消耗

### 6.3 安全性

- **Prompt 注入防护**：扫描威胁模式和隐形字符
- **技能安全扫描**：安装前检查恶意代码
- **工具守卫**：[agent/tool_guardrails.py](file:///D:/hermes-agent/agent/tool_guardrails.py) 控制工具调用权限
- **敏感信息清理**：[agent/redact.py](file:///D:/hermes-agent/agent/redact.py) 自动脱敏

### 6.4 高可用设计

- **故障转移**：自动尝试备用模型
- **重试机制**：抖动退避策略
- **安全 I/O**：`_SafeWriter` 防止管道断开导致崩溃
- **中断支持**：随时可中断长时间运行的任务

---

## 七、数据流向总结

```
用户输入
    │
    ▼
┌─────────────────────┐
│  AIAgent 会话循环   │
│  run_conversation() │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌──────────┐
│ Memory   │ │ Prompt   │
│ prefetch │ │ Builder  │
└────┬─────┘ └────┬─────┘
     │            │
     └──────┬─────┘
            ▼
┌─────────────────────┐
│ 构建完整 System Prompt │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    调用 LLM         │
│ (OpenAI/Antropic/...)│
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
 工具调用?    直接回复?
     │           │
     ▼           ▼
┌──────────┐ ┌──────────┐
│ Tool     │ │ 返回给   │
│ Execution│ │ 用户     │
└────┬─────┘ └──────────┘
     │
     ▼
┌─────────────────────┐
│  Memory sync_turn() │
│ 写入本轮对话        │
└─────────────────────┘
```

---

## 八、核心文件索引

| 模块 | 文件路径 | 说明 |
|------|----------|------|
| Agent 核心 | `run_agent.py` | AIAgent 类，会话循环 |
| 工具管理 | `model_tools.py` | 工具注册、执行、发现 |
| 工具集定义 | `toolsets.py` | 工具集分组配置 |
| 记忆管理 | `agent/memory_manager.py` | 记忆管理器 |
| 记忆提供者 | `agent/memory_provider.py` | MemoryProvider ABC |
| 提示构建 | `agent/prompt_builder.py` | 系统提示词组装 |
| 上下文压缩 | `agent/context_compressor.py` | 对话压缩 |
| 技能管理 | `tools/skills_hub.py` | Skills Hub 管理 |
| 技能策展 | `agent/curator.py` | 后台技能维护 |
| 模型提供者 | `plugins/model-providers/` | 各模型提供商插件 |
| 配置管理 | `hermes_cli/config.py` | 配置加载 |
| 认证管理 | `hermes_cli/auth.py` | API Key 解析 |
| 运行时提供者 | `hermes_cli/runtime_provider.py` | 运行时 Provider 解析