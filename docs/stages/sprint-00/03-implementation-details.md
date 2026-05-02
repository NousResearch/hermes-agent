# Sprint 00 — 关键实现细节

## 1. 代码规模

| 文件 | 行数 | 说明 |
|------|------|------|
| `run_agent.py` | 13,737 | 核心编排器 AIAgent，包含对话循环、工具分发、模型适配、会话管理 |
| `cli.py` | 11,483 | 交互式 TUI，prompt_toolkit 应用、消息渲染、/ 命令处理 |
| `tools/memory_tool.py` | 586 | 文件记忆读写（MEMORY.md / USER.md） |
| `tools/session_search_tool.py` | 591 | FTS5 搜索 + Gemini Flash 摘要 |
| `agent/` 目录 | 42 文件 | 模型适配器、Memory Manager、Skill Utils、Context Engine 等 |
| `tools/` 目录 | 62 文件 | 60+ 注册工具 |
| `gateway/` 目录 | 48 文件 | 多平台网关 |

## 2. AIAgent 核心初始化参数（run_agent.py:876-1008）

```python
class AIAgent:
    def __init__(
        self,
        base_url,           # API 端点
        model,              # 模型名称
        api_key,            # API 密钥
        session_id,         # 预生成的会话 ID
        parent_session_id,  # 父会话（压缩分裂时）
        pass_session_id,    # 是否透传 session_id 给子 Agent
        toolsets,           # 工具集名称列表
        skills,             # 启用的 skill 列表
        system_prompt,      # 自定义系统提示
        max_turns,          # 最大对话轮次
        ...
    )
```

关键属性：
- `self.session_id` → 格式 `{timestamp}_{short_uuid}`
- `self.session_log_file` → `logs/session_{session_id}.json`
- `self._session_db` → HermesState 实例
- `self._todo_store` → TodoStore 实例
- `self._memory_manager` → MemoryManager 实例

## 3. 工具注册机制（tools/registry.py）

```python
# 工具模块自注册模式：
from tools.registry import register

register(
    name="tool_name",
    schema={...},           # JSON Schema
    handler=tool_function,  # 处理函数
    toolsets=[...],         # 所属工具集
    available=lambda: True, # 可用性检查
)
```

每个 `tools/*.py` 在模块级别调用 `register()`。`model_tools.py` 在加载时导入所有工具模块触发注册。

## 4. Memory 系统实现细节

### 存储格式（tools/memory_tool.py）

使用 `§` 作为条目分隔符，条目可多行。支持 YAML frontmatter（已有基础但未充分使用 type/scope 字段）。

```markdown
§
类型: user_preference
范围: global
---
Gu 偏好直接、简洁的回答风格，不喜欢啰嗦的解释。
§
类型: project_context
---
Hermes 是 Gu 的智能秘书型多 Agent 系统。
```

### 操作类型

- `add`：追加新条目
- `replace`：子字符串匹配替换
- `remove`：子字符串匹配删除
- `read`：读取全文

### 安全扫描

写入前检查注入模式（prompt_injection、exfil_curl、ssh_backdoor 等），告警但不阻断。

## 5. Skill 系统实现细节

### SKILL.md 格式（agent/skill_utils.py）

```yaml
---
name: skill-name
description: Brief description
version: 1.0.0
platforms: [macos]
permissions:
  tools: [bash, file_read, file_write]
  network: false
triggers: [触发词1, 触发词2]
---
# Skill 正文
```

### 当前 manifest 解析能力

- `parse_frontmatter()`：YAML frontmatter → dict + body
- `PLATFORM_MAP`：{macos: darwin, linux: linux, windows: win32}
- 平台过滤：`platforms` 字段控制加载
- **已有 `permissions` 和 `triggers` 字段解析能力**，但当前不做权限校验（Sprint 6 需要增强）

## 6. 会话存储实现细节

### state.db schema（hermes_state.py）

```sql
sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,    -- 'cli', 'telegram', 'discord', ...
    model TEXT,
    model_config TEXT,       -- JSON
    system_prompt TEXT,
    parent_session_id TEXT,  -- 压缩分裂链
    started_at REAL,
    ended_at REAL,
    message_count INTEGER,
    tool_call_count INTEGER,
    input_tokens / output_tokens / cache_read_tokens / ...
)

messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT → sessions(id),
    role TEXT,               -- 'user', 'assistant', 'tool'
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,         -- JSON
    tool_name TEXT,
    timestamp REAL,
    reasoning TEXT,
    ...
)
```

FTS5 虚拟表：`messages_fts(messages(rowid), content)` 用于全文搜索。

## 7. 会话 JSONL 格式

每行一个 JSON 对象，包含 `role`、`content`、`timestamp` 等字段。存储于 `~/.hermes/sessions/{timestamp}_{uuid}.jsonl`。

## 8. Delegate 机制细节

### 子 Agent 约束

- 隔离上下文（不继承父 history）
- 受限工具集（blocked: delegate_task、clarify、memory、send_message、execute_code）
- 独立 session_id
- 父 Agent 阻塞等待，只收到摘要结果

### 关键参数

- `max_concurrent_children: 3`
- `max_spawn_depth: 1`（不允许递归委托）
- `child_timeout_seconds: 600`
- `subagent_auto_approve: false`（默认拒绝危险命令）

## 9. MCP 配置详情

```yaml
mcp_servers:
  minimax:
    enabled: true
    command: uvx
    args: [minimax-coding-plan-mcp, -y]
    env:
      MINIMAX_API_KEY: sk-...
      MINIMAX_API_HOST: https://api.minimaxi.com
  openchronicle:
    timeout: 120
    url: http://127.0.0.1:8742/mcp
```

当前只有 2 个 MCP server。未接入 firecrawl MCP 和 context MCP（任务书要求）。

## 10. Cron 调度细节

```python
# cron/scheduler.py - APScheduler 风格
# cron/jobs.py - 任务定义
```

支持 `cron.max_parallel_jobs`、`cron.wrap_response` 配置。

## 11. 上下文压缩

- `agent/context_compressor.py`：上下文引擎
- `trajectory_compressor.py`：轨迹压缩器
- 阈值：`compression.threshold: 0.5`（50% 触发）
- 保护最近 N 条消息：`compression.protect_last_n: 20`
- 硬限制：`compression.hygiene_hard_message_limit: 400`
