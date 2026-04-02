# 文件索引

## A.1 核心文件映射表

### 入口层

| 文件 | 入口命令 | 功能 |
|------|---------|------|
| `run_agent.py` | `hermes-agent` | AIAgent 核心类 |
| `cli.py` | - | CLI 编排器 |
| `hermes_cli/main.py` | `hermes` | CLI 主入口 |
| `gateway/run.py` | `hermes gateway` | 网关主循环 |
| `acp_adapter/entry.py` | `hermes-acp` | ACP 服务器 |
| `cron/scheduler.py` | `hermes cron` | 定时任务调度 |

### 核心模块

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `run_agent.py` | Agent 对话循环 | `AIAgent.run_conversation()` |
| `model_tools.py` | 工具编排 | `handle_function_call()` |
| `toolsets.py` | 工具集定义 | `_HERMES_CORE_TOOLS` |
| `hermes_state.py` | SessionDB | `SessionDB` |

### 工具系统

| 文件 | 工具数 | 关键工具 |
|------|--------|---------|
| `tools/registry.py` | 核心 | `ToolRegistry` |
| `tools/file_tools.py` | 6+ | `Read`, `Write`, `Edit` |
| `tools/terminal_tool.py` | 5+ | `Bash`, `InteractiveBash` |
| `tools/web_tools.py` | 3+ | `WebSearch`, `WebExtract` |
| `tools/browser_tool.py` | 8+ | `BrowserNavigate`, `BrowserClick` |
| `tools/mcp_tool.py` | 动态 | `MCPClient` |
| `tools/delegate_tool.py` | 2+ | `delegate_task_handler` |

### CLI 系统

| 文件 | 行数(估) | 功能 |
|------|---------|------|
| `hermes_cli/main.py` | ~200k | CLI 框架 |
| `hermes_cli/setup.py` | ~107k | 设置向导 |
| `hermes_cli/config.py` | - | 配置管理 |
| `hermes_cli/auth.py` | ~88k | 认证 |
| `hermes_cli/skin_engine.py` | ~35k | 皮肤系统 |
| `hermes_cli/models.py` | ~43k | 模型目录 |

### Gateway 系统

| 文件 | 行数(估) | 功能 |
|------|---------|------|
| `gateway/run.py` | ~287k | 网关主循环 |
| `gateway/session.py` | ~40k | 会话管理 |
| `gateway/base.py` | ~60k | Platform 基类 |
| `gateway/platforms/telegram.py` | ~92k | Telegram |
| `gateway/platforms/discord.py` | ~98k | Discord |
| `gateway/platforms/slack.py` | ~37k | Slack |
| `gateway/platforms/feishu.py` | ~136k | 飞书 |

### Agent 核心

| 文件 | 功能 |
|------|------|
| `agent/anthropic_adapter.py` | Anthropic API 适配 |
| `agent/context_compressor.py` | 上下文压缩 |
| `agent/prompt_builder.py` | Prompt 构建 |
| `agent/prompt_caching.py` | Prompt 缓存 |
| `agent/model_metadata.py` | 模型元数据 |
| `agent/usage_pricing.py` | 使用量计价 |

### 记忆系统

| 文件 | 功能 |
|------|------|
| `hermes_state.py` | SQLite + FTS5 存储 |
| `tools/memory_tool.py` | 记忆工具 |
| `honcho_integration/` | Honcho 集成 |

---

## A.2 按功能查找

### 工具注册
- `tools/registry.py` - 中心化注册表
- `model_tools.py` - 工具发现

### 消息处理
- `gateway/run.py` - 网关主循环
- `gateway/session.py` - 会话管理
- `gateway/base.py` - Platform 基类

### 配置管理
- `hermes_cli/config.py` - 默认配置
- `hermes_cli/auth.py` - 认证

### 模型适配
- `agent/anthropic_adapter.py` - Anthropic
- `agent/auxiliary_client.py` - 辅助客户端

---

## A.3 文件大小排名

```
~550k  agent/                    (Agent 核心)
~287k  gateway/run.py            (网关主程序)
~200k  hermes_cli/main.py        (CLI 主程序)
 ~98k  gateway/platforms/discord.py
 ~92k  gateway/platforms/telegram.py
 ~77k  tools/mcp_tool.py
 ~77k  tools/browser_tool.py
 ~76k  tools/web_tools.py
 ~98k  tools/skills_hub.py
 ~57k  tools/terminal_tool.py
 ~57k  tools/rl_training_tool.py
 ~48k  tools/skills_tool.py
 ~44k  tools/file_tools.py
```

---

[← 返回总目录](../SUMMARY.md) | [← RL 环境](../08-advanced/03-rl.md) | [配置参考 →](02-config-ref.md)
