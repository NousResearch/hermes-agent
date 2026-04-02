# 术语表

## C.1 核心概念

### AIAgent
Hermes 的核心 Agent 类，负责对话循环和工具调用。

**文件**: `run_agent.py`

### Tool / Toolset
- **Tool**: 单个工具，如 `Read`、`Write`、`Bash`
- **Toolset**: 工具集合，如 `files`、`terminal`

**文件**: `tools/registry.py`

### SessionDB
SQLite + FTS5 实现的会话存储，支持全文搜索。

**文件**: `hermes_state.py`

### Skill
可打包的功能单元，包含元数据、指令和工具关联。

**文件**: `tools/skills_tool.py`

### Platform
消息平台的适配器，如 Telegram、Discord。

**文件**: `gateway/base.py`

---

## C.2 配置相关

### HERMES_HOME
配置目录，默认为 `~/.hermes`，可通过环境变量覆盖。

### Toolset
工具的逻辑分组，用于启用/禁用一组相关工具。

### Skin
数据驱动的 UI 主题，使用 YAML 定义。

---

## C.3 消息平台

### ACP
Agent Communication Protocol，用于编辑器集成的通信协议。

### Gateway
消息网关，连接 Hermes Agent 和多种消息平台。

---

## C.4 RL 相关

### Atropos
Hermes 的强化学习环境，用于训练 Agent 决策能力。

### Trajectory
对话轨迹，指一次完整对话中消息和工具调用的序列。

---

## C.5 缩写

| 缩写 | 全称 | 说明 |
|------|------|------|
| FTS5 | Full-Text Search 5 | SQLite 全文搜索 |
| MCP | Model Context Protocol | 模型上下文协议 |
| ACP | Agent Communication Protocol | Agent 通信协议 |
| API | Application Programming Interface | 应用程序接口 |
| CLI | Command Line Interface | 命令行界面 |

---

[← 返回总目录](../SUMMARY.md) | [← 配置参考](02-config-ref.md)
