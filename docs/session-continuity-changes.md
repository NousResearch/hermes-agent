# Session 持久化跨会话记忆系统

## 目标
实现 session 之间的持久化记忆，让新 session 启动时能读取上一个 session 的对话记录。

## 实现方案

### 1. Gateway 内部保存（agent 级别）

**修改文件：** `gateway/run.py`, `gateway/slash_commands.py`

**新增方法：** `_auto_save_session(self, reason="unknown")`
- 从 `self.db._conn` 读取当前 session 的最近 100 条 user/assistant 消息
- 保存到 `~/.hermes/last_session_context/last_session.json`

**四个触发点：**
1. `session_expiry_watcher` — session 超时（不活跃自动关闭）
2. `_finalize_shutdown_agents` — gateway 重启且有活跃 agent
3. `idle_agent_cache_cleanup` — gateway 重启且 agent 空闲
4. `_handle_reset_command` — `/new` 和 `/reset` 命令

### 2. CLI 层面保存（进程级别）

**修改文件：** `hermes_cli/gateway.py`（源码仓库）

**保存逻辑：** `_pre_save_session_from_db()` 函数
- 在发送 SIGTERM **之前**，从 `state.db` 读取当前 session 消息
- 保存到 `~/.hermes/last_session_context/last_session.json`

**两个触发点：**
1. `gateway stop` handler — 停止 gateway 前
2. `gateway restart` handler — 重启 gateway 前

### 3. 新 session 启动读取

**通过 skill 指令：** 新 session 启动时读取 `~/.hermes/last_session_context/last_session.json`

## 文件结构

```
~/.hermes/
├── MEMORY                    # 索引文件（指向 memory/*.json）
├── memory/
│   ├── 01_douyin_platform.json
│   ├── 02_bazi.json
│   ├── 03_content_workflow.json
│   ├── 04_tools_config.json
│   └── 05_user_profile.json
└── last_session_context/
    └── last_session.json     # 自动保存的 session 对话记录
```

## 保存格式

```json
{
  "timestamp": "2026-07-19T...",
  "session_id": "...",
  "save_reason": "expiry|restart|stop|reset|new",
  "message_count": 100,
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
```

## GitHub PR

**官方仓库：** https://github.com/NousResearch/hermes-agent/pull/67272
**Fork 仓库：** https://github.com/iankwan827/hermes-agent

## 已修改文件

### 源码仓库
- `gateway/run.py` — 添加 `_auto_save_session()` 方法和调用
- `gateway/slash_commands.py` — `/new` `/reset` 命令保存逻辑
- `hermes_cli/gateway.py` — CLI 层面 stop/restart 前保存

### 安装包（venv）
- `venv/lib/python3.11/site-packages/hermes/gateway/cli.py` — stop/restart handler
- `venv/lib/python3.11/site-packages/hermes/gateway/run.py` — agent 级别保存
- `venv/lib/python3.11/site-packages/hermes/gateway/slash_commands.py` — `/new` `/reset`

## 个人同步仓库

**仓库：** `iankwan827/Hermes`
**路径：** `~/.hermes/hermes-agent/docs/session-continuity-changes.md`
**用途：** macOS ↔ Windows 通讯同步文档
