---
name: claude-session
description: Guide for using claude_session tool to delegate coding tasks to Claude Code via tmux. Includes session persistence (auto-resume via .claude/claude-session.json) and list_persisted for cross-restart continuity.
tags: ['claude-code', 'tmux', 'interactive', 'coding', 'delegation', 'persistence', 'auto-resume']
triggers:
  - "claude session"
  - "coding task"
  - "delegation"
  - "interactive session"
  - "resume session"
  - "previous session"
  - "persisted session"
version: 4.5
code_commit: see git history
required_environment_variables:
  - name: HERMES_STREAM_STALE_TIMEOUT
    prompt: "Stream stale timeout (秒，推荐 300)"
    help: "防止 claude_session 长任务时 Hermes API 流中断，默认 180s 不够用"
    optional: true
    required_for: "防止 Stream Stalled mid tool-call 错误"
  - name: HERMES_CLAUDE_SESSION_PATROL_INTERVAL
    prompt: "Patrol check interval (秒，推荐 300)"
    help: "idle 状态检查输出增长的间隔，默认 300 秒（5 分钟）"
    optional: true
    required_for: "patrol checkpoint 轮询间隔"
  - name: HERMES_CLAUDE_SESSION_STALL_THRESHOLD
    prompt: "Stall threshold (秒，推荐 1800)"
    help: "无输出增长判定为 stall 的时间，默认 1800 秒（30 分钟）"
    optional: true
    required_for: "长时间无响应检测"
  - name: HERMES_CLAUDE_SESSION_OBSERVER_POLL_INTERVAL
    prompt: "Observer poll interval (秒，推荐 5)"
    help: "observer 后台轮询间隔，默认 5 秒（足够捕捉短工具调用，180秒会错过 2-3 秒的任务）"
    optional: true
    required_for: "状态变化检测频率"
---

# Claude Session — 任务委托核心框架

## 架构原则（第一性原理）

**claude_session 是 Hermes 与 Claude Code 之间唯一的通信通道。**

```
┌─────────────────────────────────────────────────────────┐
│  Hermes (AI)                                            │
│       ↓                                                │
│  claude_session API  ←  唯一的入口，所有操作必须通过这里 │
│       ↓                                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ observer (后台线程) + OutputBuffer (输出缓冲)    │   │
│  │  - 后台追踪 tmux 状态                            │   │
│  │  - 缓冲输出供 API 查询                           │   │
│  └─────────────────────────────────────────────────┘   │
│       ↓                                                │
│  StateDetection (纯函数 detect_state)                   │
│       ↓                                                │
│  tmux_interface (终端控制)                             │
│       ↓                                                │
│  Claude Code CLI (执行者)                              │
└─────────────────────────────────────────────────────────┘
```

**核心规则**：
- 所有状态来自 API，不来自手动 tmux
- observer 后台运行，不应被绕过
- API 返回的数据已经过 buffer 整理

---

## ⚠️ 禁止行为（强制）

### 1. 禁止手动 tmux 操作

```python
# ❌ 禁止：绕过 observer + buffer 体系
terminal("tmux capture-pane -t session -p")
terminal("tmux send-keys ...")

# ✅ 正确：使用 API 获取状态和输出
claude_session(action="status")      # 返回 state + output_tail
claude_session(action="output")     # 获取完整输出
```

**原因**：手动 tmux 操作破坏状态追踪，导致：
- 状态不一致（API 说 IDLE，手动读出 THINKING）
- buffer 数据过时
- 并发问题（observer 正在读时你也在读）

### 2. 禁止直接 terminal 调用 claude

```python
# ❌ 禁止
terminal("claude -p 'fix bug'")
terminal("tmux send-keys -t session 'claude'")

# ✅ 正确
claude_session(action="send", message="fix bug")
```

### 3. 禁止反复轮询

```python
# ❌ 禁止：每30秒轮询一次
send → wait_for_idle(30) → status → wait_for_idle(30) → output → ...

# ✅ 正确：一次等到底
send → wait_for_idle(300-600) → output → stop
```

---

## 正确使用流程

### 标准模式（3轮完成）

```python
# 第1轮：启动
claude_session(action="start", name="task-name", workdir="/project", permission_mode="skip")
claude_session(action="wait_for_idle", timeout=60)  # 等待就绪

# 第2轮：执行任务
claude_session(action="send", message="Your task here")
claude_session(action="wait_for_idle", timeout=300)  # 一次等到底（工具默认900s）

# 第3轮：获取结果
result = claude_session(action="output", limit=100)
claude_session(action="stop")
```

### 状态检查

```python
# ❌ 不要手动读 tmux
terminal("tmux capture-pane -t session -p")

# ✅ 用 status API
status = claude_session(action="status")
# 返回: {"state": "THINKING", "state_duration_seconds": 42.3, "output_tail": "..."}
```

### 进度监控

```python
# wait_for_idle 返回值包含 output_since_send，不需要单独调 output
result = claude_session(action="wait_for_idle", timeout=300)
# result["output_since_send"] 已包含增量输出
```

---

## Named Sessions（多会话管理）

v3.0+ 支持通过语义化名称管理多个会话。

### 为什么 name 必须

```python
# ❌ 旧方式（自动生成 hash，不可读）
claude_session(action="start", workdir="/project")  # 生成 hermes-7935c242

# ✅ 新方式（语义化，一目了然）
claude_session(action="start", name="frontend", workdir="/project")
claude_session(action="start", name="backend", workdir="/project")
```

### 路由优先级

| 优先级 | 参数 | 说明 |
|--------|------|------|
| 1 | `session_id` | 精确指定 |
| 2 | `name` | 语义化路由 |
| 3 | 活跃会话 | 最近交互 |
| 4 | 最近创建 | 回退 |

---

## Permission Modes

| Mode | 场景 | 说明 |
|------|------|------|
| `skip` | 自动化任务 | 自动批准所有操作 |
| `normal` | 需要用户确认 | 手动响应权限 |

**建议**：多步任务用 `skip`，避免频繁权限中断。

---

## State Awareness

7个状态：`IDLE` `THINKING` `TOOL_CALL` `PERMISSION` `ERROR` `DISCONNECTED` `EXITED`

```python
# 检查状态
status = claude_session(action="status")
if status["state"] == "IDLE":
    # 可以发送任务
elif status["state"] == "PERMISSION":
    # 需要响应权限
```

---

## 常见错误（真实案例）

### 错误1：手动读 tmux 绕过 observer

**现象**：status 返回 IDLE，但 tmux pane 显示还在 thinking。

**原因**：手动 `tmux capture-pane` 读到的和 observer buffer 不一样。

**正确做法**：只用 API，用 `status["output_tail"]` 或 `output`。

### 错误2：以为 Claude 卡住了

**现象**：等待 2 分钟没有响应，判定"卡住"并重启。

**真实情况**：Claude 正在读取多个文件、分析代码，GLM-5.1 处理大上下文需要时间。

**正确做法**：
- 给够 timeout（300-600秒）
- 用 `status["state_duration_seconds"]` 判断是否真的停滞
- 除非明确超时，否则不重启

### 错误3：反复轮询 + 超时递减

**现象**：
```python
# 每轮 timeout 递减，像是轮询而不是等待
wait_for_idle(60) → 没完成 → wait_for_idle(300) → 没完成 → wait_for_idle(600)
```

**后果**：
- 每次 `wait_for_idle` 调用都生成新的 output context，Claude 每次都要处理
- Hermes 反复检查打断 Claude，Claude 丢失耐心无法完成复杂任务
- 每轮累积上下文 → 最终 Stream Stalled

**正确做法**：`wait_for_idle(900)` 一次给够 timeout，不中途打断

### 错误4：动画鬼影导致提前返回 IDLE

**现象**：`wait_for_idle` 返回 `status: idle`，但 `output_since_send` 显示的是动画（Forming/Unfurling/Jitterbugging/Stewing），命令实际结果未捕获。

**原因**：Claude Code 动画结束后，tmux pane 先显示短暂的 `❯` 提示符（触发 IDLE 检测），随后动画恢复并持续显示。observer 的 2 秒 poll 间隔捕获的是动画状态。

**正确做法**：
```python
# session.py wait_for_idle() 中，检测到 IDLE 后等待 0.5 秒再次确认
if state == SessionState.IDLE:
    time.sleep(0.5)
    confirm_pane = self._tmux.capture_pane()
    confirm_lines = clean_lines(confirm_pane)
    confirm_result = detect_state(confirm_lines)
    if confirm_result.state != SessionState.IDLE:
        continue  # 仍在过渡中，继续等待
    return {**self._build_idle_result(), "status": "idle"}
```

### 错误5：PATROL_INTERVAL 默认值过大（中低优）

**现象**：`PATROL_INTERVAL=300`（5分钟）导致 THINKING/TOOL_CALL 状态轮询间隔过长。

**现状**：代码中 THINKING 使用 `POLL_INTERVAL=180s`（3分钟），与 PATROL_INTERVAL 分离。

**建议**：保持现状，observer 使用自适应轮询（THINKING=5s 已修复），wait_for_idle 使用 POLL_INTERVAL=180s。

---

## 轮询策略

### 核心原则

用最少 tool call 完成任务。

### 错误模式（❌）

```
send → wait_for_idle(30) → output → wait_for_idle(30) → output → ... (10+轮)
```

每轮累积上下文 → 最终 Stream Stalled

### 正确模式（✅）

```
1. start + wait_for_idle(60)     # 启动，等就绪
2. send + wait_for_idle(300-600) # 发任务，一次等到底
3. output + stop                 # 取结果，关闭
```

---

## 陷阱与应对

### 陷阱0：首次启动不稳定

**现象**：wait_for_idle 超时，但实际 Claude 还在初始化。

**应对**：首次启动 timeout 设为 300 秒。

### 陷阱1：Permission 状态幽灵

**现象**：status 返回 PERMISSION，但 Claude 实际在正常工作。

**应对**：检查 `output_tail` 是否有权限提示文本，再决定是否 respond。

### 陷阱2：多行文本粘贴后不发送

**现象**：多行消息发送后 Claude 没收到。

**应对**：代码已有 10 秒延迟保护（session.py:422），不要在发送后立即 wait_for_idle。

### 陷阱3：tokens 不增长

**现象**：Claude 卡在某个 token 数不动。

**应对**：
```python
claude_session(action="cancel_input")
claude_session(action="send", message="简化指令")
```

---

## Error Recovery

1. 检查 `status` 的 state 字段
2. 检查 `output_tail` 或 `output_since_send` 的错误信息
3. 发送修正指令（最多 2 次重试）
4. 仍然失败 → 报告用户
5. **绝不等得久就放弃当前方案**

---

## API Reference

### start
```
claude_session(action="start", name="my-task", workdir="/path",
               permission_mode="skip")
```

### send
```
claude_session(action="send", message="Fix the auth bug")
claude_session(action="send", name="frontend", message="...")
```

### status
```
claude_session(action="status")
# 返回: {"state": "IDLE", "state_duration_seconds": 12.4, "output_tail": "..."}
```

### wait_for_idle
```
claude_session(action="wait_for_idle", timeout=300)
# 返回: {"status": "idle", "output_since_send": "...", "state": "IDLE"}
```

### output
```
claude_session(action="output", offset=0, limit=50)
```

### stop
```
claude_session(action="stop")
claude_session(action="stop", name="frontend")
```

### list / switch
```
claude_session(action="list")
claude_session(action="switch", name="frontend")
```

### diagnose
```
claude_session(action="diagnose")
# 检查 tmux、Claude CLI、环境变量
```

### list_persisted（持久化会话）
```
claude_session(action="list_persisted", workdir="/project")
# 返回该 workdir 下 .claude/claude-session.json 中所有持久化的会话
# 每条记录含 status（持久化时状态）+ active_in_gateway（当前 gateway 是否活跃）
```

---

## Session Persistence（自动 resume 机制）

**核心目的**：跨进程重启保留会话上下文，避免每次新建会话都从零开始。

### 存储位置
```
<workdir>/.claude/claude-session.json
```

格式示例：
```json
{
  "task-name": {
    "claude_session_uuid": "034271e5-...",
    "workdir": "/path/to/project",
    "model": "sonnet",
    "permission_mode": "skip",
    "resume_count": 3,
    "last_resume_status": "auto_resumed",
    "status": "active",
    "last_active_at": "2026-06-02T11:24:41"
  }
}
```

### 写入时机
- `start` 成功：写入/更新条目，`status: "active"`
- `stop` 成功：更新 `status: "stopped"`
- 自动 resume：增加 `resume_count`，`last_resume_status: "auto_resumed"`

### 读取时机
- `claude_session(action="list_persisted", workdir=...)`：列出 workdir 下所有持久化会话
- 启动同 `name` 的会话时：若存在条目，自动用 `claude_session_uuid` resume

### 何时应该主动调用 list_persisted
- 接手新 workdir，先查有没有可恢复的会话
- 用户提到"上次"、"之前的会话"、"继续上次"
- 不确定当前 workdir 有哪些命名会话

⚠️ 注意：`status` 字段是**持久化时的状态**（最后写入时），不代表**当前进程是否活跃**。判断当前活跃用 `active_in_gateway: true` 字段。

---

## 首次使用配置

加载 skill 时检查 `HERMES_STREAM_STALE_TIMEOUT`：

```bash
bash ~/.hermes/skills/claude-session/scripts/configure.sh
```

未配置时可能遇到 `⚠ Stream stalled mid tool-call` 错误。

配置后需重启 Gateway。