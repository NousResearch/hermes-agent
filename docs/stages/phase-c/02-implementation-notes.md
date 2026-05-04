# Phase C: Implementation Notes

Date: 2026-05-04

## Worktree Isolation

### 实现方式
- 使用 `git worktree add --detach` 创建独立工作区
- 工作区路径: `~/.hermes/worktrees/hermes-{subagent_id}/`
- 通过 `register_task_env_overrides(subagent_id, {"cwd": worktree_path})` 确保 terminal/file 操作在 worktree 中
- 不使用全局 `os.chdir()`

### 创建流程
1. `_create_worktree()` — 从父 Agent 的 git repo 创建 worktree
2. 验证 git repo 存在且 HEAD 可解析
3. 返回 `{worktree_path, worktree_branch, original_head, repo_path, error}`
4. 在 `_run_single_child` 中调用，创建失败时返回 tool error

### 清理策略 (Fail-Closed)
`_cleanup_worktree()` 按以下顺序判断：
1. 路径不存在 → 无操作
2. 路径不在 `~/.hermes/worktrees/` 下 → 拒绝删除
3. `git status` 失败 → 拒绝删除
4. `.git/index.lock` 存在 → 拒绝删除
5. 有未提交改动 → 保留并返回 `diff_summary`
6. 无改动 → 执行 `git worktree remove --force`

### 清理钩子
- 在 `_run_single_child` 的 finally 块中调用
- 清理后调用 `clear_task_env_overrides()` 注销 cwd 覆盖

## Transcript JSONL

### 路径结构
```
~/.hermes/subagents/{session_id}/{subagent_id}.jsonl
```

### 记录格式
```json
{
  "timestamp": 1700000000.0,
  "event_type": "tool_call|tool_result|final|error",
  "subagent_id": "sa-0-abc123",
  "agent_id": "kimi",
  "tool": "web_search",
  "preview": "...",
  "usage": {"input_tokens": 100, "output_tokens": 50}
}
```

### 集成点
- `_write_transcript_event()` — 追加单行 JSON 到 transcript 文件
- 在子 Agent 完成时记录 "final" 事件
- `transcript_path` 包含在 `subagent.completed` EventLog 事件和 result entry 中

## Coordinator/Swarm 预备

### Coordinator Mode
- 从 `agent-registry.json` 的 `agent_config.coordinator_mode: true` 读取
- 限制 toolsets 为 `delegation` + `file`（orchestration only）
- 阻止所有动手工具: terminal, process, write_file, patch, send_message, memory, execute_code, browser_*

### claim_task()
- 使用 SQLite WAL `INSERT OR IGNORE` 实现原子认领
- 数据库: `~/.hermes/swarm_claims.db`
- 返回 `True` 仅对第一个认领者
- 仅保证本地文件系统原子性（APFS/ext4/NTFS）

### `<task-notification>` 格式
```xml
<task-notification>
  <task-id>uuid</task-id>
  <status>completed|failed|running</status>
  <result>summary text</result>
  <usage>{"input_tokens":100,"output_tokens":50}</usage>
</task-notification>
```
Future Phase: mailbox 替换为 `<task-notification>` 协议。

## 未实现
- `run_in_background` — schema 已添加，Phase C+ 实现
- `send_message(subagent_id, message)` — 需要 agent loop 改造
- `delegate-v27.sh` 改造 — 未修改
