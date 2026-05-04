# Worktree Path Audit Report

Date: 2026-05-04
Status: 审计通过 — worktree isolation 可行

## 审计范围

| 区域 | 是否支持 cwd override | 说明 |
|---|---|---|
| Terminal session cwd | 部分（per-command workdir） | 通过 `register_task_env_overrides(task_id, {"cwd": ...})` 可设置 per-task cwd，但当前 subagent 的 task_id 全部折叠为 "default" |
| File tools path resolution | 否 | `_resolve_path_for_task()` 使用 TERMINAL_CWD env var，无 per-agent cwd |
| file_state.py | 否（按 task_id 分区） | 已使用绝对路径，按 task_id 分区，天然兼容 |
| CheckpointManager | 否（绝对路径） | 使用 SHA-256 哈希绝对路径 key 到 shadow repo，天然兼容 |
| Approval callbacks | N/A | 无 cwd 依赖 |
| delegate_tool workspace hint | 否 | 引用 `parent_agent.terminal_cwd` 和 `parent_agent.cwd` 但这两个属性在 AIAgent 上不存在（死代码） |
| AIAgent | 否 | 无 `self.cwd` 或 `self.terminal_cwd` 属性 |

## 审计结论

### 现状
- 整个代码库的 cwd 解析全部通过 `os.getenv("TERMINAL_CWD")` 或 `os.getcwd()`
- AIAgent 没有 per-instance cwd 属性
- delegate_tool 的 `_resolve_workspace_hint()` 引用了不存在的属性（死代码）
- `register_task_env_overrides()` 机制存在但 subagent 未使用

### 实现 worktree 的可行路径

1. **创建 git worktree**: 使用 `git worktree add` 在 `~/.hermes/worktrees/{subagent_id}/` 创建独立工作区
2. **设置 terminal cwd**: 使用 `register_task_env_overrides(child_task_id, {"cwd": worktree_path})` — 此机制已存在于 `tools/terminal_tool.py`，保证 terminal 命令在 worktree 中执行
3. **文件路径解析**: terminal cwd 被 file tools 的 `_get_live_tracking_cwd()` 追踪，所以 `cd` 到 worktree 后，文件操作自动跟随
4. **系统提示注入**: 通过 `_build_child_system_prompt` 的 workspace_path 参数告知子 Agent worktree 路径
5. **清理策略**: fail-closed — 无改动时删除，有改动时保留，不确定时保留

### 不使用全局 `os.chdir()`
使用 `register_task_env_overrides` + absolute worktree path，不修改进程全局状态。

### 审计结论: 通过
现有基础设施（register_task_env_overrides + task_id 分区 + 绝对路径 checkpoint/file_state）足以支持 worktree isolation。不需要大规模重构。
