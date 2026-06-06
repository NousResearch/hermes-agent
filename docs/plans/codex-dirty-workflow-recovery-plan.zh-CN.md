# Codex Dirty Workflow Recovery 设计与实施计划

> **For Hermes:** 本计划用于改造 Hermes/Codex 分阶段开发工作流。目标是让 dirty worktree 不再无意义阻塞开发，同时保持 Codex 不踩未知 dirty 的安全边界。

**Goal:** dirty worktree 触发 Hermes 自动恢复/隔离流程，而不是让 staged Codex 开发直接停止。

**Architecture:** 保持底层 `codex_staged_implement` / `codex_impl_guard.py` fail-closed；在 skill 层修正工作流习惯，在 tool metadata 层提供可操作 dirty 信息，在高层 orchestrator 中执行安全恢复、隔离 worktree、Codex implementation、Codex review、Hermes verification 和 checkpoint。

**Non-goals:** 不放宽底层 guard；不让 Codex 写入未知 dirty worktree；不自动 reset/revert/drop/覆盖用户改动；不自动 force-push/deploy/restart。

---

## 1. 已确认需求

用户确认：

- dirty 不应直接阻塞 Codex 开发。
- Hermes 应先自动处理 dirty，能安全处理就处理。
- 不确定、可能破坏、可能覆盖用户手改时再问用户。
- 处理完 dirty 后继续 Codex 分阶段开发。
- 用户授权自动完成一个大阶段时，不应因为普通 dirty 状态频繁停住。
- 仍需最终由 Hermes 审批：Codex 输出只是候选补丁，Codex review 只是审查输入。

## 2. Codex review 反馈吸收

Codex 对设计做过只读 review，结论：方向正确，但实现前必须补安全边界。

本计划吸收以下 must-fix：

1. “已验证 current-stage diff”必须有证据来源：`stage_id` / `task_id` / allowlist / touched files / review / verification / time / status snapshot。
2. standing authorization 只能授权非破坏动作，不能授权 reset/revert/drop/覆盖/force-push/deploy/restart。
3. dirty 分类保守：`source`、`unknown`、secrets、真实数据、冲突、submodule、rename/delete/chmod、large binary 默认需要用户确认或隔离。
4. Codex leftover candidate diff 必须证明来源并重新 review/verify，不能直接 checkpoint。
5. isolated worktree 必须定义基准 commit、分支名、路径、生命周期和隔离语义。

## 3. 分层设计

### 3.1 Skill 层：先修行为

修改：

- `project-dev-workflow`
- `codex-staged-development-review`

新增原则：

```text
Dirty worktree is a pre-Codex recovery step, not a stop condition.
```

要求 agent：

1. 发现 dirty 后先分类和恢复。
2. 能自动处理安全 dirty 时继续。
3. 不能判断或动作有破坏性时才问用户。
4. 成功获得 clean baseline 或 isolated worktree 后继续 staged Codex。
5. 汇报时单独列出 dirty recovery，不混进 Codex implementation。

### 3.2 Tool metadata 层：只返回结构，不 mutation

`codex_staged_implement` 保持 dirty fail-closed，不直接清理、commit、stash、worktree。

dirty 返回增加/保持：

- `dirty_paths`
- `dirty_path_classes`: `source` / `test` / `docs` / `cache` / `unknown`
- bounded `diff_stat`
- `dirty_state_id`: 基于 porcelain 状态和 bounded evidence 的快照 ID，用于防 TOCTOU
- `auto_resolvable_classes`: 默认仅 cache/generated
- `requires_user_decision`: 需要用户确认的路径和原因
- `unsafe_reasons`: source/unknown/secret/data/conflict/rename/delete/chmod/binary/submodule 等
- `resume_strategy`: `clean_worktree_required` / `isolated_worktree_recommended` / `ask_user`

原则：path class 只说明文件形态，不说明 ownership。

### 3.3 Runtime/orchestrator 层：高层恢复流程

新增高层工具：`codex_workflow_run`。

它负责完整自动流程：

1. preflight：repo、status、process、dirty state。
2. dirty recovery：安全清理、隔离、或请求用户。
3. staged implementation：调用 `codex_staged_implement`，显式 allowlist。
4. review：调用 `codex_review_guard.py` / packet-only review。
5. verification：Hermes 跑测试、diff check、缓存清理。
6. checkpoint：在 standing authorization 下，只 checkpoint 已验证 stage diff。
7. report：四层归因 + dirty recovery + git state。

底层 `codex_impl_guard.py` 不变，仍做安全拦截。

## 4. Dirty 决策表

| Dirty 类型 | 默认处理 |
|---|---|
| `__pycache__` / `.pytest_cache` / `*.pyc` | 自动清理 |
| 明确 generated/cache/build 临时物 | 自动清理 |
| 当前阶段已验证 diff | 可 checkpoint commit |
| Codex leftover candidate diff | 证明来源 + review + verify 后才 checkpoint |
| unrelated dirty 且任务需继续 | 建 isolated clean worktree 继续 |
| `source` dirty | 默认需要 stage 证据，否则问用户或隔离 |
| `unknown` | 问用户或隔离 |
| secrets / 真实数据 | 问用户，不自动处理 |
| conflict / merge 状态 | 问用户 |
| rename/delete/chmod | 问用户 |
| large binary | 问用户 |
| submodule | 问用户 |

## 5. “已验证 current-stage diff”定义

必须同时满足：

- `stage_id` / `task_id` 明确。
- 变更文件在 `allowed_files` / `allowed_globs` 内。
- `dirty_state_id` 与验证时一致。
- touched files 与记录一致。
- Codex implementation 状态记录存在：`not_run` / `candidate_diff` / `trusted_completion` 等。
- Codex review 结果记录存在；若 unavailable，必须明确标记并由 Hermes 自审补足。
- Hermes verification 命令和输出存在。
- 验证时间和当前 git status 可复核。
- 无后台 Codex 进程仍在修改工作区。

不满足则不能自动 checkpoint。

## 6. Standing authorization 语义

用户说“自动完成整个大阶段”时，视为 standing authorization，但只允许非破坏动作。

允许自动：

- 清缓存。
- 跑测试/编译/diff check。
- 调 Codex staged implementation。
- 调 Codex read-only review。
- Hermes 修真实 must-fix。
- 创建 isolated worktree。
- checkpoint commit 已验证 stage diff。
- 继续下一 slice。

不允许自动：

- reset / revert / drop。
- 覆盖用户手改。
- 删除 unknown 文件。
- force push。
- deploy / restart。
- 处理 secrets/真实数据。
- 解决 merge/rebase 冲突。

## 7. Isolated worktree 策略

当 dirty 不能安全自动处理，但任务可以从 clean commit 继续：

1. 记录原工作区 dirty baseline。
2. 使用当前 `HEAD` 或指定 clean base commit 创建 worktree。
3. 分支命名：`work/<stage>-<yyyymmdd>-<shortsha>`。
4. 路径命名：`../.hermes-worktrees/<repo>-<stage>-<shortsha>`。
5. 新 slice 只在 isolated worktree 中运行。
6. 原 dirty 工作区不清理、不覆盖、不 commit。
7. 汇报原 dirty 留存、新 worktree 路径、基准 commit、分支名。

## 8. Orchestrator 状态机

```text
START
  -> preflight
  -> dirty?
     -> no: codex_staged_implement
     -> yes: classify dirty
        -> safe auto-resolvable: cleanup/checkpoint/isolate
        -> unsafe/ambiguous: ask user or isolate
  -> re-check dirty_state_id
  -> clean baseline or isolated worktree ready?
     -> yes: codex_staged_implement
     -> no: stop with exact reason
  -> Codex implementation
  -> Codex read-only review
  -> Hermes verification
  -> checkpoint/report/next slice
```

## 9. 报告格式

每次报告必须分开：

```text
Dirty recovery:
- 发现 dirty：...
- 自动处理：...
- 未触碰：...
- worktree：...

Codex implementation:
- status：not_run / failed / candidate_diff / trusted_completion
- evidence：...

Codex review:
- status：not_run / unavailable / packet_only_passed / packet_only_failed
- must-fix：...

Hermes verification:
- commands：...
- result：...

Git state:
- branch / commit / clean|dirty / ahead|behind / push status
```

## 10. 实施阶段

### Phase 1：Skill 修正

- Patch `project-dev-workflow`：dirty 是恢复流程，不是停止条件。
- Patch `codex-staged-development-review`：加入 dirty recovery 决策表、standing authorization 边界、四层报告补充。

### Phase 2：Tool metadata

- 为 dirty fail-closed 返回添加 `dirty_state_id`。
- 添加 `auto_resolvable_classes`、`requires_user_decision`、`unsafe_reasons`、`resume_strategy`。
- 添加 focused tests：状态快照、保守分类、无 mutation、bounded 输出。

### Phase 3：最小 `codex_workflow_run`

- 新增高层工具和 tests。
- 仅支持：preflight、自动 cache cleanup、dirty 不明时建议/创建 isolated worktree、调用 staged implementation。
- 不支持自动 checkpoint source diff。

### Phase 4：Checkpoint / leftover 恢复

- 定义 verified stage ledger。
- 支持已验证 diff checkpoint。
- 支持 Codex leftover candidate 识别，但必须 review + verify 后 checkpoint。
- 加测试：无证据不得 checkpoint，dirty_state_id 变化不得 checkpoint。

## 11. 验证计划

Focused：

```bash
python -m pytest tests/tools/test_codex_staged_implement_tool.py -q -o addopts=''
python -m pytest tests/tools/test_codex_workflow_run_tool.py -q -o addopts=''
python -m pytest tests/test_toolsets.py tests/hermes_cli/test_tools_config.py -q -o addopts=''
```

Static：

```bash
python -m py_compile tools/codex_staged_implement_tool.py tools/codex_workflow_run_tool.py tests/tools/test_codex_staged_implement_tool.py tests/tools/test_codex_workflow_run_tool.py

git diff --check -- tools/codex_staged_implement_tool.py tools/codex_workflow_run_tool.py tests/tools/test_codex_staged_implement_tool.py tests/tools/test_codex_workflow_run_tool.py toolsets.py hermes_cli/tools_config.py docs/plans/codex-dirty-workflow-recovery-plan.zh-CN.md
```

收尾：

- 清理 `__pycache__` / `*.pyc` / `.pytest_cache`。
- 复查 `git status --short --branch --untracked-files=all`。
- Codex read-only review 当前实现包。
- Hermes 最终审批。

## 12. 成功标准

- 普通 dirty 不再导致 agent 停止开发。
- Codex 仍不会直接写入未知 dirty worktree。
- 安全 dirty 可自动处理。
- 不明 dirty 可通过 isolated worktree 继续开发。
- checkpoint 只发生在“已验证 current-stage diff”上。
- 报告能清楚分离 dirty recovery、Codex implementation、Codex review、Hermes verification。
