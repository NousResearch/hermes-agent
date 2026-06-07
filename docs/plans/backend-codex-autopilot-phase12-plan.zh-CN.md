# Phase 12 Backend/Codex 大阶段自动闭环计划

> **For Hermes:** 使用 `codex-staged-development-review` 分阶段执行。Codex 只产候选 patch 或只读 review；Hermes 负责 ledger、边界、验证、归因与最终判断。

**Goal:** 把后端/项目阶段开发从“Agent 记忆里的流程”收敛成可恢复、可审计、owner/session_id-aware、fail-closed 的最小自动闭环：Ownership/Provenance Contract、Stage Ledger、Dry-run Planner、Review Autopilot、Verification Matrix、Bounded Must-fix Loop、Next-stage Recommender。

**Architecture:** Phase 12 只新增 ownership/provenance、orchestrator、ledger、risk-router 层，不重写 `codex_staged_implement`、dirty guard、`codex_review_packet.py`、`codex_review_guard.py`、`codex_impl_guard.py`、`codex_stage_runner.py`。第一实现切片只做 `12A0 + 12A + 12B`：owner/session_id provenance gate + ledger/resume contract + dry-run planner。

**Tech Stack:** Python, pytest, git, existing Hermes tools, existing Codex guarded runtime scripts.

---

## 0. 当前状态与边界

### 已有基础

- `tools/codex_staged_implement_tool.py`：显式 allowlist、dirty metadata、候选 diff 生命周期。
- `tools/codex_workflow_run_tool.py`：最小 dirty recovery orchestrator、dry-run、cache cleanup、isolated worktree、checkpoint verified diff、leftover candidate。
- `scripts/runtime/codex_impl_guard.py`：单 slice Codex implementation guard。
- `scripts/runtime/codex_stage_runner.py`：JSON plan 驱动 staged runner。
- `scripts/runtime/codex_review_packet.py`：bounded review packet。
- `scripts/runtime/codex_review_guard.py`：read-only review guard、bounded JSON、flood/timeout fail-closed。

### 当前 dirty / provenance 边界

shared worktree 可能已有来源未知 dirty。Phase 12 计划与实现必须先按 owner/session_id provenance 判定，再按 path class 判定；**没有 owner 证明时默认 `unknown_unowned`，只能 preserve / block / isolate，不能清理或覆盖**。

本次事故背景：多个 QQ/Hermes 会话共用同一 `repo / branch / worktree`，Git dirty 只显示路径状态，不显示 owner/session_id；用户报告另一个 session 曾写入计划文档，后续 unknown dirty doc 被误判为污染并清理。本计划把该事件作为 `user-reported incident` 记录，用于设计 owner/session_id-aware dirty workflow；后续实现不得把该报告当作已独立验证的文件归属证据。

Live dirty baseline 必须在每个 stage 开始时重新采集。当前或历史 dirty path 只能作为 `unknown_unowned` / `other_known_session` / `current_session` 等 provenance class 处理，不得仅凭路径推断归属。

特别规则：`docs/plans/*` 的 untracked/added 文档默认 preserve。除非 provenance 证明它是当前 session 创建的临时 review artifact，否则不能把 repo 内计划文档当作 cache/review 污染清理。

不得接管、清理、delete、reset、revert、stash、drop、overwrite 或混入 Phase 12 提交，除非同时满足本计划 12A0 的 owner/session_id、allowlist、explicit authorization、before_hash gate。

### 非目标

- 不重写 dirty guard / staged implement / review packet / review guard / impl guard / stage runner。
- 不让 Codex 自动 commit、push、PR、merge、deploy、restart。
- 不自动处理 secrets、真实数据、merge/rebase 冲突、force push、权限扩大。
- 不自动 stash/reset/revert/drop/overwrite 未知 dirty。
- 不把 `review_unavailable`、timeout、flood、provider error、invalid JSON 计为通过。
- 不把“自动推荐下一阶段”等同于“自动推进下一阶段”。默认只推荐。

---

## 1. Phase 12A0：Ownership / Provenance Contract

### Objective

在 12A ledger/resume 与 12B dry-run planner 之前，先建立最小 owner/session_id provenance contract。任何 cleanup、delete、overwrite、stash/reset/revert/drop 相关决策都必须先经过 provenance 判定；无法证明为 current session owned 时，默认 `unknown_unowned` 并 preserve/block/isolate。

### Files

- Create: `agent/codex_workflow_provenance.py`
- Modify: `agent/codex_workflow_ledger.py`（12A 创建后引用 provenance snapshot / event ids）
- Modify: `tools/codex_workflow_run_tool.py`
- Test: `tests/tools/test_codex_workflow_provenance.py`
- Test: `tests/tools/test_codex_workflow_run_tool.py`

### Provenance event schema v1

最小事件字段：

```json
{
  "schema_version": 1,
  "event_id": "...",
  "repo_id": "...",
  "branch": "...",
  "head_sha": "...",
  "stage_id": "phase12a0-provenance-contract",
  "session_id": "...",
  "actor": "hermes | codex_guard | review_guard | user",
  "tool": "...",
  "operation": "create | modify | review_artifact | cleanup_candidate | delete_candidate | overwrite_candidate",
  "path": "docs/plans/backend-codex-autopilot-phase12-plan.zh-CN.md",
  "path_class": "docs | source | test | generated_cache | review_artifact | unknown",
  "before_hash": null,
  "after_hash": "...",
  "owner_session_id": "...",
  "owner_policy": "current_session | other_known_session | unknown_unowned | generated_cache | review_artifact_current_session | dangerous_conflict",
  "authorization": {"explicit": false, "reason": "..."},
  "timestamp": "..."
}
```

Rules:

- `repo_id` 与 12A ledger 一致：repo absolute path 的 SHA-256 前 16 位。
- `session_id` 必须来自当前 Hermes/gateway/session context；缺失时不得伪造，只能记录 `unknown` 并降级为 `unknown_unowned`。
- `before_hash` / `after_hash` 使用文件内容 hash；路径不存在时为 `null`。
- 事件内容必须 redaction：不得记录 token、provider payload、raw Codex logs、真实用户数据。
- provenance event 可被 ledger 引用；恢复时不能只看 path/status，必须复核 hash 与 owner policy。

### Ownership classes

| Class | Meaning | Default behavior |
|---|---|---|
| `current_session` | 当前 session 创建/修改，且 hash 与 event 匹配 | 可进入 allowlist + authorization gate |
| `other_known_session` | 有已知非当前 session owner 证据 | preserve；不得 cleanup/delete/overwrite |
| `unknown_unowned` | 无 owner 证据、owner 过期、hash 不匹配或路径新出现 | preserve/block/isolate |
| `generated_cache` | 生成缓存，且不在 repo source/doc/test 关键路径 | 可在非 dry-run 且授权后清理 |
| `review_artifact_current_session` | 当前 session 创建的 `/tmp` 或 allowlisted review artifact | 可在授权后清理 |
| `dangerous_conflict` | delete/rename/chmod/submodule/binary/conflict/secrets suspected | fail-closed，人工确认 |

### Cleanup / delete / overwrite gate

任何 repo 内路径的 cleanup、delete、overwrite 必须同时满足：

1. `owner_policy == current_session` 或 `review_artifact_current_session`。
2. path 在当前 stage allowlist 中，且不是默认 preserve 的 `docs/plans/*` 正式计划文档。
3. 用户或 standing authorization 明确覆盖该操作类型。
4. `before_hash` 与当前文件 hash 匹配；hash mismatch 立即降级为 `unknown_unowned`。
5. 操作不是 stash/reset/revert/drop/force-push/deploy/restart。

任一条件不满足：`cleanup_allowed=false`，记录原因，preserve/block/isolate，不执行文件变更。

### Stale / missing provenance rules

- provenance event 缺失：`unknown_unowned`。
- event 的 `after_hash` 与当前文件 hash 不一致：`unknown_unowned`。
- event 的 `owner_session_id` 不是当前 session：`other_known_session`。
- event 的 `head_sha` 与当前 HEAD 不一致：需要重新归因；不能沿用 current-session ownership。
- lock 只覆盖 `repo_id + branch + stage_id` 不足以保护跨 stage/session cleanup；delete/overwrite 必须额外通过 repo/branch/path/hash provenance gate。

### Tests

- `test_other_session_dirty_is_preserved`
- `test_unknown_untracked_doc_plan_is_preserved`
- `test_cleanup_owner_mismatch_blocks_delete`
- `test_delete_hash_mismatch_blocks_delete`
- `test_review_artifact_cleanup_does_not_delete_foreign_doc`
- `test_missing_session_id_degrades_to_unknown_unowned`
- `test_docs_plans_default_preserve_without_current_session_artifact_proof`

---

## 2. Phase 12A：Stage Ledger / Resume Contract

### Objective

新增持久化 stage ledger contract，使上下文压缩、会话中断、Codex timeout/flood、review unavailable 后都能恢复到可审计状态。

### Files

- Create: `agent/codex_workflow_ledger.py`
- Modify: `tools/codex_workflow_run_tool.py`
- Test: `tests/tools/test_codex_workflow_ledger.py`
- Test: `tests/tools/test_codex_workflow_run_tool.py`

### Persistence

默认 ledger root：

```text
$HERMES_HOME/runtime/codex_workflows/<repo_id>/<branch>/<stage_id>.json
```

Rules:

- `repo_id` 使用 repo absolute path 的 SHA-256 前 16 位，不直接把绝对路径放进文件名。
- ledger 内容可记录 repo path，但必须 redaction：用户 home、`HERMES_HOME`、token-like text、API key-like text 全部脱敏。
- 原子写入：写 `<file>.tmp`，`fsync`，再 `os.replace()`。
- 并发锁：同目录 `<stage_id>.lock`，同一 `repo_id + branch + stage_id` 只能一个 active run。
- schema version 初始为 `1`，未知新版本必须 fail-closed：`ledger_schema_unsupported`。

### Ledger schema v1 必填字段

```json
{
  "schema_version": 1,
  "ledger_id": "...",
  "stage_id": "phase12a-ledger-resume",
  "repo": {"path_redacted": "...", "head_sha": "...", "branch": "...", "upstream": "..."},
  "authorization": {
    "standing_authorization": false,
    "may_write_files": false,
    "may_run_codex_impl": false,
    "may_run_codex_review": true,
    "may_commit": false,
    "may_push": false,
    "may_deploy_or_restart": false
  },
  "scope": {"allowed_files": [], "allowed_globs": [], "excluded_dirty_paths": [], "risk_classes": []},
  "dirty_baseline": {
    "dirty_state_id": "...",
    "paths": [],
    "classes": [],
    "blocking_reasons": [],
    "resume_strategy": "clean_current | isolated_worktree | stop_for_user | dry_run_only"
  },
  "events": [],
  "review": {"verdict": "not_run | passed | failed | unavailable", "must_fix_count": 0},
  "verification": {"matrix_id": null, "commands": [], "results": [], "skipped": []},
  "next_stage": {"recommendation": null, "authorization_required": true}
}
```

### Resume contract

| 条件 | 行为 |
|---|---|
| ledger 缺失 | `resume_status=not_found` |
| schema unsupported | `resume_status=blocked` |
| `head_sha` 不同且 dirty 有重叠 | `resume_status=blocked` |
| `head_sha` 不同但无重叠 | `resume_status=needs_replan` |
| lock active 且 PID 存活 | `resume_status=active_elsewhere` |
| previous status `running` 但无进程 | `resume_status=interrupted_recoverable`，只能 dry-run 重算 |
| review unavailable / verification failed | 保留 fail-closed 状态，不自动升级 |

### Tests

- `test_ledger_atomic_write_and_read_roundtrip`
- `test_ledger_redacts_home_hermes_home_and_token_like_values`
- `test_ledger_rejects_unsupported_schema`
- `test_resume_blocks_on_head_change_with_overlapping_dirty`
- `test_resume_marks_interrupted_running_without_process_as_recoverable_dry_run_only`
- `test_lock_blocks_second_active_run_for_same_repo_branch_stage`

---

## 3. Phase 12B：Orchestrator Dry-run Planner

### Objective

不执行 Codex implementation、不写业务代码、不 commit/push 的情况下，生成完整 stage plan：dirty 策略、review packet 路线、verification matrix、下一阶段推荐与授权边界。

### Files

- Modify: `tools/codex_workflow_run_tool.py`
- Modify: `tests/tools/test_codex_workflow_run_tool.py`
- Optional create: `agent/codex_workflow_risk_router.py`
- Optional test: `tests/tools/test_codex_workflow_risk_router.py`

### Dry-run output contract

`codex_workflow_run(mode="dry_run")` 必须保证：

- 不调用 `codex_staged_implement`。
- 不清理 cache。
- 不创建 isolated worktree。
- 不写 checkpoint commit。
- 不调用 review guard。
- 不写 ledger、不写 provenance event、不创建/删除/覆盖任何文件；即使 `may_write_files=true` 或存在 standing authorization，`dry_run` 仍必须完全 non-mutating。
- 只返回预览字段，例如 `would_write_ledger`、`would_record_ledger_events`、`ledger_event_preview`；真实 ledger/provenance 记录必须发生在非 `dry_run` 的明确授权模式中。

返回结构新增：

```json
{
  "mode": "dry_run",
  "would_call_staged": false,
  "would_run_review": false,
  "would_run_verification": false,
  "would_create_isolated_worktree": false,
  "would_commit": false,
  "would_push": false,
  "would_deploy_or_restart": false,
  "would_write_ledger": true,
  "would_record_ledger_events": true,
  "ledger_event_preview": [],
  "dirty_ownership": [],
  "cleanup_allowed": false,
  "cleanup_blocking_reasons": [],
  "would_delete_paths": [],
  "would_overwrite_paths": [],
  "blocking_reasons": [],
  "authorization_required": [],
  "recommended_next_stage": {"stage_id": "phase12a0-provenance-contract", "allowed_files": [], "verify_cmd_ids": []}
}
```

### Dirty / ownership / isolation decision table

| Ownership class | Path class | Overlap with allowed scope | Default behavior | Authorization required |
|---|---|---:|---|---|
| `current_session` | docs/source/test | yes | may proceed only through allowlist + hash gate | yes for write |
| `other_known_session` | any | any | preserve; no cleanup/delete/overwrite | user coordination only |
| `unknown_unowned` | `docs/plans/*` | any | preserve; no cleanup/delete/overwrite | explicit re-attribution only |
| `unknown_unowned` | source/test/docs | no | preserve; prefer isolated worktree or stop | yes for isolated write stage |
| `unknown_unowned` | source/test/docs | yes | stop for user / no Codex write | yes |
| `generated_cache` | cache / pyc | no | may auto-clean only when `auto_clean_cache=true` and not dry-run | yes for real clean |
| `review_artifact_current_session` | `/tmp` or allowlisted artifact | no | may clean if hash matches and authorized | yes for real clean |
| `dangerous_conflict` | delete/rename/chmod/submodule/binary/conflict/secrets | any | stop fail-closed | manual inspection |

No row may produce automatic stash/reset/revert/drop/overwrite.

### Tests

- `test_dry_run_never_mutates_files_even_when_write_authorized`
- `test_dry_run_reports_ledger_event_preview_without_writing_ledger`
- `test_dry_run_reports_blocking_overlap_with_unknown_source_dirty`
- `test_dry_run_recommends_isolated_worktree_for_non_overlap_unknown_dirty`
- `test_dry_run_outputs_dirty_ownership_and_cleanup_block_reasons`
- `test_dry_run_never_populates_would_delete_or_overwrite_for_unknown_unowned`
- `test_dry_run_never_sets_commit_push_deploy_restart_true`
- `test_dry_run_includes_authorization_required_for_write_stage`
- `test_dry_run_preserves_leftover_candidate_metadata`

---

## 4. Phase 12C：Review Autopilot

### Objective

在明确授权时自动生成 bounded review packet，并调用现有 `scripts/runtime/codex_review_packet.py` / `scripts/runtime/codex_review_guard.py`。不可新增第二套 review guard。

### Fail-closed states

| Condition | Ledger review verdict | Stage status |
|---|---|---|
| final JSON `verdict=passed`, `must_fix=[]`, no flood/timeout | `passed` | may continue to verification |
| `verdict=failed` or `must_fix` non-empty | `failed` | stop |
| `codex-yuna` missing | `unavailable` | stop |
| timeout | `unavailable` | stop |
| source/diff/json flood | `unavailable` | stop |
| provider 5xx / auth / quota | `unavailable` | stop after one bounded retry max |
| invalid JSON / missing final file | `unavailable` | stop |
| review changed worktree | `failed` | contaminated review; stop |

### Tests

- `test_review_autopilot_uses_existing_packet_and_guard_commands`
- `test_review_unavailable_timeout_is_fail_closed`
- `test_review_invalid_json_is_fail_closed`
- `test_review_pass_requires_no_must_fix`
- `test_review_dirty_after_review_contaminated_stops`
- `test_review_packet_excludes_unknown_dirty_diff`

---

## 5. Phase 12D：Verification Matrix

### Objective

把 risk class 映射到最小可执行命令集合、证据格式和跳过理由，避免“只列 examples”。

| Risk class | Minimum commands | Evidence fields | Skip reason allowed |
|---|---|---|---|
| `docs_only` | `git diff --check -- <docs>` | exit code, changed docs | no tests needed because docs-only |
| `tool_schema` | tool focused pytest; py_compile; diff-check | test count, py_compile paths | no broad tests if only schema/doc not runtime |
| `codex_orchestrator` | workflow tool pytest; py_compile; diff-check; dry-run smoke | dry_run JSON, no mutation proof | no real Codex impl in dry-run |
| `review_guard` | review guard tests; packet smoke; py_compile; diff-check | final JSON schema, flood metadata | provider unavailable can skip external Codex review only with disclosure |
| `gateway_runtime` | gateway focused tests; py_compile; diff-check | route/mode tests, status | no restart unless explicit authorization |
| `compression` | context compressor tests; py_compile; diff-check | token/status tests | no live provider test unless user authorizes |
| `secrets/privacy` | added-line secret scan; redaction tests; focused tests | scan result, redaction cases | cannot skip without blocker |

Verification result schema records `cmd_id`、argv、exit code、bounded stdout/stderr tail、start/end time。失败或 skip 无理由则阻塞下一阶段。

### Tests

- `test_risk_router_maps_codex_orchestrator_to_required_commands`
- `test_verification_failure_blocks_next_stage`
- `test_skipped_verification_requires_reason`
- `test_secret_privacy_risk_cannot_skip_scan`
- `test_gateway_runtime_risk_recommends_no_restart_by_default`

---

## 6. Phase 12E：Bounded Must-fix Loop

### Loop rules

- 最大 2 轮：`max_fix_rounds=2`。
- 每轮：verify review claim → add/update focused regression test if practical → bounded implementation → focused verification → packet-only re-review。
- 任一条件立即停止：
  - must-fix 仍非空且已达上限。
  - review unavailable。
  - verification failed。
  - dirty overlap / allowlist escape。
  - new secret/real-data risk。
  - Codex flood/timeout repeated。
- `suggested_fixes` 默认只记录，不自动实现。

### Tests

- `test_must_fix_loop_stops_after_max_rounds`
- `test_must_fix_loop_does_not_run_without_authorization`
- `test_must_fix_loop_stops_on_review_unavailable`
- `test_must_fix_false_positive_can_be_recorded_with_evidence`
- `test_must_fix_true_positive_requires_regression_test_when_practical`

---

## 7. Phase 12F：Next-stage Recommender

### Objective

ledger、review、verification 都可审计后，给出下一整个大阶段建议，但默认不推进。

Recommendation contract:

```json
{
  "stage_id": "phase12c-review-autopilot",
  "why": "ledger + dry-run planner passed; review integration is next missing loop",
  "allowed_files": ["tools/codex_workflow_run_tool.py", "tests/tools/test_codex_workflow_run_tool.py"],
  "verify_cmd_ids": ["workflow-tool-pytest", "py-compile", "diff-check"],
  "authorization_required": true,
  "non_goals": ["commit", "push", "deploy", "restart", "force-push"]
}
```

### Tests

- `test_recommender_only_recommends_by_default`
- `test_recommender_requires_authorization_to_advance`
- `test_recommender_blocks_when_review_unavailable`
- `test_recommender_blocks_when_verification_failed`
- `test_recommender_never_recommends_deploy_restart_force_push`

---

## 8. Security and safety boundary checklist

- Secrets / real data：never write full env, tokens, provider payloads, user data, raw Codex logs into ledger。
- Dirty worktree：owner/session_id provenance before cleanup；no stash/reset/revert/drop/overwrite；unknown/other-session dirty preserved or isolated。
- Merge/rebase/conflict：fail-closed。
- Force push / deploy / restart：always false by default，separate explicit authorization required。
- Permission expansion：new capability must be allowlisted and documented。
- Allowed files/globs：pre/post evidence；untracked paths visible；new files use `git add -N` before diff-check/review。
- Logs：bounded；raw logs only referenced by redacted path；no giant source/diff injection。
- Codex flood/timeout/provider error：unavailable/takeover candidate only，never pass。
- Concurrency：repo/branch/stage lock prevents simultaneous active autopilot；repo/branch/path/hash provenance gate prevents cross-session cleanup/delete/overwrite。

---

## 9. Implementation order

1. **12A0 first slice**：Ownership/provenance contract + tests。No cleanup/delete/overwrite. No real Codex implementation beyond the bounded candidate patch. No commit/push/deploy/restart。
2. **12A + 12B second slice**：Ledger/resume + dry-run planner，必须引用 12A0 provenance snapshot / event ids。No real Codex implementation execution in dry-run. No commit/push/deploy/restart。
3. **12C third slice**：existing packet/guard integration only；fail-closed unavailable states。
4. **12D fourth slice**：risk/verification matrix。
5. **12E fifth slice**：bounded must-fix loop。
6. **12F sixth slice**：next-stage recommender。

每个 slice：

```text
plan boundary → Codex candidate implementation or Hermes tiny edit → packet-only Codex review → Hermes verification → no commit/push unless explicitly authorized
```

---

## 10. Plan review gate before implementation

实现前必须满足：

```bash
git add -N docs/plans/backend-codex-autopilot-phase12-plan.zh-CN.md
git diff --check -- docs/plans/backend-codex-autopilot-phase12-plan.zh-CN.md
```

以及 packet-only Codex review 返回：

```json
{"verdict": "passed", "must_fix": []}
```

如果 Codex review 为 `failed` 或 `unavailable`，不得启动 Phase 12 实现；先修计划或汇报 unavailable。若 review 指出 ownership/provenance gate 缺失，必须先修 12A0，再复审通过。

---

## 11. Handoff report shape

```text
阶段：计划 / 实现 / review / 验证
Codex implementation：not_run / candidate_diff / trusted_completion / ...
Hermes takeover：yes/no + reason
Codex review：packet_only_passed / packet_only_failed / unavailable + must-fix count
Hermes verification：commands + exact result
Ledger：path redacted, stage_id, status
边界：未 commit / 未 push / 未部署 / 未重启
下一步：推荐阶段 + authorization_required
```
