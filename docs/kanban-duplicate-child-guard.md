# kanban-duplicate-child-guard

`kanban-duplicate-child-guard` 是一个只读 Kanban 子任务重复检测器。它面向执行/审计链路中的 audit/review child，先产出 detector + dry-run plan + JSON receipt；当前不实现任何写入型 apply。

## 边界

- 只读读取 Kanban SQLite DB；CLI 默认用 SQLite `mode=ro` 打开数据库。
- 不修改 Hermes core、live profile、gateway、watcher 或服务进程。
- 不创建、归档、阻塞、链接或评论 Kanban task。
- 不读取或输出 token/OAuth/cookie 等凭据。
- 不输出 Telegram/Discord 等 raw 平台定位值；receipt 只记录 alias/环境是否存在。
- 不在 receipt 中输出 task title；title 是用户可控公开文本，可能夹带敏感片段或 raw locator。

## 入口

安装后入口：

```bash
kanban-duplicate-child-guard detect \
  --root-task t_example \
  --actor-profile jiangzuodajiang \
  --receipt-file /tmp/duplicate-child-receipt.json \
  --json
```

也可从源码运行：

```bash
python3 -m hermes_cli.kanban_duplicate_child_guard detect --root-task t_example --json
```

Runtime/profile smoke：

```bash
kanban-duplicate-child-guard doctor --actor-profile jiangzuodajiang --json
```

受控 apply 目前故意关闭：

```bash
kanban-duplicate-child-guard apply --json
# exit 2, no mutations performed
```

## 输入

`detect` 至少需要以下一种定位方式：

- `--root-task <task_id>`：从 root task 读取全部 descendant child graph；可重复。
- `--issue <#N|N>`：按 issue marker 找 root，再读 descendant graph。
- `--pr <N>`：按 PR marker 找 root。
- `--head <sha>`：按 head/commit marker 找 root。
- `--review-scope-key <key>` / `--audit-scope <key>`：按审计范围找 root。

可选：

- `--board <slug>`：指定 Kanban board。
- `--db-path <path>`：显式指定 SQLite DB，仍以只读模式打开。
- `--actor-profile <profile>`：写入 runtime/profile-smoke evidence，验证 `HERMES_PROFILE` 是否与预期 actor 一致。
- `--receipt-file <path>`：写 JSON receipt 文件。

## 分组规则

detector 只聚合疑似 audit/review child。判定依据是 title/body/comment/run metadata/workflow/current_step 中出现审计语义；assignee 单独叫 `reviewer` 不会被当作审计 child。

- 中文：审计、审核、复审、终审；
- 英文整词：review、re-review、audit、auditor、closeout、final-review。

进入重复分组前，child 还必须具备可验证 marker：

- subject marker：至少一个 `issues` / `prs` / `head_shas`；
- scope marker：至少一个 `review_scope_key` / `audit_scope` / `phase_scope`。

缺 marker 的 review-like child 会进入 `insufficiently_marked_review_children`，不参与 duplicate 判定，避免把两个不完整 child 用 `unknown_*` 误聚合。

重复分组 key：

- `issues`
- `prs`
- `head_shas`
- `assignee`
- `review_scope_key`
- `audit_scope`
- `phase_scope`
- `workflow_template_id`
- `current_step_key`

因此同一个 issue 下不同 phase/scope 不会被误判为重复。例如 `#159-A` 与 `#159-B` 如果 `review_scope_key` 不同，会落到不同 group。

## 状态分类

- `unrun_duplicate`：同组 child 全部处于 `triage/todo/scheduled/ready`，且没有 task_runs；dry-run plan 给出 canonical child 与冗余 child 的安全收敛建议。
- `historical_duplicate`：同组 child 全部 terminal（`done/archived` 或 `completed_at` 已有）；只记录，不倒改历史。
- `active_duplicate_manual_review`：包含 running child；不做自动收敛，只提示人工判断。
- `mixed_duplicate_manual_review`：状态混合；不做自动收敛，只提示人工判断。

当前所有 plan action 都满足：

```json
{"destructive": false, "mutation_performed": false}
```

## Receipt schema

顶层字段：

```json
{
  "schema": "kanban-duplicate-child-guard:receipt:v1",
  "mode": "dry_run",
  "ok": true,
  "generated_at": "UTC timestamp",
  "input": {},
  "board": {"slug": "...", "read_only": true},
  "graph": {},
  "detector": {
    "grouping_fields": [],
    "duplicate_groups": [],
    "non_duplicate_review_children": [],
    "insufficiently_marked_review_children": [],
    "non_review_children": []
  },
  "dry_run_plan": {
    "apply_enabled": false,
    "apply_supported": false,
    "summary": "...",
    "actions": []
  },
  "receipt_contract": {},
  "runtime_profile_smoke": {},
  "public_safety": {
    "no_mutations_performed": true,
    "write_targets": [],
    "destructive_apply_default_enabled": false,
    "raw_platform_locator_included": false,
    "user_controlled_titles_included": false,
    "credentials_or_tokens_touched": false
  }
}
```

`#159-B` 与 `#157` 这类下游消费者应读取：

- `detector.duplicate_groups[].group_key`
- `detector.duplicate_groups[].group_state`
- `detector.duplicate_groups[].members[]`（只含 task_id/status/assignee/run 状态等稳定字段，不含 title）
- `detector.insufficiently_marked_review_children[]`
- `dry_run_plan.actions[]`
- `public_safety.no_mutations_performed`

不要从 receipt 推断已经发生过 Kanban 写入。

## 测试 fixture

`tests/fixtures/kanban_duplicate_child_guard_issue155.json` 覆盖 #155 历史重复 child 类场景：同 issue、同 PR/head、同 assignee、同 review/audit scope 的两个未运行审计 child。测试会加载该 fixture 并验证输出 `unrun_duplicate` 与 dry-run actions。

目标测试：

```bash
python3 -m pytest tests/hermes_cli/test_kanban_duplicate_child_guard.py -q -o 'addopts='
```
