# Kanban Worker Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Hermes from spawning underspecified coding tasks, isolate each external coding task in its own git worktree, and prove the fix through Hermes' own gateway dispatcher.

**Architecture:** Add one pre-spawn guard to the existing dispatcher and harden the existing user-level `kanban-executor` skill. Reuse SQLite task provenance, `block_task`, git worktrees, the gateway dispatcher, the current 80% CPU quota, and `kanban.max_in_progress: 3`.

**Tech Stack:** Python, SQLite, pytest through `scripts/run_tests.sh`, Markdown skills, git worktree, systemd.

## Global Constraints

- Do not add a new dependency, service, sandbox, or orchestration layer.
- Do not use broad process kills, `rm -rf`, `reset --hard`, or `git add -A`.
- Preserve the user's untracked `/usr/local/lib/hermes-agent/.install_method`.
- Leave T2/PR #12 untouched.
- Verification tasks may open or update PRs but may not merge them.
- Workers must be started by the live Hermes gateway dispatcher, never by a manual worker command.

---

### Task 1: Block underspecified agent tasks before spawn

**Files:**
- Modify: `hermes_cli/kanban_db.py:7640-7735`
- Test: `tests/hermes_cli/test_kanban_core_functionality.py:90-125`

**Interfaces:**
- Consumes: `Task.body`, `Task.created_by`, `block_task(conn, task_id, reason=..., kind="needs_input")`.
- Produces: `dispatch_once(...)` adds the task id to `DispatchResult.auto_blocked`, leaves no worker PID, and never calls `spawn_fn`.

- [ ] **Step 1: Write the failing test**

```python
def test_dispatch_blocks_agent_task_without_body(kanban_home, all_assignees_spawnable):
    spawned = []
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="underspecified",
            body="   ",
            assignee="worker",
            created_by="user",
        )
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id),
        )
        task = kb.get_task(conn, tid)
        assert spawned == []
        assert tid in result.auto_blocked
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert task.worker_pid is None
    finally:
        conn.close()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `scripts/run_tests.sh tests/hermes_cli/test_kanban_core_functionality.py::test_dispatch_blocks_agent_task_without_body -q`

Expected: FAIL because the current dispatcher calls `spawn_fn` and moves the task to `running`.

- [ ] **Step 3: Add the minimal guard**

Change the ready-row query to select `body` and `created_by`, then add this after default-assignee resolution and before profile/spawn checks:

```python
        if row["created_by"] is not None and not (row["body"] or "").strip():
            if dry_run or block_task(
                conn,
                row["id"],
                reason="Task body is required before dispatch",
                kind="needs_input",
            ):
                result.auto_blocked.append(row["id"])
            continue
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
scripts/run_tests.sh   tests/hermes_cli/test_kanban_core_functionality.py::test_dispatch_blocks_agent_task_without_body   tests/hermes_cli/test_kanban_core_functionality.py::test_spawn_failure_auto_blocks_after_limit   tests/hermes_cli/test_kanban_core_functionality.py::test_spawned_event_emitted_with_pid -q
```

Expected: `3 passed`.

- [ ] **Step 5: Lint and commit exact files**

Run:

```bash
ruff check --fix hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_core_functionality.py
ruff format hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_core_functionality.py
pyright hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_core_functionality.py
git add hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_core_functionality.py
git commit -m "fix(kanban): block underspecified agent tasks"
```

### Task 2: Harden the external coding-worker skill

**Files:**
- Modify: `/root/.hermes/skills/kanban-executor/SKILL.md`
- Backup: `/root/.hermes/skills/kanban-executor/SKILL.md.bak-worker-safety-20260728`

**Interfaces:**
- Consumes: `HERMES_KANBAN_TASK`, target repository path, target base ref.
- Produces: a unique `/root/projects/.hermes-worktrees/<task-id>` worktree and exact-file staging rules in every external executor prompt.

- [ ] **Step 1: Run a failing policy check**

Run a Python assertion that requires all four invariants: per-task worktree path, no destructive-guard bypass, exact-file staging, and no test deletion. It must fail against the current skill because line 116 explicitly recommends `shutil.rmtree(...)` and no mandatory worktree rule exists.

- [ ] **Step 2: Back up and minimally edit the skill**

Add one mandatory safety section before executor selection:

```markdown
## Обязательная изоляция и сохранность

- Если `body` пустой — не вызывай исполнителя; заблокируй задачу как `needs_input`.
- Для существующего git-репозитория работай только в `/root/projects/.hermes-worktrees/$HERMES_KANBAN_TASK`, созданном через `git worktree add`; общий checkout не изменяй.
- Не обходи защиту от удаления через Python/`shutil`; не используй `reset --hard` и `git add -A`. Добавляй только перечисленные файлы через `git add -- <paths>`.
- Не удаляй и не ослабляй тесты ради зелёного прогона. Чужое падение зафиксируй в комментарии и заблокируй задачу.
```

Replace transport staging with `git add -- <task-slug>` and delete the line recommending the `shutil.rmtree` bypass.

- [ ] **Step 3: Run the policy check and verify GREEN**

Expected: assertions pass; `rg -n 'shutil\.rmtree|git add -A|reset --hard' /root/.hermes/skills/kanban-executor/SKILL.md` returns no matches.

### Task 3: Deploy and prove the hard guard through the live gateway

**Files:**
- Read: `/etc/systemd/system/hermes-gateway.service`
- Read: `/root/.hermes/config.yaml`
- Read/write: `/root/.hermes/kanban.db` only through Hermes CLI/runtime.

- [ ] **Step 1: Verify the service imports this checkout**

Check `systemctl cat hermes-gateway.service`, `/proc/<gateway-pid>/cwd`, and Python module paths before restarting.

- [ ] **Step 2: Restart only `hermes-gateway.service` and verify health**

Expected: Telegram reconnects; logs contain `kanban dispatcher: max_in_progress=3`; cgroup `cpu.max` remains `80000 100000`.

- [ ] **Step 3: Create a blank canary card, then wait for the gateway dispatcher**

Create an assigned card with `created_by=user`, blank body, and an idempotency key. Do not run `hermes kanban dispatch` manually.

Expected after the normal gateway tick: status `blocked`, `block_kind=needs_input`, no PID, no task run, and no increase in worker count.

### Task 4: Let Hermes launch isolated T1/T5 verification work

**Files:**
- PC repository: `/root/projects/python-path`
- PC worktrees: `/root/projects/.hermes-worktrees/<new-task-id>`

- [ ] **Step 1: Ensure the shared checkout is clean and parked on `main`**

The old T1 worker is already blocked and PR #13 exists. Switch the clean shared checkout from `feat/run-button` to `main`; do not delete its branch or commits.

- [ ] **Step 2: Create two fully specified cards with `--skill kanban-executor`**

T1 card: audit PR #13 against its recovered acceptance criteria; use a unique worktree; run tests; fix only if required; do not merge.

T5 card: inspect merged PR #11 on current `origin/main`; restore independent test coverage and validate safe code-question lookup/fraction logic; use a unique worktree; open a correction PR if needed; do not merge.

Both cards include exact repo, base ref, acceptance criteria, test command, stop rules, and idempotency keys.

- [ ] **Step 3: Wait for Hermes' gateway dispatcher to launch them**

Do not call a worker executable or manual dispatch. Poll task rows and gateway logs until both have PIDs or one is explicitly blocked with a useful reason.

- [ ] **Step 4: Verify isolation and resource bounds**

Confirm:

- both bodies are non-empty;
- worker logs show `kanban-executor` loaded;
- PC `git worktree list` contains distinct task-id paths;
- no worker edits `/root/projects/python-path` directly;
- running kanban workers never exceed three;
- gateway cgroup remains at 80% CPU;
- memory/swap are reported but unchanged by this fix.
