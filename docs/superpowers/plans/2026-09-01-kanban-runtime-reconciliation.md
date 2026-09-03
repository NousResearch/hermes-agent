# Hermes 看板运行态一致性实施计划

> **执行者要求：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项完成本计划。每个步骤用复选框记录状态。

**目标：** 让 Hermes 能可靠关联任务、当前 run 和真实 Worker，安全诊断及修复幽灵状态，并让每 profile 并发上限覆盖所有看板。

**架构：** 沿用现有 SQLite claim/run 模型，不增加表或列。PID 登记改成带 run ID 和 claim lock 的原子 compare-and-swap；诊断层只检查数据库结构；CLI `reconcile` 额外读取本机进程表，默认只读，只有“确认无同任务活进程”的状态才允许修复。跨看板 profile 计数复用已有 board 枚举与数据库路径去重逻辑。

**技术栈：** Python 3.12、SQLite/WAL、argparse、psutil 7.2.2、pytest、ruff、py_compile、Code Forge。

**规格：** `docs/superpowers/specs/2026-09-01-kanban-runtime-reconciliation-design.md`

## 全局约束

- 只在 `fix/kanban-runtime-reconciliation` 独立 worktree 工作；不得在 main/master 工作区修改。
- 不新增数据库表或列，不新增用户可见 `HERMES_*` 环境变量，不新增模型工具。
- 正式 Worker 仍只能由 dispatcher 启动；缺少 task/run/claim/db 完整身份时不得登记 PID。
- 未登记活进程、重复 Worker、登记 PID 命令行不匹配、远端或不可读进程全部 fail closed。
- 不使用 `pkill -f`、`killall`，不按任务名批量终止进程。
- `reconcile` 默认只读；`--fix` 每次写入后必须重新读取任务、run 和进程快照。
- 逻辑改动的新测试必须做真实修复点缺陷注入：同一测试先 FAIL，恢复修复后 PASS。
- 至少执行一次真实 SQLite 文件、真实子进程和真实 CLI 路径。
- Python 改动必须通过 `py_compile` 和 `ruff`；不得新增告警。
- 最终提交使用本机 Git 身份、GPG 签名和 `Signed-off-by`，消息格式 `<subsystem>/<case>: <摘要>`，说明 WHY。
- 不 push，不重启本机生产 gateway，不把 feature 分支用于本机生产环境。
- 逻辑改动必须由独立 Reviewer 完成连续三轮 Forge Review；任何确认 finding 修复后 clean 计数从零开始。

---

## 文件职责

| 文件 | 职责 |
|---|---|
| `hermes_cli/kanban_db.py` | 受 pin 的 PID 原子登记、运行态快照/安全 CAS 修复、跨看板 profile 计数。 |
| `tools/kanban_tools.py` | 从受控 Worker 环境登记当前进程，并在心跳前做幂等补登记。 |
| `cli.py` | quiet 单次 Worker 在模型初始化前调用启动登记。 |
| `hermes_cli/kanban_diagnostics.py` | PID 缺失、run 不一致和首次心跳缺失的只读结构诊断。 |
| `hermes_cli/kanban_reconcile.py` | 进程枚举、运行态分类、允许修复判定与后态复核；避免继续扩大 `kanban.py`。 |
| `hermes_cli/kanban.py` | `reconcile` 参数、board 范围校验、文本/JSON 输出和新模块编排。 |
| `tests/hermes_cli/test_kanban_worker_registration.py` | PID 登记、Worker 环境桥和真实子进程回归。 |
| `tests/hermes_cli/test_kanban_diagnostics.py` | 三类运行身份诊断的规则级测试。 |
| `tests/hermes_cli/test_kanban_review_surfaces.py` | CLI 与 Dashboard 共享诊断引擎的集成验证。 |
| `tests/hermes_cli/test_kanban_reconcile_cli.py` | 进程分类、fail-closed、CAS 修复、all-board 及真实 CLI 测试。 |
| `tests/hermes_cli/test_kanban_per_profile_cap.py` | 跨看板 profile 上限的 dry-run 与真实 claim 测试。 |
| `website/docs/user-guide/features/kanban.md` | 用户命令、分类语义、默认只读和并发口径。 |
| `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features/kanban.md` | 对应中文用户文档。 |
| `.planning/reviews/kanban-runtime-reconciliation-round-{1,2,3}.md` | 三轮独立 Forge Review 证据。 |
| `.planning/verification/kanban-runtime-reconciliation.md` | FAIL/PASS 缺陷注入、真实路径、静态检查和最终验收输出。 |

---

### 任务 1：建立受 pin 的 Worker PID 登记

**文件：**
- 修改：`hermes_cli/kanban_db.py:9340-9375`
- 新建：`tests/hermes_cli/test_kanban_worker_registration.py`

**接口：**
- 产出：
  ```python
  def register_worker_pid(
      conn: sqlite3.Connection,
      task_id: str,
      pid: int,
      *,
      expected_run_id: int,
      expected_claim_lock: str,
      source: str,
  ) -> str
  ```
- 返回值只允许 `"registered"`、`"already_registered"`、`"rejected"`。
- `_set_worker_pid(...)` 保留为内部兼容入口，但签名改为接收 `Task` 或显式 run/claim pin，最终调用 `register_worker_pid`。

- [ ] **步骤 1：写失败测试，固定原子登记契约**

在新测试文件中建立临时 `HERMES_HOME`、真实 SQLite 数据库和 `claim_task` 产生的真实 run。至少加入：

```python
def test_register_worker_pid_updates_task_and_current_run_once(conn):
    tid = kb.create_task(conn, title="worker", assignee="coder")
    claimed = kb.claim_task(conn, tid, claimer="host:claim")
    assert claimed is not None

    result = kb.register_worker_pid(
        conn,
        tid,
        43210,
        expected_run_id=claimed.current_run_id,
        expected_claim_lock=claimed.claim_lock,
        source="dispatcher",
    )
    again = kb.register_worker_pid(
        conn,
        tid,
        43210,
        expected_run_id=claimed.current_run_id,
        expected_claim_lock=claimed.claim_lock,
        source="worker_start",
    )

    task_row = conn.execute(
        "SELECT worker_pid FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    run_row = conn.execute(
        "SELECT worker_pid FROM task_runs WHERE id=?",
        (claimed.current_run_id,),
    ).fetchone()
    spawned = [e for e in kb.list_events(conn, tid) if e.kind == "spawned"]
    assert result == "registered"
    assert again == "already_registered"
    assert task_row["worker_pid"] == run_row["worker_pid"] == 43210
    assert len(spawned) == 1
    assert spawned[0].payload == {"pid": 43210, "source": "dispatcher"}
```

另加四个测试：错误 run ID、错误 claim lock、已有不同 PID、run 已结束。每个都断言返回 `rejected`，任务行和 run 行没有变化，也没有新增 `spawned` 事件。

- [ ] **步骤 2：运行测试，确认 RED**

运行：

```bash
python -m pytest tests/hermes_cli/test_kanban_worker_registration.py -q
```

预期：FAIL，`kanban_db` 尚无 `register_worker_pid`。

- [ ] **步骤 3：实现最小原子登记**

实现时在一个 `write_txn` 中读取并锁定以下事实：

```sql
SELECT t.status,
       t.current_run_id,
       t.claim_lock AS task_claim_lock,
       t.worker_pid AS task_worker_pid,
       r.status AS run_status,
       r.ended_at,
       r.claim_lock AS run_claim_lock,
       r.worker_pid AS run_worker_pid
  FROM tasks t
  LEFT JOIN task_runs r ON r.id = t.current_run_id
 WHERE t.id = ?
```

接受条件必须同时满足：任务 `running`、`current_run_id` 等于 pin、run `running` 且未结束、任务和 run claim 都等于 `expected_claim_lock`。两个 PID 都为空时同时写入；两个 PID 都等于传入 PID 时幂等成功；其他组合全部拒绝。首次写入才追加 `spawned`。

更新 ready/review 两个 spawn 调用点：

```python
registration = register_worker_pid(
    conn,
    claimed.id,
    int(pid),
    expected_run_id=int(claimed.current_run_id),
    expected_claim_lock=str(claimed.claim_lock),
    source="dispatcher",
)
if registration == "rejected":
    raise RuntimeError("spawned worker PID could not be bound to the claimed run")
```

不得在拒绝时覆盖旧 PID。

- [ ] **步骤 4：运行目标与既有生命周期测试，确认 GREEN**

运行：

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_worker_registration.py \
  tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py \
  tests/hermes_cli/test_kanban_worker_lifecycle_hooks.py \
  tests/hermes_cli/test_kanban_parent_reopen_invalidation.py -q
```

预期：全部 PASS。

- [ ] **步骤 5：在真实修复点做缺陷注入**

临时把 `register_worker_pid` 的 claim-lock 比较撤回，只保留 run ID；运行：

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_worker_registration.py::test_register_worker_pid_rejects_wrong_claim_lock -q
```

预期：FAIL，错误 claim 被接受或数据被写入。恢复真实比较后用同一命令确认 PASS。把 FAIL/PASS 原文和临时 diff 位置写入 `.planning/verification/kanban-runtime-reconciliation.md`。

- [ ] **步骤 6：提交本任务**

```bash
git add hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_worker_registration.py \
  .planning/verification/kanban-runtime-reconciliation.md
git commit -S -s -m "kanban/worker-identity: bind worker PIDs to the claimed run"
```

---

### 任务 2：让受控 Worker 启动时自登记

**文件：**
- 修改：`tools/kanban_tools.py:285-357`
- 修改：`cli.py:22033-22078`
- 修改：`tests/hermes_cli/test_kanban_worker_registration.py`
- 修改：`tests/cli/test_single_query_session_finalize.py`

**接口：**
- 消费：任务 1 的 `register_worker_pid(...) -> str`。
- 产出：
  ```python
  def register_current_worker_from_env(*, source: str = "worker_start") -> str | None
  ```
  返回 `None` 表示缺少完整受控环境或连接失败；其他返回值透传 DB 登记结果。

- [ ] **步骤 1：写环境桥失败测试**

加入以下契约：

```python
def test_register_current_worker_requires_complete_identity(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_demo")
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    assert kt.register_current_worker_from_env() is None
```

再加入真实数据库测试：设置 `HERMES_KANBAN_DB`、task、run、claim，调用后断言 task/run PID 等于 `os.getpid()`。设置过期 run 或 claim 时断言返回 `rejected` 且不覆盖。

- [ ] **步骤 2：运行桥测试，确认 RED**

```bash
python -m pytest tests/hermes_cli/test_kanban_worker_registration.py -q
```

预期：FAIL，缺少 `register_current_worker_from_env`。

- [ ] **步骤 3：实现桥和 heartbeat 补登记**

在 `tools/kanban_tools.py` 中：

1. 要求 task、run ID、claim lock、DB pin 全部存在；
2. run ID 必须能解析成正整数；
3. 调用 `_connect()` 后执行 `kb.register_worker_pid(..., pid=os.getpid(), source=source)`；
4. 捕获异常并记录 debug，不向 agent loop 抛出；
5. `heartbeat_current_worker_from_env()` 在 `heartbeat_claim` 前调用一次 `register_current_worker_from_env(source="heartbeat_repair")`。

- [ ] **步骤 4：把启动登记接入 quiet Worker 路径**

在 `cli.py` 的 `_claim_active_session` 成功后、读取任务 body 和 `_ensure_runtime_credentials()` 之前调用：

```python
if os.environ.get("HERMES_KANBAN_TASK"):
    try:
        from tools.kanban_tools import register_current_worker_from_env
        register_current_worker_from_env(source="worker_start")
    except Exception as exc:
        logger.debug("kanban worker PID registration failed: %s", exc)
```

不得因登记失败阻止 Worker 启动；冲突会由诊断和 reconcile fail closed 暴露。

在 `tests/cli/test_single_query_session_finalize.py` 的 fake quiet CLI 流程中 patch 登记 helper，断言它发生在 credentials/model 初始化前。

- [ ] **步骤 5：运行目标测试，确认 GREEN**

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_worker_registration.py \
  tests/cli/test_single_query_session_finalize.py \
  tests/cron/test_cron_kanban_env_isolation.py -q
```

预期：全部 PASS，cron/delegate 隔离不回归。

- [ ] **步骤 6：真实子进程联调**

测试必须启动真实 Python 子进程，不 mock PID：父测试建立并 claim 真实临时数据库；子进程继承四个 `HERMES_KANBAN_*` 变量，执行：

```python
from tools.kanban_tools import register_current_worker_from_env
import os
assert register_current_worker_from_env() == "registered"
print(os.getpid())
```

父进程读取 stdout PID 和数据库 task/run PID，三者必须相等。

- [ ] **步骤 7：缺陷注入**

临时删除或绕过 `cli.py` 的启动登记调用，执行启动路径集成测试：

```bash
python -m pytest \
  tests/cli/test_single_query_session_finalize.py::test_quiet_kanban_worker_registers_pid_before_credentials -q
```

预期：FAIL。恢复调用后同一测试 PASS，并把两次输出写入验证文档。

- [ ] **步骤 8：提交本任务**

```bash
git add cli.py tools/kanban_tools.py \
  tests/hermes_cli/test_kanban_worker_registration.py \
  tests/cli/test_single_query_session_finalize.py \
  .planning/verification/kanban-runtime-reconciliation.md
git commit -S -s -m "kanban/worker-start: repair missing PID registration in the child"
```

---

### 任务 3：增加运行身份结构诊断

**文件：**
- 修改：`hermes_cli/kanban_diagnostics.py:1081-1123`
- 修改：`tests/hermes_cli/test_kanban_diagnostics.py`
- 修改：`tests/hermes_cli/test_kanban_review_surfaces.py`

**接口：**
- 消费：现有 `compute_task_diagnostics(task, events, runs, now, config, graph)`。
- 产出三个 kind：
  - `running_worker_pid_missing`
  - `running_worker_run_mismatch`
  - `running_worker_heartbeat_missing`

- [ ] **步骤 1：扩充测试 fixture**

把 `_task` 默认字段补齐：

```python
"current_run_id": None,
"claim_lock": None,
"worker_pid": None,
"last_heartbeat_at": None,
"started_at": None,
```

把 `_run` 扩充为能覆盖：`status`、`claim_lock`、`worker_pid`、`last_heartbeat_at`、`started_at`、`ended_at`。

- [ ] **步骤 2：写诊断 RED 测试**

至少加入：

```python
def test_running_pid_missing_after_launch_grace_is_error():
    now = 10_000
    task = _task(
        status="running",
        started_at=now - 31,
        current_run_id=7,
        claim_lock="host:1",
        worker_pid=None,
    )
    runs = [_run(
        run_id=7,
        status="running",
        claim_lock="host:1",
        worker_pid=None,
        started_at=now - 31,
        ended_at=None,
    )]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    diag = next(d for d in diags if d.kind == "running_worker_pid_missing")
    assert diag.severity == "error"
    assert all(a.kind != "reclaim" for a in diag.actions)
```

另测：30 秒内不报；current run 不存在；run 已结束；task/run claim 不同；task/run PID 不同；120 秒无首次心跳；119 秒不报；有心跳不报；终态不报。

- [ ] **步骤 3：运行测试，确认 RED**

```bash
python -m pytest tests/hermes_cli/test_kanban_diagnostics.py -q
```

预期：新 kind 缺失导致 FAIL。

- [ ] **步骤 4：实现单一运行身份规则**

优先实现一个 `_rule_running_worker_identity(...)`，一次读取 task 和 current run 并按严重程度返回 0 到多个诊断；不要让三个规则重复查找 run。run 匹配使用 `_task_field` 兼容 dataclass、dict、sqlite row。

动作只给 `cli_hint`：

```python
DiagnosticAction(
    kind="cli_hint",
    label=f"Inspect worker identity: hermes kanban reconcile {task_id}",
    payload={"command": f"hermes kanban reconcile {task_id}"},
    suggested=True,
)
```

这里不能建议 `reclaim`。

将规则放到 `_RULES` 的运行故障靠前位置，并把三个 kind 加入 `DIAGNOSTIC_KINDS`。阈值作为模块常量：

```python
RUNNING_PID_GRACE_SECONDS = 30
RUNNING_FIRST_HEARTBEAT_GRACE_SECONDS = 120
```

- [ ] **步骤 5：验证 CLI 与 Dashboard 共用结果**

在 `test_kanban_review_surfaces.py` 新增真实 SQLite 场景：claim 后把 task/run 的 `started_at` 调整到 31 秒前，保持 PID 空；分别执行：

```python
json.loads(kc.run_slash(f"diagnostics --task {tid} --json"))
_compute_task_diagnostics(conn, task_ids=[tid])
```

两边必须都含 `running_worker_pid_missing`，severity 和 data 相同。

- [ ] **步骤 6：运行目标测试，确认 GREEN**

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_diagnostics.py \
  tests/hermes_cli/test_kanban_review_surfaces.py -q
```

预期：全部 PASS。

- [ ] **步骤 7：缺陷注入**

临时从 `_RULES` 删除 `_rule_running_worker_identity`，执行：

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_diagnostics.py::test_running_pid_missing_after_launch_grace_is_error -q
```

预期：FAIL；恢复后同一命令 PASS。记录输出。

- [ ] **步骤 8：提交本任务**

```bash
git add hermes_cli/kanban_diagnostics.py \
  tests/hermes_cli/test_kanban_diagnostics.py \
  tests/hermes_cli/test_kanban_review_surfaces.py \
  .planning/verification/kanban-runtime-reconciliation.md
git commit -S -s -m "kanban/diagnostics: expose inconsistent running worker identity"
```

---

### 任务 4：实现只读优先的进程核对与安全修复

**文件：**
- 新建：`hermes_cli/kanban_reconcile.py`
- 修改：`hermes_cli/kanban_db.py:8691-8778`
- 修改：`hermes_cli/kanban.py:536-585, 1063-1201`
- 新建：`tests/hermes_cli/test_kanban_reconcile_cli.py`
- 修改：`tests/gateway/test_kanban_reconcile_orphans.py`

**接口：**
- `kanban_reconcile.py` 产出：
  ```python
  @dataclass(frozen=True)
  class ReconcileFinding:
      board: str
      db_path: str
      task_id: str
      classification: str
      task_status: str
      current_run_id: int | None
      registered_pid: int | None
      matching_pids: tuple[int, ...]
      fix_allowed: bool
      fixed: bool = False
      detail: str = ""

  def inspect_task_runtime(
      conn: sqlite3.Connection,
      task: kb.Task,
      *,
      board: str,
      process_snapshot: "ProcessSnapshot | None" = None,
  ) -> ReconcileFinding

  def reconcile_board(
      *,
      board: str,
      task_id: str | None,
      fix: bool,
  ) -> list[ReconcileFinding]
  ```
- `kanban_db.py` 产出：
  ```python
  def reconcile_running_task_if_unchanged(
      conn: sqlite3.Connection,
      task_id: str,
      *,
      expected_run_id: int | None,
      expected_claim_lock: str | None,
      expected_worker_pid: int | None,
      reason: str,
  ) -> bool
  ```

- [ ] **步骤 1：写分类 RED 测试**

使用真实 SQLite 和 monkeypatch 的进程快照先固定纯分类：

- healthy；
- dead_registered_worker；
- orphaned_claim_no_process；
- missing_pid_no_process；
- live_process_unregistered；
- duplicate_live_workers；
- registered_pid_mismatch；
- remote_or_unreadable。

精确命令行匹配函数只能接受 argv 中连续 token：

```python
("work", "kanban", "task", task_id)
```

不能使用字符串 substring；`t_ab` 不得匹配 `t_abc`。输出不得包含完整 argv。

- [ ] **步骤 2：运行分类测试，确认 RED**

```bash
python -m pytest tests/hermes_cli/test_kanban_reconcile_cli.py -q
```

预期：模块不存在或接口缺失导致 FAIL。

- [ ] **步骤 3：实现一次性进程快照**

使用 `psutil.process_iter(["pid", "cmdline"])` 枚举一次，构建只含 PID 与 argv token 的内部快照。分别处理：

- `NoSuchProcess`：忽略，进程已消失；
- `AccessDenied`：记录 unreadable PID，使相关任务分类为 `remote_or_unreadable`；
- 其他 `psutil.Error`/`OSError`：同样 fail closed。

不得把 argv 放入 `ReconcileFinding.detail` 或 JSON。

- [ ] **步骤 4：写安全修复 RED 测试**

用真实 `sleep` 或真实 Python 子进程构造：

1. `missing_pid_no_process + --fix` 可回到 `ready`，run 关闭为 `reclaimed`；
2. 一个命令行含精确任务 token 的未登记活进程时，`--fix` 不修改；
3. 两个匹配进程时不修改；
4. 登记 PID 活着但命令行属于其他任务时不修改；
5. 扫描后改变 claim lock，CAS 修复返回 false；
6. 修复后再次 inspect，结果不再是 running 异常。

真实进程用 `sys.executable -c "import time; time.sleep(30)" work kanban task <id>`，在 `finally` 中只 terminate/kill 测试自己创建的 `Popen` 对象。

- [ ] **步骤 5：实现 CAS 修复 primitive**

`reconcile_running_task_if_unchanged` 在一个 `write_txn` 内用 NULL-safe 条件核对 task ID、status、current run、claim lock 和 worker PID。成功时：

- task 回到 `_retry_status_for_run(conn, task_id)`；
- 清空 claim、PID、heartbeat；
- `_end_run(... outcome="reclaimed", status="reclaimed")`；
- 插入 dispatcher comment；
- 追加 `reconciled` 事件，payload 带 reason 和旧快照；
- 返回 True。

`reconcile_orphaned_running` 改为调用该 primitive，保持已有 dispatcher 自动恢复行为和现有测试。

- [ ] **步骤 6：实现 `reconcile` CLI**

argparse：

```python
p_reconcile = sub.add_parser(
    "reconcile",
    help="Inspect task/run/process identity; read-only unless --fix is passed",
)
p_reconcile.add_argument("task_id", nargs="?", default=None)
p_reconcile.add_argument("--fix", action="store_true")
p_reconcile.add_argument("--all-boards", action="store_true")
p_reconcile.add_argument("--json", action="store_true")
```

约束：

- `--all-boards` 与顶层 `--board` 同时出现返回 2；
- 无 `--all-boards` 时只查显式或当前 board；
- 指定 task 不存在返回 1；
- JSON 每项固定包含 board、db_path、task_id、classification、registered_pid、matching_pids、fix_allowed、fixed、detail；
- 文本输出不打印命令行；
- `--fix` 遇到 fail-closed 分类时返回非零，并明确“no state changed”；
- 把 `reconcile` 加入 delegated child mutation denylist，因为 `--fix` 可写。

- [ ] **步骤 7：运行单元与已有 orphan 测试，确认 GREEN**

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_reconcile_cli.py \
  tests/gateway/test_kanban_reconcile_orphans.py \
  tests/hermes_cli/test_kanban_cli.py -q
```

预期：全部 PASS。

- [ ] **步骤 8：真实 CLI/SQLite/进程 smoke**

在临时 `HERMES_HOME` 中用真正入口完成：

```bash
HERMES_HOME="$tmp_home" python -m hermes_cli.main kanban init
HERMES_HOME="$tmp_home" python -m hermes_cli.main kanban create \
  "reconcile smoke" --assignee default
```

测试 helper 将该任务 claim 后清空 PID并回拨 `started_at`，再启动匹配任务 ID 的真实 sleep 进程。执行：

```bash
HERMES_HOME="$tmp_home" python -m hermes_cli.main kanban reconcile "$tid" --fix --json
```

预期：非零退出、classification=`live_process_unregistered`、`fixed=false`，数据库仍为 running。精确结束测试进程后重跑，预期 classification=`missing_pid_no_process`、`fixed=true`，后态不再 running。

- [ ] **步骤 9：缺陷注入**

临时把 `live_process_unregistered` 的 `fix_allowed` 改为 True，执行：

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_reconcile_cli.py::test_fix_refuses_live_unregistered_worker -q
```

预期：FAIL，任务被错误重排。恢复 fail-closed 条件后同一命令 PASS。记录 FAIL/PASS 原文。

- [ ] **步骤 10：提交本任务**

```bash
git add hermes_cli/kanban_reconcile.py hermes_cli/kanban_db.py \
  hermes_cli/kanban.py tests/hermes_cli/test_kanban_reconcile_cli.py \
  tests/gateway/test_kanban_reconcile_orphans.py \
  .planning/verification/kanban-runtime-reconciliation.md
git commit -S -s -m "kanban/reconcile: refuse ambiguous worker-state repair"
```

---

### 任务 5：让每 profile 上限覆盖所有看板

**文件：**
- 修改：`hermes_cli/kanban_db.py:9732-9794, 10089-10108`
- 修改：`tests/hermes_cli/test_kanban_per_profile_cap.py`

**接口：**
- 产出：
  ```python
  def count_running_tasks_by_assignee_other_boards(
      board: Optional[str] = None,
  ) -> dict[str, int]
  ```

- [ ] **步骤 1：写跨看板 RED 测试**

扩充 fixture，使用 `kb.create_board("alpha")` 和 `kb.create_board("beta")`。在 alpha claim 一个 `coder`；beta 建一个 `coder` ready 和一个 `reviewer` ready。测试：

```python
with kb.connect_closing(board="beta") as conn:
    result = kb.dispatch_once(
        conn,
        board="beta",
        spawn_fn=_fake_spawn,
        dry_run=True,
        max_in_progress_per_profile=1,
    )

assert [row[1] for row in result.spawned] == ["reviewer"]
assert result.skipped_per_profile_capped == [(coder_tid, "coder", 1)]
```

另加真实 `dry_run=False` 测试，确认 beta 的 coder 保持 ready，reviewer 转为 running。加同 DB path override 去重测试和某个其他 board 打开失败时 fail-open + warning 的测试。

- [ ] **步骤 2：运行测试，确认 RED**

```bash
python -m pytest tests/hermes_cli/test_kanban_per_profile_cap.py -q
```

预期：beta 错误派发第二个 coder。

- [ ] **步骤 3：实现跨看板 assignee 汇总**

复用 `count_running_tasks_other_boards` 的路径规范化和 board 枚举，查询：

```sql
SELECT assignee, COUNT(*) AS n
  FROM tasks
 WHERE status = 'running' AND assignee IS NOT NULL
 GROUP BY assignee
```

单个 board 失败时 `warning`，然后继续其他 board。`HERMES_KANBAN_DB` 指向同一文件时不重复计数。

在 `_per_profile_running` 初始化后合并结果：

```python
for assignee, count in count_running_tasks_by_assignee_other_boards(board).items():
    _per_profile_running[assignee] = (
        _per_profile_running.get(assignee, 0) + count
    )
```

- [ ] **步骤 4：运行 profile 与 host cap 测试，确认 GREEN**

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_per_profile_cap.py \
  tests/hermes_cli/test_kanban_host_cap.py -q
```

预期：全部 PASS。

- [ ] **步骤 5：缺陷注入**

临时删除其他看板计数的 merge，执行：

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_per_profile_cap.py::test_per_profile_cap_counts_other_boards -q
```

预期：FAIL，第二个 coder 被派发。恢复 merge 后同一命令 PASS，记录证据。

- [ ] **步骤 6：提交本任务**

```bash
git add hermes_cli/kanban_db.py \
  tests/hermes_cli/test_kanban_per_profile_cap.py \
  .planning/verification/kanban-runtime-reconciliation.md
git commit -S -s -m "kanban/profile-cap: count active workers across boards"
```

---

### 任务 6：更新用户文档与本地标准流程

**文件：**
- 修改：`website/docs/user-guide/features/kanban.md:790-796, 1151-1158`
- 修改：`website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features/kanban.md`
- 修改（Hermes skills repo，不纳入上游源码提交）：`/home/houminxi/code/hermes/skills/software-development/hermes-kanban-orchestration/SKILL.md`
- 修改（Hermes skills repo，不纳入上游源码提交）：`/home/houminxi/code/hermes/skills/software-development/hermes-kanban-orchestration/references/worker-identity-and-cross-board-capacity.md`

**接口：**
- 文档必须描述实际命令，不得引用尚未实现的参数。
- 本地 skill 与上游源码分仓处理；不得把 skill 文件复制进 hermes-agent worktree。

- [ ] **步骤 1：更新英文用户文档**

加入一个“Worker identity and reconciliation”小节，明确：

```text
hermes kanban --board <slug> reconcile [task_id] [--json]
hermes kanban --board <slug> reconcile [task_id] --fix [--json]
hermes kanban reconcile --all-boards [--fix] [--json]
```

说明默认只读、哪些分类 fail closed，以及 `max_in_progress_per_profile` 是跨所有未归档 board 的 host-level profile cap。把 `spawned` payload 更新为 `{pid, source}`。

- [ ] **步骤 2：更新中文文档**

保持与英文命令和语义一致。不要逐字机翻；用“核对”“未登记活进程”“拒绝修改”表达实际操作结果。

- [ ] **步骤 3：更新本地 skill**

用 `skill_manage` patch：

1. 在 SKILL 主流程中把正式 Worker 限定为 dispatcher 启动；
2. 汇报后台任务前必须核对 task/run/process；
3. `running + worker_pid=NULL` 不得直接 reclaim；
4. 所有状态变更显式 `--board`；
5. profile 容量跨 board 核算；
6. 引用更新后的 `worker-identity-and-cross-board-capacity.md`。

引用文件加入 `reconcile` 分类表和 fail-closed 操作顺序。所有中文文字按 `humanizer-zh` 自查。

- [ ] **步骤 4：文档一致性检查**

运行：

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_reconcile_cli.py \
  tests/hermes_cli/test_kanban_per_profile_cap.py -q
```

再搜索命令拼写和旧口径：

```bash
rg -n "kanban reconcile|max_in_progress_per_profile|spawned.*source" \
  website/docs/user-guide/features/kanban.md \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features/kanban.md
```

预期：命令与 parser 一致，文档不再称 profile cap 为单 board 口径。

- [ ] **步骤 5：提交上游文档；记录本地 skill diff**

```bash
git add website/docs/user-guide/features/kanban.md \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features/kanban.md
git commit -S -s -m "docs/kanban-recovery: document safe worker reconciliation"
```

本地 skill 单独在 skills repo 按其仓库规则提交；若用户未要求提交该仓，保留清晰 diff 并在交付报告中列出。

---

### 任务 7：完整验证、独立评审与提交卫生

**文件：**
- 修改：`.planning/verification/kanban-runtime-reconciliation.md`
- 新建：`.planning/reviews/kanban-runtime-reconciliation-round-1.md`
- 新建：`.planning/reviews/kanban-runtime-reconciliation-round-2.md`
- 新建：`.planning/reviews/kanban-runtime-reconciliation-round-3.md`

**接口：**
- 验证文档按 done-condition 记录命令、提交 SHA、源码位置、stub 边界、FAIL 原文和 PASS 原文。
- 每轮 Reviewer 报告必须写明 reviewed range、代码图谱更新结果、Forge job/receipt、finding 处置和 clean 计数。

- [ ] **步骤 1：更新代码图谱**

先执行 `build_or_update_graph_tool(repo_root=<worktree>, base=origin/main)`。若 300 秒超时，记录真实错误；随后对主仓图谱执行 stats 和 semantic search，报告 `head_matches_build`，不能把超时写成“无图谱”。Reviewer 任务体必须写“先更新代码图谱再评审”。

- [ ] **步骤 2：运行 Python 静态门禁**

```bash
python -m py_compile \
  hermes_cli/kanban_db.py \
  hermes_cli/kanban_reconcile.py \
  hermes_cli/kanban.py \
  hermes_cli/kanban_diagnostics.py \
  tools/kanban_tools.py \
  cli.py

ruff check \
  hermes_cli/kanban_db.py \
  hermes_cli/kanban_reconcile.py \
  hermes_cli/kanban.py \
  hermes_cli/kanban_diagnostics.py \
  tools/kanban_tools.py \
  cli.py \
  tests/hermes_cli/test_kanban_worker_registration.py \
  tests/hermes_cli/test_kanban_reconcile_cli.py \
  tests/hermes_cli/test_kanban_diagnostics.py \
  tests/hermes_cli/test_kanban_per_profile_cap.py \
  tests/hermes_cli/test_kanban_review_surfaces.py
```

预期：退出码 0，无新增 error/warning。

- [ ] **步骤 3：运行完整相关测试**

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_worker_registration.py \
  tests/hermes_cli/test_kanban_reconcile_cli.py \
  tests/hermes_cli/test_kanban_diagnostics.py \
  tests/hermes_cli/test_kanban_review_surfaces.py \
  tests/hermes_cli/test_kanban_per_profile_cap.py \
  tests/hermes_cli/test_kanban_host_cap.py \
  tests/hermes_cli/test_kanban_cli.py \
  tests/gateway/test_kanban_reconcile_orphans.py \
  tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py \
  tests/hermes_cli/test_kanban_worker_lifecycle_hooks.py \
  tests/cron/test_cron_kanban_env_isolation.py \
  tests/cli/test_single_query_session_finalize.py -q
```

预期：全部 PASS，零 skipped（平台明确不支持的既有 skip 除外，需逐项解释）。

- [ ] **步骤 4：重复真实路径 smoke**

重新执行任务 2 的真实子进程登记和任务 4 的真实 CLI reconcile。不得仅引用早先输出。验证文档记录临时 DB 路径、任务 ID、子进程 PID、分类前后态和退出码；不记录凭据。

- [ ] **步骤 5：范围与字符检查**

```bash
git diff --check origin/main...HEAD
git status --short
git diff --name-only origin/main...HEAD
git diff --unified=0 origin/main...HEAD | grep -P '^\+(?!\+\+\+).*[^\x00-\x7F]'
```

非 ASCII 命中只允许中文文档和既有用户文案需要的字符；Python 标识符、命令、事件 kind 和 JSON 字段必须 ASCII。清理孤儿 import、变量、scratch、`.bak` 和本任务创建的 `/tmp/draft_*`。

- [ ] **步骤 6：提交最终验证证据**

```bash
git add .planning/verification/kanban-runtime-reconciliation.md
git commit -S -s -m "tests/kanban-runtime: preserve fail-closed recovery evidence"
```

- [ ] **步骤 7：创建独立 Reviewer 看板卡**

任务体必须包含：

```markdown
## 范围
- Repo/worktree: <absolute worktree path>
- Base: origin/main
- Head: <signed SHA>
- 只读评审；不得修改代码、不得提交、不得 push。

## 评审约束
- 先更新代码图谱再评审；记录 stats/head_matches_build 和 semantic search。
- 读取规格、计划、完整 diff、被改函数和兄弟调用方。
- 检查输入信任边界：进程 argv、PID 复用、board DB path、claim/run CAS。
- 检查未登记活进程和重复 Worker 是否 fail closed。
- 运行 Forge Review；每轮报告写入指定路径。
- 任何 confirmed finding 使 clean count 归零。

## Deliverables
- .planning/reviews/kanban-runtime-reconciliation-round-1.md
- .planning/reviews/kanban-runtime-reconciliation-round-2.md
- .planning/reviews/kanban-runtime-reconciliation-round-3.md

## 结论格式
- 所有关键结论带 [KNOWN]/[COMPUTED]/[INFERRED] 与 HIGH/MED/LOW。
```

所有看板命令显式 `--board default`。架构师用 `hermes kanban --board default list/show` 验证 Reviewer 卡和进程，不信 Latest summary。

- [ ] **步骤 8：处理 Reviewer finding**

收到 request-changes 后逐条复现和处理。合理 P 级 finding 必须修复；修复期间不追加新功能。每次确认 finding 修复后，重新执行步骤 2 至 6，clean round 从零开始。

- [ ] **步骤 9：验证三轮 clean 证据**

逐文件读取三份报告，不只看 kanban 状态。确认：

- reviewed SHA 相同；
- 每轮 Forge receipt 完整；
- 无未处置 finding；
- clean counter 连续为 1、2、3；
- 报告无内部角色流转术语泄露到对外材料。

- [ ] **步骤 10：最终 Git 验证**

```bash
git status --short --branch
git diff --check origin/main...HEAD
git log --oneline origin/main..HEAD
git log -1 --show-signature --format='%H%n%an <%ae>%n%G? %GK%n%s%n%(trailers:key=Signed-off-by,valueonly)'
git worktree list
git branch --show-current
```

预期：工作区干净；分支为 `fix/kanban-runtime-reconciliation`；所有提交签名有效、key 为 `3187CF09CEE6FA15`、有 Signed-off-by；没有 main/master 直接提交。

---

## 验收映射

| 规格验收条件 | 对应任务与证据 |
|---|---|
| Worker 能补写 PID，冲突不覆盖 | 任务 1、2；原子登记测试、真实子进程测试、claim 缺陷注入。 |
| CLI 与 Dashboard 出现三类诊断 | 任务 3；规则测试和共享引擎集成测试。 |
| reconcile 默认只读且拒绝歧义 | 任务 4；真实进程 fail-closed 测试、真实 CLI smoke、CAS 测试。 |
| profile cap 跨 board | 任务 5；两真实 board 的 dry-run 和真实 claim。 |
| 每项新测试有 FAIL/PASS | 任务 1 至 5；统一记录在 verification 文档。 |
| 真实 CLI、SQLite、子进程 | 任务 2、4、7；最终重复 smoke。 |
| 连续三轮 Forge Review | 任务 7；三份逐轮报告。 |
| diff、签名和分支卫生 | 任务 7；最终 Git 命令输出。 |
