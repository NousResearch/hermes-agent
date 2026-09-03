# Hermes 看板运行态一致性治理设计

## 0. 人话摘要

看板现在把“有人领了任务”和“确实有一个受控进程在执行”混成了同一件事。结果有两种：进程已经没了，卡片还显示运行；进程仍在工作，卡片却不知道它是谁。前者占住队列，后者可能被重复派发。方案是给每次执行建立可核对的身份，发现异常时先报告、再按安全等级修复；无法确认唯一执行者时绝不自动回收。代价是多一次轻量登记和跨看板查询，但不增加数据库表，也不支持把手工启动的聊天进程冒充正式 Worker。

## 1. 背景与已验证事实

本设计基于 2026-09-01 的源码和活库排查。

- [KNOWN][HIGH] `hermes_cli/kanban_db.py:reconcile_orphaned_running` 只处理 `claim_lock` 或 `claim_expires` 缺失的运行卡。claim 完整、`worker_pid` 为空的卡不会进入该分支。
- [KNOWN][HIGH] `hermes_cli/kanban_db.py:detect_crashed_workers` 只查询 `worker_pid IS NOT NULL` 的运行卡。`running + worker_pid=NULL` 不会进入崩溃检测。
- [KNOWN][HIGH] `hermes_cli/kanban_diagnostics.py` 没有运行态身份规则。实测曾出现任务有真实进程、数据库却没有 PID 和心跳，诊断仍返回零项。
- [KNOWN][HIGH] `hermes_cli/kanban_db.py:_set_worker_pid` 由 dispatcher 在子进程启动后写入 PID；子进程没有独立的启动登记。父进程若在“启动成功”和“写入 PID”之间退出，会留下身份空洞。
- [KNOWN][HIGH] 全局 `max_in_progress` 已通过 `count_running_tasks_other_boards` 统计其他看板；`max_in_progress_per_profile` 仍只按当前数据库的 `assignee` 分组。
- [KNOWN][HIGH] 当前 `heartbeat_current_worker_from_env` 已使用任务 ID、run ID 和 claim lock 续期，但只在运行活动发生后调用，也不登记当前进程 PID。
- [KNOWN][HIGH] 现有 dispatcher Worker 会注入 `HERMES_KANBAN_TASK`、`HERMES_KANBAN_RUN_ID`、`HERMES_KANBAN_CLAIM_LOCK` 和明确的数据库路径；普通手工 `hermes chat -q "work kanban task ..."` 不具备这组受控身份。
- [KNOWN][HIGH] 基线测试 `tests/gateway/test_kanban_reconcile_orphans.py` 与 `tests/hermes_cli/test_kanban_per_profile_cap.py` 共 11 项，当前全部通过。这说明已有行为稳定，但没有覆盖本次发现的两种身份异常和跨看板 profile 上限。

源码定位以真实文件、调用点搜索和测试执行为主。静态索引工具不可用不作为“机制不存在”的证据。

## 2. 问题定义

### 2.1 当前状态表达不够

`tasks.status='running'` 只表示 claim 已建立。它不能单独证明：

- 当前 run 存在且仍处于运行态；
- 数据库 PID 属于这次 run；
- 该 PID 仍存活且命令行对应当前任务；
- 没有第二个进程执行同一任务；
- Worker 已开始产生心跳。

操作员只看状态列时，很容易把 claim 当成执行事实。

### 2.2 两种故障方向

1. 数据库显示运行，系统中没有对应进程。这是传统幽灵状态，会占用容量并阻塞后续派发。
2. 系统中有进程，数据库 PID 为空或指向别的进程。这种状态更危险：claim 到期后可能再次派发，两个 Worker 随后写同一工作区。

第二种情况不能用普通 `reclaim` 自动处理。数据库没有可验证的 canonical PID，贸然释放 claim 只会让原进程继续运行，同时允许新进程启动。

### 2.3 跨看板容量口径不一致

全局容量按所有看板计算，profile 容量只计算当前看板。配置 `max_in_progress_per_profile=N` 时，同一 profile 可以在多个看板分别达到 N。实际进程数因此超过用户设置的上限。

## 3. 目标与非目标

### 3.1 目标

1. dispatcher 和 Worker 共同确认同一份运行身份，父进程登记失败时由子进程补写。
2. Dashboard 与 CLI 对结构异常给出明确诊断，不再把 `running + worker_pid=NULL` 显示为健康。
3. 提供只读优先的运行态核对命令；只有状态唯一、进程事实明确时才允许自动修复。
4. `max_in_progress_per_profile` 对所有未归档看板使用同一统计口径。
5. 保留现有数据库格式、事件历史和正常 dispatcher 路径。

### 3.2 非目标

- 不把任意手工启动的 `hermes chat -q "work kanban task ..."` 变成正式 Worker。
- 不自动终止数据库未登记的进程，也不按命令名批量杀进程。
- 不支持跨主机共享同一看板数据库；现有 Worker 生命周期仍是单机模型。
- 不新增核心模型工具，不增加新的用户可见环境变量。
- 不顺带重写整个看板状态机或 Dashboard。
- 不解决两个独立 dispatcher 在同一瞬间跨看板抢占容量的分布式事务问题。现有 singleton gateway 加顺序 sweep 仍是生产前提；手工并发 dispatch 继续属于非标准路径。

## 4. 方案比较

### 4.1 只改操作规范

禁止手工正式 Worker，要求人工核对数据库和进程。改动小，但程序仍会静默接受错误状态，无法消除父进程启动竞态，也不能修复跨看板 profile 上限。

### 4.2 规范与 Hermes 一起修复（采用）

补齐 Worker 自登记、结构诊断、安全核对命令和跨看板计数。正常路径仍由 dispatcher 管理，异常路径默认只读。改动覆盖四个边界，但都建立在现有 claim、run、diagnostics 和 dispatcher 机制上，不引入第二套调度系统。

### 4.3 更换调度器或退回单看板

可以减少状态组合，但会丢失多角色并行和项目隔离，迁移成本明显高于修复现有缺口。当前证据不足以支持这条路线。

## 5. 运行身份契约

一次正式执行由下列四元组确定：

```text
(board_db_path, task_id, current_run_id, claim_lock)
```

PID 是该执行在本机的进程身份。任务行和当前 run 行必须同时满足：

```text
status == running
current_run_id 指向一个未结束的 running run
任务行 claim_lock == run 行 claim_lock == Worker 环境中的 claim lock
任务行 worker_pid == run 行 worker_pid == Worker 当前 PID
```

### 5.1 原子登记

在 `hermes_cli/kanban_db.py` 增加共享登记函数。建议接口：

```python
register_worker_pid(
    conn,
    task_id: str,
    pid: int,
    *,
    expected_run_id: int,
    expected_claim_lock: str,
    source: str,
) -> str
```

返回值限定为：

- `registered`：任务行和 run 行从空 PID 原子写入同一 PID；
- `already_registered`：两行已经是同一 PID，幂等成功；
- `rejected`：状态、run、claim 或已有 PID 冲突，没有写入。

写入规则：

1. `expected_run_id` 和 `expected_claim_lock` 必须存在；缺一项即拒绝。不能退化为“只凭任务 ID”写入。
2. 在一个 `write_txn` 中先核对任务行和 run 行，再同时更新。
3. 已登记为不同 PID 时拒绝覆盖。这表示可能有重复 Worker 或 PID 复用，需要人工核对。
4. 首次登记继续产生 `spawned` 事件，payload 增加 `source=dispatcher|worker_start|heartbeat_repair`。幂等调用不重复写事件。
5. 现有 `_set_worker_pid` 改为受 pin 保护的薄封装，dispatcher 传入 claim 后拿到的 run ID 和 claim lock。

### 5.2 子进程启动登记

在受控 Worker 启动时调用 `register_current_worker_from_env()`：

- 从现有四个 `HERMES_KANBAN_*` 变量取得数据库、任务、run 和 claim；
- PID 使用 `os.getpid()`；
- 在模型请求、工具执行和长任务开始前完成；
- 登记失败不得覆盖其他 PID，也不得让普通聊天进程接管任务；失败写日志，后续 heartbeat 可再次尝试同一受 pin 的幂等登记。

调用点放在 `cli.py` 的 quiet chat 启动路径中，仅当受控看板环境完整时执行。`heartbeat_current_worker_from_env()` 在写心跳前再做一次幂等登记，用于恢复启动阶段的临时数据库锁失败。

### 5.3 手工 Worker 的边界

标准流程不再允许把下面的命令当作正式执行入口：

```text
hermes -p <profile> chat -q "work kanban task <id>"
```

它可以用于只读调查，但没有完整四元组时：

- 不登记 PID；
- 不续 claim；
- 不调用任务终态工具；
- 不计入受控 Worker。

需要接管时，先按 board-qualified 流程回收并阻止自动重派，再由前台会话在原隔离 worktree 中继续。接管不伪装成后台 Worker。

## 6. 结构诊断

`hermes_cli/kanban_diagnostics.py` 增加运行身份规则。规则只读数据库结构，不在 Dashboard 请求中遍历系统进程。

### 6.1 诊断类型

| kind | 条件 | 等级 | 默认建议 |
|---|---|---:|---|
| `running_worker_pid_missing` | 运行超过 30 秒，任务行或当前 run 的 PID 为空 | error | 运行 board-qualified `reconcile`；不得直接建议 reclaim |
| `running_worker_run_mismatch` | current run 不存在、已结束、状态不是 running，或任务/run 的 claim、PID 不一致 | critical | 只读核对；禁止自动覆盖 |
| `running_worker_heartbeat_missing` | 运行超过 120 秒仍无首次心跳 | warning | 检查 Worker 日志和运行态核对报告 |

30 秒和 120 秒是内部保护阈值，不新增配置项。前者远大于正常 `Popen` 到 PID 落库时间；后者覆盖现有 60 秒自动心跳限频。

### 6.2 规则行为

- 终态卡不触发这些诊断。
- 刚 claim 的卡在 30 秒窗口内不报 PID 缺失，避免启动噪声。
- PID 缺失诊断不提供一键 `reclaim`。数据库看不到 PID，不等于系统里没有进程。
- 新 kind 加入 `DIAGNOSTIC_KINDS`。Dashboard 已使用同一规则引擎，因此无需增加新的展示通路。

## 7. 安全核对命令

新增命令：

```text
hermes kanban --board <slug> reconcile [task_id] [--fix] [--json]
hermes kanban reconcile --all-boards [--fix] [--json]
```

默认只读。`--all-boards` 与显式 `--board` 不混用；每个结果必须带 board slug 和数据库路径，避免“查错库仍返回成功”。

### 7.1 核对输入

命令一次性快照：

1. 任务行；
2. current run 行；
3. 本机进程表，只保留精确匹配 `work kanban task <id>` 的 PID；
4. 登记 PID 的存活状态与命令行归属；
5. 首次/最近心跳时间。

进程实现使用项目已依赖的 `psutil`。输出不打印完整命令行，避免把其他参数或路径带入报告；只输出 PID、匹配结果和分类。

### 7.2 分类

| 分类 | 事实 | `--fix` 行为 |
|---|---|---|
| `healthy` | task/run 一致，登记 PID 存活并匹配任务 | 不操作 |
| `dead_registered_worker` | 登记 PID 已死，没有同任务活进程 | 交给现有崩溃恢复语义处理，随后回读 |
| `orphaned_claim_no_process` | claim 结构破损，且无同任务活进程 | 允许重排并关闭泄漏 run |
| `missing_pid_no_process` | 超过启动窗口，PID 为空，且无同任务活进程 | 允许重排并关闭泄漏 run |
| `live_process_unregistered` | PID 为空，但存在一个同任务活进程 | fail closed，不修改 |
| `duplicate_live_workers` | 同任务存在多个活进程 | fail closed，不修改 |
| `registered_pid_mismatch` | 登记 PID 存活，但命令行不属于当前任务 | fail closed，不修改 |
| `remote_or_unreadable` | claim 非本机，或进程信息无法可靠读取 | fail closed，不修改 |

### 7.3 修复约束

- `--fix` 只处理表中明确允许的分类。
- 修复使用 task ID、run ID、claim lock 和旧 PID 的 compare-and-swap 条件；扫描后状态变化则拒绝写入。
- 不调用 `pkill -f`、`killall`，也不按任务名批量结束进程。
- `live_process_unregistered` 和重复进程必须由启动它的会话按精确 PID 收尾，随后重新核对。
- 每次写入后立即重新读取任务、run 和进程快照。只有后态符合预期才报告 fixed。
- 自动 dispatcher 继续执行已有轻量恢复；新的 CLI 是面向操作员的完整核对面，不把昂贵的全进程扫描放进每个 60 秒 tick。

## 8. 跨看板 profile 容量

在 `hermes_cli/kanban_db.py` 增加按 assignee 汇总其他看板运行数的 helper，复用 `list_boards()`、规范化数据库路径和“同一数据库不重复计数”逻辑。

建议接口：

```python
count_running_tasks_by_assignee_other_boards(
    board: str | None = None,
) -> dict[str, int]
```

`dispatch_once` 初始化 `_per_profile_running` 时：

1. 查询当前数据库的运行数；
2. 合并其他未归档看板的同 assignee 运行数；
3. 在 dry-run 和真实派发中使用同一累计值；
4. 当前 tick 每成功 claim 一个任务，继续递增内存计数。

沿用全局容量的错误策略：单个损坏看板不阻断健康看板派发，但必须写 warning，说明 profile 计数可能不完整。该选择保留现有可用性取舍，不把一块损坏数据库变成全机停摆。

## 9. 数据流

### 9.1 正常启动

```text
dispatcher claim
  -> 创建 current run
  -> Popen Worker，注入 task/run/claim/db
  -> dispatcher 原子登记 PID
  -> Worker 启动后幂等确认 PID
  -> 首次活动写 heartbeat
  -> 后续每 60 秒最多写一次 heartbeat
```

父子双方竞争登记同一个 PID是允许的；结果只能是一次 `registered` 和一次 `already_registered`。任何不同 PID 都拒绝覆盖。

### 9.2 父进程在登记前退出

```text
dispatcher Popen 成功
  -> dispatcher 未写 PID 就退出
  -> Worker 根据受控环境登记自身 PID
  -> task/run 恢复一致
```

### 9.3 发现未登记活进程

```text
结构诊断报 PID 缺失
  -> reconcile 扫描到同任务活进程
  -> 分类 live_process_unregistered
  -> 不回收、不重派
  -> 操作员定位启动会话并精确收尾
  -> 再次 reconcile，确认无活进程后才允许修复
```

## 10. 文件边界

实现预计修改：

- `hermes_cli/kanban_db.py`：受 pin 的 PID 登记、运行态 CAS 修复、跨看板 profile 计数。
- `tools/kanban_tools.py`：从受控环境登记当前 Worker，并在 heartbeat 前补登记。
- `cli.py`：受控 quiet Worker 的启动登记调用点。
- `hermes_cli/kanban_diagnostics.py`：三类结构诊断。
- `hermes_cli/kanban.py`：`reconcile` 参数、只读报告和 `--fix` 编排。
- `plugins/kanban/dashboard/plugin_api.py`：预计无需改变接口；验证新诊断会沿既有通路出现。
- `tests/hermes_cli/` 与 `tests/gateway/`：登记竞态、诊断、核对命令和跨看板容量测试。
- `docs/`：用户可执行的恢复命令和限制。

不新增数据库列或表。

## 11. 测试与证伪

### 11.1 Worker 登记

- dispatcher 先登记、Worker 后确认：只有一个 `spawned` 事件。
- Worker 先登记、dispatcher 后确认：结果相同。
- run ID 或 claim lock 过期：登记被拒绝，任务/run 不变。
- 已有不同 PID：登记被拒绝。
- 真实子进程使用临时 `HERMES_HOME` 和真实 SQLite 文件执行启动登记，父进程读取 task/run 验证 PID 等于子进程真实 PID。

缺陷注入：暂时移除 Worker 启动登记，让“父进程未登记”测试失败；恢复后用同一测试通过。

### 11.2 结构诊断

覆盖三种正例、启动宽限负例、健康运行负例和终态负例。Dashboard/API 集成测试确认诊断从真实 sqlite row 出现。

缺陷注入：从 `_RULES` 移除新规则，同一 PID 缺失测试必须失败；恢复后通过。

### 11.3 安全核对

至少使用真实 `sleep`/Python 子进程验证：

- 无进程的缺 PID卡可修复；
- 一个未登记活进程时 `--fix` 拒绝；
- 两个匹配进程时 `--fix` 拒绝；
- PID 被复用或命令行不匹配时拒绝；
- 修复后 task/run 后态被重新读取。

缺陷注入：在真实修复判断处去掉“活进程拒绝”条件，同一测试必须观察到错误重排；恢复条件后通过。

### 11.4 跨看板容量

建立两个真实临时 board 数据库：A 已有一个 `coder` 运行任务，B 有一个 `coder` ready 任务和一个 `reviewer` ready 任务。设置每 profile 上限为 1：B 不得派 `coder`，但可派 `reviewer`。dry-run 和真实 claim 各测一次。

缺陷注入：暂时撤回其他看板计数合并，同一测试必须错误派发第二个 `coder`；恢复后通过。

### 11.5 门禁

- `python -m py_compile` 覆盖所有改动 Python 文件；
- `ruff check` 对改动文件零新增告警；
- 目标测试、相关看板测试和真实 CLI smoke 全部通过；
- 新增行非 ASCII 检查只允许用户可见文案和现有文件约定需要的字符；
- 逻辑改动完成后由独立评审者复核，确认 finding 修复后重新复核。

## 12. 发布与回滚

1. 代码在独立分支实现，不修改运行中的生产 checkout。
2. 先合入纯兼容逻辑：登记为幂等、诊断只读、`reconcile` 默认只读、跨看板计数只收紧已配置上限。
3. 合入前不重启运行中的 gateway，不把 feature 分支直接用于生产环境。
4. 若发现兼容问题，回滚代码即可；没有数据库迁移需要逆转。新增事件仍是附加审计记录，旧版本会忽略未知 payload 字段。

## 13. 操作规则

配套的运维标准需同步以下硬规则：

- 正式 Worker 只能由 dispatcher 启动；手工聊天进程只做只读调查。
- 后台状态汇报必须同时核对任务行、current run 和真实进程。
- `running + worker_pid=NULL` 一律视为身份异常，不能直接 reclaim。
- 任何状态操作显式携带 `--board <slug>`。
- profile 容量判断必须枚举所有看板，不能只看当前 board 的统计。
- 未登记活进程、重复进程和 PID 不匹配全部 fail closed，由进程所有者按精确 PID处理。

## 14. 验收条件

全部满足才算完成：

1. 父进程漏登记时，受控 Worker 能用同一 run/claim 原子补写 PID；冲突 PID 不会被覆盖。
2. PID 缺失、task/run 不一致、首次心跳缺失会出现在 CLI 与 Dashboard 诊断中。
3. `reconcile` 默认只读，`--fix` 不会回收未登记活进程或重复 Worker。
4. 同一 profile 在多个看板上的运行总数受 `max_in_progress_per_profile` 约束。
5. 每项新增测试都有真实修复点的 FAIL/PASS 缺陷注入证据。
6. 至少一次真实 CLI、真实 SQLite 文件和真实子进程联调通过。
7. 独立评审者复核无未处置 finding。
8. diff 仅包含本任务范围的改动。
