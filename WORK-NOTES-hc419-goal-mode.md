# WORK NOTES — hc-419 长任务 Goal-mode 面 (W5-4)

## Scope
- Repo: `karlligamesvc-spec/hermes-agent` (desktop fork). Branch `feat/hc419-goal-mode`.
- PD r417 hc-419: 对标 Codex "Goal mode"（自主跑数小时~数天）。产品面缺口 = 交个大活→后台跑→随时看进度→完成通知。
- 只做桌面 UI (`apps/desktop/src`)。不碰 `apex-managed.cjs` seed。只 PR + CI 绿，不合并/不出包/不 bump pin。

## 步骤①摸底 — runtime 真实后台执行能力 (诚实评估)

两路并行读源码（fork worktree），结论如下，**带行号引用**。

### delegate_task —— 真后台但不持久 (❌ 不能做"数天自主")
- `tools/delegate_tool.py:2339` `delegate_task()`；dispatch 包装 `run_agent.py:5602` `_dispatch_delegate_task`。
- 顶层 agent 调用 **强制 background=true**（`run_agent.py:5629`），立即返回 handle（delegation_id），子 agent 完成后作为**新消息**重新入会话。
- 但执行体 = **进程内 daemon 线程**（`tools/async_delegation.py:82,93` `_DaemonThreadPoolExecutor` / `_executor`），**非独立进程**。
- **致命约束（诚实）**：`tools/delegate_tool.py:2791-2798` + `5193-5195` 明写——父 session 关闭 (/new) 或进程退出，未完成的子 agent 工作**被丢弃**；background delegation **NOT durable**，无 checkpoint/恢复。
- 并发上限默认 3（`delegation.max_async_children`），池满**拒绝**不排队（`async_delegation.py:217-227`）。
- → 结论：delegate_task 适合"单会话内并行分工、稍后回同一会话收结果"，**不适合**做"跨会话、跨重启、跑数天"的 Goal-mode 后端。

### todo 工具 —— 进程内、不落盘
- `tools/todo_tool.py:41` `TodoStore` 挂在 AIAgent 实例上（每 session 一个）。**不持久化到磁盘**；能扛 context 压缩重注入（`:111-143`），但扛不住 session 关闭/进程重启。
- 外部读取只能"调用该工具"，无直接 API/文件/SSE。
- **但**：桌面侧已有 `src/lib/todos.ts` `latestSessionTodos(messages)` —— 从 session **transcript 消息**里解析出最新 todo 列表（done/in_progress/pending）。这是我拿进度的真实数据源（不依赖工具直读）。

### session 持久 —— ✅ 落 SQLite，扛重启
- `hermes_state.py:123` SQLite `~/.hermes/state.db`（schema v17）。持久：session 元数据 + 全量 messages + 模型配置 + parent chain。
- 恢复窗口：`gateway/session.py` auto-continue 默认 1h（`get_or_create_session`）。过期不自动续。
- 子 agent(delegate) session 作为 child 存（`parent_session_id`），随父删级联删，默认不进 /resume 列表。

### ★关键正向发现 —— cron 是"真·持久后台执行引擎"，且原生支持一次性
- cron scheduler = 后端进程内**常驻** ticker 线程，每 60s tick（`gateway/run.py:19068 _start_cron_ticker` / `cron/scheduler_provider.py InProcessCronScheduler`）。桌面后端也起（`hermes_cli/web_server.py:132 _start_desktop_cron_ticker`）。
- job 持久化到 `~/.hermes/cron/jobs.json`（`cron/jobs.py:66`，跨进程文件锁）。**扛 app 重启**（只要后端进程在/重启后重读）。
- **原生 one-shot**：`cron/jobs.py:316 parse_schedule` → `kind: "once" | "interval" | "cron"`。一次性写法：`"30m"`/`"2h"`/`"1d"`（从现在起）、`"2026-02-03T14:00"`（定时）。
- 一次性跑完的终态：`cron/jobs.py:1314-1316` → `enabled=False, state="completed"`（失败递归型 → `state="error"`；一次性算 completed）。
- 每次 run 产出真实 session `cron_{job_id}_{ts}`，持久在 SQLite，`hermes_state.py:2948 list_cron_job_runs` 用索引区间扫返回 enriched 行（含 `is_active`/`preview`/`last_active`/`ended_at`/`title`）。
- REST 全就位：`hermes_cli/web_server.py:8564` GET jobs / `:8593` GET `{id}/runs` / `:8639` POST create（`schedule` 字段直穿 `parse_schedule`）/ PUT / pause / resume / trigger / DELETE。

### 诚实结论 → 做"轻版真实"而非"假的数天自主"
- **不做**假的"agent 自主跑数天"（delegate 后台线程扛不住重启，撑不起这个承诺）。
- **做**：一次性长任务 = 复用 cron `kind:"once"` 后端。真实闭环：交大活→后端持久调度→跑出真实 session→用 session 的 todos+最新产出看进度→完成/卡住系统通知→**扛 app 重启**（因为是持久化 cron job）。字段拿不到就诚实留空/标注。
- 这是当前 runtime 能力下**唯一真·持久**的后台执行路径，比 delegate 更适合"交个大活后台跑"。

## 步骤②-⑤ 实现范围

- **独立页 `/tasks`（"任务"）**，主区页（非 overlay），与 `/cron` 并列。判断：cron=定时反复，tasks=一次性长活，产品概念不同 → 独立页 + 独立 nav 入口比同页 tab 干净；复用 `/cron` 的列表/详情/状态点组件省成本。
- **数据源**：`getCronJobs()` 过滤出一次性 job（`schedule.kind === "once"` 或 display/一次性推断）。分两态：**运行中**（scheduled/running/enabled + 未完成）、**已完成/已停**（completed/error/disabled）。零新后端。
- **进度卡**：运行中任务 → 拉其最新 run session（`getCronJobRuns`）→ `getSessionMessages` → `latestSessionTodos` 得步数（已完成 N / 共 M / 当前在做 X）+ 最近文本 = 最近产出。拿不到留空。
- **通知**：store 级 watcher 轮询任务 job 状态，running→completed/error 跳变触发 `dispatchNativeNotification`（复用既有 native 通知 + `backgroundDone`/`turnError` kind）。"卡住" = 运行中但 run session `last_active` 陈旧超阈值。
- **崩溃恢复**：job + run 均后端持久 → 重开 app 由 controller 轮询 `getCronJobs` 自动找回。**诚实标注的限制**：某个 run 若在后端进程被杀的瞬间跑到一半，本功能不负责把那一轮续起来（那是 runtime 层缺口，非本票范围）。

## 步骤⑥ 测试
- vitest（`npm run test:ui`）+ `npm run typecheck` 零回归。
- 覆盖：任务分类（运行中/完成/卡住）、进度推导（todos→步数/当前）、完成通知触发、一次性 schedule 构造、重开恢复(mock hermes API)。

## Sources Read (with line refs)
- delegate: `tools/delegate_tool.py:2339,2791-2798,5193-5195`；`tools/async_delegation.py:82,93,217-227`；`run_agent.py:5602,5629`
- todo: `tools/todo_tool.py:41,111-143`；`apps/desktop/src/lib/todos.ts`
- session persist: `hermes_state.py:123,2948`；`gateway/session.py`
- cron 引擎/一次性: `cron/jobs.py:66,316,1314-1316`；`cron/scheduler_provider.py`；`gateway/run.py:19068`；`hermes_cli/web_server.py:132,8564,8593,8639`
- desktop 复用面: `apps/desktop/src/app/cron/index.tsx`, `.../cron/job-state.ts`, `.../routes.ts`, `.../chat/sidebar/index.tsx:99-130`, `.../desktop-controller.tsx:1165-1175`, `.../store/native-notifications.ts`, `.../store/notifications.ts`, `.../hermes.ts:546-608`

## Validation Log
- (填充中)
