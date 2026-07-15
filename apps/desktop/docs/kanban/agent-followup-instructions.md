# Kanban Agent 后续修复指令文档

> 本文用于指导后续 agent 完成当前 Kanban 功能的不足项。
>
> 本文不是 Kanban 总纲，不重新设计 Kanban，不重复已有 GUI / Chat / running task 文档。
>
> 本文只列出当前代码审查后仍需完成的具体补丁、文件位置、验收标准和禁止重复开发事项。

## 1. 文档位置与关系

当前 Kanban 文档体系：

```text
apps/desktop/docs/
  kanban.md
  kanban/
    gui-remediation-plan.md
    conversation-task-integration.md
    implementation-status.md
    running-task-loading.md
    agent-followup-instructions.md
```

职责关系：

| 文档 | 职责 |
| --- | --- |
| `kanban.md` | Kanban 总体设计、数据模型、IPC、长期路线 |
| `gui-remediation-plan.md` | Kanban GUI 显示和交互修复 |
| `conversation-task-integration.md` | Chat / Agent / Profile / Cron 与 Kanban 的集成路线 |
| `implementation-status.md` | 当前完成情况和旧文档差异修正 |
| `running-task-loading.md` | running task 不显示的 board identity 专项修复方案 |
| `agent-followup-instructions.md` | 当前文档：指导 agent 完成剩余不足 |

阅读顺序建议：

```text
implementation-status.md
  -> running-task-loading.md
  -> agent-followup-instructions.md
```

## 2. 当前总体状态

当前 Kanban Desktop MVP 已经完成大部分基础能力。

已完成项：

- `/kanban` 路由接入。
- Kanban Renderer 页面基础版。
- Board / Task / Comment CRUD。
- SQLite 存储，当前使用 `HERMES_HOME/kanban.db`。
- `reorderTasks` 排序持久化。
- Column 固定宽度、横向滚动、右侧详情面板。
- Task card 点击打开详情。
- 空列 droppable。
- Sidebar Kanban 名称显示。
- Assistant message -> Create Kanban Task。
- User message -> Create Kanban Task。
- Chat task 创建时保存 `messageId`。
- Board identity 已统一为 slug，解决 running task 因 board id mismatch 被过滤的问题。
- Agent plan 菜单入口已出现。
- Cron failure sync 逻辑已出现。

仍需补齐项：

1. `externalTaskId` / `externalTaskKind` 已在前端和类型里使用，但没有持久化到 SQLite。
2. `syncTodoToKanbanTasks()` 依赖 `externalTaskId`，因此 todo linked/mirrored sync 当前不可靠。
3. `syncCronFailureToKanban()` 依赖 `externalTaskId` / `externalTaskKind` 做去重，因此 cron dedup 当前不可靠。
4. `lastSyncedAt` 类型存在，但 main process 没有持久化。
5. `implementation-status.md` 需要更新，反映最新已经完成的 messageId、board identity、agent plan、cron failure 相关进展。
6. 需要补充或更新测试，避免上游 review 认为只是“表面有入口”。

## 3. 最高优先级任务

### P0：持久化 external task linkage

必须先完成：

```text
externalTaskId
externalTaskKind
lastSyncedAt
```

这些字段是 Chat / Agent / Cron 集成的关键 linkage。

如果不持久化，会导致：

- Agent plan task 创建后无法稳定与 todo item 匹配。
- linked / mirrored sync 失效或只能短期内存有效。
- Cron failure dedup 无法跨 reload / app restart 工作。
- Kanban detail 无法说明 task 来源于哪个 agent plan item / cron job。

## 4. 禁止重复开发事项

后续 agent 不要做以下事情：

- 不要新建第二个 Kanban 页面。
- 不要新建第二套 task 存储。
- 不要回退到 `kanban.json`。
- 不要新增独立的 `agent_tasks.json` / `conversation_tasks.json`。
- 不要新建第二套 Task CRUD API。
- 不要新建第二套排序 API。
- 不要绕过 `window.hermesDesktop.kanban.*` 直接让 Renderer 访问 SQLite。
- 不要把 profile 当负责人。
- 不要把 running task 特判成独立页面。
- 不要仅修改文档而不修代码。

正确方向：

```text
扩展现有 SQLite schema
  -> 扩展现有 createTask / rowToKanbanTask / updateTask
    -> 修复 sync 逻辑
      -> 更新状态文档和测试
```

## 5. 必改文件

### 5.1 Main process

```text
apps/desktop/electron/main.cjs
```

必须改：

- `ensureKanbanSchema(db)`
- `rowToKanbanTask(row)`
- `sanitizeTaskInput(input)` 或 create handler 的安全读取逻辑
- `hermes:kanban:createTask`
- `hermes:kanban:updateTask`，如需要更新 sync 字段

### 5.2 TypeScript 类型

```text
apps/desktop/src/global.d.ts
```

目前类型已经包含：

```ts
externalTaskId?: string
externalTaskKind?: string
lastSyncedAt?: number
```

需要确认：

- `updateTask` 入参是否允许更新 `syncMode` / `lastSyncedAt`。
- `KanbanTask` 返回类型和 main process row mapping 是否一致。

### 5.3 Sync 工具

```text
apps/desktop/src/lib/kanban-sync.ts
```

需要确认：

- `syncTodoToKanbanTasks()` 更新状态后是否同时更新 `lastSyncedAt`。
- `syncCronFailureToKanban()` 去重是否能跨 reload 生效。
- linked / mirrored 语义是否和文档一致。

### 5.4 Chat UI

```text
apps/desktop/src/components/assistant-ui/thread.tsx
```

需要确认：

- Assistant message 创建 task 已传 `messageId`。
- User message 创建 task 已传 `messageId`。
- Agent plan task 创建时传 `externalTaskId` / `externalTaskKind`。
- 不要在这里做 DB 细节处理。

### 5.5 文档

```text
apps/desktop/docs/kanban/implementation-status.md
apps/desktop/docs/kanban/conversation-task-integration.md
apps/desktop/docs/kanban/running-task-loading.md
```

完成代码后，应更新：

- 哪些已完成。
- 哪些仍未完成。
- 哪些不再是缺陷。
- 哪些是后续范围。

## 6. 具体代码补丁要求

### 6.1 SQLite schema 扩展

在 `ensureKanbanSchema(db)` 的 ALTER TABLE 列表中加入：

```js
"ALTER TABLE tasks ADD COLUMN external_task_id TEXT",
"ALTER TABLE tasks ADD COLUMN external_task_kind TEXT",
"ALTER TABLE tasks ADD COLUMN last_synced_at INTEGER"
```

要求：

- 保持幂等，仍然允许 column already exists。
- 不破坏旧数据库。
- 不删除旧字段。

### 6.2 rowToKanbanTask 映射

在 `rowToKanbanTask(row)` 中加入：

```js
externalTaskId: row.external_task_id || undefined,
externalTaskKind: row.external_task_kind || undefined,
lastSyncedAt: row.last_synced_at || undefined
```

要求：

- 字段命名与 `global.d.ts` 一致。
- undefined 优于 null，方便 Renderer 判断。

### 6.3 createTask 持久化

在 `hermes:kanban:createTask` 的 INSERT columns 中加入：

```sql
external_task_id,
external_task_kind,
last_synced_at
```

在 values 中加入：

```js
taskData.externalTaskId || null,
taskData.externalTaskKind || null,
taskData.lastSyncedAt || null
```

如果 `lastSyncedAt` 未提供，但 `syncMode` 是 `linked` 或 `mirrored`，可以设为 `Date.now()`：

```js
const lastSyncedAt = taskData.lastSyncedAt ||
  (taskData.syncMode === 'linked' || taskData.syncMode === 'mirrored' ? now : null)
```

推荐使用该逻辑。

### 6.4 updateTask 支持 sync 字段

`updateTask` 当前主要支持 title、description、status、priority、assignee、archived、order。

需要支持：

```js
syncMode
lastSyncedAt
externalTaskId
externalTaskKind
assigneeType
assigneeLabel
agentId
agentLabel
```

最小必须支持：

```js
syncMode
lastSyncedAt
externalTaskId
externalTaskKind
```

尤其是 todo sync 更新状态时，建议同时更新：

```js
lastSyncedAt = Date.now()
```

### 6.5 updateTask 类型同步

`global.d.ts` 中 `updateTask` 入参应增加：

```ts
syncMode: string
lastSyncedAt: number
externalTaskId: string
externalTaskKind: string
assigneeType: KanbanAssigneeType
assigneeLabel: string
agentId: string
agentLabel: string
```

可以用 `Partial<{ ... }>` 继续保持可选。

## 7. Sync 逻辑修复要求

### 7.1 Todo sync

当前目标：

```text
Todo item status -> linked/mirrored Kanban task status
```

匹配条件应为：

```ts
t.externalTaskId === todo.id &&
t.sessionId === sessionId &&
(t.syncMode === 'mirrored' || t.syncMode === 'linked')
```

更新时应调用：

```ts
await window.hermesDesktop.kanban.updateTask(matchedTask.id, {
  status: kanbanStatus,
  lastSyncedAt: Date.now()
})
```

注意：

- 如果文档规定 linked 不自动更新状态，则应只更新 mirrored。
- 当前代码如果已经允许 linked，也需要在文档中明确 linked 的实际语义。
- 不要静默吞掉所有错误后完全不可观测；至少在 development 下 console.debug 或 console.warn。

### 7.2 Agent plan import

创建 task 时必须保存：

```ts
externalTaskId: todo.id
externalTaskKind: 'agent_plan_item'
syncMode: 'linked' 或 'mirrored'
```

建议策略：

- 手动 “Send plan to Kanban” 创建的 task 使用 `syncMode = 'linked'`。
- 如果要自动跟随 todo status，则使用 `syncMode = 'mirrored'`。
- 不要同时文档说 linked 不同步、代码却同步 linked。

### 7.3 Cron failure sync

创建 task 时必须保存：

```ts
externalTaskId: job.id
externalTaskKind: 'cron_job'
syncMode: 'linked'
```

去重条件应优先用：

```ts
externalTaskId + externalTaskKind + description/currentError
```

如果后续引入 failure window，可以加入：

```ts
failureWindow = YYYY-MM-DD-HH
```

当前最小实现可以不做 failure window。

## 8. 测试要求

### 8.1 Main process / IPC 层测试

建议新增或更新：

```text
apps/desktop/electron/kanban.test.cjs
```

如果已有 Electron kanban 测试文件，优先更新已有文件，不要新建重复测试体系。

必须覆盖：

1. `createTask` 保存 `externalTaskId`。
2. `createTask` 保存 `externalTaskKind`。
3. `createTask` 保存 `lastSyncedAt` 或自动写入 linked/mirrored sync time。
4. `allTasks()` 返回这些字段。
5. `updateTask()` 可以更新 `lastSyncedAt`。
6. running task with `board_id = default` 能与 default board 对齐。
7. old task board_id random id 能迁移为 slug。

### 8.2 Renderer sync 测试

如果已有 jsdom / vitest 测试：

```text
apps/desktop/src/lib/kanban-sync.test.ts
```

建议覆盖：

1. todo `pending` -> Kanban `todo`。
2. todo `in_progress` -> Kanban `running`。
3. todo `completed` -> Kanban `done`。
4. todo `cancelled` -> Kanban `blocked`。
5. matched task 必须依赖 `externalTaskId + sessionId`。
6. cron 相同 error 不重复创建 blocked task。

## 9. 文档更新要求

代码完成后更新：

```text
apps/desktop/docs/kanban/implementation-status.md
```

至少修改：

- `externalTaskId/externalTaskKind` 从“未持久化”改为“已持久化”。
- Agent plan batch import 从“部分落实”改为“基础落实”。
- Todo linked/mirrored sync 从“不可靠”改为“基础落实”或明确保留限制。
- Cron failure dedup 从“不可靠”改为“基础落实”或说明仅内存级去重。

如修改 linked / mirrored 语义，还要更新：

```text
apps/desktop/docs/kanban/conversation-task-integration.md
```

## 10. 验收清单

Agent 完成后必须逐项确认：

### 数据层

- [ ] `tasks` 表包含 `external_task_id`。
- [ ] `tasks` 表包含 `external_task_kind`。
- [ ] `tasks` 表包含 `last_synced_at`。
- [ ] `createTask` 会写入这些字段。
- [ ] `rowToKanbanTask` 会返回这些字段。
- [ ] `updateTask` 可更新 `lastSyncedAt`。

### Agent plan

- [ ] “Send plan to Kanban” 创建的 task 有 `source = agent`。
- [ ] task 有 `externalTaskId = todo.id`。
- [ ] task 有 `externalTaskKind = agent_plan_item`。
- [ ] task 有 `sessionId`。
- [ ] task 有 `profileId`，如果当前 profile 可用。

### Todo sync

- [ ] todo `in_progress` 能更新 Kanban task 到 `running`。
- [ ] todo `completed` 能更新 Kanban task 到 `done`。
- [ ] sync 后 `lastSyncedAt` 更新。
- [ ] 没有 `externalTaskId` 的 task 不会被错误匹配。

### Cron failure

- [ ] cron job 新错误会创建 blocked task。
- [ ] task 有 `externalTaskId = job.id`。
- [ ] task 有 `externalTaskKind = cron_job`。
- [ ] 同一 job 同一 error 不重复创建。

### 回归

- [ ] manual task 创建仍正常。
- [ ] chat task 创建仍正常。
- [ ] board 创建/删除仍正常。
- [ ] task 拖拽排序仍正常。
- [ ] running task 仍能显示在 default board。

## 11. 必跑命令

在提交前运行：

```bash
cd apps/desktop
npm run typecheck
npm run lint
npm run test:ui
```

如果 Electron 层有单独测试：

```bash
cd apps/desktop
node --test electron/*.test.cjs
```

如果仓库根部有 Python / CLI 测试涉及 Kanban DB：

```bash
pytest tests/hermes_cli/test_kanban_*.py
```

如果某些命令因为本地环境缺依赖失败，必须在 PR 描述里说明：

```text
Not run: <command> — <reason>
```

## 12. PR 描述建议

PR 标题：

```text
fix(desktop): persist kanban external task linkage
```

PR 描述：

```text
This PR completes the missing linkage layer for Desktop Kanban integration.

Implemented:
- Persist externalTaskId / externalTaskKind / lastSyncedAt in SQLite tasks table
- Return linkage fields from allTasks/tasks/createTask/updateTask
- Allow sync code to match todo and cron tasks across reloads
- Update docs to reflect current implementation status

Why:
- Agent plan import already passed externalTaskId, but main process did not store it
- Todo linked/mirrored sync depended on fields that were not persisted
- Cron failure dedup depended on fields that were not persisted

Out of scope:
- Full Agent workflow orchestration
- Full cron failure windowing
- Full board picker UI
```

## 13. 完成定义

本任务完成的定义不是“菜单能点”。

本任务完成的定义是：

```text
Agent / Cron / Todo 创建或同步的 Kanban task
  -> 有稳定 external linkage
  -> reload 后仍能匹配
  -> allTasks() 返回完整 metadata
  -> sync 逻辑不依赖短期内存状态
```

只有满足以上条件，才能认为当前 `conversation-task-integration.md` 中的 Agent/Cron 基础集成真正落地。
