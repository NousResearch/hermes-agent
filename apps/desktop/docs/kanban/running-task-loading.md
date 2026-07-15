# Kanban 运行中任务加载问题开发文档

> 本文是 Kanban 文档体系中的专项缺陷与修复方案文档。
>
> 本文只处理“Desktop Kanban 无法稳定看到正在执行的 Hermes / Agent 任务”的问题，不重复定义 Kanban 总功能、GUI 修复计划或 Chat/Agent 集成路线。
>
> 本文应与以下文档配套阅读：
>
> - `apps/desktop/docs/kanban.md`
> - `apps/desktop/docs/kanban/gui-remediation-plan.md`
> - `apps/desktop/docs/kanban/conversation-task-integration.md`
> - `apps/desktop/docs/kanban/implementation-status.md`

## 1. 文档层级关系

当前 Kanban 文档层级：

```text
apps/desktop/docs/
  kanban.md
  kanban/
    gui-remediation-plan.md
    conversation-task-integration.md
    implementation-status.md
    running-task-loading.md
```

职责划分：

| 文档 | 职责 | 本文是否覆盖 |
| --- | --- | --- |
| `kanban.md` | Kanban 总体目标、数据模型、IPC、长期路线 | 不覆盖 |
| `gui-remediation-plan.md` | 页面布局、详情面板、拖拽、排序等 GUI 修复 | 不覆盖 |
| `conversation-task-integration.md` | Chat / Agent / Profile / Cron 与 Kanban 的集成路线 | 不覆盖 |
| `implementation-status.md` | 当前实现状态、已完成项、旧文档差异修正 | 引用，不重复 |
| `running-task-loading.md` | 当前专项问题：运行中任务无法稳定加载到 Desktop Kanban | 当前文档 |

本文定位：

```text
implementation-status.md
  -> 记录当前整体实现状态
running-task-loading.md
  -> 解释为什么 running task 不显示，并给出修复方案
```

## 2. 当前已完成开发项

以下内容已经完成，后续不要重复开发。

| 模块 | 完成状态 | 说明 | 是否重复开发 |
| --- | --- | --- | --- |
| Desktop `/kanban` 路由 | 已完成 | Kanban 页面已接入 Desktop route | 不要重复 |
| Kanban Renderer 页面 | 已完成基础版 | `apps/desktop/src/app/kanban/index.tsx` 已存在 | 不要重建页面 |
| Board CRUD | 已完成基础版 | main process 已有 board handlers | 不要重写 CRUD |
| Task CRUD | 已完成基础版 | main process 已有 task handlers | 不要重写 CRUD |
| Comment CRUD | 已完成基础版 | main process 已有 comment handlers | 不要重写 CRUD |
| SQLite 存储 | 已完成 | 当前使用 `HERMES_HOME/kanban.db`，不是 JSON | 不要再实现 `kanban.json` |
| `reorderTasks` IPC | 已完成基础版 | 已支持拖拽排序持久化 | 不要新增第二套排序 API |
| Kanban GUI 列宽修复 | 已完成基础版 | Column 已固定宽度并支持横向滚动 | 不要重复修布局 |
| 右侧详情面板 | 已完成基础版 | 已从 absolute overlay 改为右侧 pane | 不要重复实现详情面板 |
| Task card 点击打开详情 | 已完成 | Card 已支持选择 task | 不要重复 |
| Column droppable | 已完成基础版 | 空列可作为 drop target | 不要重复 |
| Chat message -> Kanban task | 已完成 MVP | Assistant message 可手动创建 Kanban task | 不要误认为自动同步已完成 |
| Sidebar Kanban 名称 | 已完成基础修复 | i18n nav 中已有 `Kanban / 看板` 文案 | 只需验证视觉，不要新建入口 |

## 3. 当前未完成或存在缺陷的项

以下内容仍未完成或存在缺陷，需要后续开发。

| 模块 | 状态 | 问题 | 优先级 |
| --- | --- | --- | --- |
| 运行中任务加载 | 有缺陷 | Hermes CLI / Agent dispatcher 写入的 running task 可能被 Desktop Kanban 过滤掉 | P0 |
| Board identity 一致性 | 有缺陷 | `kanban_boards.id`、`kanban_boards.slug`、`tasks.board_id` 使用语义不一致 | P0 |
| Default board 映射 | 有缺陷 | default board 在 boards 表中可能是随机 id，但 task 使用 `default` | P0 |
| Chat task `messageId` | 未完成 | 从 assistant message 创建 task 时未保存来源 message id | P1 |
| Chat task 目标 board 选择 | 未完成 | 当前可能默认写入 `default`，没有选择目标 board | P1 |
| Agent plan -> Kanban | 未完成 | Agent plan/todo 还不能批量导入 Kanban | P2 |
| Agent 状态同步 | 未完成 | running/blocked/review/done 状态还没有 mirrored sync | P2/P3 |
| Cron failure -> blocked task | 未完成 | Cron 失败不会自动生成 blocked task | P3 |

## 4. 问题描述

用户期望 Desktop Kanban 可以看到 Hermes / Agent 正在执行的任务。

当前实际表现：

```text
Hermes / Agent 任务正在执行
  -> CLI / dispatcher / worker 侧可能已经在 kanban.db.tasks 中写入 running task
  -> Desktop Kanban 页面打开后看不到这些 running task
```

这不是单纯的 UI 渲染问题，也不是 `running` 状态被显式过滤。

当前更可能的根因是：

```text
Board identity 不一致
  -> task 被 allTasks() 查出来
  -> 前端再按 activeBoardId 过滤
  -> task.boardId 与 activeBoardId 不相等
  -> running task 在 UI 中消失
```

## 5. 当前代码路径

### 5.1 Renderer 加载路径

`KanbanView` 当前通过 preload API 加载数据：

```ts
const [boardList, allTasks] = await Promise.all([
  window.hermesDesktop.kanban.boards(),
  window.hermesDesktop.kanban.allTasks()
])

setBoards(boardList)
setTasks(allTasks)
```

随后前端按 active board 过滤：

```ts
const boardTasks = tasks.filter(t => t.boardId === activeBoardId)
```

因此，即使 `allTasks()` 返回了 running task，只要 `task.boardId !== activeBoardId`，该 task 仍不会显示。

### 5.2 Main process 查询路径

`allTasks` 当前查询逻辑类似：

```sql
SELECT * FROM tasks WHERE archived = 0 ORDER BY created_at DESC
```

这说明 `allTasks` 没有排除 `running`。

所以问题重点不是：

```text
running status 被 SQL 过滤
```

而是：

```text
task 返回后被 Renderer 按 boardId 过滤掉
```

## 6. 根因分析

### 6.1 当前 Board 表

当前 main process 会创建 `kanban_boards` 表：

```sql
CREATE TABLE IF NOT EXISTS kanban_boards (
  id TEXT PRIMARY KEY,
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  description TEXT DEFAULT '',
  created_at INTEGER NOT NULL
)
```

默认 board 当前可能被插入为：

```js
INSERT INTO kanban_boards (id, slug, title, description, created_at)
VALUES (newId(), 'default', 'Default Board', '', Date.now())
```

这意味着：

```text
kanban_boards.id   = 随机 id
kanban_boards.slug = default
```

### 6.2 当前 Task 表

为了兼容 Hermes CLI 的 `tasks` 表，main process 给 `tasks` 表补了：

```sql
ALTER TABLE tasks ADD COLUMN board_id TEXT DEFAULT 'default'
```

Hermes CLI / dispatcher / worker 侧任务通常会使用 board slug，例如：

```text
tasks.board_id = default
```

### 6.3 当前 rowToKanbanTask 映射

当前 row 映射类似：

```js
boardId: row.board_id || 'default'
```

因此 UI task 的 boardId 通常是：

```text
task.boardId = default
```

### 6.4 当前 boards() 返回值

如果 `boards()` 返回：

```js
{
  id: r.id,
  title: r.title,
  ...
}
```

则 default board 的 UI id 可能是：

```text
board.id = 随机 id
```

Renderer 中：

```ts
activeBoardId = board.id
```

最终过滤条件变成：

```text
task.boardId === activeBoardId
```

也就是：

```text
default === 随机 id
```

结果为 false。

## 7. 问题影响

该问题会影响以下场景：

### 7.1 Running task 不显示

Hermes CLI / dispatcher / worker 创建或更新的 running task 使用 `board_id = default`，但 Desktop active board 使用随机 id，导致 running task 不显示。

### 7.2 Manual task 与 CLI task 分裂

如果 Desktop 创建 task 时使用随机 board id，而 CLI 创建 task 时使用 slug，则同一个“Default Board”下会出现两类 task：

```text
Desktop task: board_id = <random-board-id>
CLI task:     board_id = default
```

两者不会出现在同一个 board 过滤结果中。

### 7.3 Chat -> Kanban task 可能显示不稳定

Chat message 创建 task 如果写入：

```text
boardId = default
```

但 Desktop active board 是随机 id，则创建成功后也可能不显示在当前 board 中。

### 7.4 后续 Agent/Cron 集成会继续出错

如果不统一 board identity，后续实现：

- Agent plan -> Kanban
- Agent running sync
- Cron failure -> blocked task

都会继续遇到 task 属于某个 slug，但 UI active board 使用随机 id 的问题。

## 8. 修复目标

### 8.1 统一 Board identity

Desktop Renderer 层应统一使用 board slug 作为 board id。

目标语义：

```text
KanbanBoard.id       = slug
KanbanTask.boardId   = tasks.board_id
Default board id     = default
Non-default board id = board slug
```

也就是说：

```text
board.id === board.slug === task.boardId
```

### 8.2 保留 DB 内部 id

`kanban_boards.id` 可以继续保留为内部主键。

但 Renderer API 不应该把内部随机 id 暴露为业务 identity。

建议：

```text
DB internal id: kanban_boards.id
Business id:    kanban_boards.slug
Renderer id:    KanbanBoard.id = slug
Task board id:  tasks.board_id = slug
```

## 9. 推荐修复方案

### 9.1 修改 `boards()` 返回值

当前可能是：

```js
return {
  id: r.id,
  title: r.title,
  description: r.description,
  createdAt: r.created_at
}
```

建议改为：

```js
return {
  id: r.slug,
  title: r.title,
  description: r.description,
  createdAt: r.created_at
}
```

### 9.2 修改 `createBoard()` 返回值

当前 create board 内部可以继续生成随机 `id` 作为 DB 主键。

但返回给 Renderer 时，应返回 slug：

```js
return {
  id: slug,
  title: safeTitle,
  description: safeDesc,
  createdAt: now
}
```

### 9.3 修改 `deleteBoard()` 参数语义

如果 Renderer 传入的是 slug，则 delete handler 应改成：

```sql
UPDATE tasks SET board_id = 'default' WHERE board_id = ?
DELETE FROM kanban_boards WHERE slug = ?
```

而不是：

```sql
DELETE FROM kanban_boards WHERE id = ?
```

### 9.4 修改 `tasks(boardId)` 查询语义

`tasks(boardId)` 应明确 boardId 为 slug。

```sql
SELECT * FROM tasks
WHERE board_id = ? AND archived = 0
ORDER BY sort_order ASC, created_at ASC
```

### 9.5 保持 `rowToKanbanTask()` 映射

`rowToKanbanTask()` 可以保持：

```js
boardId: row.board_id || 'default'
```

因为它已经符合 slug 语义。

### 9.6 修复现有数据兼容

如果历史 Desktop task 已经写入随机 board id，需要迁移。

推荐迁移逻辑：

```sql
UPDATE tasks
SET board_id = (
  SELECT slug FROM kanban_boards WHERE kanban_boards.id = tasks.board_id
)
WHERE board_id IN (SELECT id FROM kanban_boards)
```

伪代码：

```js
const boardRows = db.prepare('SELECT id, slug FROM kanban_boards').all()
for (const board of boardRows) {
  if (board.id !== board.slug) {
    db.prepare('UPDATE tasks SET board_id = ? WHERE board_id = ?').run(board.slug, board.id)
  }
}
```

该迁移应在 `ensureKanbanSchema()` 中执行，并且要幂等。

## 10. 推荐实现顺序

### Step 1：修复 API identity

修改：

```text
apps/desktop/electron/main.cjs
```

完成：

- `boards()` 返回 `id: slug`。
- `createBoard()` 返回 `id: slug`。
- `deleteBoard()` 使用 slug 删除。
- `tasks(boardId)` 明确 boardId 是 slug。

### Step 2：增加历史数据迁移

修改：

```text
apps/desktop/electron/main.cjs
```

完成：

- 在 `ensureKanbanSchema()` 中把历史随机 board id 迁移到 slug。
- 确保迁移可以重复执行。

### Step 3：检查 Renderer 假设

修改：

```text
apps/desktop/src/app/kanban/index.tsx
```

检查：

- `activeBoardId` 是否仍然默认 `'default'`。
- `createTask` 是否传入 slug。
- `deleteBoard` 是否传入 slug。
- `reorderTasks` 是否传入 slug。

预期：Renderer 不需要大改，因为其 board id 语义会被 main process 修正为 slug。

### Step 4：补充测试

建议新增或更新测试：

```text
apps/desktop/electron/kanban.test.cjs
```

测试用例：

1. default board 返回 `id = 'default'`。
2. CLI-style task `board_id = 'default'` 会显示在 default board。
3. running task 不被过滤。
4. 历史 task 使用随机 board id 时，会被迁移到 slug。
5. 删除 non-default board 时，tasks 回到 `default`。

## 11. 验收标准

### 11.1 数据验收

- `boards()` 返回的 default board id 是 `default`。
- `allTasks()` 返回 running task。
- `KanbanView` 中 `task.boardId === activeBoardId` 对 default running task 成立。
- 旧数据中使用随机 board id 的 task 被迁移到 slug。

### 11.2 UI 验收

- 打开 `#/kanban` 能看到当前 running task。
- running task 出现在 Running 列。
- running task 的 source/assignee/session metadata 不阻塞渲染。
- 手动创建 task、Chat 创建 task、CLI 创建 task 都能出现在同一个 Default Board。

### 11.3 回归验收

- Board 创建仍可用。
- Board 删除仍可用。
- Task 创建、编辑、归档、删除仍可用。
- 拖拽排序仍可用。
- `npm run typecheck` 通过。
- `npm run lint` 通过。
- `npm run test:ui` 或相关 Electron test 通过。

## 12. 不要重复开发清单

修复本问题时不要做以下事情：

- 不要新建第二个 Kanban 页面。
- 不要新建 running task 专用页面。
- 不要新增第二套 running task store。
- 不要绕过 `window.hermesDesktop.kanban.*` 直接访问 SQLite。
- 不要把 `running` 特判成独立数据源。
- 不要把这个问题放到 Chat/Agent integration 里先做自动同步。
- 不要回退到 `kanban.json`。

正确方向是：

```text
先修 board identity
再做 Agent / Cron 同步
```

## 13. 与其他文档的关系

### 13.1 与 `kanban.md`

本修复改变的是当前实现细节和数据 identity 语义。

修复完成后，应同步更新 `kanban.md`：

- 本地持久化章节从 JSON 改为 SQLite。
- Board identity 说明中明确：Renderer 层 `KanbanBoard.id` 使用 slug。

### 13.2 与 `gui-remediation-plan.md`

本问题不是 GUI 布局问题。

即使 GUI 列宽、详情面板和 droppable 都已修复，如果 board identity 不一致，running task 仍会消失。

因此该问题属于数据加载/identity 层，而不是 GUI 层。

### 13.3 与 `conversation-task-integration.md`

本问题是 Agent/Chat/Cron 集成的前置依赖。

如果 board identity 不统一，后续任何从外部来源写入的 task 都可能无法显示。

因此必须先完成本文修复，再继续做：

- Agent plan 批量导入。
- Agent mirrored sync。
- Cron failure task。

## 14. 当前结论

当前 Kanban 已完成 Desktop MVP 的大部分基础能力，但存在一个 P0 数据 identity 缺陷：

```text
Desktop board 使用随机 id
CLI / Agent task 使用 board slug/default
Renderer 按 boardId 过滤
=> running task 可能被过滤掉
```

在提交给 Hermes Agent 上游作为完整 MVP 前，建议先修复该问题。

如果不修复，Kanban 可能只能稳定显示 Desktop 手动创建的任务，却不能可靠显示 Hermes / Agent 正在执行的任务。这会直接影响 Kanban 作为 Hermes 工作流任务面板的核心价值。
