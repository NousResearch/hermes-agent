# Hermes Desktop Kanban 功能开发文档

## 1. 背景

Hermes Desktop 是 Hermes Agent 的原生桌面端，当前 desktop app 基于 Electron、Vite、React、TypeScript 构建。Kanban 功能用于在 Hermes Desktop 中提供轻量级任务管理能力，让用户可以围绕项目、会话、自动化任务、agent workflow 维护任务状态、优先级、负责人和评论。

本功能的核心定位不是替代完整项目管理系统，而是在 Hermes 工作流内部提供一个和聊天、agent、文件、cron/automation 能联动的本地任务面板。

## 2. 目标

### 2.1 产品目标

- 在 Hermes Desktop 内新增 Kanban 页面。
- 支持用户创建和切换多个 Board。
- 支持任务从 Todo 到 Done 的状态流转。
- 支持拖拽任务卡片改变状态和顺序。
- 支持任务详情、优先级、负责人、评论、归档。
- 后续支持从 Chat、Agent、Cron 自动创建或更新任务。

### 2.2 技术目标

- Renderer 层通过 React 页面实现 Kanban UI。
- Electron preload 层暴露安全的 `window.hermesDesktop.kanban` API。
- Electron main process 负责数据读写，不允许 Renderer 直接访问文件系统。
- 初期使用本地 JSON 持久化，后续可迁移到 SQLite 或 Hermes backend API。
- 保持和现有 Hermes Desktop UI、主题、路由、状态栏风格一致。

## 3. 当前接入点

### 3.1 路由

当前 Kanban 页面建议接入到 desktop 主路由：

```tsx
const KanbanView = lazy(async () => ({ default: (await import('./kanban')).KanbanView }))
```

并在 `Routes` 中注册：

```tsx
<Route
  element={
    <Suspense fallback={null}>
      <KanbanView setStatusbarItemGroup={setStatusbarItemGroup} />
    </Suspense>
  }
  path="kanban"
/>
```

推荐访问路径：

```text
#/kanban
```

### 3.2 Renderer 页面

推荐页面文件：

```text
apps/desktop/src/app/kanban/index.tsx
```

页面职责：

- 加载 Board 列表和 Task 列表。
- 渲染 Board selector。
- 渲染 Kanban 状态列。
- 渲染 Task card。
- 打开 Task 创建/编辑弹窗。
- 打开 Task detail panel。
- 处理拖拽排序和跨列状态变更。
- 调用 `window.hermesDesktop.kanban.*` 完成数据读写。

### 3.3 Preload API

推荐通过 `apps/desktop/electron/preload.cjs` 暴露 Kanban API：

```js
kanban: {
  boards: () => ipcRenderer.invoke('hermes:kanban:boards'),
  createBoard: data => ipcRenderer.invoke('hermes:kanban:createBoard', data),
  deleteBoard: id => ipcRenderer.invoke('hermes:kanban:deleteBoard', id),
  tasks: boardId => ipcRenderer.invoke('hermes:kanban:tasks', boardId),
  allTasks: () => ipcRenderer.invoke('hermes:kanban:allTasks'),
  createTask: data => ipcRenderer.invoke('hermes:kanban:createTask', data),
  updateTask: (id, data) => ipcRenderer.invoke('hermes:kanban:updateTask', id, data),
  deleteTask: id => ipcRenderer.invoke('hermes:kanban:deleteTask', id),
  comments: taskId => ipcRenderer.invoke('hermes:kanban:comments', taskId),
  addComment: data => ipcRenderer.invoke('hermes:kanban:addComment', data),
  deleteComment: id => ipcRenderer.invoke('hermes:kanban:deleteComment', id)
}
```

后续为了持久化拖拽顺序，建议新增：

```js
reorderTasks: (boardId, updates) =>
  ipcRenderer.invoke('hermes:kanban:reorderTasks', boardId, updates)
```

### 3.4 TypeScript 全局类型

推荐在 `apps/desktop/src/global.d.ts` 中声明：

```ts
export interface KanbanBoard {
  id: string
  title: string
  description: string
  createdAt: number
  updatedAt?: number
}

export interface KanbanTask {
  id: string
  boardId: string
  title: string
  description: string
  status: KanbanStatus
  priority: KanbanPriority
  assignee: string
  createdBy: string
  createdAt: number
  updatedAt: number
  archived: boolean
  order: number
  labels?: string[]
  sessionId?: string
  source?: 'manual' | 'chat' | 'agent' | 'cron'
}

export interface KanbanComment {
  id: string
  taskId: string
  author: string
  body: string
  createdAt: number
}

export type KanbanStatus = 'todo' | 'ready' | 'running' | 'review' | 'done' | 'blocked'
export type KanbanPriority = 'low' | 'medium' | 'high'
```

## 4. 数据模型

### 4.1 Board

Board 表示一个看板，例如“Default Board”“Release Plan”“Bug Triage”。

字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `id` | `string` | 是 | Board 唯一 ID |
| `title` | `string` | 是 | Board 名称 |
| `description` | `string` | 否 | Board 描述 |
| `createdAt` | `number` | 是 | 创建时间戳 |
| `updatedAt` | `number` | 否 | 更新时间戳 |

### 4.2 Task

Task 是 Kanban 的核心对象。

字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `id` | `string` | 是 | Task 唯一 ID |
| `boardId` | `string` | 是 | 所属 Board |
| `title` | `string` | 是 | 标题 |
| `description` | `string` | 否 | 描述 |
| `status` | `KanbanStatus` | 是 | 所属状态列 |
| `priority` | `KanbanPriority` | 是 | 优先级 |
| `assignee` | `string` | 否 | 负责人 |
| `createdBy` | `string` | 否 | 创建者 |
| `createdAt` | `number` | 是 | 创建时间 |
| `updatedAt` | `number` | 是 | 更新时间 |
| `archived` | `boolean` | 是 | 是否归档 |
| `order` | `number` | 是 | 同一列内排序 |
| `labels` | `string[]` | 否 | 标签 |
| `sessionId` | `string` | 否 | 关联的 Hermes session |
| `source` | `'manual' \| 'chat' \| 'agent' \| 'cron'` | 否 | 任务来源 |

### 4.3 Comment

Comment 表示任务评论。

字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `id` | `string` | 是 | Comment 唯一 ID |
| `taskId` | `string` | 是 | 所属 Task |
| `author` | `string` | 否 | 评论作者 |
| `body` | `string` | 是 | 评论内容 |
| `createdAt` | `number` | 是 | 创建时间 |

## 5. 状态列设计

默认状态列：

```ts
const STATUS_COLUMNS = [
  { id: 'todo', label: 'Todo' },
  { id: 'ready', label: 'Ready' },
  { id: 'running', label: 'Running' },
  { id: 'review', label: 'Review' },
  { id: 'done', label: 'Done' },
  { id: 'blocked', label: 'Blocked' }
] as const
```

语义说明：

| 状态 | 说明 |
| --- | --- |
| `todo` | 待处理，尚未排期 |
| `ready` | 已准备，可以开始执行 |
| `running` | 正在执行 |
| `review` | 已完成初稿，等待确认或 review |
| `done` | 已完成 |
| `blocked` | 被阻塞，需要用户或外部条件处理 |

## 6. 本地持久化设计

### 6.1 数据库路径

当前使用 SQLite 作为 Kanban 存储层，与 Hermes CLI 共享同一个数据库文件：

```js
const KANBAN_DB_PATH = path.join(HERMES_HOME, 'kanban.db')
```

### 6.2 数据库表结构

#### kanban_boards — 看板表

```sql
CREATE TABLE IF NOT EXISTS kanban_boards (
  id TEXT PRIMARY KEY,
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  description TEXT DEFAULT '',
  created_at INTEGER NOT NULL
)
```

说明：

- `id` 是内部主键，由 `Date.now().toString(36) + crypto.randomBytes(8).toString('hex')` 生成。
- `slug` 是业务 identity，从 title 经过 lowercase + 去除非字母数字字符生成。
- Renderer 层使用 `slug` 作为 `KanbanBoard.id`。

#### tasks — 任务表（Hermes CLI 和 Desktop 共用）

Desktop 通过 ALTER TABLE 为 CLI 的 `tasks` 表补充以下列：

```sql
ALTER TABLE tasks ADD COLUMN board_id TEXT DEFAULT 'default'
ALTER TABLE tasks ADD COLUMN archived INTEGER NOT NULL DEFAULT 0
ALTER TABLE tasks ADD COLUMN updated_at INTEGER
ALTER TABLE tasks ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0
ALTER TABLE tasks ADD COLUMN source TEXT DEFAULT 'manual'
ALTER TABLE tasks ADD COLUMN session_id TEXT
ALTER TABLE tasks ADD COLUMN profile_id TEXT
ALTER TABLE tasks ADD COLUMN message_id TEXT
ALTER TABLE tasks ADD COLUMN assignee_type TEXT DEFAULT 'unassigned'
ALTER TABLE tasks ADD COLUMN assignee_label TEXT
ALTER TABLE tasks ADD COLUMN sync_mode TEXT DEFAULT 'manual'
```

所有 ALTER TABLE 是幂等的（列已存在时静默跳过）。

#### task_comments — 评论表

CLI 侧已有，Desktop 直接使用：

```sql
CREATE TABLE IF NOT EXISTS task_comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  author TEXT,
  body TEXT,
  created_at INTEGER NOT NULL
)
```

### 6.3 数据模型映射

DB 行通过 `rowToKanbanTask()` 映射为 UI 友好的 `KanbanTask`：

```js
function rowToKanbanTask(row) {
  return {
    id: row.id,
    boardId: row.board_id || 'default',    // board slug
    title: row.title,
    description: row.body || '',
    status: row.status || 'todo',
    priority: PRIORITY_INT_TO_STR[row.priority] || 'medium',
    assignee: row.assignee || '',
    createdBy: row.created_by || '',
    createdAt: row.created_at,
    updatedAt: row.updated_at || row.created_at,
    archived: Boolean(row.archived),
    order: row.sort_order || 0,
    source: row.source || 'manual',
    sessionId: row.session_id || undefined,
    profileId: row.profile_id || undefined,
    messageId: row.message_id || undefined,
    assigneeType: row.assignee_type || 'unassigned',
    assigneeLabel: row.assignee_label || undefined,
    syncMode: row.sync_mode || 'manual'
  }
}
```

### 6.4 连接管理

首次访问时延迟初始化 SQLite 连接：

```js
function getKanbanDb() {
  if (_kanbanDb) {
    try { _kanbanDb.prepare('SELECT 1').all(); return _kanbanDb }
    catch { _kanbanDb = null }  // 断连后重建
  }
  const { DatabaseSync } = require('node:sqlite')
  _kanbanDb = new DatabaseSync(KANBAN_DB_PATH)
  _kanbanDb.exec('PRAGMA journal_mode=WAL')
  _kanbanDb.exec('PRAGMA foreign_keys=ON')
  ensureKanbanSchema(_kanbanDb)
  return _kanbanDb
}
```

### 6.5 Schema 初始化与数据迁移

`ensureKanbanSchema()` 在首次连接时执行：

1. 创建 `kanban_boards` 表（如不存在）。
2. 确保 `slug = 'default'` 的默认 board 存在。
3. 为 `tasks` 表补充 Desktop 所需的列。
4. **历史数据迁移**：将旧 task 的 `board_id` 从随机 ID 统一为对应 board 的 slug，保证 Renderer 侧 `task.boardId === activeBoardId` 正确匹配。

### 6.6 与旧 JSON 方案的差异

> 早期方案使用 `HERMES_HOME/kanban.json`，当前已全部迁移到 SQLite。
>
> 后续开发不要再实现 JSON 写入逻辑，也不要再新增 `kanban.json`。

## 7. IPC 设计

### 7.1 Board API

#### `hermes:kanban:boards`

返回所有 board。

```ts
() => Promise<KanbanBoard[]>
```

#### `hermes:kanban:createBoard`

创建 board。

```ts
(data: { title: string; description?: string }) => Promise<KanbanBoard>
```

#### `hermes:kanban:deleteBoard`

删除 board，同时删除其 task 和 comment。

```ts
(id: string) => Promise<{ ok: boolean }>
```

### 7.2 Task API

#### `hermes:kanban:tasks`

返回指定 board 的未归档 task。

```ts
(boardId: string) => Promise<KanbanTask[]>
```

#### `hermes:kanban:allTasks`

返回所有未归档 task。

```ts
() => Promise<KanbanTask[]>
```

#### `hermes:kanban:createTask`

创建 task。

```ts
(data: {
  boardId: string
  title: string
  description?: string
  status?: KanbanStatus
  priority?: KanbanPriority
  assignee?: string
  labels?: string[]
  sessionId?: string
  source?: KanbanTask['source']
}) => Promise<KanbanTask>
```

#### `hermes:kanban:updateTask`

更新 task。

```ts
(id: string, data: Partial<KanbanTask>) => Promise<KanbanTask>
```

#### `hermes:kanban:deleteTask`

删除 task，同时删除其 comment。

```ts
(id: string) => Promise<{ ok: boolean }>
```

#### `hermes:kanban:reorderTasks`

批量更新排序和状态。

```ts
(
  boardId: string,
  updates: Array<{ id: string; status: KanbanStatus; order: number }>
) => Promise<KanbanTask[]>
```

### 7.3 Comment API

#### `hermes:kanban:comments`

返回 task 的评论。

```ts
(taskId: string) => Promise<KanbanComment[]>
```

#### `hermes:kanban:addComment`

新增评论。

```ts
(data: { taskId: string; author: string; body: string }) => Promise<KanbanComment>
```

#### `hermes:kanban:deleteComment`

删除评论。

```ts
(id: string) => Promise<{ ok: boolean }>
```

## 8. 输入校验和安全

Renderer 传入的数据不能直接信任。所有 IPC handler 都应该在 main process 做最小校验。

推荐校验规则：

```js
const VALID_STATUSES = new Set(['todo', 'ready', 'running', 'review', 'done', 'blocked'])
const VALID_PRIORITIES = new Set(['low', 'medium', 'high'])

function sanitizeString(value, maxLength) {
  return String(value || '').trim().slice(0, maxLength)
}

function sanitizeStatus(value) {
  return VALID_STATUSES.has(value) ? value : 'todo'
}

function sanitizePriority(value) {
  return VALID_PRIORITIES.has(value) ? value : 'medium'
}

function sanitizeTaskInput(input = {}) {
  return {
    title: sanitizeString(input.title, 200) || 'Untitled',
    description: String(input.description || '').slice(0, 5000),
    status: sanitizeStatus(input.status),
    priority: sanitizePriority(input.priority),
    assignee: sanitizeString(input.assignee, 120),
    labels: Array.isArray(input.labels)
      ? input.labels.map(label => sanitizeString(label, 40)).filter(Boolean).slice(0, 20)
      : []
  }
}
```

注意事项：

- 不允许 Renderer 传入任意文件路径。
- 不允许使用 task/comment 字段拼接 shell 命令。
- title、description、comment body 必须限制长度。
- status、priority 必须白名单校验。
- 删除 board/task/comment 前必须确认对象存在。

## 9. Renderer UI 设计

### 9.1 页面结构

```text
KanbanView
  Header toolbar
    Board selector
    New board button
    New task button
    Task count

  Board area
    DndContext
      KanbanColumn(todo)
      KanbanColumn(ready)
      KanbanColumn(running)
      KanbanColumn(review)
      KanbanColumn(done)
      KanbanColumn(blocked)

  TaskDialog
  TaskDetailPanel
  DeleteConfirmDialog
```

### 9.2 Header toolbar

Header toolbar 包含：

- Board selector
- 创建 Board 按钮
- 创建 Task 按钮
- 当前 Board task count
- 后续可加入搜索框、过滤器、归档入口

### 9.3 Column

Column 职责：

- 显示状态名和数量。
- 渲染当前状态下的任务。
- 支持空列 drop。
- 支持同列排序。

建议每个 column 都注册 droppable id：

```ts
const columnDroppableId = `column:${column.id}`
```

### 9.4 Task card

Task card 展示：

- 标题
- 描述摘要
- 优先级 badge
- assignee
- blocked 状态图标
- hover 后显示编辑和删除按钮

需要支持：

- 点击卡片打开详情面板。
- 拖拽卡片改变状态或顺序。
- 点击编辑按钮打开编辑弹窗。
- 点击删除按钮打开删除确认弹窗。

### 9.5 Task detail panel

详情面板展示：

- 标题
- 描述
- 当前状态
- 当前优先级
- assignee
- 快速切换状态按钮
- 归档按钮
- 评论列表
- 新增评论输入框

### 9.6 Empty state

当没有 Board 时：

```text
No boards yet. Create your first board to start tracking work.
```

当 Board 没有任务时：

```text
No tasks yet. Create a task or ask Hermes to generate tasks from a plan.
```

## 10. 拖拽排序设计

### 10.1 当前最小行为

- 跨列拖拽：更新 `task.status`。
- 同列拖拽：更新 `task.order`。
- 拖到空列：更新 `task.status`，并把 `order` 放到该列末尾。

### 10.2 推荐算法

```ts
function reorderBoardTasks(
  tasks: KanbanTask[],
  activeId: string,
  overId: string
): Array<{ id: string; status: KanbanStatus; order: number }> {
  // 1. 找到 active task。
  // 2. 根据 overId 判断目标是 task 还是 column。
  // 3. 从原列移除 active task。
  // 4. 插入目标列对应位置。
  // 5. 重新计算受影响列的 order。
  // 6. 返回 updates。
}
```

### 10.3 Optimistic update

拖拽结束后先本地更新 UI，再调用 IPC 持久化：

```ts
const previousTasks = tasks
setTasks(nextTasks)

try {
  await window.hermesDesktop.kanban.reorderTasks(activeBoardId, updates)
} catch {
  setTasks(previousTasks)
  notifyError('Failed to reorder tasks')
}
```

## 11. 和 Hermes Agent 的集成方向

### 11.1 从 Chat 创建任务

在 Chat message 菜单中增加：

```text
Create Kanban Task
```

创建数据：

```ts
{
  title: selectedText.slice(0, 120),
  description: selectedText,
  status: 'todo',
  priority: 'medium',
  source: 'chat',
  sessionId: currentSessionId
}
```

### 11.2 从 Agent plan 创建任务

当 agent 生成 todo list 或 execution plan 时，可以提供：

```text
Send plan to Kanban
```

每个 plan item 转换为一个 Task。

### 11.3 Agent 自动更新任务状态

长期方向是给 Hermes Agent 增加 Kanban tool：

```json
{
  "name": "kanban.update_task",
  "arguments": {
    "taskId": "...",
    "status": "running"
  }
}
```

用途：

- agent 开始执行任务时把任务移到 `running`。
- agent 完成初稿时把任务移到 `review`。
- 用户确认后移到 `done`。
- 工具失败或缺少权限时移到 `blocked`。

### 11.4 Cron / Automation 关联

Cron job 失败时可自动创建 blocked task：

```ts
{
  title: `Cron failed: ${jobName}`,
  description: errorMessage,
  status: 'blocked',
  priority: 'high',
  source: 'cron'
}
```

## 12. 迭代计划

### Phase 1：补齐基础闭环

- 新增或确认 `/kanban` 路由。
- 新增 Board selector。
- 新增 Task create/edit/delete/archive。
- 新增 comment create/delete。
- 支持跨列拖拽。
- 点击 task card 打开详情面板。
- 数据写入 `HERMES_HOME/kanban.json`。

### Phase 2：稳定性和数据安全

- 增加 `schemaVersion`。
- 增加 migration。
- 增加输入校验。
- 使用原子写入。
- 支持 `order` 字段。
- 支持同列排序持久化。
- 支持拖到空列。
- 增加 corrupted JSON fallback 和日志。

### Phase 3：体验增强

- 搜索任务。
- 按负责人筛选。
- 按优先级筛选。
- 按标签筛选。
- 显示归档任务。
- 支持恢复归档任务。
- 状态栏展示当前 Board 信息。
- 支持快捷键创建任务。

### Phase 4：Hermes workflow 集成

- 从 Chat message 创建任务。
- 从 agent plan 批量创建任务。
- 任务关联 session。
- Agent tool 更新任务状态。
- Cron failure 自动创建 blocked task。

### Phase 5：数据层升级

- 迁移到 SQLite 或 Hermes backend API。
- 支持 profile-aware Kanban 数据。
- 支持多设备同步。
- 支持冲突解决。

## 13. 验收标准

### 13.1 功能验收

- 可以打开 `#/kanban`。
- 可以创建 Board。
- 可以切换 Board。
- 可以创建 Task。
- 可以编辑 Task。
- 可以删除 Task。
- 可以归档 Task。
- 可以添加评论。
- 可以删除评论。
- 可以拖拽 Task 跨列改变状态。
- 可以拖拽 Task 在同列内改变顺序。
- 可以将 Task 拖到空列。
- 重启 Hermes Desktop 后数据仍然存在。

### 13.2 技术验收

- `npm run typecheck` 通过。
- `npm run lint` 通过。
- `npm run test:ui` 通过。
- Renderer 不直接访问文件系统。
- IPC 参数有校验。
- JSON 写入是原子的。
- 类型定义和 preload API 保持一致。
- 删除 board/task 时不会留下孤立 comments。

### 13.3 手动测试用例

#### Case 1：创建 Board

1. 打开 `#/kanban`。
2. 点击 New board。
3. 输入 `Release Plan`。
4. 点击 Create。
5. Board selector 显示 `Release Plan`。
6. 重启应用后 Board 仍然存在。

#### Case 2：创建 Task

1. 选择任意 Board。
2. 点击 New Task。
3. 输入标题、描述、优先级、负责人。
4. 点击 Create Task。
5. Task 出现在对应状态列。
6. 重启应用后 Task 仍然存在。

#### Case 3：拖拽状态变更

1. 创建一个 `todo` task。
2. 拖到 `running` 列。
3. Task 状态更新为 `running`。
4. 重启后 Task 仍在 `running` 列。

#### Case 4：同列排序

1. 在同一列创建三个 task。
2. 调整顺序。
3. 刷新或重启后顺序保持一致。

#### Case 5：评论

1. 点击 Task card 打开详情。
2. 添加评论。
3. 评论出现在列表中。
4. 删除评论。
5. 评论从列表中移除。

#### Case 6：归档

1. 打开 Task detail panel。
2. 点击 Archive Task。
3. Task 从当前 Board 隐藏。
4. JSON 中 task 保留且 `archived: true`。

## 14. 建议改动清单

```text
apps/desktop/src/app/kanban/index.tsx
  - 确认 KanbanView 页面结构
  - 点击 Task card 打开详情面板
  - 增加 droppable column
  - 增加空列 drop 支持
  - 增加同列排序持久化
  - 增加 optimistic update 和失败回滚

apps/desktop/electron/main.cjs
  - 增加 schemaVersion
  - 增加 normalizeKanbanData
  - 增加输入校验
  - 改为原子写入
  - 增加 reorderTasks IPC

apps/desktop/electron/preload.cjs
  - 暴露 reorderTasks

apps/desktop/src/global.d.ts
  - 增加 KanbanStatus、KanbanPriority 类型
  - 给 KanbanTask 增加 order、labels、sessionId、source
  - 给 window.hermesDesktop.kanban 增加 reorderTasks 类型

apps/desktop/src/app/chat/*
  - 后续增加从聊天创建 Kanban task 的入口
```

## 15. 风险

### 15.1 JSON 损坏

如果直接写入 `kanban.json`，进程异常退出可能导致文件损坏。需要使用临时文件加 rename 的原子写入策略。

### 15.2 并发写入

多个 IPC 同时写入可能互相覆盖。短期可以通过同步读写降低概率，长期建议引入写队列或 SQLite。

### 15.3 拖拽边界

只基于 `over.id` 判断目标 task 时，空列 drop 不可靠。需要为 column 单独注册 droppable id。

### 15.4 类型漂移

`preload.cjs`、`global.d.ts`、Renderer 调用、main IPC handler 必须保持一致。每次新增 API 都需要同步更新。

### 15.5 后续同步冲突

如果以后支持远端同步，需要提前设计 `updatedAt`、`deletedAt`、`version` 或 conflict resolution，否则多端编辑会产生覆盖问题。

## 16. 推荐优先级

当前最推荐先做：

1. 修复 Task card 点击打开详情。
2. 给 Task 增加 `order`。
3. 实现 `reorderTasks`。
4. 支持拖到空列。
5. main process 增加输入校验。
6. JSON 改为原子写入。
7. 增加 `schemaVersion` 和 migration。

完成以上内容后，Kanban 功能就具备可用的 MVP 闭环。