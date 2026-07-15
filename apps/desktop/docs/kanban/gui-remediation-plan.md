# Kanban GUI 修复与落地计划

> 本文是 `apps/desktop/docs/kanban.md` 的子级实施文档。
> 
> `kanban.md` 负责定义 Kanban 功能总目标、数据模型、IPC、长期演进和验收标准；本文只聚焦当前 GUI 显示和交互落地问题，不重复定义总功能规格，避免和主开发文档冲突。

## 1. 文档层级

当前 Kanban 文档层级建议如下：

```text
apps/desktop/docs/
  kanban.md                          # Kanban 功能总纲：目标、模型、IPC、长期路线
  kanban/
    gui-remediation-plan.md           # 当前文档：GUI 显示问题与修复计划
```

职责边界：

| 文档 | 层级 | 职责 |
| --- | --- | --- |
| `kanban.md` | 一级总纲 | 定义 Kanban 功能范围、数据模型、IPC、迭代路线、总体验收标准 |
| `kanban/gui-remediation-plan.md` | 二级实施文档 | 定义当前 GUI 问题、组件层级、修复方案、局部验收标准 |

本文不得覆盖以下内容：

- Kanban 的总产品定位。
- 长期 Agent/Cron 集成路线。
- 完整数据模型总设计。
- IPC 总 API 设计。

本文只补充：

- 当前界面显示问题。
- 组件结构调整方案。
- 拖拽和详情面板修复方案。
- GUI 层面的最小可用验收标准。

## 2. 当前问题概述

当前 Kanban 已经完成基础接入：

- `/kanban` 路由已接入 Desktop 主路由。
- Renderer 已有 `KanbanView`。
- preload 已暴露 `window.hermesDesktop.kanban`。
- Electron main process 已有本地 JSON 读写 IPC。

但 GUI 仍处于原型状态，主要问题集中在：

1. 看板列被压缩，横向布局不符合 Kanban 预期。
2. 详情面板使用 absolute 定位，容易覆盖主界面。
3. Task card 不能点击打开详情。
4. 空列不能作为可靠拖拽目标。
5. 同列排序没有真正落地。
6. Header 与 Board 内容区间距层级不一致。
7. 视觉结构没有清晰区分 Toolbar、Board、Detail Panel。

## 3. 当前组件层级问题

### 3.1 当前结构

当前 `KanbanView` 近似结构：

```text
KanbanView
  div.flex.h-full.flex-col.overflow-hidden
    Header toolbar
    Board columns area
      DndContext
        KanbanColumn[]
    TaskDetailPanel absolute right overlay
    TaskDialog
    DeleteConfirmDialog
```

主要问题：

- 根容器不是 `relative`，但详情面板使用 `absolute`。
- Board columns 和 Detail panel 不在同一个 flex 布局上下文中。
- 详情面板覆盖而不是占位。
- Column 没有固定或最小宽度。
- Column 只包含 sortable context，没有 droppable column container。

### 3.2 目标结构

目标结构应拆成清晰三层：

```text
KanbanView
  KanbanPageRoot
    KanbanToolbar
    KanbanWorkspace
      KanbanBoardPane
        DndContext
          KanbanColumn[]
      KanbanDetailPane optional
    KanbanDialogs
      TaskDialog
      DeleteConfirmDialog
```

对应职责：

| 层级 | 组件 | 职责 |
| --- | --- | --- |
| Page | `KanbanView` | 数据加载、状态管理、事件编排 |
| Toolbar | `KanbanToolbar` | Board 切换、新建 Board、新建 Task、统计信息 |
| Workspace | `KanbanWorkspace` | 管理 Board 和 Detail Pane 的左右布局 |
| Board | `KanbanBoardPane` | 横向滚动区域和 DnD 上下文 |
| Column | `KanbanColumn` | 状态列、droppable 区域、任务数量 |
| Card | `SortableTaskCard` | 任务摘要展示、选择、编辑、删除、拖拽 |
| Detail | `TaskDetailPanel` | 任务详情、评论、状态修改、归档 |
| Dialog | `TaskDialog` / `DeleteConfirmDialog` | 创建、编辑、删除确认 |

## 4. 修复目标

### 4.1 第一阶段目标

第一阶段只修 GUI 可用性，不扩大功能范围。

必须完成：

- 看板列具备合理宽度。
- 横向滚动正常。
- 详情面板不再覆盖主界面。
- 点击 Task card 可以打开详情。
- 空列可以接收拖拽。
- 跨列拖拽状态更新可靠。
- 同列排序至少能在前端视觉上立即生效。

暂不强制完成：

- 完整排序持久化。
- 标签筛选。
- 搜索。
- Agent 自动创建任务。
- Chat message 创建任务。

### 4.2 第二阶段目标

第二阶段补齐排序和数据一致性。

必须完成：

- 给 Task 增加 `order` 字段。
- 新增 `reorderTasks` IPC。
- 同列排序持久化。
- 跨列拖拽后重新计算目标列 order。
- 刷新和重启后排序保持一致。

## 5. 布局修复方案

### 5.1 Page root

当前：

```tsx
<div className="flex h-full flex-col overflow-hidden">
```

建议：

```tsx
<div className="flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-bg-primary)">
```

说明：

- `min-h-0` 保证子级滚动区域能正确收缩。
- `bg-(--ui-bg-primary)` 保证页面背景和 Hermes 主题一致。

### 5.2 Toolbar

Toolbar 应保持固定高度，不参与主内容滚动。

建议结构：

```tsx
<div className="flex shrink-0 items-center gap-3 border-b border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-4 py-2">
  {/* board selector / actions / stats */}
</div>
```

说明：

- 不建议在 Kanban Toolbar 使用正文页的 `PAGE_INSET_X`。
- Kanban 是工作区布局，左右 padding 应和 Board 区域一致。
- 推荐 Toolbar 和 Board 都使用 `px-4`。

### 5.3 Workspace

新增 Workspace 层：

```tsx
<div className="flex min-h-0 flex-1 overflow-hidden">
  <div className="min-w-0 flex-1 overflow-hidden">
    {/* board pane */}
  </div>

  {selectedTask && (
    <div className="w-80 shrink-0 border-l border-(--ui-stroke-secondary)">
      {/* detail panel */}
    </div>
  )}
</div>
```

说明：

- Detail panel 使用 flex 占位，不使用 absolute overlay。
- Board pane 自动让出右侧详情宽度。
- `min-w-0` 确保 Board pane 能被压缩并启用内部横向滚动。

### 5.4 Board pane

建议：

```tsx
<div className="h-full overflow-x-auto overflow-y-hidden p-4">
  <div className="flex h-full min-w-max gap-4">
    {/* columns */}
  </div>
</div>
```

说明：

- 外层负责横向滚动。
- 内层 `min-w-max` 保证 columns 不会被强行压缩。
- 每个 column 固定宽度或最小宽度。

### 5.5 Column 宽度

当前 Column 使用：

```tsx
<div className="flex h-full min-w-0 flex-1 flex-col overflow-hidden">
```

建议改为：

```tsx
<div className="flex h-full w-72 shrink-0 flex-col overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary)/40">
```

或：

```tsx
<div className="flex h-full min-w-[18rem] max-w-[22rem] flex-1 flex-col overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary)/40">
```

推荐第一种固定宽度，原因：

- Kanban 的主要交互是横向浏览。
- 固定列宽可保持任务卡片阅读体验稳定。
- 窄窗口下通过横向滚动解决，而不是压缩列宽。

## 6. 详情面板修复方案

### 6.1 问题

当前详情面板使用：

```tsx
{selectedTask && (
  <div className="absolute bottom-0 right-0 top-0 w-80">
    <TaskDetailPanel ... />
  </div>
)}
```

问题：

- 父容器没有明确 `relative`。
- 面板会覆盖 Board 内容。
- 面板可能覆盖 titlebar/statusbar 或其他 overlay。
- Board 不会为详情面板让出空间。

### 6.2 目标

详情面板应作为 Workspace 的右侧 pane：

```tsx
{selectedTask && (
  <aside className="h-full w-80 shrink-0 overflow-hidden border-l border-(--ui-stroke-secondary) bg-(--ui-bg-primary)">
    <TaskDetailPanel ... />
  </aside>
)}
```

### 6.3 选中任务状态同步

更新 task 后，需要同步 `selectedTask`：

```ts
setTasks(prev => prev.map(t => (t.id === updated.id ? updated : t)))
setSelectedTask(prev => (prev?.id === updated.id ? updated : prev))
```

归档或删除 selected task 后，需要关闭详情：

```ts
setSelectedTask(prev => (prev?.id === taskId ? null : prev))
```

## 7. Task card 交互修复方案

### 7.1 当前问题

当前 `SortableTaskCard` 只有：

- `onEdit`
- `onDelete`

没有 `onSelect`，根节点也没有 `onClick`。

这会导致 `TaskDetailPanel` 很难被打开。

### 7.2 目标接口

```tsx
function SortableTaskCard({
  task,
  onSelect,
  onEdit,
  onDelete
}: {
  task: KanbanTask
  onSelect: (task: KanbanTask) => void
  onEdit: (task: KanbanTask) => void
  onDelete: (task: KanbanTask) => void
})
```

### 7.3 目标行为

根节点点击打开详情：

```tsx
<div
  ref={setNodeRef}
  className={...}
  onClick={() => onSelect(task)}
  {...attributes}
  {...listeners}
>
```

编辑和删除按钮继续阻止冒泡：

```tsx
onClick={event => {
  event.stopPropagation()
  onEdit(task)
}}
```

### 7.4 注意拖拽与点击冲突

当前 PointerSensor 已设置 `activationConstraint: { distance: 4 }`，可以减少误拖拽。但为了手感更稳定，后续可考虑：

- 卡片整体可点击。
- 仅卡片 header 或 drag handle 触发拖拽。
- 编辑/删除按钮区域不触发拖拽。

第一阶段可先保持整卡拖拽。

## 8. Droppable Column 修复方案

### 8.1 当前问题

当前 `KanbanColumn` 只有 `SortableContext`，没有 `useDroppable`。

拖拽结束时通过 `overTask.status` 判断目标列：

```ts
const overTask = tasks.find(t => t.id === over.id)
if (overTask) {
  targetStatus = overTask.status
}
```

这导致：

- 拖到空列时没有 `overTask`。
- 拖到列空白区域时状态不变。
- 用户看到 `Drop tasks here`，但实际不一定能 drop。

### 8.2 目标实现

在 Column 中注册 droppable：

```tsx
import { useDroppable } from '@dnd-kit/core'

function KanbanColumn({ column, tasks, ...props }) {
  const { isOver, setNodeRef } = useDroppable({
    id: `column:${column.id}`,
    data: {
      type: 'column',
      status: column.id
    }
  })

  return (
    <div className="flex h-full w-72 shrink-0 flex-col overflow-hidden">
      <div>{/* header */}</div>
      <SortableContext items={taskIds} strategy={verticalListSortingStrategy}>
        <div ref={setNodeRef} className={cn('flex flex-1 flex-col gap-2 overflow-y-auto p-2', isOver && 'bg-(--ui-bg-tertiary)')}>
          {/* tasks */}
        </div>
      </SortableContext>
    </div>
  )
}
```

### 8.3 DragEnd 目标判断

拖拽结束时同时支持 task 和 column：

```ts
const overId = String(over.id)

if (overId.startsWith('column:')) {
  targetStatus = over.data.current?.status ?? draggedTask.status
} else if (overTask) {
  targetStatus = overTask.status
}
```

### 8.4 空列提示

空列提示应该保留，但 droppable 区域必须覆盖整个列内容区域。

```tsx
{tasks.length === 0 && (
  <div className="flex flex-1 items-center justify-center rounded-md border border-dashed border-(--ui-stroke-secondary)">
    <span className="text-[0.6875rem] text-(--ui-text-tertiary)">Drop tasks here</span>
  </div>
)}
```

## 9. 排序修复方案

### 9.1 第一阶段：前端视觉排序

当前同列排序只提示 `Task reordered`，没有更新任务顺序。

第一阶段至少应该更新本地 state：

```ts
const columnTasks = tasksByStatus[targetStatus] ?? []
const oldIdx = columnTasks.findIndex(t => t.id === active.id)
const newIdx = columnTasks.findIndex(t => t.id === over.id)

if (oldIdx >= 0 && newIdx >= 0 && oldIdx !== newIdx) {
  const reorderedColumnTasks = arrayMove(columnTasks, oldIdx, newIdx)
  setTasks(prev => {
    const otherTasks = prev.filter(t => t.status !== targetStatus || t.boardId !== activeBoardId)
    return [...otherTasks, ...reorderedColumnTasks]
  })
}
```

注意：这只是前端视觉排序，刷新后会丢失。

### 9.2 第二阶段：持久化排序

给 `KanbanTask` 增加：

```ts
order: number
```

查询时按：

```ts
status ASC, order ASC, updatedAt DESC
```

Renderer 中按：

```ts
tasks
  .filter(t => t.status === column.id)
  .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
```

新增 IPC：

```ts
reorderTasks: (
  boardId: string,
  updates: Array<{ id: string; status: KanbanStatus; order: number }>
) => Promise<KanbanTask[]>
```

Main process 中：

```js
ipcMain.handle('hermes:kanban:reorderTasks', (_event, boardId, updates) => {
  const data = readKanbanData()
  const updateMap = new Map(updates.map(update => [update.id, update]))
  const now = Date.now()

  data.tasks = data.tasks.map(task => {
    if (task.boardId !== boardId) return task
    const update = updateMap.get(task.id)
    if (!update) return task

    return {
      ...task,
      status: sanitizeStatus(update.status),
      order: Number.isFinite(update.order) ? update.order : task.order,
      updatedAt: now
    }
  })

  writeKanbanData(data)
  return data.tasks.filter(task => task.boardId === boardId && !task.archived)
})
```

## 10. 推荐修改顺序

### Step 1：布局先行

修改：

```text
apps/desktop/src/app/kanban/index.tsx
```

完成：

- 根容器增加 `min-h-0`。
- Toolbar 改为统一 `px-4`。
- 新增 Workspace flex 布局。
- Detail panel 从 absolute 改为 aside。
- Board pane 改为横向滚动。
- Column 改为固定宽度 `w-72 shrink-0`。

验收：

- 6 列不会被压扁。
- 窄窗口下可以横向滚动。
- 打开详情后 Board 不被覆盖，而是让出右侧空间。

### Step 2：Task card 打开详情

修改：

```text
SortableTaskCard
KanbanColumn
KanbanView
```

完成：

- `SortableTaskCard` 增加 `onSelect`。
- 根节点增加 `onClick`。
- `KanbanColumn` 透传 `onSelectTask`。
- `KanbanView` 中调用 `setSelectedTask(task)`。

验收：

- 点击任意 task card 打开右侧详情。
- 点击编辑不会同时打开详情。
- 点击删除不会同时打开详情。

### Step 3：Droppable column

修改：

```text
KanbanColumn
handleDragEnd
```

完成：

- 引入 `useDroppable`。
- 每列注册 `column:${status}`。
- `handleDragEnd` 支持 column drop target。
- 空列也能接收任务。

验收：

- 任务可以拖到空列。
- 任务可以拖到列空白区域。
- 跨列拖拽后状态正确更新。

### Step 4：前端排序修正

修改：

```text
handleDragEnd
tasksByStatus
```

完成：

- 同列拖拽后更新本地 state。
- 移除无效的 `notify('Task reordered')`。
- 确保 `arrayMove` 被实际使用，否则删除 import。

验收：

- 同列拖动后视觉顺序立刻变化。
- 没有 unused import。

### Step 5：持久化排序

修改：

```text
apps/desktop/src/global.d.ts
apps/desktop/electron/preload.cjs
apps/desktop/electron/main.cjs
apps/desktop/src/app/kanban/index.tsx
```

完成：

- `KanbanTask` 增加 `order`。
- preload 增加 `reorderTasks`。
- main 增加 `hermes:kanban:reorderTasks`。
- 创建 task 时自动设置 order。
- Renderer 拖拽后持久化排序。

验收：

- 刷新后排序保持。
- 重启后排序保持。

## 11. 建议代码结构拆分

当前 `index.tsx` 文件承担过多职责。GUI 修复后可以逐步拆分：

```text
apps/desktop/src/app/kanban/
  index.tsx                 # KanbanView 数据编排入口
  types.ts                  # Kanban 类型，本地 UI 类型
  constants.ts              # STATUS_COLUMNS / PRIORITY_CONFIG
  kanban-toolbar.tsx        # Toolbar
  kanban-column.tsx         # Column + droppable
  task-card.tsx             # SortableTaskCard
  task-dialog.tsx           # Create/Edit dialog
  task-detail-panel.tsx     # Detail + comments
  drag.ts                   # reorder helper
```

第一阶段不强制拆分，避免一次性改动过大。建议在 GUI 稳定后再拆。

## 12. 局部验收标准

### 12.1 GUI 显示验收

- 打开 `#/kanban` 后，页面没有明显压缩、错位或覆盖。
- Header 与 Board 左右间距一致。
- 每列宽度稳定，任务卡片可读。
- 横向滚动正常。
- 详情面板显示在右侧，不覆盖 Board。
- 窄窗口下页面仍可操作。

### 12.2 交互验收

- 点击 task card 打开详情。
- 关闭详情后 Board 恢复全宽。
- 编辑 task 后 card 和 detail panel 同步更新。
- 删除 selected task 后 detail panel 自动关闭。
- 归档 selected task 后 detail panel 自动关闭。
- 拖到空列可更新状态。
- 拖到列空白区域可更新状态。
- 同列拖拽后视觉顺序变化。

### 12.3 技术验收

- `npm run typecheck` 通过。
- `npm run lint` 通过。
- 没有 unused import。
- 没有新增直接文件系统访问到 Renderer。
- 新增 API 时同步更新 `preload.cjs` 和 `global.d.ts`。

## 13. 非目标

本轮 GUI 修复不处理：

- 多端同步。
- SQLite 迁移。
- Agent tool 设计。
- Chat message 创建任务。
- Cron failure 自动创建任务。
- 标签系统。
- 搜索系统。
- 权限系统。

这些内容仍归属 `apps/desktop/docs/kanban.md` 的长期路线，不在本文档中展开。

## 14. 最小推荐补丁范围

如果只做一次最小修复，推荐只改：

```text
apps/desktop/src/app/kanban/index.tsx
```

包含：

1. Column 固定宽度。
2. Workspace flex 布局。
3. Detail panel 从 absolute 改为 aside。
4. Task card 增加 onSelect。
5. Column 增加 useDroppable。
6. `handleDragEnd` 支持 column drop。

这个补丁可以先解决最明显的 GUI 显示问题，并且不触碰数据结构和 IPC，风险较低。

## 15. 后续与主文档的同步规则

如果本 GUI 修复引入以下变化，需要同步更新 `apps/desktop/docs/kanban.md`：

- 新增或删除数据字段。
- 新增或删除 IPC API。
- 改变 Kanban 状态列定义。
- 改变长期迭代阶段。
- 改变总体验收标准。

如果只是以下变化，不需要更新主文档：

- 调整 CSS class。
- 调整组件拆分方式。
- 修复详情面板布局。
- 修复拖拽空列体验。
- 优化 Toolbar 间距。
- 优化 card hover 样式。
