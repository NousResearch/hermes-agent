# Kanban 当前实现状态与差异修正文档

> 本文是 Kanban 开发文档体系中的状态修正文档。
>
> 它不替代 `apps/desktop/docs/kanban.md` 的总纲，也不替代 `kanban/gui-remediation-plan.md` 和 `kanban/conversation-task-integration.md` 的实施方案。
>
> 本文用于记录当前代码已经落地到什么程度、哪些旧文档描述已经过期、后续不要重复开发什么，以及新发现的 Desktop GUI 问题。

## 1. 文档层级关系

当前 Kanban 文档层级：

```text
apps/desktop/docs/
  kanban.md
  kanban/
    gui-remediation-plan.md
    conversation-task-integration.md
    implementation-status.md
```

职责划分：

| 文档 | 职责 |
| --- | --- |
| `kanban.md` | Kanban 总体目标、数据模型、IPC、长期路线 |
| `kanban/gui-remediation-plan.md` | Kanban 页面布局、详情面板、拖拽、排序等 GUI 修复计划 |
| `kanban/conversation-task-integration.md` | Chat / Agent / Profile / Cron 与 Kanban task 的集成计划 |
| `kanban/implementation-status.md` | 当前代码完成情况、文档差异修正、新增问题清单 |

阅读顺序建议：

```text
kanban.md
  -> implementation-status.md
  -> gui-remediation-plan.md
  -> conversation-task-integration.md
```

原因：`implementation-status.md` 会指出哪些总纲内容已经被当前代码超前实现或替代。

## 2. 当前结论

截至本文更新时，Kanban 已经不再只是文档或原型。

当前代码已完成：

- Desktop `/kanban` 路由接入。
- Kanban 页面基础 UI。
- Board / Task / Comment 基础 CRUD。
- SQLite 存储。
- `reorderTasks` 排序持久化。
- Column 固定宽度。
- 右侧详情面板。
- Task card 点击打开详情。
- 空列 droppable。
- 从 assistant 消息手动创建 Kanban task 的 MVP。
- Task 元数据字段扩展：`source`、`sessionId`、`profileId`、`assigneeType`、`assigneeLabel`、`syncMode` 等。
- Sidebar Kanban 名称显示 (Kanban / 看板)。
- Chat task 创建时保存 `messageId`。
- Board identity 统一为 slug，running task 不再被过滤。
- 历史数据迁移：随机 board id → slug。
- User message "..." 菜单：Create Kanban Task + Send plan to Kanban。
- Agent plan 批量导入 Kanban（Send plan to Kanban，含 `externalTaskId`）。
- Cron failure → blocked task（内存级去重）。
- Agent todo linked/mirrored sync（更新 kanban task status + `lastSyncedAt`）。
- `externalTaskId` / `externalTaskKind` / `lastSyncedAt` 持久化到 SQLite。

当前仍未完成：

- full Agent workflow orchestration (自动创建/编排任务)
- Agent 状态 mirrored sync（todo → kanban 的 linked sync 已基础完成）
- Cron failure windowing（基础去重已完成，无 failure window）
- 从 selected text 创建 task

## 3. 与旧文档的差异修正

### 3.1 存储层已经从 JSON 变成 SQLite

`kanban.md` 早期写的是：

```text
HERMES_HOME/kanban.json
```

当前代码实际已经使用：

```text
HERMES_HOME/kanban.db
```

并且 main process 使用 SQLite 作为 Kanban 存储层。

当前实现语义：

```text
Renderer
  -> window.hermesDesktop.kanban.*
    -> preload IPC
      -> main process
        -> SQLite kanban.db
```

因此，后续开发不要再实现 JSON 写入逻辑，也不要再新增 `kanban.json`。

需要后续更新 `kanban.md` 的章节：

```text
6. 本地持久化设计
```

应从 JSON 方案改成 SQLite 方案。

### 3.2 `reorderTasks` 已经落地

旧文档中 `reorderTasks` 被描述为建议新增。

当前实现已经包含：

```ts
window.hermesDesktop.kanban.reorderTasks(boardId, updates)
```

后续不要重复新增第二套排序 API。

如果继续优化排序，只应围绕现有 `reorderTasks` 做：

- 修复排序边界。
- 增加测试。
- 优化 optimistic update。
- 确保和 SQLite `sort_order` 一致。

### 3.3 GUI 第一轮修复已完成

`gui-remediation-plan.md` 中列出的核心 GUI 问题已全部落地：

| GUI 项 | 当前状态 | 说明 |
| --- | --- | --- |
| Column 固定宽度 | 已完成 | Column 使用固定宽度，避免 6 列被压缩 |
| 横向滚动 | 已完成 | Board pane 使用横向滚动容器 |
| 详情面板改为右侧 pane | 已完成 | 不再使用 absolute overlay 覆盖 Board |
| Task card 点击打开详情 | 已完成 | Card 支持 `onSelect` |
| Droppable column | 已完成 | Column 注册 droppable id |
| 空列 drop | 已完成 | 空列可作为拖拽目标 |
| 排序持久化 | 已完成 | 已使用 `order` / `sort_order` 和 `reorderTasks` |
| Sidebar 名称显示 | 已完成 | i18n 配置 `Kanban` / `看板` |
| source/assignee metadata | 已完成 | Task detail 面板已显示来源信息 |

### 3.4 Conversation task integration 已落地

`conversation-task-integration.md` 的最小目标已全部实现：

已完成：

```text
assistant message action menu
  -> Create Kanban Task
  -> window.hermesDesktop.kanban.createTask(...)
  -> source = chat, sessionId, profileId, messageId
  -> assigneeType = user, assigneeLabel = You
  -> syncMode = manual

user message action menu
  -> Create Kanban Task (同上)

Agent plan batch import
  -> Send plan to Kanban (N)
  -> each todo -> createTask with externalTaskId
  -> source = agent, syncMode = linked

Todo sync
  -> linked/mirrored todo status -> Kanban task status
  -> updates lastSyncedAt

Cron failure sync
  -> new cron error -> blocked task (内存级去重)
  -> externalTaskId + externalTaskKind 持久化
```

未完成（后续阶段）：

- 从 selected text 创建 task。
- Agent 全自动编排。
- full mirrored sync 的双向覆盖策略。

## 4. 当前完成情况总表

| 模块 | 状态 | 后续动作 |
| --- | --- | --- |
| `/kanban` 路由 | 已完成 | 不要重复 |
| Kanban 页面 | 已完成 | 继续增量优化 |
| Board CRUD | 已完成 | 不要重复 |
| Task CRUD | 已完成 | 不要重复 |
| Comment CRUD | 已完成 | 不要重复 |
| SQLite 存储 | 已完成 | 不要回退 JSON |
| `reorderTasks` 排序持久化 | 已完成 | 边界优化 |
| GUI 列宽 / 横向滚动 / 详情面板 | 已完成 | 窄屏验证 |
| 空列拖拽 | 已完成 | — |
| Sidebar Kanban 名称显示 | 已完成 | i18n `Kanban` / `看板` |
| Chat message -> Kanban | 已完成 | 含 `messageId` / `sessionId` / `profileId` |
| User message -> Kanban | 已完成 | 含 "..." 菜单 entry |
| Board identity 统一 | 已完成 | slug 替代随机 id |
| Agent plan -> Kanban | 已完成 | "Send plan to Kanban" 批量导入 |
| Agent todo linked sync | 已完成 | todo 状态 -> Kanban status |
| Cron failure -> blocked task | 已完成 | 内存级去重 |
| external linkage 持久化 | 已完成 | `externalTaskId` / `lastSyncedAt` 写入 SQLite |
| selected text -> Kanban | 未完成 | 后续增量实现 |
| Agent 全自动编排 | 未完成 | 后续阶段 |
| mirrored sync 双向覆盖 | 未完成 | 后续阶段 |
| Cron failure windowing | 未完成 | 后续阶段 |

## 5. 已关闭问题

> 以下问题已在之前实现中修复，此处留作历史记录，不再需要处理。

- Sidebar Kanban 名称显示：已通过 i18n nav 添加 `Kanban` / `看板` 文案修复。
- Chat task 未保存 messageId：`thread.tsx` 中已传入 `messageId` 参数。
- Board identity 随机 id：已统一为 slug，含历史数据迁移。
- Chat task 固定 `default` board：改为动态获取第一个可用 board。

## 6. 后续推荐优先级

### P0：已完成（Board identity + Sidebar 名称）

- ~~Board identity 统一 slug~~ ✅ 已修复
- ~~Sidebar Kanban 名称显示~~ ✅ 已修复
- ~~external linkage 持久化~~ ✅ 已修复

### P1：接近完成

统一 assignee 语义：

1. Card 和 detail panel 使用统一显示函数：

```ts
const displayAssignee = task.assigneeLabel || task.assignee || t.desktop.kanban.unassigned
```

2. 手动创建 task 时逐步从自由文本 `assignee` 迁移到：

```ts
assigneeType
assigneeId
assigneeLabel
```

3. 明确 profile 只作为上下文，不作为 assignee。

### P2：已完成（Agent / Cron 基础集成）

- ~~Agent plan 手动批量导入~~ ✅ 已完成
- ~~Linked sync（todo status → kanban status）~~ ✅ 已完成
- ~~Cron failure → blocked task~~ ✅ 已完成

### P3：仍未完成

- Mirrored sync 双向覆盖策略
- Cron failure windowing
- Selected text → Kanban

## 7. 不要重复开发清单

后续开发请继续遵守：

- 不要新建第二个 Kanban 页面。
- 不要新建第二套 task 存储。
- 不要再实现 `kanban.json` 写入路径。
- 不要新建第二套 CRUD API。
- 不要新建第二套排序 API。
- 不要把 profile 当成负责人。
- 不要把 sidebar Kanban 名称问题放到 `KanbanView` 内修；它属于 sidebar nav 层。

## 8. 当前最小下一步

以上 P0/P1/P2 项已全部完成。当前文档体系已与实际代码状态对齐。