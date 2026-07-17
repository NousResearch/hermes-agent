# Kanban 对话任务集成开发文档

> 本文是 `apps/desktop/docs/kanban.md` 的二级子文档。
>
> 本文只定义“对话 / Agent / Profile / Kanban Task”之间的集成关系、完成情况、非重复开发边界和实施顺序。
>
> 不要在本文中重复实现 Kanban 基础 CRUD、GUI 布局修复或通用数据模型总纲。

## 1. 文档层级关系

当前 Kanban 文档层级如下：

```text
apps/desktop/docs/
  kanban.md
  kanban/
    gui-remediation-plan.md
    conversation-task-integration.md
```

职责划分：

| 文档 | 层级 | 职责 | 本文是否覆盖 |
| --- | --- | --- | --- |
| `apps/desktop/docs/kanban.md` | 一级总纲 | Kanban 功能目标、基础数据模型、IPC 总体设计、长期路线 | 不覆盖 |
| `apps/desktop/docs/kanban/gui-remediation-plan.md` | 二级 GUI 实施文档 | 修复 Kanban 页面显示、详情面板、拖拽、列宽、排序体验 | 不覆盖 |
| `apps/desktop/docs/kanban/conversation-task-integration.md` | 二级集成实施文档 | 将 Chat / Agent / Profile / Cron 任务接入 Kanban | 当前文档 |

本文的定位：

```text
kanban.md                         # 定义 Kanban 是什么
  ├─ gui-remediation-plan.md       # 让 Kanban 页面先正确显示和可交互
  └─ conversation-task-integration.md # 让 Kanban 能看见对话和 agent 任务
```

## 2. 当前完成情况

### 2.1 已完成

| 模块 | 完成状态 | 说明 | 是否需要重复开发 |
| --- | --- | --- | --- |
| Kanban 路由 | 已完成 | Desktop 已接入 `/kanban` 页面 | 不要重复 |
| Kanban Renderer 页面 | 已完成 | 6 列看板、拖拽排序、详情面板、空列 drop | 不要重建 |
| Board CRUD | 已完成 | Board 创建、读取、删除能力 | 不要重复 |
| Task CRUD | 已完成 | Task 创建、读取、更新、删除能力 | 不要重复 |
| Comment CRUD | 已完成 | 评论读取、创建、删除能力 | 不要重复 |
| preload Kanban API | 已完成 | 含 `reorderTasks` 等 IPC | 只增量扩展 |
| SQLite 存储 | 已完成 | `HERMES_HOME/kanban.db` | 不要回退 JSON |
| Kanban 总开发文档 | 已完成 | `apps/desktop/docs/kanban.md` | 不要重复写总纲 |
| GUI 修复计划文档 | 已完成 | `apps/desktop/docs/kanban/gui-remediation-plan.md` | 不要重复写 GUI 计划 |
| Chat assistant message -> Kanban | 已完成 | 含 `messageId`/`sessionId`/`profileId` | 不要重复 |
| Chat user message -> Kanban | 已完成 | "..." 菜单入口 | 不要重复 |
| Agent plan 批量导入 | 已完成 | "Send plan to Kanban" | 不要重复 |
| Cron failure -> blocked task | 已完成 | 基础去重 | 不要重复 |
| Todo linked sync | 已完成 | todo 状态 -> Kanban status | 不要重复 |
| external linkage 持久化 | 已完成 | `externalTaskId`/`lastSyncedAt` 写入 SQLite | 不要重复 |
| Board identity 统一 | 已完成 | slug 替代随机 id, 含历史迁移 | 不要重复 |

### 2.2 部分完成

| 模块 | 完成状态 | 当前问题 | 后续归属 |
| --- | --- | --- | --- |
| Task assignee | 部分完成 | 新 task 使用 `assigneeType`/`assigneeLabel`，旧 task 仍可走自由文本 | 后续统一 |
| mirrored sync | 部分完成 | linked sync 已完成，双向覆盖策略未实现 | 后续阶段 |

### 2.3 未完成

| 模块 | 完成状态 | 说明 | 本文是否负责 |
| --- | --- | --- | --- |
| 从 selected text 创建 Kanban Task | 未完成 | 需要 selection toolbar 机制 | 是 |
| Agent 全自动同步编排 | 未完成 | 全自动 task 创建和管理 | 后续阶段 |
| mirrored sync 双向覆盖 | 未完成 | 用户 override 策略未实现 | 后续阶段 |
| Cron failure windowing | 未完成 | 时间窗口去重未实现 | 后续阶段 |

## 3. 不要重复开发清单

为了避免重复开发，本集成阶段明确禁止做以下事情：

### 3.1 不要重建 Kanban 页面

不要新建第二个 Kanban 页面，例如：

```text
apps/desktop/src/app/tasks/
apps/desktop/src/app/agent-tasks/
apps/desktop/src/app/conversation-kanban/
```

应继续使用：

```text
apps/desktop/src/app/kanban/index.tsx
```

所有对话任务、agent task、manual task 最终都进入同一个 Kanban 数据模型和同一个 Kanban 页面。

### 3.2 不要新建第二套 Task 存储

不要新增：

```text
conversation-tasks.json
agent-tasks.json
todos-kanban.json
```

当前 Kanban 本地存储是：

```text
HERMES_HOME/kanban.json
```

本阶段只允许扩展该数据结构，不允许并行维护第二套看板任务存储。

### 3.3 不要重复实现 Board / Task / Comment CRUD

已有 CRUD 能力继续复用。

本阶段只新增“来源关联”和“导入 / 同步”能力，例如：

- `createTaskFromMessage`
- `createTasksFromAgentPlan`
- `linkTaskToSession`
- `updateLinkedTaskStatus`

不要重新写一套 `createTask` / `updateTask` / `deleteTask`。

### 3.4 不要把 profile 当作负责人

Profile 是上下文，不是负责人。

错误理解：

```text
assignee = profile
```

正确理解：

```text
profile = 当前工作区 / 身份上下文 / 配置上下文
assignee = 负责推进任务的 actor，可以是 user 或 agent
```

### 3.5 不要让 Chat 和 Kanban 互相强耦合

Chat 不应该直接读写 `kanban.json`。

正确链路：

```text
Chat UI
  -> window.hermesDesktop.kanban.*
    -> preload IPC
      -> main process
        -> kanban.json
```

Renderer 仍然不能直接访问文件系统。

## 4. 核心概念定义

### 4.1 Profile

Profile 表示 Hermes 的工作上下文。

含义包括但不限于：

- 当前用户选择的配置上下文。
- 当前 agent/session 使用的环境上下文。
- 当前 conversation 的归属上下文。

Profile 不代表执行者，也不代表负责人。

推荐字段：

```ts
profileId?: string
profileLabel?: string
```

### 4.2 Agent

Agent 是可以执行任务的主体。

Agent 可以成为任务负责人。

推荐字段：

```ts
agentId?: string
agentLabel?: string
```

### 4.3 Assignee

Assignee 表示当前负责推进任务的 actor。

它可能是：

- 当前用户。
- 某个 agent。
- 暂未分配。

推荐字段：

```ts
type KanbanAssigneeType = 'user' | 'agent' | 'unassigned'

assigneeType: KanbanAssigneeType
assigneeId?: string
assigneeLabel?: string
```

### 4.4 Session

Session 表示任务来源对话。

如果 task 来自某次聊天，应记录：

```ts
sessionId?: string
messageId?: string
messageCreatedAt?: number
```

用途：

- 从 Kanban task 跳回原始 conversation。
- 在 conversation 中显示“已关联到 Kanban”。
- 后续同步 agent 状态。

### 4.5 Source

Source 表示 task 的来源。

```ts
type KanbanTaskSource = 'manual' | 'chat' | 'agent' | 'cron'
```

含义：

| source | 说明 |
| --- | --- |
| `manual` | 用户在 Kanban 页面手动创建 |
| `chat` | 用户从聊天内容手动创建 |
| `agent` | 从 agent plan / todo / execution 自动或半自动创建 |
| `cron` | 从 cron / automation 事件创建 |

## 5. 推荐数据模型增量

当前总模型在 `kanban.md` 中定义。本文只定义为了对话集成需要增加的字段。

### 5.1 KanbanTask 增量字段

```ts
interface KanbanTask {
  // 已有字段继续保留
  id: string
  boardId: string
  title: string
  description: string
  status: KanbanStatus
  priority: KanbanPriority
  createdAt: number
  updatedAt: number
  archived: boolean

  // 本文新增：来源关系
  source: 'manual' | 'chat' | 'agent' | 'cron'
  profileId?: string
  profileLabel?: string
  sessionId?: string
  messageId?: string
  messageCreatedAt?: number

  // 本文新增：负责人语义
  assigneeType: 'user' | 'agent' | 'unassigned'
  assigneeId?: string
  assigneeLabel?: string

  // 本文新增：agent 执行关系
  agentId?: string
  agentLabel?: string
  externalTaskId?: string
  externalTaskKind?: 'chat_todo' | 'agent_plan_item' | 'cron_job' | 'manual'

  // 本文新增：同步控制
  syncMode?: 'manual' | 'linked' | 'mirrored'
  lastSyncedAt?: number
}
```

### 5.2 字段解释

| 字段 | 说明 | 是否必须立即实现 |
| --- | --- | --- |
| `source` | 任务来源 | 是 |
| `profileId` | 来源 profile | 是，若能拿到 |
| `profileLabel` | profile 显示名 | 否 |
| `sessionId` | 来源 conversation | 是，用于 chat/agent 来源 |
| `messageId` | 来源 message | 第二阶段 |
| `assigneeType` | 负责人类型 | 是 |
| `assigneeId` | 用户或 agent id | 第二阶段 |
| `assigneeLabel` | 显示用名称 | 是 |
| `agentId` | 执行 agent id | 第二阶段 |
| `externalTaskId` | 外部 todo/plan id | 第二阶段 |
| `syncMode` | 同步模式 | 第二阶段 |
| `lastSyncedAt` | 最近同步时间 | 第二阶段 |

## 6. 同步模式设计

### 6.1 Manual

```text
syncMode = manual
```

含义：

- 任务被创建后，不自动跟随原始 conversation / agent 状态变化。
- 用户在 Kanban 中手动维护状态。

适用：

- 用户从一段聊天文本创建任务。
- 用户在 Kanban 页面手动创建任务。

第一阶段优先实现 manual。

### 6.2 Linked

```text
syncMode = linked
```

含义：

- Task 记录来源 session/message/agent plan item。
- 但状态不自动覆盖。
- UI 可以显示“来源对话”和“跳回对话”。

适用：

- 从 agent plan 创建任务。
- 从 chat todo 创建任务。

第二阶段实现。

### 6.3 Mirrored

```text
syncMode = mirrored
```

含义：

- Kanban task 和 agent task 状态自动同步。
- Agent 状态变化会更新 Kanban。
- Kanban 状态变化也可能影响 agent workflow。

适用：

- 长期自动化集成。
- Agent 执行任务流。

第三阶段以后再实现。

## 7. 实施阶段

### Phase 0：先完成 GUI 修复依赖

依赖文档：

```text
apps/desktop/docs/kanban/gui-remediation-plan.md
```

必须先完成：

- Kanban 列宽修复。
- 详情面板不覆盖主界面。
- Task card 点击打开详情。
- 空列可以拖拽。

原因：

对话任务接入后，Kanban 中的 task 数量会变多。如果 GUI 仍然存在列压缩、详情打不开、空列不能拖的问题，会让集成后的任务不可用。

完成状态：

| 事项 | 状态 |
| --- | --- |
| 文档已创建 | 已完成 |
| 代码修复 | 未完成 |
| 是否阻塞本集成文档 | 不阻塞文档，但阻塞高质量集成体验 |

### Phase 1：明确 assignee / profile / source 语义

目标：

- 不再把 `assignee` 当作任意字符串长期使用。
- 不再把 profile 当负责人。
- 所有新建 Kanban task 都带 `source`。

推荐实现：

```ts
const DEFAULT_MANUAL_TASK_META = {
  source: 'manual',
  assigneeType: 'unassigned',
  assigneeLabel: ''
}
```

从对话创建任务时：

```ts
{
  source: 'chat',
  sessionId: currentSessionId,
  profileId: currentProfileId,
  assigneeType: 'user',
  assigneeLabel: 'You'
}
```

从 agent plan 创建任务时：

```ts
{
  source: 'agent',
  sessionId: currentSessionId,
  profileId: currentProfileId,
  assigneeType: 'agent',
  assigneeId: agentId,
  assigneeLabel: agentLabel
}
```

完成状态：

| 事项 | 状态 |
| --- | --- |
| 概念定义 | 已完成，见本文 |
| TypeScript 类型更新 | 未完成 |
| main process 默认字段补齐 | 未完成 |
| UI 显示文案调整 | 未完成 |

### Phase 2：从 Chat 手动创建 Kanban Task

目标：

用户可以从当前对话中手动创建 Kanban task。

入口建议：

```text
Chat message menu
  -> Create Kanban Task
```

或：

```text
Selected text toolbar
  -> Add to Kanban
```

最小数据：

```ts
{
  boardId: activeOrDefaultBoardId,
  title: selectedTextOrMessageSummary,
  description: fullSelectedTextOrMessageText,
  status: 'todo',
  priority: 'medium',
  source: 'chat',
  sessionId: currentSessionId,
  messageId: sourceMessageId,
  profileId: currentProfileId,
  assigneeType: 'user',
  assigneeLabel: 'You',
  syncMode: 'manual'
}
```

实现要求：

- 必须复用现有 `window.hermesDesktop.kanban.createTask`。
- 如果字段不足，只扩展 `createTask` 入参，不新增第二套 create API。
- 如果没有 active board，使用默认 board 或提示用户选择 board。
- 创建成功后显示 notification。
- 可选：创建成功后提供 “Open in Kanban”。

完成状态：

| 事项 | 状态 |
| --- | --- |
| 需求定义 | 已完成，见本文 |
| Chat message menu 入口 | 未完成 |
| selected text 入口 | 未完成 |
| createTask 入参扩展 | 未完成 |
| 创建成功跳转 Kanban | 未完成 |

### Phase 3：Kanban Task 跳回来源对话

目标：

从 Kanban 详情页可以回到来源 session/message。

详情面板建议显示：

```text
Source: Chat
Profile: <profile label>
Session: Open conversation
Message: View source message
```

导航行为：

```text
Kanban task detail
  -> Open source conversation
    -> route to session
      -> optionally scroll/highlight source message
```

最小实现：

- 先只支持 `sessionId` 跳转。
- `messageId` 高亮可后续实现。

完成状态：

| 事项 | 状态 |
| --- | --- |
| 需求定义 | 已完成，见本文 |
| Kanban detail 显示 source | 未完成 |
| session 跳转 | 未完成 |
| message 高亮 | 未完成，后续 |

### Phase 4：从 Agent Plan / Todo 创建 Kanban Task

目标：

当 agent 输出 plan、todo list 或执行步骤时，用户可以批量发送到 Kanban。

入口建议：

```text
Agent plan block
  -> Send plan to Kanban
```

转换规则：

```ts
plan.items.map(item => ({
  boardId,
  title: item.title,
  description: item.description ?? '',
  status: item.status === 'done' ? 'done' : 'todo',
  priority: inferPriority(item),
  source: 'agent',
  sessionId: currentSessionId,
  profileId: currentProfileId,
  assigneeType: 'agent',
  assigneeId: currentAgentId,
  assigneeLabel: currentAgentLabel,
  externalTaskId: item.id,
  externalTaskKind: 'agent_plan_item',
  syncMode: 'linked'
}))
```

注意：

- 第一版建议是“手动批量发送”，不要一开始做全自动同步。
- 避免 agent 每次刷新 plan 都重复创建 task。
- 需要用 `externalTaskId + sessionId + source` 做去重。

完成状态：

| 事项 | 状态 |
| --- | --- |
| 需求定义 | 已完成，见本文 |
| Agent plan item 识别 | 未完成 |
| 批量创建入口 | 未完成 |
| 去重策略 | 未完成 |
| linked sync | 未完成 |

### Phase 5：Agent 状态同步到 Kanban

目标：

Agent 执行状态变化时，Kanban task 自动更新状态。

推荐映射：

| Agent 状态 | Kanban 状态 |
| --- | --- |
| planned / pending | `todo` |
| ready | `ready` |
| running | `running` |
| waiting_for_user | `blocked` |
| tool_failed | `blocked` |
| completed_needs_review | `review` |
| completed | `done` |

同步规则：

- 只同步 `syncMode = mirrored` 的 task。
- 不要覆盖用户手动修改过的 `manual` task。
- 如果用户手动把 mirrored task 改成 `blocked`，需要记录 override。

完成状态：

| 事项 | 状态 |
| --- | --- |
| 状态映射定义 | 已完成，见本文 |
| Agent 状态事件源识别 | 未完成 |
| mirrored sync | 未完成 |
| 用户 override 策略 | 未完成 |

### Phase 6：Cron / Automation 接入

目标：

Cron 或 automation 失败时可以创建 blocked task。

推荐规则：

```ts
{
  title: `Automation failed: ${jobName}`,
  description: errorMessage,
  status: 'blocked',
  priority: 'high',
  source: 'cron',
  profileId,
  assigneeType: 'user',
  assigneeLabel: 'You',
  externalTaskId: jobId,
  externalTaskKind: 'cron_job',
  syncMode: 'linked'
}
```

完成状态：

| 事项 | 状态 |
| --- | --- |
| 需求定义 | 已完成，见本文 |
| Cron event source | 未完成 |
| failed job -> task | 未完成 |
| 去重策略 | 未完成 |

## 8. 需要新增或扩展的 API

### 8.1 不建议新增重复 API

不要新增：

```ts
createConversationTask()
createAgentTask()
createCronTask()
```

这些会让 Task 创建入口分裂。

### 8.2 推荐扩展 createTask

继续使用：

```ts
window.hermesDesktop.kanban.createTask(data)
```

扩展 data 字段：

```ts
interface CreateKanbanTaskInput {
  boardId: string
  title: string
  description?: string
  status?: KanbanStatus
  priority?: KanbanPriority

  source?: 'manual' | 'chat' | 'agent' | 'cron'
  profileId?: string
  profileLabel?: string
  sessionId?: string
  messageId?: string

  assigneeType?: 'user' | 'agent' | 'unassigned'
  assigneeId?: string
  assigneeLabel?: string

  agentId?: string
  agentLabel?: string
  externalTaskId?: string
  externalTaskKind?: 'chat_todo' | 'agent_plan_item' | 'cron_job' | 'manual'

  syncMode?: 'manual' | 'linked' | 'mirrored'
}
```

### 8.3 可选新增批量 API

为了从 agent plan 批量创建 task，可以新增：

```ts
createTasks(data: CreateKanbanTaskInput[]): Promise<KanbanTask[]>
```

IPC：

```text
hermes:kanban:createTasks
```

注意：

- 这是批量 wrapper，不是第二套 Task 模型。
- 内部应复用同一套 sanitize / create task 逻辑。

### 8.4 可选新增查询 API

为了避免重复创建，可新增：

```ts
findTasksBySource(data: {
  source: KanbanTaskSource
  sessionId?: string
  externalTaskId?: string
}): Promise<KanbanTask[]>
```

IPC：

```text
hermes:kanban:findTasksBySource
```

第一阶段可不做，先在批量创建时由 main process 内部去重。

## 9. 去重策略

从对话或 agent plan 创建任务时，必须避免重复创建。

### 9.1 Chat message 去重

唯一键：

```text
source + sessionId + messageId + titleHash
```

如果没有 `messageId`，使用：

```text
source + sessionId + titleHash + descriptionHash
```

### 9.2 Agent plan item 去重

唯一键：

```text
source + sessionId + externalTaskId
```

如果没有 `externalTaskId`，使用：

```text
source + sessionId + titleHash
```

### 9.3 Cron task 去重

唯一键：

```text
source + externalTaskKind + externalTaskId + failureWindow
```

`failureWindow` 可按小时或天分桶，避免同一 job 高频失败刷屏。

## 10. UI 显示规范

### 10.1 Task card

Task card 应显示来源和负责人。

推荐显示：

```text
[High] Fix build failure
Agent: Hermes Agent
Source: Chat
```

小屏或卡片空间不足时只显示：

```text
Agent · Chat
```

### 10.2 Task detail panel

详情面板增加 metadata 区域：

```text
Metadata
  Source: Chat
  Profile: default
  Assignee: Hermes Agent
  Session: Open conversation
  Sync: Manual
```

### 10.3 Board filter

后续可以增加过滤器：

```text
All / Manual / Chat / Agent / Cron
```

第一阶段不强制。

## 11. 代码落点

### 11.1 必改文件

```text
apps/desktop/src/global.d.ts
  - 扩展 KanbanTask 类型
  - 扩展 createTask 入参类型

apps/desktop/electron/main.cjs
  - createTask 补充 source/profile/session/assignee 字段
  - 增加 sanitize
  - 可选增加 createTasks 批量创建

apps/desktop/electron/preload.cjs
  - 如新增 createTasks，则暴露 createTasks

apps/desktop/src/app/kanban/index.tsx
  - 显示 source / profile / assignee metadata
  - detail panel 支持跳转 source session
```

### 11.2 需要先定位再改的文件

以下文件路径需要实现前先通过代码搜索确认，避免猜路径：

```text
Chat message rendering component
Chat message action menu
Session route helper
Agent plan / todo rendering component
Cron / automation event source
Profile state/store
```

要求：

- 不要凭空新建新的 Chat UI 入口。
- 应接入现有 message action menu 或现有 message toolbar。
- 应复用现有 session route helper。
- 应复用现有 profile store。

## 12. 实施顺序总表

| 顺序 | 任务 | 文档归属 | 状态 | 是否可并行 |
| --- | --- | --- | --- | --- |
| 1 | 修复 Kanban GUI 基础显示 | `gui-remediation-plan.md` | 已完成 | — |
| 2 | 明确 assignee/profile/source 类型 | 当前文档 | 已完成 | — |
| 3 | 扩展 KanbanTask 元数据字段 | 当前文档 + `kanban.md` 同步 | 已完成 | — |
| 4 | Chat message -> Create Kanban Task | 当前文档 | 已完成 | — |
| 5 | Kanban detail -> Open source session | 当前文档 | 已完成 | — |
| 6 | Agent plan -> Batch create tasks | 当前文档 | 已完成 | — |
| 7 | linked sync | 当前文档 | 已完成 | — |
| 8 | mirrored sync | 当前文档 | 未完成 | 依赖 7 |
| 9 | Cron failure -> blocked task | 当前文档 | 已完成 | — |

## 13. 最小不重复开发方案

如果只做一轮最小有效集成，推荐范围如下：

```text
目标：让用户能从当前对话手动创建 Kanban task，并在 Kanban 中看到来源信息。
```

只做：

1. 扩展 `KanbanTask` 元数据字段。
2. 扩展 `createTask` 支持 `source/sessionId/profileId/assigneeType/assigneeLabel`。
3. 在 Chat message action menu 增加 `Create Kanban Task`。
4. 创建成功后写入当前 Kanban board。
5. Kanban detail 显示 `Source: Chat` 和 `Open conversation`。

不做：

- 不做自动同步。
- 不做 agent plan 批量同步。
- 不做 cron 接入。
- 不做第二套 task store。
- 不重写 Kanban 页面。

## 14. 验收标准

### 14.1 第一阶段验收

- 用户可以从一条 chat message 创建 Kanban task。
- 创建后的 task 出现在现有 Kanban 页面。
- task 的 `source` 是 `chat`。
- task 记录 `sessionId`。
- task 记录 `profileId`，如果当前上下文能拿到 profile。
- task 的 assignee 不再只是无语义自由文本，至少有 `assigneeType`。
- Kanban detail 中能看到来源信息。
- 不存在第二份 Kanban task 存储。
- 不存在第二个 Kanban 页面。

### 14.2 第二阶段验收

- 用户可以从 agent plan 批量创建 Kanban tasks。
- 重复点击不会重复创建同一批 task。
- created task 能记录 `externalTaskId` 或等价去重键。
- Kanban 中能区分 `manual`、`chat`、`agent` 来源。

### 14.3 第三阶段验收

- linked task 可以跳回来源对话。
- mirrored task 可以随 agent 状态变化更新。
- 用户手动 override 不会被无条件覆盖。

## 15. 与主文档同步规则

如果本集成实现引入以下变化，必须同步更新 `apps/desktop/docs/kanban.md`：

- `KanbanTask` 持久化字段变化。
- 新增 IPC API。
- 改变 `source` 枚举。
- 改变 `assigneeType` 枚举。
- 改变总体验收标准。

如果只是以下变化，不需要更新主文档：

- Chat message 菜单入口位置调整。
- Card metadata 的视觉样式调整。
- Detail panel 中 metadata 的排版调整。
- “Open conversation” 按钮文案调整。

## 16. 当前结论

当前 Kanban 已经具备手动任务板基础，但还不能看到对话启动的任务。

下一步不要重复做 Kanban 基础功能，而应该按以下顺序增量落地：

1. 先按 `gui-remediation-plan.md` 修复 GUI 可用性。
2. 在现有 `KanbanTask` 上补充 `source/profile/session/assignee` 元数据。
3. 复用现有 `createTask`，从 Chat message 手动创建 task。
4. 在 Kanban detail 中显示来源并支持跳回 conversation。
5. 再考虑 Agent plan 批量导入和状态同步。
