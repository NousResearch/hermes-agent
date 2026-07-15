---
description: 在 Hermes Desktop 中开发 Kanban 功能的工作指南，包含架构、开发流程、合并策略、CI 避坑
---

# Hermes Desktop Kanban 开发工作指南

## 项目概述

Hermes Agent（`NousResearch/hermes-agent`）是一个开源 AI Agent 框架。Desktop 端基于 Electron + Vite + React + TypeScript。Kanban 功能在 Desktop 内提供轻量级任务管理面板。

当前状态：Phase 1 MVP 已完成，PR #51954 已提交。

## 架构速查

### 关键文件

```
apps/desktop/
  electron/main.cjs            # IPC handlers + SQLite 存储（kanban 在文件尾部）
  electron/preload.cjs         # window.hermesDesktop.kanban.* API 桥
  src/
    global.d.ts                # KanbanBoard / KanbanTask / KanbanComment 类型
    app/kanban/index.tsx       # KanbanView 组件（6 列 + 拖拽 + 详情 + 评论）
    components/assistant-ui/thread.tsx  # Chat → Kanban 菜单入口
    lib/kanban-sync.ts         # todo sync + cron sync 逻辑
    lib/kanban-sync.test.ts    # 8 个单元测试
    i18n/en.ts, zh.ts, types.ts  # 翻译
    app/session/hooks/use-message-stream.ts  # todo 事件 → kanban sync
    app/desktop-controller.tsx  # cron 轮询 → kanban failure sync
    app/routes.ts              # 路由（含 KANBAN_ROUTE）
    app/types.ts               # SidebarNavId（含 'kanban'）
    app/chat/sidebar/index.tsx  # 侧边栏导航
docs/kanban/                   # 7 个开发文档
```

### 数据流

```
Renderer UI
  → window.hermesDesktop.kanban.*
    → preload IPC
      → main process handlers
        → SQLite (HERMES_HOME/kanban.db)
```

### 数据库

- `kanban_boards`: id(TEXT PK), slug(TEXT UNIQUE), title, description, created_at
- `tasks`: Hermes CLI 共享，Desktop 用 ALTER TABLE 加列
- `task_comments`: id(INTEGER PK), task_id, author, body, created_at

Board identity 使用 **slug**（不是随机 id），确保 CLI task 可见。

### Kanban API

11 个 IPC handler：boards, createBoard, deleteBoard, tasks, allTasks, createTask, updateTask, deleteTask, comments, addComment, deleteComment, reorderTasks

### Sync Modes

- `manual`: 用户手动，不同步
- `linked`: 记录 externalTaskId，单向状态同步
- `mirrored`: 双向自动同步（未完全实现）

## 开发流程

### 初始化

```bash
# 1. Fork 官方仓库（不要从快照创建！）
# 2. Clone
git clone https://github.com/YOUR_USER/hermes-agent.git
cd hermes-agent

# 3. 添加上游
git remote add upstream https://github.com/NousResearch/hermes-agent.git

# 4. 创建分支
git checkout -b feat/kanban-xxx

# 5. 启动开发环境
cd apps/desktop
npm run dev:fake-boot
```

### 代码修改

```bash
# 修改前先同步上游
git fetch upstream main
git rebase upstream/main

# 添加改动
# ...修改文件...

# 类型检查
npm run typecheck

# 测试
npx vitest run --environment jsdom src/lib/kanban-sync.test.ts
```

### 提交

```bash
git add <files>
git commit -s -m "feat(kanban): xxx"  # 必须加 -s（Signed-off-by）
git push origin feat/kanban-xxx
```

## PR 流程

### 首次创建

```bash
gh pr create \
  --repo NousResearch/hermes-agent \
  --head YOUR_USER:feat/kanban-xxx \
  --base main \
  --title "feat(desktop): xxx" \
  --body "..."
```

### 冲突解决（高风险！）

**不要**用 `git checkout <old-branch> -- <file>` 来覆盖文件——这会删除上游新增的代码。

正确做法：

```bash
# 1. 更新到最新上游
git fetch upstream main
git rebase upstream/main

# 2. 对于有冲突的文件，接受上游版本
git checkout --theirs <conflicting-file>

# 3. 然后手动添加我们的改动（增量，不是全量替换）
# 编辑文件，只添加我们新增的代码块

# 4. 标记已解决并继续
git add <file>
git rebase --continue

# 5. 验证没有误删上游代码
git diff upstream/main..HEAD -- <file> | grep "^-" | grep -v "^---"
```

### CI 检查清单

| 检查项 | 通过条件 |
|---|---|
| typecheck | `npm run typecheck` 通过 |
| contributor-check | commit 有 `Signed-off-by`，author email 在 `scripts/release.py` 的 `AUTHOR_MAP` 中 |
| desktop-build | Vite build 通过 |
| All required checks | 上述全部通过 |

常见 CI 坑：

1. **Contributor check**：`git commit -s`（自动加 Signed-off-by），并更新 `scripts/release.py` 的 `AUTHOR_MAP`
2. **global.d.ts 冲突**：上游新增了 `petOverlay`、`openPreviewInBrowser` 等属性，不要删它们
3. **i18n 文件冲突**：上游可能新增了 `pet`、`importedBadge` 等翻译键，保留上游版本后只添加 `kanban:` 节

## 不要做的事

- 不要新建第二个 Kanban 页面
- 不要新建第二套 task 存储
- 不要回退到 `kanban.json`
- 不要把 profile 当负责人
- 不要把 running task 特判成独立页面
- 不要让 Renderer 直接访问 SQLite

## 关键词速查

| 关键词 | 含义 |
|---|---|
| board slug | 看板业务 ID，从 title 生成，用作 `KanbanBoard.id` |
| externalTaskId | 外部系统 task ID（agent todo id / cron job id） |
| syncMode | `manual` / `linked` / `mirrored` |
| lastSyncedAt | 最近同步时间戳 |
| assigneeType | `user` / `agent` / `unassigned` |
