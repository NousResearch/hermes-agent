# Hermes Desktop Kanban — 开源协作经验总结

## 项目背景

在 NousResearch/hermes-agent 开源项目基础上，为其桌面端增加 Kanban 看板功能。从零开始，经历了需求分析 → 代码开发 → 文档清理 → 提交 PR → 解决冲突 → CI 通过的完整流程。

## 关键数据

- **开发周期**：1 个完整 session（连续对话）
- **提交次数**：本地 20+ 次 commit，上游 PR 1 个 squash commit
- **改动文件**：22 个文件，+6455/-135 行
- **PR 链接**：[NousResearch/hermes-agent#51954](https://github.com/NousResearch/hermes-agent/pull/51954)

## 经验教训

### 1. 仓库管理 — 最痛苦的环节

**问题**：仓库是独立创建的，不是官方 fork，导致和上游没有共同 git 历史。

**后果**：
- PR 无法直接创建（"no commits in common"）
- 每次 rebase 都产生大量 add/add 冲突
- 直接 `git checkout 旧分支 -- file` 覆盖了上游新增的 pet overlay、openPreviewInBrowser 等代码
- 反复冲突 → 反复修复 → 反复 CI 失败

**教训**：**永远从官方仓库 fork，永远不要从快照创建独立仓库。** 如果已经犯了，用 `git diff` + patch 文件来迁移改动，不要直接 checkout 文件。

### 2. 合并策略 — 增量胜于全量

**问题**：每次冲突都用 "覆盖整个文件" 的方式解决。

**正确做法**：以上游最新版本为基准，**只添加**我们新增的代码块，不替换整个文件。使用 `git apply --reject` 比 `git checkout` 安全得多。

### 3. CI 检查 — 细节决定成败

| 问题 | 原因 | 修复 |
|---|---|---|
| Contributor check 失败 | commit 没有 `Signed-off-by`，author email 不在 `AUTHOR_MAP` | `git commit -s` + 更新 `scripts/release.py` |
| TypeScript 编译失败 | `global.d.ts` 删了上游的 `petOverlay` 类型 | 精细合并，只加不减 |
| 类型错误 | `routes.ts` 没有 `KANBAN_ROUTE` | 确认 routes 文件是最新版 |
| WhatsApp 测试失败 | flaky test，与改动无关 | 已在 PR 中说明 |

**教训**：CI 每个失败都要仔细分析根因。快速 "修复" 往往会引入新问题。

### 4. 文档一致性

`implementation-status.md` 前后矛盾——前面写 "已完成"，后面旧章节还写 "未完成"。整理文档比写代码更花时间。

### 5. i18n 设计

项目使用自研 i18n 系统（非 react-i18next），类型安全但容易遗漏键。需要同时更新 `types.ts`（接口）、`en.ts`（英文）、`zh.ts`（中文）。

## 有效做法

- **开发文档先行**：先在 `docs/kanban/` 写清楚目标、数据模型、IPC 设计再写代码
- **增量提交**：功能拆分为独立 commit，方便 review
- **类型先行**：先扩类型定义，再实现逻辑
- **最小测试**：虽然测试晚加了，但 `kanban-sync.test.ts` 8 个测试覆盖了核心 sync 逻辑
- **Plan mode**：复杂改动前先 enter plan mode 对齐方案

## 下次改进

1. **直接 fork 官方仓库**，避免历史分裂
2. **冲突只加不减**：合并前用 `git diff` 检查是否有意外删除
3. **提前处理 contributor check**：第一个 commit 就加 `-s`
4. **CI 先跑 typecheck**：在本地跑 `npm run typecheck` 确认通过再提交
5. **文档和代码同步更新**：改完代码立即更新对应文档，不积压
