---
description: 通用开源项目协作指南 — Fork、开发、PR、冲突解决、CI 避坑全流程经验
---

# 开源项目协作工作指南

## 核心原则

1. **永远从官方仓库 fork，永远不要从快照创建独立仓库。** 否则没有共同 git 历史，PR 无法创建，每次 rebase 都是灾难。
2. **增量胜于全量**：修改已有文件时只添加新增代码块，不要用旧版本覆盖整个文件。
3. **第一个 commit 就加 `-s`**（Signed-off-by），避免 contributor check 失败后重改历史。

## 标准工作流

### 1. 初始化

```bash
# Fork 官方仓库后在本地 clone
git clone https://github.com/YOUR_USER/PROJECT.git
cd PROJECT

# 添加上游
git remote add upstream https://github.com/ORIGINAL_OWNER/PROJECT.git

# 创建功能分支
git checkout -b feat/my-feature
```

### 2. 开发

```bash
# 定期同步上游，避免积压太多冲突
git fetch upstream
git rebase upstream/main

# 改代码...
# 类型检查
npm run typecheck  # 或对应项目的检查命令

# 运行测试
npm test
```

### 3. 提交

```bash
git add <files>
git commit -s -m "feat(scope): short description"  # -s 是必须的！
git push origin feat/my-feature
```

### 4. 创建 PR

```bash
gh pr create \
  --repo ORIGINAL_OWNER/PROJECT \
  --head YOUR_USER:feat/my-feature \
  --base main \
  --title "feat(scope): description" \
  --body "Implemented:\n- ...\n\nOut of scope:\n- ..."
```

## 冲突解决（高风险操作）

### ❌ 错误做法

```bash
# 这会用旧版本覆盖整个文件，删除上游新增的代码！
git checkout <old-branch> -- src/file.ts
```

### ✅ 正确做法

```bash
# 1. 更新到最新上游
git fetch upstream main
git rebase upstream/main

# 2. 冲突时接受上游版本
git checkout --theirs <conflicting-file>

# 3. 然后手动添加我们的改动（只加不删）
# 编辑文件，找到我们新增的代码块，重新插入

# 4. 标记已解决并继续
git add <file>
git rebase --continue

# 5. 验证没有误删上游代码（非常重要！）
git diff upstream/main..HEAD -- <file> | grep "^-" | head -20
```

### 冲突验证规则

合并后必须检查 diff 中是否包含 **意外删除**：

```bash
# 只应该有我们新增的代码（+号行）
# 不应该有对上游已有代码的删除（-号行），除了我们确实要改的行
git diff upstream/main..HEAD -- <modified-file> | grep "^-" | grep -v "^---"
```

如果出现意外删除的上游代码（如 `petOverlay`、`openPreviewInBrowser` 等），说明合并策略错了，需要重做。

## CI 检查避坑

### Contributor Check / DCO

| 问题 | 原因 | 修复 |
|---|---|---|
| commit 没有签名 | 忘了加 `-s` | `git commit -s --amend`（已推送则需 force push） |
| author email 未映射 | CI 检查 `AUTHOR_MAP` | 将 email 添加到 `scripts/release.py` 或等效配置 |
| bot/noreply 邮箱被误判 | CI 过滤规则不完整 | 在 CI 脚本的 case 语句中添加过滤 |

### TypeScript 编译失败

- **global.d.ts**：上游可能新增了 window 类型属性（如 `petOverlay`），合并时不要删它们
- **routes.ts / types.ts**：如果新增了路由或类型联合，确认相关类型定义同步更新

### i18n 文件冲突

- 上游可能在翻译文件中新增了键（如 `pet`、`importedBadge`），保留上游版本后只添加我们的翻译节
- 需要同步更新：类型定义（`types.ts`）+ 英文（`en.ts`）+ 中文（`zh.ts`）

### Flaky Test

- 某些测试可能不相关地失败（如路径不存在、环境特定）
- 在 PR 描述中注明："唯一失败是 xxx test，属于 pre-existing flaky test，与改动无关"

## 分支策略

### 推荐

```
main ← 上游主分支
feat/xxx ← 功能分支（从最新上游 main 创建）
```

### 禁止

```
❌ 从快照/独立仓库创建
❌ 在 main 分支上直接开发
❌ 功能分支长期不 rebase 上游（尽量每天同步一次）
```

## PR 描述模板

```markdown
## Summary

Implemented:
- Feature 1
- Feature 2
- Feature 3

## Verification
- typecheck: ✅
- tests: ✅
- Manual testing: ...

## Out of scope
- Item 1
- Item 2
```

## 不要做的事

- 不要用 `git checkout <old-branch> -- <file>` 解决冲突
- 不要在一个 commit 里混合多个不相关的功能
- 不要提交未通过 typecheck 的代码
- 不要跳过 CI 失败不分析直接重试
- 不要把 profile 当负责人（如果项目中有这个概念）
- 不要让 Renderer/前端直接访问后端存储

## 快速自查清单

提交前逐一确认：

- [ ] 是从官方 fork 的吗？
- [ ] 分支基于最新的上游 main？
- [ ] commit 有 `Signed-off-by`（`-s` 参数）？
- [ ] typecheck 通过？
- [ ] 测试通过？
- [ ] 没有误删上游代码？
- [ ] i18n 文件同步更新了？
- [ ] 相关文档更新了？
- [ ] CI 失败不是由我们的改动引起的？
