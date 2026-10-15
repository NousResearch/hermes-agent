---
name: git-multi-branch-merge
description: 批量合并多个分支到 master 主干的标准化流程，适用于大量 v-prefixed feature 分支的仓库。
---

# Git Multi-Branch Merge Workflow

批量合并多个分支到主干的标准流程。

## 适用场景
- 维护大量 feature/v-prefixed 分支的仓库
- 一次性将所有未合并分支同步到 master

## 标准流程

```bash
# 1. 获取最新远程分支列表
git fetch --all

# 2. 找出所有 v-prefixed 本地分支
git branch --no-color | grep '^  v' | sed 's/^  //'

# 3. 对每个分支检查是否已合并，是则跳过
git merge-base --is-ancestor origin/<branch> master && echo "already merged" || echo "need merge"

# 4. 执行合并（无冲突）
git merge origin/<branch> --no-edit

# 5. 有冲突时：解决冲突
git checkout --ours <file>           # 取本地（master）版本
git add <file>
git commit -m "merge: <branch> into master"
```

## 关键经验

1. **检测分支别名**：v21-multi-persona-v2 和 v18-plugin-system 指向同一 commit（SHA fb36f40），是别名/影子分支。合并前先检查 SHA 是否相同。

2. **冲突解决策略**：deploy.yml 等 CI 配置类文件，始终用 `--ours`（master 版本），因为 master 上已修复的 CI 配置比旧分支的更正确。

3. **跳过无远程分支**：`git fetch` 后 origin/ 列表中不存在的分支直接跳过。

4. **逐个合并而非批量**：避免一次性 `git merge branch1 branch2 branch3`，难以追踪冲突。

## 验证步骤

```bash
git log --oneline -5              # 确认 merge commit
git push origin master             # 确认 push 成功
```
