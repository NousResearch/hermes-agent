---
name: github-unrelated-histories-force-replace
description: 当两个 Git 分支历史完全无关时，如何用 feature 分支强制替换 main 分支。解决 PR merge conflicts 和 "refusing to merge unrelated histories" 问题。
tags: [github, git, merge, force-push, workaround]
related_skills: [github-api-push-when-network-blocks-git]
---

# GitHub：两个分支历史无关时用 feature 替换 main

## 典型场景

两个分支 commit 历史完全不同，无共同祖先：
- `main` 分支有旧代码（deploy.yml、README 等初始配置）
- `feature/sup_wx` 分支有新代码（通过 `git init` 新建后开发）
- GitHub PR 显示 "merge conflicts" 且 squash merge 失败
- 本地 `git merge --allow-unrelated-histories` 失败或 exit 1

## 实测最简方案：reset + embedded PAT push

全程只需要 5 步，8 秒完成推送：

```bash
# 1. 用 FETCH_HEAD（之前 fetch 的 origin/main）创建本地 main
git branch main FETCH_HEAD

# 2. 切换到 main
git checkout main

# 3. 尝试 merge（如果失败，跳到步骤4）
git merge feature/sup_wx --no-edit --allow-unrelated-histories
# 常见失败原因A: "Committer identity unknown" → 先设置 git identity
# 常见失败原因B: exit 1 但无 stderr → histories 完全无关，跳到步骤4

# 4. 直接用 feature 分支重置 main（跳过 merge 冲突）
git reset --hard feature/sup_wx

# 5. 用 embedded PAT 强制推送到远程 main（8秒完成）
git push https://ghp_TOKEN@github.com/OWNER/REPO.git main -f
```

**为什么这个方案最快**：
- 不需要 REST API 的 blob→tree→commit→ref 四步流程
- 不需要解决冲突（用新分支完全覆盖旧分支）
- embedded PAT URL 避免 credential prompt 挂起
- 8秒完成推送

## 关键前提

1. 必须先 `git fetch origin main` 获取远程 main 的 FETCH_HEAD
2. 如果还没 fetch：`git fetch --depth=1 origin main`
3. 设置 git identity（merge 需要）：
   ```bash
   git config user.email "hermes@yluo.me"
   git config user.name "Hermes Agent"
   ```

## 注意事项

- 这会**丢失 main 分支上的所有旧提交**（如果 main 上有重要历史，先提取）
- 推送前确认 feature 分支包含所有需要的代码
- 删除远程分支前必须先切到其他分支

## 后续清理

```bash
# 关闭 PR（PR #1）
gh api repos/OWNER/REPO/issues/1 -X PATCH -f state=closed

# 删除远程分支（需要先切到 main）
git checkout main
git push https://ghp_TOKEN@github.com/OWNER/REPO.git --delete feature/sup_wx
# 报错 "refusing to delete current branch" → 还在 feature 分支，先 checkout main
```

## 快速验证

```python
import urllib.request, json, subprocess

token = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True).stdout.strip()

# 验证远程 main 已更新
req = urllib.request.Request(
    'https://api.github.com/repos/OWNER/REPO/branches/main',
    headers={'Authorization': 'token ' + token}
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read())
    print('Remote main SHA:', data['commit']['sha'])
    print('Expected:       ', 'FEATURE_BRANCH_LATEST_SHA')
    print('Match:', data['commit']['sha'] == 'FEATURE_BRANCH_LATEST_SHA')
```

## 实测记录

```
feature/sup_wx (6个新commit) → 替换 main（1个旧commit）
Push: 8秒
结果: 远程 main SHA = 984c5e0 (feature分支最新commit) ✓
```