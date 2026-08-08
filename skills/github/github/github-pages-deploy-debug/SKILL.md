---
name: github-pages-deploy-debug
description: GitHub Pages 部署排障 — 当 deploy-pages 失败时快速定位根因（Pages 未配置 vs workflow 问题）
---
# GitHub Pages 部署排障指南

## 典型场景
GitHub Actions `deploy-pages` 失败（exit code 1），但找不到明确原因。

## 排查步骤

### 1. 确认 GitHub Pages 是否已启用
```bash
gh api repos/{owner}/{repo}/pages
```
- 返回 404 → Pages 未配置
- 返回配置信息 → Pages 已启用

### 2. 如果 Pages 未启用（根因）
Actions workflow 使用 `actions/deploy-pages@v4` 时，如果仓库从未配置过 GitHub Pages，deploy job 会静默失败，错误信息不清晰。

**解决方案：通过 API 启用 Pages**
```bash
gh api repos/{owner}/{repo}/pages --method POST \
  -f build_type=workflow \
  -f source[branch]=master \
  -f source[path]=/

# 或使用 curl（需要 PAT）
curl -s -X POST https://api.github.com/repos/{owner}/{repo}/pages \
  -H "Authorization: token {PAT}" \
  -H "Accept: application/vnd.github.v3+json" \
  -d '{"build_type":"workflow","source":{"branch":"master","path":"/"}}'
```

### 3. 验证启用成功
```bash
curl -s -o /dev/null -w "%{http_code}" https://{owner}.github.io/{repo}/
# 应返回 200
```

### 4. 触发重新部署
启用 Pages 后不会自动部署，需要 push 新提交或手动触发 workflow：
```bash
# 手动触发 workflow
gh api repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs \
  --method POST -f ref=master
```

## 场景二：验收测到旧功能（feature 分支未同步到 gh-pages）

**症状**：GitHub Actions 显示 deploy 成功，GitHub Pages 也能访问，但新功能验收时发现是旧版 UI。

**根因**：
- feature 分支包含新代码，但 gh-pages 没有同步
- GitHub Pages 部署的是 gh-pages 分支，不是 feature 分支
- 常见于多人协作或跨分支开发后忘记合并

**排查步骤**：
```bash
# 1. 确认 GitHub Actions 最后一次 deploy 是何时
gh run list --repo {owner}/{repo} --limit 3

# 2. 确认 gh-pages 分支的最新提交
git log --oneline origin/gh-pages | head -3

# 3. 对比 feature 分支和 gh-pages 的差异
git log --oneline origin/gh-pages..origin/feature/agent
```

**解决方案**：
```bash
# 方式A：合并 feature 到 gh-pages（推荐）
git checkout gh-pages
git merge feature/agent
git push origin gh-pages

# 方式B：直接 cherry-pick 需要的功能
git checkout gh-pages
git cherry-pick {commit-hash}
git push origin gh-pages

# 方式C：对于纯 HTML 项目，直接从 feature 分支复制文件到 gh-pages
git checkout feature/agent -- path/to/file.html
git add path/to/file.html
git commit -m "sync from feature/agent"
git push origin gh-pages
```

**验证**：
```bash
# 1. GitHub Actions 自动触发，等待约1分钟
gh run list --repo {owner}/{repo} --limit 1

# 2. 确认 GitHub 实际内容已更新
curl -s https://{owner}.github.io/{repo}/ | md5sum

# 3. 确认文件修改时间
curl -sI https://{owner}.github.io/{repo}/ | grep last-modified
```

## 场景三：本地 build 成功但 GitHub Actions CI 失败

**症状**：`npm run build` 本地通过，但 GitHub Actions workflow run 失败（run failed）。

**根因**：subagent 创建了文件但从未 `git add` + `git commit`。本地 `dist/` 有旧缓存掩盖问题，但 GitHub Actions runner 执行 `npm ci` + `npm run build` 全新构建时，找不到未入 git 的源文件。

**排查步骤**：
```bash
# 1. 检查 untracked 文件（subagent 常忘记 git add）
git status --short

# 2. 检查关键文件是否在 git 历史中
git ls-files src/components/GanttChart/  # 如果为空 = 从未 commit
git ls-files src/pages/DashboardView.jsx   # 如果为空 = 从未 commit

# 3. 检查 git tree 中是否存在（对比 CI 构建用的 commit）
gh api repos/{owner}/{repo}/git/trees/{commit-sha}?recursive=1 \
  --jq '.tree[].path' | grep -E "GanttChart|Dashboard"

# 4. 验证 CI 用的 commit 包含所有源文件
git log --oneline origin/master | head -3
gh api repos/{owner}/{repo}/actions/workflows/deploy.yml/runs \
  --jq '.workflow_runs[0].head_sha'
```

**解决方案**：
```bash
# 如果关键文件从未 commit，加入 git 并 push
git add src/components/NewComponent/
git add src/pages/NewView.jsx
git add src/hooks/useNewHook.js
git status --short   # 确认只有要提交的文件
git commit -m "feat: add missing components for V3 iteration"
git push origin master

# 验证 CI
gh run list --repo {owner}/{repo} --workflow=deploy.yml --limit=1
```

**验证本地 build 真正通过（无缓存影响）**：
```bash
rm -rf dist node_modules/.vite
npm ci
npm run build
# 只有这个顺序通过，才说明构建真正独立于缓存
```

## 关键洞察
- `deploy-pages@v4` 失败的最常见原因不是 workflow 本身，而是 GitHub Pages 根本没配置
- Pages 未配置时，Actions 不会报错说"请先启用 Pages"，而是直接失败
- 通过 API 启用 Pages 是绕过 UI 登录限制的有效方法
- **验收测试前**：务必确认 GitHub Pages 部署的内容是 expected commit（不仅是 "latest deploy success"）
- **纯 HTML 项目**：可以直接从 feature 分支复制 index.html 等文件到 gh-pages 快速同步
