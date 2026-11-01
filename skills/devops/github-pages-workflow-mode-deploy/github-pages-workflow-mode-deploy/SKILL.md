---
name: github-pages-workflow-mode-deploy
description: GitHub Pages workflow mode deployment — switching from legacy to Actions-based build, avoiding gh-pages force-push mistakes
---

# GitHub Pages Workflow Mode 部署

## 背景
GitHub Pages 有两种构建模式：
- **legacy（静态站点）**：从指定分支的源码目录读取，GitHub Pages 自动构建。不读取 gh-pages 分支内容。
- **workflow（Actions）**：通过 GitHub Actions workflow 管理部署，workflow 负责构建并推送到指定分支。

legacy 模式下直接 push dist 到 gh-pages 分支是无效的——GitHub Pages 根本不从那里读取。

## 识别当前模式
```bash
gh api repos/{owner}/{repo}/pages
# 查看 build_type 字段
```

## 切换到 workflow 模式

**⚠️ 重要坑：无法通过 API 从 legacy 切换到 workflow 模式**

当 Pages 处于 legacy 模式时，`PUT /pages` 会返回：
```
GitHub Pages is not built from a workflow
```
API 无法完成此切换。**必须手动在 GitHub Web UI 操作一次**：
1. 打开 `https://github.com/{owner}/{repo}/settings/pages`
2. **Build and deployment → Source** 选择 **GitHub Actions**
3. 保存

之后才能用 API 管理 Pages 配置。

---

## 使用 actions/deploy-pages@v4（官方 Actions 部署）

这是 GitHub 官方部署 Action，比 peaceiris/actions-gh-pages 更可靠：

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: master
          path: src
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - name: Install dependencies
        run: |
          cd ${{ github.workspace }}/src
          npm config set registry https://registry.npmmirror.com
          npm install --legacy-peer-deps
      - name: Build
        run: cd ${{ github.workspace }}/src && npm run build:h5
      - name: Upload pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: src/dist/build/h5

  deploy:
    needs: build
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

**关键权限：** `pages: write` 和 `id-token: write` 是 `actions/deploy-pages` 必须的。

**⚠️ 坑：legacy 模式下 deploy-pages 不更新 branch**

当 Pages source 是 **gh-pages branch（legacy）** 时：
- `actions/deploy-pages` 部署到 CDN，但 **不更新 gh-pages 分支**
- 网站继续从旧的 gh-pages branch 取文件
- 解决：切换 Pages source 到 **GitHub Actions** 模式（手动操作一次，见上文）

## 标准 workflow（peaceiris/actions-gh-pages）
```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
permissions:
  contents: write
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
      - name: Install dependencies
        run: npm ci
      - name: Build
        run: npm run build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

## 常见坑
1. **gh-pages 分支被 legacy 构建覆盖**：legacy 模式下 GitHub Pages 从源码构建后会自动更新 gh-pages，force-push 到 gh-pages 会被下次构建覆盖。
2. **workflow 没有写入权限**：必须添加 `permissions: contents: write`。
3. **vite base 路径**：部署到子目录（如 `/todo-list/`）时，需要确保 `vite.config.js` 的 `base: '/todo-list/'` 正确，且 HTML 中的资源路径也指向子目录。

## 触发 workflow 模式的 GitHub Pages 部署

### 重新触发已有 deployment
```bash
# 重新触发指定的 workflow run
gh run rerun {run_id} --repo {owner}/{repo}
```

### 手动触发（推荐）
当 workflow 有 `workflow_dispatch` trigger 时，通过 API 触发：
```bash
# 触发指定 workflow
gh api --method POST repos/{owner}/{repo}/actions/workflows/{workflow-file}/dispatches -f ref={branch}

# 示例：触发 master 分支的 deploy.yml
gh api --method POST repos/yeluo45/pixel-pal-web/actions/workflows/deploy.yml/dispatches -f ref=master
```

**⚠️ 关键坑**：当 GitHub Pages source 设置为 "GitHub Actions" 时，`POST /repos/{owner}/{repo}/pages/builds` 返回 403 "The repository does not have a GitHub Pages site"。必须用 workflow dispatch API 触发。

### 验证部署状态
```bash
# 查看 workflow 运行状态
gh run list --workflow={workflow-file} --limit=1

# 查看 Pages 配置
gh api repos/{owner}/{repo}/pages --jq '.html_url'
```

## 关键经验

- legacy 模式下 gh-pages 分支不可靠，应该完全依赖 GitHub Pages 的自动构建。
- workflow 模式下 gh-pages 由 action 管理，不要手动 force-push 到该分支。
- 部署失败时，先确认 build_type 是否正确切换到了 workflow。
- **永远不要让 legacy build 和 workflow build 同时运行**：legacy 的 `pages build and deployment` workflow 会和自定义 workflow 互相覆盖 gh-pages，导致 404。如果两种模式冲突，先用 `build_type=workflow` 切换，然后用 workflow 唯一负责部署。
- **切换 build_type 必须用 `--input -` + JSON stdin**：`-F` flags 会导致 build_type 被静默忽略，始终保持 legacy。
- **artifact serving**：workflow mode 的 Pages 部署通过 artifact 系统，不直接操作 gh-pages 可见的 blob，但最终通过 `https://{user}.github.io/{repo}/` 可访问。

### peaceiris/actions-gh-pages@v3 失败时的解决方案

**症状**：`peaceiris/actions-gh-pages@v3` deploy 步骤失败，日志显示 `failed to determine base repo: failed to run git`。

**原因**：某些环境下（如 `GITHUB_TOKEN` 权限受限或 repo 配置特殊），`peaceiris/actions-gh-pages` 需要 git repo 可写但无法正常获取。

**解决方案**：改用 GitHub 官方 `actions/deploy-pages@v4`，同时需要：
1. 添加 `permissions: pages: write, id-token: write`
2. 使用 `actions/upload-pages-artifact@v3` 上传构建产物
3. 不需要 `github_token`（官方 action 使用 OIDC）

**验证 deployment 状态**：
```bash
# 查看 deployment 状态（不是 workflow 状态）
gh api repos/{owner}/{repo}/deployments --paginate | python3 -c "
import sys,json
for line in sys.stdin: d=json.loads(line); print(d['id'], d['environment'], d['statuses_url'])
"

# 查看具体 deployment 的 statuses
gh api {statuses_url}  # 最后一个应该是 success

# 验证 artifact 存在
gh api repos/{owner}/{repo}/actions/artifacts --jq '.artifacts[] | select(.workflow_run.head_sha == \"COMMIT_SHA\") | .id, .name, .size_in_bytes'
```

### deployment API 返回 success 但站点 404

**症状**：`/deployments/{id}/statuses` 显示 `success`，`gh run list` 显示 CI success，但 `https://user.github.io/repo/` 持续 404。

**诊断步骤**：
1. `gh api repos/{owner}/{repo}/pages` 确认 `build_type: workflow`
2. `gh run rerun {run_id}` 重新触发部署
3. 等 1-2 分钟再试（GitHub Pages CDN 有 propagation 延迟）
4. 检查 `curl -sI https://user.github.io/repo/` — 如果 header 是 `HTTP/2 404` 但内容是 GitHub Pages 自己的 "Page not found" 页面，说明 Pages 基础设施收到了请求但没有正确路由

**经验**：deployment success + 404 持续存在通常是 GitHub Pages 内部问题，不是 deployment 问题。可以多等几分钟或重新触发。

## gh-pages 被意外覆盖为源码时的修复

**场景**：执行 `git push origin master:gh-pages -f` 时，实际上从源码仓库的 master 分支推送——推送的是源码（.jsx/.html）而非 dist/ 构建产物。GitHub Pages 随后错误地提供源码文件，浏览器无法执行模块化的 JSX/JS，导致白屏。

**诊断**：
```bash
# 检查 assets 的 content-type
curl -sI "https://username.github.io/repo/assets/index-xxx.js" | grep content-type
# 如果返回 text/html → gh-pages 被源码污染
# 正确应返回 application/javascript
```

**修复步骤**：
1. 源码仓库中重新构建：`npm run build`（dist/ 被更新）
2. 进入 dist/ 目录，将其初始化为独立 git repo 并推送：
```bash
cd dist
git init
git add -A
git commit -m "deploy: $(git rev-parse --short HEAD^)"
git remote add origin https://github.com/OWNER/REPO.git
git push origin +master:gh-pages   # 必须用 +master:gh-pages，不能从源码 repo 推送
```

**预防**：永远不要在源码仓库中执行 `git push origin master:gh-pages`，这只应该从 dist/ 目录推送。
