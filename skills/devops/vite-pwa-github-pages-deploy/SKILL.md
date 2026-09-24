---
name: vite-pwa-github-pages-deploy
description: Vite + PWA 项目部署到 GitHub Pages 的完整流程，包含常见问题解决
triggers:
  - 部署 Vite 项目到 GitHub Pages
  - PWA 部署失败
  - gh-pages 分支污染
---

# Vite + PWA 项目部署到 GitHub Pages 规范流程

## 触发条件
- React/Vite 项目需要部署到 GitHub Pages (gh-pages 分支)
- 适用 PWA 项目

## 核心流程

### 1. 首次初始化（必须严格按顺序）

```bash
# 1. 创建项目目录并初始化
mkdir project-name && cd project-name
npm create vite@latest . -- --template react

# 2. 立即创建 .gitignore（必须在任何 git add 之前）
cat > .gitignore << 'EOF'
node_modules/
dist/
.DS_Store
*.local
EOF

# 3. 首次提交 .gitignore
git init
git add .gitignore
git commit -m "chore: add .gitignore"

# 4. 添加所有源码并提交
git add .
git commit -m "feat: initial commit"

# 5. 创建 GitHub 仓库并推送
gh repo create project-name --public --source=. --push

# 6. 安装依赖并构建
npm install
npm run build
```

### 2. vite.config.js 关键配置

```javascript
export default defineConfig({
  base: './',  // 重要：部署到子目录时必须用相对路径
  plugins: [react(), VitePWA({...})]
})
```

**常见错误**：`base: '/project-name/'` 会生成绝对路径，导致 GitHub Pages 无法加载资源。

### 3. 部署到 gh-pages（每次更新）

```bash
# 切换到 master 确保有最新源码
git checkout master

# 安装依赖并构建
npm install
npm run build

# 创建干净的 gh-pages（只包含 dist 内容）
git checkout --orphan gh-pages-deploy
rm -rf ./*
cp -r dist/* .
git add .
git commit -m "Deploy $(date -u +%Y%m%d-%H%M%S)"
timeout 60 git push origin gh-pages-deploy:gh-pages --force
git checkout master
```

### 4. GitHub Pages 配置

- 仓库 Settings → Pages → Source: **gh-pages** 分支, / (root)
- 首次部署后需要等 1-2 分钟让 GitHub 构建
- 检查 Actions tab 查看构建状态

## 常见问题

### 页面空白，控制台报 `ReferenceError: xxx is not defined`
- 原因：gh-pages 部署的是旧版本代码（未包含最新修复）
- 解决：重新执行部署流程，确保 dist 是最新构建

### 资源 404（JS/CSS 文件加载失败）
- 原因：vite.config.js 的 `base` 配置错误
- 解决：`base: './'` 使用相对路径

### 问题：GitHub Pages 子目录部署时 assets 路径 404（302 重定向）
**现象**：`vite.config.js` 设置 `base: '/todo-list/'`，`dist/index.html` 中路径正确为 `/todo-list/assets/main-xxx.js`，但 GitHub Pages 响应 302 重定向到 `/assets/main-xxx.js`（丢失了子目录前缀）。

**原因**：GitHub Pages static hosting 对子目录前缀处理不当。

**解决**：改用相对路径 `base: './'`，Vite 会生成相对路径 `src="assets/main-xxx.js"`，避免绝对路径问题。

### 问题：legacy build 和 workflow 部署冲突（双重写入 gh-pages）
**现象**：`pages build and deployment` workflow（legacy）从源码构建后覆盖了 `peaceiris/actions-gh-pages` 推送到 gh-pages 的文件，导致 `index.html` 有错误的 asset 路径。

**原因**：
1. GitHub Pages 配置为 `build_type: legacy` 时，每次 push 到 main 都会自动从源码重新构建
2. `peaceiris/actions-gh-pages` 推送到 gh-pages 后，legacy build 又跑一遍覆盖了结果
3. legacy build 的 Vite 构建产物路径与 workflow 构建的路径不一致

**解决（三步）**：
1. **重命名 workflow**：避免与 GitHub 内置 `pages-build-deployment` 名称冲突
2. **切换 GitHub Pages 到 workflow 模式**：API `PUT /repos/{owner}/{repo}/pages` 设置 `"build_type":"workflow"`。workflow 模式下 GitHub 的 `pages build and deployment` 找不到 artifact 会失败（不覆盖）
3. **使用 `peaceiris/actions-gh-pages`**：推送预构建的 `dist/` 到 gh-pages 分支，绕过 GitHub Pages 源码构建

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci --legacy-peer-deps
      - run: npm run build

> **CI peer dependency conflict**: `npm ci` in GitHub Actions often fails with peer dep errors (e.g. i18next@24.2.3 requires typescript@^5 but project has typescript@6). Always use `--legacy-peer-deps` flag for Vite/PWA projects in CI.
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

**注意**：`peaceiris/actions-gh-pages` 需要 GitHub Pages 在仓库设置中已开启。如果 Pages 被完全关闭，该 action 会失败。

### push 超时（gh-pages 包含 node_modules）
- 原因：之前某次推送把 node_modules 提交到了 gh-pages，远程保留了历史
- 解决：删除远程分支后重建
  ```bash
  git push origin :gh-pages  # 删除远程分支
  git push origin <干净commit>:refs/heads/gh-pages  # 重建
  ```

### git checkout 切换分支失败
- 错误：`error: Your local changes to the following files would be overwritten by checkout`
- 原因：当前分支有未提交的修改，与目标分支冲突
- 解决：`git stash` 或 `git checkout -f <branch>`

### master 被污染（包含 dist/node_modules）
- 原因：.gitignore 建立前执行了 `git add .`
- 解决：找到干净 commit `git reset --hard <干净-commit>`
  ```bash
  git log --oneline  # 找到最早的干净提交
  git reset --hard <commit-sha>
  ```

## 关键原则

1. **.gitignore 必须在首次 commit 前存在并生效**
2. **dist/ 永远不要进入任何分支的版本控制**
3. **master = 源码，gh-pages = 仅 dist 内容**
4. **部署前先在本地验证 `npm run preview`**
5. **push 超时用 `commit:refs/heads/branch` 语法绕过**

## 验证步骤

部署后访问页面，检查：
- [ ] 页面正常加载（不是空白）
- [ ] 控制台无 JS 错误
- [ ] 资源路径是相对路径（`./assets/xxx.js` 而不是 `/project-name/assets/xxx.js`）
- [ ] 点击"开始游戏"能正常进入游戏
