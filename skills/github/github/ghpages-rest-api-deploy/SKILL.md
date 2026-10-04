---
name: ghpages-rest-api-deploy
description: gh-pages 部署：网络阻塞 git push 时的 REST API + 独立 git repo 混合策略
---

# gh-pages 部署：REST API + git push 混合策略

当网络阻塞 git push 但 GitHub API 可用时的 gh-pages 部署流程。

## 核心问题
- git push 直接阻塞
- GitHub API `POST /git/blobs` 对 >1MB 的文件会超时
- git subtree split 推送到已有分支会丢文件

## 推荐方案：用临时 git repo 推送

```bash
# 1. 复制 dist 到临时目录
cp -r /path/to/dist /tmp/gh-deploy

# 2. 初始化独立 git repo
cd /tmp/gh-deploy && git init
git config user.email "hermes@agent.local"
git config user.name "Hermes Agent"

# 3. 设置带 token 的 remote（绕过网络阻塞）
git remote add origin https://ghp_TOKEN@github.com/OWNER/REPO.git

# 4. 提交并推送
git add . && git commit -m "deploy"
git push origin master:gh-pages --force
```

## REST API 方案（适合 <1MB 总量的文件）

```python
# 上传 blobs → 创建 tree → 创建 commit → 更新 ref
# 注意：>1MB 的 blob 需重试（timeout 后等待再试）
```

## GitHub Pages 模式切换

| 模式 | 说明 |
|------|------|
| legacy | 直接从 gh-pages 分支服务 |
| workflow | 由 Actions 触发，每次 push 自动部署 |

切换后必须触发 rebuild：

```bash
# 切换模式
curl -X PATCH -H "Authorization: token TOKEN" \
  -d '{"build_type": "legacy", "source": {"branch": "gh-pages", "path": "/"}}' \
  https://api.github.com/repos/OWNER/REPO/pages

# 触发重建
curl -X POST -H "Authorization: token TOKEN" \
  https://api.github.com/repos/OWNER/REPO/pages/builds
```

## 常见坑

| 问题 | 原因 | 解决 |
|------|------|------|
| 推送后 Pages 仍 404 | Pages 还在用旧 commit | POST `/pages/builds` 触发重建 |
| `PATCH /git/refs/heads/gh-pages` 返回 422 | ref 已存在 | 改用 `POST /git/refs/heads/gh-pages` |
| git subtree 丢文件 | split 出的 tree 不完整 | 用临时独立 repo 代替 |
| blob upload 超时 | 文件 >1MB | 用临时 git repo 推送 |
| workflow 模式无法覆盖 | Actions 自动重建 | 切回 legacy + 手动 rebuild |
