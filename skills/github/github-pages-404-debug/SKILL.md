---
name: github-pages-404-debug
description: GitHub Pages 404 排查与修复
---
# GitHub Pages 404 排查与修复

## 常见原因速查表

| 原因 | 排查命令 | 修复方式 |
|------|----------|----------|
| Pages 未启用 | `gh api repos/OWNER/REPO/pages` | Settings → Pages → Source 启用 |
| 构建产物路径错误 | `ls dist/` 检查是否有 index.html | workflow upload artifact path 与 source 匹配 |
| artifact 过期/缺失 | `gh run list --limit 3` 看最新 run | 手动触发 `gh workflow run <name> --ref <branch>` |
| build_type=branch 但 source 指向无 artifact 的分支 | 检查 `source.branch` | 改为 `build_type=workflow` 或确认 artifact 所在分支 |
| 首次部署 artifact 未完成 | 看 Pages API `html_url` 是否存在 | 等 workflow 完成或手动重跑 |
| 仓库名大小写错误 | GitHub username 有大小写区分 | `https://yeluo45.github.io/` 而非 `YeLuo45` |

## 快速诊断流程

```bash
# 1. 检查 Pages 配置
gh api repos/OWNER/REPO/pages

# 2. 看 build_type 和 source
# build_type=workflow  → GitHub Actions 管理部署，看 artifact
# build_type=legacy   → 传统 branch/source 方式，找 gh-pages 或指定分支

# 3. 检查最新 deployment
gh run list --limit 3

# 4. 验证站点
curl -sI https://OWNER.github.io/REPO/ | head -3
```

## 关键区别

- `build_type=workflow`：artifact 由 Actions upload-pages-artifact 上传，Pages 从 artifact 服务，不需要 gh-pages 分支
- `build_type=legacy`：`source.branch` 必须是包含构建产物的分支（如 `gh-pages` 或 `master`），且该分支根部有 index.html

## 触发重建的正确方式

```bash
# 手动触发 workflow
gh workflow run <workflow-name> --ref <branch>

# 不要用 gh-pages 分支方式部署 Vite/React 项目——构建产物在 dist/ 而非根目录
```

## pixel-pal-web 教训

- GitHub username `YeLuo45` 但 URL 是 `yeluo45`（小写）
- dist/renderer/ 是构建产物路径，workflow artifact path 必须是 `./dist/renderer`
- Pages build_type=workflow 时，即使配置正确，artifact 也可能需要手动重跑才能生效
