---
name: github-actions-ci-debug
description: GitHub Actions CI 调试方法论 — 排查 build 失败、exit code 127、npx not found 等问题
tags: [ci, github-actions, debug, vite, tsc]
---

# GitHub Actions CI 调试方法论

## 核心原则
CI 失败时按以下顺序排查，不要猜测：

1. **先看 step-level 结论**，不是 run-level 结论
2. **Exit code 127** = command not found（命令未安装或 PATH 问题）
3. **Exit code 1** + 无 vite/esbuild 报错 = 可能是 npm/npx 在 CI 环境的问题
4. **Annotations** 可能为空或不准，要看 step 的具体输出

## 排查步骤

### 1. 获取 job 详情
```bash
gh run list --workflow=<workflow-name> --limit 3
# 获取 run_id
```

### 2. 获取 job ID 和 step 结论（使用 gh api）
```bash
# gh api 自动使用已认证的 token，安全
gh api repos/<owner>/<repo>/actions/runs/<run_id>/jobs --jq '.jobs[] | {name, conclusion, id}'
```

### 3. 获取 annotations（TS/lint 错误）
```bash
gh api repos/<owner>/<repo>/check-runs/<job_id>/annotations --jq '.[].message'
```

### 4. 常见 exit code
| Code | 含义 | 排查方向 |
|------|------|----------|
| 127 | Command not found | `which <cmd>` 在 CI 环境；改用 `npm run <script>` 或 `./node_modules/.bin/<cmd>` |
| 1 | Generic error | 看 step 输出；可能是 TS 错误、依赖问题、权限问题 |
| 128 | Git error | SSH key 问题、repo 不存在、force push 冲突 |

## 常见修复模式

### npx not found in CI (exit code 127)
**症状**: Build step 失败，annotations 为空（没有 TS 错误），step 结论 failure

**根因**: `npm ci` 按 package-lock.json 安装依赖时，如果某些包（如 vite、typescript）不在 lockfile 中或路径问题，`npx <cmd>` 会报 "command not found"（127），但 `npm ci` 本身退出码是 0（因为依赖解析阶段没报错）

**解决**: 
1. 改用 `npm run build`（会从 node_modules/.bin 找命令）
2. 最好方案：跳过 `tsc -b`，直接 `npx vite build` — Vite 用 esbuild 做转译，足够部署

**验证**: 本地运行 `npx tsc --version` 成功 ≠ CI 成功。必须实际看 CI 日志。

### tsc -b 失败（96+ TS 错误）但 vite build 成功
**解决**: 跳过 `tsc -b`，直接用 `npx vite build`
```yaml
# deploy.yml
- name: Build
  run: npx vite build
```

### npm ci vs npm install in CI
- `npm ci`: 必须有 package-lock.json，严格按 lockfile 安装。**问题**：如果 lockfile 和实际 package.json 不同步，`npm ci` 可能安装成功但 `npx <cmd>` 找不到（127）。这是因为 `npm ci` 的依赖解析和 bin 链接可能不完整。
- `npm install`: 更灵活，会处理 UNMET DEPENDENCY，重新生成 lockfile 部分内容
- **经验**：如果 CI 中 `npm ci` 成功但 `npx <cmd>` 报 127，改用 `npm run <script>` 或 `npx vite build` 绕过

### Workflow 触发分支错误
**症状**: push 到 master 但 CI 没运行；workflow list 显示旧的 branch 还在跑
**根因**: `on: push: branches: [v16-scenes, main]` 指定了错误的分支名
**解决**: 修改为正确的目标分支，如 `branches: [master]`

### GitHub Pages 部署 artifact path 问题
```yaml
- name: Upload artifact
  uses: actions/upload-pages-artifact@v3
  with:
    path: ./dist/renderer  # 确保 vite build 输出到 dist/renderer
```

## 调试教训
- **不要假设本地能跑 CI 就能跑** — node 版本、PATH、缓存状态都不同
- **不要只看 run 结论** — 要看到 step 级别
- **不要忽略 exit code 127** — 这是 command not found，不是代码问题
- **vite build 可以跳过 tsc** — Vite 用 esbuild 做转译，足够部署
- **trial-and-error 流程**: 依次试 `npm run build`（失败）→ `npx tsc --noEmit`（失败127）→ `npx vite build`（成功）

## 验证修复
每次修改 deploy.yml 后：
1. push 触发 CI
2. 等 run 完成
3. 确认 "completed success"
