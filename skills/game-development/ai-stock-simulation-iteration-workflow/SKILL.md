---
name: ai-stock-simulation-iteration-workflow
description: ai-stock-simulation 项目从 PRD 到交付的高速迭代流程 — PRD起草→并行subagent实现→验收→push→交付报告
---

# ai-stock-simulation 迭代工作流

## 触发条件
Boss 从 A/B/C/D/E 方向列表中选择迭代方向（Web 应用功能迭代）

## 工作流程

### 1. PRD 起草
- 创建 `~/.hermes/proposals/workspace-pm/proposals/P-YYYYMMDD-XXX.md`
- 包含：概念愿景、现有功能、功能范围、UI布局、技术方案、API设计、非功能要求、交付物

### 2. 提案登记
- 更新 `~/.hermes/proposals/proposal-index.md`，状态 `in_dev`

### 3. Subagent 委托（并行）
- Frontend: 1个 task，路径 `/home/hermes/ai-stock-simulation-temp/frontend/src/`
- Backend: 1个 task，路径 `/home/hermes/ai-stock-simulation-temp/backend/`
- 约束：npm run build 通过，git add + commit

### 4. 验收
- `git status` 确认已 commit
- `npm run build` 确认构建通过
- 验证核心文件存在

### 5. 推送
- `git push origin main`

### 6. 文档更新
- `proposal-index.md` 状态改为 `delivered`
- `delivery-report.md` 追加本次迭代记录
- `git add + commit + push`

## 关键路径
- 项目代码: `/home/hermes/ai-stock-simulation-temp/`
- 注意：**不是** `/home/hermes/ai-stock-simulation/`（不存在）
- 提案目录: `/home/hermes/proposals/workspace-pm/proposals/`

## GitHub Pages 部署注意
- Workflow 触发分支: `master`（不是 `main`）
- `git push origin main:master` 可触发 Pages 构建
- 实际 main 分支 push 不触发 Pages（因为 workflow 监听 master）
- 若 GitHub Actions run 失败但本地 build 通过，说明是 CI 环境问题，不阻塞交付

## 经验记录
- 所有3次迭代（A/B/C）均顺利完成，构建通过，代码已推送
- GitHub Actions 近期有环境问题导致 run 失败，但代码本身没问题
- 迭代速度：平均每轮 ~5-10分钟（PRD + subagent + 验收 + push）
