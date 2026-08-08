---
name: ai-subscription-iteration-workflow
description: PRJ-20260412-008 ai-subscription 项目从 PRD 到交付的完整高速迭代流程
---

# ai-subscription 高速迭代工作流

## 项目概述

GitHub: https://github.com/YeLuo45/ai-subscription
技术栈: React 18 + Vite 5 + Ant Design + shared/(LLM层/工具层)
工作目录: `/home/hermes/ai-subscription-new/`（有 .git）
Web-only 实现（PC/小程序通过调用 Web API 复用 shared/）

## 核心流程（每迭代约 10-15 分钟）

### Step 1: 方向确认
boss 选定 P0/P1/P2/P3 方向，小墨起草 PRD。

### Step 2: PRD 起草
- 路径: `~/.hermes/proposals/workspace-pm/proposals/P-YYYYMMDD-NNN-prd.md`
- 内容: 功能范围 + 技术方案 + 交付物清单 + 验收标准
- 注意: PRD 是自足的，不需要单独 tech-solution

### Step 3: 提案登记
- 路径: `~/.hermes/proposals/proposal-index.md`
- 新条目添加到文件末尾（在 P-YYYYMMDD-003 之后插入）
- 状态: `approved_for_dev`

### Step 4: 委托 dev agent
传递:
- PRD 路径
- 项目路径: `/home/hermes/ai-subscription-new/`
- 构建命令: `npx vite build`（不是 pnpm build）
- 注意事项:
  - npm install 被网络阻塞，用 `cp -r /home/hermes/ai-subscription/web/node_modules /home/hermes/ai-subscription-new/web/` workaround
  - 零新增大型依赖
  - PC/小程序不在本 iteration 范围

### Step 5: 验收检查
```bash
# 1. 检查 commit
cd /home/hermes/ai-subscription-new && git log --oneline -2

# 2. 检查文件状态
git status --short

# 3. 构建验证
cd /home/hermes/ai-subscription-new/web && npx vite build

# 4. 推送
cd /home/hermes/ai-subscription-new && git push origin master
```

### Step 6: 更新提案登记
- 状态: `accepted`
- Notes 追加 commit hash

---

## 已知模式

### Dev agent 交付模式
- dev agent 几乎总是 hit max_iterations 在 git push 之前
- **例外**: ai-subscription 的 dev agent 在 4/5 次迭代中自己完成了 push（commit 后 `git push origin master` 成功）
- **处理**: 每次都检查 `git status --short`，如有未提交变更则手动 commit + push

### Build 命令差异
- `npx vite build` 而非 `pnpm build`
- pnpm build 会先运行 tsc 类型检查，可能因缺少 @types/node 失败
- Vite build 跳过类型检查，直接打包

### node_modules 网络问题
- pnpm install 超时（180s, 300s 均失败）
- Workaround: `cp -r /home/hermes/ai-subscription/web/node_modules /home/hermes/ai-subscription-new/web/`
- nodemailer 等部分依赖已存在于原 node_modules，复制即可用

### 目录结构
```
ai-subscription-new/
├── web/                      # React 前端
│   ├── src/
│   │   ├── api/             # API 端点
│   │   ├── components/      # React 组件
│   │   ├── services/        # 服务层
│   │   ├── pages/           # 页面组件
│   │   └── styles/          # CSS
│   └── node_modules/        # 复制自 ai-subscription/web/
├── shared/                  # LLM层/工具层（PC端通过symlink复用）
│   └── lib/ai/
│       ├── llm.ts           # callLLM/streamLLM
│       ├── tools.ts         # 工具注册
│       └── utils/           # web-search/math-eval/rss-parser
├── ai-subscription-pc/     # Electron PC端
│   └── shared -> /home/hermes/ai-subscription-new/shared/  (symlink)
├── ai-subscription-miniapp/    # uni-app 小程序
└── ai-subscription-miniapp-taro/ # Taro 小程序
```

### ai-subscription vs ai-subscription-new
- `ai-subscription/`: 所有源文件（无 .git）
- `ai-subscription-new/`: git clone，有 .git，node_modules 已复制
- subagent 有时在 ai-subscription 中工作，需要手动复制文件到 ai-subscription-new

---

## 迭代历史

| P-20260504-001 | Tool-use 能力 | ✅ 1f18f68 |
| P-20260504-002 | 内容变换输出 | ✅ 6fd61e8 |
| P-20260504-003 | 多端 AI 层同步 | ✅ aacf1f2 |
| P-20260504-004 | 实时监控+推送 | ✅ 2bb2507 |
| P-20260504-005 | 高级内容变换 | ✅ 09a12ad |
| P-20260504-006 | 数据导入/导出 | ✅ 4df5047 |
| P-20260504-007 | PWA 离线支持 | ✅ 040d2aa |

---

## 关键教训

### proposal-index.md patch 唯一性
更新已存在条目时（如 `in_acceptance` → `accepted`），old_string 必须包含至少 4 行上下文（Proposal ID + Title + Owner + Current Status），否则会匹配到文件中其他条目导致 "Found N matches" 错误。

### subagent hit max_iterations 但已 push
有时 subagent 完成了代码实现并 commit/push 成功（exit_reason: completed），但实际推送到了不同的远程或分支。需要验证 `git log --oneline -2` 确认 commit 在正确的 repo 和分支上。

### build 产物大小监控
每次 build 后记录 bundle 大小（kB）。若大小异常（如从 190KB 骤增到 300KB+），说明引入了未预期的依赖，需要回溯检查。

### FeedList.tsx 是单页应用主文件
FeedList.tsx 是唯一的主页面（600+ 行），包含 renderFeeds/renderSettings/renderHistory 等多个面板。新增功能多数需要在这里集成（import + JSX 标签插入）。

### 组件拆分注意事项
新组件放在 `web/src/components/` 下，命名风格：`MyFeature.tsx`（PascalCase）。工具函数放在 `web/src/services/` 或 `web/src/utils/`。

### 纯前端实现约束
ai-subscription 的所有迭代都是纯前端实现（IndexedDB + 浏览器 API），无服务端持久化。邮件发送使用 fetch-based email API（如 email.moeyy.cn）而非 nodemailer（SMTP）。PDF 导出使用 `window.print()` + CSS `@media print`。这些选择避免了服务端依赖，适合静态部署场景。
