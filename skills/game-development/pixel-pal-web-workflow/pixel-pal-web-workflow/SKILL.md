---
name: pixel-pal-web-workflow
description: PRJ-20260420-002 PixelPal AI companion 标准化开发流程 - React 19 + TypeScript + Electron + Vite + MUI + Zustand + IndexedDB (idb)
---

# PixelPal Web 开发流程

## 概述

PRJ-20260420-002 PixelPal AI companion 桌面应用的标准化开发流程。pixel-pal-web 迭代模式：提案登记 → dev agent 委托 → 验收 → GitHub Actions 构建部署。

**线上地址**: https://YeLuo45.github.io/pixel-pal-web
**GitHub**: YeLuo45/pixel-pal-web
**代码路径**: /home/hermes/.hermes/proposals/workspace-dev/proposals/pixel-pal-web/
**提案路径**: ~/.hermes/proposals/workspace-pm/proposals/

## 技术栈

- React 19 + TypeScript + Electron + Vite
- MUI (Material UI) + Zustand (状态管理)
- IndexedDB via `idb` (持久化存储)
- GitHub Pages + GitHub Actions (`Deploy to GitHub Pages` workflow)

## 架构概览

```
src/
├── components/PixelPal/      # 像素伙伴 UI 组件
│   ├── PixelPal.tsx          # 核心伙伴组件
│   ├── CompanionCanvas.tsx   # 伙伴容器（含动作通知层）
│   ├── ActionToast.tsx       # 主动动作 Toast
│   └── ActionBadge.tsx       # 动作计数徽章
├── hooks/
│   └── useInteractionEngine.ts  # 注册动作触发器
├── pages/
│   └── MainPage.tsx          # 主页面（使用 CompanionCanvas）
├── services/
│   ├── companion/             # 伙伴人格系统
│   │   ├── companionService.ts
│   │   └── personalityTypes.ts
│   ├── memory/               # 记忆持久化系统
│   │   ├── memoryTypes.ts    # 8 种记忆类型
│   │   ├── memoryStorage.ts  # IndexedDB 操作
│   │   ├── MemoryStore.ts    # Zustand 兼容 store
│   │   └── MemoryManager.ts  # 摘要生成 + 上下文注入
│   └── actions/              # 主动动作系统
│       ├── ActionTypes.ts    # CompanionAction 联合类型
│       ├── ActionEngine.ts   # 动作队列引擎（单例）
│       └── ActionTrigger.ts  # 触发条件评估
└── stores/                   # Zustand stores
    ├── companionStore.ts      # 人格/情绪状态
    ├── memoryStore.ts         # 记忆统计
    └── petStore.ts           # 宠物状态
```

## 标准迭代流程

### 1. 起草提案

创建 `workspace-pm/proposals/P-YYYYMMDD-NNN-intake.md`，在 `proposal-index.md` 登记为 `intake`。

### 2. 确认 PRD

与用户确认 PRD 内容，更新 proposal-index.md 为 `approved_for_dev`。

### 3. 委托开发

使用 `delegate_task` 委托 dev agent，传递：
- 项目路径
- PRD 路径
- 关键文件列表和架构说明

### 4. 等待 CI 构建

本地 `npm install` / `npm run build` 通常会失败（网络阻塞、rolldown binding 损坏）。**依赖 GitHub Actions CI 验证**：
```bash
gh run list --repo YeLuo45/pixel-pal-web
gh run view <run-id>
```

### 5. TS 错误修复轮次

通常需要 2-4 轮修复才能通过 CI：
- **第 1 轮**: 明显错误（unused imports/vars、wrong function names）
- **第 2 轮**: 隐蔽错误（type export patterns、`dueToday` 被错误识别为 unused）
- **第 3 轮**: 清理残留

关键教训：
- `dueToday = isToday(dueDate)` 在 `ActionTrigger.ts` 中 **是用于 urgency 计算的**，不要误删
- `ActionQueueItem` 类型重导出需要 `import { Type }` + `export { Type }` 在同一行
- 每次 commit 后观察 CI 结果，针对性修复

### 5a. Dev Agent 残留 TS 错误模式

dev agent 达到 max_iterations 后常留下以下错误，需手动修复：

```
# Settings.tsx 残留已删除文件的导入
error TS2307: Cannot find module './NewsSettings'
error TS2307: Cannot find module './WeatherSettings'
# 解决：删除这些 import 语句和 JSX 中的组件调用

# MUI 重复导入
error TS2300: Duplicate identifier 'Stack'
# 解决：检查 import { Stack, ... } from '@mui/material'，删除重复项

# 未使用的变量
error TS6133: 'X is declared but never used'
# 解决：用 grep -n "X" 定位，删除该变量声明

# 类型转换错误（Window cast）
error TS2352: Conversion of type 'Window & typeof globalThis' to type 'Record<string, unknown>'
# 解决：先 cast 到 unknown，再 cast 到目标类型
```

**标准修复流程**：
```bash
npm run build 2>&1 | grep "error TS"   # 列出所有 TS 错误
# 逐一修复每个错误
git add -A && git commit -m "fix: resolve TS errors" && git push
```

### 5b. 循环依赖：事件发射器模式

**问题场景**：Service A（memoryStorage）需要在状态变更时 emit 事件给 Service B（WebhookService）。如果 B 依赖 A，A 又导入 B，就形成循环依赖。

**错误模式**：
```typescript
// WebhookService.ts — 错误：导入 PluginService（不存在或不是事件总线）
import { PluginService } from '../plugin/PluginService';
const unsub = PluginService.on(event, handler); // TS2304: PluginService has no .on()

// memoryStorage.ts — 错误：导入 WebhookService
import { WebhookService } from '../webhook/WebhookService'; // 循环！
```

**正确模式**：在 WebhookService 内联一个简单事件发射器并导出：
```typescript
// WebhookService.ts 内：
const memoryEventEmitter = new (class {
  private handlers: Map<string, Array<(data: unknown) => void>[]> = new Map();
  on(event: string, handler: (data: unknown) => void): () => void { ... }
  emit(event: string, data: unknown): void { ... }
})();
export { memoryEventEmitter as memoryEvents };

// memoryStorage.ts：
import { memoryEvents } from '../webhook/WebhookService';
// 在 addMemory/updateMemory/getMemory 成功后：
memoryEvents.emit('memory:created', fullEntry);
```

**原理**：WebhookService 是事件消费者（订阅方），memoryStorage 是事件生产者（发布方）。生产者不应导入消费者。通过在内联类中创建发射器并导出，打破了循环依赖。

### 5c. PluginService 不是事件总线

`PluginService`（`src/services/plugin/PluginService.ts`）只有 `listPlugins()`、`getPlugin()`、`registerPlugin()` 方法，**没有** `.on()`/`.emit()` 事件系统。如果需要在服务间发送事件，必须自己实现事件发射器（见 5b）。

## 常见问题

### 6. 验收

GitHub Actions success 后，访问 https://YeLuo45.github.io/pixel-pal-web 验证。

### 7. 更新 proposal-index.md

- `Current Status: delivered`
- `Stage: V{N} 已交付`

## 常见问题

### 本地 npm install 失败（rolldown binding）

症状：`npm install` 报错 rolldown binding 或 ENOTEMPTY。

**解决**：
```bash
rm -rf node_modules package-lock.json
npm install
```

如果网络阻塞导致无法下载，使用 GitHub Actions CI 验证。

### 本地 npm run build 失败

症状：各种 ESM/CJS 模块错误。

**解决**：直接 push 到 GitHub，依赖 CI 构建。CI 通常比本地更稳定。

### TS 错误 "X is defined but never used"

**可能误判的情况**：
- 变量用于 `switch` case 的 urgency 分支：`dueToday = isToday(dueDate)` 用于 "today" 优先级判断
- 类型重导出需要同时 `import { Type }` 和 `export { Type }` 在同一行

### package-lock.json 被误删

如果 `package-lock.json` 被删除，先从 git 恢复：
```bash
git checkout HEAD -- package-lock.json
npm install
```

### GitHub Actions 失败

使用 `gh run view <id>` 查看构建日志，针对性修复 TS 错误。

### Git Push 超时

等待几秒后重试，或使用 `gh run list` 检查 CI 状态确认代码已 push 成功。

## 版本历史

| 版本 | 提案 | 主要内容 |
|------|------|----------|
| V1 | P-20250420-002 | 基础 AI 对话 + 文档解析 + 笔记/任务/日历/邮件 |
| V2 | P-20260503-025 | 记忆持久化（IndexedDB）+ 5 种人格 + 8 种情绪 |
| V3 | P-20260503-026 | 主动动作系统（提醒/庆祝/问候/建议/记忆召回） |
| V4 | P-20260503-029 | AI 思考过程可视化 + Thinking Panel + Model Status Indicator |
| V5 | P-20260503-030 | Multi-Persona 协作系统 + Team 管理 + 角色分工 |
| V6 | P-20260503-031 | RAG 知识库 + 文档上传/BM25 分块 + IndexedDB 持久化 |
| V7 | P-20260503-032 | TTS 语音输出 + ASR 语音输入 + VoiceSettings 面板 |
| V8 | P-20260503-033 | Mobile PWA：Hamburger 抽屉导航 + 响应式布局 + PWA manifest |
| V10 | P-20260503-035 | 高级记忆系统（实体图谱+智能检索+评分） |
| V12 | P-20260503-037 | Multi-Language i18n（中英文切换，i18next+react-i18next） |
| V13 | P-20260504-001 | 高级记忆 v2（搜索过滤+评分算法+导出导入+Timeline+词云） |
| V14 | P-20260504-002 | Webhook+插件生态（WebhookService+天气+新闻订阅+PluginHub安装卸载） |

## 下一步迭代方向

- Desktop Electron（Electron 打包 Windows exe）
- 移动端适配（响应式 Sidebar）
- 数据分析面板（交互频率热力图+情绪趋势）

## Subagent 交付验收

subagent 达到 max_iterations 后停在"未提交"状态。需检查 git status 并手动修复：

```bash
cd /home/hermes/.hermes/proposals/workspace-dev/proposals/pixel-pal-web
git status  # 检查 untracked/modified 文件
git diff <file> | head -30  # 审查改动
git add <files> && git commit -m "<message>" && git push
```

常见 subagent 未完成项：
- package.json 未更新（新依赖未添加）
- 部分文件未保存到磁盘

## Vite-plugin-pwa 安装失败处理

添加 `vite-plugin-pwa` 到依赖时，npm install 可能因 peer dependency 冲突失败：

```
npm error @vitejs/plugin-react@x.x.x conflicts with required react@19.x.x
```

**解决**：改用手动 PWA 方案，不依赖 vite-plugin-pwa：
1. 在 `public/manifest.json` 手动编写 PWA manifest
2. 在 `index.html` 手动添加 PWA meta 标签
3. 在 `vite.config.ts` 中移除 `VitePWA` 插件导入和使用
4. 创建 `public/icon-192.png` 和 `public/icon-512.png` PWA 图标

这样可以绕过 npm 依赖冲突，同时实现 PWA 功能。

## 下一步迭代方向

- Desktop Electron（Electron 打包 Windows exe）
- 移动端适配（响应式 Sidebar）
- 数据分析面板（交互频率热力图+情绪趋势）

