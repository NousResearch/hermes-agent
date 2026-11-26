---
name: pixel-pal-web-iteration-workflow
description: PixelPal V21+ 人格系统高速迭代工作流 — PRD起草→dev委托→手动收尾→commit push→报告。适用于 V21-V36+ 人格系统迭代。
category: game-development
---

# PixelPal V21+ 人格系统高速迭代工作流

## 概述

PixelPal 从 V21 开始进入"人格系统"迭代阶段（V21-V33+），采用极简高速迭代模式：boss 选方向 → 小墨起草 PRD → 委托 dev agent → 检查 agent 是否完成 commit/push → 未完成则手动收尾 → 报告并呈上下一步选项。

**当前迭代版本**: V49 情感曲线报告（7/30/90天切换 + 趋势分析 + PDF导出）
**分支链**: master ← [持续迭代中]

## V32 i18n 经验总结

PixelPal V32 完成 i18n 化，以下是关键教训：

### 常见 patch 坑点

**1. `t` 变量名与 `.map(t =>)` 冲突**
- 在组件函数内使用 `const { t } = useTranslation()` 后，如果同一个组件有 `.map((t) => ...)`，两个 `t` 会冲突
- 解决：`.map()` 的回调参数改名（如 `(trigger)` 或 `(item)`），翻译函数用 `{ t: tt }` 重命名
- 或者 `.map()` 的回调用其他变量名，让 `t` 专用于 `useTranslation`

**2. TriggerConfig Select label prop 位置错误**
- patch 替换 `label="重复"` 时，如果 `onChange` 在同一行且包含 `}`，容易把 `label=` 放到 `onChange` 闭包内部
- 正确结构：`<Select value={...} label={t(...)} onChange={...} >`（label 和 onChange 是平级属性）
- 验证：patch 后检查第 56-67 行是否有多余的 `label=` 或孤立的 `}`

**3. TextField helperText 被错误替换**
- 如果原来 `<TextField helperText="中文" />` 在 patch 时匹配了不完整的上下文，可能把 `helperText=` 那一行变成单独的空标签
- 正确做法：确保 old_string 包含完整的 `<TextField ... helperText="..."` 行
- 验证：`grep -n helperText` 检查是否还作为独立标签存在

**4. 意外删除既有变量**
- patch 替换 `{editingScene ? '编辑场景' : '新建场景'}` 时，如果同时删除了中间的 `isValid` 判断，会导致保存按钮始终 enabled
- 解决：patch 前用 `git show HEAD:file | grep -n 'isValid\|name.trim'` 确认关键变量存在
- 验证：patch 后 grep 相关变量是否还在

**5. Select/InputLabel 重复**
- InputLabel 已经包了 `{t('...')}` 后，Select 上又有 `label="重复"` 会导致 MUI 警告
- 解决：保留 InputLabel 的 `t()` 包装，去掉 Select 的 `label=` 属性（或者两者都用 t()）

### build 验证

- 本地 `npm run build` 因既有的 TS 类型错误失败（`appStore` type mismatch、`FlexSearch` namespace、`personaId` on `never`），V31/V32 都如此
- CI 能通过说明用了不同的 TypeScript 配置（`tsconfig.app.json` vs `tsconfig.json`）
- **不要因为本地 build 失败就认为代码有问题**，以 CI 为准

### i18n 替换规则

- **要换**：用户可见 UI 字符串（JSX 标签、按钮文字、placeholder、dialog 标题、helperText、chip label）
- **不换**：AI prompt、`.map()` 回调参数、数据结构、console.log、internal error messages
- **fallback 写中文原文**：`t('key', '中文原文')` 确保 key 缺失时也有兜底
- **t() 命名冲突**：当组件内有 `.map((t) => ...)` 时，`useTranslation` 的 `t` 会与之冲突。解决：`const { t: tt } = useTranslation()` + `.map((item) => ...)` 或重命名 map 回调
- **VOICE_LABELS 静态化**：如果 `const VOICE_LABELS = [...]` 在组件外定义且用了 `t()`，会编译错误。解决：改用英文 key，运行时 `t('key')` 包装

### i18n patch 常见错误模式

1. **Select label prop 位置错误**：patch 时如果 `onChange` 闭包含 `}`，容易把 `label=` 错误地放进闭包里。验证：检查 `<Select` 行附近的 `}` 数量是否平衡
2. **helperText 被误拆**：TextField 的 `helperText="中文"` 替换时上下文不完整会变成空标签。确保 old_string 包含完整的 `helperText="..."` 行
3. **删除既有变量**：patch 按钮文字时不小心删掉了中间的 `disabled={isValid}` 等判断。patch 前用 `git show HEAD:file | grep -n 'isValid'` 确认变量存在
4. **Subagent max_iterations=50**：大批量文件处理时 subagent 会在中途停止（API 50次限制）。剩余文件需手动处理，不要依赖 subagent 完成全部
5. **本地 vite build TS 错误**：既有的 TS 类型错误（appStore type mismatch、FlexSearch namespace、personaId on never）会导致本地 build 失败，但 CI 能通过。以 CI 为准
6. **useTranslation hook 插入到 destructuring 参数内**：批量添加 hook 时，如果用正则替换 `export const X = ({` → `export const X = ({\n  const { t } = useTranslation();`，hook 会出现在参数解构内部。验证：grep `'{\s*const\s+\{'` 检测，不要因为本地 build 失败就认为代码有问题

### 涉及 i18n 的文件（V32）

`RelationGraph`, `PersonaSelector`, `ActionToast`, `PersonaDetail`, `AgentPanel`, `MemoPanel`, `MultiPersonaCollaboration`, `Settings`, `TriggerConfig`, `SceneEditorDialog`, `SceneLogPanel`, `PresetScenesModal`, `SceneCard`, `QuickSceneBar`, `ScenesPage`, `ActionConfig`

## 两种迭代模式

### 人格功能迭代（创建分支）
- 版本号递增（V21, V22, ...）
- 创建新分支：`git checkout -b vXX-name`
- 完成后 PR merge 到 master
- 提案 ID 格式：`P-YYYYMMDD-VXX-001`

### 非人格功能迭代（直接 master）
- 例如 V23 移动端适配、V17 Agent 规划等
- 直接 commit 到 master（不是创建分支）
- PRD 位置：`/home/hermes/prj-proposals/PRJ-YYYYMMDD-NNN.md`
- 提案 ID 格式：`P-YYYYMMDD-VXX-XXX`（今天第几个）
- 提案状态直接 `in_dev`，不等 boss 确认

## 快速流程

### 每轮迭代（3-5分钟完成）

```
boss 选择方向 (A/B/C/D/E)
    ↓
起草 PRD → ~/.hermes/proposals/workspace-pm/proposals/P-YYYYMMDD-VXX-001-prd.md
    ↓
委托 dev agent (claude, max_iterations=30)
    ↓
等待 agent 完成
    ↓
检查 agent 是否已 commit + push
    ↓
未完成 → 手动: git checkout -b vXX-name → git add → git commit → git push
    ↓
报告 boss，提供下一步 5 个选项
```

## 提案 ID 格式

`P-YYYYMMDD-VXX-001`，例如 `P-20260505-V21-001`、`P-20260506-V28-001`。

## PRD 文件位置

两个位置都可以（按需选择）：
- `~/.hermes/proposals/workspace-pm/proposals/P-YYYYMMDD-VXX-001-prd.md` — 标准位置
- `/home/hermes/prj-proposals/PRJ-YYYYMMDD-NNN.md` — 快速草稿位置（适合非人格功能迭代）

注意：proposal-index.md 在 V22 后被重置为摘要索引，实际 PRD 数据在各文件中，**每轮迭代后仍需更新 proposal-index**（在对应 PRJ 分类下添加新条目，status: in_dev）。

## Dev Agent 委托模板

```typescript
delegate_task(
  acp_command: 'claude',
  context: `pixel-pal-web VXX — [功能名]
Baseline: vYY-name (commit HASH)
Repo: https://github.com/YeLuo45/pixel-pal-web
Working directory: /home/hermes/pixel-pal-web

V21-V(N-1) context:
- [关键状态/类型/组件说明]
- [重要文件路径]

Stay on scope.`,
  goal: `Implement VXX — [功能名] for pixel-pal-web

## Project
- Path: /home/hermes/pixel-pal-web
- GitHub: https://github.com/YeLuo45/pixel-pal-web
- Branch: create vXX-name FROM vYY-name (HASH)

## Requirements (from PRD)
[P0 功能列表]

## Implementation order:
[实现顺序]

## Files to create/modify:
[文件列表]

## Important:
- DO NOT touch [无关模块]
- Keep all V21-V(N-1) changes intact
- Commit each logical step
- When done: push branch

## Constraints:
- Build env Node 20 (needs 22+), local build fails but code correct`,
  max_iterations: 30
)
```

## Agent 未完成时的手动收尾流程

Agent 达到 max_iterations 后常停在"未 commit/push"状态。

```bash
# 1. 检查状态
cd /home/hermes/pixel-pal-web && git status --short && git log --oneline -3

# 2. 如果没有新分支，用当前 HEAD 创建
git checkout -b vXX-name

# 3. 如果已有分支在 HEAD
git checkout vXX-name

# 4. Add 所有修改的文件
git add src/.../ChangedFile.tsx

# 5. Commit
git commit -m "feat(VXX): [功能名] - [简短描述]"

# 6. Push
GIT_TERMINAL_PROMPT=0 git push origin vXX-name --quiet
```

## 协作面板组件列表（src/components/Collaboration/）

- CollabHistory.tsx
- CollaborationStatus.tsx
- CollaborationControls.tsx
- CollabHistoryDetail.tsx
- ResultSummary.tsx
- TaskBreakdown.tsx
- CollaborationChat.tsx
- DivisionView.tsx

## V46-V49 i18n 经验

### 文件级对象用 t() 的两种解决方式

**STATUS_CONFIG / ROLE_LABELS 问题**：这些对象定义在组件外，调用 `t()` 会编译错误。

**方式1（推荐）**：静态 key 映射 + 运行时 t() 包装
```typescript
// 文件级：只有英文 key，不含 t()
const ROLE_LABELS: Record<string, string> = {
  MemoryExpert: 'memoryExpert',
  EmotionAnalyst: 'emotionAnalyst',
  // ...
};
// JSX 渲染时：
t('collab.role.' + ROLE_LABELS[role])
```

**方式2**：移入组件内部
```typescript
const CollabHistoryItem: React.FC<...> = ({ entry, onDelete }) => {
  const { t } = useTranslation();
  const statusConfig = {
    completed: { color: '#4caf50', icon: CheckCircleIcon, label: t('collab.status.completed') },
    // ...
  };
};
```

### 批量替换策略

当替换 7 个组件的硬编码中文时，分三步：
1. **占位符替换**：中文 → `'PLACEHOLDER_KEY'`
2. **Hook 注入**：在每个组件函数体首行加 `const { t } = useTranslation();`
3. **占位符→t()**：所有 `'PLACEHOLDER_KEY'` → `t('collab.xxx')`

### Python 批量检测残留 CJK

```python
import re
def find_cjk_in_jsx(filepath):
    # 跳过已有 t() 的行
    # 找到不含 t() 但含 CJK 的行
    # 排除注释和 sx= 属性
```

## 版本历史

| 版本 | 改动 |
|------|------|
| V21-V33 | 人格系统迭代 |
| V41-V46 | 协作模式 → 历史 → 国际化 |
| V47 | 协作面板增强（分工视图 + 脉冲动画 + DivisionView） |
| V48 | Settings 角色图标配置（20 emoji 选项/角色） |
| V49 | 情感曲线报告（7/30/90天切换 + 趋势分析 + window.print() PDF导出） |

## 关键文件路径（V21-V49）

| 文件 | 版本 | 说明 |
|------|------|------|
| `src/store/index.ts` | V21+ | activePersonaId, setActivePersonaId, personaSystemPrompt, personaIntimacy, collabSession, appThemeMode... |
| `src/services/persona/personaStorage.ts` | V21+ | Persona 接口, CRUD, 4 preset 人格 |
| `src/services/persona/personaPrompt.ts` | V23+ | getPersonaSystemPrompt() + 亲密度注入 |
| `src/components/Layout/Sidebar.tsx` | V27+ | 可收拢侧边栏（V29 改为 floating button） |
| `src/pages/MainPage.tsx` | V27+ | 主布局，sidebarCollapsed local state |
| `src/components/ChatPanel/ChatPanel.tsx` | V28+ | 消息全宽（V28 去掉 maxWidth） |
| `src/components/Memory/MemoryPanel.tsx` | V30+ | 记忆面板全宽 |
| `src/components/Calendar/Calendar.tsx` | V30+ | 日历面板全宽 |
| `src/components/Tasks/Tasks.tsx` | V30+ | 任务面板全宽 |
| `src/components/Settings/Settings.tsx` | V30+ | Model 字段改为 TextField |

## Persona Type (V21+)

```typescript
interface Persona {
  id: string;
  name: string;
  avatar: string;  // emoji
  bio: string;
  voice: 'warm' | 'rational' | 'humorous' | 'serious';
  isDefault: boolean;
  createdAt: number;
  updatedAt: number;
  theme?: {
    primaryColor: string;
    secondaryColor: string;
    accentColor: string;
    backgroundColor: string;
    textColor: string;
  };
}
```

## 4 Preset 人格 ID

- `preset-friend` (warm, 😊)
- `preset-teacher` (rational, 📚)
- `preset-coach` (humorous, 💪)
- `preset-lover` (warm, 💕)

## Subagent 常见失败模式

### 1. 达到 max_iterations 但未 commit/push
最常见。检查：
```bash
git log --oneline -3 && git status --short
```
解决：手动 `git add → git commit → git push`

### 2. 修改了 store 但引用方未同步（V29 教训）
例如删了 `sidebarCollapsed` state，但 `MainPage.tsx` 还在用 `useStore((s) => s.sidebarCollapsed)`。
症状：TypeScript 编译错误或逻辑不工作。
解决：
```typescript
// 改前（从 store 读）
const sidebarCollapsed = useStore((s) => s.sidebarCollapsed);
const toggleSidebar = useStore((s) => s.toggleSidebar);

// 改后（用 local state）
const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
const toggleSidebar = () => setSidebarCollapsed(v => !v);
```

### 3. 意外删了 import
Agent patch 时可能删掉不该删的 import（如 `DocumentUpload`）。需手动补回。

### 4. 分支创建在旧 HEAD 上
Agent 创建分支时如果基于的 commit 不对，分支会落后。检查 `git log --oneline` 确认分支起点正确。

### 5. 意外删除既有变量/import
Patch 时如果 old_string 匹配范围不对，可能删掉不该删的变量（如 V29 删了 `DocumentUpload` import）。需手动补回。

## GitHub Pages 部署

- 地址: https://YeLuo45.github.io/pixel-pal-web
- 当前 gh-pages 是 V18 旧构建，需要 V21+ 合入 master 后 GitHub Actions 重新构建部署
- PR 链接: `https://github.com/YeLuo45/pixel-pal-web/pull/new/vXX-name`

## V27-V30 布局改造总结

| 版本 | 改动 |
|------|------|
| V27 | 侧边栏可收拢 160px↔52px，内容区全宽 |
| V28 | ChatPanel 消息区去掉 maxWidth 800px，全宽 |
| V29 | 收拢后 floating toggle button 替代内嵌收起按钮 |
| V30 | Model TextField 自由输入；Memory/Calendar/Tasks 全宽 |

## 部署地址

https://YeLuo45.github.io/pixel-pal-web
