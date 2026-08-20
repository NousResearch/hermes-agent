---
name: large-react-app-panel-strip
description: 从臃肿的 React 应用中移除不需要的模块/面板，同时保持核心功能正常运行。适用于 pixel-pal-web 这类多面板架构的精简工作。
---

# Large React App — 精简面板/移除功能工作流

## 适用场景

从臃肿的 React 应用中移除不需要的模块、面板或功能，同时保持核心功能正常运行。典型特征：
- 应用有多面板/多 Tab 架构（Sidebar 导航）
- 需要快速剥离非核心功能（砍功能但留架构）
- 历史债务积累，需要重构但不想重写

## 工作流程

### 1. 确定入口文件

先读 `src/pages/MainPage.tsx` 和 `src/App.tsx`，看面板是如何被导入和渲染的。找到 `activePanel` 状态或直接的面板组件列表。

### 2. 分析 Sidebar

Sidebar 通常定义了所有面板的入口。列出所有 `NAV_ITEMS`，确认哪些要移除。对应的组件文件也要清理。

### 3. 逐文件清理

按依赖链从外到内清理：
- `MainPage.tsx` → 移除不需要的面板 import，render 中删除对应组件
- `Sidebar.tsx` → 移除 NAV_ITEM，清理图标导入，删除未使用的状态/副作用
- `App.tsx` → 移除相关全局状态、useEffect、import
- `store/index.ts` → 确认废弃状态是否还在被使用

### 4. 语法问题排查

大规模删除后容易引入语法错误，重点检查：
- JSX 标签不匹配（多余的 `}` 或少了 `</Box>`）
- 字符串缺少引号（如 `'italic` 而非 `'italic'`）
- return 语句结构问题（用 `<>` Fragment 包裹解决）

### 5. 验证

```bash
./node_modules/.bin/tsc --noEmit
```
如果 TSC 通过但 `npm run build` 超时，直接用 tsc 结果，不等完整 build。

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `fontStyle: 'italic}` 报错 | 字符串缺闭合引号 | 修复字符串字面量 |
| return 解析报错 | JSX 结构不完整 | 用 `<>` Fragment 包裹整个 return |
| 删除后 build 报错但 tsc 不报 | 运行时 import | 确认所有 import 链被清理 |
| 状态还在被其他文件引用 | 跨文件共享状态 | 保留状态（只是不显示在 UI） |

## 验证清单

- [ ] MainPage 只渲染需要的面板
- [ ] Sidebar NAV_ITEM 与实际渲染一致
- [ ] App.tsx 没有废弃的 useEffect / import
- [ ] `tsc --noEmit` 通过
- [ ] 核心功能（聊天）仍正常工作
