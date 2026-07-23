# PRD: SessionDragSwitcher + ARD (Agentic Resource Discovery)

**Status:** Implemented  
**Branch:** `feat/ard-session-drag-switcher`  
**Author:** Hermes Agent  
**Date:** 2026-07-05

---

## 1. Problem Statement

### 1.1 SessionDragSwitcher

Hermes Desktop 用户在多个 session 之间切换只能通过侧边栏点击，缺乏快速、符合直觉的导航方式。浏览器式的手势导航（鼠标横向拖拽切换标签页）已被大量用户习惯，Hermes 应提供类似体验。

### 1.2 ARD (Agentic Resource Discovery)

当前 Hermes 的 AI 能力扩展完全依赖预配置——MCP server 需手动编辑 `config.yaml`，skill 需手动放入目录。Agent 无法在运行时自主发现并安装新能力，这限制了 Agent 的自主性和可扩展性。

ARD 是 Microsoft、Google、GoDaddy、Hugging Face 联合提出的开放协议，允许 AI Agent 在运行时搜索、发现、安装新的 AI 能力（skills、MCP servers、Spaces），无需人工预配置。

---

## 2. Solution

### 2.1 SessionDragSwitcher

在聊天区域添加鼠标横向拖拽手势：按住鼠标左键拖拽左/右超过 80px 阈值，显示方向性覆盖层（目标 session 名称），松手即导航到目标 session。

**设计决策：**
- 手势限制为**纯横向**（|dx| > 2 × |dy|），避免与纵向滚动冲突
- 忽略输入框/按钮/链接区域的拖拽，防止误触
- 循环切换（到列表末尾后回到开头）
- 加载最近 50 个 session，30s 缓存避免频繁刷新
- 覆盖层透明度随拖拽距离渐进增强（30%-100%）

### 2.2 ARD Discovery Tool

提供三个工具供 Agent 运行时使用：

| 工具 | 功能 |
|------|------|
| `ard_search` | 搜索 AI 能力（skills/MCP servers/Spaces），支持自然语言查询 |
| `ard_install_mcp` | 一键安装搜索结果中的 MCP server 到 Hermes |
| `ard_catalog` | 浏览可用的联合注册中心 |

**实现细节：**
- 后端调用 Hugging Face Discover API (`huggingface-hf-discover.hf.space/search`)
- 支持联合注册中心（federation）：一次搜索可发现其他注册中心的能力
- 懒加载 `httpx`，不影响冷启动
- `ard_install_mcp` 直接调用 `hermes mcp add` CLI

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────┐
│  Hermes Desktop Chat View                            │
│  ┌────────────────────────────────────────────────┐  │
│  │  SessionDragSwitcher (new)                     │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │  ChatRuntimeBoundary (existing)          │  │  │
│  │  │  - Messages, Composer, etc.              │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  │  > Directional overlay (left/right)             │  │
│  │  > Session list (react-query, 50 recent)        │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  ARD Tool (tools/ard_discovery.py)                   │
│                                                      │
│  ard_search(query, kind, page_size, registry_url)    │
│    └─► POST huggingface-hf-discover.hf.space/search  │
│                                                      │
│  ard_install_mcp(mcp_url, name)                      │
│    └─► hermes mcp add <name> --url <mcp_url>        │
│                                                      │
│  ard_catalog(registry_url)                           │
│    └─► GET .well-known/ai-catalog.json               │
└──────────────────────────────────────────────────────┘
```

---

## 4. Files Changed

| File | Change | Lines |
|------|--------|-------|
| `apps/desktop/src/app/chat/session-drag-switcher.tsx` | **New** — Drag gesture component | +194 |
| `apps/desktop/src/app/chat/index.tsx` | Wrap ChatRuntimeBoundary with SessionDragSwitcher | +5/-1 |
| `tools/ard_discovery.py` | **New** — ARD search/install/catalog tools | +392 |
| `toolsets.py` | Register `ard` toolset + core tools | +8 |

---

## 5. Acceptance Criteria

### SessionDragSwitcher
- [x] 鼠标横向拖拽 >80px 显示覆盖层
- [x] 覆盖层显示目标 session 名称和方向箭头
- [x] 松手导航到目标 session
- [x] 拖拽不足阈值时无操作
- [x] 纵向拖拽被忽略（不触发切换）
- [x] 输入框/按钮区域拖拽不触发
- [x] 有文本选中时不触发
- [x] 循环切换（从末尾回到开头）
- [x] 全局 mouseup 清理状态

### ARD
- [x] `ard_search("transcribe audio")` 返回相关 MCP servers
- [x] `ard_search(kind="mcp-server")` 仅返回 MCP 类型
- [x] `ard_install_mcp(url, name)` 调用 hermes CLI 安装
- [x] `ard_catalog()` 显示已知注册中心
- [x] 网络错误时返回友好 JSON 错误（不抛异常）
- [x] 作为 `ard` toolset 可被启用/禁用

---

## 6. Future Work

- ARD: 支持 skill 安装（不仅是 MCP server）
- ARD: 缓存搜索结果，减少重复请求
- SessionDragSwitcher: 触控板手势支持
- SessionDragSwitcher: 可配置手势灵敏度
