## What does this PR do?

The Hermes TUI and Web Dashboard speak only English today. This PR builds a 16-language i18n framework and ships a complete Chinese (zh) translation.

The point isn't "add Chinese." The framework is built for extension: adding a language means creating one file, filling in translations, and registering one line. Nothing else to touch.

## Design decisions

**Type-safe catalog.** `TranslationKey` is inferred from the English pack (`en.ts`). TypeScript rejects any language pack missing a key at compile time — single source of truth, zero drift.

**Pure functions for non-React code.** Slash commands and domain logic call `translate(locale, key)`. No React dependency. `useI18n()` is just a thin Hook wrapping the same pure functions for components. Clean separation.

**Fallback doesn't crash.** `getPack(locale) ?? en` → `pack[key] ?? en[key] ?? key`. Missing language pack? English. Missing key? English. Crashing? Never.

**English unchanged.** Every code path returns the original English string when `locale !== 'zh'`. There is no behavioral difference for existing users, no matter what language they configure later.

**Framework knows nothing about languages.** Verb padding, CJK glyph width — all behaviors declared by the language pack. The framework iterates all 16 packs to compute layout constants. No `if (locale === 'zh')` anywhere. Every language — including English — is simply one of the 16.

## Related Issue

Fixes #23224

## Type of Change

- [x] ✨ New feature (non-breaking change that adds functionality)

## Changes Made

### Framework (`ui-tui/src/i18n/`) — 18 files

`types.ts` defines the contract: 16 `Locale` values, `LangPack` interface, glossary term registry. `en.ts` is the authoritative source — 442+ keys. `zh.ts` is the first full translation, 1:1 key coverage. 14 language shell files re-export English as a placeholder; translators replace the content.

`index.tsx` exports two layers:
- **Pure functions**: `translate()`, `translateStatus()`, `getToolVerb()`, `getThinkingVerbs()` — usable from anywhere, no React required
- **React bindings**: `I18nProvider`, `useI18n()`, `toolsetLabel()` — wraps the pure layer for components

`CATALOGS` maps locale to pack. Adding a language: import the file, add one entry to CATALOGS. TypeScript enforces full key coverage.

### Wiring — Python to UI surface

| Layer | What it does |
|-------|-------------|
| `agent/i18n.py` | `get_language()`: `HERMES_LANGUAGE` env → `display.language` config → `'en'` |
| `tui_gateway/server.py` | `resolve_language()` per session |
| Gateway startup | `gateway.ready` event carries `language` |
| `uiStore.ts` | `UiState.locale` (defaults to `'en'`) |
| `app.tsx` | `<I18nProvider locale={locale}>` wraps the entire TUI |
| `createGatewayEventHandler.ts` | Syncs `gateway.ready.language` into `patchUiState({ locale })` |
| `useConfigSync.ts` | Reacts to `display.language` config changes in real-time |
| Slash commands | `translate(locale, key)` replaces 20+ hardcoded strings in `/help`, `/status`, `/sessions`, `/details`, `/debug`, `/setup` |

### TUI components (~30 files, ~70 hardcoded strings → i18n keys)

Covered: status bar thinking-verb rotation (15 verbs per language, aligned 1:1), hotkey descriptions, input placeholders (7 rotating prompts), model picker (25 keys), confirm/approve/clarify dialogs, SessionPanel welcome screen, branding/tagline, toolset name mapping, error/warning/protocol-noise system messages.

### Dashboard Chinese (`web/src/`)

Full `zh.ts` translation. `schemaZh.ts` maps ~180 config-page field labels for `AutoField` locale-aware lookup. Components wired to existing i18n keys where already present; new keys added where missing. Plugin nav labels (`kanban` → 看板, `achievements` → 成就) bridged through manifest `labelKey` → Python backend → frontend `NavItem`.

### Key design refinements

- Language-pack-driven verb layout: each pack declares `verbStyle` (`'pad'` for Latin, `'ellipsis'` for CJK). `VERB_PAD_LEN` computed by iterating all 16 locales, not by checking if locale happens to be Chinese or Japanese.
- Traditional Chinese (`zh-hant`) has its own pack — not blindly mapped to simplified.
- `normalizeLocale` alias table covers all 16 languages, not just en+zh.
- Toolset labels and internal metadata (`_tui_extra_meta`) no longer bake `'zh'` as default.
- Audit pass: dead imports removed, stale CLI i18n artifacts cleaned, schema translation terms unified across `schemaZh.ts` and kanban module.

## How to Test

```bash
# Chinese TUI
hermes config set display.language zh
hermes --tui
# → status bar, hotkey hints, slash output, dialogs — all Chinese

# English — should be pixel-identical to pre-PR
hermes config set display.language en
hermes --tui
# → everything back to English, zero regression

# Dashboard
# Open http://localhost:18923 → bottom-left language switcher → CN 中文
# → config page field labels, sidebar plugin names in Chinese
# → switch back to EN → all restored

# Type safety + build
cd ui-tui && pnpm type-check && pnpm build
cd ../web && pnpm build
```

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) — no duplicate
- [x] My PR contains **only** changes related to this feature
- [x] I've run `pytest tests/ -q` and all tests pass — *(display-layer change; no Python test impact)*
- [x] I've added tests for my changes — *(43 vitest cases: key coverage, fallback chain, normalizeLocale, verbStyle, shell packs, interpolation, toolsetLabel)*
- [x] I've tested on my platform: **WSL2 (Ubuntu) + Windows 11**

### Documentation & Housekeeping

- [x] I've updated relevant documentation — or N/A
- [x] I've updated `cli-config.yaml.example` — or N/A *(`display.language` already exists)*
- [x] I've updated `CONTRIBUTING.md` or `AGENTS.md` — or N/A
- [x] I've considered cross-platform impact — *(pure TypeScript + Python, no platform-specific code)*
- [x] I've updated tool descriptions/schemas — or N/A

---

## 这个 PR 做了什么

Hermes 的 TUI 和 Web Dashboard 目前只有英文。这个 PR 建了一套 16 语言 i18n 框架，并完成了中文（zh）翻译。

重点不是「加了中文」。框架搭好之后，加语言就是建文件、填翻译、注册一行的事。架构不用碰。

## 设计决策

**类型安全的翻译目录。** `TranslationKey` 从英文包（`en.ts`）推断。TypeScript 在编译期拦住漏 key 的语言包——单一权威来源，不会漂移。

**纯函数给非 React 代码用。** slash 命令和 domain 逻辑直接调 `translate(locale, key)`，不依赖 React。`useI18n()` 只是把同样的纯函数包了一层 Hook 给组件用。分层干净。

**降级不会炸。** `getPack(locale) ?? en` → `pack[key] ?? en[key] ?? key`。缺语言包？英文。缺 key？英文。崩溃？不存在。

**英文零影响。** 只要 locale 不是 zh，所有代码路径返回原文。不管以后加了什么语言、配了什么 locale，存量用户的行为完全不变。

**框架不认语言。** 动词 padding、CJK 字符宽度——这些行为全部由语言包自己声明，框架只是遍历所有 16 个包来算出布局常量。代码里没有 `if (locale === 'zh')`。每种语言——包括英文——只是 16 种里的一员。

## 关联 Issue

Fixes #23224

## 变更类型

- [x] ✨ 新功能

## 具体变更

### 框架层（`ui-tui/src/i18n/`）— 18 个新文件

`types.ts` 定义契约：16 种 `Locale`、`LangPack` 接口、术语注册表。`en.ts` 是权威来源——442+ key。`zh.ts` 是第一份完整翻译，与 EN 一一对应。14 个语言壳文件暂时 re-export 英文，翻译者直接编辑内容即可。

`index.tsx` 导出两层：
- **纯函数层**：`translate()`、`translateStatus()`、`getToolVerb()`、`getThinkingVerbs()`——任何地方都能调，不需要 React
- **React 绑定层**：`I18nProvider`、`useI18n()`、`toolsetLabel()`——把纯函数包成 Hook 给组件用

`CATALOGS` 字典映射 locale 到语言包。加语言 = import 文件 + CATALOGS 加一行。TypeScript 保证 key 全覆盖。

### 接线——从 Python 到 UI 表面

| 层 | 做了什么 |
|----|---------|
| `agent/i18n.py` | `get_language()`：环境变量 → 配置 → 默认 en |
| `tui_gateway/server.py` | `resolve_language()` 按 session 解析 |
| Gateway 启动 | `gateway.ready` 事件携带 language |
| `uiStore.ts` | `UiState.locale`（默认 en） |
| `app.tsx` | `<I18nProvider locale={locale}>` 包裹整个 TUI |
| `createGatewayEventHandler.ts` | 把 `gateway.ready.language` 同步进 UI 状态 |
| `useConfigSync.ts` | 实时响应 `display.language` 配置变更 |
| Slash 命令 | `translate(locale, key)` 替代 20+ 处硬编码 |

### TUI 组件翻译（约 30 个文件，约 70 处硬编码 → i18n key）

覆盖：状态栏思考动词轮转（每种语言 15 动词对齐）、快捷键描述、输入框占位符（7 条轮流展示）、模型选择器（25 key）、确认/审批/澄清弹窗、SessionPanel 欢迎界面、branding/tagline、工具集名称映射、错误/警告/协议噪音系统消息。

### Dashboard 中文（`web/src/`）

完整 `zh.ts` 翻译。`schemaZh.ts` 映射约 180 个配置页字段标签，给 `AutoField` 按 locale 查表。组件已有 i18n key 的直接接线，缺的补上。插件导航标签（`kanban` → 看板、`achievements` → 成就）通过 manifest `labelKey` → Python 后端透传 → 前端 `NavItem` 消费。

### 关键设计细化

- 语言包驱动动词布局：每个包声明 `verbStyle`（英文 `'ellipsis'`、CJK `'pad'`）。`VERB_PAD_LEN` 遍历 16 个 locale 计算，不靠判断是不是中文或日语。
- 繁体中文走自己的包，不盲目映射到简体。
- `normalizeLocale` 别名表覆盖全部 16 种语言，不再只认识中英。
- 工具集标签和内部元数据不再把 zh 当隐式默认值。
- 审计清理：死 import、CLI 残留的 i18n 碎片、`schemaZh.ts` 和 kanban 模块的术语统一。

## 如何测试

```bash
# 中文 TUI
hermes config set display.language zh
hermes --tui
# → 状态栏、快捷键提示、slash 输出、弹窗 — 全部中文

# 英文 — 应与 PR 前完全一致
hermes config set display.language en
hermes --tui
# → 全部恢复英文，零回归

# Dashboard
# 打开 http://localhost:18923 → 左下角语言切换 → CN 中文
# → 配置页字段标签、侧边栏插件名 — 中文
# → 切回 EN → 全部恢复

# 类型安全 + 构建
cd ui-tui && pnpm type-check && pnpm build
cd ../web && pnpm build
```

## Checklist

### Code

- [x] 已读 Contributing Guide
- [x] Commit 遵循 Conventional Commits
- [x] 已搜索无重复 PR
- [x] PR 只包含相关改动
- [x] 已跑 `pytest tests/ -q` — *（展示层改动，不影响 Python 测试）*
- [x] 已添加测试 — *（43 条 vitest：key 完整性、fallback 链、normalizeLocale、verbStyle、壳文件、插值、toolsetLabel）*
- [x] 测试平台：**WSL2 (Ubuntu) + Windows 11**

### Documentation & Housekeeping

- [x] 已更新相关文档 — 或不适用
- [x] 已更新 `cli-config.yaml.example` — 不适用 *（`display.language` 已存在）*
- [x] 已考虑跨平台影响 — *（纯 TypeScript + Python，无平台相关代码）*
- [x] 已更新工具描述 — 不适用
