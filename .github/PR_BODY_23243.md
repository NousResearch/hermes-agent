## What does this PR do?

Hermes currently has several UI surfaces that assume English: the Ink TUI, gateway-driven TUI status updates, slash-command output, and the Web Dashboard. This PR adds a typed i18n foundation for those surfaces, registers 16 supported locale codes, and ships complete Simplified Chinese (`zh`) coverage for the user-facing strings touched by this work.

This is not meant to be a one-off Chinese patch. The PR establishes the extension points that future translators need: typed language packs, locale normalization, runtime language resolution, Dashboard language switching, and config-backed propagation through `display.language`.

## Related Issue

Fixes #23224

## Type of Change

- [x] ✨ New feature (non-breaking change that adds functionality)
- [x] ✅ Tests (adding or improving test coverage)

## Changes Made

- Added a TUI i18n layer in `ui-tui/src/i18n/` with a typed English source catalog, a complete `zh` pack, 16 locale codes, locale aliases, fallback behavior, interpolation, status translation, tool verbs, and language-pack-driven verb layout rules.
- Wired language resolution from Python/config into the TUI path: `agent/i18n.py`, `tui_gateway/server.py`, `tui_gateway/ws.py`, `uiStore`, `createGatewayEventHandler`, and `useConfigSync`.
- Replaced hardcoded TUI user-facing strings across slash commands, status chrome, prompts, dialogs, tool labels, model picker, setup copy, and related components with i18n keys.
- Added Web Dashboard i18n support for the reviewed Dashboard surfaces, including the language switcher, config field labels via `schemaZh.ts`, plugin nav `labelKey` handling, OAuth/model/tool-call surfaces, and the new Channels page that landed from upstream.
- Kept non-Chinese locale packs registered as typed shells/fallback packs so future translation work can fill them without changing the framework again.
- Added test coverage for locale normalization, catalog parity, fallback behavior, interpolation, toolset labels, gateway language payloads, and Dashboard/TUI build safety.

## How to Test

### Manual verification

1. Set `display.language: zh` in `~/.hermes/config.yaml`, start `hermes --tui`, and verify TUI chrome, slash-command output, status labels, dialogs, and prompts render in Chinese.
2. Set `display.language: en`, restart/open the TUI again, and verify the same surfaces return to English.
3. Open the Web Dashboard with `display.language: zh`; verify the sidebar, config labels, language picker, plugin nav labels, and Channels page labels/buttons/status badges render from the i18n catalog.
4. Use the Dashboard language switcher and verify the chosen locale persists locally and syncs back to `display.language` for embedded TUI sessions.
5. Try unsupported locale values such as `xx` and verify the UI falls back to English rather than crashing or rendering blanks.

### Automated verification run for this PR

- `cd ui-tui && npm run type-check` — passed.
- `cd ui-tui && npm run build` — passed.
- `cd ui-tui && npm run test` — passed: 88 test files, 974 tests passed, 1 skipped.
- `cd web && npm run build` — passed.
- `cd web && npx eslint src/components/LanguageSwitcher.tsx src/pages/ChannelsPage.tsx src/i18n/*.ts` — passed.
- `python -m py_compile agent/i18n.py tui_gateway/ws.py tui_gateway/server.py tui_gateway/entry.py hermes_cli/web_server.py` — passed.
- `python -m pytest tests/agent/test_i18n.py tests/tui_gateway/test_wait_for_mcp_discovery.py tests/test_tui_gateway_server.py -q` — passed: 243 tests passed, 1 warning.
- `git diff --check` — passed.

### Full Python suite note

- `python -m pytest tests/ -q -x` was attempted. It stops at `tests/acp/test_approval_isolation.py::TestAcpExecAskGate::test_interactive_env_var_routes_to_callback`.
- The same ACP directory-order failure reproduces on current `origin/main` (`pytest tests/acp -q` -> 1 failed, 274 passed), while the specific test file passes alone (`pytest tests/acp/test_approval_isolation.py -q` -> 7 passed). Because this is an upstream order-dependent ACP test issue and not introduced by this i18n PR, the full-suite checklist item below is intentionally not checked.

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix(scope):`, `feat(scope):`, etc.)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [ ] I've run `pytest tests/ -q` and all tests pass
- [x] I've added tests for my changes (Python i18n tests, gateway language tests, and TUI Vitest coverage)
- [x] I've tested on my platform: Ubuntu/WSL2 on Windows 11

### Documentation & Housekeeping

- [x] I've updated relevant documentation (README, `docs/`, docstrings) — or N/A
- [x] I've updated `cli-config.yaml.example` if I added/changed config keys — or N/A (`display.language` already exists)
- [x] I've updated `CONTRIBUTING.md` or `AGENTS.md` if I changed architecture or workflows — or N/A
- [x] I've considered cross-platform impact (Windows, macOS) per the [compatibility guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#cross-platform-compatibility) — or N/A (string/catalog/config propagation only)
- [x] I've updated tool descriptions/schemas if I changed tool behavior — or N/A

## For New Skills

N/A — this PR does not add a skill.

## Screenshots / Logs

No screenshot is attached in this body. The verification commands above cover the typed catalogs, TUI build/test path, Dashboard build path, and gateway language payload path.

---

## 这个 PR 做了什么？

Hermes 目前有多处 UI 默认假设英文：Ink TUI、gateway 驱动的 TUI 状态更新、slash 命令输出，以及 Web Dashboard。这个 PR 为这些表面建立一套类型安全的国际化基础，注册 16 个受支持 locale，并为本次覆盖到的用户可见文案提供完整简体中文（`zh`）翻译。

这不是一次性的中文补丁。这个 PR 建的是以后翻译者能继续扩展的入口：类型化语言包、locale 归一化、运行时语言解析、Dashboard 语言切换，以及通过 `display.language` 在配置和 UI 之间传播语言选择。

## 关联 Issue

Fixes #23224

## 变更类型

- [x] ✨ 新功能（非破坏性新增能力）
- [x] ✅ 测试（新增或改进测试覆盖）

## 具体变更

- 在 `ui-tui/src/i18n/` 新增 TUI i18n 层：类型化英文源目录、完整 `zh` 包、16 个 locale、locale 别名、fallback、插值、状态翻译、工具动词，以及由语言包声明的动词布局规则。
- 打通 Python/config 到 TUI 的语言链路：`agent/i18n.py`、`tui_gateway/server.py`、`tui_gateway/ws.py`、`uiStore`、`createGatewayEventHandler` 和 `useConfigSync`。
- 将 slash 命令、状态栏、提示词、弹窗、工具标签、模型选择器、setup 文案等 TUI 用户可见硬编码文案替换为 i18n key。
- 为 Web Dashboard 接入本次审查范围内的 i18n 支持，包括语言切换器、`schemaZh.ts` 配置字段标签、插件导航 `labelKey`、OAuth/model/tool-call 相关表面，以及上游新合入的 Channels 页面。
- 其他非中文 locale 作为类型化 shell/fallback 包注册，后续翻译者可以直接填充语言包，不需要再改框架。
- 新增测试覆盖 locale 归一化、目录完整性、fallback、插值、toolset 标签、gateway language payload，以及 Dashboard/TUI 构建安全性。

## 如何测试

### 手工验证

1. 在 `~/.hermes/config.yaml` 设置 `display.language: zh`，启动 `hermes --tui`，确认 TUI chrome、slash 命令输出、状态标签、弹窗和提示词显示中文。
2. 设置 `display.language: en`，重新打开 TUI，确认同一批表面恢复英文。
3. 用 `display.language: zh` 打开 Web Dashboard，确认侧边栏、配置字段标签、语言切换器、插件导航标签和 Channels 页面按钮/状态 badge 都来自 i18n 目录。
4. 使用 Dashboard 语言切换器，确认选择会在本地持久化，并同步回 `display.language`，供嵌入式 TUI 会话读取。
5. 尝试 `xx` 这类不支持的 locale，确认 UI 回退英文，不崩溃、不空白。

### 本 PR 已跑自动化验证

- `cd ui-tui && npm run type-check` — 通过。
- `cd ui-tui && npm run build` — 通过。
- `cd ui-tui && npm run test` — 通过：88 个测试文件，974 个测试通过，1 个 skipped。
- `cd web && npm run build` — 通过。
- `cd web && npx eslint src/components/LanguageSwitcher.tsx src/pages/ChannelsPage.tsx src/i18n/*.ts` — 通过。
- `python -m py_compile agent/i18n.py tui_gateway/ws.py tui_gateway/server.py tui_gateway/entry.py hermes_cli/web_server.py` — 通过。
- `python -m pytest tests/agent/test_i18n.py tests/tui_gateway/test_wait_for_mcp_discovery.py tests/test_tui_gateway_server.py -q` — 通过：243 个测试通过，1 个 warning。
- `git diff --check` — 通过。

### Python 全量测试说明

- 已尝试 `python -m pytest tests/ -q -x`，会停在 `tests/acp/test_approval_isolation.py::TestAcpExecAskGate::test_interactive_env_var_routes_to_callback`。
- 同一个 ACP 目录顺序失败在当前 `origin/main` 也能复现（`pytest tests/acp -q` -> 1 failed, 274 passed），而该测试文件单独运行通过（`pytest tests/acp/test_approval_isolation.py -q` -> 7 passed）。这是上游当前已有的 ACP 顺序依赖问题，不是本 i18n PR 引入，所以官方全量 pytest checklist 项保持未勾选。

## Checklist

### Code

- [x] 已阅读 [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] Commit 信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)（`fix(scope):`、`feat(scope):` 等）
- [x] 已搜索 [existing PRs](https://github.com/NousResearch/hermes-agent/pulls)，确认不是重复 PR
- [x] 本 PR 只包含与此修复/功能相关的变更
- [ ] 已运行 `pytest tests/ -q` 且全部测试通过
- [x] 已为本次变更新增测试（Python i18n、gateway language，以及 TUI Vitest 覆盖）
- [x] 已在本机平台测试：Windows 11 上的 Ubuntu/WSL2

### Documentation & Housekeeping

- [x] 已更新相关文档（README、`docs/`、docstrings）— 或不适用
- [x] 如果新增/修改 config key，已更新 `cli-config.yaml.example` — 或不适用（`display.language` 已存在）
- [x] 如果改变架构或工作流，已更新 `CONTRIBUTING.md` 或 `AGENTS.md` — 或不适用
- [x] 已按兼容性指南考虑跨平台影响（Windows、macOS）— 或不适用（仅字符串/目录/config 传播）
- [x] 如果改变工具行为，已更新工具描述/schema — 或不适用

## For New Skills

不适用 — 本 PR 没有新增 skill。

## Screenshots / Logs

正文中不附截图。上面的验证命令覆盖了类型化目录、TUI 构建/测试路径、Dashboard 构建路径，以及 gateway language payload 路径。
