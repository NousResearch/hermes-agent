# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changes

- **fix(tui): clear input buffer and restore cursor after external-editor submit** — After submitting a prompt via `Cmd/Ctrl+G`, the submitted text no longer reappears in the input field and the cursor is no longer hidden. `openEditor` now calls `clearIn()` + `dispatchRef` (pointing at `dispatchSubmission`) instead of `submitRef`, eliminating the stale-closure `inputBuf` double-prepend that caused the text to reappear. `withInkSuspended` now calls `resetTerminalFocusState()` in its `finally` block so the cursor is shown immediately on the next repaint even if a focus-lost event fired while the editor owned the TTY. (Fixes [#20640](https://github.com/NousResearch/hermes-agent/issues/20640))

- **fix(agent): add `platform_hint` parameter to `AIAgent.__init__` for custom platform descriptions** — External clients such as hermes-webui can now pass `platform_hint: str` to `AIAgent.__init__`. When provided, it takes priority over the static `PLATFORM_HINTS` dict lookup and the plugin registry fallback, allowing clients with unregistered platform keys (e.g. `"webui"`) to supply accurate channel context to the agent. Existing callers are unaffected. (Fixes [#20637](https://github.com/NousResearch/hermes-agent/issues/20637))
