# Contributions Log

This file tracks open-source contributions made to the Hermes Agent project.

## 2026-04-23

### PR: feat(cli): show current session title in status bar
- **Branch:** `feat/session-title-status-bar`
- **Issue:** #14859
- **Description:** Adds session title display to the TUI/CLI status bar for wide terminals. Introduces `display.statusbar.show_session_title` and `display.statusbar.session_title_max_len` config options.
- **Status:** Ready for PR

### PR: feat(models): add deepseek-v4-pro to DeepSeek provider catalog
- **Branch:** `feat/deepseek-v4-pro-model`
- **Issue:** #14902
- **Description:** Adds `deepseek-v4-pro` to the DeepSeek provider model list and normalization logic.
- **Status:** Ready for PR

### PR: fix(cli): open dashboard browser in WSL environments
- **Branch:** `fix/wsl-dashboard-browser-open`
- **Issue:** #14897
- **Description:** Fixes dashboard browser opening in WSL by trying `cmd.exe` and `powershell.exe` before falling back to `webbrowser.open()`.
- **Status:** Ready for PR
