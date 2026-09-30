# SamAgent Local Platform Bridge for VS Code

Synchronizes your local VS Code / Cursor / VSCodium workspace with your locally installed **SamAgent Platform** (`http://127.0.0.1:8080`).

## Features
1. **Verify-on-Save (`samagent.verifyOnSave`)**: Every time you edit and save a file in VS Code (`app/main.py`, `app/static/index.html`, etc.), SamAgent automatically re-runs its L0–L4 Verification Pyramid and OWASP Top-10 security probes locally.
2. **Pre-Production Status Bar Badge**: Shows live readiness (`$(pass-filled) SamAgent: Pre-Prod Ready` or `$(error) SamAgent: Pre-Prod Gate Blocked`) in the VS Code status bar.
3. **Embedded Mission Control Sidebar**: View your Spec Contract, Isolated Worktree Waves, Live App Preview, and Pre-Production Deployment Gate directly inside VS Code's Activity Bar.
4. **Zero-Config `.vscode/` Workspace**: Every SamAgent project includes `.vscode/tasks.json`, `.vscode/launch.json` (`F5` debug), and `project.code-workspace`.
