# Hermes Code for VS Code/Cursor

Hermes Code is a thin VS Code/Cursor extension that gives Hermes Agent a Claude-like editor surface: a sidebar chat, editor/selection/diagnostics context, Git diff review, visible terminal helpers, and preview/apply flows for unified patches.

This extension intentionally keeps the Hermes brain outside VS Code. It launches the configured `hermes` command in the active workspace and delegates reasoning, tools, memory, skills, and provider routing to the existing Hermes runtime.

## Features

- **Hermes sidebar** in the Activity Bar.
- **Ask about selection** from the editor context menu.
- **Edit selection** with patch-first instructions.
- **Review current Git diff**.
- **Explain active file diagnostics**.
- **Preview unified diffs** returned by Hermes in VS Code diff editors.
- **Persistent workspace sessions** using the direct Hermes ACP backend for streaming updates/tool events, with legacy subprocess fallback.
- **Workspace instruction-file context** from `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules`, and `openspec/AGENTS.md`.
- **Quick Fix actions** on diagnostics: explain/fix with Hermes.
- **Status bar shortcut** and output-channel logging.
- **Context toggles** for active file, selection, diagnostics, Git diff, and instruction files.
- **Apply last patch** after `git apply --check` validation and explicit confirmation.
- **Run visible terminal commands** with context pasted as comments.
- **Mode picker** for Ask, Explain, Edit, Review, Test, Debug, Refactor, Security, and Commit-message workflows.
- **Markdown and code-block rendering** in chat, including copy buttons for assistant replies and fenced code blocks.
- **Collapsible tool-event cards** for streaming ACP activity.
- **Patch Review panel** with Preview, Apply, Copy, Discard, Run tests, and Revert actions after Hermes emits a unified diff.
- **Suggested starter prompts** in the empty state for diff review, file explanation, and test discovery.

## Requirements

- VS Code or a VS Code-compatible fork such as Cursor.
- Hermes installed and reachable on PATH:

```bash
hermes --version
```

If VS Code cannot find `hermes`, set `hermesCode.hermesCommand` to an absolute path.

## Useful commands

Open the Command Palette and run:

- `Hermes Code: Open Chat`
- `Hermes Code: Ask About Selection`
- `Hermes Code: Edit Selection`
- `Hermes Code: Review Current Git Diff`
- `Hermes Code: Explain Diagnostics`
- `Hermes Code: Preview Last Hermes Patch`
- `Hermes Code: Apply Last Hermes Patch`

## Patch workflow

Hermes Code asks Hermes to return edits as unified diffs. When a response contains a patch:

1. The patch is saved in extension global storage.
2. `Preview Last Hermes Patch` opens side-by-side diff editors for files the parser can materialize.
3. `Apply Last Hermes Patch` runs `git apply --check`, prompts for confirmation, then runs `git apply` in the workspace.

This is intentionally conservative: the extension previews first and applies only after validation and user confirmation.

## Current limitations

- The default backend is now a direct JSON-RPC ACP client that launches `hermes acp`, keeps one live ACP session per workspace, and renders streamed assistant/tool updates. Set `hermesCode.backend` to `subprocess` to use the older `hermes chat -q` path.
- Patch preview supports common unified diffs. Exotic rename/binary patches still fall back to opening the raw patch.
- Inline code completion is not included yet; agentic editor workflows are the first target.
