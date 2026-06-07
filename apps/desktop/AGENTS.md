# AGENTS.md — Hermes Desktop Feature Development

## Project Context
Hermes Desktop is an Electron + Vite + React 18 + TypeScript chat application that
communicates with the Hermes Python gateway via JSON-RPC over WebSocket. It uses
nanostores for state, shadcn/ui + Tailwind for UI, and a Pane-based layout system.

**Repo**: IsNoobgrammer/hermes-agent (fork of NousResearch/hermes-agent)
**App path**: `apps/desktop/`
**Backend**: `tui_gateway/` (Python, JSON-RPC via stdio/WebSocket)

## Current Phase: Feature Expansion
Adding two major features:
1. **Terminal Output Streaming** — Real-time streaming terminal output in chat
2. **Session Review Panel** — Overview tab + diff viewer for file changes

## Architecture

### Data Flow (Terminal Output)
```
Gateway (tool.start/tool.progress/tool.complete events)
  → Electron preload (window.hermesDesktop)
    → use-message-stream.ts (handleGatewayEvent)
      → chat-messages.ts (upsertToolPart)
        → tool-fallback-model.ts (buildToolView)
          → tool-fallback.tsx (ToolEntry renders output)
```

### Data Flow (File Diffs)
```
Gateway tool.complete event (inline_diff field)
  → store/tool-diffs.ts ($toolDiffs, $toolInlineDiffs)
    → tool-fallback.tsx (DiffLines inline rendering)

Gateway rollback.list / rollback.diff RPC
  → [NEW] session-diffs.ts store
    → [NEW] ReviewPanel component
```

### Layout System (Updated)
```
AppShell
  ├── Pane (chat-sidebar, left)        — session list
  ├── PaneMain (ChatView)              — chat thread
  ├── Pane (preview, right-rail)       — file preview (EDITABLE) + webview
  └── Pane (right-sidebar)             — 4 tabs: Overview, Review, Files, Terminal
```

Right sidebar tab order: Overview → Review → File system → Terminal.
Overview is the default tab when sidebar opens.

## Tech Conventions
- **State**: nanostores (`atom`, `computed`, `action`)
- **Components**: Functional React, no class components
- **Styling**: Tailwind CSS, shadcn/ui primitives in `components/ui/`
- **Types**: Strict TypeScript, interfaces in `types/` or inline
- **IPC**: `window.hermesDesktop.api<T>()` for all gateway calls
- **Notifications**: `notify()` / `notifyError()` from `store/notifications.ts`
- **Diff rendering**: ANSI-aware via `lib/ansi.ts`, inline diffs via `components/chat/diff-lines.tsx`

## File Structure Conventions
- New panels → `app/chat/right-rail/` or `app/chat/` subdirectories
- New stores → `store/` with nanostores atoms
- New components → `components/assistant-ui/` for chat-related
- Reuse `components/ui/*` — don't recreate Button, Switch, Tabs, etc.

## Testing
- Run `npm run dev` for hot-reload dev mode
- Test terminal streaming: Run a long command (e.g. `ping -n 5 google.com`) and verify
  incremental output appears in the tool card
- Test diff viewer: Use `patch` or `write_file` tool, then check Review panel shows diff
- Test Overview: Create files, run tools, verify all artifacts listed in Overview

## Explicit Boundaries
- NEVER use emojis for icons — use Tabler icons from `@/lib/icons` or Codicons via `<Codicon name="..." />`
- NEVER modify `tui_gateway/server.py` — gateway is upstream, don't touch it
- NEVER modify `src/lib/ansi.ts` — it's stable, use as-is
- NEVER use class components or lifecycle methods
- NEVER bypass `window.hermesDesktop.api()` — always go through the preload bridge
- NEVER hardcode colors — use Tailwind theme tokens
- DO reuse `DiffLines` component for any diff rendering
- DO use nanostores for any new reactive state
- DO add `notify()` on every async user action (save, toggle, etc.)

## Key Files Reference
| File | Purpose |
|------|---------|
| `src/app/session/hooks/use-message-stream.ts` | Gateway event dispatching |
| `src/lib/chat-messages.ts` | Message model, upsertToolPart |
| `src/components/assistant-ui/tool-fallback-model.ts` | Tool view model (buildToolView) |
| `src/components/assistant-ui/tool-fallback.tsx` | ToolEntry rendering |
| `src/components/chat/diff-lines.tsx` | Inline unified diff renderer |
| `src/store/tool-diffs.ts` | Per-tool-call diff storage |
| `src/store/layout.ts` | Pane/layout state |
| `src/app/chat/right-rail/preview.tsx` | Right rail tab system (preview + file) |
| `src/app/chat/right-rail/preview-file.tsx` | File preview — EDITABLE (pencil toggle) |
| `src/app/right-sidebar/index.tsx` | 4-tab sidebar: Overview, Review, Files, Terminal |
| `src/app/right-sidebar/store.ts` | Right sidebar tab state |
| `src/app/desktop-controller.tsx` | Main layout composition |
| `src/store/session-changes.ts` | [NEW] File change tracking store |
| `src/store/file-preview.ts` | [NEW] File edit dirty state |
| `src/components/assistant-ui/session-overview.tsx` | [NEW] Overview panel |
| `src/components/assistant-ui/session-review.tsx` | [NEW] Review panel |
| `src/components/assistant-ui/terminal-output.tsx` | [NEW] Enhanced terminal output |
| `src/components/assistant-ui/terminal-header.tsx` | [NEW] Terminal command header |
| `tui_gateway/server.py` | Gateway RPC handlers (READ ONLY) |
| `agent/display.py` | Inline diff capture (READ ONLY) |
