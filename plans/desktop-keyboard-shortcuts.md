# Hermes Desktop Keyboard Shortcuts Implementation Plan

**Issue**: [#38118](https://github.com/NousResearch/hermes-agent/issues/38118)
**Branch**: `feat/desktop-keyboard-shortcuts`
**Status**: Planning

---

## Overview

Add missing keyboard shortcuts and extend `Cmd/Ctrl+W` cascade behavior in Hermes Desktop.

### Already Working (no changes needed)

These shortcuts are already implemented and do NOT need new code:

| Shortcut | Location | Notes |
|---|---|---|
| `Cmd/Ctrl+B` (toggle sidebar) | `sidebar.tsx:88-100` | shadcn SidebarProvider handles this |
| `Cmd/Ctrl+N` (new chat) | `desktop-controller.tsx:390-421` | Also `Shift+N` when no input focused |
| `Cmd/Ctrl+/-/0` (zoom) | `main.cjs:2837-2861` | Uses `setZoomLevel()` with 0.1 step |
| `Cmd/Ctrl+Opt/Shift+I` (DevTools) | `main.cjs:2811` | F12 also works |
| `Cmd/Ctrl+W` (close preview rail) | `main.cjs:2825` + `desktop-controller.tsx:175-196` | Only closes preview tabs, does not cascade |
| `Escape` (close overlay) | `overlay-view.tsx:33-47` | Each overlay handles its own Escape |

### Work Required

| Task | Effort |
|---|---|
| Extend `Cmd+W` cascade to overlays and sidebars | 2-3 days |
| Add `Cmd/Ctrl+,` for Settings | 0.5 day |
| Add `Cmd/Ctrl+.` for right panel toggle | 0.5 day |
| Add `Cmd/Ctrl+K` for search / command center | 1 day |
| Testing across macOS / Windows / Linux | 2 days |
| **Total** | **~1 week** |

---

## Phase 1: Extend `Cmd/Ctrl+W` Cascade (2-3 days)

The most impactful change. Currently `Cmd+W` has only two modes: close preview or close window. It should cascade through UI layers before closing the window.

### 1.1 Architecture

The cascade must bridge main process and renderer because overlay state lives in React Router (renderer), not in the main process.

**Protocol:**
1. Main process `before-input-event` intercepts `Cmd/Ctrl+W`
2. Main sends `hermes:close-requested` to renderer (new IPC event)
3. Renderer runs the cascade and decides what to close
4. If nothing to close, renderer sends `hermes:close-window` back to main (new IPC event)
5. Main closes/minimizes the window

**File**: `apps/desktop/electron/main.cjs`

- [ ] Rename `installPreviewShortcut` to `installCloseShortcut` (or add to it)
- [ ] Always intercept `Cmd/Ctrl+W` (not just when `previewShortcutActive`)
- [ ] Always send `hermes:close-requested` to renderer instead of closing the window
- [ ] Add `ipcMain.on('hermes:close-window', ...)` handler that calls `mainWindow.close()` (Win/Linux) or `mainWindow.minimize()` (macOS)
- [ ] Keep macOS menu accelerator (`CommandOrControl+W`) in sync — route through the same IPC

**File**: `apps/desktop/electron/preload.cjs`

- [ ] Expose `hermesDesktop.onCloseRequested(callback)` — listens for `hermes:close-requested`
- [ ] Expose `hermesDesktop.closeWindow()` — sends `hermes:close-window` to main

### 1.2 Renderer Cascade Logic

**File**: `apps/desktop/src/app/desktop-controller.tsx`

Replace the existing `Cmd+W` `useEffect` (lines 175-196) with a cascade handler:

```ts
useEffect(() => {
  const CASCADE = [
    // 1. Close active overlay (settings, command-center, agents)
    () => {
      if (overlayOpen) {
        closeOverlayToPreviousRoute()
        return true
      }
    },
    // 2. Close right sidebar (file browser)
    () => {
      if ($fileBrowserOpen.get()) {
        toggleFileBrowserOpen()
        return true
      }
    },
    // 3. Close preview/file rail
    () => {
      if ($filePreviewTarget.get() || $previewTarget.get()) {
        closeActiveRightRailTab()
        return true
      }
    },
    // 4. Close left sidebar
    () => {
      if ($sidebarOpen.get()) {
        setSidebarOpen(false)
        return true
      }
    },
    // 5. Nothing to close — close/minimize the window
    () => {
      window.hermesDesktop?.closeWindow()
      return true
    },
  ]

  const onKey = (event: KeyboardEvent) => {
    if (!(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey) return
    if (event.key.toLowerCase() !== 'w') return
    event.preventDefault()
    event.stopPropagation()
    for (const step of CASCADE) if (step()) break
  }

  const unsubscribe = window.hermesDesktop?.onCloseRequested?.(() => {
    for (const step of CASCADE) if (step()) break
  })

  window.addEventListener('keydown', onKey, { capture: true })
  return () => {
    unsubscribe?.()
    window.removeEventListener('keydown', onKey, { capture: true })
  }
}, [overlayOpen, closeOverlayToPreviousRoute])
```

Note: `overlayOpen` and `closeOverlayToPreviousRoute` come from the existing `useOverlayRouting()` hook — no new overlay store needed.

### 1.3 Remove `previewShortcutActive` Flag

Once the renderer handles the full cascade, the main process no longer needs to know whether a preview is active. The `hermes:previewShortcutActive` IPC channel can be removed.

- [ ] Remove `previewShortcutActive` variable from `main.cjs`
- [ ] Remove `hermes:previewShortcutActive` IPC listener from `main.cjs`
- [ ] Remove `setPreviewShortcutActive` calls from `desktop-controller.tsx`
- [ ] Remove corresponding `send` calls from `preload.cjs`

---

## Phase 2: New Shortcuts (1 day)

All new shortcuts follow the existing pattern: `before-input-event` in main process sends IPC to renderer, renderer dispatches the action.

### 2.1 Settings (`Cmd/Ctrl+,`)

**File**: `apps/desktop/electron/main.cjs` (add to the `before-input-event` chain or as a new `install*` function)

- [ ] Intercept `Cmd/Ctrl+,` (key `,`, modifier Cmd on macOS / Ctrl on others)
- [ ] Send `hermes:open-settings` to renderer

**File**: `apps/desktop/src/app/desktop-controller.tsx`

- [ ] Listen for `hermes:open-settings` IPC
- [ ] Call `navigate(SETTINGS_ROUTE)` (same as existing `openSettings()` in `use-session-actions.ts:383`)
- [ ] If settings already open, close overlay (toggle behavior)

### 2.2 Toggle Right Panel (`Cmd/Ctrl+.`)

**File**: `apps/desktop/electron/main.cjs`

- [ ] Intercept `Cmd/Ctrl+.`
- [ ] Send `hermes:toggle-right-panel` to renderer

**File**: `apps/desktop/src/app/desktop-controller.tsx`

- [ ] Listen for `hermes:toggle-right-panel` IPC
- [ ] Call `toggleFileBrowserOpen()` from `store/layout.ts`

### 2.3 Search / Command Center (`Cmd/Ctrl+K`)

**File**: `apps/desktop/electron/main.cjs`

- [ ] Intercept `Cmd/Ctrl+K`
- [ ] Send `hermes:open-search` to renderer

**File**: `apps/desktop/src/app/desktop-controller.tsx`

- [ ] Listen for `hermes:open-search` IPC
- [ ] Call `toggleCommandCenter()` from `useOverlayRouting()` (opens command-center overlay)
- [ ] If command center already open, close it (toggle behavior)

### 2.4 Preload Bridge

**File**: `apps/desktop/electron/preload.cjs`

Add IPC listeners for the three new channels:
- `hermes:open-settings`
- `hermes:toggle-right-panel`
- `hermes:open-search`

Each exposed as `window.hermesDesktop.onXxx(callback)`.

---

## Phase 3: Testing (2 days)

### 3.1 Unit Tests

**File**: `apps/desktop/electron/keyboard-shortcuts.test.cjs` (new)

- [ ] Test that `Cmd/Ctrl+W` always sends IPC to renderer (never closes window directly)
- [ ] Test that new shortcuts send correct IPC channel names
- [ ] Test platform detection (Cmd vs Ctrl modifier)

### 3.2 Integration Tests

**File**: `apps/desktop/src/hooks/use-keyboard-shortcuts.test.ts` (new)

- [ ] Test `Cmd+W` cascade: overlay → right panel → preview → sidebar → window close
- [ ] Test `Cmd+,` opens settings
- [ ] Test `Cmd+.` toggles right panel
- [ ] Test `Cmd+K` opens command center

### 3.3 Manual Testing Checklist

**All platforms (macOS, Windows, Linux):**
- [ ] `Cmd/Ctrl+,` opens Settings; pressing again closes it
- [ ] `Cmd/Ctrl+.` toggles right sidebar (file browser)
- [ ] `Cmd/Ctrl+K` opens Command Center; pressing again closes it
- [ ] `Cmd/Ctrl+B` toggles left sidebar (existing — verify no regression)
- [ ] `Cmd/Ctrl+N` creates new chat (existing — verify no regression)
- [ ] `Cmd/Ctrl+/-/0` zoom works (existing — verify no regression)

**`Cmd/Ctrl+W` cascade:**
- [ ] With Settings overlay open → closes Settings, not the window
- [ ] With Command Center open → closes Command Center, not the window
- [ ] With right sidebar open (no overlay) → closes right sidebar
- [ ] With preview rail open (no overlay/sidebar) → closes preview rail
- [ ] With left sidebar open (nothing else) → closes left sidebar
- [ ] With nothing open → minimizes (macOS) / closes window (Win/Linux)
- [ ] `Cmd/Ctrl+W` works while focus is in a text input

---

## File Changes Summary

| File | Action | Description |
|---|---|---|
| `electron/main.cjs` | Modify | Rename `installPreviewShortcut` → `installCloseShortcut`, always forward `Cmd+W` to renderer, add `hermes:close-window` handler, add 3 new shortcut `install*` functions |
| `electron/preload.cjs` | Modify | Add 4 new IPC listeners (`onCloseRequested`, `onOpenSettings`, `onToggleRightPanel`, `onOpenSearch`) and 1 sender (`closeWindow`) |
| `src/app/desktop-controller.tsx` | Modify | Replace `Cmd+W` handler with cascade, add IPC listeners for 3 new shortcuts |
| `plans/desktop-keyboard-shortcuts.md` | Modify | This file |

No new files needed. No new npm packages. No new stores.

---

## Dependencies

- Uses Electron's built-in `before-input-event` API (already in use)
- Uses existing `useOverlayRouting()` hook for overlay state (no new store)
- Uses existing `toggleFileBrowserOpen()` from `store/layout.ts`
- Uses existing `toggleCommandCenter()` from `useOverlayRouting()`
- Uses existing IPC infrastructure in `main.cjs` and `preload.cjs`

---

## Related Issues

- #37917 - Windows zoom shortcuts broken (already fixed by `installZoomShortcuts`)
- #37915 - Keyboard navigation bugs
- #37823 - Arrow key navigation
- #38072 - Accessibility improvements

---

## Success Criteria

1. `Cmd/Ctrl+W` cascades: overlay → right panel → preview → sidebar → window close/minimize
2. `Cmd/Ctrl+,` opens Settings
3. `Cmd/Ctrl+.` toggles right sidebar
4. `Cmd/Ctrl+K` opens Command Center
5. All existing shortcuts continue to work (no regressions)
6. Cascade works on macOS, Windows, and Linux

---

## Notes

- The `previewShortcutActive` boolean flag should be removed once the renderer owns the full cascade — it becomes redundant.
- `Shift+N` (new chat when no input focused) is an existing undocumented shortcut in `desktop-controller.tsx:407-408`. It should be preserved.
- Future work: custom keybinding configuration (out of scope for this PR).
