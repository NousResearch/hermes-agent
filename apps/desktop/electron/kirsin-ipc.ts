// IPC surface for the Kirsin Agent Window (the persistent always-on-top
// floating chat). Extracted from main.ts the same way hud-ipc.ts is: the
// window handle and the open/close/reset orchestration stay injected because
// main.ts owns the window lifecycle and the persistent `Ctrl+Shift+K` shortcut
// toggles it.
//
// This is the HUD surface MINUS the transient-band machinery: no frost
// (the panel is always solid), no ignore-mouse / click-through (a persistent
// chat never fades), no cursor / game-overlay feeds (it doesn't hover over
// other apps the way the HUD band does). What remains is exactly what a
// draggable, resizable, closable floating window needs.

import { type BrowserWindow, ipcMain, screen } from 'electron'

import { createHudDragSession } from './hud-drag'
import { normalizeKirsinResizeBounds } from './kirsin-geometry'

export interface KirsinIpcDeps {
  getKirsinWindow: () => BrowserWindow | null
  openKirsinWindow: () => void
  closeKirsinWindow: () => void
  resetKirsinLayout: () => boolean
}

export function registerKirsinIpc({
  getKirsinWindow,
  openKirsinWindow,
  closeKirsinWindow,
  resetKirsinLayout
}: KirsinIpcDeps) {
  const kirsinDrag = createHudDragSession()

  ipcMain.handle('hermes:kirsin:open', () => {
    openKirsinWindow()

    return { ok: true }
  })

  ipcMain.handle('hermes:kirsin:close', () => {
    closeKirsinWindow()

    return { ok: true }
  })

  ipcMain.on('hermes:kirsin:begin-move', event => {
    const win = getKirsinWindow()

    if (!win || win.isDestroyed() || event.sender !== win.webContents) {
      return
    }

    const [x, y] = win.getPosition()
    kirsinDrag.begin(screen.getCursorScreenPoint(), { x, y })
  })

  ipcMain.on('hermes:kirsin:end-move', event => {
    const win = getKirsinWindow()

    if (win && !win.isDestroyed() && event.sender !== win.webContents) {
      return
    }

    kirsinDrag.end()
  })

  ipcMain.on('hermes:kirsin:move-by', (event, delta) => {
    const win = getKirsinWindow()

    if (!win || win.isDestroyed() || event.sender !== win.webContents) {
      return
    }

    const width = Number(delta?.width)
    const height = Number(delta?.height)

    if (!Number.isFinite(width) || !Number.isFinite(height)) {
      return
    }

    const origin = kirsinDrag.origin(screen.getCursorScreenPoint())

    if (!origin) {
      return
    }

    // Cursor − grab offset in Electron DIP (see hud-drag.ts). setBounds — NOT
    // setPosition: on Windows a transparent frameless window silently grows
    // ~1px per setPosition call. The renderer snapshots its size when the
    // drag arms and re-pins it on every move (same pattern as the HUD).
    win.setBounds({
      x: origin.x,
      y: origin.y,
      width: Math.round(width),
      height: Math.round(height)
    })
  })

  // Resize from the panel's edge/corner handles (Phase 3). The window is
  // created non-resizable (see spawnKirsinWindow), which on Windows/Linux also
  // blocks programmatic setBounds sizing — briefly flip resizable on while the
  // size actually changes, exactly like the HUD's set-bounds handler.
  ipcMain.on('hermes:kirsin:set-bounds', (event, bounds) => {
    const win = getKirsinWindow()

    if (!win || win.isDestroyed() || event.sender !== win.webContents || !bounds) {
      return
    }

    const nextBounds = normalizeKirsinResizeBounds(bounds)

    if (!nextBounds) {
      return
    }

    const { width, height } = nextBounds
    const [curW, curH] = win.getSize()
    const resizing = width !== curW || height !== curH
    const restoreResizeLock = resizing && !win.isResizable()

    try {
      if (restoreResizeLock) {
        win.setResizable(true)
      }

      win.setBounds(nextBounds)
    } catch {
      // The window may disappear between validation and the native call.
    } finally {
      if (restoreResizeLock && !win.isDestroyed()) {
        win.setResizable(false)
      }
    }
  })

  ipcMain.handle('hermes:kirsin:reset-layout', event => {
    const win = getKirsinWindow()

    if (!win || win.isDestroyed() || event.sender !== win.webContents) {
      return { ok: false }
    }

    return { ok: resetKirsinLayout() }
  })
}
