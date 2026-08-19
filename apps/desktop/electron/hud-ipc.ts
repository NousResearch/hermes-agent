// IPC surface for HUD mode (the chrome-free floating chat band). Extracted
// from main.ts; the HUD window handle and session-id latch stay injected
// because main.ts owns the window lifecycle and the close broadcast reads the
// latch when handing the session back to the app window.
import { type BrowserWindow, ipcMain } from 'electron'

export interface HudIpcDeps {
  isMac: boolean
  getHudWindow: () => BrowserWindow | null
  openHudWindow: (sessionId: null | string, profile: null | string) => void
  closeHudWindow: () => void
  setHudSessionId: (sessionId: null | string) => void
}

export function registerHudIpc({ isMac, getHudWindow, openHudWindow, closeHudWindow, setHudSessionId }: HudIpcDeps) {
  ipcMain.handle('hermes:hud:open', async (_event, request) => {
    openHudWindow(
      typeof request?.sessionId === 'string' ? request.sessionId : null,
      typeof request?.profile === 'string' ? request.profile : null
    )

    return { ok: true }
  })

  // Real frosted glass behind the band — the thing CSS backdrop-filter cannot do,
  // because Chromium composites a transparent window's page against nothing and
  // the desktop is not in its backdrop root. Vibrancy IS the window's content
  // view, so it frosts the whole rectangle; the HUD's layout leaves no dead
  // margins for that reason, and the renderer only turns it on while the band is
  // showing (idle HUD mode must be the bar and nothing else).
  ipcMain.handle('hermes:hud:vibrancy', (_event, on) => {
    const hudWindow = getHudWindow()

    if (hudWindow && !hudWindow.isDestroyed() && isMac) {
      hudWindow.setVibrancy(on ? 'hud' : null)
    }

    return { ok: true }
  })

  // Let clicks fall through the HUD wherever it isn't really there. An
  // always-on-top window eats every click inside its rectangle, and most of that
  // rectangle is a faded-out band over whatever the user is actually working in.
  // `forward` keeps mousemove flowing so the renderer can re-arm when the cursor
  // reaches the bar.
  ipcMain.on('hermes:hud:ignore-mouse', (_event, ignore) => {
    const hudWindow = getHudWindow()

    if (hudWindow && !hudWindow.isDestroyed()) {
      hudWindow.setIgnoreMouseEvents(Boolean(ignore), { forward: true })
    }
  })

  ipcMain.on('hermes:hud:move-by', (event, delta) => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed() || event.sender !== hudWindow.webContents) {
      return
    }

    const dx = Number(delta?.x)
    const dy = Number(delta?.y)
    const width = Number(delta?.width)
    const height = Number(delta?.height)

    if (!Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(width) || !Number.isFinite(height)) {
      return
    }

    const [x, y] = hudWindow.getPosition()

    // setBounds — NOT setPosition: on Windows, a transparent frameless window
    // silently grows ~1px per setPosition call (worse at >100% DPI). The renderer
    // snapshots outerWidth/outerHeight when the composer drag arms and re-pins
    // to that size on every moveBy (same pattern as the pet overlay drag).
    hudWindow.setBounds({
      x: Math.round(x + dx),
      y: Math.round(y + dy),
      width: Math.round(width),
      height: Math.round(height)
    })
  })

  // Resize from the HUD's corner handle. The window is created non-resizable
  // (see spawnHudWindow — a transparent frameless window must not expose a
  // system resize hot-zone, or dragging grows it), which on Windows/Linux also
  // blocks programmatic setBounds sizing — so briefly flip resizable on while
  // the size actually changes, exactly like the pet overlay's wheel-scale does.
  ipcMain.on('hermes:hud:set-bounds', (event, bounds) => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed() || event.sender !== hudWindow.webContents || !bounds) {
      return
    }

    const win = hudWindow
    const width = Math.max(380, Math.round(Number(bounds.width)))
    const height = Math.max(160, Math.round(Number(bounds.height)))
    const [curW, curH] = win.getSize()
    const resizing = width !== curW || height !== curH

    if (resizing && !win.isResizable()) {
      win.setResizable(true)
    }

    win.setBounds({ x: Math.round(Number(bounds.x)), y: Math.round(Number(bounds.y)), width, height })

    if (resizing) {
      win.setResizable(false)
    }
  })

  // The HUD renderer reporting which session it is on, so the close broadcast
  // can hand it back to the app window (see hudSessionId).
  ipcMain.on('hermes:hud:session', (event, sessionId) => {
    const hudWindow = getHudWindow()

    if (hudWindow && !hudWindow.isDestroyed() && event.sender === hudWindow.webContents) {
      setHudSessionId(typeof sessionId === 'string' && sessionId ? sessionId : null)
    }
  })

  ipcMain.handle('hermes:hud:close', async () => {
    closeHudWindow()

    return { ok: true }
  })
}
