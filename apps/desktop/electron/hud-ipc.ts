// IPC surface for HUD mode (the chrome-free floating chat band). Extracted
// from main.ts; the HUD window handle and session-id latch stay injected
// because main.ts owns the window lifecycle and the close broadcast reads the
// latch when handing the session back to the app window.
import { type BrowserWindow, ipcMain } from 'electron'

import { hudFrostFor, type TranslucencyState } from './translucency'

export interface HudIpcDeps {
  isMac: boolean
  /** Main's authoritative translucency state (Settings → Appearance). */
  getTranslucencyState: () => TranslucencyState
  getHudWindow: () => BrowserWindow | null
  openHudWindow: (sessionId: null | string, profile: null | string) => void
  closeHudWindow: () => void
  setHudSessionId: (sessionId: null | string) => void
}

export function registerHudIpc({
  isMac,
  getTranslucencyState,
  getHudWindow,
  openHudWindow,
  closeHudWindow,
  setHudSessionId
}: HudIpcDeps) {
  // Whether the band currently covers the window below the bar. The renderer
  // is the only party that can know this (it measures the transcript), and it
  // is half of the frost decision — the other half is the user's setting,
  // which main owns. Latched so a Settings change can re-decide without
  // waiting for the HUD to report again.
  let bandShowing = false
  let applied: null | string = null
  let appliedTo: BrowserWindow | null = null

  // Real frosted glass behind the band — the thing CSS backdrop-filter cannot do,
  // because Chromium composites a transparent window's page against nothing and
  // the desktop is not in its backdrop root. The material IS the window's content
  // view, so it frosts the whole rectangle; the HUD's layout leaves no dead
  // margins for that reason, and it only turns on while the band is showing
  // (idle HUD mode must be the bar and nothing else).
  //
  // macOS ONLY. Windows' equivalent (setBackgroundMaterial → the DWM backdrop)
  // is mutually exclusive with window transparency, so it is not called at all
  // here — see the note at the bottom of this function.
  //
  // Diffed before issuing: `setVibrancy` carries a 150ms animation that restarts
  // if re-issued, so a repeated call would keep the material from ever settling
  // (the same churn the chat windows' native-diff contract exists to prevent).
  //
  // The diff is keyed to the WINDOW as well as the value. A HUD respawn (the
  // profile switch in openHudWindow destroys and rebuilds it) hands back a
  // fresh window carrying no material, and a latch that only remembered the
  // value would recognise its own last answer and skip — leaving the new HUD
  // unfrosted until something else happened to change the signature.
  const applyHudFrost = () => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed()) {
      applied = null
      appliedTo = null

      return
    }

    const frost = hudFrostFor(getTranslucencyState(), bandShowing)
    const signature = `${frost.vibrancy ?? 'off'}:${frost.backgroundMaterial}`

    if (applied === signature && appliedTo === hudWindow) {
      return
    }

    applied = signature
    appliedTo = hudWindow

    if (isMac && typeof hudWindow.setVibrancy === 'function') {
      hudWindow.setVibrancy(frost.vibrancy)
    }

    // Windows: never touch setBackgroundMaterial on the HUD. Live-verified on
    // Win11 (Electron 40.10.2, RTX 4090): ANY setBackgroundMaterial call on a
    // transparent window — including 'none', which is what the idle HUD asks
    // for — permanently kills per-pixel alpha, and every transparent pixel
    // composites as opaque white. Neither 'auto' nor a follow-up
    // setBackgroundColor('#00000000') restores it. The DWM backdrop and window
    // transparency are mutually exclusive, so the Windows HUD keeps the CSS
    // tint the sheet already paints and skips the native frost entirely.
  }

  ipcMain.handle('hermes:hud:open', async (_event, request) => {
    openHudWindow(
      typeof request?.sessionId === 'string' ? request.sessionId : null,
      typeof request?.profile === 'string' ? request.profile : null
    )

    return { ok: true }
  })

  ipcMain.handle('hermes:hud:frost', (_event, showing) => {
    bandShowing = Boolean(showing)
    applyHudFrost()

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

  // Main re-applies the frost when the translucency SETTING changes, since the
  // band's own report only fires when the band itself moves.
  return { applyHudFrost }
}
