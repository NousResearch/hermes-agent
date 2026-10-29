// IPC surface for the pop-out companion overlay (mascot window). Extracted from
// main.ts; window handles stay injected because main.ts owns their lifecycle.
import { type BrowserWindow, ipcMain, screen } from 'electron'

import { companionOverlayClickThrough, resolveCompanionOverlayBounds } from './companion-overlay'

export interface CompanionOverlayIpcDeps {
  getMainWindow: () => BrowserWindow | null
  getCompanionOverlayWindow: () => BrowserWindow | null
  openCompanionOverlay: (bounds: unknown) => void
  closeCompanionOverlay: () => void
}

// Where the overlay may actually sit: `bounds` kept wholly on a connected
// display, or re-centered on the main window's display when it is on none
// (see resolveCompanionOverlayBounds). Null for garbage bounds.
export function placeCompanionOverlay(bounds, mainWindow: BrowserWindow | null) {
  let anchor = null

  try {
    if (mainWindow && !mainWindow.isDestroyed()) {
      anchor = mainWindow.getContentBounds()
    }
  } catch {
    // Resolve falls back to the primary display when the anchor is unknown.
  }

  return resolveCompanionOverlayBounds(bounds, screen.getAllDisplays(), anchor)
}

export function registerCompanionOverlayIpc({
  getMainWindow,
  getCompanionOverlayWindow,
  openCompanionOverlay,
  closeCompanionOverlay
}: CompanionOverlayIpcDeps) {
  // `request` is `{ bounds, screen }`. A fresh pop-out passes viewport-space
  // bounds (screen=false): convert to screen space by adding the main window's
  // content origin so the companion lands where it sat in-window. A remembered/dragged
  // spot passes screen-space bounds (screen=true) and is used as-is. We return the
  // resolved screen bounds so the renderer can persist exactly where it opened.
  ipcMain.handle('hermes:companion-overlay:open', async (_event, request) => {
    const bounds = request && request.bounds ? request.bounds : request
    const isScreen = Boolean(request && request.screen)
    const mainWindow = getMainWindow()
    let screenBounds = bounds

    try {
      if (bounds && !isScreen && mainWindow && !mainWindow.isDestroyed()) {
        const content = mainWindow.getContentBounds()
        screenBounds = {
          x: content.x + (bounds.x || 0),
          y: content.y + (bounds.y || 0),
          width: bounds.width,
          height: bounds.height
        }
      }
    } catch {
      // Fall back to raw bounds if the window geometry is unavailable.
    }

    // Both paths land wholly inside a display's work area: a fresh pop-out near
    // the window edge or a spot saved on a since-unplugged monitor would
    // otherwise open the transparent, non-activating overlay (mostly)
    // off-screen where the companion can't be found. The corrected bounds are echoed
    // so the renderer persists the on-screen spot (self-healing).
    screenBounds = placeCompanionOverlay(screenBounds, mainWindow) ?? screenBounds

    openCompanionOverlay(screenBounds)

    return { ok: true, bounds: screenBounds }
  })
  ipcMain.handle('hermes:companion-overlay:close', async () => {
    closeCompanionOverlay()

    return { ok: true }
  })
  // Drag/resize: the overlay reports new absolute screen bounds (it already knows
  // the pointer's screen coords). Drag keeps the size constant; the wheel-to-scale
  // gesture grows/shrinks it so the sprite is never cropped by the window edge.
  // The window is created non-resizable (no stray edge-drag on the transparent
  // frameless panel), which on Windows/Linux also blocks programmatic setBounds
  // sizing — so briefly flip resizable on whenever the size actually changes.
  ipcMain.on('hermes:companion-overlay:set-bounds', (_event, bounds) => {
    const companionOverlayWindow = getCompanionOverlayWindow()

    if (!companionOverlayWindow || companionOverlayWindow.isDestroyed() || !bounds) {
      return
    }

    const next = placeCompanionOverlay(bounds, getMainWindow())

    if (!next) {
      return
    }

    const win = companionOverlayWindow
    const { width, height } = next
    const [curW, curH] = win.getSize()
    const resizing = width !== curW || height !== curH

    if (resizing && !win.isResizable()) {
      win.setResizable(true)
    }

    win.setBounds(next)

    if (resizing) {
      win.setResizable(false)
    }
  })
  // Click-through: the overlay window is a full rectangle but only the companion pixels
  // should be interactive. The renderer toggles this as the cursor enters/leaves
  // the sprite so transparent margins pass clicks to whatever is behind.
  ipcMain.on('hermes:companion-overlay:ignore-mouse', (_event, ignore) => {
    const companionOverlayWindow = getCompanionOverlayWindow()

    // No forward on Linux: an ignoring overlay could never be re-armed.
    if (Boolean(ignore) && !companionOverlayClickThrough()) {
      return
    }

    if (companionOverlayWindow && !companionOverlayWindow.isDestroyed()) {
      companionOverlayWindow.setIgnoreMouseEvents(Boolean(ignore), { forward: true })
    }
  })
  // The overlay is a non-activating panel (focusable:false) so it never steals
  // the app's cmd/alt-tab anchor from the main window. But the pop-up composer
  // needs the keyboard, so the renderer asks us to flip it focusable + focus it
  // while the composer is open, then back to non-activating when it closes.
  ipcMain.on('hermes:companion-overlay:set-focusable', (_event, focusable) => {
    const companionOverlayWindow = getCompanionOverlayWindow()

    if (!companionOverlayWindow || companionOverlayWindow.isDestroyed()) {
      return
    }

    companionOverlayWindow.setFocusable(Boolean(focusable))

    if (focusable) {
      companionOverlayWindow.focus()
    }
  })
  // Main renderer → overlay: forward the latest companion state for the overlay to render.
  ipcMain.on('hermes:companion-overlay:state', (_event, payload) => {
    const companionOverlayWindow = getCompanionOverlayWindow()

    if (companionOverlayWindow && !companionOverlayWindow.isDestroyed()) {
      companionOverlayWindow.webContents.send('hermes:companion-overlay:state', payload)
    }
  })
  // Overlay → main renderer: control messages (pop back in, composer submit).
  ipcMain.on('hermes:companion-overlay:control', (_event, payload) => {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    // Double-click toggles the app window: hide it away if it's up front, bring it
    // back if it's minimized/buried. Pure window control — nothing for the
    // renderer to do, so don't forward it.
    if (payload && payload.type === 'toggle-app') {
      if (mainWindow.isMinimized() || !mainWindow.isVisible()) {
        mainWindow.show()
        mainWindow.focus()
      } else {
        mainWindow.minimize()
      }

      return
    }

    // The mail icon means "take me to the app": raise the main window (it may be
    // minimized or buried) before the renderer navigates to the latest thread.
    if (payload && payload.type === 'open-app') {
      if (mainWindow.isMinimized()) {
        mainWindow.restore()
      }

      mainWindow.show()
      mainWindow.focus()
    }

    // A drag/resize end reports the rect the overlay asked for; persist where
    // set-bounds actually put it (clamped on-screen), not the raw request.
    if (payload && payload.type === 'bounds') {
      payload = { ...payload, bounds: placeCompanionOverlay(payload.bounds, mainWindow) ?? payload.bounds }
    }

    mainWindow.webContents.send('hermes:companion-overlay:control', payload)
  })
}
