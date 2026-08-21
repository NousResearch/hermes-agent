// IPC surface for the pop-out pet overlay (mascot window). Extracted from
// main.ts; window handles stay injected because main.ts owns their lifecycle.
import { type BrowserWindow, ipcMain } from 'electron'

export interface PetOverlayIpcDeps {
  getMainWindow: () => BrowserWindow | null
  getPetOverlayWindow: () => BrowserWindow | null
  openPetOverlay: (bounds: unknown) => void
  closePetOverlay: () => void
}

export function registerPetOverlayIpc({
  getMainWindow,
  getPetOverlayWindow,
  openPetOverlay,
  closePetOverlay
}: PetOverlayIpcDeps) {
  // `request` is `{ bounds, screen }`. A fresh pop-out passes viewport-space
  // bounds (screen=false): convert to screen space by adding the main window's
  // content origin so the pet lands where it sat in-window. A remembered/dragged
  // spot passes screen-space bounds (screen=true) and is used as-is. We return the
  // resolved screen bounds so the renderer can persist exactly where it opened.
  ipcMain.handle('hermes:pet-overlay:open', async (_event, request) => {
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

    openPetOverlay(screenBounds)

    return { ok: true, bounds: screenBounds }
  })
  ipcMain.handle('hermes:pet-overlay:close', async () => {
    closePetOverlay()

    return { ok: true }
  })
  // Drag/resize: the overlay reports new absolute screen bounds (it already knows
  // the pointer's screen coords). Drag keeps the size constant; the wheel-to-scale
  // gesture grows/shrinks it so the sprite is never cropped by the window edge.
  // The window is created non-resizable (no stray edge-drag on the transparent
  // frameless panel), which on Windows/Linux also blocks programmatic setBounds
  // sizing — so briefly flip resizable on whenever the size actually changes.
  ipcMain.on('hermes:pet-overlay:set-bounds', (_event, bounds) => {
    const petOverlayWindow = getPetOverlayWindow()

    if (!petOverlayWindow || petOverlayWindow.isDestroyed() || !bounds) {
      return
    }

    const win = petOverlayWindow
    const width = Math.max(80, Math.round(bounds.width))
    const height = Math.max(80, Math.round(bounds.height))
    const [curW, curH] = win.getSize()
    const resizing = width !== curW || height !== curH

    if (resizing && !win.isResizable()) {
      win.setResizable(true)
    }

    win.setBounds({ x: Math.round(bounds.x), y: Math.round(bounds.y), width, height })

    if (resizing) {
      win.setResizable(false)
    }
  })
  // Click-through: the overlay window is a full rectangle but only the pet pixels
  // should be interactive. The renderer toggles this as the cursor enters/leaves
  // the sprite so transparent margins pass clicks to whatever is behind.
  ipcMain.on('hermes:pet-overlay:ignore-mouse', (_event, ignore) => {
    const petOverlayWindow = getPetOverlayWindow()

    if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
      petOverlayWindow.setIgnoreMouseEvents(Boolean(ignore), { forward: true })
    }
  })
  // The overlay is a non-activating panel (focusable:false) so it never steals
  // the app's cmd/alt-tab anchor from the main window. But the pop-up composer
  // needs the keyboard, so the renderer asks us to flip it focusable + focus it
  // while the composer is open, then back to non-activating when it closes.
  ipcMain.on('hermes:pet-overlay:set-focusable', (_event, focusable) => {
    const petOverlayWindow = getPetOverlayWindow()

    if (!petOverlayWindow || petOverlayWindow.isDestroyed()) {
      return
    }

    petOverlayWindow.setFocusable(Boolean(focusable))

    if (focusable) {
      petOverlayWindow.focus()
    }
  })
  // Main renderer → overlay: forward the latest pet state for the overlay to render.
  ipcMain.on('hermes:pet-overlay:state', (_event, payload) => {
    const petOverlayWindow = getPetOverlayWindow()

    if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
      petOverlayWindow.webContents.send('hermes:pet-overlay:state', payload)
    }
  })
  // Overlay → main renderer: control messages (pop back in, composer submit).
  ipcMain.on('hermes:pet-overlay:control', (_event, payload) => {
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

    mainWindow.webContents.send('hermes:pet-overlay:control', payload)
  })
}
