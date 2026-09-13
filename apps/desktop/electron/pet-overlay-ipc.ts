// IPC surface for the pop-out pet overlay (mascot window). Extracted from
// main.ts; window handles stay injected because main.ts owns their lifecycle.
import { type BrowserWindow, ipcMain, screen } from 'electron'

import { overlayWindowBoundsToDip } from './pet-overlay-geometry'
import { captureVisualLedgeBelow } from './pet-overlay-visual-surfaces'
import { enumerateWindowsFrontToBack, enumerationFailed } from './window-below'

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
  let cachedRoamWindows:
    | {
        capturedAt: number
        displayId: string
        revision: number
        windows: Array<{ height: number; width: number; x: number; y: number }>
      }
    | undefined

  let roamWindowRevision = 0

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
  // Native roam geometry for the display currently containing the pet. Window
  // enumeration is already shared with read_window_below and the HUD; expose
  // bounds only (never app names, titles, or pixels). If the platform cannot
  // enumerate other apps, an empty list leaves the work-area floor available.
  ipcMain.handle('hermes:pet-overlay:roam-environment', async (_event, probe) => {
    const petOverlayWindow = getPetOverlayWindow()

    if (!petOverlayWindow || petOverlayWindow.isDestroyed()) {
      return null
    }

    const display = screen.getDisplayMatching(petOverlayWindow.getBounds())
    const workArea = display.workArea
    const reuseCapture = Boolean(probe?.reuseCapture)
    const requestedMaxAge = Number(probe?.maxCacheAgeMs)

    const maximumCacheAgeMs = Number.isFinite(requestedMaxAge)
      ? Math.max(0, Math.min(2000, Math.round(requestedMaxAge)))
      : 750

    const displayId = String(display.id)

    const cachedWindowsAreFresh =
      reuseCapture &&
      cachedRoamWindows?.displayId === displayId &&
      Date.now() - cachedRoamWindows.capturedAt <= maximumCacheAgeMs

    const scanMode = reuseCapture
      ? probe?.scanMode === 'support'
        ? 'support'
        : 'landing'
      : probe?.scanMode === 'destination'
        ? 'destination'
        : 'landing'

    const [enumerated, visualCapture] = await Promise.all([
      cachedWindowsAreFresh ? Promise.resolve(undefined) : enumerateWindowsFrontToBack(process.pid, false),
      captureVisualLedgeBelow(
        petOverlayWindow,
        probe?.petWidth,
        reuseCapture,
        scanMode,
        probe?.petHeight,
        maximumCacheAgeMs
      )
    ])

    let windows = cachedRoamWindows?.displayId === displayId ? cachedRoamWindows.windows : []
    let windowRevision = cachedRoamWindows?.displayId === displayId ? cachedRoamWindows.revision : roamWindowRevision

    if (enumerated !== undefined) {
      windowRevision = ++roamWindowRevision

      if (enumerationFailed(enumerated)) {
        cachedRoamWindows = undefined
        windows = []
      } else {
        const right = workArea.x + workArea.width
        const bottom = workArea.y + workArea.height

        windows = enumerated
          .map(windowInfo => ({
            ...windowInfo,
            bounds: overlayWindowBoundsToDip(windowInfo.bounds, process.platform, bounds =>
              screen.screenToDipRect(null, bounds)
            )
          }))
          .filter(({ bounds, pid }) => {
            if (pid === process.pid || bounds.width <= 0 || bounds.height <= 0) {
              return false
            }

            return (
              bounds.x < right &&
              bounds.x + bounds.width > workArea.x &&
              bounds.y < bottom &&
              bounds.y + bounds.height > workArea.y
            )
          })
          .map(({ bounds }) => bounds)

        cachedRoamWindows = { capturedAt: Date.now(), displayId, revision: windowRevision, windows }
      }
    }

    return {
      sceneRevision: `${windowRevision}:${visualCapture.revision}`,
      visualLedges: visualCapture.ledges,
      windows,
      workArea
    }
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
    const x = Math.round(bounds.x)
    const y = Math.round(bounds.y)
    const width = Math.max(80, Math.round(bounds.width))
    const height = Math.max(80, Math.round(bounds.height))
    const [curW, curH] = win.getSize()
    const resizing = width !== curW || height !== curH

    if (resizing && !win.isResizable()) {
      win.setResizable(true)
    }

    if (resizing) {
      win.setBounds({ x, y, width, height })
    } else {
      // Roaming only changes position. Avoid asking Windows to renegotiate the
      // whole transparent window rectangle on every animation frame.
      win.setPosition(x, y)
    }

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
