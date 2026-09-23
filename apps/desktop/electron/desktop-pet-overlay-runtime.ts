import { pathToFileURL } from 'node:url'

import { BrowserWindow } from 'electron'

import { attachRendererConsoleCapture } from './renderer-log'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'
import { zoomWiringForWindowKind } from './zoom'

export function createDesktopPetOverlayRuntime(deps: {
  DEV_SERVER: string | undefined
  IS_MAC: boolean
  PRELOAD_PATH: string
  getMainWindow: () => BrowserWindow | null
  loadWindowUrl: (...args: any[]) => any
  rememberLog: (...args: any[]) => any
  resolveRendererIndex: () => string
  wireCommonWindowHandlers: (...args: any[]) => any
  wireWindowReveal: (...args: any[]) => any
}) {
  const {
    DEV_SERVER,
    IS_MAC,
    PRELOAD_PATH,
    getMainWindow,
    loadWindowUrl,
    rememberLog,
    resolveRendererIndex,
    wireCommonWindowHandlers,
    wireWindowReveal
  } = deps

  // The pet overlay: a single transparent, frameless, always-on-top window that
  // hosts ONLY the floating mascot. Shift-clicking the in-window pet "pops it out"
  // here so it can leave the app's bounds and stay visible while Hermes is
  // minimized (Codex-style task-completion glance). It carries no gateway
  // connection of its own — the main renderer is the single source of truth and
  // pushes pet state over IPC (hermes:pet-overlay:state); the overlay just renders
  // it. Control flows back (pop-in, composer submit) via hermes:pet-overlay:control.
  let petOverlayWindow: BrowserWindow | null = null

  function petOverlayUrl() {
    if (DEV_SERVER) {
      return `${DEV_SERVER.endsWith('/') ? DEV_SERVER.slice(0, -1) : DEV_SERVER}/?win=overlay#/`
    }

    return `${pathToFileURL(resolveRendererIndex()).toString()}?win=overlay#/`
  }

  function spawnPetOverlayWindow(bounds) {
    const win = new BrowserWindow({
      width: Math.max(80, Math.round(bounds?.width || 220)),
      height: Math.max(80, Math.round(bounds?.height || 220)),
      x: Number.isFinite(bounds?.x) ? Math.round(bounds.x) : undefined,
      y: Number.isFinite(bounds?.y) ? Math.round(bounds.y) : undefined,
      frame: false,
      transparent: true,
      resizable: false,
      movable: true,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      // Windows/Linux need this so the helper window does not get its own
      // taskbar/alt-tab entry. On macOS, cmd-tab is app-level and this can make
      // the whole app look like it vanished when the only newly-created visible
      // window is a frameless overlay. Use NSPanel + Mission Control hiding below
      // instead, leaving the main Hermes app as the Dock/cmd-tab anchor.
      skipTaskbar: !IS_MAC,
      hasShadow: false,
      alwaysOnTop: true,
      // macOS panels are non-activating helper windows and can float over full
      // screen spaces without becoming the app's main switcher window.
      type: IS_MAC ? 'panel' : undefined,
      hiddenInMissionControl: IS_MAC,
      // Non-activating: the overlay must never become the app's key/main window,
      // or it (a frameless, taskbar-skipping panel) becomes the app's switcher
      // anchor and the Hermes icon drops out of cmd/alt-tab — especially when the
      // main window is minimized. We flip this on only while the composer needs
      // the keyboard (see hermes:pet-overlay:set-focusable).
      focusable: false,
      show: false,
      // Fully transparent — the renderer paints only the sprite + bubble.
      backgroundColor: '#00000000',
      webPreferences: {
        preload: PRELOAD_PATH,
        contextIsolation: true,
        sandbox: true,
        nodeIntegration: false,
        devTools: true,
        // Keep the sprite animating + bubble updating while the main window is
        // minimized/blurred — the whole point of the overlay.
        backgroundThrottling: false
      }
    })

    // Float above other apps and follow the user across desktops so the pet is
    // always reachable. `floating` + `type: panel` is the macOS NSPanel path; the
    // more aggressive `screen-saver` level can interfere with normal app/window
    // switching semantics.
    win.setAlwaysOnTop(true, IS_MAC ? 'floating' : 'screen-saver')
    win.setHiddenInMissionControl?.(true)

    try {
      // Electron docs: macOS may transform process type on each
      // setVisibleOnAllWorkspaces() call unless skipTransformProcessType=true,
      // which briefly hides the Dock/cmd-tab presence. Keep Hermes in the normal
      // ForegroundApplication class so shift-clicking the pet never drops the app
      // out of app switchers.
      win.setVisibleOnAllWorkspaces(
        true,
        IS_MAC ? { visibleOnFullScreen: true, skipTransformProcessType: true } : undefined
      )
    } catch {
      // Not supported everywhere — best effort.
    }

    // Pet overlay opts out of global UI zoom (see zoomWiringForWindowKind): it
    // owns its window-fit + scale, and inheriting zoom would crop the sprite.
    wireCommonWindowHandlers(win, zoomWiringForWindowKind('petOverlay'))

    wireWindowReveal(win, { show: () => win.showInactive() })

    // Log-only renderer lifecycle (#81290): a dead overlay must never resurrect
    // itself over the app, but its loss belongs in desktop.log.
    installWindowRendererLifecycle(win, { kind: 'overlay', callbacks: { log: rememberLog } })

    win.on('closed', () => {
      if (petOverlayWindow === win) {
        petOverlayWindow = null
      }

      // If the overlay went away on its own (e.g. ⌘W), tell the main renderer to
      // pop the pet back in so it doesn't stay hidden. Harmless echo when we're
      // the ones who closed it (popInPet already cleared the active flag).
      const mainWindow = getMainWindow()

      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('hermes:pet-overlay:control', { type: 'pop-in' })
      }
    })

    attachRendererConsoleCapture(win, 'pet-overlay', rememberLog)
    loadWindowUrl(win, petOverlayUrl(), 'Pet overlay')

    return win
  }

  function openPetOverlay(bounds) {
    if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
      if (bounds) {
        petOverlayWindow.setBounds({
          x: Math.round(bounds.x),
          y: Math.round(bounds.y),
          width: Math.max(80, Math.round(bounds.width)),
          height: Math.max(80, Math.round(bounds.height))
        })
      }

      petOverlayWindow.showInactive()

      return petOverlayWindow
    }

    petOverlayWindow = spawnPetOverlayWindow(bounds)

    return petOverlayWindow
  }

  function closePetOverlay() {
    if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
      petOverlayWindow.close()
    }

    petOverlayWindow = null
  }

  return {
    getPetOverlayWindow: () => petOverlayWindow,
    openPetOverlay,
    closePetOverlay
  }
}
