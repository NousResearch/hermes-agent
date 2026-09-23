import fs from 'node:fs'
import path from 'node:path'

// Electron owns native window geometry, zoom persistence, icon decode, and OS
// window/permission facts. Late IPC registration stays at main's original site.
export function createDesktopNativeWindowServicesRuntime(deps: {
  APP_ICON_PATHS: string[]
  BrowserWindow: any
  DESKTOP_WINDOW_STATE_PATH: string
  IS_MAC: boolean
  app: any
  debounce: (...args: any[]) => any
  getMainWindow: () => Electron.BrowserWindow | null
  ipcMain: any
  readWindowBelow: (...args: any[]) => any
  rememberLog: (message: string) => void
  resolveAppIcon: (...args: any[]) => any
  sanitizeWindowState: (...args: any[]) => any
  systemPreferences: any
  writeFileAtomic: (...args: any[]) => any
}) {
  const {
    APP_ICON_PATHS,
    BrowserWindow,
    DESKTOP_WINDOW_STATE_PATH,
    IS_MAC,
    app,
    debounce,
    getMainWindow,
    ipcMain,
    readWindowBelow,
    rememberLog,
    resolveAppIcon,
    sanitizeWindowState,
    systemPreferences,
    writeFileAtomic
  } = deps

  // ─── Main-window geometry persistence (window-state.json) ──────────────────

  function readWindowState() {
    try {
      return sanitizeWindowState(JSON.parse(fs.readFileSync(DESKTOP_WINDOW_STATE_PATH, 'utf8')))
    } catch {
      return null
    }
  }

  // Persist the window's restored (non-maximized) bounds plus its maximized flag.
  // getNormalBounds() keeps the pre-maximize size, so un-maximizing next session
  // lands back where the user actually sized the window.
  function persistWindowState() {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed() || mainWindow.isMinimized()) {
      return
    }

    try {
      const { x, y, width, height } = mainWindow.getNormalBounds()
      fs.mkdirSync(path.dirname(DESKTOP_WINDOW_STATE_PATH), { recursive: true })
      writeFileAtomic(
        DESKTOP_WINDOW_STATE_PATH,
        JSON.stringify({ x, y, width, height, isMaximized: mainWindow.isMaximized() }, null, 2)
      )
    } catch (err) {
      rememberLog(`[window-state] persist failed: ${err?.message || err}`)
    }
  }

  // move/resize fire many times mid-drag; debounce to one write.
  const schedulePersistWindowState = debounce(persistWindowState, 250)

  // Zoom's primary store is a main-process JSON file. The renderer localStorage
  // mirror lives under Electron's cache/storage folders, which crash recovery
  // can move or recreate — wiping the zoom setting exactly when the user just
  // recovered from a crash (#56726). JSON survives; localStorage is kept as a
  // secondary mirror so pre-JSON installs migrate transparently on first read.
  const DESKTOP_ZOOM_STATE_PATH = path.join(app.getPath('userData'), 'zoom-state.json')

  function readZoomState() {
    try {
      const raw = JSON.parse(fs.readFileSync(DESKTOP_ZOOM_STATE_PATH, 'utf8'))
      const level = Number(raw?.zoomLevel)

      return Number.isFinite(level) ? level : null
    } catch {
      return null
    }
  }

  function writeZoomState(zoomLevel) {
    try {
      fs.mkdirSync(path.dirname(DESKTOP_ZOOM_STATE_PATH), { recursive: true })
      writeFileAtomic(DESKTOP_ZOOM_STATE_PATH, JSON.stringify({ zoomLevel }, null, 2))
    } catch (error) {
      rememberLog(`[zoom] json persist failed: ${error?.message || error}`)
    }
  }

  function getAppIconPath() {
    // Fail-soft: skip candidates that exist but don't decode (truncated PNG in a
    // packaged app.asar previously crashed createWindow mid-session). Missing
    // every candidate is fine — the window then uses the platform default icon.
    try {
      return resolveAppIcon(APP_ICON_PATHS)
    } catch {
      return undefined
    }
  }

  function registerNativeWindowServicesIpc() {
    ipcMain.handle('hermes:requestMicrophoneAccess', async () => {
      if (!IS_MAC || typeof systemPreferences.askForMediaAccess !== 'function') {
        return true
      }

      return systemPreferences.askForMediaAccess('microphone')
    })

    // read_window_below tool: which OS window is directly underneath this one.
    // Metadata only (app, title, bounds) — never pixels. On macOS, other apps'
    // window titles are gated behind the Screen Recording permission; pass titles
    // through only when it is ALREADY granted, and never prompt for it here.
    ipcMain.handle('hermes:window:readBelow', async event => {
      const win = BrowserWindow.fromWebContents(event.sender)

      if (!win || win.isDestroyed()) {
        return null
      }

      const titlesAvailable = IS_MAC ? systemPreferences.getMediaAccessStatus?.('screen') === 'granted' : true

      const [x, y] = win.getPosition()
      const [width, height] = win.getSize()

      return readWindowBelow(process.pid, { x, y, width, height }, titlesAvailable)
    })

  }

  return {
    readWindowState,
    schedulePersistWindowState,
    readZoomState,
    writeZoomState,
    getAppIconPath,
    registerNativeWindowServicesIpc
  }
}
