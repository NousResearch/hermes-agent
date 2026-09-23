import path from 'node:path'

import { app, BrowserWindow, screen } from 'electron'

import {
  BROWSER_WINDOW_HEIGHT,
  BROWSER_WINDOW_MIN_HEIGHT,
  BROWSER_WINDOW_MIN_WIDTH,
  BROWSER_WINDOW_WIDTH,
  buildBrowserWindowUrl
} from './browser-windows'
import { type DesktopProfileRoute, resolveDesktopWindowLaunch } from './desktop-profile'
import { createMinimizeToTray } from './minimize-to-tray'
import { attachRendererConsoleCapture } from './renderer-log'
import {
  buildInstanceWindowUrl,
  buildSessionWindowUrl,
  chatWindowWebPreferences,
  createSessionWindowRegistry,
  instanceWindowBounds,
  SESSION_WINDOW_MIN_HEIGHT,
  SESSION_WINDOW_MIN_WIDTH
} from './session-windows'
import { bindWindowChromeEvents } from './window-chrome-events'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'
import { computeWindowOptions, MIN_HEIGHT as WINDOW_MIN_HEIGHT, MIN_WIDTH as WINDOW_MIN_WIDTH } from './window-state'
import { zoomWiringForWindowKind } from './zoom'

// Late-created main-process services remain live through getters. In particular,
// the stream throttle and connection-route registry are initialized after the
// window runtime, exactly as they were in main.ts.
export function createDesktopSecondaryWindowRuntime(deps: {
  DEV_SERVER: string | undefined
  IS_MAC: boolean
  PRELOAD_PATH: string
  RENDERER_RELOAD_MAX: number
  RENDERER_RELOAD_WINDOW_MS: number
  WINDOW_BUTTON_POSITION: Electron.Point
  appearance: any
  createWindow: () => any
  ensureMainWindow: (...args: any[]) => any
  getAppIconPath: () => any
  getIsQuittingForHandoff: () => boolean
  getMainWindow: () => BrowserWindow | null
  getStreamThrottle: () => any
  getWindowConnectionRoutes: () => any
  loadWindowUrl: (...args: any[]) => any
  primaryProfileKey: () => string
  readWindowState: () => any
  recordWindowConnectionRoute: (...args: any[]) => any
  rememberLog: (...args: any[]) => any
  rendererReloadTimesRef: { current: number[] }
  resolveRendererIndex: () => string
  sendWindowStateChanged: (...args: any[]) => any
  validateDesktopProfileRoute: (route: DesktopProfileRoute) => any
  wireCommonWindowHandlers: (...args: any[]) => any
  wireWindowReveal: (...args: any[]) => any
}) {
  const {
    DEV_SERVER,
    IS_MAC,
    PRELOAD_PATH,
    RENDERER_RELOAD_MAX,
    RENDERER_RELOAD_WINDOW_MS,
    WINDOW_BUTTON_POSITION,
    appearance,
    createWindow,
    ensureMainWindow,
    getAppIconPath,
    getIsQuittingForHandoff,
    getMainWindow,
    getStreamThrottle,
    getWindowConnectionRoutes,
    loadWindowUrl,
    primaryProfileKey,
    readWindowState,
    recordWindowConnectionRoute,
    rememberLog,
    rendererReloadTimesRef,
    resolveRendererIndex,
    sendWindowStateChanged,
    validateDesktopProfileRoute,
    wireCommonWindowHandlers,
    wireWindowReveal
  } = deps

  // Secondary "session windows" — one extra OS window per chat so a user can
  // work with multiple chats side by side. The registry guarantees one window
  // per sessionId (re-opening focuses the existing window) and self-cleans on
  // close. The primary mainWindow is never tracked here. Pure logic + the URL
  // builder live in session-windows.ts so they stay unit-testable.
  const sessionWindows = createSessionWindowRegistry()

  const minimizeToTray = createMinimizeToTray({
    preferencesPath: path.join(app.getPath('userData'), 'minimize-to-tray.json'),
    getIconPath: getAppIconPath,
    restoreMainWindow: () => ensureMainWindow(getMainWindow(), { isReady: app.isReady(), createWindow, focusWindow }),
    isQuittingForHandoff: () => getIsQuittingForHandoff(),
    log: rememberLog
  })

  function focusWindow(win) {
    if (!win || win.isDestroyed()) {
      return
    }

    if (win.isMinimized()) {
      win.restore()
    }

    if (!win.isVisible()) {
      win.show()
    }

    win.focus()
  }

  function spawnSecondaryWindow({
    sessionId,
    profile,
    watch
  }: { sessionId?: string; profile?: null | string; watch?: boolean } = {}) {
    const icon = getAppIconPath()

    const win = new BrowserWindow({
      width: SESSION_WINDOW_MIN_WIDTH,
      height: SESSION_WINDOW_MIN_HEIGHT,
      minWidth: SESSION_WINDOW_MIN_WIDTH,
      minHeight: SESSION_WINDOW_MIN_HEIGHT,
      title: 'Hermes',
      titleBarStyle: 'hidden',
      titleBarOverlay: appearance.getTitleBarOverlayOptions(),
      trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : undefined,
      ...appearance.chatWindowSurfaceOptions(),
      icon,
      // Don't show until the renderer's first themed paint is ready. macOS
      // `vibrancy` ignores `backgroundColor` and paints a translucent OS
      // material (which follows the OS appearance, not the app theme), so a
      // dark-themed app on a light-mode Mac flashes white until the renderer
      // covers it. ready-to-show fires after the boot-time paint in
      // themes/context.tsx, so the window appears already themed.
      show: false,
      webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
    })

    // Chat-surface registration: applyWindowTranslucency swaps this window's
    // backing between opaque-themed and alpha-0 when glass toggles.
    minimizeToTray.registerWindow(win)
    appearance.registerChatWindow(win)

    if (IS_MAC) {
      win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
    }

    wireWindowReveal(win)

    bindWindowChromeEvents(win, sendWindowStateChanged)

    getStreamThrottle().register(win)
    wireCommonWindowHandlers(win, zoomWiringForWindowKind('chat'))
    attachRendererConsoleCapture(win, 'session-window', rememberLog)

    // Renderer lifecycle diagnostics + recovery (#81290): a dead session-window
    // renderer used to log nothing and stay black; now it logs with its window
    // kind and reloads under the shared crash-loop budget, exactly like the
    // primary window, without touching any other window.
    installWindowRendererLifecycle(win, {
      kind: 'secondary',
      callbacks: {
        log: rememberLog,
        reload: () => {
          win.webContents.reload()
        }
      },
      reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
      reloadMax: RENDERER_RELOAD_MAX,
      recentReloadTimesRef: rendererReloadTimesRef
    })

    loadWindowUrl(
      win,
      buildSessionWindowUrl(sessionId, {
        devServer: DEV_SERVER,
        profile,
        rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex(),
        watch
      }),
      'Session window'
    )

    return win
  }

  // Open (or focus) a standalone window for a single chat session.
  function createSessionWindow(sessionId, { profile = null, watch = false } = {}) {
    return sessionWindows.openOrFocus(sessionId, () => spawnSecondaryWindow({ sessionId, profile, watch }))
  }

  // Popped-out in-app Browser: same webview + address bar as a docked Browser
  // tab, in its own OS window. One window per tab id (re-open focuses); closing
  // it tells the other renderers so they can dock the tab again.
  const browserWindows = createSessionWindowRegistry()

  function notifyBrowserPopoutClosed(tabId) {
    if (typeof tabId !== 'string' || !tabId) {
      return
    }

    for (const other of BrowserWindow.getAllWindows()) {
      if (!other.isDestroyed()) {
        other.webContents.send('hermes:browser-popout:closed', tabId)
      }
    }
  }

  function spawnBrowserWindow(tabId) {
    const icon = getAppIconPath()

    const win = new BrowserWindow({
      width: BROWSER_WINDOW_WIDTH,
      height: BROWSER_WINDOW_HEIGHT,
      minWidth: BROWSER_WINDOW_MIN_WIDTH,
      minHeight: BROWSER_WINDOW_MIN_HEIGHT,
      title: 'Hermes',
      titleBarStyle: 'hidden',
      titleBarOverlay: appearance.getTitleBarOverlayOptions(),
      trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : undefined,
      ...appearance.chatWindowSurfaceOptions(),
      icon,
      show: false,
      webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
    })

    appearance.registerChatWindow(win)

    if (IS_MAC) {
      win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
    }

    wireWindowReveal(win)

    bindWindowChromeEvents(win, sendWindowStateChanged)

    getStreamThrottle().register(win)
    wireCommonWindowHandlers(win, zoomWiringForWindowKind('chat'))
    attachRendererConsoleCapture(win, 'browser-window', rememberLog)

    installWindowRendererLifecycle(win, {
      kind: 'browser',
      callbacks: {
        log: rememberLog,
        reload: () => {
          win.webContents.reload()
        }
      },
      reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
      reloadMax: RENDERER_RELOAD_MAX,
      recentReloadTimesRef: rendererReloadTimesRef
    })

    minimizeToTray.registerWindow(win)
    win.on('closed', () => notifyBrowserPopoutClosed(tabId))

    loadWindowUrl(
      win,
      buildBrowserWindowUrl(tabId, {
        devServer: DEV_SERVER,
        rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex()
      }),
      'Browser window'
    )

    return win
  }

  function createBrowserWindow(tabId) {
    return browserWindows.openOrFocus(tabId, () => spawnBrowserWindow(tabId))
  }

  // Additional full "instance" windows — peers of the primary that render the
  // COMPLETE app (sidebar, routing, its own draft) against the shared backend, so
  // a user can run multiple GUI windows at once (⌘⇧N / the "New Window" palette
  // command). Unlike the compact session windows they carry no `?win` flag; a
  // separate `peer=1` marker prevents them from replaying app-launch source
  // restoration after joining that shared backend. The primary mainWindow stays
  // the notification / deep-link / pet-overlay anchor and
  // is NOT tracked here. The set holds a strong reference so an open peer isn't
  // garbage-collected, and drops it on close.
  const instanceWindows = new Set<any>()

  // Cascade a new instance off whichever window spawned it so it doesn't land
  // exactly on top of its source. Falls back to the persisted primary geometry
  // when there's no live source window (e.g. all windows closed on macOS). The
  // pure cascade math lives in session-windows.ts (instanceWindowBounds).
  function nextInstanceBounds(source: BrowserWindow | null = BrowserWindow.getFocusedWindow() || getMainWindow()) {
    const displays = screen.getAllDisplays()
    const fallback = computeWindowOptions(readWindowState(), displays)
    const base = source && !source.isDestroyed() ? source.getBounds() : null

    return instanceWindowBounds(base, fallback, displays)
  }

  // Open a new full-chrome instance window. Mirrors createWindow()'s window
  // options (shared chatWindowWebPreferences + streamThrottle registration so a
  // streamed answer never stalls in the background) but is a peer, not the
  // primary: it never overwrites mainWindow or re-homes the source. Its renderer
  // joins the requested pooled backend through its own connection/profile route.
  function createInstanceWindow(
    options?: DesktopProfileRoute,
    source: BrowserWindow | null = BrowserWindow.getFocusedWindow() || getMainWindow()
  ) {
    const route = resolveDesktopWindowLaunch(
      options,
      source && !source.isDestroyed() ? getWindowConnectionRoutes().get(source.webContents.id) : null,
      { connectionId: null, profile: primaryProfileKey() }
    )

    validateDesktopProfileRoute(route)
    const icon = getAppIconPath()

    const win = new BrowserWindow({
      ...nextInstanceBounds(source),
      minWidth: WINDOW_MIN_WIDTH,
      minHeight: WINDOW_MIN_HEIGHT,
      title: 'Hermes',
      titleBarStyle: 'hidden',
      titleBarOverlay: appearance.getTitleBarOverlayOptions(),
      trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : undefined,
      ...appearance.chatWindowSurfaceOptions(),
      icon,
      show: false,
      webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
    })

    instanceWindows.add(win)
    minimizeToTray.registerWindow(win)
    recordWindowConnectionRoute(win.webContents, { ...route, registryScoped: route.connectionId !== null })

    // Chat-surface registration: see applyWindowTranslucency.
    appearance.registerChatWindow(win)

    if (IS_MAC) {
      win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
    }

    wireWindowReveal(win)

    bindWindowChromeEvents(win, sendWindowStateChanged)

    getStreamThrottle().register(win)
    wireCommonWindowHandlers(win, zoomWiringForWindowKind('chat'))

    // Renderer lifecycle diagnostics + recovery (#81290), same policy as the
    // primary and session windows: a crashed instance renderer logs with its
    // window kind and reloads under the shared crash-loop budget.
    installWindowRendererLifecycle(win, {
      kind: 'instance',
      callbacks: {
        log: rememberLog,
        reload: () => {
          win.webContents.reload()
        }
      },
      reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
      reloadMax: RENDERER_RELOAD_MAX,
      recentReloadTimesRef: rendererReloadTimesRef
    })

    win.on('closed', () => {
      instanceWindows.delete(win)
    })

    attachRendererConsoleCapture(win, 'instance', rememberLog)
    loadWindowUrl(
      win,
      buildInstanceWindowUrl({
        ...route,
        devServer: DEV_SERVER,
        rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex()
      }),
      'Instance window'
    )

    return win
  }

  return { minimizeToTray, focusWindow, createSessionWindow, createBrowserWindow, createInstanceWindow }
}
