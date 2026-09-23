import { beforeEach, expect, test, vi } from 'vitest'

const native = vi.hoisted(() => {
  let nextId = 1

  class FakeWindow {
    static windows: FakeWindow[] = []
    static getAllWindows = () => FakeWindow.windows
    static getFocusedWindow = () => null

    readonly webContents = { id: nextId++, reload: vi.fn(), send: vi.fn() }
    readonly listeners = new Map<string, (() => void)[]>()
    readonly focus = vi.fn()
    readonly restore = vi.fn()
    readonly show = vi.fn()
    readonly setWindowButtonPosition = vi.fn()
    destroyed = false

    constructor(readonly options: Record<string, unknown>) {
      FakeWindow.windows.push(this)
    }

    on(event: string, listener: () => void) {
      this.listeners.set(event, [...(this.listeners.get(event) ?? []), listener])

      return this
    }

    once(event: string, listener: () => void) {
      return this.on(event, listener)
    }

    emit(event: string) {
      for (const listener of this.listeners.get(event) ?? []) {
        listener()
      }
    }

    isDestroyed() {
      return this.destroyed
    }

    isMinimized() {
      return false
    }

    isVisible() {
      return true
    }

    getBounds() {
      return { height: 620, width: 420, x: 20, y: 30 }
    }
  }

  return {
    app: { getPath: () => 'C:/test-user-data', isReady: () => true },
    FakeWindow,
    screen: {
      getAllDisplays: () => [
        { bounds: { height: 1000, width: 1600, x: 0, y: 0 }, workArea: { height: 1000, width: 1600, x: 0, y: 0 } }
      ]
    }
  }
})

const tray = vi.hoisted(() => ({ registerWindow: vi.fn() }))

vi.mock('electron', () => ({ app: native.app, BrowserWindow: native.FakeWindow, screen: native.screen }))
vi.mock('./minimize-to-tray', () => ({ createMinimizeToTray: () => tray }))
vi.mock('./renderer-log', () => ({ attachRendererConsoleCapture: vi.fn() }))
vi.mock('./window-chrome-events', () => ({ bindWindowChromeEvents: vi.fn() }))
vi.mock('./window-renderer-lifecycle', () => ({ installWindowRendererLifecycle: vi.fn() }))

import { createDesktopSecondaryWindowRuntime } from './desktop-secondary-window-runtime'

function makeRuntime() {
  const loadWindowUrl = vi.fn()
  const register = vi.fn()
  const recordWindowConnectionRoute = vi.fn()
  const routeGet = vi.fn(() => ({ connectionId: 'registry-a', profile: 'work', registryScoped: true }))

  const appearance = {
    chatWindowSurfaceOptions: () => ({}),
    getTitleBarOverlayOptions: () => ({}),
    registerChatWindow: vi.fn()
  }

  const runtime = createDesktopSecondaryWindowRuntime({
    DEV_SERVER: 'http://127.0.0.1:5174',
    IS_MAC: false,
    PRELOAD_PATH: 'C:/desktop/preload.js',
    RENDERER_RELOAD_MAX: 3,
    RENDERER_RELOAD_WINDOW_MS: 60_000,
    WINDOW_BUTTON_POSITION: { x: 0, y: 0 },
    appearance,
    createWindow: vi.fn(),
    ensureMainWindow: vi.fn(),
    getAppIconPath: () => 'C:/desktop/icon.png',
    getIsQuittingForHandoff: () => false,
    getMainWindow: () => null,
    getStreamThrottle: () => ({ register }),
    getWindowConnectionRoutes: () => ({ get: routeGet }),
    loadWindowUrl,
    primaryProfileKey: () => 'default',
    readWindowState: () => ({ height: 800, isMaximized: false, width: 1220 }),
    recordWindowConnectionRoute,
    rememberLog: vi.fn(),
    rendererReloadTimesRef: { current: [] },
    resolveRendererIndex: () => 'C:/desktop/index.html',
    sendWindowStateChanged: vi.fn(),
    validateDesktopProfileRoute: vi.fn(),
    wireCommonWindowHandlers: vi.fn(),
    wireWindowReveal: vi.fn()
  })

  return { appearance, loadWindowUrl, recordWindowConnectionRoute, register, routeGet, runtime }
}

beforeEach(() => {
  native.FakeWindow.windows.length = 0
  tray.registerWindow.mockClear()
})

test('reopening a session focuses its window and retains the owning profile route', () => {
  const { appearance, loadWindowUrl, register, runtime } = makeRuntime()
  const first = runtime.createSessionWindow('session-1', { profile: 'work', watch: true })
  const second = runtime.createSessionWindow('session-1', { profile: 'work', watch: true })

  expect(second).toBe(first)
  expect(native.FakeWindow.windows).toHaveLength(1)
  expect(first?.focus).toHaveBeenCalledTimes(1)
  expect(loadWindowUrl).toHaveBeenCalledWith(
    first,
    'http://127.0.0.1:5174/?win=secondary&watch=1&profile=work#/session-1',
    'Session window'
  )
  expect(appearance.registerChatWindow).toHaveBeenCalledWith(first)
  expect(register).toHaveBeenCalledWith(first)
})

test('closing a browser popout notifies the other live renderer to dock its tab', () => {
  const { runtime } = makeRuntime()
  const session = runtime.createSessionWindow('session-1')!
  const browser = runtime.createBrowserWindow('tab-1')!

  browser.destroyed = true
  browser.emit('closed')

  expect(session.webContents.send).toHaveBeenCalledWith('hermes:browser-popout:closed', 'tab-1')
  expect(browser.webContents.send).not.toHaveBeenCalled()
})

test('a peer window inherits the source connection route and registers its own scope', () => {
  const { loadWindowUrl, recordWindowConnectionRoute, routeGet, runtime } = makeRuntime()
  const source = runtime.createSessionWindow('session-1')!
  const peer = runtime.createInstanceWindow(undefined, source as never)

  expect(routeGet).toHaveBeenCalledWith(source.webContents.id)
  expect(recordWindowConnectionRoute).toHaveBeenCalledWith(peer.webContents, {
    connectionId: 'registry-a',
    profile: 'work',
    profileWindow: false,
    registryScoped: true
  })
  expect(loadWindowUrl.mock.lastCall?.[1]).toContain('peer=1')
})
