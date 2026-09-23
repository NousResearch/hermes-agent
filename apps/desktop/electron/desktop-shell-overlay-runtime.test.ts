import { beforeEach, expect, test, vi } from 'vitest'

const native = vi.hoisted(() => {
  let nextId = 1

  class FakeWindow {
    static windows: FakeWindow[] = []
    static getAllWindows = () => FakeWindow.windows

    readonly listeners = new Map<string, (() => void)[]>()
    readonly webListeners = new Map<string, (() => void)[]>()
    readonly webContents = {
      id: nextId++,
      getZoomFactor: () => 1,
      on: (event: string, listener: () => void) => {
        this.webListeners.set(event, [...(this.webListeners.get(event) ?? []), listener])
      },
      send: vi.fn()
    }
    readonly focus = vi.fn()
    readonly setBounds = vi.fn()
    readonly setAlwaysOnTop = vi.fn()
    readonly setHiddenInMissionControl = vi.fn()
    readonly setVisibleOnAllWorkspaces = vi.fn()
    destroyed = false
    visible = false

    constructor(readonly options: Record<string, unknown> = {}) {
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

    emitWeb(event: string) {
      for (const listener of this.webListeners.get(event) ?? []) {
        listener()
      }
    }

    isDestroyed() {
      return this.destroyed
    }

    isVisible() {
      return this.visible
    }

    hide() {
      this.visible = false
    }

    show() {
      this.visible = true
    }

    destroy() {
      this.destroyed = true
      this.emit('closed')
    }

    close() {
      this.destroy()
    }

    removeAllListeners(event: string) {
      this.listeners.delete(event)
    }

    getBounds() {
      return { height: 200, width: 500, x: 20, y: 30 }
    }

    getNormalBounds() {
      return this.getBounds()
    }
  }

  return {
    FakeWindow,
    app: { getPath: () => 'C:/test-overlay-user-data' },
    globalShortcut: {
      isRegistered: vi.fn(() => false),
      register: vi.fn((_accelerator: string, _callback: () => void) => true),
      unregister: vi.fn()
    },
    screen: {
      getAllDisplays: () => [{ workArea: { height: 1000, width: 1600, x: 0, y: 0 } }],
      getCursorScreenPoint: () => ({ x: 400, y: 300 }),
      getDisplayNearestPoint: () => ({ workArea: { height: 1000, width: 1600, x: 0, y: 0 } })
    },
    systemPreferences: { getMediaAccessStatus: () => 'granted' }
  }
})

vi.mock('electron', () => ({ ...native, BrowserWindow: native.FakeWindow }))
vi.mock('./hud-close', () => ({
  requestHudClose: (window: InstanceType<typeof native.FakeWindow>) => window.destroy()
}))
vi.mock('./hud-game-overlay', () => ({ startHudGameOverlayWatch: () => () => undefined }))
vi.mock('./hud-overlay', () => ({
  applyHudElectronOverlay: (window: InstanceType<typeof native.FakeWindow>) => window.setAlwaysOnTop(true),
  promoteHudOverlay: vi.fn()
}))
vi.mock('./renderer-log', () => ({ attachRendererConsoleCapture: vi.fn() }))
vi.mock('./window-renderer-lifecycle', () => ({ installWindowRendererLifecycle: vi.fn() }))
vi.mock('./window-below', () => ({ enumerateWindowsFrontToBack: vi.fn(), enumerationFailed: () => false }))

import { createDesktopShellOverlayRuntime } from './desktop-shell-overlay-runtime'

function makeRuntime() {
  let mainWindow: InstanceType<typeof native.FakeWindow> | null = null
  let streamRegister = vi.fn()
  const loadWindowUrl = vi.fn()
  const focusWindow = vi.fn()
  const wireCommonWindowHandlers = vi.fn()

  const runtime = createDesktopShellOverlayRuntime({
    DEV_SERVER: 'http://127.0.0.1:5174',
    HUD_WINDOW_TITLE: 'Hermes HUD',
    IS_MAC: false,
    PRELOAD_PATH: 'C:/desktop/preload.js',
    bindGeometryPersistence: vi.fn(),
    focusWindow,
    getMainWindow: () => mainWindow as never,
    getStreamThrottle: () => ({ register: streamRegister }),
    loadWindowUrl,
    rememberLog: vi.fn(),
    resolveRendererIndex: () => 'C:/desktop/index.html',
    wireCommonWindowHandlers,
    wireWindowReveal: (window, options) => {
      options.show?.()
      options.onRevealed?.()

      return window
    },
    writeFileAtomic: vi.fn()
  })

  return {
    focusWindow,
    loadWindowUrl,
    runtime,
    setMainWindow: (window: InstanceType<typeof native.FakeWindow>) => {
      mainWindow = window
    },
    setStreamRegister: (register: any) => {
      streamRegister = register
    },
    wireCommonWindowHandlers
  }
}

beforeEach(() => {
  native.FakeWindow.windows.length = 0
  native.globalShortcut.register.mockClear()
  native.globalShortcut.unregister.mockClear()
})

test('HUD profile respawn keeps the replacement authoritative and hands its session back on close', () => {
  const { focusWindow, loadWindowUrl, runtime, setMainWindow, setStreamRegister, wireCommonWindowHandlers } =
    makeRuntime()

  const main = new native.FakeWindow()
  const laterRegister = vi.fn()
  setMainWindow(main)
  setStreamRegister(laterRegister)

  const first = runtime.openHudWindow('session-a', 'work')
  const second = runtime.openHudWindow('session-b', 'review')

  expect(first).not.toBe(second)
  expect(first?.isDestroyed()).toBe(true)
  expect(runtime.getHudWindow()).toBe(second)
  expect(focusWindow).not.toHaveBeenCalled()
  expect(loadWindowUrl.mock.lastCall?.[1]).toContain('profile=review')
  expect(laterRegister).toHaveBeenCalledWith(first)
  expect(laterRegister).toHaveBeenCalledWith(second)
  expect(wireCommonWindowHandlers).toHaveBeenCalledWith(second, { zoom: true })
  expect(main.webContents.send).not.toHaveBeenCalledWith('hermes:hud:changed', {
    open: false,
    sessionId: 'session-b'
  })

  runtime.setHudSessionId('session-live')
  runtime.closeHudWindow()

  expect(runtime.getHudWindow()).toBeNull()
  expect(focusWindow).toHaveBeenCalledWith(main)
  expect(main.webContents.send).toHaveBeenCalledWith('hermes:hud:changed', {
    open: false,
    sessionId: 'session-live'
  })
})

test('quick entry replays late state and re-summons without creating a second window', () => {
  const { runtime } = makeRuntime()
  const state = { connected: true, sessions: ['session-a'] }
  runtime.pushQuickEntryState(state)
  runtime.applyQuickEntrySettings(runtime.readQuickEntrySettings())
  const summon = native.globalShortcut.register.mock.calls.find(call => call[0] === 'CommandOrControl+Shift+Space')?.[1]

  expect(summon).toBeTypeOf('function')

  summon!()
  const window = native.FakeWindow.windows[0]
  window.emitWeb('did-finish-load')

  expect(window.webContents.send).toHaveBeenCalledWith('hermes:quick-entry:state', state)
  const nextState = { connected: false, sessions: [] }
  runtime.pushQuickEntryState(nextState)
  expect(window.webContents.send).toHaveBeenCalledWith('hermes:quick-entry:state', nextState)
  summon!()
  expect(window.isVisible()).toBe(false)
  summon!()
  expect(native.FakeWindow.windows).toHaveLength(1)
  expect(window.webContents.send).toHaveBeenCalledWith('hermes:quick-entry:shown')
})

test('quit destroys the HUD without restoring the app and releases quick entry shortcut', () => {
  const { focusWindow, runtime, setMainWindow } = makeRuntime()
  const main = new native.FakeWindow()
  setMainWindow(main)
  runtime.openHudWindow('session-a', 'work')
  runtime.applyQuickEntrySettings(runtime.readQuickEntrySettings())

  runtime.closeHudWindowForQuit()
  runtime.closeQuickEntryWindow()

  expect(runtime.getHudWindow()).toBeNull()
  expect(focusWindow).not.toHaveBeenCalled()
  expect(main.webContents.send).not.toHaveBeenCalledWith('hermes:hud:changed', {
    open: false,
    sessionId: 'session-a'
  })
  expect(native.globalShortcut.unregister).toHaveBeenCalledWith('CommandOrControl+Shift+Space')
})
