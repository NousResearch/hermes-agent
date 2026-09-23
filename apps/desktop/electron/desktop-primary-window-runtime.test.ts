import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import { createDesktopPrimaryWindowRuntime } from './desktop-primary-window-runtime'

class FakeWindow extends EventEmitter {
  readonly options: Record<string, unknown>
  readonly webContents = Object.assign(new EventEmitter(), {
    id: 11,
    reloads: 0,
    reload() {
      this.reloads += 1
    }
  })

  constructor(options: Record<string, unknown>) {
    super()
    this.options = options
  }

  maximize() {}
  isDestroyed() {
    return false
  }
}

function fixture(defaultRoute: null | { connectionId: string; profile: string } = null) {
  let mainWindow: FakeWindow | null = null
  let rendererReadyClears = 0
  const events: string[] = []
  const markerArgs: unknown[] = []
  const routes: unknown[] = []
  const lifecycle: Array<{ callbacks: Record<string, (...args: any[]) => void> }> = []
  const persist = Object.assign(() => events.push('persist'), { flush: () => events.push('flush') })

  const sandboxState = {
    fallbackActive: false,
    fallbackSticky: false,
    fallbackReason: 'boot-loop',
    noSandboxRelaunchAttempted: false
  }

  const deps = {
    app: { getPath: () => '/tmp/desktop', getVersion: () => '1.0', dock: { setIcon: () => {} } },
    BrowserWindow: FakeWindow,
    DEV_SERVER: 'http://127.0.0.1:5174',
    IS_MAC: false,
    IS_WINDOWS: false,
    PRELOAD_PATH: '/tmp/preload.js',
    RENDERER_RELOAD_MAX: 2,
    RENDERER_RELOAD_WINDOW_MS: 1000,
    WINDOW_BUTTON_POSITION: { x: 1, y: 1 },
    WINDOW_MIN_HEIGHT: 400,
    WINDOW_MIN_WIDTH: 600,
    alreadyHasNoSandbox: () => false,
    appearance: {
      getTitleBarOverlayOptions: () => ({}),
      chatWindowSurfaceOptions: () => ({}),
      registerChatWindow: () => events.push('appearance'),
      installNativeThemeListener: () => events.push('native-theme')
    },
    attachRendererConsoleCapture: () => events.push('console'),
    backendShutdown: { hasStarted: () => false },
    bindGeometryPersistence: () => events.push('geometry'),
    bindWindowChromeEvents: () => events.push('chrome'),
    buildNoSandboxRelaunchArgs: () => [],
    chatWindowWebPreferences: () => ({ preload: '/tmp/preload.js' }),
    clearRendererReadyForDeepLink: () => {
      rendererReadyClears += 1
    },
    closePetOverlay: () => events.push('pet-close'),
    computeWindowOptions: () => ({ width: 800, height: 600 }),
    connectDesktopProfileRoute: () => {
      events.push('connect-route')

      return Promise.resolve()
    },
    desktopProfilePreferences: { getDefault: () => defaultRoute },
    exitAfterBackendShutdown: () => Promise.resolve(),
    fallbackMarker: () => ({}),
    firstRunBoot: { broadcastBootProgress: () => events.push('boot-progress') },
    getAppIconPath: () => '/tmp/icon.png',
    getIsQuittingForHandoff: () => false,
    getMainWindow: () => mainWindow,
    getStreamThrottle: () => ({ register: () => events.push('throttle') }),
    installWindowRendererLifecycle: (_window: FakeWindow, options: any) => lifecycle.push(options),
    introRevealController: { destroy: () => events.push('intro-close') },
    loadRendererLoadErrorPage: () => Promise.resolve(),
    loadWindowUrl: () => events.push('load-url'),
    markerAfterSuccessfulBoot: (options: unknown) => {
      markerArgs.push(options)

      return options
    },
    minimizeToTray: { registerWindow: () => events.push('tray') },
    notifyLauncherWindowRevealed: () => events.push('launcher-revealed'),
    readWindowState: () => null,
    recordWindowConnectionRoute: (_sender: unknown, route: unknown) => routes.push(route),
    rememberLog: () => {},
    rendererReloadTimesRef: { current: [] },
    resolveRendererIndex: () => '/tmp/index.html',
    resolveRendererIndexWithMissing: () => ({ index: '/tmp/index.html', missing: [] }),
    sandboxState,
    schedulePersistWindowState: persist,
    screen: { getAllDisplays: () => [] },
    sendWindowStateChanged: () => events.push('state-changed'),
    setMainWindow: (window: FakeWindow | null) => {
      mainWindow = window
    },
    shouldRelaunchForRendererSandboxCrashLoop: (_options: unknown) => false,
    startHermes: () => {
      events.push('start-hermes')

      return Promise.resolve()
    },
    wakeIndicatorController: { close: () => events.push('wake-close') },
    wireCommonWindowHandlers: () => events.push('common'),
    wireWindowReveal: (_window: FakeWindow, { onRevealed }: { onRevealed: () => void }) => ({
      reveal: onRevealed
    }),
    writeSandboxMarker: () => {},
    zoomWiringForWindowKind: () => ({})
  }

  return {
    deps,
    events,
    lifecycle,
    markerArgs,
    routes,
    getMainWindow: () => mainWindow,
    getRendererReadyClears: () => rendererReadyClears,
    replaceMainWindow: (window: FakeWindow) => {
      mainWindow = window
    }
  }
}

test('primary window wires the selected route and starts its backend alongside renderer loading', () => {
  const f = fixture({ connectionId: 'remote-1', profile: 'alpha' })
  let throttleReady = false

  f.deps.getStreamThrottle = () => {
    assert.equal(throttleReady, true)

    return { register: () => f.events.push('throttle') }
  }

  const runtime = createDesktopPrimaryWindowRuntime(f.deps)
  throttleReady = true
  runtime.createWindow()

  const window = f.getMainWindow()
  assert.ok(window)
  assert.equal(window.options.show, false)
  assert.deepEqual(f.routes, [{ connectionId: 'remote-1', profile: 'alpha', registryScoped: true }])
  assert.ok(f.events.indexOf('load-url') < f.events.indexOf('connect-route'))
  assert.equal(f.events.includes('start-hermes'), false)
  assert.equal(f.lifecycle.length, 1)

  window.webContents.emit('did-finish-load')
  assert.ok(f.events.includes('boot-progress'))
  assert.ok(f.events.includes('state-changed'))
})

test('late primary-window callbacks read the live window and a stale close cannot clear its replacement', () => {
  const f = fixture()
  createDesktopPrimaryWindowRuntime(f.deps).createWindow()

  const original = f.getMainWindow()
  assert.ok(original)
  assert.ok(f.events.includes('start-hermes'))

  const replacement = new FakeWindow({})
  f.replaceMainWindow(replacement)
  f.lifecycle[0].callbacks.reload()
  assert.equal(original.webContents.reloads, 0)
  assert.equal(replacement.webContents.reloads, 1)

  original.emit('closed')
  assert.equal(f.getMainWindow(), replacement)
  assert.equal(f.getRendererReadyClears(), 0)
  assert.ok(f.events.includes('pet-close'))
  assert.ok(f.events.includes('wake-close'))
  assert.ok(f.events.includes('intro-close'))
})

test('reveal and crash callbacks read sandbox state at event time', () => {
  const f = fixture()

  let reveal: () => void = () => {
    throw new Error('reveal was not registered')
  }

  const decisions: unknown[] = []

  f.deps.IS_WINDOWS = true

  f.deps.wireWindowReveal = (_window, { onRevealed }) => {
    reveal = onRevealed

    return { reveal: onRevealed }
  }

  f.deps.shouldRelaunchForRendererSandboxCrashLoop = options => {
    decisions.push(options)

    return false
  }

  createDesktopPrimaryWindowRuntime(f.deps).createWindow()
  f.deps.sandboxState.fallbackSticky = true
  f.deps.sandboxState.fallbackReason = 'renderer-crash-loop'
  f.deps.sandboxState.fallbackActive = true
  f.deps.sandboxState.noSandboxRelaunchAttempted = true
  reveal()
  f.lifecycle[0].callbacks.onCrashLoopSuppressed({ reason: 'crashed', exitCode: 123 })

  assert.deepEqual(f.markerArgs, [{ fallbackActive: true, reason: 'renderer-crash-loop', appVersion: '1.0' }])
  assert.deepEqual(decisions, [{ reason: 'crashed', exitCode: 123, alreadyNoSandbox: true, relaunchAttempted: true }])
})
