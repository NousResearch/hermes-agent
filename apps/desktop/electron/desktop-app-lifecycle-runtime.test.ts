import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test, vi } from 'vitest'

import { createDesktopAppLifecycleRuntime } from './desktop-app-lifecycle-runtime'

function setup(options: { lock?: boolean } = {}) {
  const events = new EventEmitter()
  const handlers = new Map<string, () => unknown>()
  const exit = vi.fn()
  const quit = vi.fn()
  const calls: string[] = []
  const app = Object.assign(events, {
    requestSingleInstanceLock: () => options.lock ?? true,
    exit,
    quit,
    isReady: () => false,
    whenReady: () => new Promise<void>(() => void 0)
  })
  const window = {
    isDestroyed: () => false,
    isMinimized: () => false,
    restore: vi.fn(),
    focus: vi.fn(),
    webContents: { send: vi.fn() }
  }
  let rendererReady = false
  let pendingUpdates = false
  let mainWindow: typeof window | null = null
  const deps = {
    app,
    ipcMain: { handle: (channel: string, handler: () => unknown) => handlers.set(channel, handler) },
    HERMES_PROTOCOL: 'hermes',
    DEV_SERVER: undefined,
    getMainWindow: () => mainWindow,
    getRendererReadyForDeepLink: () => rendererReady,
    setRendererReadyForDeepLink: (ready: boolean) => {
      rendererReady = ready
    },
    getPendingOpenUpdates: () => pendingUpdates,
    setPendingOpenUpdates: (pending: boolean) => {
      pendingUpdates = pending
    },
    sendOpenUpdatesRequested: () => calls.push('updates'),
    ensureMainWindow: (_window: unknown, options: { focusExisting: boolean }) =>
      calls.push(`ensure:${options.focusExisting}`),
    createWindow: vi.fn(),
    focusWindow: vi.fn(),
    rememberLog: vi.fn()
  }

  return {
    app,
    handlers,
    exit,
    quit,
    window,
    calls,
    deps,
    setMainWindow: (next: typeof window | null) => {
      mainWindow = next
    },
    setPendingUpdates: (pending: boolean) => {
      pendingUpdates = pending
    }
  }
}

test('a lock loser exits immediately before ready and never registers a second-instance handler', () => {
  const fixture = setup({ lock: false })
  const runtime = createDesktopAppLifecycleRuntime(fixture.deps as any)

  assert.equal(runtime.isPrimaryInstance, false)
  assert.deepEqual(fixture.exit.mock.calls, [[0]])
  assert.equal(fixture.app.listenerCount('second-instance'), 0)
  assert.equal(fixture.app.listenerCount('open-url'), 1)
  assert.equal(fixture.handlers.has('hermes:deep-link-ready'), true)
})

test('an early URL queues until renderer readiness and flushes updates before the single latest link', () => {
  const fixture = setup()
  const runtime = createDesktopAppLifecycleRuntime(fixture.deps as any)
  fixture.setPendingUpdates(true)

  runtime.handleDeepLink('hermes://open/first?tab=one')
  runtime.handleDeepLink('hermes://open/second?tab=two')
  fixture.setMainWindow(fixture.window)
  fixture.handlers.get('hermes:deep-link-ready')?.()

  assert.deepEqual(fixture.calls, ['updates'])
  assert.deepEqual(fixture.window.webContents.send.mock.calls, [
    ['hermes:deep-link', { kind: 'open', name: 'second', params: { tab: 'two' } }]
  ])
  fixture.handlers.get('hermes:deep-link-ready')?.()
  assert.equal(fixture.window.webContents.send.mock.calls.length, 1)
})

test('second-instance resolves the live window and lets deep-link delivery own focus', () => {
  const fixture = setup()
  createDesktopAppLifecycleRuntime(fixture.deps as any)
  fixture.setMainWindow(fixture.window)
  fixture.app.emit('second-instance', {}, ['hermes://open/settings'])

  assert.deepEqual(fixture.calls, ['ensure:false'])
  assert.equal(fixture.window.webContents.send.mock.calls.length, 0)
  fixture.handlers.get('hermes:deep-link-ready')?.()
  assert.deepEqual(fixture.window.webContents.send.mock.calls, [
    ['hermes:deep-link', { kind: 'open', name: 'settings', params: {} }]
  ])
})

test('open-url prevents default before readiness and delivers through the queued path', () => {
  const fixture = setup()
  createDesktopAppLifecycleRuntime(fixture.deps as any)
  const event = { preventDefault: vi.fn() }

  fixture.app.emit('open-url', event, 'hermes://plugin/install?repo=owner%2Frepo')
  assert.equal(event.preventDefault.mock.calls.length, 1)
  fixture.setMainWindow(fixture.window)
  fixture.handlers.get('hermes:deep-link-ready')?.()
  assert.deepEqual(fixture.window.webContents.send.mock.calls, [
    ['hermes:deep-link', { kind: 'plugin', name: 'install', params: { repo: 'owner/repo' } }]
  ])
})

test('ready startup keeps credential migration ahead of window-facing installers', () => {
  const fixture = setup()
  const order: string[] = []
  let onReady: (() => void) | undefined
  const record = (name: string) => () => order.push(name)

  ;(fixture.app as any).whenReady = () => ({
    then: (callback: () => void) => {
      onReady = callback
    }
  })
  ;(fixture.app as any).commandLine = { getSwitchValue: () => '' }
  ;(fixture.app as any).setAsDefaultProtocolClient = record('protocol')
  const deps = {
    ...fixture.deps,
    path: { resolve: () => 'entry' },
    tls: {},
    safeStorage: {},
    session: {
      defaultSession: { availableSpellCheckerLanguages: ['en-US'], setSpellCheckerLanguages: record('spellcheck') }
    },
    screen: { on: vi.fn() },
    Menu: { setApplicationMenu: vi.fn() },
    DEV_SERVER: 'http://localhost:5173',
    CRASH_DIAGNOSTICS: false,
    IS_MAC: false,
    installWindowsSystemCaTrust: () => ({ applied: false }),
    ensureLoginShellPath: record('shell-path'),
    enableBasicPasswordStoreEncryption: record('password-store'),
    migrateLegacyEncryptedSecretsOnce: record('credential-migration'),
    installMediaPermissions: record('permissions'),
    installDownloadHandling: record('downloads'),
    registerMediaProtocol: record('media-protocol'),
    installEmbedReferer: record('embed-referer'),
    installRemoteHeaderRules: record('remote-headers'),
    installPreviewGuestPreload: record('preview-preload'),
    ensureWslWindowsFonts: record('fonts'),
    registerPowerResumeListeners: record('resume-listeners'),
    keepAwake: { set: record('keep-awake') },
    minimizeToTray: { start: record('tray') },
    readPersistedKeepAwake: () => false,
    readPersistedDisableF12: () => false,
    setF12Blocked: record('f12'),
    primaryProfileKey: () => 'default',
    setActiveGatewayProfile: record('active-profile'),
    setWslBridgeProfileState: record('wsl-profile'),
    primaryBackendIsRemote: () => false,
    readQuickEntrySettings: () => ({}),
    applyQuickEntrySettings: record('quick-entry'),
    installCommandScreenshot: record('screenshot'),
    installHudModifierTap: record('hud-tap'),
    resumeManagedSshRecoveries: record('ssh-recovery'),
    installApplicationMenuAfterFirstWindow: record('menu'),
    buildApplicationMenu: vi.fn(),
    resolveRendererIndex: () => 'index.html',
    pathToFileURL: () => ({ toString: () => 'file://index.html' }),
    backendShutdown: { hasStarted: () => false },
    openHudWindow: vi.fn(),
    wakeIndicatorController: { reposition: vi.fn() }
  }

  createDesktopAppLifecycleRuntime(deps as any)
  assert.ok(onReady)
  onReady()

  assert.deepEqual(order, [
    'shell-path',
    'password-store',
    'credential-migration',
    'permissions',
    'downloads',
    'media-protocol',
    'embed-referer',
    'remote-headers',
    'protocol',
    'preview-preload',
    'fonts',
    'spellcheck',
    'resume-listeners',
    'keep-awake',
    'tray',
    'f12',
    'active-profile',
    'wsl-profile',
    'quick-entry',
    'screenshot',
    'hud-tap',
    'ssh-recovery',
    'menu'
  ])
})
