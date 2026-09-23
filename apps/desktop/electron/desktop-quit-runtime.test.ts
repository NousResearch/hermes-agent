import { describe, expect, it, vi } from 'vitest'

import { registerDesktopQuitRuntime } from './desktop-quit-runtime'

function setupQuitRuntime(options: { holdActiveWork?: boolean; updateWait?: Promise<void> } = {}) {
  const handlers = new Map<string, (event?: any) => void>()
  const order: string[] = []

  const app = {
    getPath: () => 'test-user-data',
    on: vi.fn((name: string, handler: (event?: any) => void) => handlers.set(name, handler)),
    quit: vi.fn()
  }

  const minimizeToTray = { beginQuit: vi.fn(() => order.push('tray-quit')) }

  const sshBootstrapCoordinator = {
    promises: () => [],
    shutdown: vi.fn(() => order.push('ssh-shutdown'))
  }

  const backendShutdown = {
    isPending: () => false,
    run: vi.fn(async () => order.push('backend-shutdown'))
  }

  const managedUpdateQuitState = { wait: null as Promise<void> | null, done: false }
  const managedConnectionUpdates = new Map<string, Promise<any>>()

  if (options.updateWait) {
    managedConnectionUpdates.set('update', options.updateWait)
  }

  registerDesktopQuitRuntime({
    IS_WINDOWS: false,
    app,
    backendConnectionState: { getPendingPromise: () => null, getProcess: () => null },
    backendQuitNeedsWait: () => false,
    backendShutdown,
    closePetOverlay: vi.fn(),
    closeQuickEntryWindow: vi.fn(),
    flushDesktopLogBufferSync: vi.fn(),
    getBootstrapAbortController: () => null,
    getIsQuittingForHandoff: () => false,
    getWindowsSandboxFallbackSticky: () => false,
    heldQuitForActiveWork: () => Boolean(options.holdActiveWork),
    introRevealController: { destroy: vi.fn() },
    localBackendLifecycle: { hasPending: () => false },
    managedConnectionRecoveries: new Map(),
    managedConnectionUpdates,
    managedUpdateQuitState,
    markerAfterSuccessfulBoot: vi.fn(),
    minimizeToTray,
    poolStopper: { hasPending: () => false },
    previewTargetRuntime: { closePreviewWatchers: vi.fn() },
    quitTeardown: { begin: vi.fn(() => false) },
    shellOverlayRuntime: { closeHudWindowForQuit: vi.fn() },
    sshBootstrapCoordinator,
    sshConnections: new Map(),
    sshTeardowns: { hasPending: () => false },
    stopDesktopLogFlushTimer: vi.fn(),
    teardownSshForQuit: vi.fn(async () => {}),
    terminalIpc: { disposeAllTerminalSessions: vi.fn() },
    waitForManagedUpdateOperations: () => options.updateWait ?? Promise.resolve(),
    wakeIndicatorController: { close: vi.fn() },
    writeSandboxMarker: vi.fn()
  })

  return { app, handlers, managedUpdateQuitState, minimizeToTray, order, sshBootstrapCoordinator }
}

describe('desktop quit registration', () => {
  it('leaves all teardown untouched while the active-work prompt holds quit', () => {
    const runtime = setupQuitRuntime({ holdActiveWork: true })
    const event = { preventDefault: vi.fn() }

    runtime.handlers.get('before-quit')!(event)

    expect(runtime.minimizeToTray.beginQuit).not.toHaveBeenCalled()
    expect(runtime.sshBootstrapCoordinator.shutdown).not.toHaveBeenCalled()
    expect(event.preventDefault).not.toHaveBeenCalled()
  })

  it('joins a managed update before sealing SSH and resumes teardown on re-entry', async () => {
    let finishUpdate!: () => void

    const updateWait = new Promise<void>(resolve => {
      finishUpdate = resolve
    })

    const runtime = setupQuitRuntime({ updateWait })
    const event = { preventDefault: vi.fn() }

    runtime.handlers.get('before-quit')!(event)

    expect(event.preventDefault).toHaveBeenCalledOnce()
    expect(runtime.sshBootstrapCoordinator.shutdown).not.toHaveBeenCalled()
    expect(runtime.managedUpdateQuitState.done).toBe(false)

    finishUpdate()
    await updateWait
    await Promise.resolve()

    expect(runtime.managedUpdateQuitState.done).toBe(true)
    expect(runtime.app.quit).toHaveBeenCalledOnce()

    runtime.handlers.get('before-quit')!({ preventDefault: vi.fn() })

    expect(runtime.order.indexOf('ssh-shutdown')).toBeGreaterThan(runtime.order.indexOf('tray-quit'))
    expect(runtime.order.indexOf('backend-shutdown')).toBeGreaterThan(runtime.order.indexOf('ssh-shutdown'))
  })
})
