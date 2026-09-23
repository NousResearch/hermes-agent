import { expect, test, vi } from 'vitest'

import { createDesktopPrimaryTeardownRuntime, sendBackendExitToLiveWindow } from './desktop-primary-teardown-runtime'

test('soft primary rehome invalidates the slot and suppresses recovery without resetting boot progress', () => {
  const child = { pid: 42 }
  const stopped = vi.fn(async () => undefined)
  const clearFailures = vi.fn()
  const suppressPrimaryRecovery = vi.fn()
  const updateBootProgress = vi.fn()
  const clearProfilePin = vi.fn()

  const runtime = createDesktopPrimaryTeardownRuntime({
    firstRunBoot: { updateBootProgress },
    localBackendLifecycle: { stop: stopped },
    rememberLog: vi.fn(),
    clearFailures,
    remoteLiveness: { clear: vi.fn() },
    suppressPrimaryRecovery,
    backendConnectionState: { invalidate: () => child, getProcess: () => child },
    forceKillProcessTree: vi.fn(),
    IS_WINDOWS: false,
    readActiveDesktopProfile: vi.fn(),
    backendPool: new Map(),
    sshConnections: new Map(),
    sshBootstrapCoordinator: { active: [], cancelAndWait: vi.fn() },
    stopPoolBackend: vi.fn(),
    teardownSshConnection: vi.fn(),
    clearProfilePin
  } as any)

  runtime.resetHermesConnection({ soft: true })
  expect(clearFailures).toHaveBeenCalledOnce()
  expect(suppressPrimaryRecovery).toHaveBeenCalledOnce()
  expect(stopped).toHaveBeenCalledExactlyOnceWith(child)
  expect(updateBootProgress).not.toHaveBeenCalled()
  expect(runtime.isSoftRehomeInProgress()).toBe(false)
})

test('backend exit notification reaches only a live primary window outside soft rehome', () => {
  const send = vi.fn()
  let softRehome = true
  let destroyed = false
  let contentDestroyed = false
  const primaryTeardown = { isSoftRehomeInProgress: () => softRehome }

  const getMainWindow = () => ({
    isDestroyed: () => destroyed,
    webContents: { isDestroyed: () => contentDestroyed, send }
  })

  const payload = { reason: 'process-exited' }

  sendBackendExitToLiveWindow(primaryTeardown, getMainWindow, payload)
  expect(send).not.toHaveBeenCalled()

  softRehome = false
  destroyed = true
  sendBackendExitToLiveWindow(primaryTeardown, getMainWindow, payload)
  expect(send).not.toHaveBeenCalled()

  destroyed = false
  contentDestroyed = true
  sendBackendExitToLiveWindow(primaryTeardown, getMainWindow, payload)
  expect(send).not.toHaveBeenCalled()

  contentDestroyed = false
  sendBackendExitToLiveWindow(primaryTeardown, getMainWindow, payload)
  expect(send).toHaveBeenCalledExactlyOnceWith('hermes:backend-exit', payload)
})
