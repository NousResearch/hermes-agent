import { waitForBackendExit as waitForBackendExitImpl } from './backend-child'
import { backendScopePrefix } from './connection-registry'
import { PrimaryProfilePin } from './primary-profile-pin'

export function sendBackendExitToLiveWindow(
  primaryTeardown: { isSoftRehomeInProgress: () => boolean },
  getMainWindow: () => {
    isDestroyed: () => boolean
    webContents?: { isDestroyed: () => boolean; send: (channel: string, payload: unknown) => void }
  } | null,
  payload: unknown
) {
  // Intentional soft re-home (gateway mode apply) kills the child on purpose —
  // don't surface the "backend stopped" error toast / boot-failure path.
  if (primaryTeardown.isSoftRehomeInProgress()) {return}

  const mainWindow = getMainWindow()

  if (!mainWindow || mainWindow.isDestroyed()) {return}

  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {return}

  webContents.send('hermes:backend-exit', payload)
}

// Owns intentional primary invalidation, exit-wait deduplication, and cleanup
// of every scope behind a removed registry connection. Late pool callbacks stay
// lazy until the pool owner and SSH bootstrap runtime are initialized.
export function createDesktopPrimaryTeardownRuntime(deps: any) {
  const {
    firstRunBoot, localBackendLifecycle, rememberLog, clearFailures,
    remoteLiveness, suppressPrimaryRecovery, backendConnectionState,
    forceKillProcessTree, IS_WINDOWS, readActiveDesktopProfile,
    backendPool, sshConnections, sshBootstrapCoordinator,
    stopPoolBackend, teardownSshConnection
  } = deps

  let softRehomeInProgress = false

  function resetBootProgressForReconnect() {
    firstRunBoot.updateBootProgress(
      {
        error: null,
        message: 'Restarting desktop connection',
        phase: 'backend.resolve',
        progress: 4,
        running: true
      },
      { allowDecrease: true }
    )
  }

  function stopBackendChild(child) {
    void localBackendLifecycle.stop(child).catch(error => rememberLog(`Backend teardown failed: ${error.message}`))
  }

  // Soft gateway-mode apply: tear down the primary without resetting boot UI or
  // reloading the renderer. The shell stays up; the renderer wipes session lists
  // (so skeletons retrigger) and re-dials. Distinct from hard re-home (profile
  // switch / crash recovery), which still resets boot progress + reloads.
  function resetHermesConnection({ soft = false } = {}) {
    clearFailures()
    remoteLiveness.clear()
    // The next startHermes() re-reads active-profile.json for its launch profile.
    primaryProfilePin.clear()
    const hermesProcess = invalidatePrimaryConnection()
    stopBackendChild(hermesProcess)

    if (!soft) {
      resetBootProgressForReconnect()
    }
  }

  // Every deliberate emptying of the primary slot goes through here so the
  // dying child's stale exit reads as intentional (see primaryRecoverySuppressed).
  function invalidatePrimaryConnection() {
    suppressPrimaryRecovery()

    return backendConnectionState.invalidate()
  }

  // Re-home the primary backend: reset connection state, then wait for the live
  // dashboard process to actually exit (SIGKILL after 5s) so the next
  // startHermes() spawns fresh instead of racing the dying one. Shared by the
  // connection-config and profile switch flows.
  async function teardownPrimaryBackendAndWait({ soft = false } = {}) {
    // Capture the reference before resetHermesConnection() invalidates it.
    const hermesProcess = backendConnectionState.getProcess()
    const dying = hermesProcess && !hermesProcess.killed ? hermesProcess : null

    if (soft) {
      softRehomeInProgress = true
    }

    try {
      resetHermesConnection({ soft })
      await waitForBackendExit(dying)
    } finally {
      if (soft) {
        softRehomeInProgress = false
      }
    }
  }

  const backendExitWaits = new Map<any, Promise<void>>()

  function waitForBackendExit(child, timeoutMs = 5000) {
    const existing = backendExitWaits.get(child)

    if (existing) {
      return existing
    }

    const waiting = waitForBackendExitImpl(child, { forceKillProcessTree, isWindows: IS_WINDOWS }, timeoutMs)
    backendExitWaits.set(child, waiting)
    void waiting.then(
      () => backendExitWaits.delete(child),
      () => backendExitWaits.delete(child)
    )

    return waiting
  }

  // The profile the primary (window) backend was actually LAUNCHED as. Pinned by
  // startHermes() and cleared when the primary is torn down; while a primary is
  // live this must NOT follow active-profile.json (see primary-profile-pin.ts).
  const primaryProfilePin = new PrimaryProfilePin()

  function primaryProfileKey() {
    return primaryProfilePin.resolve(readActiveDesktopProfile)
  }

  // Managed SSH restore borrows the same gate, pools, and coordinator that startup
  // and before-quit use. It is composed after connection admission is ready.

  // Stop every pooled backend and ssh scope owned by a registry connection —
  // called when the connection is removed from the registry.
  async function stopRegistryConnectionBackends(connectionId) {
    const prefix = backendScopePrefix(connectionId)

    for (const key of [...backendPool.keys()]) {
      if (String(key).startsWith(prefix)) {
        stopPoolBackend(key)
      }
    }

    const sshScopes = new Set([
      ...[...sshConnections.keys()].filter(scope => String(scope).startsWith(prefix)),
      ...[...sshBootstrapCoordinator.active].map(entry => entry.scope).filter(scope => String(scope).startsWith(prefix))
    ])

    await Promise.all(
      [...sshScopes].map(async scope => {
        await sshBootstrapCoordinator.cancelAndWait(scope)
        await teardownSshConnection(scope)
      })
    )
  }


  return {
    stopBackendChild, resetHermesConnection, invalidatePrimaryConnection,
    teardownPrimaryBackendAndWait, waitForBackendExit,
    primaryProfilePin, primaryProfileKey, stopRegistryConnectionBackends,
    isSoftRehomeInProgress: () => softRehomeInProgress
  }
}
