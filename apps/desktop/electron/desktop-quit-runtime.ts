// Register after the active-work guard and app lifecycle setup: listener order
// is part of the shutdown contract. The managed update joins before the SSH
// bootstrap coordinator is sealed, then normal teardown re-enters before-quit.
export function registerDesktopQuitRuntime(deps: {
  IS_WINDOWS: boolean
  app: any
  backendConnectionState: any
  backendQuitNeedsWait: (...args: any[]) => boolean
  backendShutdown: any
  closePetOverlay: () => void
  closeQuickEntryWindow: () => void
  flushDesktopLogBufferSync: () => void
  getBootstrapAbortController: () => AbortController | null
  getIsQuittingForHandoff: () => boolean
  getWindowsSandboxFallbackSticky: () => boolean
  heldQuitForActiveWork: (event: Electron.Event) => boolean
  introRevealController: any
  localBackendLifecycle: any
  managedConnectionRecoveries: Map<string, Promise<void>>
  managedConnectionUpdates: Map<string, Promise<any>>
  managedUpdateQuitState: { wait: Promise<void> | null; done: boolean }
  markerAfterSuccessfulBoot: (...args: any[]) => any
  minimizeToTray: any
  poolStopper: any
  previewTargetRuntime: any
  quitTeardown: any
  shellOverlayRuntime: any
  sshBootstrapCoordinator: any
  sshConnections: Map<string, any>
  sshTeardowns: any
  stopDesktopLogFlushTimer: () => void
  teardownSshForQuit: () => Promise<void>
  terminalIpc: any
  waitForManagedUpdateOperations: (...args: any[]) => Promise<void>
  wakeIndicatorController: any
  writeSandboxMarker: (...args: any[]) => any
}) {
  const {
    IS_WINDOWS,
    app,
    backendConnectionState,
    backendQuitNeedsWait,
    backendShutdown,
    closePetOverlay,
    closeQuickEntryWindow,
    flushDesktopLogBufferSync,
    getBootstrapAbortController,
    getIsQuittingForHandoff,
    getWindowsSandboxFallbackSticky,
    heldQuitForActiveWork,
    introRevealController,
    localBackendLifecycle,
    managedConnectionRecoveries,
    managedConnectionUpdates,
    managedUpdateQuitState,
    markerAfterSuccessfulBoot,
    minimizeToTray,
    poolStopper,
    previewTargetRuntime,
    quitTeardown,
    shellOverlayRuntime,
    sshBootstrapCoordinator,
    sshConnections,
    sshTeardowns,
    stopDesktopLogFlushTimer,
    teardownSshForQuit,
    terminalIpc,
    waitForManagedUpdateOperations,
    wakeIndicatorController,
    writeSandboxMarker
  } = deps

  app.on('before-quit', event => {
    // Runs ahead of every teardown below, so "Keep Running" leaves the app
    // exactly as it was.
    if (heldQuitForActiveWork(event)) {
      return
    }

    minimizeToTray.beginQuit()

    // A detached remote updater can outlive this Electron process. Do not tear
    // down its SSH observer/restore transaction at the generic SSH shutdown
    // deadline: join it first (BEFORE sealing the bootstrap coordinator, whose
    // shutdown would refuse the restore dials), then re-enter before-quit for
    // normal teardown. A crash still fails closed on next launch via the remote
    // install-marker preflight in both POSIX and Windows lifecycle
    // implementations.
    if (
      !managedUpdateQuitState.done &&
      (managedUpdateQuitState.wait || managedConnectionUpdates.size > 0 || managedConnectionRecoveries.size > 0)
    ) {
      event.preventDefault()

      if (!managedUpdateQuitState.wait) {
        managedUpdateQuitState.wait = waitForManagedUpdateOperations(() => [
          ...managedConnectionUpdates.values(),
          ...managedConnectionRecoveries.values()
        ]).finally(() => {
          managedUpdateQuitState.done = true
          app.quit()
        })
      }

      return
    }

    // A prevented first quit leaves the renderer alive while teardown runs.
    // Seal the SSH coordinator before touching connections so reconnect
    // callbacks cannot recreate a backend for a registration whose app is
    // already quitting (#91668).
    sshBootstrapCoordinator.shutdown()

    const backendNeedsWait = backendQuitNeedsWait({
      connectionPending: backendConnectionState.getPendingPromise() !== null || localBackendLifecycle.hasPending(),
      poolPending: poolStopper.hasPending(),
      processAttached: backendConnectionState.getProcess() !== null,
      shutdownPending: backendShutdown.isPending()
    })

    const sshNeedsWait =
      sshConnections.size > 0 || sshBootstrapCoordinator.promises().length > 0 || sshTeardowns.hasPending()

    const teardownTasks = [{ run: () => backendShutdown.run(), waitForCompletion: backendNeedsWait }]

    if (sshNeedsWait) {
      teardownTasks.push({ run: teardownSshForQuit, waitForCompletion: true })
    }

    if (quitTeardown.begin(teardownTasks)) {
      event.preventDefault()
    }

    // Clean quit mid-boot should not trip next-launch --no-sandbox (#38216).
    // FATAL GPU aborts skip before-quit, leaving the `booting` marker in place.
    // Keyed on sticky (not active): a manual --no-sandbox run still records a
    // clean quit, while an engaged fallback keeps its sticky marker.
    if (IS_WINDOWS && !getWindowsSandboxFallbackSticky()) {
      try {
        writeSandboxMarker(app.getPath('userData'), markerAfterSuccessfulBoot({ fallbackActive: false }))
      } catch {
        void 0
      }
    }

    // The always-on-top overlay isn't a "real" app window; close it so a stray
    // pet can't keep the process alive or float over a quit app.
    closePetOverlay()
    wakeIndicatorController.close()
    introRevealController.destroy()

    // Same for the HUD — an always-on-top panel outliving the app would leave a
    // floating composer with nothing behind it. Close it directly rather than via
    // closeHudWindow(): that also re-shows the main window, which is wrong on the
    // way out (and `hudRestoreMainWindow` may still be armed from entering HUD).
    shellOverlayRuntime.closeHudWindowForQuit()

    // Same for the Quick Entry composer — and release its global accelerator so a
    // quitting Hermes never keeps another app's chord hostage.
    closeQuickEntryWindow()

    // Quitting mid-install should stop the installer, not orphan it.
    if (getBootstrapAbortController()) {
      try {
        getBootstrapAbortController()?.abort()
      } catch {
        void 0
      }
    }

    stopDesktopLogFlushTimer()
    flushDesktopLogBufferSync()
    previewTargetRuntime.closePreviewWatchers()

    // Kill open PTYs before environment teardown to avoid the node-pty#904
    // ThreadSafeFunction SIGABRT race.
    terminalIpc.disposeAllTerminalSessions()

    void backendShutdown.run()
  })

  app.on('window-all-closed', () => {
    // macOS convention: keep the process alive in the Dock when the user closes
    // the last window. But when we're handing off to a detached updater / swap /
    // uninstall script, the process MUST exit so the script can replace or remove
    // the bundle and relaunch — without this the script's PID-wait spins to its
    // full timeout and the user is left with an invisible app (or an uninstall
    // that appears to do nothing).
    if (process.platform !== 'darwin' || getIsQuittingForHandoff()) {
      app.quit()
    }
  })
}
