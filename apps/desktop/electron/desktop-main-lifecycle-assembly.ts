import { createDesktopAppLifecycleRuntime } from './desktop-app-lifecycle-runtime'
import { createDesktopHeldQuitRuntime } from './desktop-held-quit-runtime'
import { registerDesktopQuitRuntime } from './desktop-quit-runtime'

// Listener order is part of the quit contract: app lifecycle first, active
// work confirmation second, then backend and SSH teardown. The callbacks for
// state initialized elsewhere remain live getters rather than copied values.
export function installDesktopMainLifecycle(deps: any) {
  const { startup, windows, connections, primaryTeardown } = deps

  const {
    CHROMIUM_LOG_PATH, CRASH_DIAGNOSTICS, rememberLog,
    startChromiumLogWatcher, SKIP_QUIT_CONFIRM,
    flushDesktopLogBufferSync, stopDesktopLogFlushTimer
  } = startup

  const {
    createWindow, focusWindow, minimizeToTray, openHudWindow,
    readQuickEntrySettings, installPreviewGuestPreload,
    wakeIndicatorController, applyQuickEntrySettings, closePetOverlay,
    closeQuickEntryWindow, introRevealController, shellOverlayRuntime
  } = windows

  const {
    installRemoteHeaderRules, migrateLegacyEncryptedSecretsOnce,
    primaryBackendIsRemote, managedConnectionRecoveries,
    managedConnectionUpdates, managedUpdateQuitState,
    sshBootstrapCoordinator, sshConnections, sshTeardowns
  } = connections

  const { primaryProfileKey } = primaryTeardown

  const {
    app, ipcMain, Menu, screen, session, safeStorage, path, tls,
    pathToFileURL, DEV_SERVER, HERMES_PROTOCOL, IS_MAC, IS_WINDOWS,
    backendShutdown, buildApplicationMenu, ensureLoginShellPath,
    ensureMainWindow, ensureWslWindowsFonts,
    enableBasicPasswordStoreEncryption, getIsQuittingForHandoff,
    getMainWindow, getPendingOpenUpdates, getRendererReadyForDeepLink,
    installDownloadHandling, installApplicationMenuAfterFirstWindow,
    installCommandScreenshot, installEmbedReferer, installHudModifierTap,
    installMediaPermissions, installWindowsSystemCaTrust, keepAwake,
    readPersistedDisableF12, readPersistedKeepAwake,
    registerMediaProtocol, registerPowerResumeListeners,
    resolveRendererIndex, resumeManagedSshRecoveries,
    sendOpenUpdatesRequested, setActiveGatewayProfile, setF12Blocked,
    setPendingOpenUpdates, setRendererReadyForDeepLink,
    setWslBridgeProfileState, BrowserWindow, dialog,
    activeWorkByWebContents, backendConnectionState,
    backendQuitNeedsWait, getBootstrapAbortController,
    getWindowsSandboxFallbackSticky, localBackendLifecycle,
    markerAfterSuccessfulBoot, poolStopper, previewTargetRuntime,
    quitTeardown, teardownSshForQuit, terminalIpc,
    waitForManagedUpdateOperations, writeSandboxMarker
  } = deps

  const desktopAppLifecycle = createDesktopAppLifecycleRuntime({
    app,
    ipcMain,
    Menu,
    screen,
    session,
    safeStorage,
    path,
    tls,
    pathToFileURL,
    CHROMIUM_LOG_PATH,
    CRASH_DIAGNOSTICS,
    DEV_SERVER,
    HERMES_PROTOCOL,
    IS_MAC,
    backendShutdown,
    buildApplicationMenu,
    createWindow,
    ensureLoginShellPath,
    ensureMainWindow,
    ensureWslWindowsFonts,
    enableBasicPasswordStoreEncryption,
    focusWindow,
    getIsQuittingForHandoff,
    getMainWindow,
    getPendingOpenUpdates,
    getRendererReadyForDeepLink,
    installDownloadHandling,
    installApplicationMenuAfterFirstWindow,
    installCommandScreenshot,
    installEmbedReferer,
    installHudModifierTap,
    installMediaPermissions,
    installPreviewGuestPreload,
    installRemoteHeaderRules,
    installWindowsSystemCaTrust,
    keepAwake,
    migrateLegacyEncryptedSecretsOnce,
    minimizeToTray,
    openHudWindow,
    primaryBackendIsRemote,
    primaryProfileKey,
    readPersistedDisableF12,
    readPersistedKeepAwake,
    readQuickEntrySettings,
    registerMediaProtocol,
    registerPowerResumeListeners,
    rememberLog,
    resolveRendererIndex,
    resumeManagedSshRecoveries,
    sendOpenUpdatesRequested,
    setActiveGatewayProfile,
    setF12Blocked,
    setPendingOpenUpdates,
    setRendererReadyForDeepLink,
    setWslBridgeProfileState,
    startChromiumLogWatcher,
    wakeIndicatorController,
    applyQuickEntrySettings
  })

  const isPrimaryInstance = desktopAppLifecycle.isPrimaryInstance

  // Register after lifecycle setup so the held quit precedes normal teardown.
  const heldQuitForActiveWork = createDesktopHeldQuitRuntime({
    app,
    BrowserWindow,
    dialog,
    activeWorkByWebContents,
    minimizeToTray,
    getIsQuittingForHandoff,
    skipQuitConfirm: SKIP_QUIT_CONFIRM
  })

  registerDesktopQuitRuntime({
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
  })

  return { desktopAppLifecycle, isPrimaryInstance, heldQuitForActiveWork }
}
