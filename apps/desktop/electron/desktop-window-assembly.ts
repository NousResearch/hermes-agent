import { registerChatOnboardingWindow } from './chat-onboarding-window'
import { createDesktopPetOverlayRuntime } from './desktop-pet-overlay-runtime'
import { createDesktopPrimaryWindowRuntime } from './desktop-primary-window-runtime'
import { createDesktopSecondaryWindowRuntime } from './desktop-secondary-window-runtime'
import { createDesktopShellOverlayRuntime } from './desktop-shell-overlay-runtime'
import { createDesktopWindowWiringRuntime } from './desktop-window-wiring-runtime'
import { createIntroRevealWindowController } from './intro-reveal-window'
import { createWakeIndicatorWindowController } from './wake-indicator-window'
import { WindowConnectionRouteRegistry } from './window-connection-route'

// Keep the native window owners in their original construction order. Secondary
// windows capture createWindow lazily; the primary owner is initialized before
// any app lifecycle callback can create the first window.
export function createDesktopWindowAssembly(deps: any) {
  const {
    DEV_SERVER, PREVIEW_GUEST_PRELOAD_PATH, app, createWindowOpenHandler,
    createWindowRevealController, installBrowserNavGestures,
    installContextMenuBridge, installDevToolsShortcut, installFindShortcut,
    installPreviewShortcut, installZoomReassertOnNavigation,
    installZoomReassertOnWindowEvents, installZoomShortcuts, openExternalUrl,
    rememberLog, restorePersistedZoomLevel, IS_MAC, PRELOAD_PATH,
    RENDERER_RELOAD_MAX, RENDERER_RELOAD_WINDOW_MS, WINDOW_BUTTON_POSITION,
    appearance, ensureMainWindow, getAppIconPath, getIsQuittingForHandoff,
    getMainWindow, getStreamThrottle, loadWindowUrl, primaryProfileKey,
    readWindowState, recordWindowConnectionRoute, rendererReloadTimesRef,
    resolveRendererIndex, sendWindowStateChanged, validateDesktopProfileRoute,
    GUEST_ONBOARDING, zoomWiringForWindowKind, HUD_WINDOW_TITLE,
    bindGeometryPersistence, writeFileAtomic, BrowserWindow, screen,
    IS_WINDOWS, WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH, alreadyHasNoSandbox,
    attachRendererConsoleCapture, backendShutdown, bindWindowChromeEvents,
    buildNoSandboxRelaunchArgs, chatWindowWebPreferences,
    clearRendererReadyForDeepLink, computeWindowOptions,
    connectDesktopProfileRoute, desktopProfilePreferences,
    exitAfterBackendShutdown, fallbackMarker, firstRunBoot,
    installWindowRendererLifecycle, loadRendererLoadErrorPage,
    markerAfterSuccessfulBoot, notifyLauncherWindowRevealed,
    resolveRendererIndexWithMissing, sandboxState, schedulePersistWindowState,
    setMainWindow, shouldRelaunchForRendererSandboxCrashLoop, startHermes,
    writeSandboxMarker
  } = deps

  const { wireCommonWindowHandlers, installPreviewGuestPreload, wireWindowReveal } =
    createDesktopWindowWiringRuntime({
      DEV_SERVER,
      PREVIEW_GUEST_PRELOAD_PATH,
      app,
      createWindowOpenHandler,
      createWindowRevealController,
      installBrowserNavGestures,
      installContextMenuBridge,
      installDevToolsShortcut,
      installFindShortcut,
      installPreviewShortcut,
      installZoomReassertOnNavigation,
      installZoomReassertOnWindowEvents,
      installZoomShortcuts,
      openExternalUrl,
      rememberLog,
      restorePersistedZoomLevel
    })

  const { minimizeToTray, focusWindow, createSessionWindow, createBrowserWindow, createInstanceWindow } =
    createDesktopSecondaryWindowRuntime({
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
      getWindowConnectionRoutes: () => windowConnectionRoutes,
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
    })

  // A macOS-only ambient wake cue. It is deliberately a gateway-less helper
  // window: the active renderer owns voice state and sends only the visual phase.
  const wakeIndicatorController = createWakeIndicatorWindowController({
    devServer: DEV_SERVER,
    isMac: IS_MAC,
    loadWindowUrl,
    log: rememberLog,
    preloadPath: PRELOAD_PATH,
    rendererIndex: resolveRendererIndex,
    wireWindow: window => wireCommonWindowHandlers(window, zoomWiringForWindowKind('wakeIndicator'))
  })

  const introRevealController = createIntroRevealWindowController({
    devServer: DEV_SERVER,
    enabled: GUEST_ONBOARDING,
    isMac: IS_MAC,
    loadWindowUrl,
    log: rememberLog,
    mainWindow: getMainWindow,
    preloadPath: PRELOAD_PATH,
    rendererIndex: resolveRendererIndex,
    showMain: () => {
      const mainWindow = getMainWindow()
      mainWindow.show()
      mainWindow.focus()
    },
    wireWindow: window => wireCommonWindowHandlers(window, zoomWiringForWindowKind('petOverlay'))
  })

  registerChatOnboardingWindow({
    enabled: GUEST_ONBOARDING,
    mainWindow: getMainWindow
  })

  const { getPetOverlayWindow, openPetOverlay, closePetOverlay } = createDesktopPetOverlayRuntime({
    DEV_SERVER,
    IS_MAC,
    PRELOAD_PATH,
    getMainWindow,
    loadWindowUrl,
    rememberLog,
    resolveRendererIndex,
    wireCommonWindowHandlers,
    wireWindowReveal
  })

  const shellOverlayRuntime = createDesktopShellOverlayRuntime({
    DEV_SERVER,
    HUD_WINDOW_TITLE,
    IS_MAC,
    PRELOAD_PATH,
    bindGeometryPersistence,
    focusWindow,
    getMainWindow,
    getStreamThrottle,
    loadWindowUrl,
    rememberLog,
    resolveRendererIndex,
    wireCommonWindowHandlers,
    wireWindowReveal,
    writeFileAtomic
  })

  const {
    applyQuickEntrySettings,
    closeHudWindow,
    closeQuickEntryWindow,
    hideQuickEntryWindow,
    openHudWindow,
    readQuickEntrySettings,
    resetHudWindowLayout,
    writeQuickEntrySettings
  } = shellOverlayRuntime

  const primaryWindowRuntime = createDesktopPrimaryWindowRuntime({
    app,
    BrowserWindow,
    screen,
    DEV_SERVER,
    IS_MAC,
    IS_WINDOWS,
    PRELOAD_PATH,
    RENDERER_RELOAD_MAX,
    RENDERER_RELOAD_WINDOW_MS,
    WINDOW_BUTTON_POSITION,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    alreadyHasNoSandbox,
    appearance,
    attachRendererConsoleCapture,
    backendShutdown,
    bindGeometryPersistence,
    bindWindowChromeEvents,
    buildNoSandboxRelaunchArgs,
    chatWindowWebPreferences,
    clearRendererReadyForDeepLink,
    closePetOverlay,
    computeWindowOptions,
    connectDesktopProfileRoute,
    desktopProfilePreferences,
    exitAfterBackendShutdown,
    fallbackMarker,
    firstRunBoot,
    getAppIconPath,
    getIsQuittingForHandoff,
    getMainWindow,
    getStreamThrottle,
    installWindowRendererLifecycle,
    introRevealController,
    loadRendererLoadErrorPage,
    loadWindowUrl,
    markerAfterSuccessfulBoot,
    minimizeToTray,
    notifyLauncherWindowRevealed,
    readWindowState,
    recordWindowConnectionRoute,
    rememberLog,
    rendererReloadTimesRef,
    resolveRendererIndex,
    resolveRendererIndexWithMissing,
    sandboxState,
    schedulePersistWindowState,
    sendWindowStateChanged,
    setMainWindow,
    shouldRelaunchForRendererSandboxCrashLoop,
    startHermes,
    wakeIndicatorController,
    wireCommonWindowHandlers,
    wireWindowReveal,
    writeSandboxMarker,
    zoomWiringForWindowKind
  })

  // The secondary-window runtime receives this declaration before the primary
  // runtime is initialized; the callback is invoked only after app startup.
  function createWindow() {
    return primaryWindowRuntime.createWindow()
  }

  const windowConnectionRoutes = new WindowConnectionRouteRegistry()

  return {
    wireCommonWindowHandlers, installPreviewGuestPreload, wireWindowReveal,
    minimizeToTray, focusWindow, createSessionWindow, createBrowserWindow,
    createInstanceWindow, wakeIndicatorController, introRevealController,
    getPetOverlayWindow, openPetOverlay, closePetOverlay, shellOverlayRuntime,
    applyQuickEntrySettings, closeHudWindow, closeQuickEntryWindow,
    hideQuickEntryWindow, openHudWindow, readQuickEntrySettings,
    resetHudWindowLayout, writeQuickEntrySettings, createWindow,
    windowConnectionRoutes
  }
}
