import { pathToFileURL } from 'node:url'

// The primary window keeps the main process as the owner of mutable state.
// Accessors preserve the live bindings observed by late Electron callbacks.
interface DesktopPrimaryWindowRuntimeDeps {
  app: any
  BrowserWindow: new (options: any) => any
  screen: { getAllDisplays: () => any[] }
  DEV_SERVER: string | undefined
  IS_MAC: boolean
  IS_WINDOWS: boolean
  PRELOAD_PATH: string
  RENDERER_RELOAD_MAX: number
  RENDERER_RELOAD_WINDOW_MS: number
  WINDOW_BUTTON_POSITION: any
  WINDOW_MIN_HEIGHT: number
  WINDOW_MIN_WIDTH: number
  alreadyHasNoSandbox: (...args: any[]) => boolean
  appearance: any
  attachRendererConsoleCapture: (...args: any[]) => any
  backendShutdown: { hasStarted: () => boolean }
  bindGeometryPersistence: (...args: any[]) => any
  bindWindowChromeEvents: (...args: any[]) => any
  buildNoSandboxRelaunchArgs: (...args: any[]) => any
  chatWindowWebPreferences: (...args: any[]) => any
  clearRendererReadyForDeepLink: () => void
  closePetOverlay: () => void
  computeWindowOptions: (...args: any[]) => any
  connectDesktopProfileRoute: (...args: any[]) => Promise<any>
  desktopProfilePreferences: any
  exitAfterBackendShutdown: (...args: any[]) => any
  fallbackMarker: (...args: any[]) => any
  firstRunBoot: { broadcastBootProgress: () => void }
  getAppIconPath: () => any
  getIsQuittingForHandoff: () => boolean
  getMainWindow: () => any
  getStreamThrottle: () => any
  installWindowRendererLifecycle: (...args: any[]) => any
  introRevealController: { destroy: () => void }
  loadRendererLoadErrorPage: (...args: any[]) => any
  loadWindowUrl: (...args: any[]) => any
  markerAfterSuccessfulBoot: (...args: any[]) => any
  minimizeToTray: { registerWindow: (...args: any[]) => any }
  notifyLauncherWindowRevealed: () => void
  readWindowState: () => any
  recordWindowConnectionRoute: (...args: any[]) => any
  rememberLog: (message: string) => void
  rendererReloadTimesRef: { current: number[] }
  resolveRendererIndex: () => string
  resolveRendererIndexWithMissing: () => any
  sandboxState: {
    fallbackActive: boolean
    fallbackSticky: boolean
    fallbackReason: string
    noSandboxRelaunchAttempted: boolean
  }
  schedulePersistWindowState: ((...args: any[]) => any) & { flush: () => any }
  sendWindowStateChanged: (...args: any[]) => any
  setMainWindow: (window: any) => void
  shouldRelaunchForRendererSandboxCrashLoop: (...args: any[]) => boolean
  startHermes: () => Promise<any>
  wakeIndicatorController: { close: () => void }
  wireCommonWindowHandlers: (...args: any[]) => any
  wireWindowReveal: (...args: any[]) => any
  writeSandboxMarker: (...args: any[]) => any
  zoomWiringForWindowKind: (...args: any[]) => any
}

export function createDesktopPrimaryWindowRuntime(deps: DesktopPrimaryWindowRuntimeDeps) {
  const {
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
  } = deps

  function createWindow() {
    const icon = getAppIconPath()
    const savedWindowState = readWindowState()
    setMainWindow(
      new BrowserWindow({
        ...computeWindowOptions(savedWindowState, screen.getAllDisplays()),
        minWidth: WINDOW_MIN_WIDTH,
        minHeight: WINDOW_MIN_HEIGHT,
        title: 'Hermes',
        // Frameless title bar on every platform so the renderer can paint the
        // "hide sidebar" button (and other left-side titlebar tools) flush with
        // the top edge — matching the macOS layout where the traffic lights sit
        // inside the same band. On Windows/Linux, titleBarOverlay tells Electron
        // to paint native min/max/close in the top-right of the renderer; on
        // macOS it just reserves a content inset alongside the traffic lights.
        titleBarStyle: 'hidden',
        titleBarOverlay: appearance.getTitleBarOverlayOptions(),
        trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : undefined,
        ...appearance.chatWindowSurfaceOptions(),
        icon,
        // Hidden until the first themed paint so macOS `vibrancy` (which ignores
        // `backgroundColor` and follows the OS appearance) can't flash a light
        // material before the renderer paints the app theme. See createSessionWindow.
        show: false,
        // Shared with the secondary session windows (chatWindowWebPreferences);
        // stream-aware throttling is applied per-window via getStreamThrottle() so a
        // live answer keeps painting while the window is blurred or minimized,
        // without pinning visibilityState to 'visible' at idle. See
        // session-windows.ts and stream-throttle.ts.
        webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
      })
    )

    const createdMainWindow = getMainWindow()
    minimizeToTray.registerWindow(createdMainWindow, { closeToTray: true })
    const defaultRoute = desktopProfilePreferences.getDefault()

    if (defaultRoute) {
      recordWindowConnectionRoute(getMainWindow().webContents, {
        ...defaultRoute,
        registryScoped: defaultRoute.connectionId !== null
      })
    }

    // Chat-surface registration: see applyWindowTranslucency.
    appearance.registerChatWindow(getMainWindow())

    if (IS_MAC) {
      getMainWindow().setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)

      if (icon) {
        app.dock?.setIcon(icon)
      }
    }

    if (!IS_MAC) {
      appearance.installNativeThemeListener()
    }

    if (savedWindowState?.isMaximized) {
      getMainWindow().maximize()
    }

    const revealController = wireWindowReveal(createdMainWindow, {
      onRevealed: () => {
        // Persist geometry as soon as the window is visible so a crash before the
        // first clean resize/move/close still captures the restored bounds (#56726).
        schedulePersistWindowState()

        // #111906: the Linux launcher holds back its .desktop entry write until the
        // window is on screen (a STARTING gnome-shell app must not see its entry change).
        notifyLauncherWindowRevealed()

        // #38216: clear the mid-boot marker only after a window is actually usable.
        // Keep sticky `fallback` when we launched with --no-sandbox so the next
        // Start Menu click does not re-enter the GPU FATAL crash loop. The marker
        // records the app version so the next update re-probes the sandbox.
        if (IS_WINDOWS) {
          try {
            writeSandboxMarker(
              app.getPath('userData'),
              markerAfterSuccessfulBoot({
                fallbackActive: sandboxState.fallbackSticky,
                reason: sandboxState.fallbackReason,
                appVersion: app.getVersion()
              })
            )
          } catch (error) {
            rememberLog(`[sandbox] marker update after main-window reveal failed: ${error?.message || error}`)
          }
        }
      }
    })

    // Under Playwright testing, instantly show the window: `ready-to-show`
    // doesn't fire in some testing envs, and the suite can't wait out the
    // production fallback.
    if (process.env.TEST_WORKER_INDEX !== undefined) {
      revealController.reveal()
    }

    bindWindowChromeEvents(getMainWindow(), sendWindowStateChanged)

    // Reopen where the user left off. close is the backstop, flushed
    // synchronously before the window is gone.
    bindGeometryPersistence(getMainWindow(), schedulePersistWindowState)
    getMainWindow().on('maximize', schedulePersistWindowState)
    getMainWindow().on('unmaximize', schedulePersistWindowState)
    getMainWindow().on('close', () => schedulePersistWindowState.flush())

    // the closed wrapper remains truthy, so clear only the window this callback owns.
    getMainWindow().on('closed', () => {
      closePetOverlay()
      wakeIndicatorController.close()
      introRevealController.destroy()

      if (getMainWindow() === createdMainWindow) {
        setMainWindow(null)
        // the replacement renderer must register before queued links can be delivered.
        clearRendererReadyForDeepLink()
      }
    })

    getStreamThrottle().register(getMainWindow())
    wireCommonWindowHandlers(getMainWindow(), zoomWiringForWindowKind('chat'))

    // Per-window renderer lifecycle diagnostics + recovery (#81290). The reload
    // policy (crashed/oom → bounded reload via the shared rolling budget, then
    // the #38216 Windows sandbox relaunch check on suppression) is the same
    // policy this window used before it moved into the shared helper, so a
    // crashed peer renderer now logs and recovers exactly like the primary one.
    installWindowRendererLifecycle(getMainWindow(), {
      kind: 'main',
      callbacks: {
        log: rememberLog,
        reload: () => {
          getMainWindow().webContents.reload()
        },
        onCrashLoopSuppressed: details => {
          // #38216 renderer flavor (same recovery as #56726, credit @Sahil-SS9):
          // a deterministic Windows renderer crash loop with the sandbox
          // breakpoint signature gets one --no-sandbox relaunch instead of a
          // dead window. Gated on the exit code so unrelated crash loops don't
          // silently drop the sandbox.
          if (
            !shouldRelaunchForRendererSandboxCrashLoop({
              reason: details?.reason,
              exitCode: details?.exitCode,
              alreadyNoSandbox: sandboxState.fallbackActive || alreadyHasNoSandbox(process.argv, process.env),
              relaunchAttempted: sandboxState.noSandboxRelaunchAttempted
            })
          ) {
            return
          }

          sandboxState.noSandboxRelaunchAttempted = true
          sandboxState.fallbackActive = true
          sandboxState.fallbackSticky = true
          sandboxState.fallbackReason = 'renderer-crash-loop'

          try {
            writeSandboxMarker(app.getPath('userData'), fallbackMarker('renderer-crash-loop', app.getVersion()))
          } catch {
            void 0
          }

          rememberLog('[renderer] Windows sandbox crash loop detected; relaunching once with --no-sandbox (#38216)')

          try {
            app.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) })
            void exitAfterBackendShutdown(0)
          } catch (err) {
            rememberLog(`[renderer] --no-sandbox relaunch failed: ${err?.message || err}`)
          }
        },
        // #95575: a renderer that repeatedly fails to load (torn bundle after
        // an update, file locked by AV, missing index.html) used to sit on a
        // white screen with only a desktop.log line. Once the bounded reload
        // budget is exhausted, put the VISIBLE error page in the window so the
        // user sees what is wrong and how to repair it.
        onFailedLoadBudgetExhausted: details => {
          rememberLog(
            `[renderer:main] load-failure budget exhausted; loading visible error page` +
              `${details?.errorCode === undefined ? '' : ` code=${String(details.errorCode)}`}`
          )
          void loadRendererLoadErrorPage(getMainWindow(), {
            errorCode: details?.errorCode,
            url: details?.url,
            errorDescription: 'The desktop renderer failed to load repeatedly after the update.',
            repairHint: 'hermes desktop --force-build',
            reloadUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString()
          })
        },
        // #116472: the OS/Chromium can SIGKILL a renderer while the window is live (memory
        // reclaim, an external kill). Hermes never does this itself and never reloads it
        // (a killed-after-close window must not pop back up), so without this the window sat
        // silent with only a desktop.log line. Surface the reason + a recovery button instead.
        onRendererTerminated: details => {
          // An intentional quit/handoff also tears the renderer down; never pop a
          // recovery page for it (its window may still be alive when this fires).
          if (getIsQuittingForHandoff() || backendShutdown.hasStarted() || getMainWindow().isDestroyed()) {
            return
          }

          const reason = details?.reason ? String(details.reason) : 'unknown'
          const exit = details?.exitCode === undefined ? '' : `, exit code ${String(details.exitCode)}`
          rememberLog(
            `[renderer:main] renderer terminated while live (reason=${reason}${exit}); surfacing recovery page`
          )
          void loadRendererLoadErrorPage(getMainWindow(), {
            title: 'Hermes desktop UI was terminated',
            errorDescription:
              `The desktop UI process was terminated unexpectedly (reason: ${reason}${exit}). ` +
              'Your sessions and the background gateway are unaffected — reload to continue.',
            reloadUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString()
          })
        }
      },
      reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
      reloadMax: RENDERER_RELOAD_MAX,
      recentReloadTimesRef: rendererReloadTimesRef,
      reloadOnFailedLoad: true
    })

    // Electron always passes the event first. The canonical (Electron 36+) shape
    // is (event, messageDetails); the deprecated positional shape is
    // (event, level, message, line, sourceId). Handled in renderer-log.ts, which
    // every renderer-content window shares (#79428: crashes in secondary/HUD/
    // quick-entry windows used to vanish without a trace).
    attachRendererConsoleCapture(getMainWindow(), 'main', rememberLog)

    // #95575: a torn renderer bundle (update replaced the app while its files
    // were locked) loads fine and then dies on the first lazy import — a white
    // screen with no error surface. resolveRendererIndex already logs the torn
    // copies; here we refuse to load one into the PRIMARY window and put the
    // visible repair page in it instead. The Reload button re-attempts the
    // bundle in case the file lock cleared since boot.
    const resolvedRenderer = DEV_SERVER ? null : resolveRendererIndexWithMissing()
    const rendererIndex = resolvedRenderer?.index ?? null
    const tornAssets = resolvedRenderer?.missing ?? []

    if (!DEV_SERVER && rendererIndex && tornAssets.length > 0) {
      rememberLog(
        `[renderer] primary window: chosen renderer bundle ${rendererIndex} is incomplete ` +
          `(${tornAssets.length} missing asset(s)); loading visible repair page instead of a white screen`
      )
      void loadRendererLoadErrorPage(getMainWindow(), {
        errorCode: 'ERR_FILE_NOT_FOUND',
        errorDescription: `The desktop renderer bundle is incomplete after the last update (${tornAssets.length} missing file(s)).`,
        missingAssets: tornAssets,
        repairHint: 'hermes desktop --force-build',
        reloadUrl: pathToFileURL(rendererIndex).toString()
      })
    } else {
      loadWindowUrl(
        getMainWindow(),
        DEV_SERVER || pathToFileURL(rendererIndex || resolveRendererIndex()).toString(),
        'Renderer'
      )
    }

    // Start the Python backend NOW, in parallel with the renderer load — not on
    // did-finish-load. The backend cold boot (spawn → port announce → /api/status)
    // is the dominant startup cost, and serializing it behind Chromium's load
    // added the whole renderer load time to first-usable-composer. The promise is
    // shared (backendConnectionState), so the renderer's getConnection() joins
    // this in-flight boot instead of duplicating it; early boot-progress events
    // the renderer misses are recovered by its getBootProgress() pull on mount.
    const startup = defaultRoute ? connectDesktopProfileRoute(defaultRoute) : startHermes()
    startup.catch(error => rememberLog(error.stack || error.message))

    getMainWindow().webContents.once('did-finish-load', () => {
      // Zoom restore is handled by wireCommonWindowHandlers (shared with session
      // windows); no need to reapply it here.
      firstRunBoot.broadcastBootProgress()
      sendWindowStateChanged()
    })
  }

  return { createWindow }
}
