// The main process owns window and readiness state. Accessors keep callbacks
// bound to their live values after registration with Electron.
interface DesktopAppLifecycleDeps {
  app: any
  ipcMain: any
  Menu: any
  screen: any
  session: any
  safeStorage: any
  path: any
  tls: any
  pathToFileURL: (...args: any[]) => any
  CHROMIUM_LOG_PATH: string
  CRASH_DIAGNOSTICS: ReturnType<typeof import('./linux-crash-diagnostics').linuxCrashDiagnostics>
  DEV_SERVER: string | undefined
  HERMES_PROTOCOL: string
  IS_MAC: boolean
  backendShutdown: { hasStarted: () => boolean }
  buildApplicationMenu: (...args: any[]) => any
  createWindow: () => any
  ensureLoginShellPath: () => any
  ensureMainWindow: (...args: any[]) => any
  ensureWslWindowsFonts: (...args: any[]) => any
  enableBasicPasswordStoreEncryption: (...args: any[]) => any
  focusWindow: (...args: any[]) => any
  getIsQuittingForHandoff: () => boolean
  getMainWindow: () => any
  getPendingOpenUpdates: () => boolean
  getRendererReadyForDeepLink: () => boolean
  installDownloadHandling: (...args: any[]) => any
  installApplicationMenuAfterFirstWindow: (...args: any[]) => any
  installCommandScreenshot: (...args: any[]) => any
  installEmbedReferer: (...args: any[]) => any
  installHudModifierTap: (...args: any[]) => any
  installMediaPermissions: (...args: any[]) => any
  installPreviewGuestPreload: (...args: any[]) => any
  installRemoteHeaderRules: (...args: any[]) => any
  installWindowsSystemCaTrust: (...args: any[]) => any
  keepAwake: { set: (...args: any[]) => any }
  migrateLegacyEncryptedSecretsOnce: (...args: any[]) => any
  minimizeToTray: any
  openHudWindow: (...args: any[]) => any
  primaryBackendIsRemote: () => boolean
  primaryProfileKey: () => any
  readPersistedDisableF12: () => boolean
  readPersistedKeepAwake: () => boolean
  readQuickEntrySettings: (...args: any[]) => any
  registerMediaProtocol: (...args: any[]) => any
  registerPowerResumeListeners: (...args: any[]) => any
  rememberLog: (message: string) => void
  resolveRendererIndex: () => string
  resumeManagedSshRecoveries: (...args: any[]) => any
  sendOpenUpdatesRequested: (...args: any[]) => any
  setActiveGatewayProfile: (...args: any[]) => any
  setF12Blocked: (blocked: boolean) => void
  setPendingOpenUpdates: (pending: boolean) => void
  setRendererReadyForDeepLink: (ready: boolean) => void
  setWslBridgeProfileState: (...args: any[]) => any
  startChromiumLogWatcher: (...args: any[]) => any
  wakeIndicatorController: { reposition: () => void }
  applyQuickEntrySettings: (...args: any[]) => any
}

export function createDesktopAppLifecycleRuntime(deps: DesktopAppLifecycleDeps) {
  const {
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
  } = deps

  // ---------------------------------------------------------------------------
  // hermes:// deep links (e.g. hermes://blueprint/morning-brief?time=08:00,
  // hermes://mcp/install?name=NAME&config=B64 — the vendor "Add to Hermes"
  // button, or hermes://plugin/install?repo=owner/repo). Dev
  // (`HERMES_DESKTOP_DEV_SERVER`) registers hermes-dev:// instead — bare
  // Electron or a stale OS handler often owns hermes:// on dev machines.
  // Parsing is generic ({kind, name, params}); the renderer routes per kind
  // and anything install-shaped requires explicit user confirmation there.
  // A docs/dashboard "Send to App" button opens this URL; we route it into the
  // running app. Three delivery paths: macOS 'open-url',
  // Win/Linux running-app 'second-instance' (argv), Win/Linux cold-start argv.
  // ---------------------------------------------------------------------------
  /** Schemes accepted when parsing inbound URLs (dev accepts both). */
  const DEEPLINK_SCHEMES = DEV_SERVER ? ['hermes-dev', 'hermes'] : ['hermes']
  let _pendingDeepLink: { kind: string; name: string; params: Record<string, string> } | null = null

  function _extractDeepLink(argv) {
    if (!Array.isArray(argv)) {
      return null
    }

    return argv.find(a => typeof a === 'string' && DEEPLINK_SCHEMES.some(s => a.startsWith(`${s}://`))) || null
  }

  function handleDeepLink(url) {
    if (!url || typeof url !== 'string') {
      return
    }

    let parsed

    try {
      parsed = new URL(url)
    } catch {
      rememberLog(`[deeplink] ignoring malformed url: ${url}`)

      return
    }

    const scheme = parsed.protocol.replace(/:$/, '')

    if (!DEEPLINK_SCHEMES.includes(scheme)) {
      rememberLog(`[deeplink] ignoring scheme ${scheme} (expected ${DEEPLINK_SCHEMES.join(' or ')})`)

      return
    }

    // hermes://blueprint/<key>?slot=val  -> host="blueprint", path="/<key>"
    const kind = parsed.hostname || ''
    const name = decodeURIComponent((parsed.pathname || '').replace(/^\//, ''))
    const params = {}
    parsed.searchParams.forEach((v, k) => {
      params[k] = v
    })
    const payload = { kind, name, params }

    if (!getRendererReadyForDeepLink() || !getMainWindow() || getMainWindow().isDestroyed()) {
      _pendingDeepLink = payload

      return
    }

    try {
      if (getMainWindow().isMinimized()) {
        getMainWindow().restore()
      }

      getMainWindow().focus()
      getMainWindow().webContents.send('hermes:deep-link', payload)
      rememberLog(`[deeplink] delivered ${kind}/${name}`)
    } catch (err) {
      rememberLog(`[deeplink] delivery failed: ${err.message}`)
    }
  }

  // Renderer calls this (via IPC) once it has mounted its deep-link listener, so
  // a link that arrived during boot/install is flushed exactly once.
  ipcMain.handle('hermes:deep-link-ready', () => {
    setRendererReadyForDeepLink(true)

    if (getPendingOpenUpdates()) {
      setPendingOpenUpdates(false)
      sendOpenUpdatesRequested()
    }

    if (_pendingDeepLink) {
      const queued = _pendingDeepLink
      _pendingDeepLink = null
      handleDeepLink(
        `${HERMES_PROTOCOL}://${queued.kind}/${encodeURIComponent(queued.name)}` +
          (Object.keys(queued.params).length ? '?' + new URLSearchParams(queued.params).toString() : '')
      )
    }

    return { ok: true }
  })

  function registerDeepLinkProtocol() {
    try {
      if (process.defaultApp && process.argv.length >= 2) {
        // Dev: register with the electron exec path + entry script so the OS can
        // relaunch us with the URL. argv[1] is usually "." when launched via
        // `electron .` from apps/desktop — resolve against cwd.
        const entry = path.resolve(process.argv[1])
        app.setAsDefaultProtocolClient(HERMES_PROTOCOL, process.execPath, [entry])
      } else {
        app.setAsDefaultProtocolClient(HERMES_PROTOCOL)
      }

      rememberLog(`[deeplink] registered ${HERMES_PROTOCOL}:// handler`)
    } catch (err) {
      rememberLog(`[deeplink] protocol registration failed: ${err.message}`)
    }
  }

  // Single-instance lock: deep links on a running app (Win/Linux) arrive as a
  // second-instance argv. Without the lock a second `hermes://` launch spawns a
  // whole new app instead of routing into the running one.
  const _gotSingleInstanceLock = app.requestSingleInstanceLock()
  const isPrimaryInstance = _gotSingleInstanceLock

  if (!isPrimaryInstance) {
    // Hard-exit, not app.quit(): the before-quit teardown coordinator defers a
    // plain quit (event.preventDefault + async backend shutdown), and in that
    // window `ready` still fires — the lock-losing instance then runs the full
    // startup (shortcut registration, createWindow → startHermes), whose
    // reapOrphans() SIGTERMs the running instance's live backend (#87295).
    // app.exit() terminates immediately, before `ready`, so a second launch
    // routes into the running window and never touches backend machinery.
    app.exit(0)
  } else {
    app.on('second-instance', (_event, argv) => {
      const url = _extractDeepLink(argv)

      if (url) {
        handleDeepLink(url)
      }

      ensureMainWindow(getMainWindow(), {
        isReady: app.isReady(),
        createWindow,
        focusWindow,
        // deep-link delivery focuses a live window after its renderer is ready.
        focusExisting: !url
      })
    })
  }

  // macOS delivers deep links via 'open-url' — register early (can fire before
  // whenReady; handleDeepLink queues until the renderer is ready).
  app.on('open-url', (event, url) => {
    event.preventDefault()
    handleDeepLink(url)
  })

  app.whenReady().then(() => {
    // Warm the login-shell PATH resolution immediately so it usually completes
    // before the backend start path awaits the same single-flight promise.
    void ensureLoginShellPath()

    if (CRASH_DIAGNOSTICS) {
      startChromiumLogWatcher(CHROMIUM_LOG_PATH)
    }

    const systemCa = installWindowsSystemCaTrust(tls)

    if (systemCa.applied) {
      rememberLog(
        `[tls] trusting ${systemCa.systemCertificateCount} Windows system CA certificate(s) for backend connections`
      )
    } else if (systemCa.error) {
      rememberLog(`[tls] could not load Windows system CA certificates: ${systemCa.error}`)
    }

    // Keyring-less Linux `--password-store=basic` support. This must run before
    // createWindow() and anything that could touch safeStorage; the narrow
    // platform/switch/guard semantics live in the extracted helper.
    enableBasicPasswordStoreEncryption({
      platform: process.platform,
      passwordStoreSwitch: app.commandLine.getSwitchValue('password-store'),
      safeStorageApi: safeStorage
    })

    // Keychain encryption is opt-in (default OFF). One-shot: rewrite any
    // legacy safeStorage-encrypted secrets as plain so no later launch ever
    // touches the OS keychain unless the user turns encryption on in
    // Settings → Gateway. Must run before createWindow() and the first
    // connection resolution.
    migrateLegacyEncryptedSecretsOnce()

    installMediaPermissions()
    installDownloadHandling()
    registerMediaProtocol()
    installEmbedReferer()
    installRemoteHeaderRules()
    registerDeepLinkProtocol()
    installPreviewGuestPreload()

    ensureWslWindowsFonts()
    configureSpellChecker()
    registerPowerResumeListeners()
    keepAwake.set(readPersistedKeepAwake())
    void minimizeToTray.start()
    setF12Blocked(readPersistedDisableF12())
    // Seed this before the first window exists: a picker can open before
    // startHermes() finishes resolving the configured backend.
    const primaryProfile = primaryProfileKey()

    setActiveGatewayProfile(primaryProfile)
    setWslBridgeProfileState(primaryProfile, !primaryBackendIsRemote())
    // Quick Entry's global chord — registered on ready so a cold launch restores
    // it without the renderer visiting Settings. A failed registration is logged
    // here and surfaced in Settings via the IPC state (never silent).
    applyQuickEntrySettings(readQuickEntrySettings())
    installCommandScreenshot({ rendererUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString() })
    installHudModifierTap({
      rendererUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString(),
      summon: () => {
        if (!getIsQuittingForHandoff() && !backendShutdown.hasStarted()) {
          openHudWindow(null, null)
        }
      }
    })

    if (IS_MAC) {
      const reposition = () => wakeIndicatorController.reposition()

      screen.on('display-added', reposition)

      screen.on('display-metrics-changed', reposition)

      screen.on('display-removed', reposition)
    }

    // A hard crash can interrupt the in-memory restore loop after exact remote
    // serves were drained. The owner-only recovery journal survives that crash;
    // its worker waits for the install marker to clear, then reopens every scope
    // captured by the original transaction before removing the journal entry.
    void resumeManagedSshRecoveries()
    installApplicationMenuAfterFirstWindow({
      isMac: IS_MAC,
      buildMenu: buildApplicationMenu,
      setApplicationMenu: menu => Menu.setApplicationMenu(menu),
      createWindow
    })

    // Win/Linux cold start: the launching hermes:// URL is in our own argv.
    const _coldStartLink = _extractDeepLink(process.argv)

    if (_coldStartLink) {
      handleDeepLink(_coldStartLink)
    }

    app.on('activate', () => {
      // Recreate the primary window if it's gone. Guard on mainWindow directly
      // (not just total window count) so a dock click still restores the main
      // window when only secondary session windows remain open.
      if (!getMainWindow() || getMainWindow().isDestroyed()) {
        createWindow()
      } else {
        focusWindow(getMainWindow())
      }
    })
  })

  // Seed Chromium's spellchecker with the system locale (falling back to en-US).
  // On macOS Electron uses the native spellchecker which ignores this list, but
  // on Windows/Linux Chromium downloads Hunspell dictionaries on demand and
  // won't enable any without an explicit language.
  function configureSpellChecker() {
    try {
      const defaultSession = session.defaultSession

      if (!defaultSession || typeof defaultSession.setSpellCheckerLanguages !== 'function') {
        return
      }

      const available = defaultSession.availableSpellCheckerLanguages || []
      const locale = (app.getLocale && app.getLocale()) || 'en-US'
      const candidates = [locale, locale.split('-')[0], 'en-US', 'en']
      const chosen = candidates.find(lang => available.includes(lang)) || 'en-US'

      defaultSession.setSpellCheckerLanguages([chosen])
    } catch (error) {
      rememberLog(`Spellchecker setup failed: ${error.message}`)
    }
  }

  return { isPrimaryInstance, handleDeepLink }
}
