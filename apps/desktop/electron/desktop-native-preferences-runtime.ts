import fs from 'node:fs'
import path from 'node:path'

// Preserve registration order: appearance/keep-awake listeners register before
// Quick Entry; F12 registers after it. Both use main's live native services.
export function createDesktopNativePreferencesRuntime(deps: {
  GLASS_SUPPORTED: boolean
  GUEST_ONBOARDING: boolean
  SKIP_INTRO: boolean
  TRANSLUCENCY_SUPPORTED: boolean
  app: any
  appearance: any
  createKeepAwake: (...args: any[]) => any
  destroyKeepaliveAgents: () => void
  hudIpc: any
  ipcMain: any
  nativeNotifications: any
  powerSaveBlocker: any
  quitFinalization: any
  rememberLog: (message: string) => void
  sshIsolatedKeepalives: any
}) {
  const {
    GLASS_SUPPORTED,
    GUEST_ONBOARDING,
    SKIP_INTRO,
    TRANSLUCENCY_SUPPORTED,
    app,
    appearance,
    createKeepAwake,
    destroyKeepaliveAgents,
    hudIpc,
    ipcMain,
    nativeNotifications,
    powerSaveBlocker,
    quitFinalization,
    rememberLog,
    sshIsolatedKeepalives
  } = deps

  ipcMain.on('hermes:titlebar-theme', (_event, payload) => appearance.setTitleBarTheme(payload))
  ipcMain.on('hermes:native-theme', (_event, mode) => appearance.setNativeTheme(mode))
  app.on('before-quit', () => appearance.flushTranslucencyWrite())

  // Close the pooled keep-alive sockets on quit so lingering connections can't
  // hold the event loop open or leak FDs past app teardown.
  app.on('will-quit', () => {
    sshIsolatedKeepalives.stopAll()
    destroyKeepaliveAgents()
    nativeNotifications.dispose()
    quitFinalization.arm()
  })

  app.on('quit', () => {
    quitFinalization.cancel()
  })

  // Answered synchronously so preload can publish the verdict before the
  // renderer's first script — see the note there on why it cannot decide this
  // itself. Registered at module scope, which runs long before any window.
  ipcMain.on('hermes:translucency:support', event => {
    event.returnValue = { glass: GLASS_SUPPORTED, translucency: TRANSLUCENCY_SUPPORTED }
  })

  // Launch-flag facts the renderer needs before first paint (same sendSync
  // pattern as translucency). `--local` gates every local-models GUI surface;
  // it arrives from `hermes desktop --local` or directly on Hermes.exe (a
  // shortcut edit), and survives self-relaunches because collectRelaunchArgs
  // only strips internal flags.
  ipcMain.on('hermes:launch-flags', event => {
    event.returnValue = {
      localModels: process.argv.includes('--local') || process.platform === 'win32' || process.platform === 'darwin',
      guestOnboarding: GUEST_ONBOARDING,
      skipIntro: SKIP_INTRO
    }
  })

  ipcMain.on('hermes:translucency', (_event, payload) => {
    appearance.setTranslucency(payload, () => hudIpc.applyHudFrost())
  })

  // Keep-awake: hold the machine awake for long/overnight runs. Main owns the one
  // blocker and its persisted state so a cold launch restores it (applied on
  // ready — powerSaveBlocker needs the app ready). The renderer toggles it from
  // Settings → Advanced over IPC. See store/keep-awake.
  const KEEP_AWAKE_CONFIG_PATH = path.join(app.getPath('userData'), 'keep-awake.json')
  const keepAwake = createKeepAwake(powerSaveBlocker)

  function readPersistedKeepAwake() {
    try {
      return JSON.parse(fs.readFileSync(KEEP_AWAKE_CONFIG_PATH, 'utf8')).on === true
    } catch {
      return false
    }
  }

  ipcMain.on('hermes:keep-awake', (_event, on) => {
    const enabled = Boolean(on)
    keepAwake.set(enabled)

    try {
      fs.mkdirSync(path.dirname(KEEP_AWAKE_CONFIG_PATH), { recursive: true })
      fs.writeFileSync(KEEP_AWAKE_CONFIG_PATH, JSON.stringify({ on: enabled }, null, 2), 'utf8')
    } catch (error) {
      rememberLog(`[keep-awake] write failed: ${error.message}`)
    }
  })

  return { keepAwake, readPersistedKeepAwake }
}

export function registerDesktopF12PreferenceIpc(deps: {
  app: any
  f12State: { blocked: boolean }
  ipcMain: any
  rememberLog: (message: string) => void
}) {
  const { app, f12State, ipcMain, rememberLog } = deps

  // Disable F12 DevTools: maintained in the main process so a cold launch
  // restores it before any window is shown (applied on ready). The renderer
  // toggles it from Settings → Advanced over IPC. See store/disable-f12.
  const DISABLE_F12_CONFIG_PATH = path.join(app.getPath('userData'), 'disable-f12.json')

  function readPersistedDisableF12() {
    try {
      return JSON.parse(fs.readFileSync(DISABLE_F12_CONFIG_PATH, 'utf8')).on === true
    } catch {
      return false
    }
  }

  ipcMain.on('hermes:devtools:disable-f12', (_event, on) => {
    f12State.blocked = Boolean(on)

    try {
      fs.mkdirSync(path.dirname(DISABLE_F12_CONFIG_PATH), { recursive: true })
      fs.writeFileSync(DISABLE_F12_CONFIG_PATH, JSON.stringify({ on: f12State.blocked }, null, 2), 'utf8')
    } catch (error) {
      rememberLog(`[disable-f12] write failed: ${error.message}`)
    }
  })

  return { readPersistedDisableF12 }
}
