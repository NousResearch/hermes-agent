// Quick Entry's OS shortcut and live settings remain owned by the shell
// overlay runtime. These channels register after HUD IPC, as in main.
export function registerDesktopQuickEntryIpc(deps: {
  applyQuickEntrySettings: (...args: any[]) => any
  getMainWindow: () => Electron.BrowserWindow | null
  hideQuickEntryWindow: () => void
  ipcMain: any
  readQuickEntrySettings: (...args: any[]) => any
  rememberLog: (message: string) => void
  sanitizeQuickEntrySettings: (...args: any[]) => any
  shellOverlayRuntime: any
  writeQuickEntrySettings: (...args: any[]) => any
}) {
  const {
    applyQuickEntrySettings,
    getMainWindow,
    hideQuickEntryWindow,
    ipcMain,
    readQuickEntrySettings,
    rememberLog,
    sanitizeQuickEntrySettings,
    shellOverlayRuntime,
    writeQuickEntrySettings
  } = deps

  // Quick Entry: the renderer reads the live registration state on settings mount
  // and writes the preference back. Main is authoritative — it owns the OS
  // accelerator — so both handlers return the state that ACTUALLY resulted,
  // including `registered: false` + `error: 'taken'` when another app owns the
  // chord. See electron/quick-entry.ts + store/quick-entry.
  ipcMain.handle('hermes:quick-entry:settings:get', async () => {
    const settings = readQuickEntrySettings()
    const state = shellOverlayRuntime.currentQuickEntryShortcutState()

    // Ground truth is what the last apply produced; the shortcut we report is the
    // live one (a saved-but-rejected chord still shows what the user asked for).
    return {
      enabled: settings.enabled,
      error: state.error,
      registered: state.registered,
      shortcut: settings.enabled ? state.shortcut : settings.shortcut
    }
  })

  ipcMain.handle('hermes:quick-entry:settings:set', async (_event, patch) => {
    const current = readQuickEntrySettings()

    const next = sanitizeQuickEntrySettings({
      enabled: patch?.enabled === undefined ? current.enabled : patch.enabled === true,
      shortcut: typeof patch?.shortcut === 'string' && patch.shortcut.trim() ? patch.shortcut : current.shortcut
    })

    writeQuickEntrySettings(next)

    return applyQuickEntrySettings(next)
  })

  // Quick window → main → PRIMARY renderer. We never submit here: the renderer
  // owns the one prompt-submit path, and forwarding keeps it that way. The
  // payload is `{ target, text }` — target routing (current chat / a picked
  // session / new) is the renderer's job too.
  ipcMain.on('hermes:quick-entry:submit', (_event, payload) => {
    hideQuickEntryWindow()

    const text = typeof payload?.text === 'string' ? payload.text.trim() : ''

    if (!text) {
      return
    }

    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      rememberLog('[quick-entry] dropped a submit: no primary window to route it to')

      return
    }

    // Deliberately does NOT raise/focus the main window — the user asked to fire
    // a prompt from wherever they were, not to be yanked into the app.
    mainWindow.webContents.send('hermes:quick-entry:submit', {
      target: typeof payload?.target === 'string' && payload.target ? payload.target : 'current',
      text
    })
  })

  // Primary renderer → main → quick window: gateway connection state + the
  // recent-session list for the target picker. Cached so a quick window spawned
  // AFTER the last push still boots from truth instead of "disconnected".
  ipcMain.on('hermes:quick-entry:state', (_event, payload) => {
    shellOverlayRuntime.pushQuickEntryState(payload)
  })

  ipcMain.on('hermes:quick-entry:dismiss', () => hideQuickEntryWindow())

}
