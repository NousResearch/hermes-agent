import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

// Register native window, terminal handoff, wake indicator, and zoom IPC in
// the same order as main's original call site. Main retains live services.
export function registerDesktopWindowIpcRuntime(deps: {
  BrowserWindow: any
  DEFAULT_ZOOM_LEVEL: number
  HERMES_HOME: string
  app: any
  buildTerminalScript: (...args: any[]) => any
  createBrowserWindow: (...args: any[]) => any
  createInstanceWindow: (...args: any[]) => any
  createSessionWindow: (...args: any[]) => any
  findOnPath: (...args: any[]) => any
  ipcMain: any
  percentToZoomLevel: (...args: any[]) => any
  registerWindowControlIpc: (...args: any[]) => any
  rememberLog: (message: string) => void
  resolveHermesBackend: (...args: any[]) => Promise<any>
  resolveTerminalLaunch: (...args: any[]) => any
  sanitizeWorkspaceCwd: (...args: any[]) => any
  setAndPersistZoomLevel: (...args: any[]) => any
  terminalScriptEnv: (...args: any[]) => any
  terminalScriptExtension: (...args: any[]) => any
  tuiResumeArgs: (...args: any[]) => any
  wakeIndicatorController: any
  zoomLevelToPercent: (...args: any[]) => any
}) {
  const {
    BrowserWindow,
    DEFAULT_ZOOM_LEVEL,
    HERMES_HOME,
    app,
    buildTerminalScript,
    createBrowserWindow,
    createInstanceWindow,
    createSessionWindow,
    findOnPath,
    ipcMain,
    percentToZoomLevel,
    registerWindowControlIpc,
    rememberLog,
    resolveHermesBackend,
    resolveTerminalLaunch,
    sanitizeWorkspaceCwd,
    setAndPersistZoomLevel,
    terminalScriptEnv,
    terminalScriptExtension,
    tuiResumeArgs,
    wakeIndicatorController,
    zoomLevelToPercent
  } = deps

  ipcMain.handle('hermes:window:openSession', async (_event, sessionId, opts) => {
    if (typeof sessionId !== 'string' || !sessionId.trim()) {
      return { ok: false, error: 'invalid-session-id' }
    }

    createSessionWindow(sessionId.trim(), {
      profile: typeof opts?.profile === 'string' ? opts.profile : null,
      watch: opts?.watch === true
    })

    return { ok: true }
  })
  ipcMain.handle('hermes:window:openInstance', async (event, options) => {
    createInstanceWindow(options, BrowserWindow.fromWebContents(event.sender))

    return { ok: true }
  })
  registerWindowControlIpc(ipcMain, sender => BrowserWindow.fromWebContents(sender))
  ipcMain.handle('hermes:window:openBrowser', async (_event, tabId) => {
    if (typeof tabId !== 'string' || !tabId.trim()) {
      return { ok: false, error: 'invalid-tab-id' }
    }

    createBrowserWindow(tabId.trim())

    return { ok: true }
  })

  // Hand a session to the user's OWN terminal emulator, running the TUI against
  // it (`hermes --tui --resume <id>`). Not the in-app terminal pane: the point is
  // to continue the chat in the terminal they already live in.
  //
  // The desktop's runtime is usually a venv Python invoked as
  // `python -m hermes_cli.main`, so we resolve the SAME backend the app itself
  // launches and carry its argv + PYTHONPATH into a launcher script rather than
  // hoping a `hermes` exists on the user's interactive PATH. Resolution only —
  // never ensureRuntime(), which would kick off a first-run install from a menu
  // click; an unresolved runtime is reported instead.
  ipcMain.handle('hermes:window:openInTerminal', async (_event, sessionId, opts) => {
    if (typeof sessionId !== 'string' || !sessionId.trim()) {
      return { ok: false, error: 'invalid-session-id' }
    }

    try {
      const profile = typeof opts?.profile === 'string' ? opts.profile.trim() : ''
      const backend = await resolveHermesBackend(tuiResumeArgs(sessionId.trim(), profile || undefined))

      if (!backend.command) {
        return { ok: false, error: 'Hermes is not installed yet' }
      }

      const { cwd } = sanitizeWorkspaceCwd(opts?.cwd)
      const scriptDir = path.join(app.getPath('userData'), 'open-in-terminal')
      fs.mkdirSync(scriptDir, { recursive: true })

      const scriptPath = path.join(
        scriptDir,
        `hermes-${crypto.randomBytes(6).toString('hex')}${terminalScriptExtension()}`
      )

      fs.writeFileSync(
        scriptPath,
        buildTerminalScript({
          args: backend.args,
          command: backend.command,
          cwd,
          env: terminalScriptEnv(backend.env, HERMES_HOME)
        }),
        { mode: 0o700 }
      )

      const launch = resolveTerminalLaunch({ findOnPath, scriptPath })

      if (!launch) {
        return { ok: false, error: 'No terminal emulator found' }
      }

      rememberLog(`[terminal] opening session ${sessionId} via ${launch.command}`)

      // Detached + unref'd: the terminal window outlives the desktop app, and
      // never inherits our stdio (a closed pipe would kill the TUI).
      const child = spawn(launch.command, launch.args, { detached: true, stdio: 'ignore' })
      child.unref()

      return { ok: true }
    } catch (error) {
      rememberLog(`[terminal] open in terminal failed: ${error.message}`)

      return { ok: false, error: error.message }
    }
  })
  ipcMain.handle('hermes:wake-indicator:get', () => wakeIndicatorController.getState())
  ipcMain.on('hermes:wake-indicator:set', (_event, state) => {
    wakeIndicatorController.setState(state)
  })

  // --- Text size (zoom) -------------------------------------------------------
  // The settings UI drives the same clamped zoom scale as the Ctrl/Cmd
  // shortcuts and the View menu. Reads and writes target the asking window.
  ipcMain.handle('hermes:zoom:get', event => {
    const window = BrowserWindow.fromWebContents(event.sender)

    const level = window && !window.isDestroyed() ? window.webContents.getZoomLevel() : DEFAULT_ZOOM_LEVEL

    return { level, percent: zoomLevelToPercent(level) }
  })
  ipcMain.on('hermes:zoom:set-percent', (event, percent) => {
    const window = BrowserWindow.fromWebContents(event.sender)

    if (!window || window.isDestroyed()) {
      return
    }

    setAndPersistZoomLevel(window, percentToZoomLevel(Number(percent)))
  })

}
