import fs from 'node:fs'
import path from 'node:path'

import { GIT_UNUSABLE } from './select-runnable-binary'

interface BackendPoolIpcDeps {
  ipcMain: any
  touchPoolBackend: (...args: any[]) => void
  getPoolLimits: () => { maxBackends: number; idleMs: number }
  setPoolLimits: (limits: { maxBackends: number; idleMs: number }) => any
  gatewayWsUrlIpcResult: (getUrl: () => Promise<any>) => any
  freshGatewayWsUrl: (profile: any) => Promise<any>
}

export function registerBackendPoolIpc(deps: BackendPoolIpcDeps) {
  const { ipcMain, touchPoolBackend, getPoolLimits, setPoolLimits, gatewayWsUrlIpcResult, freshGatewayWsUrl } = deps

  ipcMain.handle('hermes:backend:touch', async (_event, profile, options) => {
    touchPoolBackend(profile, options)

    return { ok: true }
  })
  // Pool sizing (Settings → Advanced): device-local, live-applied. Main is
  // authoritative (it owns the pool and the persisted copy); the returned
  // limits are what actually took effect post-clamp.
  ipcMain.handle('hermes:pool-limits:get', async () => ({ ...getPoolLimits() }))
  ipcMain.handle('hermes:pool-limits:set', async (_event, raw) => {
    const poolLimits = getPoolLimits()

    const next = setPoolLimits({
      maxBackends: typeof raw?.maxBackends === 'number' ? raw.maxBackends : poolLimits.maxBackends,
      idleMs: typeof raw?.idleMs === 'number' ? raw.idleMs : poolLimits.idleMs
    })

    return { ok: true, limits: next }
  })
  ipcMain.handle('hermes:gateway:ws-url', async (_event, profile) => {
    return gatewayWsUrlIpcResult(() => freshGatewayWsUrl(profile))
  })
}

interface WorkspaceAndLogIpcDeps {
  ipcMain: any
  app: any
  dialog: any
  shell: any
  readDefaultProjectDir: () => string | null
  resolveHermesCwd: () => string
  sanitizeWorkspaceCwd: (cwd: any) => any
  writeDefaultProjectDir: (dir: string | null) => void
  mkdirSync: (dir: string, options: { recursive: boolean }) => void
  desktopLogPath: string
  fileExists: (filePath: string) => boolean
  appendFile: (filePath: string, content: string) => Promise<any>
  hermesLog: string[]
  rememberLog: (line: string) => void
  formatRendererBoundaryReport: (...args: any[]) => string
  flushDesktopLogBufferSync: () => void
  fetchLinkTitle: (url: any) => any
  resolveFaviconCached: (url: any) => any
  registerFsIpc: (deps: any) => void
  registerGitIpc: (deps: any) => void
  registerMcpOauthCallbackIpc: () => void
  registerTerminalIpc: (deps: any) => any
  hermesHome: string
  readActiveDesktopProfile: () => any
  expandUserPath: (...args: any[]) => any
  resolveRequestedPathForIpc: (...args: any[]) => any
  directoryExists: (filePath: string) => boolean
  resolveGitBinary: () => string
  resolveGhBinary: () => string
  isWindows: boolean
  findOnPath: (command: string) => string | null
  activeSshTerminalTarget: (...args: any[]) => any
  ensureTerminalBackend: (webContentsId: number) => any
  getSshConnectionState: (scope: string) => any
}

export function registerWorkspaceAndLogIpc(deps: WorkspaceAndLogIpcDeps) {
  const {
    ipcMain, app, dialog, shell, readDefaultProjectDir, resolveHermesCwd, sanitizeWorkspaceCwd,
    writeDefaultProjectDir, mkdirSync, desktopLogPath, fileExists, appendFile, hermesLog,
    rememberLog, formatRendererBoundaryReport, flushDesktopLogBufferSync, fetchLinkTitle,
    resolveFaviconCached, registerFsIpc, registerGitIpc, registerMcpOauthCallbackIpc,
    registerTerminalIpc, hermesHome, readActiveDesktopProfile, expandUserPath,
    resolveRequestedPathForIpc, directoryExists, resolveGitBinary, resolveGhBinary,
    isWindows, findOnPath, activeSshTerminalTarget, ensureTerminalBackend, getSshConnectionState
  } = deps

  // User-configurable default project directory. The renderer reads this on
  // settings mount and seeds the value into the picker; writing back persists
  // it via writeDefaultProjectDir so resolveHermesCwd picks it up on the next
  // session spawn (no app restart needed).
  ipcMain.handle('hermes:setting:defaultProjectDir:get', async () => ({
    dir: readDefaultProjectDir(),
    defaultLabel: app.getPath('home'),
    resolvedCwd: resolveHermesCwd()
  }))

  ipcMain.handle('hermes:workspace:sanitize', async (_event, cwd) => sanitizeWorkspaceCwd(cwd))

  ipcMain.handle('hermes:setting:defaultProjectDir:set', async (_event, dir) => {
    const next = typeof dir === 'string' && dir.trim() ? dir.trim() : null

    if (next) {
      try {
        mkdirSync(next, { recursive: true })
      } catch (error) {
        throw new Error(`Could not create directory: ${error.message}`)
      }
    }

    writeDefaultProjectDir(next)

    return { dir: next }
  })
  ipcMain.handle('hermes:setting:defaultProjectDir:pick', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Choose default project directory',
      properties: ['openDirectory', 'createDirectory'],
      defaultPath: readDefaultProjectDir() || app.getPath('home')
    })

    if (result.canceled || result.filePaths.length === 0) {
      return { canceled: true, dir: null }
    }

    return { canceled: false, dir: result.filePaths[0] }
  })

  ipcMain.handle('hermes:fetchLinkTitle', (_event, url) => fetchLinkTitle(url))
  ipcMain.handle('hermes:resolveFavicon', (_event, url) => resolveFaviconCached(url))

  ipcMain.handle('hermes:logs:reveal', async () => {
    try {
      await fs.promises.mkdir(path.dirname(desktopLogPath), { recursive: true })

      if (!fileExists(desktopLogPath)) {
        await appendFile(desktopLogPath, '')
      }

      shell.showItemInFolder(desktopLogPath)

      return { ok: true, path: desktopLogPath }
    } catch (error) {
      return { ok: false, path: desktopLogPath, error: error.message }
    }
  })

  ipcMain.handle('hermes:logs:recent', async () => ({ path: desktopLogPath, lines: hermesLog.slice(-200) }))

  // Renderer error-boundary catches (#79428 defect B): the component stack only
  // exists in renderer memory, so the boundary posts it here and we persist it
  // via the desktop.log pipeline. `on`, not `handle` — the sender may be mid-
  // crash and must not await. Flush immediately: a crashing window can be gone
  // before the debounced flush timer fires.
  ipcMain.on('hermes:logs:renderer-error', (_event, report) => {
    const { label, boundary, message, componentStack } = report && typeof report === 'object' ? report : {}
    rememberLog(formatRendererBoundaryReport(label, boundary, message, componentStack))
    flushDesktopLogBufferSync()
  })

  // Local filesystem + plugin-root IPC (readDir/reveal/rename/trash/…) — see fs-ipc.ts.
  registerFsIpc({
    hermesHome,
    readActiveDesktopProfile,
    expandUserPath,
    resolveRequestedPathForIpc,
    directoryExists,
    resolveGitBinary
  })

  // Git-driven features (worktrees, review pane, repo scan) — see git-ipc.ts.
  registerGitIpc({ resolveGitBinary, resolveGhBinary })

  // Client-side loopback callback for MCP OAuth against remote backends — see
  // mcp-oauth-callback-ipc.ts.
  registerMcpOauthCallbackIpc()

  // Embedded terminal PTY host (hermes:terminal:*) — see terminal-ipc.ts.
  return registerTerminalIpc({
    isWindows,
    findOnPath,
    rememberLog,
    activeSshTerminalTarget,
    ensureBackend: webContentsId => ensureTerminalBackend(webContentsId),
    getSshConnectionState
  })
}

interface DesktopOperationsIpcDeps {
  ipcMain: any
  checkUpdates: (opts: { force: boolean }) => Promise<any>
  applyUpdates: (payload: any) => Promise<any>
  readDesktopUpdateConfig: () => any
  writeDesktopUpdateConfig: (config: { branch: string }) => void
  defaultUpdateBranch: string
  desktopShellRuntime: any
  fetchMarketplaceThemes: (id: string) => Promise<any>
  searchMarketplaceThemes: (query: string, limit: number) => Promise<any>
}

export function registerDesktopOperationsIpc(deps: DesktopOperationsIpcDeps) {
  const {
    ipcMain, checkUpdates, applyUpdates, readDesktopUpdateConfig, writeDesktopUpdateConfig,
    defaultUpdateBranch, desktopShellRuntime, fetchMarketplaceThemes, searchMarketplaceThemes
  } = deps

  ipcMain.handle('hermes:updates:check', async (_event, opts) =>
    checkUpdates({ force: Boolean(opts?.force) }).catch(error => ({
      supported: true,
      branch: readDesktopUpdateConfig().branch,
      error: error?.kind === GIT_UNUSABLE ? GIT_UNUSABLE : 'check-failed',
      message: error?.message || String(error),
      fetchedAt: Date.now()
    }))
  )

  ipcMain.handle('hermes:updates:apply', async (_event, payload) =>
    applyUpdates(payload || {}).catch(error => ({
      ok: false,
      error: 'apply-failed',
      message: error?.message || String(error)
    }))
  )

  ipcMain.handle('hermes:updates:branch:get', async () => readDesktopUpdateConfig())

  ipcMain.handle('hermes:updates:branch:set', async (_event, name) => {
    const branch = typeof name === 'string' && name.trim() ? name.trim() : defaultUpdateBranch
    writeDesktopUpdateConfig({ branch })

    return { branch }
  })

  ipcMain.handle('hermes:version', async () => desktopShellRuntime.getVersionInfo())
  ipcMain.handle('hermes:app:relaunch', async () => desktopShellRuntime.relaunchAfterBundleSwap())
  ipcMain.handle('hermes:machine:profile', async () => desktopShellRuntime.getMachineProfile())
  ipcMain.handle('hermes:uninstall:summary', async () => desktopShellRuntime.getUninstallSummary())
  ipcMain.handle('hermes:uninstall:run', async (_event, payload) => {
    const mode = payload && typeof payload === 'object' ? payload.mode : payload

    return desktopShellRuntime.runDesktopUninstall(String(mode || ''))
  })

  // Download a VS Code Marketplace extension and return the raw color-theme JSON
  // it contributes. No theme code is executed — we only read JSON from the .vsix.
  ipcMain.handle('hermes:vscode-theme:fetch', async (_event, id) => fetchMarketplaceThemes(String(id || '')))

  // Search the Marketplace for color-theme extensions (empty query = top installs).
  ipcMain.handle('hermes:vscode-theme:search', async (_event, query) => searchMarketplaceThemes(String(query || ''), 20))
}
