import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registerBackendPoolIpc, registerDesktopOperationsIpc, registerWorkspaceAndLogIpc } from './desktop-operational-ipc'

test('backend pool settings keep current values when a patch omits a field', async () => {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  const limits = { maxBackends: 4, idleMs: 120_000 }
  registerBackendPoolIpc({
    ipcMain: { handle: (name, fn) => handlers.set(name, fn) },
    touchPoolBackend: () => {},
    getPoolLimits: () => limits,
    setPoolLimits: next => next,
    freshGatewayWsUrl: async () => 'ws://localhost',
    gatewayWsUrlIpcResult: fn => fn()
  } as any)

  assert.deepEqual(await handlers.get('hermes:pool-limits:set')!(null, { idleMs: 250_000 }), {
    ok: true,
    limits: { maxBackends: 4, idleMs: 250_000 }
  })
  assert.equal(await handlers.get('hermes:gateway:ws-url')!(null, 'main'), 'ws://localhost')
})

test('workspace and log IPC persists the selected directory and flushes renderer errors', async () => {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  const listeners = new Map<string, (...args: any[]) => void>()
  const writes: Array<string | null> = []
  const calls: string[] = []
  registerWorkspaceAndLogIpc({
    ipcMain: {
      handle: (name, fn) => handlers.set(name, fn),
      on: (name, fn) => listeners.set(name, fn)
    },
    app: { getPath: () => 'C:/home' },
    dialog: { showOpenDialog: async () => ({ canceled: true, filePaths: [] }) },
    shell: { showItemInFolder: () => {} },
    readDefaultProjectDir: () => null,
    resolveHermesCwd: () => 'C:/home',
    sanitizeWorkspaceCwd: value => value,
    writeDefaultProjectDir: value => { writes.push(value) },
    mkdirSync: value => { calls.push(`mkdir:${value}`) },
    desktopLogPath: 'C:/home/desktop.log',
    fileExists: () => true,
    appendFile: async () => {},
    hermesLog: [],
    rememberLog: message => { calls.push(message) },
    formatRendererBoundaryReport: () => 'renderer boundary',
    flushDesktopLogBufferSync: () => { calls.push('flushed') },
    fetchLinkTitle: () => '',
    resolveFaviconCached: () => '',
    registerFsIpc: () => {},
    registerGitIpc: () => {},
    registerMcpOauthCallbackIpc: () => {},
    registerTerminalIpc: () => ({ disposeTerminalSession: () => {} })
  } as any)

  assert.deepEqual(await handlers.get('hermes:setting:defaultProjectDir:set')!(null, ' C:/work '), { dir: 'C:/work' })
  assert.deepEqual(writes, ['C:/work'])
  assert.deepEqual(calls, ['mkdir:C:/work'])
  listeners.get('hermes:logs:renderer-error')!(null, { message: 'boom' })
  assert.deepEqual(calls.slice(1), ['renderer boundary', 'flushed'])
})

test('desktop update and app IPC preserve recoverable result contracts', async () => {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  registerDesktopOperationsIpc({
    ipcMain: { handle: (name, fn) => handlers.set(name, fn) },
    checkUpdates: async () => { throw new Error('offline') },
    applyUpdates: async () => { throw new Error('apply broke') },
    readDesktopUpdateConfig: () => ({ branch: 'main' }),
    writeDesktopUpdateConfig: () => {},
    defaultUpdateBranch: 'main',
    desktopShellRuntime: {
      getVersionInfo: () => ({ version: '1.0' }),
      relaunchAfterBundleSwap: () => {},
      getMachineProfile: () => ({}),
      getUninstallSummary: () => ({}),
      runDesktopUninstall: () => ({})
    },
    fetchMarketplaceThemes: () => ({}),
    searchMarketplaceThemes: () => []
  } as any)

  assert.match((await handlers.get('hermes:updates:check')!(null, {})).message, /offline/)
  assert.deepEqual(await handlers.get('hermes:updates:apply')!(null, {}), {
    ok: false,
    error: 'apply-failed',
    message: 'apply broke'
  })
  assert.deepEqual(await handlers.get('hermes:version')!(), { version: '1.0' })
})
