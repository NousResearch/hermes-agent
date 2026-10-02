/** Callable tests of main's actual route/IPC nodes, without loading Electron main.
 * Only those AST nodes are transpiled. Filesystem, IPC, windows and backend
 * starts are explicit fixture boundaries; safe policy helpers are real imports.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'

import ts from 'typescript'
import { test, vi } from 'vitest'

import { BackendDialClaims, runForegroundRetryingDialClaim } from './backend-dial-claim'
import { backendScopeKey, registryDialConnectionId, resolvedConnectionId } from './connection-registry'
import { liveWindowState, overlayWindowState } from './connection-window-state'
import { resolveDesktopConnectionRequest } from './desktop-profile'
import {
  assertPoolEntryStillOwned,
  isBackgroundCapacitySkip,
  isBackgroundSlotRetryDeferred,
  LocalBackendBackgroundCapacityError
} from './pool-spawn-coordinator'

type Handler = (...args: any[]) => Promise<any>

function fixture() {
  const source = fs.readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
  const ast = ts.createSourceFile('main.ts', source, ts.ScriptTarget.Latest, true)

  const declarations = ast.statements.filter(
    node =>
      ts.isFunctionDeclaration(node) &&
      ['spawnPriorityFrom', 'connectDesktopProfileRoute'].includes(node.name?.text ?? '')
  )

  const channels = ['hermes:connection', 'hermes:connection:for']

  const registrations = ast.statements.filter(node => {
    if (!ts.isExpressionStatement(node) || !ts.isCallExpression(node.expression)) {
      return false
    }
    const call = node.expression

    return (
      call.expression.getText(ast) === 'ipcMain.handle' &&
      ts.isStringLiteral(call.arguments[0]) &&
      channels.includes(call.arguments[0].text)
    )
  })

  assert.equal(declarations.length, 2)
  assert.equal(registrations.length, 2)
  const handlers = new Map<string, Handler>()
  const registrationCounts = new Map<string, number>()
  const claims = new BackendDialClaims()
  const windows = new Map<object, any>()
  const defaultWindow = { isDestroyed: () => false, state: { isFullscreen: false, isMaximized: false } }
  const registry: any = { primary: 'remote-primary', connections: [] }
  const descriptor = { url: 'http://127.0.0.1:8080', isFullscreen: false }
  const ensureBackend = vi.fn(async (_profile: unknown, _opts: unknown): Promise<any> => ({ ...descriptor }))

  const ensureRegistryBackend = vi.fn(
    async (_id: unknown, _profile: unknown, _correlation: unknown, _opts: unknown): Promise<any> => ({ ...descriptor })
  )

  const released: string[] = []
  const routes = new Map<number, any>()

  const deps = {
    ipcMain: {
      handle: (channel: string, handler: Handler) => {
        registrationCounts.set(channel, (registrationCounts.get(channel) ?? 0) + 1)
        handlers.set(channel, handler)
      }
    },
    windowConnectionRoutes: routes,
    primaryProfileKey: () => 'default',
    resolveDesktopConnectionRequest,
    backendScopeKey,
    applySpawnPriority: (key: string, _priority: unknown) => () => released.push(key),
    backendDialClaims: claims,
    runForegroundRetryingDialClaim,
    ensureBackend,
    ensureRegistryBackend,
    isBackgroundCapacitySkip,
    isBackgroundSlotRetryDeferred,
    liveWindowState,
    overlayWindowState,
    BrowserWindow: { fromWebContents: (sender: object) => windows.get(sender) ?? null },
    getWindowState: (window: any) => window.state,
    mainWindow: defaultWindow,
    readDesktopConnectionsRegistry: () => registry,
    registryDialConnectionId,
    resolvedConnectionId
  }

  const code = [...declarations, ...registrations].map(node => node.getText(ast)).join('\n')

  const javascript = ts.transpileModule(code, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None }
  }).outputText

  const connect = new Function(...Object.keys(deps), `${javascript}\nreturn connectDesktopProfileRoute;`)(
    ...Object.values(deps)
  ) as Handler

  return {
    connect,
    handlers,
    registrationCounts,
    claims,
    windows,
    defaultWindow,
    ensureBackend,
    ensureRegistryBackend,
    released,
    routes
  }
}

test('main registers each connection IPC exactly once and forwards priority/speculative and live sender state', async () => {
  const f = fixture()
  assert.deepEqual(
    [...f.registrationCounts],
    [
      ['hermes:connection', 1],
      ['hermes:connection:for', 1]
    ]
  )
  const sender = { id: 7 }
  const window = { isDestroyed: () => false, state: { isFullscreen: true, isMaximized: true } }
  f.windows.set(sender, window)

  const result = await f.handlers.get('hermes:connection:for')!(
    { sender },
    {
      connectionId: 'remote-work',
      profile: ' work ',
      priority: 'background',
      speculative: true
    }
  )

  assert.deepEqual(f.ensureRegistryBackend.mock.calls, [
    ['remote-work', 'work', '', { spawnPriority: 'background', speculative: true }]
  ])
  assert.equal(result.connectionId, 'remote-work')
  assert.equal(result.registryScoped, true)
  assert.equal(result.isFullscreen, true)
  assert.equal(result.isMaximized, true)
  assert.equal(f.ensureBackend.mock.calls.length, 0)
  assert.deepEqual(f.released, [backendScopeKey('remote-work', 'work')])
})

test('main refuses an empty registry identity instead of dialing the configured primary', async () => {
  const f = fixture()
  await assert.rejects(
    f.handlers.get('hermes:connection:for')!({ sender: { id: 8 } }, { connectionId: '', profile: 'work' }),
    /No connection with id/
  )
  assert.equal(f.ensureRegistryBackend.mock.calls.length, 0)
  assert.equal(f.ensureBackend.mock.calls.length, 0)
})

test('main connection follows the requesting window route and samples window state at reply time', async () => {
  const f = fixture()
  const sender = { id: 9 }
  const window = { isDestroyed: () => false, state: { isFullscreen: false } }
  f.windows.set(sender, window)
  f.routes.set(sender.id, { connectionId: 'pinned-source', profile: 'research', registryScoped: true })
  f.ensureRegistryBackend.mockImplementationOnce(async () => {
    window.state.isFullscreen = true

    return { url: 'http://127.0.0.1:8080', isFullscreen: false }
  })

  const result = await f.handlers.get('hermes:connection')!({ sender }, undefined, {
    priority: 'foreground',
    speculative: false
  })

  assert.deepEqual(f.ensureRegistryBackend.mock.calls[0], [
    'pinned-source',
    'research',
    '',
    { spawnPriority: 'foreground', speculative: false }
  ])
  assert.equal(result.isFullscreen, true)
})

test('main abort propagates without retrying or retaining the dial/priority ownership', async () => {
  const f = fixture()
  const aborted = new Error('speculative start cancelled')
  aborted.name = 'AbortError'
  f.ensureBackend.mockRejectedValueOnce(aborted)
  await assert.rejects(
    f.connect({ connectionId: null, profile: 'work' }, 'background', true),
    error => error === aborted
  )
  assert.equal(f.ensureBackend.mock.calls.length, 1)
  assert.equal(f.claims.inFlight(backendScopeKey(null, 'work')), false)
  assert.deepEqual(f.released, [backendScopeKey(null, 'work')])
  await f.connect({ connectionId: null, profile: 'work' }, 'foreground', false)
  assert.deepEqual(f.ensureBackend.mock.calls[1], ['work', { spawnPriority: 'foreground', speculative: false }])
})

test('a foreground main dial retries a coalesced saturated speculative claim once', async () => {
  const f = fixture()
  let refuse!: (error: Error) => void
  f.ensureBackend.mockImplementationOnce(
    () =>
      new Promise((_resolve, reject) => {
        refuse = reject
      })
  )
  const route = { connectionId: null, profile: 'work' }
  const speculative = f.connect(route, 'background', true)
  const speculativeResult = assert.rejects(speculative, LocalBackendBackgroundCapacityError)
  const foreground = f.connect(route, 'foreground', false)
  refuse(new LocalBackendBackgroundCapacityError('work'))
  await speculativeResult
  await foreground
  assert.deepEqual(f.ensureBackend.mock.calls, [
    ['work', { spawnPriority: 'background', speculative: true }],
    ['work', { spawnPriority: 'foreground', speculative: false }]
  ])
  assert.equal(f.claims.inFlight(backendScopeKey(null, 'work')), false)
  assert.equal(f.released.length, 2)
})

test('a destroyed sender window never fabricates live chrome state in a main reply', async () => {
  const f = fixture()
  const sender = { id: 10 }
  f.windows.set(sender, { isDestroyed: () => true, state: { isFullscreen: true, isMaximized: true } })
  const result = await f.connect({ connectionId: null, profile: 'work' }, 'foreground', false, sender)
  assert.equal(result.isFullscreen, false)
  assert.equal(result.isMaximized, undefined)
})

test('cancelled pool ownership releases pre-spawn work once and keeps spawned child capacity until exit', () => {
  const signal = new AbortController()
  const release = vi.fn()
  const entry = { process: null, releaseLocalBackendSlot: release }
  signal.abort()
  const pool = new Map([['work', entry]])
  assert.throws(() => assertPoolEntryStillOwned('work', entry, pool, signal.signal), /cancelled during start/)
  assert.equal(release.mock.calls.length, 1)
  assert.throws(() => assertPoolEntryStillOwned('work', entry, pool, signal.signal), /cancelled during start/)
  assert.equal(release.mock.calls.length, 1)
  const childRelease = vi.fn()
  const spawned = { process: {}, releaseLocalBackendSlot: childRelease }
  assert.throws(() => assertPoolEntryStillOwned('work', spawned, new Map(), signal.signal), /cancelled during start/)
  assert.equal(childRelease.mock.calls.length, 0)
})
