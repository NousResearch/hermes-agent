import assert from 'node:assert/strict'

import { test } from 'vitest'

import { backendScopeKey, backendScopePrefix } from './connection-registry'
import { createDesktopConnectionAdmissionRuntime } from './desktop-connection-admission-runtime'

function fixture() {
  const events: string[] = []
  const pool = new Map<string, any>()
  const registry: any = { primary: 'local', connections: [{ id: 'local', kind: 'local', label: 'This device' }] }
  const spawned: Array<{ key: string; options: any }> = []
  const bootstraps: any[][] = []
  let globalRemote = false
  let isolatedBackend = false
  let waitedBaseUrl = ''
  let probeFailures = 0

  const runtime = createDesktopConnectionAdmissionRuntime({
    backendPool: pool,
    bootstrapSshConnection: async (...args: any[]) => {
      bootstraps.push(args)
      return { baseUrl: 'http://127.0.0.1:49001', mode: 'remote', token: 'served-token' }
    },
    buildRemoteConnection: async (url: string, authMode: string, token: string, source: string) => ({
      baseUrl: url,
      mode: 'remote',
      authMode,
      token,
      source
    }),
    decryptDesktopSecret: (token: string) => `plain-${token}`,
    effectiveSshConfigFingerprint: async () => 'effective-fingerprint',
    evictLruPoolBackends: async () => events.push('evict'),
    fetchJsonForBackend: async () => {
      if (probeFailures-- > 0) throw new Error('connect ECONNREFUSED')
      return { ok: true }
    },
    getWindowState: () => ({ windowReady: true }),
    globalRemoteActive: () => globalRemote,
    hermesLog: ['line one'],
    localBackendLifecycle: { assertCanStart: () => events.push('lifecycle-assert') },
    logPoolSpawnFailure: () => events.push('spawn-failed'),
    managedConnectionUpdateGate: { assertCanDial: () => events.push('managed-gate') },
    poolMaxBackends: () => 4,
    poolRetirer: { assertCanOpen: () => events.push('retirer-assert') },
    poolStopper: { inFlight: () => null },
    primaryProfileKey: () => 'default',
    profileDeletionGate: { assertCanStart: () => events.push('deletion-assert') },
    profileHasRemoteOverride: () => false,
    profileRouteOptions: () => ({
      backendProfile: '',
      globalRemote,
      isolatedBackend,
      ownEntry: false,
      primaryProfile: 'default',
      primaryRemoteActive: false,
      profileRemoteOverride: false
    }),
    promotePoolEntry: () => events.push('promote'),
    readDesktopConnectionsRegistry: () => registry,
    registryDispatchRevalidation: { run: async (_promise: Promise<any>, run: () => Promise<any>) => run() },
    rememberLog: () => events.push('log'),
    setWslBridgeProfileState: (profile: string, active: boolean) => events.push(`wsl:${profile}:${active}`),
    spawnPoolBackend: async (key: string, _entry: any, options: any) => {
      spawned.push({ key, options })
      return { baseUrl: 'http://127.0.0.1:49002', mode: 'local', profile: key }
    },
    spawnPriorityFrom: (value: string) => value || 'foreground',
    sshBootstrapCoordinator: { cancelAndWait: async () => events.push('cancel-bootstrap') },
    startHermes: async () => {
      events.push('start-primary')
      return { baseUrl: 'http://127.0.0.1:49000', mode: 'local' }
    },
    startPoolIdleReaper: () => events.push('idle-reaper'),
    stopPoolBackend: async (key: string) => {
      pool.delete(key)
      events.push('stop-pool')
    },
    teardownFailedLocalBackend: async () => events.push('teardown-failed'),
    teardownSshConnection: async () => events.push('teardown-ssh'),
    waitForHermes: async (baseUrl: string) => {
      waitedBaseUrl = baseUrl
      events.push('wait-remote')
    }
  } as any)

  return {
    runtime,
    events,
    pool,
    registry,
    spawned,
    bootstraps,
    get waitedBaseUrl() {
      return waitedBaseUrl
    },
    setGlobalRemote: (value: boolean) => {
      globalRemote = value
    },
    setIsolatedBackend: (value: boolean) => {
      isolatedBackend = value
    },
    failNextProbes: (count: number) => {
      probeFailures = count
    }
  }
}

test('primary admission preserves the selected profile and asserts lifecycle before starting', async () => {
  const f = fixture()
  const connection = await f.runtime.ensureBackend('default')

  assert.deepEqual(f.events, [
    'lifecycle-assert',
    'retirer-assert',
    'deletion-assert',
    'start-primary',
    'wsl:default:true'
  ])
  assert.equal(connection.profile, 'default')
  assert.equal(connection.sharedPrimary, undefined)
  assert.equal(f.pool.size, 0)
})

test('registry local source uses a separate forced-local pool when v1 routes remotely', async () => {
  const f = fixture()
  f.setGlobalRemote(true)

  const connection = await f.runtime.ensureRegistryBackend('local', 'work')
  const key = `${backendScopePrefix('local')}work`

  assert.equal(f.spawned.length, 1)
  assert.deepEqual(f.spawned[0], { key: 'work', options: { forceLocal: true, poolKey: key } })
  assert.equal(f.pool.get(key)?.connectionPromise instanceof Promise, true)
  assert.equal(connection.mode, 'local')
  assert.ok(f.events.includes('idle-reaper'))
})

test('passive registry admission refuses a cold forced-local spawn', async () => {
  const f = fixture()
  f.setGlobalRemote(true)

  await assert.rejects(f.runtime.ensureRegistryBackend('local', 'work', '', { passive: true }), /Passive read/)

  assert.equal(f.spawned.length, 0)
  assert.equal(f.pool.size, 0)
})

test('unknown registry identity is rejected before any backend opens', async () => {
  const f = fixture()

  await assert.rejects(f.runtime.ensureRegistryBackend('missing', 'work'), /No connection with id "missing"/)

  assert.equal(f.spawned.length, 0)
  assert.equal(f.pool.size, 0)
  assert.deepEqual(f.events, [])
})

test('registry remote dial waits for the gateway and preserves connection and profile identity', async () => {
  const f = fixture()
  f.registry.connections.push({
    id: 'remote-one',
    kind: 'remote',
    label: 'Remote one',
    url: 'https://remote.example',
    authMode: 'token',
    token: 'ciphertext'
  })

  const connection = await f.runtime.ensureRegistryBackend('remote-one', 'work')
  const key = backendScopeKey('remote-one', 'work')

  assert.equal(f.waitedBaseUrl, 'https://remote.example')
  assert.equal(connection.connectionId, 'remote-one')
  assert.equal(connection.profile, 'work')
  assert.equal(connection.sharedRemote, true)
  assert.equal(connection.token, 'plain-ciphertext')
  assert.equal(connection.windowReady, true)
  assert.equal(f.pool.get(key)?.remoteBaseUrl, 'https://remote.example')
  assert.ok(f.events.includes('wsl:work:false'))
})

test('dispatch retires a dead cached remote before reconnecting the same scope', async () => {
  const f = fixture()
  f.registry.connections.push({
    id: 'remote-one',
    kind: 'remote',
    label: 'Remote one',
    url: 'https://remote.example',
    authMode: 'token',
    token: 'ciphertext'
  })
  const key = backendScopeKey('remote-one', 'work')
  const stale = { baseUrl: 'https://old.example', mode: 'remote', token: 'old-token' }
  const stalePromise = Promise.resolve(stale)
  f.pool.set(key, { connectionPromise: stalePromise, lastActiveAt: 0 })
  f.failNextProbes(1)

  const connection = await f.runtime.ensureRegistryBackend('remote-one', 'work')

  assert.equal(connection.baseUrl, 'https://remote.example')
  assert.equal(f.pool.get(key)?.connectionPromise === stalePromise, false)
  assert.equal(f.events.filter(event => event === 'stop-pool').length, 1)
  assert.equal(f.waitedBaseUrl, 'https://remote.example')
})

test('registry SSH dial carries effective fingerprint and managed scope through bootstrap', async () => {
  const f = fixture()
  f.registry.connections.push({
    id: 'ssh-one',
    kind: 'ssh',
    label: 'SSH one',
    host: 'remote.example',
    user: 'agent',
    token: 'ciphertext',
    remoteProfile: 'remote-work'
  })

  const connection = await f.runtime.ensureRegistryBackend('ssh-one', 'work', 'correlation-one')
  const key = backendScopeKey('ssh-one', 'work')
  const [scope, sshConfig, token, source, fingerprint, metadata] = f.bootstraps[0]

  assert.equal(f.events[0], 'managed-gate')
  assert.equal(scope, key)
  assert.equal(sshConfig.remoteProfile, 'remote-work')
  assert.equal(token, 'plain-ciphertext')
  assert.equal(source, 'registry:ssh-one')
  assert.equal(fingerprint, 'effective-fingerprint')
  assert.equal(metadata.managedUpdateCorrelation, 'correlation-one')
  assert.equal(metadata.registryConnectionId, 'ssh-one')
  assert.equal(connection.remoteProfile, 'remote-work')
  assert.equal(connection.connectionId, 'ssh-one')
  assert.equal(f.pool.get(key)?.remoteBaseUrl, 'http://127.0.0.1:49001')
  assert.ok(f.events.includes('wsl:work:false'))
})
