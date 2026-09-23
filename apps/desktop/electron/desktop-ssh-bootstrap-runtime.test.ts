import assert from 'node:assert/strict'
import crypto from 'node:crypto'

import { test } from 'vitest'

import { createDesktopSshBootstrapRuntime } from './desktop-ssh-bootstrap-runtime'
import { createBootstrapCoordinator } from './ssh-bootstrap-coordinator'

function fixture(options: { platform?: 'Linux' | 'Windows'; failRollback?: boolean } = {}) {
  const events: string[] = []
  const connections = new Map<string, any>()
  let gateBlocked = false
  let blockAfterConnect = false
  let terminationExpected: any = null
  let sshInstance: FakeSsh | null = null
  let execArgs: string[] | null = null

  class FakeSsh {
    constructor(
      _config: any,
      public options: any
    ) {
      sshInstance = this
    }

    async open() {
      events.push('open')
    }

    async close() {
      events.push('close')
    }

    async cancelForward(localPort: number, remotePort: number) {
      events.push(`cancel:${localPort}:${remotePort}`)
    }

    async forward() {
      return undefined
    }

    async isAlive() {
      return true
    }
  }

  const result = {
    baseUrl: 'http://127.0.0.1:42001',
    token: 'served-token',
    localPort: 42001,
    remotePort: 42002,
    pid: 321,
    ownershipId: 'owner-work',
    spawnNonce: 'nonce-work',
    platform: { os: options.platform || 'Linux' },
    hermesPath: '/opt/hermes',
    hermesHome: '/home/work/.hermes',
    hermesVersion: '1.0',
    reused: false
  }

  const connect = async (input: any) => {
    events.push('connect')
    assert.equal(input.profile, 'work')
    assert.equal(input.ownershipId, 'owner-work')
    assert.equal(input.guestOnboarding, true)
    assert.equal(input.ssh, sshInstance)
    if (blockAfterConnect) {
      gateBlocked = true
    }
    return result
  }

  const runtime = createDesktopSshBootstrapRuntime({
    GUEST_ONBOARDING: true,
    SshConnection: FakeSsh,
    adoptServedDashboardToken: async () => 'served-token',
    buildRemoteConnection: async (baseUrl: string, _authMode: string, _token: string, source: string, host: string) => {
      events.push('build-remote')
      return { baseUrl, source, remoteHost: host }
    },
    connectWindowsRemote: connect,
    detectRemotePlatform: async () => {
      events.push('detect')
      return result.platform
    },
    execText: async (_ssh: string, args: string[], config: any) => {
      execArgs = args
      assert.equal(config.timeout, 10_000)
      return 'hostname resolved.example\nuser work\n'
    },
    managedConnectionUpdateGate: {
      assertCanDial: (id: string) => {
        assert.equal(id, 'connection-work')
        events.push('gate')
        if (gateBlocked) {
          throw new Error('managed update paused this connection')
        }
      }
    },
    persistSshConnectionToken: (_profile: string, _source: string, token: string, id: string) => {
      assert.equal(token, 'served-token')
      assert.equal(id, 'connection-work')
      events.push('persist-token')
    },
    pickLocalPort: async () => 42001,
    remoteLifecycle: {
      connect,
      terminateOwnedDashboardForUpdate: async (_ssh: any, expected: any) => {
        events.push('terminate-posix')
        terminationExpected = expected
        if (options.failRollback) {
          throw new Error('exact termination failed')
        }
      }
    },
    resolveRemoteSshDashboardProfile: (remoteProfile: string, profile: string) => remoteProfile || profile,
    sshBootstrapCoordinator: createBootstrapCoordinator(),
    sshConnections: connections,
    sshIsolatedKeepalives: {
      start: (scope: string, target: any) => {
        assert.equal(scope, 'scope-work')
        assert.equal(target.token, 'served-token')
        events.push('keepalive-start')
      },
      stop: () => events.push('keepalive-stop')
    },
    sshOwnershipKey: (profile: string) => `owner-${profile}`,
    sshProbeReuseProof: async () => 'reusable',
    sshRememberLog: () => events.push('log'),
    sshScopeKey: (profile: string) => `scope-${profile}`,
    teardownSshConnection: async () => events.push('teardown'),
    terminateOwnedWindowsDashboardForUpdate: async (_ssh: any, _install: any, expected: any) => {
      events.push('terminate-windows')
      terminationExpected = expected
      if (options.failRollback) {
        throw new Error('exact termination failed')
      }
    },
    waitForHermes: async () => true
  } as any)

  return {
    runtime,
    events,
    connections,
    result,
    get sshInstance() {
      return sshInstance
    },
    get execArgs() {
      return execArgs
    },
    get terminationExpected() {
      return terminationExpected
    },
    blockAfterConnect: () => {
      blockAfterConnect = true
    }
  }
}

const config = { host: 'remote.example', user: 'work', port: 2222, keyPath: '/keys/work' }
const metadata = { registryConnectionId: 'connection-work', primaryRegistryScope: true }

test('effective SSH fingerprint resolves the live ssh -G configuration', async () => {
  const f = fixture()
  const digest = await f.runtime.effectiveSshConfigFingerprint(config)

  assert.deepEqual(f.execArgs, ['-G', '-p', '2222', '-i', '/keys/work', '--', 'work@remote.example'])
  assert.equal(digest, crypto.createHash('sha256').update('hostname resolved.example\nuser work\n').digest('hex'))
})

test('bootstrap publishes token and tunnel only after the final managed gate assertion', async () => {
  const f = fixture()
  const connection = await f.runtime.bootstrapSshConnection(
    'work',
    config,
    '',
    'registry:connection-work',
    'resolved',
    metadata
  )

  assert.deepEqual(f.events.slice(0, 7), [
    'open',
    'gate',
    'detect',
    'connect',
    'gate',
    'persist-token',
    'keepalive-start'
  ])
  assert.equal(f.execArgs, null)
  assert.equal(f.connections.get('scope-work')?.ssh, f.sshInstance)
  assert.equal(f.connections.get('scope-work')?.registryConnectionId, 'connection-work')
  assert.equal(f.connections.get('scope-work')?.primaryRegistryScope, true)
  assert.equal(connection.remoteHost, 'work@remote.example')
  assert.equal(connection.ssh.effectiveConfigFingerprint, 'resolved')
})

test('a gate claim after connect rolls back the exact POSIX serve before publishing state', async () => {
  const f = fixture()
  f.blockAfterConnect()

  await assert.rejects(
    f.runtime.bootstrapSshConnection('work', config, '', 'registry:connection-work', 'resolved', metadata),
    /managed update paused/
  )

  assert.deepEqual(f.events, [
    'open',
    'gate',
    'detect',
    'connect',
    'gate',
    'terminate-posix',
    'cancel:42001:42002',
    'close'
  ])
  assert.equal(f.terminationExpected?.ownershipId, 'owner-work')
  assert.equal(f.terminationExpected?.pid, 321)
  assert.equal(f.connections.size, 0)
})

test('failed exact Windows rollback marks the bootstrap unsafe while still closing its forward and transport', async () => {
  const f = fixture({ platform: 'Windows', failRollback: true })
  f.blockAfterConnect()

  await assert.rejects(
    f.runtime.bootstrapSshConnection('work', config, '', 'registry:connection-work', 'resolved', metadata),
    (error: any) => error.code === 'managed-update-bootstrap-fence-failed' && error.unsafeManagedBootstrap === true
  )

  assert.deepEqual(f.events, [
    'open',
    'gate',
    'detect',
    'connect',
    'gate',
    'terminate-windows',
    'cancel:42001:42002',
    'close'
  ])
  assert.equal(f.terminationExpected?.spawnNonce, 'nonce-work')
  assert.equal(f.connections.size, 0)
})
