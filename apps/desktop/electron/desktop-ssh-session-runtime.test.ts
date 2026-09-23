import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopSshSessionRuntime } from './desktop-ssh-session-runtime'

test('a non-SSH profile route does not inherit a cached global SSH terminal', () => {
  const sshConnections = new Map([['', { ssh: { id: 'old-global-tunnel' } }]])

  const runtime = createDesktopSshSessionRuntime({
    createDesktopSshBootstrapRuntime: () => ({
      effectiveSshConfigFingerprint: async () => '',
      bootstrapSshConnection: async () => undefined
    }),
    primaryProfileKey: () => 'default',
    readDesktopConnectionConfig: () => ({}),
    readDesktopConnectionsRegistry: () => ({ connections: [] }),
    resolveDesktopRemoteRoute: () => ({ kind: 'remote' }),
    sshConnections,
    windowConnectionRoutes: new Map([[42, { profile: 'mara' }]])
  })

  assert.equal(runtime.activeSshTerminalTarget(42), null)
  assert.equal(sshConnections.size, 1)
})
