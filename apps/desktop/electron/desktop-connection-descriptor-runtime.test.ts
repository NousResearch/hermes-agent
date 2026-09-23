import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopConnectionDescriptorRuntime } from './desktop-connection-descriptor-runtime'

test('switching from a Cloud source does not carry its org and agent identity into a remote source', () => {
  const existing = {
    mode: 'cloud',
    remote: {
      url: 'https://old.example',
      authMode: 'oauth',
      org: 'old-org',
      name: 'Old agent'
    },
    profiles: {}
  }

  const runtime = createDesktopConnectionDescriptorRuntime({
    readDesktopConnectionConfig: () => existing,
    decryptDesktopSecret: () => '',
    decryptRemoteHeaders: () => ({}),
    encryptDesktopSecret: () => ({}),
    probeSecureTokenStorage: () => true,
    hasNativeSession: () => false,
    hasLiveOauthSession: async () => false,
    mintGatewayWsTicket: async () => '',
    rememberRemoteWsHeaders: () => {}
  })

  const next = runtime.coerceDesktopConnectionConfig({
    mode: 'remote',
    remoteAuthMode: 'oauth',
    remoteUrl: 'https://new.example'
  })

  assert.equal(next.mode, 'remote')
  assert.equal(next.remote.url, 'https://new.example')
  assert.equal(next.remote.org, undefined)
  assert.equal(next.remote.name, undefined)
})
