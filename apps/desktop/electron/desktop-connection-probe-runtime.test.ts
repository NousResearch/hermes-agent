import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopConnectionProbeRuntime } from './desktop-connection-probe-runtime'

test('public gateway status selects OAuth while malformed provider rows are excluded', async () => {
  const requested: string[] = []

  const runtime = createDesktopConnectionProbeRuntime({
    fetchPublicJson: async (url: string) => {
      requested.push(url)

      return url.endsWith('/api/status')
        ? { auth_required: true, version: 'test-version' }
        : { providers: [{ name: 'nous', display_name: 'Nous Research' }, null, { display_name: 'missing-name' }] }
    }
  } as any)

  const result = await runtime.probeRemoteAuthMode('https://gateway.example/')

  assert.equal(result.reachable, true)
  assert.equal(result.authMode, 'oauth')
  assert.deepEqual(result.providers, [{ name: 'nous', displayName: 'Nous Research', supportsPassword: false }])
  assert.deepEqual(requested, ['https://gateway.example/api/status', 'https://gateway.example/api/auth/providers'])
})
