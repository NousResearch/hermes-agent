import { expect, test, vi } from 'vitest'

import { resolveLegacyApiConnection, resolveProfileBackendRoute } from './connection-config'
import { redialLocalGateway } from './local-gateway'

// B2-15: an unscopable mutating REST call keeps its pooled backend only while the routing decision
// sees the request; a stale-ticket redial must resolve that same route and owner again.
test('the REST redial preserves the failed request route and rejects an owner replacement', async () => {
  const opts = { primaryProfile: 'default', globalRemote: false, profileRemoteOverride: false, isolatedBackend: false }
  const write = { ...opts, requestMethod: 'POST', requestPath: '/api/not-a-real-route/destroy' }
  expect(resolveProfileBackendRoute('work', write).backend).toBe('pool')
  expect(resolveProfileBackendRoute('work', opts).backend).toBe('primary')

  const original = { mode: 'remote', baseUrl: 'https://original.example', token: 'original-token' }
  const replacement = { ...original, token: 'replacement-token' }
  let current = original
  const ensureBackend = vi.fn(async () => current)
  const request = { legacyConnection: original, legacyConnectionProfile: 'work' }
  const ensureOpts = { request: { method: 'POST', path: '/api/not-a-real-route/destroy' } }

  const use = vi.fn(async () => {
    throw new Error('Gateway ticket expired')
  })

  await expect(
    redialLocalGateway({
      ensure: () => resolveLegacyApiConnection(request, 'work', ensureBackend, ensureOpts),
      forget: () => {
        current = replacement
      },
      use
    })
  ).rejects.toThrow('Backend changed')
  expect(use).toHaveBeenCalledOnce()
  expect(ensureBackend).toHaveBeenNthCalledWith(1, 'work', { passive: undefined, ...ensureOpts })
  expect(ensureBackend).toHaveBeenNthCalledWith(2, 'work', { passive: undefined, ...ensureOpts })
})
