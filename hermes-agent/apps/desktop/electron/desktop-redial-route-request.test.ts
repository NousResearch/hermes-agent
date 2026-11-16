import { expect, test } from 'vitest'

import { resolveProfileBackendRoute } from './connection-config'

// B2-15: an unscopable mutating REST call keeps its pooled backend only while the routing decision
// sees the request; the stale-ticket redial (ensure + forget) must resolve that same route, or the
// retry lands on the shared primary home (wrong-profile write) and forget clears the primary.
test('the REST redial re-ensures and forgets the route the failed request used', async () => {
  const opts = { primaryProfile: 'default', globalRemote: false, profileRemoteOverride: false, isolatedBackend: false }
  const write = { ...opts, requestMethod: 'POST', requestPath: '/api/not-a-real-route/destroy' }
  expect(resolveProfileBackendRoute('work', write).backend).toBe('pool')
  expect(resolveProfileBackendRoute('work', opts).backend).toBe('primary')

  // main.ts is not importable under vitest; pin the one REST redial seam's wiring instead.
  const fs = await import('node:fs/promises')
  const source = await fs.readFile(new URL('./main.ts', import.meta.url), 'utf8')
  const first = source.indexOf('connection = await ensureBackend(routeProfile, ensureOpts)')
  const seam = source.slice(first, source.indexOf('use: send', first))
  expect(first).toBeGreaterThan(0)
  expect(seam).toContain('ensure: () => ensureBackend(routeProfile, ensureOpts)')
  expect(seam).toContain('forget: () => forgetLocalGatewayDescriptor(routeProfile, ensureOpts.request)')
  expect(source).toContain('resolveProfileBackendRoute(key, profileRouteOptions(key, request))')
})
