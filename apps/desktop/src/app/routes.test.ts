import { describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { contributedRoutes, NEW_CHAT_ROUTE, primaryRouteSelectedSessionId, ROUTES_AREA, sessionRoute, SETTINGS_ROUTE } from './routes'

const SESS_A = 'sess-a'
const SESS_B = 'sess-b'

describe('primaryRouteSelectedSessionId', () => {
  it('prefers the routed session id over a stale/different store selection (#59305)', () => {
    // The route already committed to B while the store selection hasn't
    // caught up yet (still reads A) — the route wins.
    expect(primaryRouteSelectedSessionId(sessionRoute(SESS_B), SESS_A)).toBe(SESS_B)
  })

  it('returns null on the new-chat route even with a leftover selection from the previous chat', () => {
    expect(primaryRouteSelectedSessionId(NEW_CHAT_ROUTE, SESS_A)).toBeNull()
  })

  it('falls back to the store selection on a non-chat route (settings, overlays)', () => {
    expect(primaryRouteSelectedSessionId(SETTINGS_ROUTE, SESS_A)).toBe(SESS_A)
  })

  it('falls back to the store selection when the route matches the same session', () => {
    expect(primaryRouteSelectedSessionId(sessionRoute(SESS_A), SESS_A)).toBe(SESS_A)
  })

  it('returns null on a non-chat route with no store selection', () => {
    expect(primaryRouteSelectedSessionId(SETTINGS_ROUTE, null)).toBeNull()
  })
})

describe('contributedRoutes', () => {
  const page = (id: string, path: string) => ({
    id,
    area: ROUTES_AREA,
    data: { path },
    render: () => null
  })

  it('reads the registry when called with no arguments', () => {
    const dispose = registry.register(page('helper-contract', '/helper-contract'))

    try {
      expect(contributedRoutes().some(route => route.path === '/helper-contract')).toBe(true)
    } finally {
      dispose()
    }
  })

  it('uses an explicit snapshot instead of the registry (#109063)', () => {
    // React consumers pass their useContributions(ROUTES_AREA) snapshot so the
    // compiler-visible subscription stays a dependency of the derived routes.
    // Pin the contract: the snapshot wins even when the registry disagrees.
    const dispose = registry.register(page('helper-contract', '/helper-contract'))

    try {
      const routes = contributedRoutes([page('explicit', '/explicit')])

      expect(routes.map(route => route.path)).toEqual(['/explicit'])
    } finally {
      dispose()
    }
  })
})
