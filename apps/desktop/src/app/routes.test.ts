import { describe, expect, it } from 'vitest'

import {
  NEW_CHAT_ROUTE,
  primaryRouteSelectedSessionId,
  routeSessionId,
  routeSessionProfile,
  sessionRoute,
  SETTINGS_ROUTE
} from './routes'

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

describe('profile-qualified session routes', () => {
  it('round-trips the profile owner independently from the stored id', () => {
    const route = sessionRoute('same/id?', 'research ops')

    expect(route).toBe('/same%2Fid%3F?profile=research%20ops')
    expect(routeSessionId(route)).toBe('same/id?')
    expect(routeSessionProfile(route)).toBe('research ops')
  })

  it('keeps legacy profile-less routes profile-less', () => {
    expect(sessionRoute(SESS_A)).toBe('/sess-a')
    expect(routeSessionProfile(sessionRoute(SESS_A))).toBeNull()
  })
})
