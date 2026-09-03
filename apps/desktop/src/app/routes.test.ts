import { describe, expect, it } from 'vitest'

import {
  appViewForPath,
  NEW_CHAT_ROUTE,
  OVERLAY_ROUTE_PATHS,
  OVERLAY_VIEWS,
  primaryRouteSelectedSessionId,
  routeSessionId,
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

describe('OVERLAY_ROUTE_PATHS', () => {
  // The workspace route table drives its retained-chat routes off this list, so
  // a new overlay that never reaches it silently reintroduces the detachment.
  it('derives a workspace route path for every overlay view', () => {
    expect(OVERLAY_ROUTE_PATHS).toHaveLength(OVERLAY_VIEWS.size)
    expect(new Set(OVERLAY_ROUTE_PATHS.map(path => appViewForPath(`/${path}`)))).toEqual(new Set(OVERLAY_VIEWS))
  })

  it('emits workspace-relative paths (the workspace route table is nested)', () => {
    for (const path of OVERLAY_ROUTE_PATHS) {
      expect(path.startsWith('/')).toBe(false)
    }
  })

  // Mounting the chat at an overlay path must not make ChatView think the URL
  // selects a session: a truthy routeSessionId sets isRoutedSessionView, which
  // drives routeSessionMismatch and blanks the transcript behind a loader.
  it('leaves every overlay path out of the session-id parser', () => {
    for (const path of OVERLAY_ROUTE_PATHS) {
      expect(routeSessionId(`/${path}`)).toBeNull()
    }
  })

  it('keeps a full-page workspace route out of the overlay set', () => {
    expect(OVERLAY_ROUTE_PATHS).not.toContain('skills')
  })
})
