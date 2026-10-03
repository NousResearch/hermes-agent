import { describe, expect, it } from 'vitest'

import {
  appViewForPath,
  isWorkspacePageRoute,
  NEW_CHAT_ROUTE,
  primaryRouteSelectedSessionId,
  PROJECTS_ROUTE,
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
})

describe('projects route', () => {
  it('is a workspace page, never mistaken for a session id, with or without a selected project', () => {
    for (const to of [PROJECTS_ROUTE, `${PROJECTS_ROUTE}?project=p_1`]) {
      expect(appViewForPath(to)).toBe('projects')
      expect(isWorkspacePageRoute(to)).toBe(true)
      expect(routeSessionId(to)).toBeNull()
    }
  })
})
