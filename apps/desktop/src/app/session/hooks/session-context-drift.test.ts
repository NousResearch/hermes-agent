import { describe, expect, it } from 'vitest'

import { NEW_CHAT_ROUTE, sessionRoute, SETTINGS_ROUTE } from '../../routes'

import { routeTargetFromToken, sessionContextDrift } from './session-context-drift'

const SESS_A = 'sess-a'
const SESS_B = 'sess-b'

// Build a route token the way desktop-controller does: pathname:search:hash.
const routeToken = (pathname: string, search = '', hash = '') => `${pathname}:${search}:${hash}`

describe('routeTargetFromToken', () => {
  it('maps a session route to its session id, a non-chat route to null, and the new-chat route to __new__', () => {
    expect(routeTargetFromToken(routeToken(sessionRoute(SESS_A)))).toBe(SESS_A)
    expect(routeTargetFromToken(routeToken(SETTINGS_ROUTE))).toBeNull()
    expect(routeTargetFromToken(routeToken(NEW_CHAT_ROUTE))).toBe('__new__')
  })

  it('ignores search and hash — only the pathname selects the chat', () => {
    expect(routeTargetFromToken(routeToken(sessionRoute(SESS_A), '?panel=preview', '#reply'))).toBe(SESS_A)
  })

  it('treats a colon-free token as a bare pathname', () => {
    expect(routeTargetFromToken(sessionRoute(SESS_A))).toBe(SESS_A)
  })
})

describe('sessionContextDrift', () => {
  it('does not drift on search/hash-only route churn', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_A), '?panel=preview', '#reply'),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A
    })

    expect(reason).toBeNull()
  })

  it('does not drift on a selection null-reset (gateway/profile switch, reconnect)', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_A)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: null,
      submitTargetStoredId: SESS_A
    })

    expect(reason).toBeNull()
  })

  it('drifts when selection moves to a different non-null stored session', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_A)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_B,
      submitTargetStoredId: SESS_A
    })

    expect(reason).toBe('selection:sess-a->sess-b')
  })

  it('drifts when the routed session id changes to another session', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_B)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A
    })

    expect(reason).toBe('route:sess-a->sess-b')
  })

  it('drifts when the route moves to the new-chat route mid-submit', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(NEW_CHAT_ROUTE),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A
    })

    expect(reason).toBe('route:sess-a->__new__')
  })

  it('does not drift when the route moves to a non-chat route (null target)', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(SETTINGS_ROUTE),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A
    })

    expect(reason).toBeNull()
  })

  it('does not drift when route and selection re-home onto the submit target (the create pipeline re-home)', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(NEW_CHAT_ROUTE),
      nowRouteToken: routeToken(sessionRoute(SESS_A)),
      startSelectedStoredId: null,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A
    })

    expect(reason).toBeNull()
  })

  it('drifts when a new-chat draft with no target yet is switched to an existing chat', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(NEW_CHAT_ROUTE),
      nowRouteToken: routeToken(sessionRoute(SESS_B)),
      startSelectedStoredId: null,
      nowSelectedStoredId: SESS_B,
      submitTargetStoredId: null
    })

    expect(reason).toBe('route:__new__->sess-b')
  })
})
