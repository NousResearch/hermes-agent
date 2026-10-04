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

  it('does not drift when composerScope matches the resolved (lineage) submit target', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_A)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A,
      composerScope: SESS_A,
      submitTargetComposerScope: SESS_A
    })

    expect(reason).toBeNull()
  })

  it('drifts (composer prong) when the loaded composer scope disagrees with the resolved submit target (#59305)', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_A)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A,
      composerScope: SESS_B,
      submitTargetComposerScope: SESS_A
    })

    expect(reason).toBe('composer:sess-b->sess-a')
  })

  // A sessionless new chat loads NO chat into the composer: ChatBar scopes its
  // draft to the per-lifecycle fresh key (`__new__:<uuid>`), which is the
  // successor of the plain `null` the prong has always ignored. Treating it as
  // a chat name made every first send on a new chat abort right after
  // session.create — the session was minted, the route re-homed onto it, and
  // prompt.submit never ran, so no row was ever persisted and the window sat
  // on an id the REST transcript answers with 404 "Session not found".
  it.each([['__new__'], ['__new__:0f0b1b4a-1f3e-4b21-9f31-2c6b0d7a9e55']])(
    'does not drift when the composer still carries the fresh-draft scope %s (pre-create)',
    freshScope => {
      const reason = sessionContextDrift({
        startRouteToken: routeToken(NEW_CHAT_ROUTE),
        nowRouteToken: routeToken(NEW_CHAT_ROUTE),
        startSelectedStoredId: null,
        nowSelectedStoredId: null,
        submitTargetStoredId: null,
        composerScope: freshScope,
        // resolveComposerSessionKey(null, sessions) — no stored target yet.
        submitTargetComposerScope: null
      })

      expect(reason).toBeNull()
    }
  )

  it('does not drift when a fresh-draft composer scope meets the chat the submit just created', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_A)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A,
      composerScope: '__new__:0f0b1b4a-1f3e-4b21-9f31-2c6b0d7a9e55',
      submitTargetComposerScope: SESS_A
    })

    expect(reason).toBeNull()
  })

  it('still drifts (composer prong) for a real chat whose name merely starts like the fresh bucket', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_A)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: SESS_A,
      composerScope: SESS_B,
      submitTargetComposerScope: SESS_A
    })

    expect(reason).toBe('composer:sess-b->sess-a')
  })

  it('does not drift when the session has rotated via compression (composerScope is the lineage root, submitTargetStoredId is the live tip)', () => {
    const ROOT_ID = 'stored-root'
    const TIP_ID = 'stored-tip-after-compression'

    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(TIP_ID)),
      nowRouteToken: routeToken(sessionRoute(TIP_ID)),
      startSelectedStoredId: TIP_ID,
      nowSelectedStoredId: TIP_ID,
      submitTargetStoredId: TIP_ID,
      composerScope: ROOT_ID,
      // What submit.ts actually passes: resolveComposerSessionKey(TIP_ID, sessions),
      // which resolves to the lineage root for a session that has compressed.
      submitTargetComposerScope: ROOT_ID
    })

    expect(reason).toBeNull()
  })
})
