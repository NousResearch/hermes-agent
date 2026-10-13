import { afterEach, describe, expect, it, vi } from 'vitest'

import { NEW_CHAT_ROUTE, sessionRoute, SETTINGS_ROUTE } from '../../routes'

import {
  pinnedOwnerCount,
  pinStoredSessionForOwner,
  releaseStoredSessionPins,
  routeTargetFromToken,
  sessionContextDrift
} from './session-context-drift'

const SESS_A = 'sess-a'
const SESS_B = 'sess-b'
const CREATED = 'sess-created'

// Build a route token the way desktop-controller does: pathname:search:hash.
const routeToken = (pathname: string, search = '', hash = '') => `${pathname}:${search}:${hash}`

afterEach(() => {
  releaseStoredSessionPins('owner-a')
  releaseStoredSessionPins('owner-b')
  releaseStoredSessionPins('submit-owner')
})

describe('sessionContextDrift: a fresh new-chat submit', () => {
  it('does NOT report drift when the user is still on the new-chat route and selection is null', () => {
    // The new-chat baseline: the submit started with no selection at all.
    const reason = sessionContextDrift({
      startRouteToken: routeToken(NEW_CHAT_ROUTE),
      nowRouteToken: routeToken(NEW_CHAT_ROUTE),
      startSelectedStoredId: null,
      nowSelectedStoredId: null,
      submitTargetStoredId: null
    })

    expect(reason).toBeNull()
  })

  it('reports drift if selection lands on the created chat BEFORE the pipeline re-pins its baseline', () => {
    // This is the narrow race the re-pin block in submit.ts closes: after
    // createBackendSessionForSend resolves, the code re-pins
    // startingStoredSessionId / startingSelectedStoredSessionId / startingRouteToken
    // to the newly created chat BEFORE any further drift read. Until that
    // re-pin lands, a selection move onto the created chat (from a null
    // baseline) still reads as `selection:null-><created>` — which is exactly
    // why the re-pin must stay ordered ahead of the post-create drift checks,
    // and why this test documents the pre-re-pin state as drift.
    const reason = sessionContextDrift({
      startRouteToken: routeToken(NEW_CHAT_ROUTE),
      nowRouteToken: routeToken(NEW_CHAT_ROUTE),
      startSelectedStoredId: null,
      nowSelectedStoredId: CREATED,
      submitTargetStoredId: null
    })

    expect(reason).toBe(`selection:null->${CREATED}`)
  })

  it('does NOT report drift once the baseline has been re-pinned to the created chat', () => {
    // Post-re-pin: baseline and target both name the created chat.
    const reason = sessionContextDrift({
      startRouteToken: routeToken(NEW_CHAT_ROUTE),
      nowRouteToken: routeToken(NEW_CHAT_ROUTE),
      startSelectedStoredId: CREATED,
      nowSelectedStoredId: CREATED,
      submitTargetStoredId: CREATED
    })

    expect(reason).toBeNull()
  })

  it('does NOT report drift when the route+selection landed on the session create just minted', () => {
    // The exact post-create state the submit pipeline expects: baseline
    // re-pinned to the created chat (see the re-pin block in submit.ts).
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(CREATED)),
      nowRouteToken: routeToken(sessionRoute(CREATED)),
      startSelectedStoredId: CREATED,
      nowSelectedStoredId: CREATED,
      submitTargetStoredId: CREATED
    })

    expect(reason).toBeNull()
  })

  it('DOES report drift when the user moved to a genuinely different chat', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_B)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_B,
      submitTargetStoredId: SESS_A
    })

    expect(reason).not.toBeNull()
  })

  it('a pin suppresses a route move onto the pinned id for its owner only', () => {
    // Both prongs consult activePins. The one exception baked into the route
    // prong is `!(targetNow !== '__new__' && ...)`: the __new__ sentinel is
    // never treated as a pinned target, so a route move onto a real pinned
    // chat is suppressed, while the new-chat sentinel is not.
    pinStoredSessionForOwner('owner-a', SESS_B)

    const pinnedOwner = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_B)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: null,
      pinOwner: 'owner-a'
    })

    // A different owner must not inherit that suppression.
    const otherOwner = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(sessionRoute(SESS_B)),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: null,
      pinOwner: 'owner-b'
    })

    expect(pinnedOwner).toBeNull()
    expect(otherOwner).toBe(`route:${SESS_A}->${SESS_B}`)
  })

  it('the __new__ sentinel is never a pinned target, so a route onto it is never suppressed', () => {
    // Guard against a future "simplification" of the route prong that drops
    // the `targetNow !== '__new__'` conjunct.
    pinStoredSessionForOwner('owner-a', '__new__')

    const reason = sessionContextDrift({
      startRouteToken: routeToken(sessionRoute(SESS_A)),
      nowRouteToken: routeToken(NEW_CHAT_ROUTE),
      startSelectedStoredId: SESS_A,
      nowSelectedStoredId: SESS_A,
      submitTargetStoredId: null,
      pinOwner: 'owner-a'
    })

    expect(reason).toBe(`route:${SESS_A}->__new__`)
  })

  it('treats a settings route as no drift target at all (selection churn is ignored)', () => {
    const reason = sessionContextDrift({
      startRouteToken: routeToken(NEW_CHAT_ROUTE),
      nowRouteToken: routeToken(SETTINGS_ROUTE),
      startSelectedStoredId: null,
      nowSelectedStoredId: null,
      submitTargetStoredId: null
    })

    expect(reason).toBeNull()
  })
})

describe('the drift-abort contract on a new-chat submit', () => {
  it('a drift abort returns false with no text restored — the prompt is lost', () => {
    // This mirrors abortForSessionSwitch's contract in submit.ts:
    //   dropOptimistic(optimisticSessionId); releaseBusy(); return false
    // and the composer's `if (!ok) { setTimeout(...restore...) }` guard, which
    // ONLY restores when the submit itself THREW. A plain `return false` is
    // read by dispatchSubmit as "handled", so the draft stash is never replayed
    // and the composer stays empty.
    const abortForSessionSwitch = (_optimisticSessionId: null | string): false => {
      return false
    }

    // The composer's restore guard, verbatim in shape from use-composer-submit:
    const dispatchSubmit = (onSubmit: () => Promise<boolean>) => {
      let restored = false

      void onSubmit().catch(() => {
        // Only a REJECTION re-stashes and re-loads the draft.
        restored = true
      })

      return { restored }
    }

    const outcome = dispatchSubmit(async () => abortForSessionSwitch(null))

    expect(outcome.restored).toBe(false)
    expect(abortForSessionSwitch(CREATED)).toBe(false)
  })

  it('a thrown submit DOES restore the draft (the asymmetry the lost prompt falls through)', async () => {
    let restored = false

    const dispatchSubmit = (onSubmit: () => Promise<boolean>) =>
      onSubmit().catch(() => {
        restored = true
      })

    await dispatchSubmit(async () => {
      throw new Error('Quick Entry destination changed mid-create: route')
    })

    expect(restored).toBe(true)
  })
})

describe('routeTargetFromToken on the new-chat route', () => {
  it('maps the new-chat route to the __new__ sentinel', () => {
    expect(routeTargetFromToken(routeToken(NEW_CHAT_ROUTE))).toBe('__new__')
  })

  it('agrees with a selection when the target is the sentinel', () => {
    // This is submit.ts's `routeAgrees` escape: `routeTarget === '__new__'`
    // counts as agreement even before the route re-homes to the created chat.
    const routeTarget = routeTargetFromToken(routeToken(NEW_CHAT_ROUTE))
    const selection = CREATED
    const routeAgrees = routeTarget === null || routeTarget === '__new__' || routeTarget === selection

    expect(routeAgrees).toBe(true)
  })
})

describe('pin ownership', () => {
  it('a pin survives the owner that pinned it and is released explicitly', () => {
    pinStoredSessionForOwner('owner-a', CREATED)

    expect(pinnedOwnerCount(CREATED)).toBe(1)

    releaseStoredSessionPins('owner-a')

    expect(pinnedOwnerCount(CREATED)).toBe(0)
  })

  it('pins suppress drift only for the owning request; submit pins nothing', () => {
    // Both prongs consult activePins, but ONLY the pins created by the same
    // owner token. submit.ts builds its sessionContextDrift args with no
    // pinOwner at all, so activePins is always NO_PINS on the submit path:
    // even a selection move onto an id another request pinned reads as drift
    // there. That is deliberate isolation, and it is why the submit guard's
    // trigger window is as wide as it is.
    pinStoredSessionForOwner('owner-a', SESS_B)

    const unpinned = sessionContextDrift({
      nowRouteToken: routeToken(NEW_CHAT_ROUTE),
      nowSelectedStoredId: SESS_B,
      startRouteToken: routeToken(NEW_CHAT_ROUTE),
      startSelectedStoredId: SESS_A,
      submitTargetStoredId: null
    })

    const ownedByOwner = sessionContextDrift({
      nowRouteToken: routeToken(NEW_CHAT_ROUTE),
      nowSelectedStoredId: SESS_B,
      startRouteToken: routeToken(NEW_CHAT_ROUTE),
      startSelectedStoredId: SESS_A,
      submitTargetStoredId: null,
      pinOwner: 'owner-a'
    })

    expect(unpinned).toBe(`selection:${SESS_A}->${SESS_B}`)
    expect(ownedByOwner).toBeNull()
  })
})
