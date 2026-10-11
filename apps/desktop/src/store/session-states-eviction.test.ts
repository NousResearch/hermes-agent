import { beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $providerWaitSessions, setSessionProviderWait } from '@/store/provider-wait'
import { $activeSessionId, $selectedStoredSessionId, $sessions, $unreadFinishedSessionIds } from '@/store/session'
import {
  $sessionStates,
  $sessionTiles,
  clearAllSessionStates,
  closeSessionTile,
  publishSessionState,
  retainAfterReclaim
} from '@/store/session-states'

/**
 * The closed-tile leak: gateway events keep publishing for sessions whose
 * surface is gone, and every parked transcript taxes every later publish (map
 * spread + the status projections run per entry per message delta). A settled
 * state nothing references must release its transcript; lightweight status
 * stays so sidebar projections remain available.
 */

const state = (storedId: string, patch: Partial<ReturnType<typeof createClientSessionState>> = {}) => ({
  ...createClientSessionState(storedId),
  messages: [{ id: `${storedId}-m`, role: 'assistant' as const, parts: [{ type: 'text' as const, text: 'hi' }] }],
  ...patch
})

beforeEach(() => {
  $sessionStates.set({})
  $sessionTiles.set([])
  $sessions.set([])
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $unreadFinishedSessionIds.set([])
  $providerWaitSessions.set({})
})

/** The reclaim handler's retention, driven at the store seam where the
 *  bookkeeping it must eventually release is observable. */
function retainAndRebind(reclaimedRuntimeId: string, reboundRuntimeId: string) {
  $activeSessionId.set(reclaimedRuntimeId)
  publishSessionState(reclaimedRuntimeId, state('stored-1', { busy: true }))
  retainAfterReclaim(reclaimedRuntimeId)

  // The durable resume rebinds the pane onto a FRESH runtime id (the backend
  // re-mints on resume), so the atom leaves the reclaimed one.
  $activeSessionId.set(reboundRuntimeId)
}

describe('publish-time eviction', () => {
  it('releases an unreferenced settled transcript while keeping status and its unread dot', () => {
    publishSessionState('rt-1', state('stored-1', { busy: true }))
    expect($sessionStates.get()['rt-1']).toBeDefined()

    publishSessionState('rt-1', state('stored-1', { busy: false }))

    expect($sessionStates.get()['rt-1']?.messages).toEqual([])
    expect($sessionStates.get()['rt-1']).toMatchObject({ storedSessionId: 'stored-1', busy: false })
    // The settle transition still fired: the sidebar's unread marker landed.
    expect($unreadFinishedSessionIds.get()).toContain('stored-1')
  })

  it('keeps a busy session with no surface — its background turn feeds the sidebar dot', () => {
    publishSessionState('rt-1', state('stored-1', { busy: true }))
    publishSessionState('rt-1', state('stored-1', { busy: true, awaitingResponse: true }))

    expect($sessionStates.get()['rt-1']).toBeDefined()
  })

  it('keeps a needsInput session with no surface — the attention dot reads it', () => {
    publishSessionState('rt-1', state('stored-1', { busy: true }))
    publishSessionState('rt-1', state('stored-1', { busy: false, needsInput: true }))

    expect($sessionStates.get()['rt-1']).toBeDefined()
  })

  it('keeps a settled session an open tile references, by runtime or stored id', () => {
    $sessionTiles.set([{ runtimeId: 'rt-1', storedSessionId: 'stored-1' }])
    publishSessionState('rt-1', state('stored-1', { busy: true }))
    publishSessionState('rt-1', state('stored-1', { busy: false }))
    expect($sessionStates.get()['rt-1']).toBeDefined()

    // Mid-resume a tile holds only the stored id (runtime binding not patched
    // in yet) — that reference must count too.
    $sessionTiles.set([{ storedSessionId: 'stored-2' }])
    publishSessionState('rt-2', state('stored-2', { busy: true }))
    publishSessionState('rt-2', state('stored-2', { busy: false }))
    expect($sessionStates.get()['rt-2']).toBeDefined()
  })

  it("keeps the primary view's settled session", () => {
    $activeSessionId.set('rt-1')
    publishSessionState('rt-1', state('stored-1', { busy: true }))
    publishSessionState('rt-1', state('stored-1', { busy: false }))

    expect($sessionStates.get()['rt-1']).toBeDefined()
  })

  it('always lands a FIRST publish — resume can publish before the surface points at the runtime', () => {
    publishSessionState('rt-1', state('stored-1', { busy: false }))

    expect($sessionStates.get()['rt-1']).toBeDefined()
  })
})

describe('closeSessionTile eviction', () => {
  it("drops a settled session's state on close — no later publish may come", () => {
    $sessionTiles.set([{ runtimeId: 'rt-1', storedSessionId: 'stored-1' }])
    publishSessionState('rt-1', state('stored-1', { busy: false }))

    closeSessionTile('stored-1')

    expect($sessionStates.get()['rt-1']).toBeUndefined()
  })

  it("keeps a busy session's state on close — the background turn is still running", () => {
    $sessionTiles.set([{ runtimeId: 'rt-1', storedSessionId: 'stored-1' }])
    publishSessionState('rt-1', state('stored-1', { busy: true }))

    closeSessionTile('stored-1')

    expect($sessionStates.get()['rt-1']).toBeDefined()

    // ... and its settle publish releases only the heavy transcript.
    publishSessionState('rt-1', state('stored-1', { busy: false }))
    expect($sessionStates.get()['rt-1']?.messages).toEqual([])
    expect($sessionStates.get()['rt-1']).toMatchObject({ storedSessionId: 'stored-1', busy: false })
  })
})

/**
 * #122507 follow-up: `session.reclaimed` keeps the visible primary transcript
 * while its runtime is durably resumed. The backend re-mints a fresh runtime id
 * on resume, so nothing publishes against the dead one again — the leak
 * `session-states.ts` names for closed tiles would otherwise park one full
 * transcript per idle-timeout of the visible chat, forever.
 */
describe('post-reclaim retention', () => {
  it('drops the retained transcript and its bookkeeping once the resume rebinds the atom', () => {
    setSessionProviderWait('rt-1', 'loading model into memory')
    retainAndRebind('rt-1', 'rt-2')

    // The dead runtime is neither atom-visible nor tile-bound any more, so its
    // state AND the per-runtime ledgers dropSessionState clears are gone.
    expect($sessionStates.get()['rt-1']).toBeUndefined()
    // A mid-load reap leaves the status row keyed to a runtime nothing reads.
    expect($providerWaitSessions.get()['rt-1']).toBeUndefined()
  })

  // The negative that matters: a drop guard written as "drop whatever the atom
  // left" would pass every other case here and destroy the warm-switch
  // repaint, which reads the outgoing session's slice.
  it('leaves an unretained session alone across an unrelated atom change', () => {
    $activeSessionId.set('rt-warm')
    publishSessionState('rt-warm', state('stored-1', { busy: true }))
    publishSessionState('rt-warm', state('stored-1', { busy: false }))

    $activeSessionId.set('rt-other')

    expect($sessionStates.get()['rt-warm']).toBeDefined()
  })

  // The #122507 shape at the store seam: the reclaim publish clears busy on a
  // state that is now unreferenced, so publish-time eviction would strip the
  // transcript the primary view is still rendering.
  it('keeps a retained transcript whose settle publish would otherwise evict it', () => {
    $activeSessionId.set('rt-1')
    publishSessionState('rt-1', state('stored-1', { busy: true }))
    retainAfterReclaim('rt-1')

    // The reclaim handler's own publish: dead runtime, activity flags cleared.
    publishSessionState('rt-1', state('stored-1', { busy: false }))

    expect($sessionStates.get()['rt-1']?.messages).toHaveLength(1)
  })

  it('does not grow the retention set across a gateway wipe', () => {
    retainAndRebind('rt-1', 'rt-2')
    clearAllSessionStates()

    // A stale retention entry would make a runtime id minted again after the
    // switch look reclaim-retained and get dropped on its first atom change.
    $activeSessionId.set('rt-1')
    publishSessionState('rt-1', state('stored-1', { busy: true }))

    $activeSessionId.set('rt-2')

    expect($sessionStates.get()['rt-1']).toBeDefined()
  })
})
