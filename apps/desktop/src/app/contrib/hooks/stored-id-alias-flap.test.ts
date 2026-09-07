import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $activeSessionId,
  $activeSessionStoredIdRotation,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  setActiveSessionStoredIdRotation
} from '@/store/session'
import { $sessionStates, clearAllSessionStates, publishSessionState } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { rehydrateLiveSessionStatuses } from './use-background-sync'

/**
 * Auto-compression rotates a conversation's tip id while its lineage root
 * stays constant, and the gateway's surfaces do not all rename at once:
 * `session.active_list` keeps reporting the session_key the runtime was
 * started with while the sidebar rows (and session.info) carry the rotated
 * tip. Both names ALIAS the same conversation — `sessionMatchesStoredId`
 * resolves either against the same row.
 *
 * The live-status rehydrate used to compare those names with `!==` and
 * republish the snapshot's alias over the renderer's on EVERY poll. Each
 * republish ran handleTransition's rotation detector, which emitted a
 * stored-id "rotation" for what is actually the same conversation; the
 * route-follow effect then chased it and moved $selectedStoredSessionId.
 * With the renderer-side paths (reconcile / session.info) writing the other
 * alias back between polls, the selection flapped A→B→A forever — once per
 * sessions.changed beat — and every flap re-ran the transcript's
 * settle-to-bottom effect, yanking the reader to the newest message (the
 * 10s "scroll to bottom while idle" bug).
 */
describe('rehydrateLiveSessionStatuses — lineage-alias stored ids must not rotate', () => {
  const row = {
    id: 'tip-2',
    _lineage_root_id: 'root-1',
    last_active: 1000,
    profile: 'default',
    title: 'compressed conversation'
  } as unknown as SessionInfo

  beforeEach(() => {
    $sessions.set([row])
    $messagingSessions.set([])
    $activeSessionId.set('runtime-x')
    $selectedStoredSessionId.set('tip-2')
    setActiveSessionStoredIdRotation(null)
  })

  afterEach(() => {
    clearAllSessionStates()
    $sessions.set([])
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    setActiveSessionStoredIdRotation(null)
  })

  it('does not oscillate when snapshot and renderer paths keep writing different aliases', () => {
    publishSessionState('runtime-x', { ...createClientSessionState('tip-2'), busy: true })
    setActiveSessionStoredIdRotation(null)

    const rotations: string[] = []

    const unbind = $activeSessionStoredIdRotation.listen(event => {
      if (event) {
        rotations.push(`${event.previousStoredSessionId}->${event.nextStoredSessionId}`)
      }
    })

    // Two idle beats: each beat = one active_list rehydrate (root alias) and
    // one renderer-side republish of the tip alias (session.info / reconcile).
    for (let beat = 0; beat < 2; beat += 1) {
      rehydrateLiveSessionStatuses({
        sessions: [{ id: 'runtime-x', session_key: 'root-1', status: 'working' }]
      })

      expect($activeSessionStoredIdRotation.get()).toBeNull()
      expect($sessionStates.get()['runtime-x']?.storedSessionId).toBe('tip-2')

      const current = $sessionStates.get()['runtime-x']!
      publishSessionState('runtime-x', { ...current, storedSessionId: 'tip-2' })
    }

    unbind()

    expect(rotations).toEqual([])
  })

  it('still rotates for a genuinely different conversation id', () => {
    publishSessionState('runtime-x', { ...createClientSessionState('tip-2'), busy: true })
    setActiveSessionStoredIdRotation(null)

    // No row aliases tip-2 with unrelated-9: this is a real handoff.
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-x', session_key: 'unrelated-9', status: 'working' }]
    })

    expect($activeSessionStoredIdRotation.get()).toEqual({
      nextStoredSessionId: 'unrelated-9',
      previousStoredSessionId: 'tip-2',
      runtimeSessionId: 'runtime-x'
    })
  })
})
