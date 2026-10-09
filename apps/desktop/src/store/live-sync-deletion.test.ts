import { afterEach, expect, it } from 'vitest'

import { handleLifecycleEvent } from '@/app/session/hooks/use-message-stream/gateway-event/lifecycle'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { makeSessionInfo } from '@/test/session-info'

import { $cronSessions, $messagingSessions, $sessions, setCronSessions, setMessagingSessions, setSessions } from './session'
import { notifySessionsDeleted } from './session-live-deletion'
import {
  $removedSessionIds,
  captureSessionTombstoneGenerations,
  sessionRemovalIntersected,
  untombstoneSessions
} from './session-removal'
import { $sessionStates, clearAllSessionStates, publishSessionState } from './session-states'

afterEach(() => {
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  clearAllSessionStates()
  untombstoneSessions([...$removedSessionIds.get()])
})

it('evicts exact sidebar IDs synchronously and fences pages even after tree pruning', () => {
  const snapshot = captureSessionTombstoneGenerations()
  const rows = [makeSessionInfo({ id: 'delete-id' }), makeSessionInfo({ id: 'keep-id' })]
  setSessions(rows)
  setCronSessions(rows)
  setMessagingSessions(rows)
  notifySessionsDeleted({ session_ids: ['delete-id', 'not-loaded-id'], profile: 'default' })

  for (const store of [$sessions, $cronSessions, $messagingSessions]) {
    expect(store.get().map(row => row.id)).toEqual(['keep-id'])
  }

  expect($removedSessionIds.get().has('delete-id')).toBe(true)
  // The tree's catch-up prune cannot let an older sidebar page resurrect rows.
  $removedSessionIds.set(new Set())
  expect(sessionRemovalIntersected(snapshot, 'delete-id')).toBe(true)
  expect(sessionRemovalIntersected(snapshot, 'not-loaded-id')).toBe(true)
  expect(sessionRemovalIntersected(snapshot, 'keep-id')).toBe(false)
})

it('routes native deletion only from the active connection without touching transcript state', () => {
  const payload = { session_ids: ['wire-id'], profile: 'default' }
  setMessagingSessions([makeSessionInfo({ id: 'wire-id' })])
  publishSessionState('idle-runtime', createClientSessionState('wire-id'))
  const transcript = $sessionStates.get()

  const context = {
    event: { type: 'sessions.deleted', payload },
    payload,
    deps: {},
    fromActiveSource: () => false
  } as unknown as GatewayEventContext

  expect(handleLifecycleEvent(context)).toBe(true)
  expect($messagingSessions.get()).toHaveLength(1)
  context.fromActiveSource = () => true
  expect(handleLifecycleEvent(context)).toBe(true)
  expect($messagingSessions.get()).toHaveLength(0)
  expect($sessionStates.get()).toBe(transcript)
})

it.each(['busy', 'awaitingResponse', 'turnLive', 'needsInput'] as const)(
  'preserves another profile and a %s writer',
  flag => {
    setMessagingSessions([
      makeSessionInfo({ id: 'busy-id', profile: 'default' }),
      makeSessionInfo({ id: 'other-id', profile: 'other' }),
      makeSessionInfo({ id: 'collision-id', profile: 'default' }),
      makeSessionInfo({ id: 'collision-id', profile: 'other' })
    ])
    publishSessionState('busy-runtime', { ...createClientSessionState('busy-id'), [flag]: true })
    notifySessionsDeleted({ session_ids: ['busy-id', 'other-id', 'collision-id'], profile: 'default' })
    expect($messagingSessions.get()).toHaveLength(4)
    expect($removedSessionIds.get().size).toBe(0)
  }
)
