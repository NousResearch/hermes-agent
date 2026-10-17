import { afterEach, expect, it } from 'vitest'

import { handleLifecycleEvent } from '@/app/session/hooks/use-message-stream/gateway-event/lifecycle'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { makeSessionInfo } from '@/test/session-info'

import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  setCronSessions,
  setMessagingSessions,
  setSessions
} from './session'
import { notifySessionsDeleted } from './session-live-deletion'
import {
  $removedSessionIds,
  captureSessionTombstoneGenerations,
  sessionRemovalIntersected,
  untombstoneSessions
} from './session-removal'
import { $sessionStates, clearAllSessionStates, publishSessionState, recordSessionEventScope } from './session-states'

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

  // External events must never poison the legacy, profile-blind overlay.
  expect($removedSessionIds.get().has('delete-id')).toBe(false)
  // The tree's catch-up prune cannot let an older sidebar page resurrect rows.
  $removedSessionIds.set(new Set())
  expect(sessionRemovalIntersected(snapshot, 'delete-id')).toBe(true)
  expect(sessionRemovalIntersected(snapshot, 'not-loaded-id')).toBe(true)
  expect(sessionRemovalIntersected(snapshot, 'keep-id')).toBe(false)
})

it('fences an unseen deleted ID without fencing a foreign-profile page', () => {
  const snapshot = captureSessionTombstoneGenerations()
  notifySessionsDeleted({ session_ids: ['unseen-collision'], profile: 'default' })
  expect(sessionRemovalIntersected(snapshot, 'unseen-collision', 'default')).toBe(true)
  expect(sessionRemovalIntersected(snapshot, 'unseen-collision', 'other')).toBe(false)
  // A fresh authoritative read may re-admit an ID restored after deletion.
  expect(sessionRemovalIntersected(captureSessionTombstoneGenerations(), 'unseen-collision', 'default')).toBe(false)
})

it.each([setCronSessions, setMessagingSessions])('protects a live lineage tip in a dedicated sidebar list', setRows => {
  setRows([makeSessionInfo({ id: 'live-tip', profile: 'default', _lineage_ids: ['live-root', 'live-tip'] })])
  publishSessionState('lineage-writer', { ...createClientSessionState('live-root'), busy: true })
  notifySessionsDeleted({ session_ids: ['live-tip'], profile: 'default' })
  expect([...$cronSessions.get(), ...$messagingSessions.get()].map(row => row.id)).toEqual(['live-tip'])
})

it('does not let a known foreign writer block deletion of an owned twin', () => {
  const snapshot = captureSessionTombstoneGenerations()
  setSessions([makeSessionInfo({ id: 'writer-twin', profile: 'default' })])
  recordSessionEventScope({ session_id: 'foreign-writer', connectionId: 'local', profile: 'other' })
  publishSessionState('foreign-writer', { ...createClientSessionState('writer-twin'), busy: true })
  notifySessionsDeleted({ session_ids: ['writer-twin'], profile: 'default' })
  expect($sessions.get()).toHaveLength(0)
  expect(sessionRemovalIntersected(snapshot, 'writer-twin', 'default')).toBe(true)
  expect($sessionStates.get()['foreign-writer'].busy).toBe(true)
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
    expect($messagingSessions.get().map(row => [row.id, row.profile])).toEqual([
      ['busy-id', 'default'],
      ['other-id', 'other'],
      ['collision-id', 'other']
    ])
    expect($removedSessionIds.get().size).toBe(0)
  }
)
