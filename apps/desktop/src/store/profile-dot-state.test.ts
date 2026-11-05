import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import { $fleetRoster } from './fleet-roster'
import { $activeGatewayProfile, $profiles } from './profile'
import { $profileDotStateByScope, profileDotScopeKey } from './profile-dot-state'
import {
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  setSessions
} from './session'
import { clearAllSessionStates, publishSessionState, recordSessionEventScope } from './session-states'
import { $sessionSeenCounts, $unreadFinishedMarkers } from './session-unread'

const session = (over: Partial<SessionInfo>): SessionInfo => makeSessionInfo({ id: 'live', ...over })

const profileInfo = (name: string, is_default = false) => ({
  has_env: false,
  is_default,
  model: null,
  name,
  path: `/profiles/${name}`,
  provider: null,
  skill_count: 0
})

function resetAll() {
  clearAllSessionStates()
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $selectedStoredSessionId.set(null)
  $activeGatewayProfile.set('default')
  $sessionSeenCounts.set({})
  $unreadFinishedMarkers.set({})
  $unreadFinishedSessionIds.set([])
  $fleetRoster.set(null)
  $profiles.set([profileInfo('default', true), profileInfo('writer')])
}

const summaryFor = (connectionId: null | string, profile: string) =>
  $profileDotStateByScope.get()[profileDotScopeKey(connectionId, profile)]

describe('profile rail dot state ($profileDotStateByScope)', () => {
  beforeEach(resetAll)
  afterEach(resetAll)

  it("lights the inactive profile's scope when its session finishes unread (#91710)", () => {
    // Profile writer's turn runs; the user is homed on default.
    setSessions([session({ id: 's1', message_count: 3, profile: 'writer' })])

    publishSessionState('r1', { ...createClientSessionState('s1'), busy: true })
    publishSessionState('r1', { ...createClientSessionState('s1'), busy: false })

    // The busy→idle edge persisted the marker into the ROW's profile bucket.
    expect($unreadFinishedMarkers.get()['writer']).toBeDefined()

    // Profile switch drains the list: the marker alone must keep the square lit.
    setSessions([])

    const writer = summaryFor(null, 'writer')

    expect(writer).toBeDefined()
    expect(writer?.state).toBe('unread')
    expect(writer?.unreadCount).toBe(1)
    expect(summaryFor(null, 'default')).toBeUndefined()
  })

  it('ranks needs-input over working within one profile', () => {
    setSessions([
      session({ id: 's1', message_count: 3, profile: 'writer' }),
      session({ id: 's2', message_count: 3, profile: 'writer' })
    ])

    publishSessionState('r1', { ...createClientSessionState('s1'), busy: true })
    publishSessionState('r2', { ...createClientSessionState('s2'), needsInput: true })

    const writer = summaryFor(null, 'writer')

    expect(writer?.state).toBe('needs-input')
    expect(writer?.needsInputCount).toBe(1)
    expect(writer?.workingCount).toBe(1)
  })

  it('ranks working over unread within one profile', () => {
    setSessions([
      session({ id: 's1', message_count: 3, profile: 'writer' }),
      session({ id: 's2', message_count: 3, profile: 'writer' })
    ])

    publishSessionState('r1', { ...createClientSessionState('s1'), busy: true })
    publishSessionState('r2', { ...createClientSessionState('s2'), busy: true })
    // s2 finishes unread while s1 is still running.
    publishSessionState('r2', { ...createClientSessionState('s2'), busy: false })

    const writer = summaryFor(null, 'writer')

    expect(writer?.state).toBe('working')
    expect(writer?.workingCount).toBe(1)
    expect(writer?.unreadCount).toBe(1)
  })

  it('keeps a same-id session in another profile from cross-marking', () => {
    // writer's finished session shares default's stored id; only writer holds
    // the marker, so only writer's square lights.
    $unreadFinishedMarkers.set({ writer: ['s1'] })
    setSessions([session({ id: 's1', message_count: 3, profile: 'default' })])

    expect(summaryFor(null, 'writer')?.unreadCount).toBe(1)
    expect(summaryFor(null, 'default')).toBeUndefined()
  })

  it('counts a backend-unread row in its own profile', () => {
    setSessions([session({ id: 's9', message_count: 2, profile: 'default', unread: true })])

    const summary = summaryFor(null, 'default')

    expect(summary?.state).toBe('unread')
    expect(summary?.unreadCount).toBe(1)
  })

  it('counts a marker-backed unread row once (row and marker agree)', () => {
    $unreadFinishedMarkers.set({ writer: ['s1'] })
    $sessionSeenCounts.set({ writer: { s1: 3 } })
    setSessions([session({ id: 's1', message_count: 5, profile: 'writer' })])

    expect(summaryFor(null, 'writer')?.unreadCount).toBe(1)
  })

  it('does not attribute ambiguous bare-profile markers when two gateways share the profile name', () => {
    $unreadFinishedMarkers.set({ writer: ['s1'] })
    $fleetRoster.set({
      agents: [
        {
          connectionId: 'remote-1',
          connectionKind: 'ssh',
          connectionLabel: 'Box',
          handle: 'h1',
          profile: 'writer'
        }
      ],
      sources: []
    })

    // The local gateway exposes 'writer' too, so the persisted marker (bucketed
    // by bare profile name) cannot say which machine owns it: no dot, rather
    // than a wrong one.
    expect(summaryFor(null, 'writer')).toBeUndefined()
  })

  it('attributes an unlisted working session to its owning gateway scope', () => {
    recordSessionEventScope({ connectionId: 'conn-1', profile: 'writer', session_id: 'r4' })
    publishSessionState('r4', { ...createClientSessionState('s4'), busy: true })

    const summary = summaryFor('conn-1', 'writer')

    expect(summary?.state).toBe('working')
    expect(summary?.workingCount).toBe(1)
  })

  it('writes the finish marker to the socket-proven profile, not the active one', () => {
    // No loaded row: without the event-proven owner the marker would fall back
    // to the ACTIVE gateway profile and light the wrong square (#91710).
    recordSessionEventScope({ connectionId: 'conn-9', profile: 'writer', session_id: 'r3' })

    publishSessionState('r3', { ...createClientSessionState('s3'), busy: true })
    publishSessionState('r3', { ...createClientSessionState('s3'), busy: false })

    expect($unreadFinishedMarkers.get()['writer']).toBeDefined()
    expect($unreadFinishedMarkers.get()['default']).toBeUndefined()
  })
})
