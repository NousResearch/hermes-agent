import { beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $selectedStoredSessionId, $sessions, $unreadFinishedSessionIds } from '@/store/session'
import { $sessionStates, $sessionTiles, closeSessionTile, publishSessionState } from '@/store/session-states'

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
})

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
