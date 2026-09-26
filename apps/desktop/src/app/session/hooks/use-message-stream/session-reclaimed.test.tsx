import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { resetRuntimeGoneHealing } from '@/store/runtime-gone'
import { $activeSessionId, $sessionResumeRequest } from '@/store/session'
import { $sessionStates, $sessionTiles, publishSessionState } from '@/store/session-states'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

// `session.reclaimed`: the backend tore down a live session we're still
// holding (idle TTL, LRU cap, WS-orphan reap). Before this event the runtime id
// stayed cached until something failed against it, which read to the user as
// the session vanishing rather than being reclaimed.

const ACTIVE_SID = 'session-active'
const ACTIVE_PROFILE = 'compass'
let stream: MessageStreamHarness
let queryClient: QueryClient
let wiringCache: Map<string, ClientSessionState>

function mountStream(sessionId: string = ACTIVE_SID) {
  stream = renderMessageStream(sessionId, { activeGatewayProfile: ACTIVE_PROFILE, queryClient })
  wiringCache = stream.states
}

const reclaim = (sessionId: string, reason = 'ws_orphan_reap') =>
  act(() =>
    stream.handleEvent({
      payload: { reason, session_id: sessionId, stored_session_id: 'stored-1' },
      session_id: '',
      type: 'session.reclaimed'
    } as GatewayEvent)
  )

beforeEach(() => {
  queryClient = new QueryClient()
  resetRuntimeGoneHealing()
  $sessionStates.set({})
  $sessionTiles.set([])
  $activeSessionId.set(null)
  $sessionResumeRequest.set(null)
})

afterEach(() => {
  cleanup()
  resetRuntimeGoneHealing()
  $sessionStates.set({})
  $sessionTiles.set([])
  $activeSessionId.set(null)
  $sessionResumeRequest.set(null)
  vi.restoreAllMocks()
})

describe('session.reclaimed', () => {
  it('leaves every other live session alone', () => {
    mountStream('observer')
    publishSessionState('live-gone', createClientSessionState())
    publishSessionState('live-kept', createClientSessionState())

    reclaim('live-gone')

    // Both halves matter: the target was dropped, the bystander stayed. Asserting
    // only the survivor would pass with no handler at all.
    expect($sessionStates.get()['live-gone']).toBeUndefined()
    expect($sessionStates.get()['live-kept']).toBeDefined()
  })

  it('ignores a payload with no runtime id instead of clearing everything', () => {
    mountStream()
    publishSessionState('live-a', createClientSessionState())
    publishSessionState('live-b', createClientSessionState())

    reclaim('')

    // A malformed/empty id must be a no-op, never a blanket wipe.
    expect(Object.keys($sessionStates.get()).sort()).toEqual(['live-a', 'live-b'])
  })

  // A TILE bound to the reclaimed runtime is the #82620 blank-pane case: the
  // state drop above empties the tile's view, but with `runtimeId` still set
  // the tile's resume effect (gated on !runtimeId) never refires — an empty
  // transcript under live chrome, unrecoverable without closing the tab.
  it('unbinds a tile holding the reclaimed runtime so its resume can refire', () => {
    mountStream()
    publishSessionState('live-gone', createClientSessionState('stored-1'))
    $sessionTiles.set([
      { runtimeId: 'live-gone', storedSessionId: 'stored-1' },
      { runtimeId: 'live-kept', storedSessionId: 'stored-2' }
    ])

    reclaim('live-gone')

    const tiles = $sessionTiles.get()
    // The reclaimed tile survives as a pane (its stored session is intact) but
    // sheds the dead runtime; the bystander tile keeps its live binding.
    expect(tiles.find(t => t.storedSessionId === 'stored-1')?.runtimeId).toBeUndefined()
    expect(tiles.find(t => t.storedSessionId === 'stored-2')?.runtimeId).toBe('live-kept')
  })

  // The wiring cache is resumeTile's warm path: a leftover entry for the dead
  // runtime would be handed straight back on the re-resume, rebinding the tile
  // to the same reclaimed id the backend just forgot.
  it('purges the wiring cache entry for the reclaimed runtime', () => {
    mountStream()
    wiringCache.set('live-gone', createClientSessionState('stored-1'))
    wiringCache.set('live-kept', createClientSessionState('stored-2'))

    reclaim('live-gone')

    expect(wiringCache.has('live-gone')).toBe(false)
    expect(wiringCache.has('live-kept')).toBe(true)
  })

  it('keeps the atom-visible transcript when the imperative routing ref is stale', () => {
    mountStream('stale-ref')
    $activeSessionId.set('live-visible')
    publishSessionState('live-visible', {
      ...createClientSessionState('stored-1'),
      awaitingResponse: true,
      busy: true,
      messages: [{ id: 'user-1', parts: [{ text: 'visible turn', type: 'text' }], role: 'user' }],
      needsInput: true,
      streamId: 'stream-1',
      turnLive: true
    })

    // The event names the atom-visible runtime explicitly; the hook's ref still
    // points at a different runtime on purpose.
    act(() =>
      stream.handleEvent({
        payload: { reason: 'ws_orphan_reap', session_id: 'live-visible', stored_session_id: 'stored-1' },
        session_id: 'live-visible',
        type: 'session.reclaimed'
      } as GatewayEvent)
    )

    // The rendered primary view follows the atom, not the deliberately mutable
    // write-routing ref. Preserve the slice that actually remains on screen.
    expect($sessionStates.get()['live-visible']).toMatchObject({
      awaitingResponse: false,
      busy: false,
      messages: [{ id: 'user-1' }],
      needsInput: false,
      streamId: null,
      turnLive: false
    })
    expect($sessionStates.get()['stale-ref']).toBeUndefined()
    expect($sessionResumeRequest.get()?.sessionId).toBe('stored-1')
  })

  it('resumes the atom-visible durable session when it has no local state slice', () => {
    mountStream('stale-ref')
    $activeSessionId.set(ACTIVE_SID)

    reclaim(ACTIVE_SID)

    expect($sessionResumeRequest.get()?.sessionId).toBe('stored-1')
    expect($sessionStates.get()[ACTIVE_SID]).toBeUndefined()
  })

  it('does not navigate the primary chat when a background runtime is reclaimed', () => {
    mountStream()
    $activeSessionId.set('live-kept')
    publishSessionState('live-gone', createClientSessionState('stored-1'))
    publishSessionState('live-kept', createClientSessionState('stored-2'))

    reclaim('live-gone')

    expect($sessionResumeRequest.get()).toBeNull()
  })
})
