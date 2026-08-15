import { beforeEach, describe, expect, it } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessionStates, $sessionTiles, releaseSessionTranscript } from '@/store/session-states'

import { SessionStateCache } from './session-state-cache'

function transcript(id: string, text = id): ChatMessage[] {
  return [
    { id: `${id}-user`, role: 'user', parts: [{ type: 'text', text }] },
    { id: `${id}-assistant`, role: 'assistant', parts: [{ type: 'text', text: `reply ${text}` }] }
  ]
}

function settled(storedSessionId: string, text = storedSessionId): ClientSessionState {
  return { ...createClientSessionState(storedSessionId), messages: transcript(storedSessionId, text) }
}

describe('SessionStateCache', () => {
  beforeEach(() => {
    $sessionStates.set({})
    $sessionTiles.set([])
  })

  it('bounds warm settled transcripts by LRU count and cleans ownership atomically', () => {
    const owners = new Map<string, string>()
    const evicted: string[] = []

    const cache = new SessionStateCache(
      {
        isReferenced: () => false,
        onEvict: (runtimeId, state) => {
          if (state.storedSessionId && owners.get(state.storedSessionId) === runtimeId) {
            owners.delete(state.storedSessionId)
          }

          evicted.push(runtimeId)
        }
      },
      { maxBytes: Number.POSITIVE_INFINITY, maxCount: 2 }
    )

    for (const id of ['a', 'b', 'c']) {
      owners.set(`stored-${id}`, `runtime-${id}`)
      cache.set(`runtime-${id}`, settled(`stored-${id}`))
    }

    // A read makes A warmer than B, so B is the oldest when pruning.
    cache.get('runtime-a')
    cache.prune()

    expect([...cache.keys()].sort()).toEqual(['runtime-a', 'runtime-c'])
    expect(evicted).toEqual(['runtime-b'])
    expect(owners.has('stored-b')).toBe(false)

    // A recycled reverse mapping is not owned by the evicted runtime and must
    // survive cleanup.
    owners.set('stored-a', 'runtime-new-owner')
    cache.set('runtime-d', settled('stored-d'))
    owners.set('stored-d', 'runtime-d')
    cache.prune()
    expect(owners.get('stored-a')).toBe('runtime-new-owner')
  })

  it('uses transcript bytes as well as count', () => {
    const evicted: string[] = []

    const cache = new SessionStateCache(
      { isReferenced: () => false, onEvict: runtimeId => evicted.push(runtimeId) },
      { maxBytes: 600, maxCount: 10 }
    )

    cache.set('small', settled('small', 'x'))
    cache.set('large', settled('large', 'x'.repeat(500)))
    cache.prune()

    expect(evicted).toEqual(['small', 'large'])
    expect(cache.size).toBe(0)
  })

  it.each([
    ['active', (state: ClientSessionState) => state, true],
    ['tiled', (state: ClientSessionState) => state, true],
    ['busy', (state: ClientSessionState) => ({ ...state, busy: true }), false],
    ['awaiting', (state: ClientSessionState) => ({ ...state, awaitingResponse: true }), false],
    ['needs input', (state: ClientSessionState) => ({ ...state, needsInput: true }), false]
  ])('never evicts %s transcripts', (_label, decorate, referenced) => {
    const protectedState = decorate(settled('protected'))

    const cache = new SessionStateCache(
      {
        isReferenced: runtimeId => referenced && runtimeId === 'protected',
        onEvict: () => undefined
      },
      { maxBytes: 0, maxCount: 0 }
    )

    cache.set('protected', protectedState)
    cache.prune()

    expect(cache.get('protected')).toBe(protectedState)
  })

  it('keeps unsaved drafts and pending messages out of the eviction pool', () => {
    const draft = { ...createClientSessionState(null), messages: transcript('draft') }
    const pending = settled('pending')
    pending.messages = [{ id: 'pending-assistant', role: 'assistant', parts: [], pending: true }]

    const cache = new SessionStateCache(
      { isReferenced: () => false, onEvict: () => undefined },
      { maxBytes: 0, maxCount: 0 }
    )

    cache.set('draft', draft)
    cache.set('pending', pending)
    cache.prune()

    expect(cache.has('draft')).toBe(true)
    expect(cache.has('pending')).toBe(true)
  })

  it('retains lightweight status while releasing an evicted transcript', () => {
    const state = { ...settled('stored'), needsInput: false }
    $sessionStates.set({ runtime: state })

    releaseSessionTranscript('runtime')

    expect($sessionStates.get().runtime).toMatchObject({ storedSessionId: 'stored', busy: false, needsInput: false })
    expect($sessionStates.get().runtime.messages).toEqual([])
  })
})
