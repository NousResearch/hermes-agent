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

  describe('authoritative liveness probe (#95189)', () => {
    function cacheWithAuthority(evicted: string[]): SessionStateCache {
      return new SessionStateCache(
        {
          isReferenced: () => false,
          onEvict: runtimeId => evicted.push(runtimeId),
          isAuthoritativelyActive: runtimeId => {
            const live = $sessionStates.get()[runtimeId]

            return Boolean(live && (live.busy || live.awaitingResponse))
          }
        },
        { maxBytes: 0, maxCount: 0 }
      )
    }

    it.each([
      ['busy', (state: ClientSessionState) => ({ ...state, busy: true })],
      ['awaiting', (state: ClientSessionState) => ({ ...state, awaitingResponse: true })]
    ])('evicts an orphaned %s transcript once the authoritative record settles', (_label, decorate) => {
      const evicted: string[] = []
      const cache = cacheWithAuthority(evicted)
      const orphaned = decorate(settled('orphaned'))

      // Mid-turn the snapshot and the authoritative record agree: protection
      // must hold exactly as it does without the probe.
      $sessionStates.set({ orphaned })
      cache.set('orphaned', orphaned)
      cache.prune()
      expect(cache.get('orphaned')).toBe(orphaned)

      // The minting connection dies mid-turn. Reconnect reconciliation
      // settles the authoritative record, but the respawned backend re-mints
      // runtime ids — no event will ever reach this snapshot again, so its
      // frozen in-flight flags must stop pinning the transcript.
      $sessionStates.set({ orphaned: { ...orphaned, busy: false, awaitingResponse: false } })
      cache.prune()

      expect(cache.has('orphaned')).toBe(false)
      expect(evicted).toEqual(['orphaned'])
    })

    it('evicts an in-flight transcript whose authoritative record was dropped entirely', () => {
      const evicted: string[] = []
      const cache = cacheWithAuthority(evicted)
      const working = { ...settled('working'), busy: true }

      $sessionStates.set({ working })
      cache.set('working', working)
      cache.prune()
      expect(cache.has('working')).toBe(true)

      // A soft gateway-mode apply wipes every authoritative state; surviving
      // snapshots describe dead runtimes (#95189 reconnect churn).
      $sessionStates.set({})
      cache.prune()

      expect(cache.has('working')).toBe(false)
      expect(evicted).toEqual(['working'])
    })

    it('keeps an in-flight transcript pinned while the authoritative store still claims work', () => {
      const evicted: string[] = []
      const cache = cacheWithAuthority(evicted)
      const working = { ...settled('working'), busy: true }

      $sessionStates.set({ working })
      cache.set('working', working)
      cache.prune()

      expect(cache.get('working')).toBe(working)
      expect(evicted).toEqual([])
    })

    it.each([
      ['busy', (state: ClientSessionState) => ({ ...state, busy: true })],
      ['awaiting', (state: ClientSessionState) => ({ ...state, awaitingResponse: true })]
    ])('still never evicts %s transcripts when no authority probe is wired', (_label, decorate) => {
      // Legacy construction: without the probe there is no way to tell a live
      // turn from an orphaned snapshot, so the flags keep blocking eviction.
      const working = decorate(settled('working'))
      $sessionStates.set({ working: { ...working, busy: false, awaitingResponse: false } })

      const cache = new SessionStateCache(
        { isReferenced: () => false, onEvict: () => undefined },
        { maxBytes: 0, maxCount: 0 }
      )

      cache.set('working', working)
      cache.prune()

      expect(cache.get('working')).toBe(working)
    })
  })

  describe('stalled session eviction (#95276)', () => {
    function busy(storedSessionId: string, text = storedSessionId): ClientSessionState {
      return { ...settled(storedSessionId, text), busy: true }
    }

    it('evicts entries gone silent past the stall window even under no size pressure', () => {
      let nowMs = 0
      const evicted: string[] = []

      const cache = new SessionStateCache(
        { isReferenced: () => false, onEvict: runtimeId => evicted.push(runtimeId) },
        // No count/byte pressure at all: only the stall guard can bound these.
        {
          maxBytes: Number.POSITIVE_INFINITY,
          maxCount: Number.POSITIVE_INFINITY,
          stalledMs: 10_000,
          now: () => nowMs
        }
      )

      // Mid-turn orphans: committed as busy, then no further events, ever.
      for (const id of ['a', 'b', 'c']) {
        cache.set(`runtime-${id}`, busy(`stored-${id}`, 'x'.repeat(2048)))
      }

      cache.prune()
      expect(cache.size).toBe(3)

      nowMs += 10_001
      cache.prune()

      // Pre-fix this stays 3 forever: busy transcripts are invisible to the LRU.
      expect(cache.size).toBe(0)
      expect([...evicted].sort()).toEqual(['runtime-a', 'runtime-b', 'runtime-c'])
    })

    it.each([
      ['busy', (state: ClientSessionState) => ({ ...state, busy: true }), true],
      ['awaiting response', (state: ClientSessionState) => ({ ...state, awaitingResponse: true }), true],
      ['needs input', (state: ClientSessionState) => ({ ...state, needsInput: true }), false]
    ])('%s past the stall window is evictable exactly when the wait is not on the user', (_label, decorate, stalls) => {
      let nowMs = 0
      const evicted: string[] = []

      const cache = new SessionStateCache(
        { isReferenced: () => false, onEvict: runtimeId => evicted.push(runtimeId) },
        { stalledMs: 5_000, now: () => nowMs }
      )

      cache.set('runtime', decorate(settled('stored')))
      nowMs += 5_001
      cache.prune()

      expect(cache.has('runtime')).toBe(!stalls)
      expect(evicted).toEqual(stalls ? ['runtime'] : [])
    })

    it('keeps a busy transcript alive while events keep arriving and evicts only after silence', () => {
      let nowMs = 0
      const evicted: string[] = []

      const cache = new SessionStateCache(
        { isReferenced: () => false, onEvict: runtimeId => evicted.push(runtimeId) },
        { stalledMs: 10_000, now: () => nowMs }
      )

      cache.set('runtime', busy('stored', 'chunk-0'))

      // A healthy long turn keeps mutating state; each event resets the window.
      for (let tick = 1; tick <= 4; tick += 1) {
        nowMs += 8_000
        cache.set('runtime', busy('stored', `chunk-${tick}`))
        cache.prune()
        expect(cache.has('runtime')).toBe(true)
      }

      // Events stop; silence outlasts the window.
      nowMs += 10_001
      cache.prune()

      expect(cache.has('runtime')).toBe(false)
      expect(evicted).toEqual(['runtime'])
    })

    it('never stall-evicts drafts, in-flight messages, or referenced states', () => {
      let nowMs = 0

      const cache = new SessionStateCache(
        { isReferenced: runtimeId => runtimeId === 'referenced', onEvict: () => undefined },
        { stalledMs: 1, now: () => nowMs }
      )

      const pending = busy('stored-pending')
      pending.messages = [{ id: 'pending-assistant', role: 'assistant', parts: [], pending: true }]
      const draft = { ...createClientSessionState(null), messages: transcript('draft'), busy: true }

      cache.set('pending', pending)
      cache.set('draft', draft)
      cache.set('referenced', busy('stored-referenced'))
      nowMs += 2
      cache.prune()

      expect(cache.has('pending')).toBe(true)
      expect(cache.has('draft')).toBe(true)
      expect(cache.has('referenced')).toBe(true)
    })
  })
})
