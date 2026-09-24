import { describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import {
  createPersistedDisplayTranscriptProvenance,
  hasPersistedDisplayTranscriptProvenance,
  invalidatePersistedDisplayTranscriptAuthority,
  suppressTranscriptForView,
  withoutTranscriptProvenance
} from './transcript-provenance'

const expected = createPersistedDisplayTranscriptProvenance({
  lineageRootId: 'root-1',
  scope: { connectionId: 'conn-1', profile: 'coder' },
  storedSessionId: 'stored-1'
})

describe('transcript provenance', () => {
  it('matches only the same connection, profile, stored id, and lineage', () => {
    const state = createClientSessionState('stored-1')
    state.transcriptProvenance = expected

    expect(hasPersistedDisplayTranscriptProvenance(state, expected)).toBe(true)
    expect(
      hasPersistedDisplayTranscriptProvenance(state, {
        ...expected,
        lineageRootId: 'root-2'
      })
    ).toBe(false)
    expect(
      hasPersistedDisplayTranscriptProvenance(state, {
        ...expected,
        profile: 'default'
      })
    ).toBe(false)
  })

  it('strips proof and bumps the authority epoch on invalidation', () => {
    const state = createClientSessionState('stored-1')
    state.transcriptProvenance = expected
    state.transcriptAuthorityEpoch = 3

    const next = invalidatePersistedDisplayTranscriptAuthority(state)

    expect(next.transcriptProvenance).toBeUndefined()
    expect(next.transcriptAuthorityEpoch).toBe(4)
    expect(withoutTranscriptProvenance(state).transcriptProvenance).toBeUndefined()
  })

  it('hides messages from the view without dropping the cache entry', () => {
    const state = createClientSessionState('stored-1')
    state.messages = [{ id: 'u1', role: 'user', parts: [{ type: 'text', text: 'hi' }] }]

    // No gate held: the state reaches the view verbatim.
    expect(suppressTranscriptForView(state, null)).toBe(state)

    // Gate held with no captured prefix: fail-closed, everything hides.
    expect(suppressTranscriptForView(state, { cutoffIds: new Set() }).messages).toEqual([])
    expect(state.messages).toHaveLength(1)
  })

  it('drops only the cached prefix and keeps rows appended after the arm (#117867)', () => {
    const state = createClientSessionState('stored-1')
    state.messages = [
      { id: 'cached-1', role: 'user', parts: [{ type: 'text', text: 'old' }] },
      { id: 'live-1', role: 'assistant', parts: [{ type: 'text', text: 'streaming' }] }
    ]

    const suppressed = suppressTranscriptForView(state, { cutoffIds: new Set(['cached-1']) })

    expect(suppressed.messages.map(message => message.id)).toEqual(['live-1'])

    // Nothing else in the state is touched, and the input is not mutated.
    expect(state.messages).toHaveLength(2)
  })
})
