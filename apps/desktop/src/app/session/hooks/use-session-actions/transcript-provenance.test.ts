import { afterEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'

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
  afterEach(() => clearClarifyRequest())
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

  it('projects only the current owned request through a held transcript, even with no cached messages', () => {
    const state = createClientSessionState('stored-1')
    const request = { requestId: 'live', sessionId: 'runtime-1', question: 'Trusted?', choices: ['Yes'], multiSelect: false }
    setClarifyRequest(request)
    const emptyProjection = suppressTranscriptForView(state, true, 'runtime-1')
    expect(emptyProjection.messages).toHaveLength(1)
    state.messages = [{ id: 'old', role: 'assistant', parts: [{ type: 'text', text: 'UNVERIFIED' }] }]
    const projected = suppressTranscriptForView(state, true, 'runtime-1')
    expect(projected.messages).toEqual(emptyProjection.messages)
    expect(projected.messages[0].parts).toEqual([expect.objectContaining({ toolCallId: 'live', args: { question: 'Trusted?', choices: ['Yes'] } })])
    expect(projected.transcriptProvenance).toBeUndefined()
    expect(state.messages[0].id).toBe('old')
    expect(suppressTranscriptForView(state, true, 'runtime-2').messages).toEqual([])
    $clarifyRequests.set({ 'runtime-1': { ...request, sessionId: 'runtime-2' } })
    expect(suppressTranscriptForView(state, true, 'runtime-1').messages).toEqual([])
    clearClarifyRequest()
    expect(suppressTranscriptForView(state, true, 'runtime-1').messages).toEqual([])
  })

  it('hides messages from the view without dropping the cache entry', () => {
    const state = createClientSessionState('stored-1')
    state.messages = [{ id: 'u1', role: 'user', parts: [{ type: 'text', text: 'hi' }] }]

    expect(suppressTranscriptForView(state, false)).toBe(state)
    expect(suppressTranscriptForView(state, true).messages).toEqual([])
    expect(state.messages).toHaveLength(1)
  })
})
