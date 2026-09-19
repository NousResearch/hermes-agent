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

    expect(suppressTranscriptForView(state, false)).toBe(state)
    expect(suppressTranscriptForView(state, true).messages).toEqual([])
    expect(state.messages).toHaveLength(1)
  })

  it('keeps the open clarify part correlated to the active input request', () => {
    const state = createClientSessionState('stored-1')
    state.needsInput = true
    state.streamId = 'clarify-1'
    state.messages = [
      {
        id: 'clarify-1',
        role: 'assistant',
        pending: true,
        parts: [
          { type: 'text', text: 'unproven commentary' },
          {
            type: 'tool-call',
            toolCallId: 'request-1',
            toolName: 'clarify',
            args: { question: 'Which path?' },
            argsText: '{"question":"Which path?"}'
          }
        ]
      }
    ]

    expect(suppressTranscriptForView(state, true).messages).toEqual([
      {
        ...state.messages[0],
        parts: [expect.objectContaining({ toolCallId: 'request-1', toolName: 'clarify' })]
      }
    ])
  })

  it('hides a matching pending clarify when the state does not need input', () => {
    const state = createClientSessionState('stored-1')
    state.needsInput = false
    state.streamId = 'clarify-1'
    state.messages = [
      {
        id: 'clarify-1',
        role: 'assistant',
        pending: true,
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'request-1',
            toolName: 'clarify',
            args: { question: 'Which path?' },
            argsText: '{"question":"Which path?"}'
          }
        ]
      }
    ]

    expect(suppressTranscriptForView(state, true).messages).toEqual([])
  })

  it('hides a correlated pending clarify from a non-assistant message', () => {
    const state = createClientSessionState('stored-1')
    state.needsInput = true
    state.streamId = 'clarify-1'
    state.messages = [
      {
        id: 'clarify-1',
        role: 'user',
        pending: true,
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'request-1',
            toolName: 'clarify',
            args: { question: 'Which path?' },
            argsText: '{"question":"Which path?"}'
          }
        ]
      }
    ]

    expect(suppressTranscriptForView(state, true).messages).toEqual([])
  })

  it.each([null, 'other-message'])('hides a pending clarify without a matching stream id (%s)', streamId => {
    const state = createClientSessionState('stored-1')
    state.needsInput = true
    state.streamId = streamId
    state.messages = [
      {
        id: 'clarify-1',
        role: 'assistant',
        pending: true,
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'request-1',
            toolName: 'clarify',
            args: { question: 'Which path?' },
            argsText: '{"question":"Which path?"}'
          }
        ]
      }
    ]

    expect(suppressTranscriptForView(state, true).messages).toEqual([])
  })
})
