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
  displayRevision: 7,
  lineageRootId: 'root-1',
  resolvedTipId: 'tip-2',
  scope: { connectionId: 'conn-1', profile: 'coder' },
  storedSessionId: 'stored-1'
})!

describe('transcript provenance', () => {
  it.each([
    ['source', 'other'],
    ['connectionId', 'conn-2'],
    ['profile', 'default'],
    ['storedSessionId', 'stored-2'],
    ['lineageRootId', 'root-2'],
    ['resolvedTipId', 'tip-3'],
    ['displayRevision', 8],
    ['coverage', 'other-page']
  ] as const)('rejects a %s mismatch', (field, value) => {
    const state = createClientSessionState('stored-1')
    state.transcriptProvenance = { ...expected, [field]: value } as never

    expect(hasPersistedDisplayTranscriptProvenance(state, expected)).toBe(false)
  })

  it('matches the exact persisted-display proof', () => {
    const state = createClientSessionState('stored-1')
    state.transcriptProvenance = expected

    expect(hasPersistedDisplayTranscriptProvenance(state, expected)).toBe(true)
  })

  it.each([
    ['lineageRootId', null],
    ['lineageRootId', ''],
    ['resolvedTipId', null],
    ['resolvedTipId', ''],
    ['displayRevision', Number.NaN],
    ['displayRevision', Number.POSITIVE_INFINITY],
    ['displayRevision', -1],
    ['displayRevision', 1.5],
    ['displayRevision', '7']
  ] as const)('does not create proven provenance with invalid %s=%s', (field, value) => {
    const candidate = createPersistedDisplayTranscriptProvenance({
      displayRevision: 7,
      lineageRootId: 'root-1',
      resolvedTipId: 'tip-2',
      scope: { connectionId: 'conn-1', profile: 'coder' },
      storedSessionId: 'stored-1',
      [field]: value
    } as never)

    expect(candidate).toBeNull()
  })

  it('normalizes scope and identity values before creating proof', () => {
    expect(
      createPersistedDisplayTranscriptProvenance({
        displayRevision: 7,
        lineageRootId: ' root-1 ',
        resolvedTipId: ' tip-2 ',
        scope: { connectionId: ' conn-1 ', profile: ' ' },
        storedSessionId: ' stored-1 '
      })
    ).toEqual({
      connectionId: 'conn-1',
      coverage: 'latest-page',
      displayRevision: 7,
      lineageRootId: 'root-1',
      profile: 'default',
      resolvedTipId: 'tip-2',
      source: 'persisted-display',
      storedSessionId: 'stored-1'
    })
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
})
