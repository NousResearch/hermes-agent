import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { acknowledgedTranscriptBoundary, conflictingTranscriptIdentity } from './pending-turn-identity'

const user = (id: string, clientMessageId: string, rowId?: number): ChatMessage => ({
  id,
  role: 'user',
  parts: [{ type: 'text', text: 'same repeated prompt' }],
  clientMessageId,
  ...(rowId === undefined ? {} : { rowId })
})

describe('pending turn identity', () => {
  it('lets stable client identity bridge a gateway queue-row replacement', () => {
    expect(conflictingTranscriptIdentity(user('local', 'client-a'), user('stored', 'client-a', 11))).toBe(false)
    expect(conflictingTranscriptIdentity(user('local', 'client-a'), user('stored', 'client-b', 11))).toBe(true)

    expect(conflictingTranscriptIdentity(user('local', 'client-a', 11), user('stored', 'client-b', 11))).toBe(false)
    expect(
      conflictingTranscriptIdentity(user('queued-accept', 'client-a', 10), user('drained-turn', 'client-a', 11))
    ).toBe(false)
  })

  it('finds the acknowledged boundary by client id without sorting timestamps', () => {
    const previous = [user('optimistic-newer-clock', 'client-a')]
    previous[0].timestamp = 2_000
    const next = [user('hydrated-older-clock', 'client-a', 42)]
    next[0].timestamp = 1_000

    expect(acknowledgedTranscriptBoundary(next, previous)).toEqual({ localIndex: 0, storedIndex: 0 })
  })
})
