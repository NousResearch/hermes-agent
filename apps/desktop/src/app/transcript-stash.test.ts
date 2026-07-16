// In-memory transcript stash (instant session switch). Contract:
//  1. stash + read round-trips rows; read refreshes LRU position.
//  2. Bounded LRU: oldest entry evicted past MAX_ENTRIES (8).
//  3. Empty rows clear the entry (never paint a stale blank).
//  4. cull removes an entry (delete wire); clear drops everything.
//  5. null/undefined ids are safe no-ops.
import { beforeEach, describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import {
  clearStashedTranscripts,
  cullStashedTranscript,
  readStashedTranscript,
  stashTranscript,
  stashedTranscriptCount
} from './transcript-stash'

const rows = (id: string): ChatMessage[] => [
  { id: `${id}-1`, role: 'user', parts: [{ type: 'text', text: `hello ${id}` }], timestamp: 1 } as ChatMessage
]

describe('transcript-stash', () => {
  beforeEach(() => {
    clearStashedTranscripts()
  })

  it('round-trips rows for a stashed session', () => {
    const r = rows('a')

    stashTranscript('a', r)
    expect(readStashedTranscript('a')).toBe(r)
  })

  it('misses on unknown, null, and empty ids', () => {
    expect(readStashedTranscript('nope')).toBeNull()
    expect(readStashedTranscript(null)).toBeNull()
    expect(readStashedTranscript(undefined)).toBeNull()
    expect(readStashedTranscript('')).toBeNull()
  })

  it('stashing empty rows clears the entry', () => {
    stashTranscript('a', rows('a'))
    stashTranscript('a', [])
    expect(readStashedTranscript('a')).toBeNull()
    expect(stashedTranscriptCount()).toBe(0)
  })

  it('evicts the least-recently-used entry past the cap', () => {
    for (let i = 0; i < 9; i++) {
      stashTranscript(`s${i}`, rows(`s${i}`))
    }

    expect(stashedTranscriptCount()).toBe(8)
    expect(readStashedTranscript('s0')).toBeNull() // oldest evicted
    expect(readStashedTranscript('s8')).not.toBeNull()
  })

  it('a read refreshes LRU position', () => {
    for (let i = 0; i < 8; i++) {
      stashTranscript(`s${i}`, rows(`s${i}`))
    }

    // Touch s0 so it becomes newest, then push one more entry.
    expect(readStashedTranscript('s0')).not.toBeNull()
    stashTranscript('s8', rows('s8'))

    expect(readStashedTranscript('s0')).not.toBeNull() // survived
    expect(readStashedTranscript('s1')).toBeNull() // evicted instead
  })

  it('re-stashing refreshes LRU position', () => {
    for (let i = 0; i < 8; i++) {
      stashTranscript(`s${i}`, rows(`s${i}`))
    }

    stashTranscript('s0', rows('s0-new'))
    stashTranscript('s8', rows('s8'))

    expect(readStashedTranscript('s0')).not.toBeNull()
    expect(readStashedTranscript('s1')).toBeNull()
  })

  it('cull removes the entry; null cull is a no-op', () => {
    stashTranscript('a', rows('a'))
    cullStashedTranscript('a')
    expect(readStashedTranscript('a')).toBeNull()
    cullStashedTranscript(null)
    cullStashedTranscript(undefined)
  })

  it('stash with a null id is a no-op', () => {
    stashTranscript(null, rows('x'))
    stashTranscript(undefined, rows('x'))
    expect(stashedTranscriptCount()).toBe(0)
  })
})
