// In-memory transcript stash (instant session switch). Contract:
//  1. stash + read round-trips rows; read refreshes LRU position.
//  2. Bounded LRU: oldest entry evicted past MAX_STASH_ENTRIES.
//  3. Empty rows clear the entry (never paint a stale blank).
//  4. cull removes an entry (delete wire); clear drops everything.
//  5. null/undefined ids are safe no-ops.
import { beforeEach, describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { MAX_PRELOAD } from './transcript-preload'
import {
  clearStashedTranscripts,
  cullStashedTranscript,
  MAX_STASH_ENTRIES,
  readStashedTranscript,
  stashedTranscriptCount,
  stashTranscript
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

  it('holds the whole boot preload set plus visited sessions without eviction', () => {
    // The cap exists to KEEP the preload set resident — regression guard for
    // sizing it below MAX_PRELOAD + a working set of clicks. Machine-checked
    // against the actual preload constant, not a hardcoded floor.
    expect(MAX_STASH_ENTRIES).toBeGreaterThanOrEqual(MAX_PRELOAD + 4)
  })

  it('evicts the least-recently-used entry past the cap', () => {
    for (let i = 0; i <= MAX_STASH_ENTRIES; i++) {
      stashTranscript(`s${i}`, rows(`s${i}`))
    }

    expect(stashedTranscriptCount()).toBe(MAX_STASH_ENTRIES)
    expect(readStashedTranscript('s0')).toBeNull() // oldest evicted
    expect(readStashedTranscript(`s${MAX_STASH_ENTRIES}`)).not.toBeNull()
  })

  it('a read refreshes LRU position', () => {
    for (let i = 0; i < MAX_STASH_ENTRIES; i++) {
      stashTranscript(`s${i}`, rows(`s${i}`))
    }

    // Touch s0 so it becomes newest, then push one more entry.
    expect(readStashedTranscript('s0')).not.toBeNull()
    stashTranscript('extra', rows('extra'))

    expect(readStashedTranscript('s0')).not.toBeNull() // survived
    expect(readStashedTranscript('s1')).toBeNull() // evicted instead
  })

  it('re-stashing refreshes LRU position', () => {
    for (let i = 0; i < MAX_STASH_ENTRIES; i++) {
      stashTranscript(`s${i}`, rows(`s${i}`))
    }

    stashTranscript('s0', rows('s0-new'))
    stashTranscript('extra', rows('extra'))

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
