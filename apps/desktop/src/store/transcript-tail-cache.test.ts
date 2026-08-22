import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import {
  clearTranscriptTails,
  dropTranscriptTail,
  loadTranscriptTail,
  saveTranscriptTail
} from './transcript-tail-cache'

const msg = (id: string, chars = 10): ChatMessage =>
  ({ id, parts: [{ text: 'x'.repeat(chars), type: 'text' }], role: 'assistant' }) as never

beforeEach(() => {
  window.localStorage.clear()
})

afterEach(() => {
  window.localStorage.clear()
})

describe('transcript tail cache', () => {
  it('round-trips a session tail keyed by stored id', () => {
    saveTranscriptTail('sess-a', [msg('1'), msg('2')])

    const loaded = loadTranscriptTail('sess-a')

    expect(loaded).toHaveLength(2)
    expect(loaded?.[0].id).toBe('1')
    expect(loadTranscriptTail('sess-other')).toBeNull()
  })

  it('persists only the bounded tail of a long transcript', () => {
    const long = Array.from({ length: 200 }, (_, index) => msg(`m${index}`))
    saveTranscriptTail('sess-long', long)

    const loaded = loadTranscriptTail('sess-long')

    expect(loaded).toHaveLength(40)
    expect(loaded?.[0].id).toBe('m160')
    expect(loaded?.[39].id).toBe('m199')
  })

  it('falls back to a shorter tail rather than caching an oversized entry', () => {
    // ~40 x 30KB ≈ 1.2MB > 256KB cap; 8 x 30KB ≈ 240KB fits.
    const heavy = Array.from({ length: 40 }, (_, index) => msg(`h${index}`, 30_000))
    saveTranscriptTail('sess-heavy', heavy)

    const loaded = loadTranscriptTail('sess-heavy')

    expect(loaded).toHaveLength(8)
    expect(loaded?.[7].id).toBe('h39')
  })

  it('ignores empty saves and blank ids', () => {
    saveTranscriptTail('', [msg('1')])
    saveTranscriptTail('sess-empty', [])

    expect(loadTranscriptTail('')).toBeNull()
    expect(loadTranscriptTail('sess-empty')).toBeNull()
  })

  it('self-evicts a corrupt entry instead of returning garbage', () => {
    window.localStorage.setItem('hermes.transcript-tail.v1:sess-bad', '{not json')

    expect(loadTranscriptTail('sess-bad')).toBeNull()
    expect(window.localStorage.getItem('hermes.transcript-tail.v1:sess-bad')).toBeNull()
  })

  it('drops a deleted session and wipes everything on a gateway re-home', () => {
    saveTranscriptTail('sess-1', [msg('a')])
    saveTranscriptTail('sess-2', [msg('b')])

    dropTranscriptTail('sess-1')
    expect(loadTranscriptTail('sess-1')).toBeNull()
    expect(loadTranscriptTail('sess-2')).not.toBeNull()

    clearTranscriptTails()
    expect(loadTranscriptTail('sess-2')).toBeNull()
  })

  it('LRU-evicts the oldest sessions past the entry cap', () => {
    for (let index = 0; index < 55; index += 1) {
      saveTranscriptTail(`sess-${index}`, [msg(`m${index}`)])
    }

    // 50-entry cap: the first five saved are evicted, the newest survive.
    expect(loadTranscriptTail('sess-0')).toBeNull()
    expect(loadTranscriptTail('sess-4')).toBeNull()
    expect(loadTranscriptTail('sess-5')).not.toBeNull()
    expect(loadTranscriptTail('sess-54')).not.toBeNull()
  })
})
