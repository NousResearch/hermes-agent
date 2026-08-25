import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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
    window.localStorage.setItem('hermes.transcript-tail.v2:sess-bad', '{not json')

    expect(loadTranscriptTail('sess-bad')).toBeNull()
    expect(window.localStorage.getItem('hermes.transcript-tail.v2:sess-bad')).toBeNull()
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

  it('repairs a poisoned persisted tail carrying a duplicate toolCallId (#87857)', () => {
    // A tail written by an older build can hold one message with two tool-call
    // parts sharing an id. This path paints DIRECTLY into the view and the same
    // bytes are re-read every launch — without repair-on-read, an affected
    // install crash-loops forever even after upgrading.
    const tool = (toolCallId: string) => ({
      type: 'tool-call',
      toolCallId,
      toolName: 'terminal',
      args: {},
      argsText: ''
    })

    const poisoned = {
      messages: [{ id: 'assistant-p', role: 'assistant', parts: [tool('call-b'), tool('call-b')] }],
      savedAt: Date.now()
    }

    window.localStorage.setItem('hermes.transcript-tail.v2:sess-poisoned', JSON.stringify(poisoned))

    const loaded = loadTranscriptTail('sess-poisoned')

    expect(loaded).toHaveLength(1)

    const ids = (loaded![0].parts as { type: string; toolCallId?: string }[])
      .filter(part => part.type === 'tool-call')
      .map(part => part.toolCallId)

    expect(ids).toHaveLength(2)
    expect(new Set(ids).size).toBe(2)
    expect(ids[0]).toBe('call-b')
  })

  // ── Profile scoping (#94828) ─────────────────────────────────────────────
  // Stored session ids are only unique WITHIN a profile's state.db, and
  // localStorage survives profile switches in the same window. An unscoped
  // key lets profile A's tail be painted against profile B's backend — the
  // view then retries a session id that does not exist there ("session not
  // found") on every wake. The durable cache must carry the same
  // {connectionId, profile} scope as its in-memory twin (transcript-tail.ts).
  describe('profile scoping (#94828)', () => {
    it('never returns a tail cached under another profile', () => {
      saveTranscriptTail('sess-x', [msg('a')], { profile: 'ai-energy' })

      expect(loadTranscriptTail('sess-x', { profile: 'ai-energy' })).toHaveLength(1)
      expect(loadTranscriptTail('sess-x', { profile: 'default' })).toBeNull()
      expect(loadTranscriptTail('sess-x')).toBeNull()
    })

    it('keeps same-id tails on different connections distinct', () => {
      saveTranscriptTail('sess-y', [msg('local-row')], { connectionId: 'local', profile: 'default' })
      saveTranscriptTail('sess-y', [msg('remote-row')], { connectionId: 'conn:mimir', profile: 'default' })

      expect(loadTranscriptTail('sess-y', { connectionId: 'local', profile: 'default' })?.[0].id).toBe('local-row')
      expect(loadTranscriptTail('sess-y', { connectionId: 'conn:mimir', profile: 'default' })?.[0].id).toBe(
        'remote-row'
      )
    })

    it('accepts a plain-profile string scope like the in-memory twin', () => {
      saveTranscriptTail('sess-s', [msg('s')], 'rwa-africa')

      expect(loadTranscriptTail('sess-s', 'rwa-africa')).toHaveLength(1)
      expect(loadTranscriptTail('sess-s', 'other')).toBeNull()
    })

    it('drops only the scoped entry when a scope is given', () => {
      saveTranscriptTail('sess-d', [msg('p1-row')], { profile: 'p1' })
      saveTranscriptTail('sess-d', [msg('p2-row')], { profile: 'p2' })

      dropTranscriptTail('sess-d', { profile: 'p1' })

      expect(loadTranscriptTail('sess-d', { profile: 'p1' })).toBeNull()
      expect(loadTranscriptTail('sess-d', { profile: 'p2' })).not.toBeNull()
    })

    it('purges pre-scoping v1 entries so a stale tail can never paint again', async () => {
      const stale = {
        messages: [{ id: 'old', parts: [{ text: 'x', type: 'text' }], role: 'assistant' }],
        savedAt: Date.now()
      }

      window.localStorage.setItem('hermes.transcript-tail.v1:sess-legacy', JSON.stringify(stale))
      window.localStorage.setItem('hermes.transcript-tail.v1-index', JSON.stringify(['sess-legacy']))

      // Fresh module instance: the sweep runs once per window, and earlier
      // tests in this file have already touched storage.
      vi.resetModules()

      try {
        const fresh = await import('./transcript-tail-cache')

        expect(fresh.loadTranscriptTail('sess-legacy')).toBeNull()
        expect(fresh.loadTranscriptTail('sess-legacy', { profile: 'ai-energy' })).toBeNull()
        expect(window.localStorage.getItem('hermes.transcript-tail.v1:sess-legacy')).toBeNull()
        expect(window.localStorage.getItem('hermes.transcript-tail.v1-index')).toBeNull()
      } finally {
        vi.resetModules()
      }
    })
  })
})
