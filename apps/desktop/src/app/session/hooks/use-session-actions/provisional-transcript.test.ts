import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

const state = vi.hoisted(() => ({ messages: [] as ChatMessage[] }))

vi.mock('@/store/session', () => ({
  $messages: { get: () => state.messages },
  setMessages: (next: ChatMessage[]) => {
    state.messages = next
  }
}))

import { loadTranscriptTail, saveTranscriptTail } from '@/store/transcript-tail-cache'

import { provisionalTranscriptPaint } from './provisional-transcript'

const SCOPE = { connectionId: 'local', profile: 'default' }

const msg = (id: string): ChatMessage =>
  ({ id, parts: [{ text: id, type: 'text' }], role: 'assistant' }) as never

beforeEach(() => {
  window.localStorage.clear()
  state.messages = []
})

describe('provisionalTranscriptPaint rollback (#120215)', () => {
  it('rolls back an unreconciled paint and evicts the entry so the next wake re-fetches', () => {
    saveTranscriptTail('sess-1', [msg('stale-a'), msg('stale-b')], SCOPE)

    const provisional = provisionalTranscriptPaint('sess-1', () => true)
    provisional.paint(SCOPE)

    // The stale tail painted at ~0ms; resume RPC + REST fallback both failed
    // (detached websocket), so nothing authoritative ever replaced it.
    expect(state.messages.map(m => m.id)).toEqual(['stale-a', 'stale-b'])

    provisional.rollback(SCOPE)

    expect(state.messages).toEqual([])
    expect(loadTranscriptTail('sess-1', SCOPE)).toBeNull()
  })

  it('keeps the entry when an authoritative transcript already replaced the paint', () => {
    saveTranscriptTail('sess-1', [msg('stale-a')], SCOPE)

    const provisional = provisionalTranscriptPaint('sess-1', () => true)
    provisional.paint(SCOPE)

    // The success path painted REST truth and refreshed the entry.
    const authoritative = [msg('fresh-a'), msg('fresh-b')]
    state.messages = authoritative
    saveTranscriptTail('sess-1', authoritative, SCOPE)

    provisional.rollback(SCOPE)

    expect(state.messages).toBe(authoritative)
    expect(loadTranscriptTail('sess-1', SCOPE)?.map(m => m.id)).toEqual(['fresh-a', 'fresh-b'])
  })

  it('is a no-op without a paint and leaves other sessions intact', () => {
    saveTranscriptTail('sess-other', [msg('kept')], SCOPE)

    const provisional = provisionalTranscriptPaint('sess-1', () => true)
    provisional.rollback(SCOPE)

    expect(state.messages).toEqual([])
    expect(loadTranscriptTail('sess-other', SCOPE)?.map(m => m.id)).toEqual(['kept'])
  })
})
