import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { graftRefreshedTailOntoBackfill } from './transcript-backfill'

/**
 * A live reply with no stored id used to trail the newest reply after a tail
 * refresh whose page starts inside an older tool turn, even though the page
 * already held that turn durably. The streaming-level repro lives in
 * use-message-stream/orphaned-turn-tail-merge.test.tsx; these pin the guards.
 */

const call = (id: string) => [{ id, type: 'function', function: { name: 'terminal', arguments: '{}' } }]

const rows = (finalA: string, finalB: string): SessionMessage[] => [
  { id: 1, role: 'user', content: 'Old prompt', timestamp: 1 },
  { id: 2, role: 'assistant', content: 'Old narration', tool_calls: call('t1'), timestamp: 2 },
  { id: 3, role: 'tool', content: 'ok', tool_call_id: 't1', tool_name: 'terminal', timestamp: 3 },
  { id: 4, role: 'assistant', content: '', tool_calls: call('t2'), timestamp: 4 },
  { id: 5, role: 'tool', content: 'ok', tool_call_id: 't2', tool_name: 'terminal', timestamp: 5 },
  { id: 6, role: 'assistant', content: 'Old final', timestamp: 6 },
  { id: 9, role: 'user', content: 'Prompt N', timestamp: 9 },
  { id: 12, role: 'assistant', content: finalA, timestamp: 12 },
  { id: 13, role: 'user', content: 'Prompt N+1', timestamp: 13 },
  { id: 16, role: 'assistant', content: finalB, timestamp: 16 }
]

const bubble = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role: 'assistant',
  parts: [{ type: 'text', text }],
  ...extra
})

/** Window as the stream leaves it: hydrated old turn, turn N's unbound bubbles, turn N+1 bound. */
const windowWith = (live: ChatMessage[], tail: SessionMessage[] = []): ChatMessage[] => {
  const all = rows('unused', 'FINAL B')

  return [
    ...toChatMessages(all.slice(0, 6)),
    ...toChatMessages([all[6]]),
    ...live,
    ...toChatMessages([all[8], all[9]]),
    ...toChatMessages(tail)
  ]
}

// The newest page starts inside the older tool turn (row 4), so the overlap merge decides.
const page = (finalA: string, finalB = 'FINAL B') => toChatMessages(rows(finalA, finalB).slice(3))

const texts = (messages: ChatMessage[]) => messages.map(chatMessageText)

describe('overlap merge vs live copies of a turn the page holds', () => {
  it('drops the unbound narration and answer, keeping the page order', () => {
    const merged = graftRefreshedTailOntoBackfill(
      page('FINAL A'),
      windowWith([bubble('n', 'FINAL A', { interim: true }), bubble('a', 'FINAL A')])
    )

    expect(texts(merged).filter(t => t === 'FINAL A')).toHaveLength(1)
    expect(texts(merged).slice(-4)).toEqual(['Prompt N', 'FINAL A', 'Prompt N+1', 'FINAL B'])
  })

  it('keeps a still-streaming bubble', () => {
    const merged = graftRefreshedTailOntoBackfill(
      page('FINAL A'),
      windowWith([bubble('a', 'FINAL A', { pending: true })])
    )

    expect(texts(merged).filter(t => t === 'FINAL A')).toHaveLength(2)
  })

  it('keeps a local assistant error that repeats the text', () => {
    const merged = graftRefreshedTailOntoBackfill(
      page('FINAL A'),
      windowWith([bubble('e', 'FINAL A', { error: 'Failed locally' })])
    )

    expect(merged.some(m => m.id === 'e')).toBe(true)
  })

  it('keeps an identical reply that belongs to another turn', () => {
    // The page holds a different reply for turn N; the identical text is only
    // inside turn N+1's bracket, not the bubble's own.
    const merged = graftRefreshedTailOntoBackfill(page('Something else', 'Done.'), windowWith([bubble('a', 'Done.')]))

    expect(texts(merged).filter(t => t === 'Done.')).toHaveLength(2)
  })

  it('keeps a bubble the page does not hold', () => {
    const merged = graftRefreshedTailOntoBackfill(page('Stored text'), windowWith([bubble('a', 'Local only text')]))

    expect(merged.some(m => m.id === 'a')).toBe(true)
  })

  it('keeps a bubble with no stored row after it (turn may still be running)', () => {
    const all = rows('FINAL A', 'FINAL B')
    const previous = [...toChatMessages(all.slice(0, 7)), bubble('a', 'FINAL A')]

    const merged = graftRefreshedTailOntoBackfill(toChatMessages(all.slice(3, 8)), previous)

    expect(merged.some(m => m.id === 'a')).toBe(true)
  })

  it('keeps a bubble whose own prompt the page does not carry', () => {
    // The page starts after prompt N: it holds a reply with the same text but
    // not the turn's start, so equal prose alone proves nothing.
    const all = rows('FINAL A', 'FINAL B')

    const merged = graftRefreshedTailOntoBackfill(toChatMessages(all.slice(7)), windowWith([bubble('a', 'FINAL A')]))

    expect(merged.some(m => m.id === 'a')).toBe(true)
  })

  it('keeps a bubble that no prompt opens or follows', () => {
    const all = rows('FINAL A', 'FINAL B')
    const hydrated = toChatMessages(all.slice(0, 6))
    const closing = toChatMessages([all[9]])

    // Stored assistant rows on both sides, no prompt: the bubble is not framed as a turn.
    const merged = graftRefreshedTailOntoBackfill(page('FINAL A'), [...hydrated, bubble('a', 'FINAL A'), ...closing])

    expect(merged.some(m => m.id === 'a')).toBe(true)
  })

  it('keeps a queued prompt and a stopped partial', () => {
    const queued: ChatMessage = { id: 'queued', role: 'user', parts: [{ type: 'text', text: 'next' }] }

    const merged = graftRefreshedTailOntoBackfill(
      page('FINAL A'),
      windowWith([bubble('stopped', 'FINAL A', { interrupted: true })], []).concat(queued)
    )

    expect(merged.some(m => m.id === 'stopped')).toBe(true)
    expect(merged.at(-1)?.id).toBe('queued')
  })
})
