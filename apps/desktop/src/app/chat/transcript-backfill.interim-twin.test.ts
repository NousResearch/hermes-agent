import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { graftRefreshedTailOntoBackfill } from './transcript-backfill'

const row = (rowId: number, role: ChatMessage['role'], text: string): ChatMessage => ({
  id: `row-${rowId}`,
  role,
  parts: [{ type: 'text', text }],
  rowId
})

const interimBubble = (text: string): ChatMessage => ({
  id: 'live-interim',
  role: 'assistant',
  parts: [{ type: 'text', text }],
  interim: true
})

const texts = (messages: ChatMessage[]): string[] =>
  messages.map(m => (m.parts[0] as { text?: string }).text ?? '')

// #134415: the completion event never landed on the client (dropped socket, or
// a replayed `message.interim` with already_streamed=false after a refresh had
// already hydrated the turn), so a sealed interim bubble lingers unstored in
// the window while the refreshed page carries the same turn as a durable row.
// The overlap merge must fold the bubble instead of trailing it behind the
// hydrated row, which painted the turn twice.
describe('graftRefreshedTailOntoBackfill vs a lingering interim bubble', () => {
  it('folds the interim bubble when the page hydrates its settled turn', () => {
    const previous = [
      row(8, 'user', 'request access to a service'),
      interimBubble('read-only access check paragraph'),
      row(15, 'user', 'next prompt after the stalled turn')
    ]

    const refreshedTail = [
      row(11, 'assistant', 'read-only access check paragraph'),
      row(15, 'user', 'next prompt after the stalled turn')
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedTail, previous)

    expect(texts(merged).filter(t => t === 'read-only access check paragraph')).toHaveLength(1)
    expect(texts(merged)).toEqual([
      'request access to a service',
      'read-only access check paragraph',
      'next prompt after the stalled turn'
    ])
  })

  it('keeps a bubble whose bracket carries no matching durable text', () => {
    const previous = [
      row(8, 'user', 'run the check'),
      interimBubble('the same conclusion'),
      row(15, 'user', 'run it again'),
    ]

    const refreshedTail = [
      row(11, 'assistant', 'a different reply inside the bracket'),
      row(15, 'user', 'run it again'),
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedTail, previous)

    // No assistant row inside the bubble's bracket (8, 15) repeats its text:
    // the turn has no hydrated twin yet, so the live copy survives the merge.
    expect(texts(merged).filter(t => t === 'the same conclusion')).toHaveLength(1)
  })

  it('replaces a still-live bubble through the anchor path without doubling it', () => {
    const previous = [row(8, 'user', 'request'), interimBubble('streaming commentary')]

    const refreshedTail = [
      row(8, 'user', 'request'),
      row(11, 'assistant', 'streaming commentary and the parts that followed'),
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedTail, previous)

    // The page anchors at the window's first row and takes over the turn:
    // one hydrated copy, never the bubble next to it.
    expect(texts(merged)).toEqual(['request', 'streaming commentary and the parts that followed'])
  })
})
