// Regression tests for the disappearing-final-response defect, which live with
// the `settled-reply-over-shell` admission they pin rather than in the already
// oversized `utils.test.ts`.
//
// A refreshed transcript page can carry the turn's text-less reasoning/tool FOLD
// without the final prose row it would fold (that row is committed only at
// completion, so a mid-turn or just-after-completion read misses it). The fold is
// addressed by the turn's FIRST row while the settled reply bubble is addressed by
// the final prose row, so the two look like different turns. The settled reply
// used to fall through every admission clause and be dropped, leaving the
// text-less shell on screen — a reasoning-only turn ("Thought for…") with no
// answer and no footer.
import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'

import { preserveLocalPendingTurnMessages } from './utils'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

// A live assistant row carrying the structure the gateway's text-only inflight
// snapshot cannot: reasoning and tool calls, with or without any text yet.
const streamingMsg = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({
    id,
    role: 'assistant',
    parts: [
      { type: 'reasoning', text: 'planning' },
      { type: 'tool-call', toolCallId: 'call-1', toolName: 'terminal', result: 'done' },
      ...(text ? [{ type: 'text', text } as ChatMessage['parts'][number]] : [])
    ],
    pending: true,
    ...extra
  }) as ChatMessage

// The page's text-less reasoning/tool fold of a turn: settled, no prose, and
// addressed by the turn's FIRST row.
const textlessShell = (id: string, rowId: number, callId = 'call-1'): ChatMessage =>
  ({
    id,
    role: 'assistant',
    parts: [
      { type: 'reasoning', text: 'planning' },
      { type: 'tool-call', toolCallId: callId, toolName: 'terminal', result: 'done' }
    ],
    pending: false,
    rowId
  }) as ChatMessage

// A settled reply bubble as completion leaves it when no live stream id resolved:
// an `assistant-<ts>` id, and the committed final row bound by the receipt
// (withPersistedIdentity).
const settledReply = (id: string, rowId: number | undefined, text: string, callId = 'call-1'): ChatMessage =>
  ({
    id,
    role: 'assistant',
    parts: [
      { type: 'reasoning', text: 'planning' },
      { type: 'tool-call', toolCallId: callId, toolName: 'terminal', result: 'done' },
      { type: 'text', text }
    ],
    pending: false,
    interim: false,
    ...(rowId === undefined ? {} : { rowId })
  }) as ChatMessage

describe('preserveLocalPendingTurnMessages — settled reply over a text-less shell', () => {
  it('keeps a settled non-stream reply over a text-less assistant shell', () => {
    const previous = [msg('1-user', 'user', 'run the tools'), settledReply('assistant-99', 7420, 'The result is 71.')]

    const next = [msg('1-user', 'user', 'run the tools'), textlessShell('row-1-assistant', 7414)]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    // The shell is replaced by the reply, not kept beside it.
    expect(preserved.map(message => message.id)).toEqual(['1-user', 'assistant-99'])
    expect(preserved.filter(message => chatMessageText(message) === 'The result is 71.')).toHaveLength(1)
  })

  it('keeps a settled non-stream reply over a text-less shell when the page carries no reply row', () => {
    const previous = [
      msg('1-user', 'user', 'run the tools'),
      settledReply('assistant-99', undefined, 'The result is 71.')
    ]

    const next = [msg('1-user', 'user', 'run the tools')]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user',
      'assistant-99'
    ])
  })

  it('keeps the authoritative answer when the page already carries it (exactly one answer)', () => {
    const previous = [msg('1-user', 'user', 'run the tools'), settledReply('assistant-99', 7420, 'The result is 71.')]

    const next = [
      msg('1-user-stored', 'user', 'run the tools', { rowId: 7413 }),
      msg('row-1-assistant', 'assistant', 'The result is 71.', { rowId: 7414 })
    ]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved.filter(message => chatMessageText(message) === 'The result is 71.')).toHaveLength(1)
    expect(preserved.map(message => message.id)).not.toContain('assistant-99')
  })

  it('handles differing fold and final-answer row ids without a duplicate bubble', () => {
    const previous = [msg('1-user', 'user', 'run the tools'), settledReply('assistant-99', 7420, 'The result is 71.')]

    const next = [msg('1-user', 'user', 'run the tools'), textlessShell('row-1-assistant', 7414)]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    // One assistant bubble, addressed by the FINAL prose row (not the fold's first).
    expect(preserved.filter(message => message.role === 'assistant')).toHaveLength(1)
    expect(preserved[1].rowId).toBe(7420)
  })

  it('keeps existing behavior for a stream-id reply over a text-less shell', () => {
    // A SETTLED stream-id row (completion settled it): admitted as pending by id
    // prefix, and now resolved by the shared replacement path.
    const previous = [
      msg('1-user', 'user', 'run the tools'),
      settledReply('assistant-stream-live', 7420, 'The result is 71.')
    ]

    const next = [msg('1-user', 'user', 'run the tools'), textlessShell('row-1-assistant', 7414)]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved.filter(message => chatMessageText(message) === 'The result is 71.')).toHaveLength(1)
    expect(preserved.filter(message => message.role === 'assistant')).toHaveLength(1)
  })

  // Unchanged by this correction (requirement: preserve existing pending /
  // stream-id behavior). A rowId-less PENDING reply resolves through
  // `localPendingSupersedes` / `settledReplyOverEmptyShell`, and both refuse a
  // text-less authoritative row — so the page's shell is kept as-is and the
  // pending row is not appended. Pin that so a future broadening is deliberate.
  it('leaves pending-row resolution unchanged when a text-less shell occupies the ordinal', () => {
    const previous = [msg('1-user', 'user', 'run the tools'), streamingMsg('assistant-stream-live', 'The result is 71.')]

    const next = [msg('1-user', 'user', 'run the tools'), textlessShell('row-1-assistant', 7414)]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved.map(message => message.id)).toEqual(['1-user', 'row-1-assistant'])
  })

  it('does not admit a reasoning-only local row over a text-less shell', () => {
    const reasoningOnly = {
      ...settledReply('assistant-99', 7420, ''),
      parts: [{ type: 'reasoning' as const, text: 'only reasoning' }]
    } as ChatMessage

    const previous = [msg('1-user', 'user', 'run the tools'), reasoningOnly]
    const next = [msg('1-user', 'user', 'run the tools'), textlessShell('row-1-assistant', 7414)]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved).toHaveLength(2)
    expect(preserved.map(message => message.id)).not.toContain('assistant-99')
  })

  it('does not admit an interim local row over a text-less shell', () => {
    const interim = {
      ...settledReply('assistant-interim-1', 7420, 'The result is 71.'),
      interim: true
    } as ChatMessage

    const previous = [msg('1-user', 'user', 'run the tools'), interim]
    const next = [msg('1-user', 'user', 'run the tools'), textlessShell('row-1-assistant', 7414)]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).not.toContain(
      'assistant-interim-1'
    )
  })

  it('does not resurrect a superseded reply when the page carries a newer answer', () => {
    const previous = [
      msg('1-user', 'user', 'run the tools'),
      settledReply('assistant-99', 7420, 'the pre-compaction answer')
    ]

    const next = [
      msg('1-user-stored', 'user', 'run the tools', { rowId: 7413 }),
      msg('row-1-assistant', 'assistant', 'the rewritten answer', { rowId: 7414 })
    ]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved.map(message => chatMessageText(message))).not.toContain('the pre-compaction answer')
    expect(preserved.filter(message => chatMessageText(message) === 'the rewritten answer')).toHaveLength(1)
  })
})
