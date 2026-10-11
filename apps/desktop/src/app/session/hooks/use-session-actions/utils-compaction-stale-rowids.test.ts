import { describe, expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart, chatMessageText } from '@/lib/chat-messages'

import { preserveLocalPendingTurnMessages } from './utils'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

const tool = (toolCallId: string) =>
  ({ type: 'tool-call', toolCallId, toolName: 'terminal', result: 'ok' }) as ChatMessagePart

const sealed = (id: string, parts: ChatMessagePart[], extra: Partial<ChatMessage> = {}) =>
  ({ id, role: 'assistant', parts, pending: false, interim: true, ...extra }) as ChatMessage

const lunaTurn = (prefix: string, callPrefix: string) => [
  sealed(`assistant-stream-${prefix}-ack`, [{ type: 'text', text: `${prefix}: on it, reading the logs.` }]),
  sealed(`assistant-stream-${prefix}-progress-1`, [
    tool(`${callPrefix}-1`),
    { type: 'text', text: `${prefix}: logs clean.` }
  ]),
  sealed(`assistant-stream-${prefix}-progress-2`, [
    tool(`${callPrefix}-2`),
    { type: 'text', text: `${prefix}: config fixed.` }
  ]),
  sealed(
    `assistant-stream-${prefix}-final`,
    [tool(`${callPrefix}-3`), { type: 'text', text: `${prefix}: all done.` }],
    { interim: false }
  )
]

const lunaFold = (id: string, prefix: string, callPrefix: string) =>
  ({
    id,
    role: 'assistant',
    parts: [
      ...[`${prefix}: on it, reading the logs.`, `${prefix}: logs clean.`, `${prefix}: config fixed.`].flatMap(
        (text, at) => [{ type: 'text', text } as ChatMessagePart, tool(`${callPrefix}-${at + 1}`)]
      ),
      { type: 'text', text: `${prefix}: all done.` }
    ]
  }) as ChatMessage

/**
 * Compaction rewrites a committed turn under NEW row ids (#117867), which marks
 * the turn's committed rows as a transcript-identity CONFLICT for the local
 * bubble that still carries its pre-compaction rowId. The still-pending local
 * bubble then survives every same-turn guard and lands in `preserved`, so the
 * answer renders twice — the tail duplication reported from long sessions.
 */
describe('preserveLocalPendingTurnMessages / compaction re-inserted row ids', () => {
  it('retires a still-pending bubble whose compacted turn carries new row ids', () => {
    const user = msg('1-user', 'user', 'run the tools', { rowId: 10 })

    const localTurn = [
      ...lunaTurn('a', 'call-a')
        .slice(0, 3)
        .map((row, at) => ({ ...row, rowId: 11 + at })),
      { ...lunaTurn('a', 'call-a')[3], rowId: 14, pending: true, interim: false } as ChatMessage
    ]

    const next = [
      msg('200-user', 'user', 'run the tools', { rowId: 210 }),
      { ...lunaFold('2-assistant', 'a', 'call-a'), rowId: 211 } as ChatMessage
    ]

    const out = preserveLocalPendingTurnMessages(next, [user, ...localTurn])

    expect(out.map(message => message.id)).toEqual(['200-user', '2-assistant'])
    expect(out.filter(message => message.role === 'assistant')).toHaveLength(1)
    expect(chatMessageText(out[1])).toContain('a: all done.')
  })

  /**
   * Measured shape (CDP + the React-hook store) after a compaction re-keyed the
   * whole page: the turn's live bubbles were still in the store next to the
   * re-inserted committed copy, so the transcript rendered the turn twice. The
   * sealed segment bubbles are NOT prefixes of the folded row, so the
   * prefix-only arms (`isStrictAnswerTextExtension`) missed them.
   */
  it('retires text-only segment bubbles the re-keyed committed row already spells out', () => {
    const local = [
      { id: 'assistant-stream-x-1', role: 'assistant', parts: [{ type: 'text', text: 'seg one' }], pending: false, interim: true },
      { id: 'assistant-stream-x-2', role: 'assistant', parts: [{ type: 'text', text: 'seg two' }], pending: false, interim: true },
      { id: 'assistant-stream-x-3', role: 'assistant', parts: [{ type: 'text', text: 'seg three' }], pending: false, interim: false }
    ] as ChatMessage[]

    const next = [
      msg('200-user', 'user', 'run the tools', { rowId: 210 }),
      {
        id: '2-assistant',
        role: 'assistant',
        rowId: 211,
        parts: [{ type: 'text', text: 'seg one\n\nseg two\n\nseg three' }]
      } as ChatMessage
    ]

    const out = preserveLocalPendingTurnMessages(next, [
      msg('10-user', 'user', 'run the tools', { rowId: 10 }),
      ...local
    ])

    expect(out.map(message => message.id)).toEqual(['200-user', '2-assistant'])
    expect(out.filter(message => message.role === 'assistant')).toHaveLength(1)
  })

  /**
   * Measured leak (CDP + the React-hook store): the refresh's base carries the
   * stale live bubble forward through `graftRefreshedTailOntoBackfill`, so it
   * rendered at index 18 right next to the committed fold at 17. Refusing to
   * preserve it was not enough — it had to leave the base.
   */
  it('drops a stale live bubble the refresh carried forward in its own base', () => {
    const fold = {
      id: '107-assistant',
      role: 'assistant',
      rowId: 3907,
      parts: [{ type: 'text', text: 'seg one\n\nseg two' }]
    } as ChatMessage

    const stale = {
      id: 'assistant-stream-x-1',
      role: 'assistant',
      parts: [{ type: 'text', text: 'seg two' }],
      pending: false,
      interim: true
    } as ChatMessage

    const listing = [msg('106-user', 'user', 'hello world', { rowId: 3906 }), fold, stale]

    const out = preserveLocalPendingTurnMessages(listing, listing)

    expect(out.map(message => message.id)).toEqual(['106-user', '107-assistant'])
  })

  it('keeps a current live bubble when the same prompt was sent twice', () => {
    const earlier = {
      id: '107-assistant',
      role: 'assistant',
      rowId: 3907,
      parts: [{ type: 'text', text: 'seg two' }]
    } as ChatMessage

    const currentLive = {
      id: 'assistant-stream-y-1',
      role: 'assistant',
      parts: [{ type: 'text', text: 'seg two' }],
      pending: false,
      interim: true
    } as ChatMessage

    const listing = [
      msg('106-user', 'user', 'same words', { rowId: 3906 }),
      earlier,
      msg('108-user', 'user', 'same words', { rowId: 3908 }),
      currentLive
    ]

    const out = preserveLocalPendingTurnMessages(listing, listing)

    expect(out.map(message => message.id)).toContain('assistant-stream-y-1')
  })
})
