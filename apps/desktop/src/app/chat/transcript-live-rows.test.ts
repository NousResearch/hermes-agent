import { describe, expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart } from '@/lib/chat-messages'

import { reconcileLiveRowsWithPage } from './transcript-live-rows'

const assistant = (id: string, parts: ChatMessagePart[], extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role: 'assistant',
  parts,
  ...extra
})

const call = (row: number, result?: string, uid?: string): ChatMessagePart => ({
  type: 'tool-call',
  toolCallId: 'call_0',
  toolName: 'terminal',
  sourceCallRowId: row,
  ...(uid ? { sourceMessageUid: uid } : {}),
  ...(result === undefined ? {} : { result })
})

const said = (row: number, text: string, uid?: string): ChatMessagePart => ({
  type: 'text',
  text,
  sourceRowId: row,
  ...(uid ? { sourceMessageUid: uid } : {})
})

const results = (list: ChatMessage[]) =>
  list.flatMap(message =>
    message.parts.flatMap(part => (part.type === 'tool-call' && typeof part.result === 'string' ? [part.result] : []))
  )

describe('live rows paired with a refreshed page by the stored row they name', () => {
  it('keeps the earlier call when the page only carries a later call with the same provider id', () => {
    const live = [
      assistant('first-live', [call(10, 'first result')]),
      assistant('second-live', [call(12, 'second result')])
    ]

    const page = [assistant('second-stored', [call(12, 'second result')])]

    const { page: nextPage, window } = reconcileLiveRowsWithPage(live, page)

    expect(window.map(message => message.id)).toEqual(['first-live'])
    expect([...results(window), ...results(nextPage)].sort()).toEqual(['first result', 'second result'])
  })

  it('fills a stored call slot from the only live result, once', () => {
    const live = [assistant('live', [call(12, 'second result')])]
    const page = [assistant('stored', [call(10, 'first result'), call(12)])]

    const { page: nextPage, window } = reconcileLiveRowsWithPage(live, page)

    expect(window).toEqual([])
    expect(results(nextPage)).toEqual(['first result', 'second result'])
  })

  it('pairs by uid when a compaction re-issued the row id, and never by a stale id', () => {
    const live = [assistant('live', [said(4, 'narration', 'uid-a'), call(4, 'ok', 'uid-a')])]
    const moved = [assistant('stored', [said(8, 'narration', 'uid-a'), call(8, 'ok', 'uid-a')])]
    // Another row now holds id 4 only in this contrived page; with a uid on both sides the id proves nothing.
    const reused = [assistant('stored', [said(4, 'narration', 'uid-b')])]

    expect(reconcileLiveRowsWithPage(live, moved).window).toEqual([])
    expect(reconcileLiveRowsWithPage(live, reused).window).toEqual(live)
  })

  it('pairs by row id with a page from a backend that stores no uids', () => {
    const live = [assistant('live', [said(4, 'narration', 'uid-a')])]

    expect(reconcileLiveRowsWithPage(live, [assistant('stored', [said(4, 'narration')])]).window).toEqual([])
  })

  it('keeps what the page does not store: the streaming text, an error card, an unaddressed part', () => {
    const page = [assistant('stored', [said(4, 'narration'), call(4, 'ok')])]

    const live = [
      assistant('streaming', [call(4, 'ok'), { type: 'text', text: 'still typing' }], { pending: true }),
      assistant('failed', [said(4, 'narration')], { error: 'provider exploded' }),
      assistant('older-host', [{ type: 'text', text: 'narration' }]),
      assistant('thoughts', [{ type: 'reasoning', text: 'why' }, said(4, 'narration')])
    ]

    const { window } = reconcileLiveRowsWithPage(live, page)

    expect(window.map(message => [message.id, message.parts.map(part => part.type)])).toEqual([
      ['streaming', ['text']],
      ['failed', []],
      ['older-host', ['text']]
    ])
  })
})
