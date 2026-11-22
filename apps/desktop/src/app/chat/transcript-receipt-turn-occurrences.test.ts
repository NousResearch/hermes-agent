import { describe, expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart } from '@/lib/chat-messages'

import { reconcileLiveTurnsWithPage } from './transcript-receipt-turns'

const assistant = (id: string, parts: ChatMessagePart[]): ChatMessage => ({ id, role: 'assistant', parts })

const call = (result?: string): ChatMessagePart => ({
  type: 'tool-call',
  toolCallId: 'call_0',
  toolName: 'terminal',
  ...(result === undefined ? {} : { result })
})

const finishedResults = ({ page, window }: ReturnType<typeof reconcileLiveTurnsWithPage>) =>
  [...window, ...page].flatMap(message =>
    message.parts.flatMap(part => (part.type === 'tool-call' && typeof part.result === 'string' ? [part.result] : []))
  )

describe('a refreshed page that begins inside a streamed turn', () => {
  it('keeps the earlier result when the page only carries a later call with the same ID', () => {
    const turn = 'machine-stream'

    const live = [
      { ...assistant('first-live', [call('first result')]), streamTurnId: turn },
      { ...assistant('second-live', [call('second result')]), streamTurnId: turn }
    ]

    const stored = [assistant('second-stored', [{ ...call('second result'), storedTurnId: turn, sourceCallRowId: 12 }])]

    expect(finishedResults(reconcileLiveTurnsWithPage(live, stored)).sort()).toEqual(['first result', 'second result'])
  })

  it('keeps the only live result when an attachment missed an earlier call with the same ID', () => {
    const turn = 'machine-stream'

    const live = [{ ...assistant('second-live', [call('second result')]), streamTurnId: turn }]

    const stored = [
      assistant('first-stored', [{ ...call('first result'), storedTurnId: turn, sourceCallRowId: 10 }]),
      assistant('second-stored', [{ ...call(), storedTurnId: turn, sourceCallRowId: 12 }])
    ]

    expect(finishedResults(reconcileLiveTurnsWithPage(live, stored)).sort()).toEqual(['first result', 'second result'])
  })
})
