import { describe, expect, it } from 'vitest'

import { restorePendingClarifyToolCall, settlePendingClarifyToolCall, toChatMessages } from '@/lib/chat-messages'
import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

// #133770: a clarify re-ARM (the request is still open — live clarify.request,
// resume/activate snapshot replay) can land on a transcript whose provider row
// already carries a stored result (the result row was persisted before the
// request left `open_requests`, or a replay raced the row). The re-arm used to
// look at open parts only, so the settled row was invisible and the re-arm
// appended a second, interactive card next to the settled one — the duplicate
// that survives answering (only the provider row settles via tool.complete)
// and is rebuilt by every session activation. Correlation must reopen the
// settled row in place instead.

const CALL_ID = 'call_01_Ps2vwFrrCnE01efLi1So8337'
const REQUEST_ID = 'req-133770'

const QUESTIONS = [
  { choices: ['A', 'B'], qid: 'q0', question: 'Drink?' },
  { choices: ['X', 'Y'], qid: 'q1', question: 'Productive when?' }
]

const userRow: SessionMessage = { content: 'plan my homelab', id: 29452, role: 'user', timestamp: 900 }

const callRow: SessionMessage = {
  content: 'A few things first.',
  id: 29453,
  role: 'assistant',
  timestamp: 1_000,
  tool_calls: [
    {
      function: {
        arguments: JSON.stringify({ questions: QUESTIONS.map(q => ({ question: q.question })) }),
        name: 'clarify'
      },
      id: CALL_ID,
      type: 'function'
    }
  ]
}

const resultRow: SessionMessage = {
  content: JSON.stringify({ outcome: 'submitted', responses: [] }),
  id: 29754,
  role: 'tool',
  timestamp: 2_000,
  tool_call_id: CALL_ID,
  tool_name: 'clarify'
}

const otherQuestions = [{ choices: ['P', 'Q'], qid: 'q0', question: 'Unrelated?' }]

const otherCallRow: SessionMessage = {
  content: '',
  id: 28_000,
  role: 'assistant',
  timestamp: 500,
  tool_calls: [
    {
      function: {
        arguments: JSON.stringify({ questions: otherQuestions.map(q => ({ question: q.question })) }),
        name: 'clarify'
      },
      id: 'call_old',
      type: 'function'
    }
  ]
}

const otherResultRow: SessionMessage = {
  content: JSON.stringify({ outcome: 'submitted', responses: [] }),
  id: 28_001,
  role: 'tool',
  timestamp: 600,
  tool_call_id: 'call_old',
  tool_name: 'clarify'
}

const openRequest = {
  questions: QUESTIONS.map(q => ({ choices: q.choices, multiSelect: false, qid: q.qid, question: q.question })),
  receivedAt: 2_100,
  requestId: REQUEST_ID,
  sessionId: 'sess-1'
}

const openRequestPayload = (): Parameters<typeof restorePendingClarifyToolCall>[1] => ({
  args: {
    questions: QUESTIONS.map(question => ({ choices: question.choices, question: question.question }))
  },
  tool_id: REQUEST_ID
})

function clarifyParts(messages: ChatMessage[]): Extract<ChatMessagePart, { type: 'tool-call' }>[] {
  return messages
    .flatMap(m => m.parts)
    .filter(
      (p): p is Extract<ChatMessagePart, { type: 'tool-call' }> => p.type === 'tool-call' && p.toolName === 'clarify'
    )
}

describe('clarify re-arm over a settled provider row (#133770)', () => {
  it('reopens the settled row in place instead of appending a second card', () => {
    const settled = toChatMessages([userRow, callRow, resultRow])
    expect(clarifyParts(settled)).toHaveLength(1)
    expect(clarifyParts(settled)[0].result).toBeDefined()

    const projection = restorePendingClarifyToolCall(settled, openRequestPayload())

    const parts = clarifyParts(projection.messages)
    expect(parts).toHaveLength(1)
    expect(parts[0].toolCallId).toBe(CALL_ID)
    expect(parts[0].result).toBeUndefined()
    expect(parts[0].completedAt).toBeUndefined()
    // In place: the row count and the clarify row's message position are unchanged.
    expect(projection.messages).toHaveLength(settled.length)
  })

  it('keeps the reopened card answerable and settle-able in one row', () => {
    const settled = toChatMessages([userRow, callRow, resultRow])
    const reopened = restorePendingClarifyToolCall(settled, openRequestPayload())

    // request.cancel for the still-open wait settles the same row — no residue.
    const projection = settlePendingClarifyToolCall(reopened.messages, openRequestPayload(), false)

    const parts = clarifyParts(projection.messages)
    expect(parts).toHaveLength(1)
    expect(parts[0].toolCallId).toBe(CALL_ID)
    expect(parts[0].result).toBeDefined()
    expect(parts[0].completedAt).toBeDefined()
  })

  it('does not overwrite a settled answer when the wait ends without a reopen', () => {
    const settled = toChatMessages([userRow, callRow, resultRow])
    const before = clarifyParts(settled)[0].result

    // A settle arriving on an already-settled row (late cancel for a request
    // that already returned) must leave the stored answer untouched.
    const projection = settlePendingClarifyToolCall(settled, openRequestPayload(), false)

    expect(clarifyParts(projection.messages)).toHaveLength(1)
    expect(clarifyParts(projection.messages)[0].result).toBe(before)
  })

  it('does not adopt an unrelated settled clarify as the re-arm target', () => {
    // History settles its own clarifies; a re-arm whose questions match none of
    // them still takes the append fallback (the tool.start row was missed) and
    // must not reopen an old, answered card.
    const history = toChatMessages([userRow, otherCallRow, otherResultRow, callRow, resultRow])
    expect(clarifyParts(history)).toHaveLength(2)
    expect(clarifyParts(history).every(part => part.result !== undefined)).toBe(true)

    const unrelatedPayload = {
      args: { questions: [{ question: 'Brand new question?' }] },
      tool_id: 'req-new'
    }

    const projection = restorePendingClarifyToolCall(history, unrelatedPayload)

    const parts = clarifyParts(projection.messages)
    expect(parts).toHaveLength(3)
    // Both historical rows stay settled; the new request is its own open row.
    expect(parts.filter(part => part.result === undefined)).toHaveLength(1)
    expect(parts.find(part => part.result === undefined)?.toolCallId).toBe('req-new')
  })

  it('still merges a re-arm onto an open (unanswered) provider row in place', () => {
    const open = toChatMessages([userRow, callRow])
    expect(clarifyParts(open)[0].result).toBeUndefined()

    const projection = restorePendingClarifyToolCall(open, openRequestPayload())

    const parts = clarifyParts(projection.messages)
    expect(parts).toHaveLength(1)
    expect(parts[0].toolCallId).toBe(CALL_ID)
    expect(parts[0].result).toBeUndefined()
  })
})
