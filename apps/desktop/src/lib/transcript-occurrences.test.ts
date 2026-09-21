import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import {
  mergeInFlightMessages,
  persistInFlightTurnState,
  readInFlightTurnJournal,
  recoverInFlightTurnJournal,
  resetInFlightTurnJournalStateForTests
} from '@/lib/inflight-turn-journal'
import type { SessionMessage } from '@/types/hermes'

const prompt: SessionMessage = { id: 1, role: 'user', content: 'Inspect each phase', timestamp: 1 }

const round = (id: number, text: string): SessionMessage[] => [
  {
    id,
    role: 'assistant',
    content: text,
    timestamp: id,
    tool_calls: [{ id: `tool-${id}`, type: 'function', function: { name: 'read_file', arguments: '{}' } }]
  },
  { id: id + 1, role: 'tool', content: 'result', timestamp: id + 1, tool_call_id: `tool-${id}` }
]

const assistantTexts = (rows: ChatMessage[]) => rows.filter(row => row.role === 'assistant').map(chatMessageText)

beforeEach(() => {
  vi.useFakeTimers()
  localStorage.clear()
  resetInFlightTurnJournalStateForTests()
})
afterEach(() => {
  resetInFlightTurnJournalStateForTests()
  localStorage.clear()
  vi.useRealTimers()
})

it('hydrates authored occurrences rather than deduplicating equal text across tool rounds', () => {
  const first = round(2, 'Status unchanged.')
  const second = round(4, 'Status unchanged.')
  const final: SessionMessage = { id: 6, role: 'assistant', content: 'Status unchanged.', timestamp: 6 }
  const messages = toChatMessages([prompt, ...first, ...second, final])
  const textParts = messages.flatMap(row => row.parts.filter(part => part.type === 'text'))

  expect(textParts.map(part => part.text)).toEqual(['Inspect each phase', ...Array(3).fill('Status unchanged.')])
  expect(messages.at(-1)?.durableComplete).toBe(true)
  expect(toChatMessages([prompt, ...first, ...second]).at(-1)?.durableComplete).toBe(false)
  // The same physical row is not another authored occurrence.
  expect(assistantTexts(toChatMessages([prompt, ...first, first[0]]))).toEqual(['Status unchanged.'])
})

it('recovers only the uncommitted journal suffix until the complete reply becomes durable', () => {
  const stored = toChatMessages([prompt, ...round(2, 'Checking the file.')])

  const tail: ChatMessage = {
    id: 'assistant-stream-runtime', role: 'assistant', pending: true,
    parts: [{ type: 'text', text: 'The unfinished conclusion.' }]
  }

  const state = {
    storedSessionId: 'occurrence-session', busy: true, awaitingResponse: false,
    streamId: tail.id, turnStartedAt: 1000, messages: [...stored, tail]
  }

  persistInFlightTurnState(state)
  vi.advanceTimersByTime(400)

  const first = recoverInFlightTurnJournal(state.storedSessionId, stored)
  expect(first.caughtUp).toBe(false)
  expect(assistantTexts(first.messages)).toEqual(['Checking the file.', 'The unfinished conclusion.'])
  expect(first.streamId).toBeNull()
  // Publishing idle recovered state must not delete its only persistent copy.
  persistInFlightTurnState({ ...state, messages: first.messages, busy: false, streamId: null })
  vi.advanceTimersByTime(400)
  resetInFlightTurnJournalStateForTests()
  const reopened = recoverInFlightTurnJournal(state.storedSessionId, stored)
  expect(assistantTexts(reopened.messages)).toEqual(assistantTexts(first.messages))
  expect(readInFlightTurnJournal(state.storedSessionId)).not.toBeNull()

  const completed = toChatMessages([
    prompt, ...round(2, 'Checking the file.'),
    { id: 4, role: 'assistant', content: 'The unfinished conclusion. Now complete.', timestamp: 4 }
  ])

  expect(recoverInFlightTurnJournal(state.storedSessionId, completed).caughtUp).toBe(true)
  expect(readInFlightTurnJournal(state.storedSessionId)).toBeNull()
})

it.each(['missing prompt', 'repeated prompt', 'no prompt'])('does not clear an unknown journal occurrence by global text: %s', shape => {
  const older = toChatMessages([prompt, { id: 2, role: 'assistant', content: 'Still working.', timestamp: 2 }])

  const user: ChatMessage = {
    id: 'new-user', rowId: 3, role: 'user',
    parts: [{ type: 'text', text: shape === 'repeated prompt' ? 'Inspect each phase' : 'A new question' }]
  }

  const answer: ChatMessage = {
    id: 'assistant-stream-new', role: 'assistant', pending: true,
    parts: [{ type: 'text', text: 'Still working.' }]
  }

  persistInFlightTurnState({
    storedSessionId: 'repeated-session', messages: [...(shape === 'no prompt' ? [] : [user]), answer],
    busy: true, awaitingResponse: false, streamId: answer.id, turnStartedAt: 3000
  })
  vi.advanceTimersByTime(400)
  const recovered = recoverInFlightTurnJournal('repeated-session', older)
  expect(recovered.caughtUp).toBe(false)
  expect(assistantTexts(recovered.messages)).toEqual(['Still working.', 'Still working.'])
  expect(readInFlightTurnJournal('repeated-session')).not.toBeNull()
})

it.each([true, false])('journal replay preserves sealed occurrences (live projection: %s)', hasLiveProjection => {
  const user = toChatMessages([prompt])[0]

  const sealed: ChatMessage = {
    id: 'assistant-stream-sealed', role: 'assistant', interim: true, pending: false,
    parts: [{ type: 'text', text: 'Checking.' }]
  }

  const live: ChatMessage = {
    id: 'assistant-stream-live', role: 'assistant', pending: true,
    parts: [{ type: 'text', text: 'Still working.' }]
  }

  const base = [user, sealed, live]
  const recovered = mergeInFlightMessages(hasLiveProjection ? base : base.slice(0, -1), base, { keepPending: true })
  expect(recovered.messages.map(row => row.id)).toEqual(base.map(row => row.id))
  expect(assistantTexts(recovered.messages)).toEqual(assistantTexts(base))
  expect(recovered.streamId).toBe(live.id)
})
