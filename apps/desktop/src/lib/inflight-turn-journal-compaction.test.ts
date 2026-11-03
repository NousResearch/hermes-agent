import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import {
  persistInFlightTurnState,
  readInFlightTurnJournal,
  recoverInFlightTurnJournal,
  resetInFlightTurnJournalStateForTests
} from '@/lib/inflight-turn-journal'
import type { SessionMessage } from '@/types/hermes'

const prompt = 'Check this run'
const commentary = 'Checking the run.'

const rows = (userId: number, assistantId: number): SessionMessage[] => [
  { id: userId, role: 'user', content: prompt, timestamp: 100 },
  {
    id: assistantId,
    role: 'assistant',
    content: commentary,
    timestamp: 101,
    tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
  }
]

const before = () => toChatMessages(rows(10, 11))

const after = () =>
  toChatMessages([{ id: 19, role: 'assistant', content: 'Earlier response.', timestamp: 90 }, ...rows(20, 21)])

const persist = (messages: ChatMessage[]) => {
  persistInFlightTurnState({
    storedSessionId: 'rewritten-session',
    messages,
    busy: true,
    awaitingResponse: false,
    streamId: messages.at(-1)!.id,
    turnStartedAt: 100000
  })
  vi.advanceTimersByTime(400)
  expect(readInFlightTurnJournal('rewritten-session')).not.toBeNull()
  resetInFlightTurnJournalStateForTests()
}

beforeEach(() => {
  vi.useFakeTimers()
  window.localStorage.clear()
  resetInFlightTurnJournalStateForTests()
})
afterEach(() => {
  resetInFlightTurnJournalStateForTests()
  window.localStorage.clear()
  vi.useRealTimers()
})

it.each([false, true])(
  'does not recover a rewritten tool turn as a second user/reply (keepPending=%s)',
  keepPending => {
    persist(before())
    const restored = after()
    const result = recoverInFlightTurnJournal('rewritten-session', restored, { keepPending })
    expect(result.messages).toBe(restored)
    expect(result.messages.filter(message => message.role === 'user').map(chatMessageText)).toEqual([prompt])
  }
)

it('keeps the journal-only suffix when its saved prefix was reinserted', () => {
  const cached = before()
  cached.at(-1)!.parts.push({ type: 'text', text: 'Still running.' })
  persist(cached)
  const result = recoverInFlightTurnJournal('rewritten-session', after())
  expect(result.messages.filter(message => message.role === 'user').map(chatMessageText)).toEqual([prompt])
  expect(result.messages.filter(message => message.role === 'assistant').map(chatMessageText)).toEqual([
    'Earlier response.',
    commentary,
    'Still running.'
  ])
  expect(result.messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call')).toHaveLength(1)
})

it.each(['timestamp', 'tool identity', 'user boundary'])(
  'does not fold an equal prompt without same-turn evidence: %s',
  difference => {
    persist(before())
    const raw = rows(20, 21)

    if (difference === 'timestamp') {
      raw[0].timestamp = 200
    }

    if (difference === 'tool identity') {
      raw[1].tool_calls = [{ id: 'other-call', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
    }

    if (difference === 'user boundary') {
      raw.splice(1, 0, { id: 22, role: 'user', content: 'A different turn', timestamp: 102 })
    }

    const result = recoverInFlightTurnJournal(
      'rewritten-session',
      toChatMessages([{ id: 19, role: 'assistant', content: 'Earlier response.', timestamp: 90 }, ...raw])
    )

    expect(
      result.messages.filter(message => message.role === 'user' && chatMessageText(message) === prompt)
    ).toHaveLength(2)
  }
)
