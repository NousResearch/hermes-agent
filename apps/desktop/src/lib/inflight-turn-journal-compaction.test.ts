import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { reconcileAuthoritativeChatMessages } from '@/app/session/hooks/use-session-actions/resume-reconcile'
import { preserveEquivalentTranscript } from '@/app/session/hooks/use-session-actions/utils'
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
const userUid = 'c'.repeat(32)

const rows = (userId: number, assistantId: number, messageUid?: string): SessionMessage[] => [
  { id: userId, message_uid: messageUid, role: 'user', content: prompt, timestamp: 100 },
  {
    id: assistantId,
    role: 'assistant',
    content: commentary,
    timestamp: 101,
    tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
  }
]

const before = () => toChatMessages(rows(10, 11, userUid))

const after = () =>
  toChatMessages([{ id: 19, role: 'assistant', content: 'Earlier response.', timestamp: 90 }, ...rows(20, 21, userUid)])

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

it('publishes newly hydrated stable identity before journaling an unchanged prompt', () => {
  // Arrange
  const legacy = toChatMessages(rows(10, 11))
  const hydrated = before()

  // Act
  const published = preserveEquivalentTranscript(legacy, hydrated)
  persist(published)

  // Assert
  expect(published).toBe(hydrated)
  expect(readInFlightTurnJournal('rewritten-session')?.messages[0]).toMatchObject({ rowId: 10, messageUid: userUid })
})

it('retains known stable identity when a same-row history projection omits it', () => {
  // Arrange
  const previous = before()
  const runtime = toChatMessages(rows(10, 11))

  // Act
  persist(reconcileAuthoritativeChatMessages(runtime, previous))

  // Assert
  expect(readInFlightTurnJournal('rewritten-session')?.messages[0]).toMatchObject({ rowId: 10, messageUid: userUid })
})

it.each([
  { name: 'different UIDs', savedUid: 'a'.repeat(32), restoredUid: 'b'.repeat(32) },
  { name: 'legacy rows without UIDs', savedUid: undefined, restoredUid: undefined },
  { name: 'saved UID only', savedUid: 'a'.repeat(32), restoredUid: undefined },
  { name: 'restored UID only', savedUid: undefined, restoredUid: 'b'.repeat(32) }
])('retains journal-only output across conflicting physical rows ($name)', ({ savedUid, restoredUid }) => {
  // Arrange
  const savedRows = rows(10, 11)
  const restoredRows = rows(20, 21)

  const saved = toChatMessages([
    { ...savedRows[0], message_uid: savedUid },
    { ...savedRows[1], content: 'Unique journal output' }
  ])

  const restored = toChatMessages([
    { id: 19, role: 'assistant', content: 'Earlier response.', timestamp: 90 },
    { ...restoredRows[0], message_uid: restoredUid },
    { ...restoredRows[1], content: 'Other output' },
    { id: 22, role: 'assistant', content: 'Other final', timestamp: 102 }
  ])

  persist(saved)

  // Act
  const result = recoverInFlightTurnJournal('rewritten-session', restored)

  // Assert
  expect(result.caughtUp).toBe(false)
  expect(result.messages.filter(message => message.role === 'user').map(chatMessageText)).toEqual([prompt, prompt])
  expect(result.messages.map(chatMessageText)).toContain('Unique journal output')
  expect(readInFlightTurnJournal('rewritten-session')).not.toBeNull()
})

it.each([false, true])(
  'does not recover a rewritten tool turn as a second user/reply (keepPending=%s)',
  keepPending => {
    // Arrange
    persist(before())
    const restored = after()

    // Act
    const result = recoverInFlightTurnJournal('rewritten-session', restored, { keepPending })

    // Assert
    expect(result.messages).toBe(restored)
    expect(result.caughtUp).toBe(false)
    expect(result.messages.filter(message => message.role === 'user').map(chatMessageText)).toEqual([prompt])
    expect(readInFlightTurnJournal('rewritten-session')).not.toBeNull()
  }
)

it.each([
  { name: 'unknown identity on different rows', restoredId: 20, savedUid: undefined, restoredUid: undefined },
  { name: 'conflicting UIDs on the same row', restoredId: 10, savedUid: 'a'.repeat(32), restoredUid: 'b'.repeat(32) }
])('preserves journal output despite colliding renderer ids ($name)', ({ restoredId, savedUid, restoredUid }) => {
  // Arrange
  const savedRows = rows(10, 11, savedUid)
  const restoredRows = rows(restoredId, restoredId + 1, restoredUid)

  const saved = toChatMessages([
    savedRows[0],
    {
      ...savedRows[1],
      message_uid: savedUid === undefined ? undefined : 'd'.repeat(32),
      content: 'Unique journal output'
    }
  ])

  const restored = toChatMessages([
    restoredRows[0],
    {
      ...restoredRows[1],
      message_uid: restoredUid === undefined ? undefined : 'e'.repeat(32),
      content: 'Other output'
    },
    { id: restoredId + 2, role: 'assistant', content: 'Other final', timestamp: 102 }
  ])

  expect(saved.map(message => message.id)).toEqual(restored.map(message => message.id))
  persist(saved)

  // Act
  const result = recoverInFlightTurnJournal('rewritten-session', restored)
  const repeated = recoverInFlightTurnJournal('rewritten-session', result.messages)

  // Assert
  expect(result.caughtUp).toBe(false)
  expect(result.messages.map(chatMessageText)).toContain('Unique journal output')
  expect(new Set(result.messages.map(message => message.id)).size).toBe(result.messages.length)
  expect(repeated.messages.map(chatMessageText)).toEqual(result.messages.map(chatMessageText))
  expect(readInFlightTurnJournal('rewritten-session')).not.toBeNull()
})

it('retires a completed rewritten turn by stable UID without a shared provider call', () => {
  // Arrange
  persist(before())

  const restored = toChatMessages([
    { id: 19, role: 'assistant', content: 'Earlier response.', timestamp: 90 },
    { id: 20, message_uid: userUid, role: 'user', content: prompt, timestamp: 100 },
    { id: 21, role: 'assistant', content: 'Completed response.', timestamp: 102 }
  ])

  // Act
  const result = recoverInFlightTurnJournal('rewritten-session', restored)

  // Assert
  expect(result.caughtUp).toBe(true)
  expect(result.messages).toBe(restored)
  expect(readInFlightTurnJournal('rewritten-session')).toBeNull()
})

it.each([
  { name: 'same UID', savedUid: 'd'.repeat(32), restoredUid: 'd'.repeat(32), caughtUp: true },
  { name: 'different UIDs', savedUid: 'd'.repeat(32), restoredUid: 'e'.repeat(32), caughtUp: false },
  { name: 'legacy identity', savedUid: undefined, restoredUid: undefined, caughtUp: false }
])(
  'requires assistant identity to retire an unanchored rewritten tail ($name)',
  ({ savedUid, restoredUid, caughtUp }) => {
    // Arrange
    const saved = toChatMessages([
      { id: 11, message_uid: savedUid, role: 'assistant', content: 'Repeated answer', timestamp: 101 }
    ])

    const restored = toChatMessages([
      { id: 21, message_uid: restoredUid, role: 'assistant', content: 'Repeated answer', timestamp: 101 }
    ])

    persist(saved)

    // Act
    const result = recoverInFlightTurnJournal('rewritten-session', restored)

    // Assert
    expect(result.caughtUp).toBe(caughtUp)
    expect(result.messages.map(chatMessageText)).toEqual(
      caughtUp ? ['Repeated answer'] : ['Repeated answer', 'Repeated answer']
    )
    expect(readInFlightTurnJournal('rewritten-session') === null).toBe(caughtUp)
  }
)

it('keeps the journal-only suffix when its stable-UID prefix was reinserted', () => {
  // Arrange
  const cached = before()
  cached.at(-1)!.parts.push({ type: 'text', text: 'Still running.' })
  persist(cached)

  // Act
  const result = recoverInFlightTurnJournal('rewritten-session', after())

  // Assert
  expect(result.caughtUp).toBe(false)
  expect(result.messages.filter(message => message.role === 'user').map(chatMessageText)).toEqual([prompt])
  expect(result.messages.filter(message => message.role === 'assistant').map(chatMessageText)).toEqual([
    'Earlier response.',
    commentary,
    'Still running.'
  ])
  expect(result.messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call')).toHaveLength(1)
  expect(readInFlightTurnJournal('rewritten-session')).not.toBeNull()
})

it.each(['timestamp', 'tool identity', 'user boundary'])(
  'does not infer missing stable identity from legacy %s',
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
