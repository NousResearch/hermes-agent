import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import {
  clearInFlightTurnJournal,
  type JournalableSessionState,
  mergeInFlightMessages,
  persistInFlightTurnState,
  readInFlightTurnJournal,
  recoverInFlightTurnJournal
} from '@/lib/inflight-turn-journal'

const STORAGE_KEY = 'hermes.desktop.inflightTurnJournal.v1'

function user(id: string, text: string): ChatMessage {
  return { id, role: 'user', parts: [{ type: 'text', text }] }
}

function assistant(id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, role: 'assistant', parts: [{ type: 'text', text }], ...extra }
}

function assistantWithTool(id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id,
    role: 'assistant',
    parts: [
      { type: 'tool-call', toolCallId: 'tc-1', toolName: 'terminal', args: { command: 'ls' } },
      { type: 'text', text }
    ],
    ...extra
  }
}

function journalState(overrides: Partial<JournalableSessionState> = {}): JournalableSessionState {
  return {
    awaitingResponse: false,
    busy: true,
    messages: [user('u1', 'do the thing'), assistant('assistant-stream-1', 'partial answer', { pending: true })],
    storedSessionId: 'stored-1',
    streamId: 'assistant-stream-1',
    turnStartedAt: 1000,
    ...overrides
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  window.localStorage.clear()
})

afterEach(() => {
  clearInFlightTurnJournal('stored-1')
  vi.useRealTimers()
})

describe('persistInFlightTurnState', () => {
  it('journals the running turn tail after the throttle window', () => {
    persistInFlightTurnState(journalState())

    expect(readInFlightTurnJournal('stored-1')).toBeNull()

    vi.advanceTimersByTime(400)

    const entry = readInFlightTurnJournal('stored-1')
    expect(entry).not.toBeNull()
    expect(entry?.streamId).toBe('assistant-stream-1')
    expect(entry?.turnStartedAt).toBe(1000)
    expect(entry?.messages.map(m => m.role)).toEqual(['user', 'assistant'])
  })

  it('coalesces rapid updates into one write carrying the latest state', () => {
    persistInFlightTurnState(journalState())
    persistInFlightTurnState(
      journalState({
        messages: [
          user('u1', 'do the thing'),
          assistant('assistant-stream-1', 'partial answer grew', { pending: true })
        ]
      })
    )

    vi.advanceTimersByTime(400)

    const entry = readInFlightTurnJournal('stored-1')
    const tail = entry?.messages.find(m => m.role === 'assistant')
    expect(tail?.parts).toEqual([{ type: 'text', text: 'partial answer grew' }])
  })

  it('clears the entry the moment the turn settles, cancelling pending writes', () => {
    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)
    expect(readInFlightTurnJournal('stored-1')).not.toBeNull()

    persistInFlightTurnState(journalState({ messages: [] }))
    persistInFlightTurnState(journalState({ busy: false, awaitingResponse: false, streamId: null }))

    expect(readInFlightTurnJournal('stored-1')).toBeNull()

    vi.advanceTimersByTime(1000)
    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })

  it('does not journal a turn with no recoverable assistant content yet', () => {
    persistInFlightTurnState(journalState({ messages: [user('u1', 'do the thing')], streamId: null }))

    vi.advanceTimersByTime(400)
    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })

  it('expires entries older than the max age', () => {
    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)

    const raw = JSON.parse(window.localStorage.getItem(STORAGE_KEY)!)
    raw.entries['stored-1'].updatedAt = Date.now() - 8 * 24 * 60 * 60 * 1000
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(raw))

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })
})

describe('recoverInFlightTurnJournal', () => {
  function journalEntry(messages: ChatMessage[]) {
    persistInFlightTurnState(journalState({ messages, streamId: messages.at(-1)?.id ?? null }))
    vi.advanceTimersByTime(400)
  }

  it('is a reference-preserving no-op when nothing is journaled', () => {
    const base = [user('u1', 'do the thing')]
    const result = recoverInFlightTurnJournal('stored-1', base)

    expect(result.applied).toBe(false)
    expect(result.messages).toBe(base)
  })

  it('appends the full tail when the base transcript never saw the turn', () => {
    journalEntry([
      user('u1', 'do the thing'),
      assistantWithTool('assistant-stream-1', 'working on it', { pending: true })
    ])

    const base = [user('u0', 'earlier turn'), assistant('a0', 'earlier reply')]
    const result = recoverInFlightTurnJournal('stored-1', base)

    expect(result.applied).toBe(true)
    expect(result.messages.map(m => m.id)).toEqual(['u0', 'a0', 'u1', 'assistant-stream-1'])
    expect(result.streamId).toBe('assistant-stream-1')
  })

  it('appends only the assistant tail when the user row was persisted', () => {
    journalEntry([
      user('u1', 'do the thing'),
      assistantWithTool('assistant-stream-1', 'working on it', { pending: true })
    ])

    const base = [user('db-u1', 'do the thing')]
    const result = recoverInFlightTurnJournal('stored-1', base, { keepPending: false })

    expect(result.applied).toBe(true)
    expect(result.messages.map(m => m.id)).toEqual(['db-u1', 'assistant-stream-1'])
    const tail = result.messages.at(-1)!
    expect(tail.pending).toBe(false)
    expect(tail.parts[0]).toMatchObject({ type: 'tool-call' })
  })

  it('detects a committed reply as caught up and clears the entry', () => {
    journalEntry([user('u1', 'do the thing'), assistant('assistant-stream-1', 'partial', { pending: true })])

    const base = [user('db-u1', 'do the thing'), assistant('db-a1', 'full committed reply')]
    const result = recoverInFlightTurnJournal('stored-1', base)

    expect(result.applied).toBe(false)
    expect(result.caughtUp).toBe(true)
    expect(result.messages).toBe(base)
    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })

  it('overlays the backend text-only projection instead of dropping local tool progress', () => {
    // Sweeper regression on #44339: a backend `inflight` assistant snapshot
    // (text only) used to mark the richer local tail "caught up" and delete
    // locally recorded tool calls.
    journalEntry([
      user('u1', 'do the thing'),
      assistantWithTool('assistant-stream-old', 'local part', { pending: true })
    ])

    const base = [
      user('db-u1', 'do the thing'),
      assistant('assistant-stream-rt9', 'longer partial text from the backend snapshot', { pending: true })
    ]

    const result = recoverInFlightTurnJournal('stored-1', base, { keepPending: true })

    expect(result.applied).toBe(true)
    expect(result.caughtUp).toBe(false)
    expect(result.messages).toHaveLength(2)

    const merged = result.messages.at(-1)!
    // Keeps the BASE projection row id so live deltas keep landing on it.
    expect(merged.id).toBe('assistant-stream-rt9')
    expect(result.streamId).toBe('assistant-stream-rt9')
    // Journal structure survives; the longer backend text wins.
    expect(merged.parts[0]).toMatchObject({ type: 'tool-call', toolName: 'terminal' })
    expect(merged.parts[1]).toMatchObject({ type: 'text', text: 'longer partial text from the backend snapshot' })
    // Still in flight — the journal must NOT be cleared.
    expect(readInFlightTurnJournal('stored-1')).not.toBeNull()
  })

  it('keeps the journal text when it is longer than the projection text', () => {
    journalEntry([
      user('u1', 'do the thing'),
      assistantWithTool('assistant-stream-old', 'a much longer locally journaled partial answer', { pending: true })
    ])

    const base = [user('db-u1', 'do the thing'), assistant('assistant-stream-rt9', 'thin', { pending: true })]
    const result = recoverInFlightTurnJournal('stored-1', base, { keepPending: true })

    const merged = result.messages.at(-1)!
    expect(merged.id).toBe('assistant-stream-rt9')
    expect(merged.parts[1]).toMatchObject({ type: 'text', text: 'a much longer locally journaled partial answer' })
  })
})

describe('mergeInFlightMessages', () => {
  it('treats an error-bearing assistant row as recoverable content', () => {
    const tail = [user('u1', 'do the thing'), assistant('a-err', '', { error: 'provider exploded' })]
    const result = mergeInFlightMessages([user('db-u1', 'do the thing')], tail)

    expect(result.applied).toBe(true)
    expect(result.messages.at(-1)?.error).toBe('provider exploded')
  })

  it('ignores hidden rows when extracting nothing to recover', () => {
    const result = mergeInFlightMessages([], [user('u1', 'x')])

    expect(result.applied).toBe(false)
    expect(result.caughtUp).toBe(false)
  })
})
