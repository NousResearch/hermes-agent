import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import {
  clearInFlightTurnJournal,
  type JournalableSessionState,
  mergeInFlightMessages,
  persistInFlightTurnState,
  readInFlightTurnJournal,
  recoverInFlightTurnJournal,
  resetInFlightTurnJournalStateForTests
} from '@/lib/inflight-turn-journal'

const STORAGE_KEY = 'hermes.desktop.inflightTurnJournal.v1'
const STORAGE_PREFIX = 'hermes.desktop.inflightTurnJournal.v2:'
const MIGRATION_KEY = 'hermes.desktop.inflightTurnJournal.v2.migrated'

const sessionStorageKey = (storedSessionId: string) => `${STORAGE_PREFIX}${encodeURIComponent(storedSessionId)}`

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
  resetInFlightTurnJournalStateForTests()
  vi.useFakeTimers()
  window.localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  clearInFlightTurnJournal('stored-1')
  vi.useRealTimers()
})

describe('persistInFlightTurnState', () => {
  it('sweeps expired and oldest session entries once before the first write', () => {
    const now = Date.now()

    for (let index = 0; index < 25; index += 1) {
      const sessionId = `old-${index}`

      const snapshot = {
        messages: [
          user(`u-${index}`, `prompt-${index}`),
          assistant(`a-${index}`, `partial-${index}`, { pending: true })
        ],
        streamId: `a-${index}`,
        turnStartedAt: index,
        updatedAt: now - index * 1_000
      }

      window.localStorage.setItem(sessionStorageKey(sessionId), JSON.stringify(snapshot))
    }

    window.localStorage.setItem(
      sessionStorageKey('expired'),
      JSON.stringify({
        messages: [user('expired-u', 'expired'), assistant('expired-a', 'expired', { pending: true })],
        streamId: 'expired-a',
        turnStartedAt: 0,
        updatedAt: now - 8 * 24 * 60 * 60 * 1_000
      })
    )

    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)

    const sessionKeys = Array.from({ length: window.localStorage.length }, (_, index) =>
      window.localStorage.key(index)
    ).filter((key): key is string => key?.startsWith(STORAGE_PREFIX) === true)

    expect(sessionKeys).toHaveLength(24)
    expect(window.localStorage.getItem(sessionStorageKey('expired'))).toBeNull()
    expect(window.localStorage.getItem(sessionStorageKey('old-24'))).toBeNull()
    expect(window.localStorage.getItem(sessionStorageKey('stored-1'))).not.toBeNull()
  })

  it('writes only the current session instead of reading and rewriting the aggregate journal', () => {
    const localStorage = window.localStorage
    const storageConstructor = window.Storage

    const spyTarget =
      typeof storageConstructor === 'function' && localStorage instanceof storageConstructor
        ? storageConstructor.prototype
        : localStorage

    const getItem = vi.spyOn(spyTarget, 'getItem')
    const setItem = vi.spyOn(spyTarget, 'setItem')

    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)

    expect(getItem).not.toHaveBeenCalledWith(STORAGE_KEY)
    expect(setItem).not.toHaveBeenCalledWith(STORAGE_KEY, expect.any(String))
    expect(setItem).toHaveBeenCalledWith(sessionStorageKey('stored-1'), expect.any(String))
  })

  it('keeps another session snapshot when one session settles', () => {
    persistInFlightTurnState(journalState())
    persistInFlightTurnState(journalState({ storedSessionId: 'stored-2' }))
    vi.advanceTimersByTime(400)

    expect(window.localStorage.getItem(sessionStorageKey('stored-1'))).not.toBeNull()
    expect(window.localStorage.getItem(sessionStorageKey('stored-2'))).not.toBeNull()

    clearInFlightTurnJournal('stored-2')

    expect(readInFlightTurnJournal('stored-1')).not.toBeNull()
    expect(readInFlightTurnJournal('stored-2')).toBeNull()
  })

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

  it('preserves a long user prompt exactly so recovery still matches its transcript row', () => {
    const prompt = 'prompt '.repeat(8_000)

    persistInFlightTurnState(
      journalState({
        messages: [user('u1', prompt), assistant('assistant-stream-1', 'partial', { pending: true })]
      })
    )
    vi.advanceTimersByTime(400)

    const result = recoverInFlightTurnJournal('stored-1', [user('db-u1', prompt)])

    expect(result.messages.map(message => message.id)).toEqual(['db-u1', 'assistant-stream-1'])
  })

  it('preserves user attachment refs exactly so recovery still matches its transcript row', () => {
    const attachmentRefs = Array.from({ length: 25 }, (_, index) => `@file:/tmp/input-${index}.txt`)
    const prompt = user('u1', 'inspect these files')
    prompt.attachmentRefs = attachmentRefs

    persistInFlightTurnState(
      journalState({ messages: [prompt, assistant('assistant-stream-1', 'partial', { pending: true })] })
    )
    vi.advanceTimersByTime(400)

    const restoredPrompt = user('db-u1', 'inspect these files')
    restoredPrompt.attachmentRefs = attachmentRefs
    const result = recoverInFlightTurnJournal('stored-1', [restoredPrompt])

    expect(result.messages.map(message => message.id)).toEqual(['db-u1', 'assistant-stream-1'])
  })

  it('trims oldest sealed rows when bounded parts exceed the entry cap', () => {
    const text = 'x'.repeat(60 * 1024)

    const messages = [
      user('u1', 'do the thing'),
      assistant('a1', text, { pending: false }),
      assistant('a2', text, { pending: false }),
      assistant('a3', text, { pending: false }),
      assistant('a4', text, { pending: true })
    ]

    persistInFlightTurnState(journalState({ messages, streamId: 'a4' }))
    vi.advanceTimersByTime(400)

    const raw = window.localStorage.getItem(sessionStorageKey('stored-1'))
    const snapshot = JSON.parse(raw!)

    expect(raw?.length).toBeLessThanOrEqual(160 * 1024)
    expect(snapshot.messages.map((message: ChatMessage) => message.id)).toEqual(['u1', 'a3', 'a4'])
  })

  it('keeps an older recoverable snapshot when the newest row alone is too large', () => {
    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)

    const hugeAssistant: ChatMessage = {
      id: 'assistant-stream-1',
      role: 'assistant',
      parts: Array.from({ length: 3 }, () => ({ type: 'text' as const, text: 'x'.repeat(64 * 1024) })),
      pending: true
    }

    persistInFlightTurnState(
      journalState({ messages: [user('u1', 'do the thing'), hugeAssistant], streamId: hugeAssistant.id })
    )
    vi.advanceTimersByTime(400)

    const snapshot = JSON.parse(window.localStorage.getItem(sessionStorageKey('stored-1'))!)

    expect(snapshot.messages[1].parts).toEqual([{ type: 'text', text: 'partial answer' }])
  })

  it('skips a pathological user prompt instead of truncating its recovery join key', () => {
    const prompt = 'x'.repeat(64 * 1024 + 1)

    persistInFlightTurnState(
      journalState({
        messages: [user('u1', prompt), assistant('assistant-stream-1', 'partial', { pending: true })]
      })
    )
    vi.advanceTimersByTime(400)

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })

  it('does not parse the legacy aggregate when a pathological write discards its session', () => {
    const legacy = {
      messages: [user('legacy-u1', 'old prompt'), assistant('legacy-a1', 'old partial', { pending: true })],
      streamId: 'legacy-a1',
      turnStartedAt: 1,
      updatedAt: Date.now()
    }

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ entries: { 'stored-1': legacy }, version: 1 }))
    const getItem = vi.spyOn(Storage.prototype, 'getItem')
    const prompt = 'x'.repeat(64 * 1024 + 1)

    persistInFlightTurnState(
      journalState({
        messages: [user('u1', prompt), assistant('assistant-stream-1', 'partial', { pending: true })]
      })
    )
    vi.advanceTimersByTime(400)

    expect(getItem).not.toHaveBeenCalledWith(STORAGE_KEY)
    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })

  it('preserves a tombstone while sweeping before legacy migration', () => {
    const legacy = {
      messages: [user('legacy-u1', 'old prompt'), assistant('legacy-a1', 'old partial', { pending: true })],
      streamId: 'legacy-a1',
      turnStartedAt: 1,
      updatedAt: Date.now()
    }

    window.localStorage.setItem(sessionStorageKey('stored-1'), '0')
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ entries: { 'stored-1': legacy }, version: 1 }))

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
    expect(window.localStorage.getItem(sessionStorageKey('stored-1'))).toBeNull()
  })

  it('removes tombstones after the one-shot legacy migration has completed', () => {
    const key = sessionStorageKey('stored-1')

    window.localStorage.setItem(MIGRATION_KEY, '1')
    window.localStorage.setItem(key, '0')

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
    expect(window.localStorage.getItem(key)).toBeNull()
  })

  it('strips pathological 5 MiB tool payloads before attempting a storage write', () => {
    const setItem = vi.spyOn(Storage.prototype, 'setItem')

    const oversized: ChatMessage = {
      id: 'assistant-stream-1',
      role: 'assistant',
      parts: [
        {
          type: 'tool-call',
          toolCallId: 'tc-1',
          toolName: 'terminal',
          args: { command: 'x'.repeat(5 * 1024 * 1024) },
          result: 'x'.repeat(5 * 1024 * 1024),
          isError: true
        },
        { type: 'text', text: 'still useful' }
      ],
      pending: true
    }

    persistInFlightTurnState(journalState({ messages: [user('u1', 'do the thing'), oversized] }))
    vi.advanceTimersByTime(400)

    const raw = window.localStorage.getItem(sessionStorageKey('stored-1'))
    const snapshot = JSON.parse(raw!)
    const persistedTool = snapshot.messages[1].parts[0]

    expect(raw?.length).toBeLessThan(256 * 1024)
    expect(persistedTool).toEqual({
      args: {},
      isError: true,
      result: {},
      toolCallId: 'tc-1',
      toolName: 'terminal',
      type: 'tool-call'
    })
    expect(snapshot.messages[1].parts[1]).toEqual({ type: 'text', text: 'still useful' })
    expect(setItem.mock.calls.every(([, value]) => value.length <= 256 * 1024)).toBe(true)
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

    const key = sessionStorageKey('stored-1')
    const raw = JSON.parse(window.localStorage.getItem(key)!)
    raw.updatedAt = Date.now() - 8 * 24 * 60 * 60 * 1000
    window.localStorage.setItem(key, JSON.stringify(raw))

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })

  it('isolates storage read, write, and removal failures', () => {
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('read denied')
    })

    expect(() => readInFlightTurnJournal('stored-1')).not.toThrow()
    getItem.mockRestore()

    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota')
    })

    expect(() => {
      persistInFlightTurnState(journalState())
      vi.advanceTimersByTime(400)
    }).not.toThrow()
    setItem.mockRestore()

    const removeItem = vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
      throw new Error('remove denied')
    })

    expect(() => clearInFlightTurnJournal('stored-1')).not.toThrow()
  })

  it('discards malformed optional message metadata instead of throwing during recovery', () => {
    window.localStorage.setItem(
      sessionStorageKey('stored-1'),
      JSON.stringify({
        messages: [
          { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'prompt' }], attachmentRefs: '@file:bad' },
          { id: 'a1', role: 'assistant', parts: [{ type: 'text', text: 'partial' }], pending: true }
        ],
        streamId: 'a1',
        turnStartedAt: 1,
        updatedAt: Date.now()
      })
    )
    const base = [user('db-u1', 'prompt')]

    expect(() => recoverInFlightTurnJournal('stored-1', base)).not.toThrow()
    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })
})

describe('legacy journal migration', () => {
  it('migrates the bounded v1 aggregate once and recovers its sessions', () => {
    const first = {
      messages: [user('u1', 'one'), assistant('a1', 'partial one', { pending: true })],
      streamId: 'a1',
      turnStartedAt: 1,
      updatedAt: Date.now()
    }

    const second = {
      messages: [
        user('u2', 'two'),
        {
          id: 'a2',
          role: 'assistant' as const,
          parts: [
            {
              type: 'tool-call' as const,
              toolCallId: 'tc-legacy',
              toolName: 'terminal',
              args: { command: 'large-output' },
              result: 'x'.repeat(1024 * 1024)
            },
            { type: 'text' as const, text: 'partial two' }
          ],
          pending: true
        }
      ],
      streamId: 'a2',
      turnStartedAt: 2,
      updatedAt: Date.now()
    }

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ entries: { one: first, two: second }, version: 1 }))

    expect(readInFlightTurnJournal('one')).toEqual(first)
    const migratedSecondRaw = window.localStorage.getItem(sessionStorageKey('two'))!
    const migratedSecond = JSON.parse(migratedSecondRaw)

    expect(migratedSecondRaw.length).toBeLessThan(256 * 1024)
    expect(migratedSecond.messages[1].parts).toEqual([
      { args: {}, result: {}, toolCallId: 'tc-legacy', toolName: 'terminal', type: 'tool-call' },
      { text: 'partial two', type: 'text' }
    ])
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
    expect(window.localStorage.getItem(MIGRATION_KEY)).toBe('1')

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ entries: { three: first }, version: 1 }))
    expect(readInFlightTurnJournal('three')).toBeNull()
    expect(window.localStorage.getItem(STORAGE_KEY)).not.toBeNull()
  })

  it('drops an oversized legacy aggregate without parsing it', () => {
    window.localStorage.setItem(STORAGE_KEY, 'x'.repeat(2 * 1024 * 1024 + 1))

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
    expect(window.localStorage.getItem(MIGRATION_KEY)).toBe('1')
  })

  it('does not overwrite a newer per-session snapshot while migrating another legacy session', () => {
    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)

    const legacyCurrent = {
      messages: [user('legacy-u1', 'old prompt'), assistant('legacy-a1', 'old partial', { pending: true })],
      streamId: 'legacy-a1',
      turnStartedAt: 1,
      updatedAt: Date.now() - 1_000
    }

    const legacyOther = {
      messages: [user('u2', 'other prompt'), assistant('a2', 'other partial', { pending: true })],
      streamId: 'a2',
      turnStartedAt: 2,
      updatedAt: Date.now()
    }

    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ entries: { 'stored-1': legacyCurrent, other: legacyOther }, version: 1 })
    )

    expect(readInFlightTurnJournal('other')).toEqual(legacyOther)
    expect(readInFlightTurnJournal('stored-1')?.messages[0]).toEqual(user('u1', 'do the thing'))
  })

  it('does not resurrect a legacy entry after that session settles before migration', () => {
    const legacyCurrent = {
      messages: [user('legacy-u1', 'old prompt'), assistant('legacy-a1', 'old partial', { pending: true })],
      streamId: 'legacy-a1',
      turnStartedAt: 1,
      updatedAt: Date.now()
    }

    const legacyOther = {
      messages: [user('u2', 'other prompt'), assistant('a2', 'other partial', { pending: true })],
      streamId: 'a2',
      turnStartedAt: 2,
      updatedAt: Date.now()
    }

    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ entries: { 'stored-1': legacyCurrent, other: legacyOther }, version: 1 })
    )

    persistInFlightTurnState(journalState({ busy: false, awaitingResponse: false, streamId: null }))

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
    expect(readInFlightTurnJournal('other')).toEqual(legacyOther)
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
    // locally recorded tool calls. After #76444, longer text wins only when it
    // is a strict extension of the journal answer (flat thinking dumps must
    // not replace structured answer text).
    journalEntry([
      user('u1', 'do the thing'),
      assistantWithTool('assistant-stream-old', 'local part', { pending: true })
    ])

    const base = [
      user('db-u1', 'do the thing'),
      assistant('assistant-stream-rt9', 'local part and more from the backend snapshot', { pending: true })
    ]

    const result = recoverInFlightTurnJournal('stored-1', base, { keepPending: true })

    expect(result.applied).toBe(true)
    expect(result.caughtUp).toBe(false)
    expect(result.messages).toHaveLength(2)

    const merged = result.messages.at(-1)!
    // Keeps the BASE projection row id so live deltas keep landing on it.
    expect(merged.id).toBe('assistant-stream-rt9')
    expect(result.streamId).toBe('assistant-stream-rt9')
    // Journal structure survives; strict-extension backend text wins.
    expect(merged.parts[0]).toMatchObject({ type: 'tool-call', toolName: 'terminal' })
    expect(merged.parts[1]).toMatchObject({ type: 'text', text: 'local part and more from the backend snapshot' })
    // Still in flight — the journal must NOT be cleared.
    expect(readInFlightTurnJournal('stored-1')).not.toBeNull()
  })

  it('keeps journal answer text when a longer flat dump is not a strict extension (#76444)', () => {
    journalEntry([user('u1', 'do the thing'), assistantWithTool('assistant-stream-old', 'partial', { pending: true })])

    const base = [
      user('db-u1', 'do the thing'),
      assistant(
        'assistant-stream-rt9',
        'thinking chatter\nRan terminal\npartial and unrelated dump longer than answer',
        { pending: true }
      )
    ]

    const result = recoverInFlightTurnJournal('stored-1', base, { keepPending: true })
    const merged = result.messages.at(-1)!

    expect(merged.parts[0]).toMatchObject({ type: 'tool-call', toolName: 'terminal' })
    expect(merged.parts[1]).toMatchObject({ type: 'text', text: 'partial' })
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

describe('mid-turn redirect corrections', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  // A redirect inserts its correction as a second user row directly before the
  // live reply, so the turn opens with a RUN of user rows. Journaling only back
  // to the nearest one lost the prompt that actually started the turn — the
  // vanishing user bubble.
  it('journals the whole user run, not just the correction', () => {
    persistInFlightTurnState({
      awaitingResponse: false,
      busy: true,
      messages: [
        user('user-1', 'remove the session counts'),
        user('user-2', 'hurry up'),
        assistant('assistant-stream-1', 'Moving.', { pending: true })
      ],
      storedSessionId: 'stored-redirect',
      streamId: 'assistant-stream-1',
      turnStartedAt: Date.now()
    })
    vi.advanceTimersByTime(400)

    const journaled = readInFlightTurnJournal('stored-redirect')?.messages ?? []

    expect(journaled.map(message => message.parts.map(part => (part as { text: string }).text).join(''))).toEqual([
      'remove the session counts',
      'hurry up',
      'Moving.'
    ])
  })

  it('still stops at an assistant boundary so prior turns are not journaled', () => {
    persistInFlightTurnState({
      awaitingResponse: false,
      busy: true,
      messages: [
        user('user-old', 'an earlier turn'),
        assistant('assistant-old', 'an earlier answer'),
        user('user-1', 'the live prompt'),
        assistant('assistant-stream-1', 'Moving.', { pending: true })
      ],
      storedSessionId: 'stored-boundary',
      streamId: 'assistant-stream-1',
      turnStartedAt: Date.now()
    })
    vi.advanceTimersByTime(400)

    const journaled = readInFlightTurnJournal('stored-boundary')?.messages ?? []

    expect(journaled.map(message => message.id)).toEqual(['user-1', 'assistant-stream-1'])
  })
})
