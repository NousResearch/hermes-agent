import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import {
  clearInFlightTurnJournal,
  type JournalableSessionState,
  mergeInFlightMessages,
  persistInFlightTurnState,
  readInFlightTurnJournal,
  recoverInFlightTurnJournal,
  resetInFlightTurnJournalForTests
} from '@/lib/inflight-turn-journal'

const LEGACY_KEY = 'hermes.desktop.inflightTurnJournal.v1'
const KEY_PREFIX = 'hermes.desktop.inflightTurnJournal.v2.'
const KILL_SWITCH_KEY = 'hermes.desktop.inflightTurnJournal.disabled'

const sessionKey = (id: string) => `${KEY_PREFIX}${id}`

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
  resetInFlightTurnJournalForTests()
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

  it('writes one v2 key per session and never an aggregate key', () => {
    persistInFlightTurnState(journalState())
    persistInFlightTurnState(journalState({ storedSessionId: 'stored-2' }))

    vi.advanceTimersByTime(400)

    expect(window.localStorage.getItem(sessionKey('stored-1'))).not.toBeNull()
    expect(window.localStorage.getItem(sessionKey('stored-2'))).not.toBeNull()
    expect(window.localStorage.getItem(LEGACY_KEY)).toBeNull()

    clearInFlightTurnJournal('stored-2')
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

    const raw = JSON.parse(window.localStorage.getItem(sessionKey('stored-1'))!)
    raw.updatedAt = Date.now() - 8 * 24 * 60 * 60 * 1000
    window.localStorage.setItem(sessionKey('stored-1'), JSON.stringify(raw))

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
    expect(window.localStorage.getItem(sessionKey('stored-1'))).toBeNull()
  })

  it('honors the kill switch without touching storage', () => {
    window.localStorage.setItem(KILL_SWITCH_KEY, '1')
    resetInFlightTurnJournalForTests()

    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)

    expect(window.localStorage.getItem(sessionKey('stored-1'))).toBeNull()
  })
})

describe('bounded projection', () => {
  it('bounds a multi-megabyte tool result while preserving tool identity and recovery', () => {
    const hugeResult = 'x'.repeat(5 * 1024 * 1024)

    const messages: ChatMessage[] = [
      user('u1', 'run the big thing'),
      {
        id: 'assistant-stream-1',
        role: 'assistant',
        pending: true,
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'tc-big',
            toolName: 'terminal',
            args: { command: 'dump' },
            argsText: '{"command":"dump"}',
            result: hugeResult,
            isError: false
          },
          { type: 'text', text: 'digesting the output' }
        ]
      }
    ]

    persistInFlightTurnState(journalState({ messages }))
    vi.advanceTimersByTime(400)

    const raw = window.localStorage.getItem(sessionKey('stored-1'))!
    expect(raw.length).toBeLessThan(256 * 1024)

    const entry = readInFlightTurnJournal('stored-1')!
    const tool = entry.messages.at(-1)!.parts[0] as { isError: boolean; result?: string; toolCallId: string; toolName: string }
    expect(tool.toolCallId).toBe('tc-big')
    expect(tool.toolName).toBe('terminal')
    expect(tool.isError).toBe(false)
    expect((tool.result ?? '').length).toBeLessThanOrEqual(2 * 1024 + 20)

    // The bounded entry still merges back onto a restored transcript.
    const recovered = recoverInFlightTurnJournal('stored-1', [user('db-u1', 'run the big thing')])
    expect(recovered.applied).toBe(true)
    expect(recovered.messages.at(-1)!.parts[0]).toMatchObject({ type: 'tool-call', toolCallId: 'tc-big' })
  })

  it('keeps the user prompt in full so recovery matching still works', () => {
    const longPrompt = 'p'.repeat(100 * 1024)

    persistInFlightTurnState(
      journalState({
        messages: [user('u1', longPrompt), assistant('assistant-stream-1', 'ok', { pending: true })]
      })
    )
    vi.advanceTimersByTime(400)

    const recovered = recoverInFlightTurnJournal('stored-1', [
      user('db-u1', longPrompt),
      assistant('db-a1', 'full committed reply')
    ])

    expect(recovered.caughtUp).toBe(true)
  })
})

describe('boot migration from the v1 aggregate blob', () => {
  it('migrates non-expired entries to bounded v2 keys and deletes the v1 key', () => {
    const hugeResult = 'y'.repeat(3 * 1024 * 1024)

    const v1 = {
      version: 1,
      entries: {
        'stored-normal': {
          messages: [user('u1', 'small turn'), assistant('assistant-stream-1', 'partial', { pending: true })],
          streamId: 'assistant-stream-1',
          turnStartedAt: 111,
          updatedAt: Date.now()
        },
        'stored-oversized': {
          messages: [
            user('u2', 'big turn'),
            {
              id: 'assistant-stream-2',
              role: 'assistant',
              pending: true,
              parts: [
                { type: 'tool-call', toolCallId: 'tc-2', toolName: 'terminal', args: {}, result: hugeResult },
                { type: 'text', text: 'working' }
              ]
            }
          ],
          streamId: 'assistant-stream-2',
          turnStartedAt: 222,
          updatedAt: Date.now()
        },
        'stored-expired': {
          messages: [user('u3', 'old'), assistant('a3', 'stale')],
          streamId: null,
          turnStartedAt: null,
          updatedAt: Date.now() - 8 * 24 * 60 * 60 * 1000
        }
      }
    }

    window.localStorage.setItem(LEGACY_KEY, JSON.stringify(v1))
    resetInFlightTurnJournalForTests()

    // Any journal entry point boots the module.
    expect(readInFlightTurnJournal('stored-normal')).not.toBeNull()

    expect(window.localStorage.getItem(LEGACY_KEY)).toBeNull()
    expect(window.localStorage.getItem(sessionKey('stored-expired'))).toBeNull()

    const oversized = window.localStorage.getItem(sessionKey('stored-oversized'))!
    expect(oversized.length).toBeLessThan(256 * 1024)
    expect(readInFlightTurnJournal('stored-oversized')?.turnStartedAt).toBe(222)

    clearInFlightTurnJournal('stored-normal')
    clearInFlightTurnJournal('stored-oversized')
  })

  it('respects new-key-wins when a v2 key already exists', () => {
    persistInFlightTurnState(journalState({ storedSessionId: 'stored-dup' }))
    vi.advanceTimersByTime(400)
    const existing = window.localStorage.getItem(sessionKey('stored-dup'))!

    window.localStorage.setItem(
      LEGACY_KEY,
      JSON.stringify({
        version: 1,
        entries: {
          'stored-dup': {
            messages: [user('legacy-u', 'legacy turn'), assistant('legacy-a', 'legacy partial')],
            streamId: null,
            turnStartedAt: 999,
            updatedAt: Date.now()
          }
        }
      })
    )
    resetInFlightTurnJournalForTests()

    expect(readInFlightTurnJournal('stored-dup')).not.toBeNull()
    expect(window.localStorage.getItem(sessionKey('stored-dup'))).toBe(existing)
    expect(window.localStorage.getItem(LEGACY_KEY)).toBeNull()

    clearInFlightTurnJournal('stored-dup')
  })

  it('deletes an unparseable v1 blob instead of keeping it around', () => {
    window.localStorage.setItem(LEGACY_KEY, '{not json at all')
    resetInFlightTurnJournalForTests()

    expect(readInFlightTurnJournal('anything')).toBeNull()
    expect(window.localStorage.getItem(LEGACY_KEY)).toBeNull()
  })
})

describe('quota and write failures', () => {
  it('keeps streaming unaffected when setItem throws', () => {
    persistInFlightTurnState(journalState())

    const spy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('quota', 'QuotaExceededError')
    })

    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})

    try {
      expect(() => vi.advanceTimersByTime(400)).not.toThrow()
      expect(warn).toHaveBeenCalledTimes(1)
    } finally {
      spy.mockRestore()
      warn.mockRestore()
    }

    expect(readInFlightTurnJournal('stored-1')).toBeNull()

    // A later write (storage healthy again) succeeds.
    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)
    expect(readInFlightTurnJournal('stored-1')).not.toBeNull()
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
    resetInFlightTurnJournalForTests()
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

describe('external review hardening', () => {
  it('previews object-valued tool results without traversing the full structure', () => {
    let elementReads = 0
    const bigArray = Array.from({ length: 200_000 }, (_, index) => `row-${index}-${'z'.repeat(24)}`)

    const counted = new Proxy(bigArray, {
      get(target, prop, receiver) {
        if (typeof prop === 'string' && /^\d+$/.test(prop)) {
          elementReads += 1
        }

        return Reflect.get(target, prop, receiver)
      }
    })

    persistInFlightTurnState(
      journalState({
        messages: [
          user('u1', 'run it'),
          {
            id: 'assistant-stream-1',
            role: 'assistant',
            pending: true,
            parts: [
              {
                type: 'tool-call',
                toolCallId: 'tc-obj',
                toolName: 'terminal',
                args: {},
                result: counted as unknown as string
              },
              { type: 'text', text: 'reading' }
            ]
          }
        ]
      })
    )
    vi.advanceTimersByTime(400)

    const raw = window.localStorage.getItem(sessionKey('stored-1'))!
    expect(raw.length).toBeLessThan(256 * 1024)
    // The 2KB preview budget stops the walk after ~70 rows of ~30 chars; a
    // full JSON.stringify of the object would have read all 200k elements.
    expect(elementReads).toBeLessThan(1000)

    const entry = readInFlightTurnJournal('stored-1')!
    const tool = entry.messages.at(-1)!.parts[0] as { result?: string; toolCallId: string }
    expect(tool.toolCallId).toBe('tc-obj')
    expect((tool.result ?? '').length).toBeLessThanOrEqual(2 * 1024 + 20)
  })

  it('skips the write entirely when even the tight projection exceeds the hard ceiling', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    // Only an un-truncatable user prompt can defeat the tight caps.
    const enormousPrompt = 'w'.repeat(600 * 1024)

    try {
      persistInFlightTurnState(
        journalState({
          messages: [user('u1', enormousPrompt), assistant('assistant-stream-1', 'ok', { pending: true })]
        })
      )
      vi.advanceTimersByTime(400)

      expect(window.localStorage.getItem(sessionKey('stored-1'))).toBeNull()
      expect(warn).toHaveBeenCalledTimes(1)
      expect(String(warn.mock.calls[0][0])).toContain('hard size ceiling')

      // Once per session, not once per tick.
      persistInFlightTurnState(
        journalState({
          messages: [user('u1', enormousPrompt), assistant('assistant-stream-1', 'ok then', { pending: true })]
        })
      )
      vi.advanceTimersByTime(400)
      expect(warn).toHaveBeenCalledTimes(1)
    } finally {
      warn.mockRestore()
    }
  })

  it('degrades to no journal when a storage read fails during recovery', () => {
    persistInFlightTurnState(journalState())
    vi.advanceTimersByTime(400)
    expect(readInFlightTurnJournal('stored-1')).not.toBeNull()

    const spy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError')
    })

    try {
      expect(() => readInFlightTurnJournal('stored-1')).not.toThrow()
      expect(readInFlightTurnJournal('stored-1')).toBeNull()

      const recovered = recoverInFlightTurnJournal('stored-1', [user('db-u1', 'do the thing')])
      expect(recovered.applied).toBe(false)
      expect(recovered.messages).toHaveLength(1)
    } finally {
      spy.mockRestore()
    }
  })

  it('enforces the global entry cap at runtime, not only at boot', () => {
    for (let index = 0; index < 25; index += 1) {
      persistInFlightTurnState(journalState({ storedSessionId: `stored-cap-${index}` }))
      // Each write lands at a distinct Date.now() so eviction order is stable.
      vi.advanceTimersByTime(401)
    }

    const journalKeys = Object.keys(window.localStorage).filter(key => key.startsWith(KEY_PREFIX))
    expect(journalKeys).toHaveLength(24)
    // The oldest write was evicted by the sweep the 25th session triggered.
    expect(window.localStorage.getItem(sessionKey('stored-cap-0'))).toBeNull()
    expect(window.localStorage.getItem(sessionKey('stored-cap-24'))).not.toBeNull()

    for (let index = 0; index < 25; index += 1) {
      clearInFlightTurnJournal(`stored-cap-${index}`)
    }
  })

  it('does nothing at boot when the kill switch is set — no migration, no sweep', () => {
    const legacyBlob = JSON.stringify({
      version: 1,
      entries: {
        'stored-legacy': {
          messages: [user('u1', 'legacy'), assistant('a1', 'legacy partial')],
          streamId: null,
          turnStartedAt: 1,
          updatedAt: Date.now()
        }
      }
    })

    window.localStorage.setItem(LEGACY_KEY, legacyBlob)
    window.localStorage.setItem(KILL_SWITCH_KEY, '1')
    resetInFlightTurnJournalForTests()

    const getItem = vi.spyOn(Storage.prototype, 'getItem')
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    const removeItem = vi.spyOn(Storage.prototype, 'removeItem')

    try {
      persistInFlightTurnState(journalState())
      vi.advanceTimersByTime(1000)

      // The kill-switch probe is the subsystem's ONLY storage access.
      expect(getItem).toHaveBeenCalledTimes(1)
      expect(getItem).toHaveBeenCalledWith(KILL_SWITCH_KEY)
      expect(setItem).not.toHaveBeenCalled()
      expect(removeItem).not.toHaveBeenCalled()
    } finally {
      getItem.mockRestore()
      setItem.mockRestore()
      removeItem.mockRestore()
    }

    // The legacy blob is untouched, preserved for the revert case.
    expect(window.localStorage.getItem(LEGACY_KEY)).toBe(legacyBlob)
  })
})
