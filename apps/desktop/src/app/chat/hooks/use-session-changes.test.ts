import { act, cleanup, render } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import type { ClientSessionState } from '@/app/types'
import type { StatusResponse } from '@/types/hermes'

import {
  advanceCursorAfterRows,
  appendFetchedMessages,
  createSessionChangesController,
  adoptZombieFooters,
  discardUnstampedOptimisticTranscriptRows,
  dropZombieOptimisticRows,
  extractCommittedMessageIds,
  maxCommittedMessageId,
  sessionChangesSupported,
  stampOptimisticTranscriptRows,
  useSessionChanges
} from './use-session-changes'

const SID = 'session-1'
const STORED_SID = '20260101_000000_stored1'
type TestGatewayRequest = (method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<unknown>

function message(id: string, role: ChatMessage['role'] = 'user'): ChatMessage {
  return {
    id,
    role,
    parts: [{ type: 'text', text: `${role}:${id}` }]
  }
}

function status(capabilities: Record<string, unknown> = { session_changes: true }): StatusResponse {
  return {
    active_sessions: 0,
    capabilities,
    config_path: '',
    config_version: 1,
    env_path: '',
    gateway_exit_reason: null,
    gateway_health_url: null,
    gateway_pid: null,
    gateway_platforms: {},
    gateway_running: true,
    gateway_state: 'running',
    gateway_updated_at: null,
    hermes_home: '',
    latest_config_version: 1,
    release_date: '',
    version: 'test'
  } as StatusResponse
}

function setFocused(focused: boolean) {
  Object.defineProperty(document, 'hasFocus', { configurable: true, value: () => focused })
}

interface HarnessProps {
  activeSessionId?: string | null
  busy?: boolean
  currentView?: string
  initialMessages?: ChatMessage[]
  onState?: (state: ClientSessionState) => void
  requestGateway: TestGatewayRequest
  statusSnapshot?: StatusResponse | null
  storedSessionId?: string | null
}

function Harness({
  activeSessionId = SID,
  busy = false,
  currentView = 'chat',
  initialMessages = [message('4')],
  onState,
  requestGateway,
  statusSnapshot = status(),
  storedSessionId = STORED_SID
}: HarnessProps) {
  const state: ClientSessionState = {
    awaitingResponse: false,
    branch: '',
    busy: false,
    cwd: '',
    fast: false,
    interrupted: false,
    messages: initialMessages,
    model: '',
    needsInput: false,
    pendingBranchGroup: null,
    personality: '',
    provider: '',
    reasoningEffort: '',
    sawAssistantPayload: false,
    serviceTier: '',
    storedSessionId: SID,
    streamId: null,
    turnStartedAt: null,
    yolo: false
  }

  useSessionChanges({
    activeSessionId,
    busy,
    currentView,
    messages: initialMessages,
    requestGateway: requestGateway as <T = unknown>(
      method: string,
      params?: Record<string, unknown>,
      timeoutMs?: number
    ) => Promise<T>,
    statusSnapshot,
    storedSessionId,
    updateSessionState: (_sessionId, updater) => {
      const next = updater(state)
      onState?.(next)

      return next
    }
  })

  return null
}

beforeEach(() => {
  vi.useFakeTimers()
  setFocused(true)
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('useSessionChanges B1', () => {
  it('gates polling on the session_changes capability', () => {
    expect(sessionChangesSupported(status())).toBe(true)
    expect(sessionChangesSupported(status({}))).toBe(false)

    const requestGateway = vi.fn(async () => ({ messages: [], last_id: 0 }))
    render(createElement(Harness, { requestGateway, statusSnapshot: status({}) }))

    act(() => vi.advanceTimersByTime(10_000))

    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('initializes the in-memory cursor from the loaded transcript max id', () => {
    expect(maxCommittedMessageId([message('1'), message('12'), message('temp-user')])).toBe(12)
    expect(createSessionChangesController([]).cursor).toBe(0)
  })

  it('stops polling while blurred and coalesces refocus to exactly one immediate poll', async () => {
    const requestGateway = vi.fn(async () => ({ messages: [], last_id: 4 }))

    render(createElement(Harness, { requestGateway }))

    act(() => {
      window.dispatchEvent(new Event('blur'))
      vi.advanceTimersByTime(7_500)
    })

    expect(requestGateway).not.toHaveBeenCalled()

    await act(async () => {
      window.dispatchEvent(new Event('focus'))
      window.dispatchEvent(new Event('focus'))
      vi.advanceTimersByTime(999)
      await Promise.resolve()
    })

    expect(requestGateway).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(1)
      await Promise.resolve()
    })

    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(requestGateway).toHaveBeenCalledWith('session.changes', {
      session_id: STORED_SID,
      since_message_id: 4
    })
  })

  it('puts the STORED session id on the wire, never the runtime id (4044 contract)', async () => {
    const requestGateway = vi.fn(async () => ({ messages: [], last_id: 4 }))

    render(createElement(Harness, { requestGateway }))

    await act(async () => {
      vi.advanceTimersByTime(2_500)
      await Promise.resolve()
    })

    expect(requestGateway).toHaveBeenCalled()
    for (const call of requestGateway.mock.calls as unknown[][]) {
      if (call[0] !== 'session.changes') continue
      expect((call[1] as { session_id: string }).session_id).toBe(STORED_SID)
      expect((call[1] as { session_id: string }).session_id).not.toBe(SID)
    }
  })

  it('falls back to the runtime id only when no stored id exists yet', async () => {
    const requestGateway = vi.fn(async () => ({ messages: [], last_id: 4 }))

    render(createElement(Harness, { requestGateway, storedSessionId: null }))

    await act(async () => {
      vi.advanceTimersByTime(2_500)
      await Promise.resolve()
    })

    const changesCalls = requestGateway.mock.calls.filter(call => (call as unknown[])[0] === 'session.changes')
    expect(changesCalls.length).toBeGreaterThan(0)
    expect(((changesCalls[0] as unknown[])[1] as { session_id: string }).session_id).toBe(SID)
  })

  it('stops quietly on feature-disabled errors without advancing the cursor', async () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined)
    const requestGateway = vi.fn(async () => {
      throw new Error('session changes disabled')
    })

    render(createElement(Harness, { requestGateway }))

    await act(async () => {
      vi.advanceTimersByTime(2_500)
      await Promise.resolve()
    })

    await act(async () => {
      vi.advanceTimersByTime(10_000)
      await Promise.resolve()
    })

    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(info).toHaveBeenCalledTimes(1)
  })

  it('does not advance the cursor over unfetched or unrendered rows', () => {
    expect(
      advanceCursorAfterRows(
        9,
        [
          { id: 10, role: 'user', content: 'own' },
          { id: 11, role: 'assistant', content: 'remote' }
        ],
        [message('10')]
      )
    ).toBe(10)
  })
})

describe('useSessionChanges B2 materialization', () => {
  it('dedupes already-rendered committed ids and appends new rows through the resume materializer', () => {
    const result = appendFetchedMessages([message('10', 'user')], [
      { id: 10, role: 'user', content: 'already rendered' },
      { id: 11, role: 'assistant', content: 'new assistant', timestamp: 11 }
    ])

    expect(result.messages).toHaveLength(2)
    expect(result.messages.map(row => row.id)).toEqual(['10', '11'])
    expect(result.messages[1]?.parts).toEqual([{ type: 'text', text: 'new assistant' }])
  })

  it('advances the cursor past tool-result rows merged into the preceding assistant message (Greptile #268)', () => {
    // toChatMessages collapses assistant-with-tool_calls (id=12) + its tool
    // result row (id=13) into ONE ChatMessage carrying id=12. Row 13 is
    // consumed, not surfaced — the cursor must still advance past it, or
    // every subsequent poll re-fetches 13 and appends a duplicate tool card.
    const result = appendFetchedMessages([], [
      {
        id: 12,
        role: 'assistant',
        content: '',
        timestamp: 12,
        tool_calls: [{ id: 'call_x', function: { name: 'search_files', arguments: '{}' } }]
      },
      { id: 13, role: 'tool', content: '{}', tool_call_id: 'call_x', timestamp: 13 }
    ])

    expect(result.cursor).toBe(13)
    expect(result.renderedIds.has('13')).toBe(true)

    // Re-poll returning row 13 again (unchanged-cursor path) must be a
    // no-op, not a duplicate — renderedIds carry consumed row ids forward.
    const again = appendFetchedMessages(result.messages, [
      { id: 13, role: 'tool', content: '{}', tool_call_id: 'call_x', timestamp: 13 }
    ], result.renderedIds)
    expect(again.messages).toHaveLength(result.messages.length)
  })
})

describe('useSessionChanges B3 partial turns', () => {
  it('renders a polled assistant tool-call prefix as the existing pending tool-call shape', () => {
    const result = appendFetchedMessages([], [
      {
        id: 20,
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'call-1',
            function: { name: 'search_files', arguments: { query: 'needle' } }
          }
        ]
      }
    ])

    expect(result.messages).toHaveLength(1)
    expect(result.messages[0]?.id).toBe('20')
    const [part] = result.messages[0]?.parts ?? []

    expect(part).toEqual(
      expect.objectContaining({
        toolCallId: 'call-1',
        toolName: 'search_files',
        type: 'tool-call'
      })
    )
    expect(part && 'result' in part).toBe(false)
  })

  it('keeps the cursor at the last rendered/deduped id when a fetched row is not rendered', () => {
    expect(
      advanceCursorAfterRows(
        20,
        [
          { id: 21, role: 'assistant', content: 'rendered' },
          { id: 22, role: 'assistant', content: 'not rendered yet' }
        ],
        [message('21', 'assistant')]
      )
    ).toBe(21)
  })
})

describe('useSessionChanges B4 own-turn suspension helpers', () => {
  it('stamps optimistic transcript rows from completion frame committed ids', () => {
    const stamped = stampOptimisticTranscriptRows(
      [
        message('user-temp', 'user'),
        { ...message('assistant-stream-1', 'assistant'), pending: true }
      ],
      extractCommittedMessageIds({ message_ids: [100, 103] })
    )

    expect(stamped.messages.map(row => row.id)).toEqual(['100', '103'])
    expect([...stamped.stampedIds]).toEqual(['100', '103'])
  })

  it('drops own rows re-returned by the post-completion poll after stamping', () => {
    const stamped = stampOptimisticTranscriptRows(
      [
        message('user-temp', 'user'),
        { ...message('assistant-stream-1', 'assistant'), pending: true }
      ],
      ['100', '101']
    )
    const result = appendFetchedMessages(stamped.messages, [
      { id: 100, role: 'user', content: 'own user' },
      { id: 101, role: 'assistant', content: 'own assistant' }
    ])

    expect(result.messages.map(row => row.id)).toEqual(['100', '101'])
  })

  it('stamps an ALREADY-COMPLETED assistant row (pending cleared before stamping)', () => {
    // Real runtime ordering: completeAssistantMessage() clears the streamed
    // assistant row's `pending` flag BEFORE markTurnComplete() runs the stamp.
    // So the row arriving at stampOptimisticTranscriptRows is pending:false and
    // carries a non-`user-` id (`assistant-<ts>`). If the stamp predicate only
    // accepts `user-`-prefixed OR pending rows, the assistant row is skipped,
    // keeps its optimistic id, and the post-completion poll then appends the
    // committed integer row as a DUPLICATE (the reported desktop bug).
    const stamped = stampOptimisticTranscriptRows(
      [
        message('user-temp', 'user'),
        { ...message('assistant-1', 'assistant'), pending: false }
      ],
      extractCommittedMessageIds({ message_ids: [100, 103] })
    )

    expect(stamped.messages.map(row => row.id)).toEqual(['100', '103'])
    expect([...stamped.stampedIds]).toEqual(['100', '103'])
  })

  it('does NOT duplicate the assistant row end-to-end when the poll returns own committed rows', () => {
    // Full seam: optimistic user + completed assistant → stamp from completion
    // ids → post-completion poll returns the same committed rows → no dup.
    const stamped = stampOptimisticTranscriptRows(
      [
        message('user-temp', 'user'),
        { ...message('assistant-1', 'assistant'), pending: false }
      ],
      extractCommittedMessageIds({ message_ids: [100, 101] })
    )
    const result = appendFetchedMessages(stamped.messages, [
      { id: 100, role: 'user', content: 'own user' },
      { id: 101, role: 'assistant', content: 'own assistant' }
    ])

    expect(result.messages.map(row => row.id)).toEqual(['100', '101'])
  })

  it('renders remote ids interleaved between own ids and advances only through rendered rows', () => {
    const stamped = stampOptimisticTranscriptRows(
      [
        message('user-temp', 'user'),
        { ...message('assistant-stream-1', 'assistant'), pending: true }
      ],
      ['100', '103']
    )
    const result = appendFetchedMessages(stamped.messages, [
      { id: 100, role: 'user', content: 'own user' },
      { id: 101, role: 'user', content: 'remote user' },
      { id: 102, role: 'assistant', content: 'remote assistant' },
      { id: 103, role: 'assistant', content: 'own assistant' }
    ])

    expect(result.messages.map(row => row.id)).toEqual(['100', '101', '102', '103'])
    expect(advanceCursorAfterRows(99, [{ id: 100, role: 'user', content: '' }], [message('100')])).toBe(100)
  })
})

describe('useSessionChanges B5 watchdog and hatch', () => {
  it('does not hatch during a long quiet tool call while active_list still reports working', async () => {
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => undefined)
    const states: ClientSessionState[] = []
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.active_list') {
        return { sessions: [{ session_id: SID, status: 'working' }] }
      }

      return { messages: [], last_id: 0 }
    })

    render(
      createElement(Harness, {
        busy: true,
        initialMessages: [message('user-temp', 'user'), { ...message('assistant-stream-1', 'assistant'), pending: true }],
        onState: state => states.push(state),
        requestGateway
      })
    )

    await act(async () => {
      vi.advanceTimersByTime(90_000)
      await Promise.resolve()
    })

    const probes = requestGateway.mock.calls.filter(([method]) => method === 'session.active_list')
    const polls = requestGateway.mock.calls.filter(([method]) => method === 'session.changes')

    expect(probes.length).toBeLessThanOrEqual(3)
    expect(polls).toHaveLength(0)
    expect(states.some(state => state.messages.length === 0)).toBe(false)
    expect(debug).toHaveBeenCalledWith('session.changes watchdog probe', { sessionId: SID })
  })

  it('runs the refocus hatch before polling so unstamped optimistic rows cannot dedupe against committed ids', async () => {
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => undefined)
    const states: ClientSessionState[] = []
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.changes') {
        return {
          last_id: 2,
          messages: [
            { id: 1, role: 'user', content: 'own user' },
            { id: 2, role: 'assistant', content: 'own assistant' }
          ]
        }
      }

      return { sessions: [] }
    })

    render(
      createElement(Harness, {
        busy: true,
        initialMessages: [message('user-temp', 'user'), { ...message('assistant-stream-1', 'assistant'), pending: true }],
        onState: state => states.push(state),
        requestGateway
      })
    )

    act(() => {
      window.dispatchEvent(new Event('blur'))
    })

    await act(async () => {
      window.dispatchEvent(new Event('focus'))
      vi.advanceTimersByTime(1_000)
      await Promise.resolve()
    })

    expect(debug).toHaveBeenCalledWith('session.changes hatch fired', { trigger: 'refocus' })
    expect(states[0]?.messages).toEqual([])
    expect(requestGateway.mock.calls.find(([method]) => method === 'session.changes')).toBeTruthy()
  })

  it('discards only transcript rows, leaving queued composer entries out of hatch scope', () => {
    const queuedChip = { id: 'queue-chip-1', role: 'system', parts: [{ type: 'text', text: 'queued' }] } as ChatMessage
    const remaining = discardUnstampedOptimisticTranscriptRows(
      [message('user-temp', 'user'), queuedChip],
      new Set(['user-temp'])
    )

    expect(remaining).toEqual([queuedChip])
  })
})

describe('reconnect-seam zombie optimistic rows (severed message.complete stamp)', () => {
  // Live incident 2026-07-15: a backend restart severed the message.complete
  // frame, so an optimistic assistant row (`assistant-stream-<ts>`) was never
  // stamped to its committed id. The poll then re-fetched the committed row
  // and — with an id-only dedupe — painted it as a DUPLICATE beside the
  // zombie. DB clean; pure render corruption. These drive the REAL
  // appendFetchedMessages/dropZombieOptimisticRows, no mocks.

  function textMessage(id: string, role: ChatMessage['role'], text: string, pending = false): ChatMessage {
    return { id, role, pending, parts: [{ type: 'text', text }] }
  }

  it('drops the zombie when its committed twin arrives via the poll (no double paint)', () => {
    const current = [
      textMessage('100', 'user', 'hello'),
      textMessage('assistant-stream-1784172446416', 'assistant', 'the reply')
    ]
    const result = appendFetchedMessages(current, [
      { id: 101, role: 'assistant', content: 'the reply' }
    ])

    const texts = result.messages.map(row => row.parts.map(p => (p as { text?: string }).text ?? '').join(''))
    expect(texts.filter(t => t === 'the reply')).toHaveLength(1)
    expect(result.messages.map(row => row.id)).toEqual(['100', '101'])
  })

  it('drops a zombie USER row too (user optimistic id, committed twin polled)', () => {
    const current = [
      textMessage('user-1784173429344-vh9u3f', 'user', 'i see the footer!!!')
    ]
    const result = appendFetchedMessages(current, [
      { id: 200, role: 'user', content: 'i see the footer!!!' }
    ])

    expect(result.messages.map(row => row.id)).toEqual(['200'])
  })

  it('never drops a PENDING (actively streaming) optimistic row', () => {
    const current = [textMessage('assistant-stream-1', 'assistant', 'partial text', true)]
    const result = appendFetchedMessages(current, [
      { id: 300, role: 'assistant', content: 'partial text' }
    ])

    // orderCommittedMessages (pre-existing) sorts optimistic rows after committed
    // ones; the contract here is SURVIVAL of the pending row, not its position.
    expect(result.messages.map(row => row.id).sort()).toEqual(['300', 'assistant-stream-1'])
  })

  it('never drops committed rows and never over-collapses distinct content', () => {
    const current = [
      textMessage('400', 'assistant', 'same words'),
      textMessage('assistant-stream-2', 'assistant', 'different words')
    ]
    const result = appendFetchedMessages(current, [
      { id: 401, role: 'assistant', content: 'same words' }
    ])

    // committed 400 untouched; zombie has different content so it stays
    // (position governed by pre-existing orderCommittedMessages, assert survival)
    expect(result.messages.map(row => row.id).sort()).toEqual(['400', '401', 'assistant-stream-2'])
  })

  it('one incoming row consumes at most ONE zombie (genuine repeats survive)', () => {
    const zombies = [
      textMessage('user-a', 'user', 'continue'),
      textMessage('user-b', 'user', 'continue')
    ]
    const remaining = dropZombieOptimisticRows(zombies, [textMessage('500', 'user', 'continue')])

    expect(remaining.messages.map(row => row.id)).toEqual(['user-b'])
  })


  it('transplants the zombie footer onto the adopting committed twin', () => {
    // The runtime footer travels ONLY on the message.complete frame and lives
    // on the streamed optimistic row; DB rows never carry it. Sweeping a
    // footer-bearing zombie must NOT lose the footer (live regression
    // 2026-07-16: footer visible live, gone after the poll adopted the twin).
    const zombie = { ...textMessage('assistant-stream-7', 'assistant', 'the reply'), footer: 'model · 5% · 3s' }
    const result = appendFetchedMessages([zombie], [
      { id: 800, role: 'assistant', content: 'the reply' }
    ])

    expect(result.messages.map(row => row.id)).toEqual(['800'])
    expect(result.messages[0].footer).toBe('model · 5% · 3s')
  })

  it('does not graft a footer onto a committed row that already has one', () => {
    const zombie = { ...textMessage('assistant-stream-8', 'assistant', 'x'), footer: 'zombie-footer' }
    const committed = { ...textMessage('900', 'assistant', 'x'), footer: 'own-footer' }
    const adopted = adoptZombieFooters([committed], new Map([['assistant\u0000x', 'zombie-footer']]))

    expect(adopted[0].footer).toBe('own-footer')
    void zombie
  })

  it('role must match for a zombie drop (same text, different role stays)', () => {
    const zombies = [textMessage('assistant-stream-3', 'assistant', 'ok')]
    const remaining = dropZombieOptimisticRows(zombies, [textMessage('600', 'user', 'ok')])

    expect(remaining.messages.map(row => row.id)).toEqual(['assistant-stream-3'])
  })

  it('MEDIA-tag edge: raw streamed zombie matches its media-rendered committed twin', () => {
    // Committed rows pass through assistantTextPart -> renderMediaTags; the
    // streamed zombie may hold the raw MEDIA: line. The join key normalizes
    // both sides so the zombie still drops (Greptile #361 P2).
    const zombie = textMessage('assistant-stream-9', 'assistant', 'here you go\nMEDIA:/tmp/pic.png')
    const committed = appendFetchedMessages([zombie], [
      { id: 700, role: 'assistant', content: 'here you go\nMEDIA:/tmp/pic.png' }
    ])

    expect(committed.messages.map(row => row.id)).toEqual(['700'])
  })
})

