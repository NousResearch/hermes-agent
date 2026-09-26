import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import {
  persistInFlightTurnState,
  readInFlightTurnJournal,
  resetInFlightTurnJournalStateForTests
} from '@/lib/inflight-turn-journal'
import { $activeGatewayProfile } from '@/store/profile'
import {
  _resetSessionOwnerHintsForTests,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setConnection,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { useMessageStream } from './use-message-stream'
import { useSessionActions } from './use-session-actions'
import { appendLiveSessionProjection } from './use-session-actions/utils'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))
vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

const storedId = 'resume-occurrences'
const runtimeId = 'resume-runtime'
const prompt = 'Inspect both phases'
const commentary = 'Checking the phase.'
const tail = 'The result is still growing.'
const noop = async () => undefined
const user: SessionMessage = { id: 1, role: 'user', content: prompt, timestamp: 1 }

function history(comments: string[]): SessionMessage[] {
  return [
    { ...user },
    ...comments.flatMap((content, index): SessionMessage[] => [
      {
        id: 2 + index * 2,
        role: 'assistant',
        content,
        timestamp: 2 + index * 2,
        tool_calls: [{ id: `call-${index}`, type: 'function', function: { name: 'read_file', arguments: '{}' } }]
      },
      { id: 3 + index * 2, role: 'tool', content: 'fixture', tool_call_id: `call-${index}`, timestamp: 3 + index * 2 }
    ])
  ]
}

function mount(snapshot: SessionResumeResult) {
  const requestGateway = vi.fn().mockResolvedValue(snapshot)

  const hook = renderHook(() => {
    const busyRef = useRef(false)
    const creatingSessionRef = useRef(false)
    const queryClient = useRef(new QueryClient()).current

    const cache = useSessionStateCache({
      activeSessionId: null,
      selectedStoredSessionId: null,
      busyRef,
      setMessages,
      setBusy,
      setAwaitingResponse
    })

    const actions = useSessionActions({
      ...cache,
      activeSessionId: null,
      selectedStoredSessionId: null,
      busyRef,
      creatingSessionRef,
      getRouteToken: () => 'A',
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn(),
      requestGateway
    })

    const stream = useMessageStream({
      ...cache,
      queryClient,
      hydrateFromStoredSession: noop,
      refreshHermesConfig: noop,
      refreshSessions: noop
    })

    return { cache, actions, stream }
  })

  return { ...hook, requestGateway }
}

beforeEach(() => {
  vi.useFakeTimers()
  localStorage.clear()
  clearAllSessionStates()
  resetInFlightTurnJournalStateForTests()
  _resetSessionOwnerHintsForTests()
  $activeGatewayProfile.set('default')
  setConnection(null)
  setMessages([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setBusy(false)
  setAwaitingResponse(false)
  setSessions([
    {
      id: storedId,
      title: storedId,
      source: 'desktop',
      message_count: 5,
      tool_call_count: 2,
      is_active: false,
      started_at: 1,
      last_active: 1,
      ended_at: null,
      model: null,
      preview: null,
      input_tokens: 0,
      output_tokens: 0
    }
  ])
  vi.mocked(getLatestSessionMessages).mockReset()
})
afterEach(() => {
  cleanup()
  clearAllSessionStates()
  resetInFlightTurnJournalStateForTests()
  localStorage.clear()
  setSessions([])
  setMessages([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setBusy(false)
  setAwaitingResponse(false)
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it.each([
  ['cold', true, 'equal'],
  ['sparse', true, 'equal'],
  ['warm', true, 'equal'],
  ['warm', false, 'equal'],
  ['warm', false, 'snapshot-ahead'],
  ['warm', false, 'local-ahead'],
  ['cold', false, 'missing-ids'],
  ['warm', false, 'delta-during-history']
] as const)(
  'reconciles ordered tool-round occurrences, not assistant ordinals (%s, repeated: %s, %s)',
  async (cacheKind, repeated, race) => {
    // Equal commentary is intentional; the producer suppresses its second interim,
    // not its second delta. Distinct commentary exercises the other live row shape.
    const comments = [commentary, repeated ? commentary : 'Checking another phase.']

    const snapshot: SessionResumeResult = {
      session_id: runtimeId,
      resumed: storedId,
      messages: [],
      message_count: 0,
      running: true,
      inflight: {
        user: prompt,
        assistant: [...comments, race === 'local-ahead' ? tail.slice(0, 10) : tail].join('\n\n'),
        streaming: true
      }
    }

    const { result } = mount(snapshot)

    if (cacheKind !== 'cold') {
      act(() => {
        result.current.cache.activeSessionIdRef.current = runtimeId
        result.current.cache.selectedStoredSessionIdRef.current = storedId
        result.current.cache.updateSessionState(
          runtimeId,
          state => ({ ...state, messages: toChatMessages([user]) }),
          storedId
        )
      })
    }

    if (cacheKind === 'warm') {
      const send = (type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) =>
        act(() => result.current.stream.handleGatewayEvent({ session_id: runtimeId, type, payload }))

      send('message.start')
      comments.forEach((text, index) => {
        send('message.delta', { text: `${index ? '\n\n' : ''}${text}` })

        if (!index || !repeated) {
          send('message.interim', { text, already_streamed: true })
        }

        send('tool.start', { name: 'read_file', tool_id: `call-${index}`, args: {} })
        send('tool.complete', { name: 'read_file', tool_id: `call-${index}`, result: 'fixture' })
      })
      send('message.delta', { text: `\n\n${race === 'snapshot-ahead' ? 'The res' : tail}` })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(150)
      })
    }

    const durable = history(comments)

    if (race === 'missing-ids') {
      durable.forEach(row => {
        delete row.id
      })
    }

    let release!: () => void

    const held = new Promise<Awaited<ReturnType<typeof getLatestSessionMessages>>>(resolve => {
      release = () => resolve({ session_id: storedId, messages: durable })
    })

    vi.mocked(getLatestSessionMessages).mockReturnValue(held)
    let pending!: Promise<void>
    await act(async () => {
      pending = result.current.actions.resumeSession(storedId, true)
    })

    if (race === 'delta-during-history') {
      expect(getLatestSessionMessages).toHaveBeenCalled()
      act(() =>
        result.current.stream.handleGatewayEvent({
          session_id: runtimeId,
          type: 'message.delta',
          payload: { text: ' More.' }
        })
      )
      await act(async () => {
        await vi.advanceTimersByTimeAsync(150)
      })
    }

    await act(async () => {
      release()
      await pending
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(150)
    })
    const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages

    const displayed = rows
      .filter(row => row.role === 'assistant')
      .map(chatMessageText)
      .join('\n\n')
      .replace(/\s+/g, '')

    // Missing row identity cannot prove coverage: prefer a duplicate over silently
    // consuming an accepted live occurrence. The live tail must still be visible.
    if (race === 'missing-ids') {
      expect(displayed).toContain(tail.replace(/\s+/g, ''))
    } else {
      expect(displayed).toBe(
        [...comments, tail, race === 'delta-during-history' ? ' More.' : ''].join('').replace(/\s+/g, '')
      )
    }

    expect(
      rows.flatMap(row => row.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : [])))
    ).toEqual(comments.map((_, index) => `call-${index}`))

    if (race !== 'missing-ids') {
      // Activation feeds its previous projection back into the warm cache.
      // Replaying an unchanged snapshot must not create another occurrence.
      for (let resume = 0; resume < 2; resume++) {
        await act(async () => {
          await result.current.actions.resumeSession(storedId, true)
        })
        const repeatedRows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages

        expect(repeatedRows.filter(row => row.role === 'assistant').map(chatMessageText)).toEqual(
          rows.filter(row => row.role === 'assistant').map(chatMessageText)
        )
        expect(
          repeatedRows.flatMap(row => row.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : [])))
        ).toEqual(comments.map((_, index) => `call-${index}`))
        expect(new Set(repeatedRows.map(row => row.id)).size).toBe(repeatedRows.length)
      }
    }
  }
)

it.each(['snapshot-ahead', 'history-ahead', 'unpersisted-before-correction'] as const)(
  'uses raw producer offsets and consumes corrections by occurrence (%s)',
  async race => {
    const media = 'An artifact 📄: MEDIA:/workspace/report.pdf'
    const beforeCorrection = race === 'unpersisted-before-correction' ? '\n\nBefore the correction.' : ''
    const correction = prompt // A correction may deliberately repeat the prompt.
    const raw = [media + beforeCorrection, 'Following the correction.'].join('\n\n')

    const snapshot: SessionResumeResult = {
      session_id: runtimeId,
      resumed: storedId,
      messages: [],
      message_count: 0,
      running: true,
      inflight: {
        user: prompt,
        assistant: race === 'history-ahead' ? media : `${raw}\n\n${tail}`,
        corrections: [correction, correction],
        correction_offsets: [Array.from(media + beforeCorrection).length, Array.from(raw).length],
        streaming: true
      }
    }

    // The second accepted equal correction is not yet durable.
    const durable = history([media])
    durable.push({ id: 4, role: 'user', content: correction, display_kind: 'steer', timestamp: 4 })

    if (race === 'history-ahead') {
      durable.push({
        id: 5,
        role: 'assistant',
        content: 'Following the correction.',
        timestamp: 5,
        tool_calls: [{ id: 'later-call', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
      })
    }

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: durable })
    const { result } = mount(snapshot)
    await act(async () => {
      await result.current.actions.resumeSession(storedId, true)
    })
    const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages

    expect(rows.filter(row => row.role === 'user').map(chatMessageText)).toEqual([prompt, correction, correction])

    const texts = rows
      .filter(row => row.role === 'assistant')
      .map(chatMessageText)
      .join('\n')

    expect(texts.match(/An artifact 📄:/g)).toHaveLength(1)
    expect(texts).toContain('Following the correction.')

    if (race === 'snapshot-ahead') {
      expect(texts).toContain(tail)
    }

    if (beforeCorrection) {
      const beforeIndex = rows.findIndex(row => chatMessageText(row).includes(beforeCorrection.trim()))
      const correctionIndex = rows.findIndex(row => row.rowId === 4)
      expect(beforeIndex).toBeLessThan(correctionIndex)
    }
  }
)

it.each([prompt, 'Inspect the next file'])(
  'projects the accepted next-turn queue once across cold and repeated warm resume (%s)',
  async queued => {
    const durable = history([commentary])

    const snapshot: SessionResumeResult = {
      session_id: runtimeId,
      resumed: storedId,
      messages: [],
      message_count: 0,
      running: true,
      inflight: { user: prompt, assistant: `${commentary}\n\n${tail}`, streaming: true },
      queued: { user: queued }
    }

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: durable })

    const { result } = mount(snapshot)

    for (let resume = 0; resume < 3; resume++) {
      if (resume === 2) {
        // Later text-only arrivals extend the same backend queue slot.
        snapshot.queued!.user = `${queued}\n\nThen inspect the README`
      }

      await act(async () => {
        await result.current.actions.resumeSession(storedId, true)
      })
      const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages

      expect(rows.filter(row => row.role === 'user').map(chatMessageText)).toEqual([prompt, snapshot.queued!.user])
      expect(new Set(rows.map(row => row.id)).size).toBe(rows.length)
      expect(
        rows.flatMap(row => row.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : [])))
      ).toEqual(['call-0'])
    }
  }
)

it('settles a retained idle error once across repeated resume and a changed error snapshot', async () => {
  const partial = 'The partial result'

  const snapshot: SessionResumeResult = {
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    message_count: 0,
    running: false,
    inflight: {
      user: prompt,
      assistant: partial,
      streaming: false,
      status: 'error',
      recoverable: true,
      error: 'Connection reset',
      error_surface: { layer: 'streaming', code: 'stream_drop', retryable: true }
    }
  }

  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: [user] })

  const { result } = mount(snapshot)

  for (let resume = 0; resume < 3; resume++) {
    if (resume === 2) {
      snapshot.inflight!.error = 'Upstream timed out'
      snapshot.inflight!.error_surface = { layer: 'provider', code: 'timeout', retryable: true }
    }

    await act(async () => {
      await result.current.actions.resumeSession(storedId, true)
    })
    const state = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!

    expect(state.messages.filter(row => row.role === 'assistant').map(chatMessageText)).toEqual([partial])
    expect(state.messages.filter(row => row.error)).toHaveLength(1)
    expect(state.messages.at(-1)).toMatchObject({
      error: snapshot.inflight!.error,
      errorSurface: snapshot.inflight!.error_surface,
      pending: false
    })
    expect(new Set(state.messages.map(row => row.id)).size).toBe(state.messages.length)
    expect(state.busy).toBe(false)
    expect(state.awaitingResponse).toBe(false)
  }
})

it('keeps runtime provenance when a visible wake has persisted tool commentary', async () => {
  const durable = history([commentary])
  durable[0] = { ...user, display_kind: 'internal_notification', user_originated: false }

  const snapshot: SessionResumeResult = {
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    message_count: 0,
    running: true,
    turn_started_at: 0.5,
    queued: { user: 'Queued human prompt' },
    inflight: {
      user: prompt,
      user_originated: false,
      display_kind: 'internal_notification',
      assistant: `${commentary}\n\n${tail}`,
      streaming: true
    }
  }

  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: durable })
  const { result } = mount(snapshot)

  for (let resume = 0; resume < 2; resume++) {
    await act(async () => {
      await result.current.actions.resumeSession(storedId, true)
    })
    const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages
    const notice = rows.find(row => row.rowId === user.id)!
    expect(notice).toMatchObject({ userOriginated: false, runtimeTurnStartedAt: snapshot.turn_started_at })
    expect(rows.filter(row => chatMessageText(row) === prompt)).toHaveLength(1)
    expect(rows.filter(row => chatMessageText(row).trim() === tail)).toHaveLength(1)
    expect(rows.find(row => row.id === `user-queued-${runtimeId}`)?.runtimeTurnStartedAt).toBeUndefined()
    expect(
      rows.filter(row => row.role === 'assistant').every(row => row.runtimeTurnStartedAt === snapshot.turn_started_at)
    ).toBe(true)
    expect(new Set(rows.map(row => row.id)).size).toBe(rows.length)
  }

  // A later human can deliberately submit the same text as that runtime wake.
  // The older runtime's durable prompt must not absorb the accepted human row.
  snapshot.turn_started_at = 10
  delete snapshot.queued
  snapshot.inflight = { user: prompt, user_originated: true, assistant: 'Human response', streaming: true }
  await act(async () => {
    await result.current.actions.resumeSession(storedId, true)
  })
  const rows = result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages
  expect(rows.filter(row => row.role === 'user' && row.userOriginated === true).map(chatMessageText)).toEqual([prompt])
})

it.each([1, 2])('consumes %i durable runtime tool occurrences before recovering a cold journal', async rounds => {
  const comments = Array.from({ length: rounds }, () => commentary)
  const durable = history(comments)
  durable[0] = { ...user, display_kind: 'internal_notification', user_originated: false }

  const snapshot: SessionResumeResult = {
    session_id: runtimeId,
    resumed: storedId,
    messages: [],
    message_count: 0,
    running: true,
    turn_started_at: 0.5,
    inflight: {
      user: prompt,
      user_originated: false,
      display_kind: 'internal_notification',
      assistant: '',
      streaming: true
    }
  }

  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: [durable[0]] })
  const live = mount(snapshot)
  await act(async () => {
    await live.result.current.actions.resumeSession(storedId, true)
  })

  const send = (type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) =>
    act(() => live.result.current.stream.handleGatewayEvent({ session_id: runtimeId, type, payload }))

  send('message.start')
  comments.forEach((text, index) => {
    send('message.delta', { text: `${index ? '\n\n' : ''}${text}` })

    if (!index) {
      send('message.interim', { text, already_streamed: true })
    }

    send('tool.start', { name: 'read_file', tool_id: `call-${index}`, args: {} })
    send('tool.complete', { name: 'read_file', tool_id: `call-${index}`, result: 'fixture' })
  })
  send('message.delta', { text: `\n\n${tail}` })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(150)
  })
  persistInFlightTurnState(live.result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(500)
  })
  expect(readInFlightTurnJournal(storedId)?.messages.some(row => row.runtimeTurnStartedAt === 0.5)).toBe(true)

  live.unmount()
  clearAllSessionStates()
  resetInFlightTurnJournalStateForTests()
  setMessages([])
  snapshot.inflight!.assistant = [...comments, tail].join('\n\n')
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: durable })
  const recovered = mount(snapshot)

  for (let resume = 0; resume < 2; resume++) {
    await act(async () => {
      await recovered.result.current.actions.resumeSession(storedId, true)
    })
    const rows = recovered.result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!.messages
    expect(rows.filter(row => row.role === 'assistant').map(chatMessageText).join('').replace(/\s+/g, ''))
      .toBe([...comments, tail].join('').replace(/\s+/g, ''))
    expect(rows.flatMap(row => row.parts.flatMap(part => part.type === 'tool-call' ? [part.toolCallId] : [])))
      .toEqual(comments.map((_, index) => `call-${index}`))
    expect(new Set(rows.map(row => row.id)).size).toBe(rows.length)
  }
})

it.each([false, true].flatMap(coalesced => [false, true].flatMap(repeated =>
  [0, 1, 2, 3].map(persisted => ({ coalesced, repeated, persisted }))
)))('retains accepted runtime humans across two cold cache commits (%j)', async ({ coalesced, repeated, persisted }) => {
  const humans = ['Preserve the source files', repeated ? 'Preserve the source files' : 'Inspect the tests', 'Then inspect the logs']
  const durable = history([commentary])
  durable[0] = { ...user, display_kind: 'internal_notification', user_originated: false }
  durable.unshift(
    { id: 10, role: 'user', content: 'Earlier task', timestamp: -2 },
    { id: 11, role: 'assistant', content: 'Earlier answer', timestamp: -1 }
  )

  const snapshot: SessionResumeResult = {
    session_id: runtimeId, resumed: storedId, messages: [], message_count: 0, running: true, turn_started_at: 0.5,
    inflight: { user: prompt, user_originated: false, display_kind: 'internal_notification',
      assistant: commentary, streaming: true, corrections: humans.slice(0, 2),
      correction_offsets: [commentary.length, commentary.length] },
    queued: { user: humans[2] }
  }

  const messages = appendLiveSessionProjection(toChatMessages(durable), snapshot)
  const streamId = messages.findLast(row => row.role === 'assistant')!.id
  const live = mount(snapshot)
  act(() => live.result.current.cache.updateSessionState(runtimeId, state => ({
    ...state, messages, streamId, busy: true, awaitingResponse: false, turnStartedAt: 500
  }), storedId))
  act(() => vi.advanceTimersByTime(400))

  const humanTexts = (rows: typeof messages) => rows.filter(row => row.role === 'user' && row.userOriginated !== false)
    .map(chatMessageText)

  expect(humanTexts(readInFlightTurnJournal(storedId)!.messages)).toEqual(humans)
  live.unmount()

  const corrections = humans.slice(0, Math.min(persisted, 2))

  const durableHumans = coalesced && corrections.length
    ? [
        '[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered once at this position; not tool output and not a new delivery when replayed from conversation history]\n' +
        corrections.join('\n') + '\n[/OUT-OF-BAND USER MESSAGE]',
        ...humans.slice(corrections.length, persisted)
      ]
    : humans.slice(0, persisted)

  durable.push(...durableHumans.map((content, index): SessionMessage => ({
    id: 20 + index, role: 'user', content, user_originated: true, timestamp: 20 + index,
    ...(coalesced && corrections.length && index === 0 ? { display_kind: 'steer' as const } : {})
  })))
  const expectedHumans = [...durableHumans, ...humans.slice(persisted)]
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: storedId, messages: durable })

  for (let reload = 0; reload < 2; reload++) {
    clearAllSessionStates()
    resetInFlightTurnJournalStateForTests()
    setMessages([])

    const cold = mount({ session_id: runtimeId, resumed: storedId, messages: [], message_count: durable.length,
      messages_omitted: true, running: false })

    await act(async () => { await cold.result.current.actions.resumeSession(storedId, true) })
    act(() => vi.advanceTimersByTime(400))
    const state = cold.result.current.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)!
    expect(humanTexts(state.messages)).toEqual(['Earlier task', ...expectedHumans])
    expect(state.messages.map(chatMessageText)).toEqual(['Earlier task', 'Earlier answer', prompt, commentary, ...expectedHumans])
    expect(state.messages.flatMap(row => row.parts.filter(part => part.type === 'tool-call').map(part => part.toolCallId)))
      .toEqual(['call-0'])
    expect(state.busy).toBe(false)
    expect(state.awaitingResponse).toBe(false)
    expect(state.streamId).toBeNull()
    expect(readInFlightTurnJournal(storedId) !== null).toBe(persisted < humans.length)
    cold.unmount()
  }
})
