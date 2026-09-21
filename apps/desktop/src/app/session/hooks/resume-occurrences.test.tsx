import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { resetInFlightTurnJournalStateForTests } from '@/lib/inflight-turn-journal'
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
      send('message.delta', { text: `\n\n${race === 'snapshot-ahead' ? tail.slice(0, 10) : tail}` })
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
