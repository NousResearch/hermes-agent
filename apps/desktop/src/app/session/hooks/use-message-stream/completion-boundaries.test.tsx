import type { GatewayEventName } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import {
  clearInFlightTurnJournal,
  readInFlightTurnJournal,
  recoverInFlightTurnJournal
} from '@/lib/inflight-turn-journal'
import { $messages, setActiveSessionId, setMessages } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { usePromptActions } from '../use-prompt-actions'
import { useSessionStateCache } from '../use-session-state-cache'

import { useMessageStream } from './index'

const SID = 'completion-runtime'
const STORED = 'completion-stored'
const ANSWER = 'The answer is unchanged.'
const noop = async () => undefined

const requestGateway = async <T,>(method: string): Promise<T> =>
  (method === 'prompt.submit' ? { status: 'started' } : { status: 'redirected' }) as T

// Real submit/redirect, stream reducer, cache and view publication; only RPC
// acceptance and history/metadata I/O are stand-ins.
function mount() {
  const queryClient = new QueryClient()
  const hydrate = vi.fn(noop)

  const hook = renderHook(() => {
    const busyRef = useRef(false)

    const cache = useSessionStateCache({
      activeSessionId: SID,
      selectedStoredSessionId: STORED,
      busyRef,
      setAwaitingResponse: () => undefined,
      setBusy: value => {
        busyRef.current = value
      },
      setMessages
    })

    const stream = useMessageStream({
      activeSessionIdRef: cache.activeSessionIdRef,
      sessionStateByRuntimeIdRef: cache.sessionStateByRuntimeIdRef,
      updateSessionState: cache.updateSessionState,
      queryClient,
      hydrateFromStoredSession: hydrate,
      refreshHermesConfig: noop,
      refreshSessions: noop
    })

    const actions = usePromptActions({
      activeSessionId: SID,
      activeSessionIdRef: cache.activeSessionIdRef,
      branchCurrentSession: async () => true,
      busyRef,
      createBackendSessionForSend: async () => SID,
      getRoutedStoredSessionId: () => STORED,
      getRuntimeIdForStoredSession: () => SID,
      getRouteToken: () => 'completion-route',
      handleSkinCommand: () => '',
      openMemoryGraph: () => undefined,
      refreshSessions: noop,
      requestGateway,
      resumeStoredSession: noop,
      runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
      selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
      startFreshSessionDraft: () => undefined,
      sttEnabled: false,
      updateSessionState: cache.updateSessionState
    })

    return { cache, stream, actions }
  })

  act(() => hook.result.current.cache.ensureSessionState(SID, STORED))

  return {
    hydrate,
    state: () => hook.result.current.cache.sessionStateByRuntimeIdRef.current.get(SID)!,
    send: (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
      act(() => hook.result.current.stream.handleGatewayEvent({ type, payload, session_id: SID })),
    submit: () =>
      act(async () => {
        expect(await hook.result.current.actions.submitText('Give the answer.', { attachments: [] })).toBe(true)
      }),
    redirect: () =>
      act(async () => {
        expect(await hook.result.current.actions.redirectPrompt('Revise the answer.')).toBe(true)
      }),
    update: (updater: Parameters<typeof hook.result.current.cache.updateSessionState>[1]) =>
      act(() => {
        hook.result.current.cache.updateSessionState(SID, updater)
      }),
    dispose: () => {
      hook.unmount()
      queryClient.clear()
    }
  }
}

const timeline = (messages: ChatMessage[]) => messages.map(message => [message.role, chatMessageText(message)])

const flush = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(50)
  })

beforeEach(() => {
  vi.useFakeTimers()
  localStorage.clear()
  $sessionStates.set({})
  setMessages([])
  setActiveSessionId(SID)
})
afterEach(() => {
  cleanup()
  clearInFlightTurnJournal(STORED)
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it.each([false, true].flatMap(previousFailure => [ANSWER, ''].map(reply => ({ previousFailure, reply }))))(
  'keeps a new completion after a real second submit (previousFailure=$previousFailure, reply="$reply")',
  async ({ previousFailure, reply }) => {
    const h = mount()
    await h.submit()
    await h.send('message.start')
    await h.send('message.delta', { text: ANSWER })
    await h.send('message.complete', {
      text: ANSWER,
      ...(previousFailure ? { status: 'error', error: 'Interrupted provider', partial: true } : {})
    })
    const firstAnswer = h.state().messages.at(-1)!
    expect(h.hydrate).not.toHaveBeenCalled()
    await flush()
    await h.submit() // Even identical prompt text is a new occurrence.
    await h.send('message.start')
    await h.send('message.complete', { text: reply })

    const expected = [
      ['user', 'Give the answer.'],
      ['assistant', ANSWER],
      ['user', 'Give the answer.'],
      ...(reply ? [['assistant', reply]] : [])
    ]

    expect(timeline(h.state().messages)).toEqual(expected)
    expect(timeline($messages.get())).toEqual(expected)
    expect(h.state().messages.find(message => message.id === firstAnswer.id)).toEqual(firstAnswer)
    expect(h.hydrate).toHaveBeenCalledWith(3, STORED, SID)

    if (reply) {
      const ids = h.state().messages.map(message => message.id)
      await h.send('message.complete', { text: reply })
      expect(h.state().messages.map(message => message.id)).toEqual(ids)
      expect(timeline($messages.get())).toEqual(expected)
    }

    h.dispose()
  }
)

it.each([
  { before: 'Answer', after: 'Answer with details' },
  { before: 'Answer with details', after: 'Answer' },
  { before: 'Answer', after: 'Rewritten reply', previewed: true },
  { before: 'Answer', after: 'Answer with details', hidden: true },
  { before: 'Answer', after: 'Answer with details', staleStream: true }
])('keeps redirect boundaries and settles duplicates: $before → $after ($hidden, $staleStream)', async fixture => {
  const h = mount()
  await h.submit()
  await h.send('message.start')
  await h.send('message.delta', { text: fixture.before })
  await h.send('message.interim', { text: fixture.before, already_streamed: true })
  expect(timeline(h.state().messages)).toEqual([
    ['user', 'Give the answer.'],
    ['assistant', fixture.before]
  ])
  const before = h.state().messages.at(-1)!
  await flush()
  await h.redirect()

  // Visibility is not identity; a hidden user row still divides occurrences.
  // A stale stream reference must not authorize completion across that row.
  if (fixture.hidden || fixture.staleStream) {
    h.update(state => ({
      ...state,
      streamId: fixture.staleStream ? before.id : state.streamId,
      messages: fixture.hidden
        ? state.messages.map(message => (message === state.messages.at(-1) ? { ...message, hidden: true } : message))
        : state.messages
    }))
  }

  await h.send('message.complete', { text: fixture.after, response_previewed: fixture.previewed })

  const expected = [
    ['user', 'Give the answer.'],
    ['assistant', fixture.before],
    ['user', 'Revise the answer.'],
    ['assistant', fixture.after]
  ]

  expect(timeline(h.state().messages)).toEqual(expected)
  expect(h.state().messages.find(message => message.id === before.id)).toEqual(before)
  expect(h.state().messages.at(-1)?.interim).toBeFalsy()
  const ids = h.state().messages.map(message => message.id)
  await h.send('message.complete', { text: fixture.after, response_previewed: fixture.previewed })
  expect(h.state().messages.map(message => message.id)).toEqual(ids)
  expect(timeline($messages.get())).toEqual(expected)
  h.dispose()
})

// A completion whose text IS the sealed pre-redirect reply, with nothing
// streamed since, is that reply's own completion (rejected redirect race, or a
// steer the model absorbed without new output): it settles the seal and the
// correction stays the tail — see steer-arrival-order.test.tsx for the
// streaming-side tripwire.
it('settles an equal no-delta completion onto the sealed pre-redirect reply', async () => {
  const h = mount()
  await h.submit()
  await h.send('message.start')
  await h.send('message.delta', { text: 'Answer' })
  await h.send('message.interim', { text: 'Answer', already_streamed: true })
  await flush()
  await h.redirect()
  await h.send('message.complete', { text: 'Answer' })

  expect(timeline(h.state().messages)).toEqual([
    ['user', 'Give the answer.'],
    ['assistant', 'Answer'],
    ['user', 'Revise the answer.']
  ])
  expect(h.state().messages[1].interim).toBeFalsy()
  h.dispose()
})

it.each(['fallback', 'stream', 'tool-interim'] as const)(
  'retires recovered metadata when completion settles the %s row',
  async path => {
    const h = mount()
    await h.submit()
    await h.send('message.start')
    await h.send('message.delta', { text: ANSWER })

    if (path === 'tool-interim') {
      await h.send('message.interim', { text: ANSWER, already_streamed: true })
    }

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(readInFlightTurnJournal(STORED)).not.toBeNull()
    const recovered = recoverInFlightTurnJournal(STORED, h.state().messages.slice(0, 1), { keepPending: false })
    expect(recovered.applied).toBe(true)
    const recoveredId = recovered.messages.at(-1)!.id
    h.update(state => ({
      ...state,
      busy: false,
      awaitingResponse: false,
      turnLive: false,
      streamId: path === 'stream' ? recoveredId : null,
      messages: recovered.messages
    }))
    expect(h.state().messages.at(-1)).toMatchObject({ recovered: true, pending: false })

    if (path === 'tool-interim') {
      await h.send('tool.start', { name: 'terminal', tool_id: 'after-recovery' })
      await h.send('tool.complete', { name: 'terminal', tool_id: 'after-recovery', result: 'ok' })
    }

    await h.send('message.complete', { text: ANSWER })
    expect(h.state().messages).toHaveLength(2)
    expect(h.state().messages.at(-1)).toMatchObject({ id: recoveredId, recovered: false, pending: false })
    expect(chatMessageText(h.state().messages.at(-1)!)).toBe(ANSWER)
    expect(readInFlightTurnJournal(STORED)).toBeNull()
    expect(h.state()).toMatchObject({ busy: false, awaitingResponse: false, streamId: null })
    h.dispose()
  }
)
