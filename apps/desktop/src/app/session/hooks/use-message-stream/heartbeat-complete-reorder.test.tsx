import type { GatewayEventName } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { clearInFlightTurnJournal } from '@/lib/inflight-turn-journal'
import { $messages, setActiveSessionId, setMessages } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { usePromptActions } from '../use-prompt-actions'
import { useSessionStateCache } from '../use-session-state-cache'

import { useMessageStream } from './index'

const SID = 'race119569-runtime'
const STORED = 'race119569-stored'
const noop = async () => undefined

// The canonical admission receipt echoes the client's submission id.
const requestGateway = async <T,>(method: string, params?: Record<string, unknown>): Promise<T> =>
  (method === 'prompt.submit'
    ? { admission_id: params?.submission_id, session_id: SID, status: 'started' }
    : { status: 'redirected' }) as T

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
      getRouteToken: () => 'race-route',
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
        expect(await hook.result.current.actions.submitText('Search flights.', { attachments: [] })).toBe(true)
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

const settle = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(5000)
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

const FULL = 'Voos de Porto Alegre para Sao Paulo: opcao 1 ... opcao 2 ... (long reply)'

async function toolHeavyTurn(h: ReturnType<typeof mount>) {
  await h.submit()
  await h.send('message.start')
  await h.send('message.delta', { text: 'Searching flights ' })
  await h.send('tool.start', { name: 'mcp__kiwi__search_flight', tool_id: 't1' })
  await h.send('tool.complete', { name: 'mcp__kiwi__search_flight', tool_id: 't1', result: 'ok-1' })
  await h.send('tool.start', { name: 'mcp__kiwi__search_flight', tool_id: 't2' })
  await h.send('tool.complete', { name: 'mcp__kiwi__search_flight', tool_id: 't2', result: 'ok-2' })
  await h.send('tool.start', { name: 'mcp__kiwi__search_flight', tool_id: 't3' })
  await h.send('tool.complete', { name: 'mcp__kiwi__search_flight', tool_id: 't3', result: 'ok-3' })
  await h.send('message.delta', { text: FULL })
  await flush()
}

it('#119569 V1: tool-heavy turn keeps the just-delivered reply across complete', async () => {
  const h = mount()
  await toolHeavyTurn(h)
  await h.send('message.complete', { text: `Searching flights ${FULL}` })
  await settle()

  const texts = timeline(h.state().messages)
  const viewTexts = timeline($messages.get())
  expect(texts.filter(([, t]) => (t as string).includes('Voos de Porto Alegre'))).toHaveLength(1)
  expect(viewTexts).toEqual(texts)
  h.dispose()
})

it('#119569 V2: interim-sealed tool turn keeps both segments across complete', async () => {
  const h = mount()
  await h.submit()
  await h.send('message.start')
  await h.send('message.delta', { text: 'Let me look that up.' })
  await h.send('tool.start', { name: 'mcp__kiwi__search_flight', tool_id: 't1' })
  await h.send('tool.complete', { name: 'mcp__kiwi__search_flight', tool_id: 't1', result: 'ok' })
  await h.send('message.interim', { text: 'Let me look that up.', already_streamed: true })
  await h.send('message.delta', { text: FULL })
  await flush()
  await h.send('message.complete', { text: FULL })
  await settle()

  const texts = timeline(h.state().messages)
  expect(texts.filter(([, t]) => (t as string).includes('Voos de Porto Alegre'))).toHaveLength(1)
  expect(timeline($messages.get())).toEqual(texts)
  h.dispose()
})

it('#119569 V3: running=false heartbeat before complete must not drop the reply', async () => {
  const h = mount()
  await toolHeavyTurn(h)
  // Agent-loop finally-block heartbeat wins the race against the terminal frame.
  await h.send('session.info', { running: false, stored_session_id: STORED })
  await h.send('message.complete', { text: `Searching flights ${FULL}` })
  await settle()

  const texts = timeline(h.state().messages)
  // The reordered heartbeat must not orphan the late complete into a second
  // bubble, and must not fire a stored-history hydrate that can race the
  // gateway commit and drop the just-delivered reply from view (#119569).
  expect(h.hydrate).not.toHaveBeenCalled()
  expect(texts.filter(([, t]) => (t as string).includes('Voos de Porto Alegre'))).toHaveLength(1)
  expect(timeline($messages.get())).toEqual(texts)
  h.dispose()
})

it('#119569 V6: same-ms redelivered complete frames must not key-collide bubbles away', async () => {
  const h = mount()
  await toolHeavyTurn(h)
  await h.send('message.complete', { text: `Searching flights ${FULL}` })
  // Redelivery / retry of the terminal frame inside the same millisecond.
  await h.send('message.complete', { text: `Searching flights ${FULL} (updated)` })
  await settle()

  const ids = h.state().messages.map(m => m.id)
  expect(new Set(ids).size).toBe(ids.length)
  const texts = timeline(h.state().messages)
  expect(texts.filter(([, t]) => (t as string).includes('Voos de Porto Alegre')).length).toBeGreaterThanOrEqual(1)
  expect(timeline($messages.get())).toEqual(texts)
  h.dispose()
})
