import { useStore } from '@nanostores/react'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { sessionRoute } from '@/app/routes'
import { textPart } from '@/lib/chat-messages'
import { requestGatewayForAgent, requestGatewayForProfile } from '@/store/gateway'
import {
  $activeSessionId,
  $activeSessionStoredIdRotation,
  $messages,
  $selectedStoredSessionId,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { handleSessionInfoEvent } from '../use-message-stream/gateway-event/session-info'
import { useSessionStateCache } from '../use-session-state-cache'

import { clearSingleFlightSessionResumeState, registerRecoveredRuntime } from './single-flight-resume'

import { usePromptActions } from '.'

// Real prompt hooks, cache, ownership router, and dispatcher; only the external
// gateway edge is substituted. No test injects a corrupted ownership mapping.
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

const busyRef = { current: false }
let routedStoredId: string | null = 'stored-B'
let handle: { actions: ReturnType<typeof usePromptActions>; cache: ReturnType<typeof useSessionStateCache> }

function Harness() {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)

  const cache = useSessionStateCache({
    activeSessionId,
    selectedStoredSessionId,
    busyRef,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const requestGateway = createSessionRpcDispatcher({
    ...cache,
    ambientRequest: async () => {
      throw new Error('unexpected ambient request')
    }
  })

  const actions = usePromptActions({
    activeSessionId,
    ...cache,
    busyRef,
    branchCurrentSession: async () => false,
    createBackendSessionForSend: async () => {
      throw new Error('unexpected create')
    },
    getRoutedStoredSessionId: () => routedStoredId,
    getRouteToken: () => `${routedStoredId ? sessionRoute(routedStoredId) : '/'}::`,
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: async () => {
      throw new Error('unexpected foreground resume')
    },
    startFreshSessionDraft: () => undefined,
    sttEnabled: false
  })

  handle = { actions, cache }

  return null
}

function seed() {
  routedStoredId = 'stored-B'
  busyRef.current = false
  // Same profile name on distinct backends: profile equality is not ownership.
  setSessions(
    ['A', 'B'].map(id => ({
      id: `stored-${id}`,
      connection_id: `connection-${id}`,
      profile: 'default',
      source: 'desktop',
      message_count: 1
    })) as SessionInfo[]
  )
  setSelectedStoredSessionId('stored-B')
  setActiveSessionId('rt-B')
  render(<Harness />)
  act(() => {
    for (const id of ['A', 'B']) {
      handle.cache.updateSessionState(
        `rt-${id}`,
        state => ({
          ...state,
          messages: [{ id: `history-${id}`, role: 'assistant', parts: [textPart(`history ${id}`)] }]
        }),
        `stored-${id}`
      )
    }
  })
}

function navigateToA() {
  act(() => {
    routedStoredId = 'stored-A'
    setSelectedStoredSessionId('stored-A')
    setActiveSessionId('rt-A')
  })
  act(() => {
    handle.cache.syncSessionStateToView('rt-A', handle.cache.sessionStateByRuntimeIdRef.current.get('rt-A')!)
  })
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  clearSingleFlightSessionResumeState()
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  setBusy(false)
  setAwaitingResponse(false)
  setMessages([])
  window.localStorage.clear()
  vi.clearAllMocks()
})

it.each(['redirectPrompt', 'injectHiddenPrompt'] as const)(
  '%s refuses a source whose route, selection, and runtime do not agree',
  async action => {
    vi.mocked(requestGatewayForAgent).mockResolvedValue({ status: 'queued' })
    seed()
    const originalB = handle.cache.sessionStateByRuntimeIdRef.current.get('rt-B')
    // Route publication can precede both selection and runtime publication.
    routedStoredId = 'stored-A'
    await act(async () => {
      expect(await handle.actions[action]('not B input')).toBe(false)
    })
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
    expect(handle.cache.sessionStateByRuntimeIdRef.current.get('rt-B')).toBe(originalB)

    // Selection/route can also agree while the active runtime still lags.
    act(() => {
      setSelectedStoredSessionId('stored-A')
    })
    // Missing forward proof must not turn a known foreign runtime into A's.
    handle.cache.runtimeIdByStoredSessionIdRef.current.delete('stored-A')
    await act(async () => {
      expect(await handle.actions[action]('A input')).toBe(false)
    })
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
    expect(handle.cache.runtimeIdByStoredSessionIdRef.current.has('stored-A')).toBe(false)
    expect(handle.cache.runtimeIdByStoredSessionIdRef.current.get('stored-B')).toBe('rt-B')

    // A fresh draft must not steer the previous chat's still-live runtime.
    act(() => {
      routedStoredId = null
      setSelectedStoredSessionId(null)
    })
    await act(async () => {
      expect(await handle.actions[action]('new chat input')).toBe(false)
    })
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
  }
)

const steeringActions = [
  { action: 'redirectPrompt' as const, method: 'session.redirect', status: 'redirected' },
  { action: 'redirectPrompt' as const, method: 'session.redirect', status: 'queued' },
  { action: 'injectHiddenPrompt' as const, method: 'session.steer', status: 'queued' }
]

const recoveryCases = (['before-stale-error', 'during-resume', 'stay'] as const).flatMap(navigation =>
  [false, true].flatMap(expiredRecovery => steeringActions.map(action => ({ ...action, navigation, expiredRecovery })))
)

it.each(recoveryCases)(
  '$action recovery ($status, $navigation, expired cache: $expiredRecovery) retains its owner without stealing another chat',
  async ({ action, method, status, navigation, expiredRecovery }) => {
    const first = deferred<unknown>()
    const resume = deferred<{ session_id: string }>()
    vi.mocked(requestGatewayForAgent).mockImplementation(async (_connection, _profile, rpc, params) => {
      if (rpc === method && params?.session_id === 'rt-B') {
        return first.promise
      }

      if (rpc === 'session.resume') {
        return resume.promise
      }

      if (rpc === method && params?.session_id === 'rt-B-cached') {
        throw new Error('session not found')
      }

      if (rpc === method) {
        return { status }
      }

      if (rpc === 'prompt.submit') {
        // Durable admission protocol: a successful submit is acknowledged with
        // the receipt for THIS submission, never a bare legacy status.
        return { admission_id: params?.submission_id, status: 'started' }
      }

      throw new Error(`unexpected ${rpc}`)
    })
    seed()

    if (expiredRecovery) {
      registerRecoveredRuntime('stored-B', 'rt-B-cached')
    }

    let pending!: Promise<boolean>
    act(() => {
      pending = handle.actions[action]('B correction')
    })
    await waitFor(() =>
      expect(requestGatewayForAgent).toHaveBeenCalledWith('connection-B', 'default', method, {
        session_id: 'rt-B',
        text: 'B correction'
      })
    )

    if (navigation === 'before-stale-error') {
      navigateToA()
    }

    await act(async () => {
      first.reject(new Error('session not found'))
    })
    await waitFor(() =>
      expect(requestGatewayForAgent).toHaveBeenCalledWith(
        'connection-B',
        'default',
        'session.resume',
        expect.objectContaining({ session_id: 'stored-B' })
      )
    )

    if (expiredRecovery) {
      expect(requestGatewayForAgent).toHaveBeenCalledWith('connection-B', 'default', method, {
        session_id: 'rt-B-cached',
        text: 'B correction'
      })
      expect($activeSessionId.get()).toBe(navigation === 'before-stale-error' ? 'rt-A' : 'rt-B-cached')
    }

    if (navigation === 'during-resume') {
      navigateToA()
    }

    await act(async () => {
      resume.resolve({ session_id: 'rt-B2' })
      expect(await pending).toBe(true)
    })

    expect(requestGatewayForAgent).toHaveBeenCalledWith('connection-B', 'default', method, {
      session_id: 'rt-B2',
      text: 'B correction'
    })
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
    expect(handle.cache.runtimeIdByStoredSessionIdRef.current.get('stored-A')).toBe('rt-A')
    expect(handle.cache.runtimeIdByStoredSessionIdRef.current.get('stored-B')).toBe('rt-B2')
    expect(handle.cache.sessionStateByRuntimeIdRef.current.get('rt-B2')?.storedSessionId).toBe('stored-B')

    const correctionRows = [...handle.cache.sessionStateByRuntimeIdRef.current.values()]
      .flatMap(state => state.messages)
      .filter(message => message.role === 'user')

    expect(correctionRows).toHaveLength(action === 'redirectPrompt' ? 1 : 0)
    expect($activeSessionId.get()).toBe(navigation === 'stay' ? 'rt-B2' : 'rt-A')
    expect(handle.cache.activeSessionIdRef.current).toBe($activeSessionId.get())
    expect($selectedStoredSessionId.get()).toBe(navigation === 'stay' ? 'stored-B' : 'stored-A')

    if (navigation === 'stay') {
      navigateToA()
    }

    expect($messages.get().map(message => message.id)).toEqual(['history-A'])
    // An ordinary foreground send must still route to A after B's recovery.
    await act(async () => {
      expect(await handle.actions.submitText('A next prompt', { attachments: [], composerScope: 'stored-A' })).toBe(
        true
      )
    })
    expect(requestGatewayForAgent).toHaveBeenLastCalledWith(
      'connection-A',
      'default',
      'prompt.submit',
      expect.objectContaining({ session_id: 'rt-A', text: 'A next prompt' }),
      expect.any(Number),
      undefined
    )
  }
)

const rebuiltRuntimeCases = steeringActions.flatMap(action =>
  [false, true].flatMap(recover => ['stored-B', 'stored-B-tip'].map(selection => ({ ...action, recover, selection })))
)

it.each(rebuiltRuntimeCases)(
  '$action ($status, recovery: $recover, selection: $selection) preserves a rebuilt runtime’s tip binding and durable route',
  async ({ action, method, status, recover, selection }) => {
    seed()
    const rotation = $activeSessionStoredIdRotation.get()

    act(() => {
      setSessions([
        {
          id: 'stored-B-tip',
          _lineage_root_id: 'stored-B',
          profile: 'default',
          connection_id: 'connection-B',
          source: 'desktop',
          message_count: 2
        }
      ] as SessionInfo[])
      handleSessionInfoEvent({
        deps: {
          ...handle.cache,
          activeGatewayProfile: 'default',
          compactedTurnRef: { current: new Set() },
          lastCwdInfoSessionRef: { current: null },
          nativeSubagentSessionsRef: { current: new Set() },
          appendAssistantDelta: vi.fn(),
          appendReasoningDelta: vi.fn(),
          completeAssistantMessage: vi.fn(),
          failAssistantMessage: vi.fn(),
          flushQueuedDeltas: vi.fn(),
          dropQueuedDeltas: vi.fn(),
          finalizeInterimAssistantMessage: vi.fn(),
          hydrateFromStoredSession: vi.fn(async () => undefined),
          queryClient: new QueryClient(),
          refreshHermesConfig: vi.fn(async () => undefined),
          scheduleSessionsRefresh: vi.fn(),
          sessionInterrupted: () => false,
          upsertToolCall: vi.fn()
        },
        event: { profile: 'default', session_id: 'rt-B-rebuilt', type: 'session.info' },
        explicitSid: 'rt-B-rebuilt',
        fromActiveSource: () => true,
        isActiveEvent: false,
        occurredAt: Date.now() / 1000,
        payload: { stored_session_id: 'stored-B-tip', model: 'fixture-model', running: true },
        scheduleConfigRefresh: vi.fn(),
        sessionId: 'rt-B-rebuilt'
      })
    })
    // The real reducer adopts this runtime without rotating the durable selection.
    expect($activeSessionId.get()).toBe('rt-B-rebuilt')
    expect($selectedStoredSessionId.get()).toBe('stored-B')
    expect(handle.cache.sessionStateByRuntimeIdRef.current.get('rt-B-rebuilt')?.storedSessionId).toBe('stored-B-tip')

    // A selection may also follow the tip while the durable route stays on root.
    act(() => {
      setSelectedStoredSessionId(selection)
    })

    vi.mocked(requestGatewayForAgent).mockImplementation(async (_connection, _profile, rpc, params) => {
      if (rpc === 'session.resume') {
        return { session_id: 'rt-B2' }
      }

      if (recover && params?.session_id === 'rt-B-rebuilt') {
        throw new Error('session not found')
      }

      return { status }
    })
    await act(async () => {
      expect(await handle.actions[action]('same chat correction')).toBe(true)
    })

    const runtimeId = recover ? 'rt-B2' : 'rt-B-rebuilt'
    expect(requestGatewayForAgent).toHaveBeenLastCalledWith('connection-B', 'default', method, {
      session_id: runtimeId,
      text: 'same chat correction'
    })

    if (recover) {
      expect(requestGatewayForAgent).toHaveBeenCalledWith(
        'connection-B',
        'default',
        'session.resume',
        expect.objectContaining({ session_id: 'stored-B-tip' })
      )
    }

    expect($activeSessionId.get()).toBe(runtimeId)
    expect(handle.cache.activeSessionIdRef.current).toBe(runtimeId)
    expect($selectedStoredSessionId.get()).toBe(selection)
    expect(routedStoredId).toBe('stored-B')
    expect(handle.cache.sessionStateByRuntimeIdRef.current.get('rt-B-rebuilt')?.storedSessionId).toBe('stored-B-tip')
    expect(handle.cache.sessionStateByRuntimeIdRef.current.get(runtimeId)?.storedSessionId).toBe('stored-B-tip')
    expect(handle.cache.runtimeIdByStoredSessionIdRef.current.get('stored-B-tip')).toBe(runtimeId)
    expect(handle.cache.runtimeIdByStoredSessionIdRef.current.get('stored-B')).toBe('rt-B')
    expect($activeSessionStoredIdRotation.get()).toBe(rotation)
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
  }
)
