import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect, useRef } from 'react'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { useMessageStream } from '../use-message-stream'

import type { SubmitTextOptions } from './utils'

import { usePromptActions } from '.'

// Shared fixtures for the usePromptActions suites (index.test.tsx and its
// split siblings). Each test file keeps its own vi.mock calls.
// The active id the desktop holds is the *runtime* session id from
// session.create — deliberately distinct from the stored DB id here, because
// that mismatch is the bug: the REST renameSession endpoint resolves against
// the stored sessions table and 404s on a runtime id. session.title accepts
// the runtime id directly.
export const RUNTIME_SESSION_ID = 'rt-abc123'

/** Harness props spell "use the fixture runtime id" as undefined (null stays null). */
const orRuntimeDefault = (id: null | string | undefined): null | string => (id === undefined ? RUNTIME_SESSION_ID : id)

export function sessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: RUNTIME_SESSION_ID,
    input_tokens: 0,
    is_active: true,
    last_active: 0,
    message_count: 3,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: 'Old title',
    tool_call_count: 0,
    ...overrides
  }
}

// Wrap render() in act() so the Harness's useEffect (onReady callback +
// internal state from usePromptActions) flushes synchronously instead of
// spilling async state updates outside act().
export async function actRender(ui: React.ReactElement) {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(ui)
  })

  return result!
}

export interface HarnessHandle {
  handleEvent: (event: GatewayEvent) => void
  state: () => ClientSessionState
  activeSessionIdRef: MutableRefObject<string | null>
  cancelRun: () => Promise<void>
  editMessage: (edited: Parameters<ReturnType<typeof usePromptActions>['editMessage']>[0]) => Promise<void>
  reloadFromMessage: (parentId: null | string) => Promise<void>
  restoreToMessage: (messageId: string, target?: { text?: string; userOrdinal?: number | null }) => Promise<void>
  redirectPrompt: (text: string) => Promise<boolean>
  /** @deprecated Use `redirectPrompt`. */
  steerPrompt: (text: string) => Promise<boolean>
  submitTextRaw: (text: string, options?: SubmitTextOptions) => Promise<boolean>
  submitText: (text: string, options?: SubmitTextOptions) => Promise<boolean>
}

export function Harness({
  activeSessionIdRef: activeSessionIdRefProp,
  busyRef,
  getRoutedStoredSessionId,
  getRuntimeIdForStoredSession,
  getRouteToken,
  onUpdateState,
  onReady,
  onSeedState,
  openMemoryGraph,
  refreshSessions,
  requestGateway,
  resumeStoredSession,
  runtimeIdByStoredSessionIdRef: runtimeIdByStoredSessionIdRefProp,
  seedMessages,
  seedStreamId,
  seedTurnStartedAt,
  rawAdmissionReceipts = false,
  selectedStoredSessionIdRef: selectedStoredSessionIdRefProp,
  storedSessionId,
  activeSessionId,
  createBackendSessionForSend
}: {
  activeSessionIdRef?: MutableRefObject<string | null>
  busyRef?: MutableRefObject<boolean>
  getRoutedStoredSessionId?: () => null | string
  getRuntimeIdForStoredSession?: (storedSessionId: string) => null | string
  getRouteToken?: () => string
  onUpdateState?: (
    sessionId: string,
    storedSessionId: null | string | undefined,
    state: Record<string, unknown>
  ) => void
  onReady: (handle: HarnessHandle) => void
  onSeedState?: (state: Record<string, unknown>) => void
  openMemoryGraph?: () => void
  refreshSessions: () => Promise<void>
  requestGateway: <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>
  resumeStoredSession?: (storedSessionId: string) => Promise<void> | void
  runtimeIdByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
  seedMessages?: unknown[]
  seedStreamId?: null | string
  seedTurnStartedAt?: null | number
  rawAdmissionReceipts?: boolean
  selectedStoredSessionIdRef?: MutableRefObject<string | null>
  storedSessionId?: null | string
  activeSessionId?: null | string
  createBackendSessionForSend?: (preview?: null | string) => Promise<null | string>
}) {
  const localActiveSessionIdRef = useRef<string | null>(orRuntimeDefault(activeSessionId))

  const activeSessionIdRef = activeSessionIdRefProp ?? localActiveSessionIdRef

  const selectedStoredSessionIdRef: MutableRefObject<string | null> = selectedStoredSessionIdRefProp ?? {
    current: orRuntimeDefault(storedSessionId)
  }

  const defaultStoredSessionId = orRuntimeDefault(storedSessionId)
  const defaultRuntimeSessionId = orRuntimeDefault(activeSessionId)

  const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = runtimeIdByStoredSessionIdRefProp ?? {
    current:
      defaultStoredSessionId && defaultRuntimeSessionId
        ? new Map([[defaultStoredSessionId, defaultRuntimeSessionId]])
        : new Map()
  }

  const localBusyRef = busyRef ?? { current: false }

  const stateRef = useRef({
    ...createClientSessionState(),
    messages: seedMessages ?? [],
    busy: false,
    awaitingResponse: false,
    interrupted: true,
    streamId: seedStreamId ?? null,
    turnStartedAt: seedTurnStartedAt ?? null,
    interimBoundaryPending: false
  } as never)

  const sessionStates = useRef(new Map<string, ClientSessionState>())
  const queryClient = useRef(new QueryClient())

  const updateSessionState: Parameters<typeof useMessageStream>[0]['updateSessionState'] = (
    sessionId,
    updater,
    storedId
  ) => {
    const next = updater(stateRef.current)
    stateRef.current = next as never
    sessionStates.current.set(sessionId, next)
    onSeedState?.(next as unknown as Record<string, unknown>)
    onUpdateState?.(sessionId, storedId, next as unknown as Record<string, unknown>)

    return next
  }

  const { handleGatewayEvent } = useMessageStream({
    activeSessionIdRef,
    sessionStateByRuntimeIdRef: sessionStates,
    queryClient: queryClient.current,
    updateSessionState,
    hydrateFromStoredSession: async () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions
  })

  const actions = usePromptActions({
    activeSessionId: orRuntimeDefault(activeSessionId),
    activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef: localBusyRef,
    createBackendSessionForSend: createBackendSessionForSend ?? (async () => RUNTIME_SESSION_ID),
    getRoutedStoredSessionId: getRoutedStoredSessionId ?? (() => null),
    getRuntimeIdForStoredSession: getRuntimeIdForStoredSession ?? (() => null),
    getRouteToken: getRouteToken ?? (() => 'token'),
    handleSkinCommand: () => '',
    openMemoryGraph: openMemoryGraph ?? (() => undefined),
    refreshSessions,
    // Older fixture peers acknowledged successful submits with an empty object.
    // Keep those peers successful under the durable protocol; receipt tests opt
    // out so missing/mismatched acknowledgements still exercise production.
    requestGateway: async (method, params, timeoutMs) => {
      const result = await (timeoutMs === undefined
        ? requestGateway(method, params)
        : requestGateway(method, params, timeoutMs))

      if (
        !rawAdmissionReceipts &&
        method === 'prompt.submit' &&
        result &&
        typeof result === 'object' &&
        (Object.keys(result).length === 0 || ('ok' in result && result.ok === true))
      ) {
        return { admission_id: params?.submission_id, status: 'started' } as never
      }

      return result as never
    },
    resumeStoredSession: resumeStoredSession ?? (() => undefined),
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState
  })

  useEffect(() => {
    onReady({
      handleEvent: event => act(() => handleGatewayEvent(event)),
      state: () => stateRef.current,
      activeSessionIdRef,
      cancelRun: (...args: Parameters<typeof actions.cancelRun>) =>
        act(async () => actions.cancelRun(...args)) as Promise<void>,
      editMessage: (...args: Parameters<typeof actions.editMessage>) =>
        act(async () => actions.editMessage(...args)) as Promise<void>,
      reloadFromMessage: (...args: Parameters<typeof actions.reloadFromMessage>) =>
        act(async () => actions.reloadFromMessage(...args)) as Promise<void>,
      restoreToMessage: (...args: Parameters<typeof actions.restoreToMessage>) =>
        act(async () => actions.restoreToMessage(...args)) as Promise<void>,
      redirectPrompt: (...args: Parameters<typeof actions.redirectPrompt>) =>
        act(async () => actions.redirectPrompt(...args)) as Promise<boolean>,
      steerPrompt: (...args: Parameters<typeof actions.steerPrompt>) =>
        act(async () => actions.steerPrompt(...args)) as Promise<boolean>,
      submitTextRaw: actions.submitText,
      submitText: (...args: Parameters<typeof actions.submitText>) =>
        act(async () => actions.submitText(...args)) as Promise<boolean>
    })
  }, [
    actions.cancelRun,
    actions.editMessage,
    actions.reloadFromMessage,
    actions.restoreToMessage,
    actions.redirectPrompt,
    actions.steerPrompt,
    actions.submitText,
    activeSessionIdRef,
    handleGatewayEvent,
    onReady
  ])

  return null
}
