import { act, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useCallback, useEffect, useRef } from 'react'

import type { SessionInfo } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

import type { SubmitTextOptions } from './utils'

import { usePromptActions } from '.'

// The active id the desktop holds is the *runtime* session id from
// session.create — deliberately distinct from the stored DB id here, because
// that mismatch is the bug: the REST renameSession endpoint resolves against
// the stored sessions table and 404s on a runtime id. session.title accepts
// the runtime id directly.
export const RUNTIME_SESSION_ID = 'rt-abc123'
// Every typed command also fires this (fire-and-forget); these tests assert the command's own traffic.
export const SLASH_METRIC = 'shared_metrics.slash_command'

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
  updateSessionState: (sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => void
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
  selectedStoredSessionIdRef?: MutableRefObject<string | null>
  storedSessionId?: null | string
  activeSessionId?: null | string
  createBackendSessionForSend?: (preview?: null | string) => Promise<null | string>
}) {
  const localActiveSessionIdRef = useRef<string | null>(
    activeSessionId === undefined ? RUNTIME_SESSION_ID : activeSessionId
  )

  const activeSessionIdRef = activeSessionIdRefProp ?? localActiveSessionIdRef

  const selectedStoredSessionIdRef: MutableRefObject<string | null> = selectedStoredSessionIdRefProp ?? {
    current: storedSessionId === undefined ? RUNTIME_SESSION_ID : storedSessionId
  }

  const defaultStoredSessionId = storedSessionId === undefined ? RUNTIME_SESSION_ID : storedSessionId
  const defaultRuntimeSessionId = activeSessionId === undefined ? RUNTIME_SESSION_ID : activeSessionId

  const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = runtimeIdByStoredSessionIdRefProp ?? {
    current:
      defaultStoredSessionId && defaultRuntimeSessionId
        ? new Map([[defaultStoredSessionId, defaultRuntimeSessionId]])
        : new Map()
  }

  const localBusyRef = busyRef ?? { current: false }

  const stateRef = useRef({
    messages: seedMessages ?? [],
    busy: false,
    awaitingResponse: false,
    interrupted: true,
    streamId: seedStreamId ?? null,
    turnStartedAt: seedTurnStartedAt ?? null,
    interimBoundaryPending: false
  } as never)

  const updateSessionState = useCallback((
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: null | string
  ) => {
    // Seed with interrupted:true so we can prove a fresh submit clears it.
    const next = updater(stateRef.current) as unknown as Record<string, unknown>
    stateRef.current = next as never
    onSeedState?.(next)
    onUpdateState?.(sessionId, storedSessionId, next)

    return next as never
  }, [onSeedState, onUpdateState])

  const actions = usePromptActions({
    activeSessionId: activeSessionId === undefined ? RUNTIME_SESSION_ID : activeSessionId,
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
    requestGateway,
    resumeStoredSession: resumeStoredSession ?? (() => undefined),
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState
  })

  useEffect(() => {
    onReady({
      activeSessionIdRef,
      updateSessionState,
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
    onReady,
    updateSessionState
  ])

  return null
}
