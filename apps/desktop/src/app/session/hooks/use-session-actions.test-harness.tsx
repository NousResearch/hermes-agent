import { useStore } from '@nanostores/react'
import { type MutableRefObject, useEffect, useRef } from 'react'
import { vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages
} from '@/store/session'

import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'
import { useSessionStateCache } from './use-session-state-cache'

export const RUNTIME_SESSION_ID = 'rt-new-001'

export type HarnessHandle = Pick<
  ReturnType<typeof useSessionActions>,
  | 'archiveSession'
  | 'branchStoredSession'
  | 'createBackendSessionForSend'
  | 'openNewSessionTile'
  | 'removeSession'
  | 'selectSidebarItem'
  | 'startFreshSessionDraft'
> &
  Pick<ReturnType<typeof useSessionActions>, 'submitTextToNewSession'>

export function storedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'stored',
    tool_call_count: 0,
    ...overrides
  }
}

export function Harness({
  activeSessionId = null,
  activeSessionIdRef: activeSessionIdRefOverride,
  getRouteToken: getRouteTokenOverride,
  getRoutedStoredSessionId = () => null,
  navigate = vi.fn(),
  onReady,
  requestGateway,
  runtimeIdByStoredSessionIdRef: runtimeIdByStoredSessionIdRefOverride,
  selectedStoredSessionId = null,
  selectedStoredSessionIdRef: selectedStoredSessionIdRefOverride,
  updateSessionState: updateSessionStateOverride
}: {
  activeSessionId?: null | string
  activeSessionIdRef?: MutableRefObject<null | string>
  getRouteToken?: () => string
  getRoutedStoredSessionId?: () => null | string
  navigate?: ReturnType<typeof vi.fn>
  onReady: (handle: HarnessHandle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  runtimeIdByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
  selectedStoredSessionId?: null | string
  selectedStoredSessionIdRef?: MutableRefObject<null | string>
  updateSessionState?: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: null | string
  ) => ClientSessionState
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId,
    activeSessionIdRef: activeSessionIdRefOverride ?? ref(activeSessionId),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: getRouteTokenOverride ?? (() => 'token'),
    getRoutedStoredSessionId,
    navigate: navigate as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: runtimeIdByStoredSessionIdRefOverride ?? ref(new Map<string, string>()),
    selectedStoredSessionId,
    selectedStoredSessionIdRef: selectedStoredSessionIdRefOverride ?? ref(selectedStoredSessionId),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: updateSessionStateOverride ?? (() => ({}) as ClientSessionState)
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

export function StoredIdRotationHarness({
  activeSessionIdRef,
  getRoutedStoredSessionId,
  navigate,
  selectedStoredSessionIdRef
}: {
  activeSessionIdRef: MutableRefObject<string | null>
  getRoutedStoredSessionId: () => null | string
  navigate: (to: string, options?: { replace?: boolean }) => void
  selectedStoredSessionIdRef: MutableRefObject<string | null>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  useSessionActions({
    activeSessionId: activeSessionIdRef.current,
    activeSessionIdRef,
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId,
    navigate: navigate as never,
    requestGateway: async () => ({}) as never,
    resetViewSync: vi.fn(),
    routedSessionId: getRoutedStoredSessionId(),
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: selectedStoredSessionIdRef.current,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  return null
}

export function StoredIdRotationProducerHarness({
  getRoutedStoredSessionId,
  navigate,
  onCache
}: {
  getRoutedStoredSessionId: () => null | string
  navigate: (to: string, options?: { replace?: boolean }) => void
  onCache: (cache: ReturnType<typeof useSessionStateCache>) => void
}) {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const busyRef = useRef(false)

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  onCache(cache)

  return (
    <StoredIdRotationHarness
      activeSessionIdRef={cache.activeSessionIdRef}
      getRoutedStoredSessionId={getRoutedStoredSessionId}
      navigate={navigate}
      selectedStoredSessionIdRef={cache.selectedStoredSessionIdRef}
    />
  )
}
