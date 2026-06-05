import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $freshDraftReady, $yoloActive } from '@/store/session'

import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'

vi.mock('@/hermes', () => ({
  deleteSession: vi.fn(),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSessionMessages: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

type SessionActions = ReturnType<typeof useSessionActions>

function clientSessionState(): ClientSessionState {
  return {
    awaitingResponse: false,
    branch: '',
    busy: false,
    cwd: '',
    interrupted: false,
    messages: [],
    needsInput: false,
    pendingBranchGroup: null,
    sawAssistantPayload: false,
    storedSessionId: null,
    streamId: null
  }
}

function Harness({ onReady }: { onReady: (actions: SessionActions) => void }) {
  const requestGateway = vi.fn(async () => ({})) as <T>(
    method: string,
    params?: Record<string, unknown>
  ) => Promise<T>

  const actions = useSessionActions({
    activeSessionId: 'runtime-1',
    activeSessionIdRef: { current: 'runtime-1' } as MutableRefObject<string | null>,
    busyRef: { current: true },
    creatingSessionRef: { current: false },
    ensureSessionState: clientSessionState,
    getRouteToken: () => '',
    navigate: vi.fn(),
    requestGateway,
    runtimeIdByStoredSessionIdRef: { current: new Map() },
    selectedStoredSessionId: 'stored-1',
    selectedStoredSessionIdRef: { current: 'stored-1' } as MutableRefObject<string | null>,
    sessionStateByRuntimeIdRef: { current: new Map() },
    syncSessionStateToView: vi.fn(),
    updateSessionState: (_sessionId, updater) => updater(clientSessionState())
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

afterEach(() => {
  cleanup()
  $freshDraftReady.set(false)
  $yoloActive.set(false)
  vi.restoreAllMocks()
})

describe('useSessionActions', () => {
  it('does not carry session YOLO into a fresh chat draft', () => {
    let actions: SessionActions | null = null
    $yoloActive.set(true)

    render(<Harness onReady={next => (actions = next)} />)

    act(() => {
      actions!.startFreshSessionDraft()
    })

    expect($freshDraftReady.get()).toBe(true)
    expect($yoloActive.get()).toBe(false)
  })
})
