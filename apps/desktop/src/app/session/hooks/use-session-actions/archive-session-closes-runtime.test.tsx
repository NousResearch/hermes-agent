// Regression (#75489): archive only flipped the stored flag, so the chat's
// runtime kept its max_concurrent_sessions slot until the backend exited.
// Enough archived chats filled the cap and every surface got "Hermes is at the
// active session limit". Archiving an idle chat must close its runtime the way
// removeSession does; a busy one keeps streaming.
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setSessionArchived } from '@/hermes'
import { setSessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

const STORED_ID = 'chat-1'
const RUNTIME_ID = 'runtime-1'

type Handle = Pick<ReturnType<typeof useSessionActions>, 'archiveSession'>

interface HarnessProps {
  foregroundBusy: boolean
  onReady: (handle: Handle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  selected: boolean
}

function Harness({ foregroundBusy, onReady, requestGateway, selected }: HarnessProps) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })
  const activeRuntimeId = selected ? RUNTIME_ID : 'runtime-other'
  const selectedStoredId = selected ? STORED_ID : 'chat-other'

  const actions = useSessionActions({
    activeSessionId: activeRuntimeId,
    activeSessionIdRef: ref<string | null>(activeRuntimeId),
    busyRef: ref(foregroundBusy),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map([[STORED_ID, RUNTIME_ID]])),
    selectedStoredSessionId: selectedStoredId,
    selectedStoredSessionIdRef: ref<string | null>(selectedStoredId),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady({ archiveSession: actions.archiveSession })
  }, [actions, onReady])

  return null
}

async function archive({ foregroundBusy = false, selected = false } = {}) {
  const requestGateway = vi.fn().mockResolvedValue({})
  let handle: Handle | undefined

  render(
    <Harness
      foregroundBusy={foregroundBusy}
      onReady={h => (handle = h)}
      requestGateway={requestGateway}
      selected={selected}
    />
  )
  await waitFor(() => expect(handle).toBeDefined())
  await act(() => handle!.archiveSession(STORED_ID))

  return requestGateway
}

describe('archiveSession releases the active-session slot', () => {
  beforeEach(() => {
    setSessions([makeSessionInfo({ id: STORED_ID, message_count: 2, source: 'desktop' })])
    vi.mocked(setSessionArchived).mockReset().mockResolvedValue({ ok: true })
  })

  afterEach(() => {
    cleanup()
    setSessions([])
    $sessionStates.set({})
  })

  it.each([
    ['selected', true],
    ['background', false]
  ])('closes the idle runtime of a %s chat', async (_label, selected) => {
    const requestGateway = await archive({ selected })

    expect(setSessionArchived).toHaveBeenCalledWith(STORED_ID, true, undefined)
    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('session.close', { session_id: RUNTIME_ID }))
    expect(requestGateway).not.toHaveBeenCalledWith('session.close', { session_id: 'runtime-other' })
  })

  it.each([
    ['streaming', { busy: true }],
    ['waiting on the user', { needsInput: true }],
    ['awaiting a response', { awaitingResponse: true }],
    ['running on the backend', { turnLive: true }]
  ])('leaves a runtime that is %s alone', async (_label, live) => {
    $sessionStates.set({ [RUNTIME_ID]: live as ClientSessionState })

    const requestGateway = await archive()

    expect(setSessionArchived).toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalledWith('session.close', expect.anything())
  })

  it('leaves the selected chat alone while the foreground is busy', async () => {
    const requestGateway = await archive({ foregroundBusy: true, selected: true })

    expect(requestGateway).not.toHaveBeenCalledWith('session.close', expect.anything())
  })

  it('keeps the runtime when the archive itself fails', async () => {
    vi.mocked(setSessionArchived).mockRejectedValue(new Error('archive failed'))

    const requestGateway = await archive()

    expect(requestGateway).not.toHaveBeenCalledWith('session.close', expect.anything())
  })
})
