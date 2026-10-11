// Regression (#75489): archive only flipped the stored flag, so the chat's
// runtime kept its max_concurrent_sessions slot until the backend exited.
// Enough archived chats filled the cap and every surface got "Hermes is at the
// active session limit". Archiving an idle chat must close its runtime the way
// removeSession does; a busy one keeps streaming.
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionInfo, setSessionArchived } from '@/hermes'
import { setSessions } from '@/store/session'

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

function storedSession(): SessionInfo {
  return {
    ended_at: null,
    id: STORED_ID,
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'idle chat',
    tool_call_count: 0
  } as SessionInfo
}

type Handle = Pick<ReturnType<typeof useSessionActions>, 'archiveSession'>

interface HarnessProps {
  busy?: Partial<ClientSessionState>
  onReady: (handle: Handle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  selected: boolean
}

function Harness({ busy, onReady, requestGateway, selected }: HarnessProps) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })
  const state = { ...busy } as ClientSessionState

  const actions = useSessionActions({
    activeSessionId: selected ? RUNTIME_ID : 'runtime-other',
    activeSessionIdRef: ref<string | null>(selected ? RUNTIME_ID : 'runtime-other'),
    busyRef: ref(selected && Boolean(busy?.busy)),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map([[STORED_ID, RUNTIME_ID]])),
    selectedStoredSessionId: selected ? STORED_ID : 'chat-other',
    selectedStoredSessionIdRef: ref<string | null>(selected ? STORED_ID : 'chat-other'),
    sessionStateByRuntimeIdRef: ref(new Map([[RUNTIME_ID, state]])),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady({ archiveSession: actions.archiveSession })
  }, [actions, onReady])

  return null
}

async function archive(props: Omit<HarnessProps, 'onReady'>) {
  let handle: Handle | undefined
  render(<Harness {...props} onReady={h => (handle = h)} />)
  await waitFor(() => expect(handle).toBeDefined())
  await act(() => handle!.archiveSession(STORED_ID))
}

describe('archiveSession releases the active-session slot', () => {
  beforeEach(() => {
    setSessions([storedSession()])
    vi.mocked(setSessionArchived).mockReset()
  })

  afterEach(() => {
    cleanup()
    setSessions([])
  })

  it.each([
    ['selected', true],
    ['background', false]
  ])('closes the idle runtime of a %s chat', async (_label, selected) => {
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })
    const requestGateway = vi.fn().mockResolvedValue({})

    await archive({ requestGateway, selected })

    expect(setSessionArchived).toHaveBeenCalledWith(STORED_ID, true, undefined)
    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('session.close', { session_id: RUNTIME_ID }))
    expect(requestGateway).not.toHaveBeenCalledWith('session.close', { session_id: 'runtime-other' })
  })

  it.each([
    ['streaming', { busy: true }],
    ['waiting on the user', { needsInput: true }],
    ['awaiting a response', { awaitingResponse: true }]
  ])('leaves a runtime that is %s alone', async (_label, busy) => {
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })
    const requestGateway = vi.fn().mockResolvedValue({})

    await archive({ busy, requestGateway, selected: false })

    expect(setSessionArchived).toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalledWith('session.close', expect.anything())
  })

  it('keeps the runtime when the archive itself fails', async () => {
    vi.mocked(setSessionArchived).mockRejectedValue(new Error('archive failed'))
    const requestGateway = vi.fn().mockResolvedValue({})

    await archive({ requestGateway, selected: false })

    expect(requestGateway).not.toHaveBeenCalledWith('session.close', expect.anything())
  })
})
