import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionInfo, setSessionArchived } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $sessions, setSessions } from '@/store/session'

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

function listedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    ended_at: null,
    id: 'chat-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'chat',
    tool_call_count: 0,
    ...overrides
  } as SessionInfo
}

type Handle = Pick<ReturnType<typeof useSessionActions>, 'archiveSession'>

function Harness({ onReady }: { onReady: (handle: Handle) => void }) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway: vi.fn().mockResolvedValue(undefined),
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady({ archiveSession: actions.archiveSession })
  }, [actions, onReady])

  return null
}

async function mountHarness(): Promise<Handle> {
  let handle: Handle | undefined
  render(<Harness onReady={h => (handle = h)} />)
  await waitFor(() => expect(handle).toBeDefined())

  return handle!
}

// Archiving from the sidebar is one click away from a misclick; the toast's
// Undo must put the row, its archived flag and its pin back.
describe('archiveSession undo', () => {
  beforeEach(() => {
    setSessions([])
    $pinnedSessionIds.set([])
    clearNotifications()
    vi.mocked(setSessionArchived).mockReset()
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })
  })

  afterEach(() => {
    cleanup()
    setSessions([])
    $pinnedSessionIds.set([])
    clearNotifications()
  })

  it('Undo on the Archived toast restores the row, the flag and the pin position', async () => {
    setSessions([listedSession({ id: 'other' }), listedSession()])
    $pinnedSessionIds.set(['p0', 'chat-1', 'p2'])

    const handle = await mountHarness()

    await act(() => handle.archiveSession('chat-1'))

    expect($sessions.get().map(session => session.id)).toEqual(['other'])
    expect($pinnedSessionIds.get()).toEqual(['p0', 'p2'])

    const toast = $notifications.get().find(item => item.action)

    expect(toast?.action?.label).toBe('Undo')

    await act(async () => toast!.action!.onClick())

    await waitFor(() => expect(vi.mocked(setSessionArchived)).toHaveBeenLastCalledWith('chat-1', false, undefined))
    await waitFor(() => expect($sessions.get().map(session => session.id).sort()).toEqual(['chat-1', 'other']))
    expect($pinnedSessionIds.get()).toEqual(['p0', 'chat-1', 'p2'])
  })
})
