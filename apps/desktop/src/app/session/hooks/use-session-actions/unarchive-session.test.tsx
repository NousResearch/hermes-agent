// Regression (#98813): the sidebar's Archived view reuses the row action
// menu, so its shared archive verb must RESTORE an already-archived row
// instead of re-archiving it. unarchiveSession is the restore path the
// wiring dispatches to: flip the persisted flag off, drop the archived-view
// row, and resurface the session in $sessions without a full refresh — the
// same shape as the Settings → Archived Chats restore.
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionInfo, setSessionArchived } from '@/hermes'
import { $sessions, setSessions } from '@/store/session'
import { $archivedSessions } from '@/store/sidebar-archive'

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

function archivedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: true,
    ended_at: null,
    id: 'arch-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'archived row',
    tool_call_count: 0,
    ...overrides
  } as SessionInfo
}

type Handle = Pick<ReturnType<typeof useSessionActions>, 'unarchiveSession'>

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
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady({ unarchiveSession: actions.unarchiveSession })
  }, [actions, onReady])

  return null
}

async function mountHarness(): Promise<Handle> {
  let handle: Handle | undefined
  render(<Harness onReady={h => (handle = h)} />)
  await waitFor(() => expect(handle).toBeDefined())

  return handle!
}

describe('unarchiveSession', () => {
  beforeEach(() => {
    setSessions([])
    $archivedSessions.set([])
    vi.mocked(setSessionArchived).mockReset()
  })

  afterEach(() => {
    cleanup()
    setSessions([])
    $archivedSessions.set([])
  })

  it('restores an archived row: flips the RPC off, drops the view row, resurfaces it in $sessions', async () => {
    $archivedSessions.set([archivedSession(), archivedSession({ id: 'arch-2', title: 'keep me' })])
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    const handle = await mountHarness()

    await act(() => handle.unarchiveSession('arch-1'))

    expect(vi.mocked(setSessionArchived)).toHaveBeenCalledWith('arch-1', false, undefined)
    expect($archivedSessions.get().map(session => session.id)).toEqual(['arch-2'])
    expect($sessions.get().map(session => session.id)).toEqual(['arch-1'])
    expect($sessions.get()[0]?.archived).toBe(false)
  })

  it('passes the archived row profile to setSessionArchived', async () => {
    $archivedSessions.set([archivedSession({ profile: 'sage' } as Partial<SessionInfo>)])
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    const handle = await mountHarness()

    await act(() => handle.unarchiveSession('arch-1'))

    expect(vi.mocked(setSessionArchived)).toHaveBeenCalledWith('arch-1', false, 'sage')
  })

  it('keeps the archived row when the restore RPC fails', async () => {
    $archivedSessions.set([archivedSession()])
    vi.mocked(setSessionArchived).mockRejectedValue(new Error('backend down'))

    const handle = await mountHarness()

    await act(() => handle.unarchiveSession('arch-1'))

    expect($archivedSessions.get().map(session => session.id)).toEqual(['arch-1'])
    expect($sessions.get()).toEqual([])
  })
})
