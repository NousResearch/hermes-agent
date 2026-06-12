import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { deleteSession, setSessionArchived } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import { $sessions, setSessions } from '@/store/session'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

function ref<T>(value: T): MutableRefObject<T> {
  return { current: value }
}

type Actions = ReturnType<typeof useSessionActions>

function Harness({
  onReady,
  refreshSessions
}: {
  onReady: (actions: Actions) => void
  refreshSessions: (profileOverride?: string) => Promise<void>
}) {
  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    navigate: vi.fn() as never,
    refreshSessions,
    requestGateway: vi.fn(async () => ({})) as never,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

async function withActions(
  refreshSessions: (profileOverride?: string) => Promise<void>
): Promise<Actions> {
  let actions: Actions | null = null
  render(<Harness onReady={a => (actions = a)} refreshSessions={refreshSessions} />)
  await waitFor(() => expect(actions).not.toBeNull())
  return actions!
}

describe('removeSession refresh', () => {
  afterEach(() => {
    cleanup()
    setSessions([])
    $pinnedSessionIds.set([])
    vi.restoreAllMocks()
  })

  it('calls refreshSessions with the removed session profile after deleteSession succeeds', async () => {
    const refreshSessions = vi.fn().mockResolvedValue(undefined)
    vi.mocked(deleteSession).mockResolvedValue({ ok: true })

    setSessions([
      {
        id: 'stored-1',
        _lineage_root_id: 'root-1',
        profile: 'aux',
        message_count: 1
      } as never
    ])

    const actions = await withActions(refreshSessions)

    await act(async () => {
      await actions.removeSession('stored-1')
    })

    expect(deleteSession).toHaveBeenCalledWith('stored-1', 'aux')
    expect(refreshSessions).toHaveBeenCalledTimes(1)
    expect(refreshSessions).toHaveBeenCalledWith('aux')
  })

  it('does not call refreshSessions if deleteSession rejects', async () => {
    const refreshSessions = vi.fn().mockResolvedValue(undefined)
    vi.mocked(deleteSession).mockRejectedValue(new Error('boom'))

    setSessions([
      {
        id: 'stored-1',
        _lineage_root_id: 'root-1',
        profile: 'default',
        message_count: 1
      } as never
    ])

    const actions = await withActions(refreshSessions)

    await act(async () => {
      await actions.removeSession('stored-1')
    })

    expect(refreshSessions).not.toHaveBeenCalled()
    // Error path restores the session.
    expect($sessions.get().some(s => s.id === 'stored-1')).toBe(true)
  })
})

describe('archiveSession refresh', () => {
  afterEach(() => {
    cleanup()
    setSessions([])
    $pinnedSessionIds.set([])
    vi.restoreAllMocks()
  })

  it('calls refreshSessions with the archived session profile after setSessionArchived succeeds', async () => {
    const refreshSessions = vi.fn().mockResolvedValue(undefined)
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    setSessions([
      {
        id: 'stored-2',
        _lineage_root_id: 'root-2',
        profile: 'dev',
        message_count: 1
      } as never
    ])

    const actions = await withActions(refreshSessions)

    await act(async () => {
      await actions.archiveSession('stored-2')
    })

    expect(setSessionArchived).toHaveBeenCalledWith('stored-2', true, 'dev')
    expect(refreshSessions).toHaveBeenCalledTimes(1)
    expect(refreshSessions).toHaveBeenCalledWith('dev')
  })

  it('does not call refreshSessions if setSessionArchived rejects', async () => {
    const refreshSessions = vi.fn().mockResolvedValue(undefined)
    vi.mocked(setSessionArchived).mockRejectedValue(new Error('boom'))

    setSessions([
      {
        id: 'stored-2',
        _lineage_root_id: 'root-2',
        profile: 'default',
        message_count: 1
      } as never
    ])

    const actions = await withActions(refreshSessions)

    await act(async () => {
      await actions.archiveSession('stored-2')
    })

    expect(refreshSessions).not.toHaveBeenCalled()
    // Error path restores the session.
    expect($sessions.get().some(s => s.id === 'stored-2')).toBe(true)
  })
})
