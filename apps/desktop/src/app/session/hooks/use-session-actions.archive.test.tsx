import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import type { NavigateFunction } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setSessionArchived } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import { $notifications, clearNotifications } from '@/store/notifications'
import {
  $archivedSessions,
  $archivedSessionsTotal,
  $sessions,
  $sessionsTotal,
  setArchivedSessions,
  setArchivedSessionsTotal,
  setSessions,
  setSessionsTotal
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'

vi.mock('@/hermes', () => ({
  deleteSession: vi.fn(),
  getHermesConfigRecord: vi.fn(async () => ({})),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSessionMessages: vi.fn(),
  saveHermesConfig: vi.fn(async () => ({ ok: true })),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn(async () => ({ ok: true }))
}))

function sessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id: 'session-1',
    input_tokens: 0,
    is_active: false,
    last_active: 20,
    message_count: 3,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: 'tui',
    started_at: 10,
    title: 'Session',
    tool_call_count: 0,
    ...overrides
  }
}

type SessionActions = ReturnType<typeof useSessionActions>

function clientSessionState(overrides: Partial<ClientSessionState> = {}): ClientSessionState {
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
    streamId: null,
    turnStartedAt: null,
    ...overrides
  }
}

function Harness({ onReady }: { onReady: (actions: SessionActions) => void }) {
  const activeSessionIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const selectedStoredSessionIdRef = useRef<string | null>(null)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: vi.fn(() => clientSessionState()),
    getRouteToken: () => 'route-token',
    navigate: vi.fn() as unknown as NavigateFunction,
    requestGateway: vi.fn(async () => ({})) as unknown as <T>(
      method: string,
      params?: Record<string, unknown>
    ) => Promise<T>,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView: vi.fn(),
    updateSessionState: vi.fn((_, updater) => updater(clientSessionState()))
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

describe('useSessionActions archive restore flow', () => {
  beforeEach(() => {
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })
    setSessions(() => [])
    setSessionsTotal(0)
    setArchivedSessions(() => [])
    setArchivedSessionsTotal(0)
    $pinnedSessionIds.set([])
    clearNotifications()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    setSessions(() => [])
    setSessionsTotal(0)
    setArchivedSessions(() => [])
    setArchivedSessionsTotal(0)
    $pinnedSessionIds.set([])
    clearNotifications()
  })

  it('optimistically restores an archived session into the live sidebar and persists archived=false', async () => {
    const archived = sessionInfo({ archived: true, id: 'archived-1', _lineage_root_id: 'root-1', title: 'Archived' })
    const live = sessionInfo({ id: 'live-1', title: 'Live' })
    let actions!: SessionActions

    setSessions(() => [live])
    setSessionsTotal(1)
    setArchivedSessions(() => [archived])
    setArchivedSessionsTotal(2)

    render(<Harness onReady={next => (actions = next)} />)

    await act(async () => {
      await actions.unarchiveSession('archived-1')
    })

    expect(setSessionArchived).toHaveBeenCalledWith('archived-1', false, 'default')
    expect($archivedSessions.get().map(session => session.id)).toEqual([])
    expect($archivedSessionsTotal.get()).toBe(1)
    expect($sessions.get().map(session => session.id)).toEqual(['archived-1', 'live-1'])
    expect($sessions.get()[0]).toMatchObject({ archived: false, id: 'archived-1', _lineage_root_id: 'root-1' })
    expect($sessionsTotal.get()).toBe(2)
    expect($notifications.get()[0]).toMatchObject({ kind: 'success', message: 'Restored' })
  })

  it('keeps pinned lineage ids while archiving so restore brings the pin back', async () => {
    const pinned = sessionInfo({ id: 'tip-1', _lineage_root_id: 'root-1', title: 'Pinned work' })
    let actions!: SessionActions

    setSessions(() => [pinned])
    setSessionsTotal(1)
    $pinnedSessionIds.set(['root-1'])

    render(<Harness onReady={next => (actions = next)} />)

    await act(async () => {
      await actions.archiveSession('tip-1')
    })

    expect(setSessionArchived).toHaveBeenCalledWith('tip-1', true, 'default')
    expect($sessions.get().map(session => session.id)).toEqual([])
    expect($archivedSessions.get().map(session => session.id)).toEqual(['tip-1'])
    expect($pinnedSessionIds.get()).toEqual(['root-1'])

    await act(async () => {
      await actions.unarchiveSession('tip-1')
    })

    expect($sessions.get()[0]).toMatchObject({ archived: false, id: 'tip-1', _lineage_root_id: 'root-1' })
    expect($pinnedSessionIds.get()).toEqual(['root-1'])
  })

  it('rolls back both live and archive lists when restoring fails', async () => {
    const archived = sessionInfo({ archived: true, id: 'archived-1', title: 'Archived' })
    const archivedOther = sessionInfo({ archived: true, id: 'archived-2', title: 'Archived 2' })
    const live = sessionInfo({ id: 'live-1', title: 'Live' })
    let actions!: SessionActions

    vi.mocked(setSessionArchived).mockRejectedValueOnce(new Error('backend unavailable'))
    setSessions(() => [live])
    setSessionsTotal(3)
    setArchivedSessions(() => [archived, archivedOther])
    setArchivedSessionsTotal(2)

    render(<Harness onReady={next => (actions = next)} />)

    await act(async () => {
      await actions.unarchiveSession('archived-1')
    })

    expect($sessions.get().map(session => session.id)).toEqual(['live-1'])
    expect($sessionsTotal.get()).toBe(3)
    expect($archivedSessions.get().map(session => session.id)).toEqual(['archived-1', 'archived-2'])
    expect($archivedSessionsTotal.get()).toBe(2)
    expect($notifications.get()[0]).toMatchObject({ kind: 'error', title: 'Unarchive failed' })
  })
})
