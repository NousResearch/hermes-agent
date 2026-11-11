import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { deleteSession, type SessionInfo } from '@/hermes'
import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearQueuedPrompts, enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'
import { $backgroundStatusBySession } from '@/store/composer-status'
import { requestGatewayForAgent } from '@/store/gateway'
import { $pinnedSessionIds } from '@/store/layout'
import { $projectTree, getProjectTreeConnectionId, refreshProjectTree } from '@/store/projects'
import {
  $activeSessionId,
  $messages,
  $selectedStoredSessionId,
  $sessions,
  $unlistedSessionOwnerRows,
  _resetSessionOwnerHintsForTests,
  getSessionOwnerHint,
  setActiveSessionId,
  setMessages,
  setSelectedStoredSessionId,
  setSessionOwnerHint,
  setSessions,
  setUnlistedSessionOwnerRows
} from '@/store/session'
import { $removedSessionIds, $sessionMutationsInFlight } from '@/store/session-removal'
import type { SessionOwnerRoute } from '@/store/session-request-router'
import { $sessionStates, $sessionTiles, clearAllSessionStates, recordSessionEventScope } from '@/store/session-states'
import { $archivedSessions } from '@/store/sidebar-archive'
import { clearTranscriptTails, loadTranscriptTail, saveTranscriptTail } from '@/store/transcript-tail-cache'

import { deferred } from '../../../../test/deferred'
import { NEW_CHAT_ROUTE } from '../../../routes'
import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn()
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn().mockResolvedValue({})
}))
vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProjectTreeConnectionId: vi.fn().mockReturnValue(null),
  refreshProjectTree: vi.fn().mockResolvedValue(undefined)
}))

const id = 'delete-owner-shared'
const runtimeId = 'delete-owner-runtime'
const ownerA: SessionOwnerRoute = { connectionId: 'local', profile: 'default' }
const ownerB: SessionOwnerRoute = { connectionId: 'local', profile: 'research' }
const messages = [{ id: 'reply', role: 'assistant' as const, parts: [textPart('Still streaming')] }]

function row(owner: SessionOwnerRoute, hidden = false): SessionInfo {
  return {
    connection_id: owner.connectionId,
    profile: owner.targetProfile ?? owner.profile,
    hidden,
    id,
    ended_at: null,
    input_tokens: 0,
    is_active: true,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: hidden ? 'Bot Chat' : 'Other owner',
    tool_call_count: 0
  }
}

function mountDeletion(owner: SessionOwnerRoute) {
  const state: ClientSessionState = { ...createClientSessionState(id, messages), busy: true, streamId: 'stream-a' }
  const activeSessionIdRef = { current: runtimeId as string | null }
  const selectedStoredSessionIdRef = { current: id as string | null }
  const runtimeIdByStoredSessionIdRef = { current: new Map([[id, runtimeId]]) }
  const sessionStateByRuntimeIdRef = { current: new Map([[runtimeId, state]]) }
  const navigate = vi.fn()
  const routeTokenRef = { current: 'delete-route' }
  const requestGateway = vi.fn().mockResolvedValue({})

  const updateSessionState = vi.fn((runtime: string, updater: (value: ClientSessionState) => ClientSessionState) => {
    const next = updater(sessionStateByRuntimeIdRef.current.get(runtime) ?? createClientSessionState(id))
    sessionStateByRuntimeIdRef.current.set(runtime, next)
    $sessionStates.set({ ...$sessionStates.get(), [runtime]: next })

    return next
  })

  setActiveSessionId(runtimeId)
  setSelectedStoredSessionId(id)
  setMessages(messages)
  setSessionOwnerHint(id, owner)
  recordSessionEventScope({
    connectionId: owner.connectionId,
    profile: owner.targetProfile ?? owner.profile,
    session_id: runtimeId
  })
  $sessionStates.set({ [runtimeId]: state })
  $sessionTiles.set([{ storedSessionId: id, runtimeId, ownerRoute: owner }])
  enqueueQueuedPrompt(id, { attachments: [], text: 'Next question' })
  enqueueQueuedPrompt(runtimeId, { attachments: [], text: 'Runtime question' })
  $backgroundStatusBySession.set({
    [id]: [{ id: 'stored-job', state: 'done', title: 'Stored task', type: 'background' }],
    [runtimeId]: [{ id: 'runtime-job', state: 'done', title: 'Runtime task', type: 'background' }]
  })
  $pinnedSessionIds.set([id])

  const hook = renderHook(() =>
    useSessionActions({
      activeSessionId: runtimeId,
      activeSessionIdRef,
      busyRef: { current: true },
      creatingSessionRef: { current: false },
      ensureSessionState: () => state,
      getRouteToken: () => routeTokenRef.current,
      getRoutedStoredSessionId: () => selectedStoredSessionIdRef.current,
      navigate,
      requestGateway,
      resetViewSync: vi.fn(),
      routedSessionId: id,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionId: id,
      selectedStoredSessionIdRef,
      sessionStateByRuntimeIdRef,
      syncSessionStateToView: vi.fn(),
      updateSessionState
    })
  )

  return {
    ...hook,
    activeSessionIdRef,
    selectedStoredSessionIdRef,
    runtimeIdByStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    navigate,
    requestGateway,
    routeTokenRef,
    updateSessionState,
    state
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(deleteSession).mockResolvedValue({ ok: true })
  vi.mocked(getProjectTreeConnectionId).mockReturnValue(null)
  $removedSessionIds.set(new Set())
  $sessionMutationsInFlight.set(new Set())
})

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $sessionTiles.set([])
  setSessions([])
  setUnlistedSessionOwnerRows([])
  $archivedSessions.set([])
  $projectTree.set([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setMessages([])
  clearQueuedPrompts(id)
  clearQueuedPrompts(runtimeId)
  $backgroundStatusBySession.set({})
  $pinnedSessionIds.set([])
  $removedSessionIds.set(new Set())
  $sessionMutationsInFlight.set(new Set())
  _resetSessionOwnerHintsForTests({ storage: true })
  clearTranscriptTails()
})

it.each([
  { owner: ownerA, deleted: ownerB, tiled: true },
  { owner: ownerA, deleted: ownerB, tiled: false },
  { owner: ownerA, deleted: { connectionId: 'remote-b', profile: 'default' }, tiled: true },
  { owner: { ...ownerA, targetProfile: 'backend-a' }, deleted: ownerA, tiled: true }
])(
  'preserves hidden streaming A when deleting search-only B ($deleted, tiled=$tiled)',
  async ({ owner, deleted, tiled }) => {
    const hidden = row(owner, true)
    setUnlistedSessionOwnerRows([hidden])
    const view = mountDeletion(owner)

    if (!tiled) {
      $sessionTiles.set([])
    }
    saveTranscriptTail(id, messages, {
      connectionId: owner.connectionId,
      profile: owner.targetProfile ?? owner.profile
    })
    saveTranscriptTail(id, messages, deleted)
    const queued = getQueuedPrompts(id)
    const runtimeQueued = getQueuedPrompts(runtimeId)
    const background = $backgroundStatusBySession.get()
    const tiles = $sessionTiles.get()

    await act(() => view.result.current.removeSession(id, row(deleted)))

    expect(deleteSession).toHaveBeenCalledWith(id, deleted)
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
    expect(view.requestGateway).not.toHaveBeenCalled()
    expect(view.navigate).not.toHaveBeenCalled()
    expect(view.updateSessionState).not.toHaveBeenCalled()
    expect($selectedStoredSessionId.get()).toBe(id)
    expect($activeSessionId.get()).toBe(runtimeId)
    expect($messages.get()).toEqual(messages)
    expect($sessionStates.get()[runtimeId]).toBe(view.state)
    expect(view.runtimeIdByStoredSessionIdRef.current.get(id)).toBe(runtimeId)
    expect(view.sessionStateByRuntimeIdRef.current.get(runtimeId)).toBe(view.state)
    expect($sessionTiles.get()).toBe(tiles)
    expect(getQueuedPrompts(id)).toEqual(queued)
    expect(getQueuedPrompts(runtimeId)).toEqual(runtimeQueued)
    expect($backgroundStatusBySession.get()).toBe(background)
    expect(getSessionOwnerHint(id)).toEqual(owner)
    expect($unlistedSessionOwnerRows.get()).toEqual([hidden])
    expect($pinnedSessionIds.get()).toEqual([id])
    expect($removedSessionIds.get().has(id)).toBe(false)
    expect($sessionMutationsInFlight.get().has(id)).toBe(false)
    expect(
      loadTranscriptTail(id, { connectionId: owner.connectionId, profile: owner.targetProfile ?? owner.profile })
    ).toEqual(messages)
    expect(loadTranscriptTail(id, deleted)).toBeNull()
    expect(refreshProjectTree).toHaveBeenCalledOnce()
  }
)

it('removes only the matching owner from listed and archived same-id rows', async () => {
  const kept = row(ownerA)
  const removed = row(ownerB)
  setSessions([kept, removed])
  $archivedSessions.set([kept, removed])
  const view = mountDeletion(ownerA)

  await act(() => view.result.current.removeSession(id, removed))

  expect($sessions.get()).toEqual([kept])
  expect($archivedSessions.get()).toEqual([kept])
  expect(view.navigate).not.toHaveBeenCalled()
  expect(getQueuedPrompts(id)).toHaveLength(1)
})

it.each([true, false])('cleans up a primary-local project session (listed=%s)', async listed => {
  const removed = row(ownerB)
  setSessions(listed ? [removed] : [])
  vi.mocked(getProjectTreeConnectionId).mockReturnValue('local')
  $projectTree.set([
    {
      id: 'local-project',
      label: 'Local project',
      path: null,
      sessionCount: 1,
      repos: [],
      previewSessions: [{ ...removed, connection_id: undefined }]
    }
  ])
  const view = mountDeletion(ownerB)

  await act(() => view.result.current.removeSession(id, listed ? removed : undefined))

  expect(requestGatewayForAgent).toHaveBeenCalledWith(ownerB.connectionId, ownerB.profile, 'session.interrupt', {
    session_id: runtimeId
  })
  expect(requestGatewayForAgent).toHaveBeenCalledWith(ownerB.connectionId, ownerB.profile, 'session.close', {
    session_id: runtimeId
  })
  expect(view.navigate).toHaveBeenCalledWith(NEW_CHAT_ROUTE, { replace: true })
  expect($selectedStoredSessionId.get()).toBeNull()
  expect($activeSessionId.get()).toBeNull()
  expect($sessions.get()).toEqual([])
  expect($sessionTiles.get()).toEqual([])
  expect(view.runtimeIdByStoredSessionIdRef.current.has(id)).toBe(false)
  expect($sessionStates.get()[runtimeId]).toBeUndefined()
  expect(getQueuedPrompts(id)).toEqual([])
  expect(getQueuedPrompts(runtimeId)).toEqual([])
  expect($backgroundStatusBySession.get()).toEqual({})
  expect($pinnedSessionIds.get()).toEqual([])
  expect($removedSessionIds.get().has(id)).toBe(true)
})

it('preserves shared slots when a bare project row came from another machine', async () => {
  const removed = row(ownerB)
  setSessions([removed])
  vi.mocked(getProjectTreeConnectionId).mockReturnValue('remote-b')
  $projectTree.set([
    {
      id: 'remote-project',
      label: 'Remote project',
      path: null,
      sessionCount: 1,
      repos: [],
      previewSessions: [{ ...removed, connection_id: undefined }]
    }
  ])
  const view = mountDeletion(ownerB)
  const queued = getQueuedPrompts(id)

  await act(() => view.result.current.removeSession(id, removed))

  expect($sessions.get()).toEqual([])
  expect(getQueuedPrompts(id)).toEqual(queued)
  expect($pinnedSessionIds.get()).toEqual([id])
  expect($removedSessionIds.get().has(id)).toBe(false)
})

it.each([true, false])(
  'preserves a newly selected twin when an outstanding delete settles (success=%s)',
  async success => {
    const pending = deferred<{ ok: boolean }>()
    vi.mocked(deleteSession).mockReturnValue(pending.promise)
    const removed = row(ownerB)
    setSessions([removed])
    const view = mountDeletion(ownerB)
    let deleting!: Promise<void>
    act(() => {
      deleting = view.result.current.removeSession(id, removed)
    })
    await waitFor(() => expect(deleteSession).toHaveBeenCalledWith(id, ownerB))

    act(() => {
      const next = { ...createClientSessionState(id, messages), busy: true, streamId: 'new-stream' }
      view.activeSessionIdRef.current = runtimeId
      view.selectedStoredSessionIdRef.current = id
      view.runtimeIdByStoredSessionIdRef.current.set(id, runtimeId)
      view.sessionStateByRuntimeIdRef.current.set(runtimeId, next)
      setActiveSessionId(runtimeId)
      setSelectedStoredSessionId(id)
      setSessionOwnerHint(id, ownerA)
      recordSessionEventScope({ connectionId: ownerA.connectionId, profile: ownerA.profile, session_id: runtimeId })
      setUnlistedSessionOwnerRows([row(ownerA, true)])
      $sessionStates.set({ [runtimeId]: next })
      $sessionTiles.set([{ storedSessionId: id, runtimeId, ownerRoute: ownerA }])
      setMessages(messages)
    })
    const state = $sessionStates.get()[runtimeId]
    view.navigate.mockClear()
    await act(async () => {
      if (success) {
        pending.resolve({ ok: true })
      } else {
        pending.reject(new Error('delete rejected'))
      }

      await deleting
    })

    expect(view.navigate).not.toHaveBeenCalled()
    expect($activeSessionId.get()).toBe(runtimeId)
    expect($selectedStoredSessionId.get()).toBe(id)
    expect($sessionStates.get()[runtimeId]).toBe(state)
    expect(view.runtimeIdByStoredSessionIdRef.current.get(id)).toBe(runtimeId)
    expect($sessionTiles.get()[0]?.ownerRoute).toEqual(ownerA)
    expect(getQueuedPrompts(id)).toHaveLength(1)
    expect(getQueuedPrompts(runtimeId)).toHaveLength(1)
    expect(getSessionOwnerHint(id, ownerA)).toEqual(ownerA)
    expect($removedSessionIds.get().has(id)).toBe(false)
  }
)

it('restores a same-owner deletion failure after its own new-chat navigation commits', async () => {
  const pending = deferred<{ ok: boolean }>()
  vi.mocked(deleteSession).mockReturnValue(pending.promise)
  const removed = row(ownerB)
  setSessions([removed])
  const view = mountDeletion(ownerB)
  let deleting!: Promise<void>
  act(() => {
    deleting = view.result.current.removeSession(id, removed)
  })
  await waitFor(() => expect(deleteSession).toHaveBeenCalledWith(id, ownerB))

  view.routeTokenRef.current = `${NEW_CHAT_ROUTE}::`
  await act(async () => {
    pending.reject(new Error('delete rejected'))
    await deleting
  })

  expect($selectedStoredSessionId.get()).toBe(id)
  expect($activeSessionId.get()).toBe(runtimeId)
  expect($messages.get()).toEqual(messages)
  expect($sessions.get()).toEqual([removed])
  expect(getQueuedPrompts(id)).toHaveLength(1)
  expect($removedSessionIds.get().has(id)).toBe(false)
})

it('rolls back only deleted-owner rows while preserving concurrent updates to the surviving twin', async () => {
  const pending = deferred<{ ok: boolean }>()
  vi.mocked(deleteSession).mockReturnValue(pending.promise)
  const kept = row(ownerA)
  const removed = row(ownerB)
  setSessions([kept, removed])
  $archivedSessions.set([kept, removed])
  const view = mountDeletion(ownerA)
  let deleting!: Promise<void>
  act(() => {
    deleting = view.result.current.removeSession(id, removed)
  })
  await waitFor(() => expect(deleteSession).toHaveBeenCalledWith(id, ownerB))

  const updated = { ...kept, message_count: kept.message_count + 1 }
  act(() => {
    setSessions([updated])
    $archivedSessions.set([updated])
  })
  await act(async () => {
    pending.reject(new Error('delete rejected'))
    await deleting
  })

  expect($sessions.get()).toEqual([removed, updated])
  expect($archivedSessions.get()).toEqual([removed, updated])
  expect(view.navigate).not.toHaveBeenCalled()
  expect(getQueuedPrompts(id)).toHaveLength(1)
  expect(getSessionOwnerHint(id)).toEqual(ownerA)
})
