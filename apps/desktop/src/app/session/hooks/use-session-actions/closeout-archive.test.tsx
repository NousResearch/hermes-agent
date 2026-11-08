import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $pinnedSessionIds } from '@/store/layout'
import { $sessions, setSessions } from '@/store/session'
import { $sessionTiles, clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { $archivedSessions } from '@/store/sidebar-archive'
import { makeSessionInfo } from '@/test/session-info'

import { useSessionActions } from './index'

const { request, plainArchive } = vi.hoisted(() => ({
  request: vi.fn(),
  plainArchive: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@/store/gateway', async original => ({ ...(await original<object>()), requestGatewayForAgent: request }))
vi.mock('@/hermes', async original => ({ ...(await original<object>()), setSessionArchived: plainArchive }))

const receipt = () => ({
  closeout_id: 'closeout',
  status: 'preserved',
  archived: true,
  settled: false,
  old: { logical_session_id: 'old', stored_session_id: 'old', runtime_id: 'runtime-old', profile: 'work' },
  receipt_id: 'durable',
  destinations: { memory: 'pending', tasks: 'pending' },
  next_create_request_id: 'new-stable'
})

function mount() {
  const ref = <T,>(current: T) => ({ current })
  const state = { ...createClientSessionState('old'), busy: true }
  publishSessionState('runtime-old', state)

  return renderHook(() =>
    useSessionActions({
      activeSessionId: null,
      activeSessionIdRef: ref<string | null>(null),
      busyRef: ref(false),
      creatingSessionRef: ref(false),
      ensureSessionState: () => state,
      getRouteToken: () => 'route',
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn() as never,
      requestGateway: request,
      resetViewSync: vi.fn(),
      routedSessionId: null,
      runtimeIdByStoredSessionIdRef: ref(new Map([['old', 'runtime-old']])),
      selectedStoredSessionId: null,
      selectedStoredSessionIdRef: ref<string | null>(null),
      sessionStateByRuntimeIdRef: ref(new Map([['runtime-old', state]])),
      syncSessionStateToView: vi.fn(),
      updateSessionState: () => state
    })
  )
}

beforeEach(() => {
  request.mockReset()
  plainArchive.mockClear()
  clearAllSessionStates()
  setSessions([makeSessionInfo({ id: 'old', connection_id: 'cloud-a', profile: 'work', pinned: true })])
  $pinnedSessionIds.set(['old'])
  $sessionTiles.set([
    { storedSessionId: 'old', runtimeId: 'runtime-old', ownerRoute: { connectionId: 'cloud-a', profile: 'work' } }
  ])
})
afterEach(() => {
  cleanup()
  clearAllSessionStates()
  setSessions([])
  $sessionTiles.set([])
})

it('keeps the transcript and pins visible until a durable preserved archive receipt, including pending online destinations', async () => {
  let finish!: (value: unknown) => void
  request.mockImplementation(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  const { result } = mount()
  let archived!: Promise<unknown>
  act(() => {
    archived = result.current.archiveSession('old')
  })
  await vi.waitFor(() => expect(request).toHaveBeenCalled())
  expect($sessions.get().map(row => row.id)).toContain('old')
  expect($pinnedSessionIds.get()).toContain('old')
  await act(async () => {
    finish(receipt())
    await archived
  })
  expect(plainArchive).not.toHaveBeenCalled()
  expect(request).toHaveBeenCalledWith(
    'cloud-a',
    'work',
    'session.closeout',
    expect.objectContaining({ intent: 'archive', session_id: 'runtime-old' })
  )
  expect($sessions.get().map(row => row.id)).not.toContain('old')
})

it('keeps history intact when the backend lacks preservation; never falls back to a plain archive', async () => {
  request.mockRejectedValue(new Error('Method not found: session.closeout'))
  const { result } = mount()
  await act(async () => {
    await result.current.archiveSession('old')
  })
  expect(plainArchive).not.toHaveBeenCalled()
  expect($sessions.get().map(row => row.id)).toContain('old')
  expect($pinnedSessionIds.get()).toContain('old')
})

it('archives a restored conversation’s new work under a fresh request identity', async () => {
  const operations = new Map<string, ReturnType<typeof receipt>>()
  request.mockImplementation(async (_connection, _profile, _method, params) => {
    if (!operations.has(params.request_id)) {
      operations.set(params.request_id, { ...receipt(), closeout_id: `operation-${operations.size}` })
    }

    return operations.get(params.request_id)
  })
  const original = $sessions.get()[0]!
  const { result } = mount()
  await act(async () => { await result.current.archiveSession('old') })
  $archivedSessions.set([{ ...original, archived: true }])
  await act(async () => { await result.current.unarchiveSession('old') })
  expect(plainArchive).toHaveBeenCalledWith('old', false, 'work')
  expect($sessions.get().some(row => row.id === 'old')).toBe(true)
  publishSessionState('runtime-old', { ...createClientSessionState('old'), messages: [{ id: 'new-work', role: 'user', parts: [{ type: 'text', text: 'new work after restore' }] }] })
  await act(async () => { await result.current.archiveSession('old') })
  expect(operations.size).toBe(2)
  expect($sessions.get().some(row => row.id === 'old')).toBe(false)
})
