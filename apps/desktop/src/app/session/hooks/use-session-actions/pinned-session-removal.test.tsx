// Regression for #139350: optimistic pin cleanup must not race the fenced removal.
import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { deleteSession, type SessionInfo, setSessionArchived, setSessionPinnedRemote } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import { $sessions, setSessions } from '@/store/session'
import { resetSessionPinMirror, watchSessionPins } from '@/store/session-pin-sync'
import { $removedSessionIds, $sessionMutationsInFlight } from '@/store/session-removal'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  setSessionArchived: vi.fn(),
  setSessionPinnedRemote: vi.fn().mockResolvedValue({ ok: true })
}))

type Handle = Pick<ReturnType<typeof useSessionActions>, 'archiveSession' | 'removeSession'>

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

  onReady(actions)

  return null
}

beforeAll(() => {
  window.hermesDesktop = {} as typeof window.hermesDesktop
  watchSessionPins()
})

beforeEach(() => {
  resetSessionPinMirror()
  setSessions([])
  $pinnedSessionIds.set([])
  $removedSessionIds.set(new Set())
  $sessionMutationsInFlight.set(new Set())
  vi.clearAllMocks()
  // Pins use the lineage root while archive/delete targets the current tip.
  setSessions([
    {
      id: 'tip',
      _lineage_root_id: 'root',
      pinned: true,
      profile: 'work',
      message_count: 2,
      source: 'desktop',
      started_at: 1,
      title: 'pinned chat'
    } as SessionInfo
  ])
  expect($pinnedSessionIds.get()).toEqual(['root'])
})

afterEach(() => {
  cleanup()
  resetSessionPinMirror()
  setSessions([])
  $pinnedSessionIds.set([])
  $removedSessionIds.set(new Set())
  $sessionMutationsInFlight.set(new Set())
})

it.each(['archiveSession', 'removeSession'] as const)('%s does not issue a competing unpin', async action => {
  let handle!: Handle
  render(
    <Harness
      onReady={value => {
        handle = value
      }}
    />
  )
  const mutate = action === 'archiveSession' ? vi.mocked(setSessionArchived) : vi.mocked(deleteSession)
  let resolve!: (value: { ok: boolean }) => void
  mutate.mockImplementation(
    () =>
      new Promise(done => {
        resolve = done
      })
  )

  await act(async () => {
    const removal = handle[action]('tip')
    // The row disappears immediately while the single server mutation is pending.
    expect($sessions.get()).toEqual([])
    expect($pinnedSessionIds.get()).toEqual([])
    expect(setSessionPinnedRemote).not.toHaveBeenCalled()
    resolve({ ok: true })
    await removal
  })

  expect(mutate).toHaveBeenCalledTimes(1)
  expect($pinnedSessionIds.get()).toEqual([])
  expect(setSessionPinnedRemote).not.toHaveBeenCalled()
})

it.each(['archiveSession', 'removeSession'] as const)(
  '%s restores a rejected removal without rewriting its pin',
  async action => {
    let handle!: Handle
    render(
      <Harness
        onReady={value => {
          handle = value
        }}
      />
    )
    const mutate = action === 'archiveSession' ? vi.mocked(setSessionArchived) : vi.mocked(deleteSession)
    mutate.mockRejectedValue(new Error('permission_denied'))

    await act(() => handle[action]('tip'))

    expect($sessions.get().map(row => row.id)).toEqual(['tip'])
    expect($pinnedSessionIds.get()).toEqual(['root'])
    expect(setSessionPinnedRemote).not.toHaveBeenCalled()
  }
)
