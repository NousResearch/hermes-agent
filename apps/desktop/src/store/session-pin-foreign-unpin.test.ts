import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { buildSessionByAnyId, resolvePinnedSessions } from '@/app/chat/sidebar/session-index'
import type { HermesConnection } from '@/global'
import { connectionScopeSuffix } from '@/lib/connection-scoped'
import { storedStringArray } from '@/lib/storage'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $pinnedSessionIds, pinSession } from '@/store/layout'
import { $cronSessions, $messagingSessions, $sessions, setConnection } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

const { patch, notifyError } = vi.hoisted(() => ({
  patch: vi.fn().mockResolvedValue({ ok: true }),
  notifyError: vi.fn()
}))

vi.mock('@/hermes', () => ({ setApiRequestProfile: () => {}, setSessionPinnedRemote: patch }))
vi.mock('@/store/notifications', () => ({ notifyError }))

import { $unconfirmedPinWrites, resetSessionPinMirror, unpinSession, watchSessionPins } from './session-pin-sync'

const connection = (connectionId?: string): HermesConnection =>
  ({
    connectionId,
    mode: connectionId === 'local' ? 'local' : 'remote',
    baseUrl: `https://${connectionId ?? 'primary'}.example`,
    profile: 'work',
    token: 'test',
    wsUrl: 'ws://test'
  }) as HermesConnection

const row = (connectionId: string, pinned = true): SessionInfo =>
  ({
    id: 'tip',
    _lineage_root_id: 'root',
    connection_id: connectionId,
    profile: 'work',
    pinned,
    source: 'cli',
    message_count: 1,
    started_at: 0,
    title: 'Pinned chat'
  }) as SessionInfo

function visiblePins(): SessionInfo[] {
  const rows = [...$sessions.get(), ...$cronSessions.get(), ...$messagingSessions.get()]

  return resolvePinnedSessions(
    $pinnedSessionIds.get(),
    buildSessionByAnyId(rows, [], []),
    rows,
    $unconfirmedPinWrites.get()
  )
}

beforeAll(() => {
  watchSessionPins()
})

beforeEach(() => {
  window.localStorage.clear()
  setApiRequestConnection(null)
  Object.assign(window, {
    hermesDesktop: { getConnectionFor: vi.fn(async ({ connectionId }) => connection(connectionId)) }
  })
  $primaryGatewayRoute.set({ connectionId: null, mode: 'remote' })
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  setConnection(connection())
  $pinnedSessionIds.set([])
  resetSessionPinMirror()
  patch.mockReset().mockResolvedValue({ ok: true })
  notifyError.mockClear()
})

afterEach(() => {
  setApiRequestConnection(null)
  // A real confirming page retires the owner-scoped intent, just as in the app.
  $sessions.set([row('local', false), row('remote-owner', false)])
  $cronSessions.set([])
  $messagingSessions.set([])
  $pinnedSessionIds.set([])
  resetSessionPinMirror()
})

it.each(['local', 'remote-owner'])(
  'unpins %s through its owner and cannot resurrect across gateway switches',
  async ownerId => {
    const owner = connection(ownerId)
    const pinnedRow = row(ownerId)
    const cacheKey = `hermes.desktop.pinnedSessions${connectionScopeSuffix(owner, false)}`
    setConnection(owner)
    $sessions.set([pinnedRow])
    expect($pinnedSessionIds.get()).toEqual(['root'])

    setConnection(connection())
    expect($pinnedSessionIds.get()).toEqual([])
    expect(visiblePins()).toHaveLength(1)
    patch.mockClear()

    let resolveOwner!: (value: HermesConnection) => void
    vi.mocked(window.hermesDesktop.getConnectionFor!).mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveOwner = resolve
        })
    )
    unpinSession('root')
    setConnection(owner) // descriptor lookup has not resolved: purge before loading the cache
    expect($pinnedSessionIds.get()).toEqual([])
    expect(storedStringArray(cacheKey)).toEqual([])
    setConnection(connection())
    resolveOwner(owner)
    await vi.waitFor(() =>
      expect(patch).toHaveBeenCalledWith('root', false, { connectionId: ownerId, profile: 'work' })
    )
    expect(visiblePins()).toEqual([])
    expect(storedStringArray(cacheKey)).toEqual([])
    const twin = { ...pinnedRow, connection_id: 'unrelated' }
    $messagingSessions.set([twin])
    expect(visiblePins()).toMatchObject([twin])
    expect($messagingSessions.get()[0].pinned).toBe(true)
    $messagingSessions.set([])

    // An optimistic value-copy is not a server confirmation. Unrelated page
    // reconciliation republishes the fence and must not drop this intent.
    $sessions.set($sessions.get().map(session => ({ ...session })))
    $cronSessions.set([pinnedRow])
    expect(visiblePins()).toEqual([])

    setConnection(connection('third'))
    $sessions.set([pinnedRow]) // stale page received after PATCH ACK + rescope
    expect(visiblePins()).toEqual([])
    setConnection(owner)
    expect($pinnedSessionIds.get()).toEqual([])
    expect(storedStringArray(cacheKey)).toEqual([])
    setConnection(connection())
    setConnection(owner) // reload storage again: no suppressed persistence leak
    expect(patch.mock.calls).toEqual([['root', false, { connectionId: ownerId, profile: 'work' }]])

    $cronSessions.set([])
    $sessions.set([row(ownerId, false)])
    $sessions.set([pinnedRow]) // a later genuine server pin can win again
    expect(visiblePins()).toHaveLength(1)

    setConnection(connection())
    unpinSession('root')
    await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(2))
    setConnection(owner)
    pinSession('root') // a deliberate re-pin supersedes the old unpin guard
    expect($pinnedSessionIds.get()).toEqual(['root'])
    expect(patch).toHaveBeenLastCalledWith('root', true, 'work')

    // Legacy remote descriptors can omit the id while the API still names
    // an explicit registry owner. That pin remains valid routing authority.
    if (ownerId === 'remote-owner') {
      setApiRequestConnection(ownerId)
      setConnection({ ...owner, connectionId: undefined })
      $sessions.set([{ ...pinnedRow, id: 'descriptor-tip', _lineage_root_id: 'descriptor-root' }])
      expect($pinnedSessionIds.get()).toContain('descriptor-root')
    }
  }
)

it('rolls back a rejected foreign unpin without writing to the active gateway', async () => {
  const pinnedRow = row('local')
  setConnection(connection('local'))
  $messagingSessions.set([pinnedRow])
  setConnection(connection())
  const failure = new Error('permission denied')
  patch.mockRejectedValueOnce(failure)

  unpinSession('root')
  await vi.waitFor(() => expect(notifyError).toHaveBeenCalledWith(failure, expect.any(String)))
  expect(patch.mock.calls).toEqual([['root', false, { connectionId: 'local', profile: 'work' }]])
  expect($pinnedSessionIds.get()).toEqual([])
  expect(visiblePins()).toMatchObject([pinnedRow])
  expect(storedStringArray('hermes.desktop.pinnedSessions')).toEqual(['root'])

  patch.mockResolvedValue({ ok: true })
  unpinSession('root') // rollback leaves the same action available to retry
  await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(2))
  expect(visiblePins()).toEqual([])
})
