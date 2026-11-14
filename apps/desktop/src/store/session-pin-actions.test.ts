import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { buildSessionByAnyId, resolvePinnedSessions } from '@/app/chat/sidebar/session-index'
import type { HermesConnection } from '@/global'
import { connectionScopeSuffix } from '@/lib/connection-scoped'
import { storedStringArray } from '@/lib/storage'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $pinnedSessionIds } from '@/store/layout'
import { $cronSessions, $messagingSessions, $sessions, setConnection } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

const { patch, notifyError } = vi.hoisted(() => ({
  patch: vi.fn().mockResolvedValue({ ok: true }),
  notifyError: vi.fn()
}))

vi.mock('@/hermes', () => ({ setApiRequestProfile: () => {}, setSessionPinnedRemote: patch }))
vi.mock('@/store/notifications', () => ({ notifyError }))

import {
  $unconfirmedPinWrites,
  pinSession,
  resetSessionPinMirror,
  unpinSession,
  watchSessionPins
} from './session-pin-sync'

const connection = (id: string): HermesConnection =>
  ({
    connectionId: id,
    mode: id === 'local' ? 'local' : 'remote',
    baseUrl: `https://${id}.example`,
    profile: id,
    token: 'synthetic',
    wsUrl: 'ws://synthetic'
  }) as HermesConnection

const row = (id: string, pinned: boolean): SessionInfo =>
  ({
    id: `session-${id}`,
    connection_id: id,
    profile: id,
    pinned,
    title: 'Synthetic chat',
    message_count: 1,
    source: 'local',
    started_at: 0
  }) as SessionInfo

const cache = (owner: HermesConnection) =>
  storedStringArray(`hermes.desktop.pinnedSessions${connectionScopeSuffix(owner, false)}`)

const flush = async () => {
  for (let n = 0; n < 10; n++) {
    await Promise.resolve()
  }
}

const visiblePins = () =>
  resolvePinnedSessions(
    $pinnedSessionIds.get(),
    buildSessionByAnyId($sessions.get(), [], []),
    $sessions.get(),
    $unconfirmedPinWrites.get()
  )

beforeAll(() => watchSessionPins())
beforeEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $pinnedSessionIds.set([])
  resetSessionPinMirror()
  window.localStorage.clear()
  setApiRequestConnection(null)
  $primaryGatewayRoute.set({ connectionId: 'a', mode: 'remote' })
  setConnection(connection('a'))
  Object.assign(window, {
    hermesDesktop: { getConnectionFor: vi.fn(async ({ connectionId }) => connection(connectionId)) }
  })
  patch.mockReset().mockResolvedValue({ ok: true })
  notifyError.mockReset()
})
afterEach(() => {
  // Real pages confirm both outcomes and retire owner-scoped write guards.
  $sessions.set([row('b', true), row('local', true)])
  $sessions.set([row('b', false), row('local', false)])
  $sessions.set([])
  $pinnedSessionIds.set([])
  resetSessionPinMirror()
})

it('does not reassert a stale legacy cache after a confirmed unpin on a local sibling', async () => {
  const remote = { ...connection('a'), connectionId: undefined }
  const local = { ...connection('local'), connectionId: undefined }
  const session = { ...row('local', true), connection_id: undefined }
  $primaryGatewayRoute.set({ connectionId: null, mode: 'remote' })
  setConnection(remote)
  $sessions.set([session])
  setConnection(local)
  unpinSession(session.id, session)
  await flush()
  expect(patch.mock.calls).toEqual([[session.id, false, 'local']])

  $sessions.set([{ ...session, pinned: false }])
  patch.mockClear()
  setConnection(remote)
  await flush()
  expect(patch).not.toHaveBeenCalled()
  expect($pinnedSessionIds.get()).toEqual([])
  expect(cache(remote)).toEqual([])
  setConnection(local)
  setConnection(remote)
  await flush()
  expect(patch).not.toHaveBeenCalled()
  expect($pinnedSessionIds.get()).toEqual([])
})

it.each(['b', 'local'])('pins a selected %s row through its owner across switches and stale pages', async ownerId => {
  const owner = connection(ownerId)
  const session = { ...row(ownerId, false), id: 'tip', _lineage_root_id: 'root' }
  const twin = { ...session, connection_id: 'a' }
  $sessions.set([twin, session])
  let acknowledge!: () => void
  patch.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        acknowledge = () => resolve({ ok: true })
      })
  )

  pinSession('root', session, 0)
  await flush()
  expect(patch.mock.calls).toEqual([['root', true, { connectionId: ownerId, profile: ownerId }]])
  expect(cache(connection('a'))).toEqual([])
  expect(cache(owner)).toEqual(['root'])
  expect(visiblePins()).toMatchObject([{ connection_id: ownerId, pinned: true }])
  expect($sessions.get()[0].pinned).toBe(false)

  setConnection(owner)
  $sessions.set([session]) // stale false while the write is pending
  expect($pinnedSessionIds.get()).toEqual(['root'])
  expect(visiblePins()).toMatchObject([{ pinned: true }])
  setConnection(connection('c'))
  acknowledge()
  await flush()
  setConnection(owner)
  expect(patch).toHaveBeenCalledTimes(1)
  expect($pinnedSessionIds.get()).toEqual(['root'])

  $sessions.set([{ ...session, pinned: true }])
  $sessions.set([session]) // a later real server unpin wins
  expect($pinnedSessionIds.get()).toEqual([])
  expect(cache(owner)).toEqual([])
})

it('rolls back a rejected foreign pin and leaves it retryable', async () => {
  const session = row('b', false)
  $sessions.set([session])
  const failure = new Error('permission denied')
  patch.mockRejectedValueOnce(failure)
  pinSession(session.id, session)
  await flush()
  expect(notifyError).toHaveBeenCalledWith(failure, expect.any(String))
  expect(visiblePins()).toEqual([])
  expect(cache(connection('b'))).toEqual([])
  expect(cache(connection('a'))).toEqual([])

  pinSession(session.id, session)
  await flush()
  expect(patch).toHaveBeenCalledTimes(2)
  expect(visiblePins()).toMatchObject([{ pinned: true }])
})

it('serializes a foreign pin followed by unpin while its PATCH is pending', async () => {
  const session = row('b', false)
  $sessions.set([session])
  let acknowledge!: () => void
  patch.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        acknowledge = () => resolve({ ok: true })
      })
  )
  pinSession(session.id, session)
  await flush()
  unpinSession(session.id, $sessions.get()[0])
  await flush()
  expect(patch).toHaveBeenCalledTimes(1)
  expect(visiblePins()).toEqual([])
  expect(cache(connection('b'))).toEqual([])

  acknowledge()
  await flush()
  expect(patch.mock.calls).toEqual([
    [session.id, true, { connectionId: 'b', profile: 'b' }],
    [session.id, false, { connectionId: 'b', profile: 'b' }]
  ])
  setConnection(connection('b'))
  expect($pinnedSessionIds.get()).toEqual([])
})

it('restores the original state when two opposite pending foreign writes both fail', async () => {
  const session = row('b', false)
  $sessions.set([session])
  let rejectPin!: (error: Error) => void
  patch.mockImplementationOnce(
    () =>
      new Promise((_, reject) => {
        rejectPin = reject
      })
  )
  pinSession(session.id, session)
  await flush()
  unpinSession(session.id, $sessions.get()[0])
  await flush()
  patch.mockRejectedValueOnce(new Error('unpin denied'))
  rejectPin(new Error('pin denied'))
  await flush()
  expect(patch).toHaveBeenCalledTimes(2)
  expect(visiblePins()).toEqual([])
  expect(cache(connection('b'))).toEqual([])
  expect(notifyError).toHaveBeenCalledTimes(1)
})
