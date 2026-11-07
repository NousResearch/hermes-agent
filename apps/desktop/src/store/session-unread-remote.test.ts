import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileScope } from '@/hermes'
import type { SessionInfo } from '@/types/hermes'

const patch = vi.fn<(id: string, unread: boolean, profile?: ProfileScope) => Promise<{ ok: boolean }>>(() =>
  Promise.resolve({ ok: true })
)

vi.mock('@/hermes', () => ({
  // The store only needs the REST mutation; keep the mock minimal.
  setApiRequestProfile: () => {},
  setSessionUnreadRemote: (id: string, unread: boolean, profile?: ProfileScope) => patch(id, unread, profile)
}))

import { $sessions } from '@/store/session'

import {
  $unreadWriteGuard,
  clearUnreadOnOpen,
  markSessionUnread,
  unreadWriteKey,
  watchUnreadWriteGuard
} from './session-unread-remote'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

beforeEach(() => {
  $sessions.set([])
  $unreadWriteGuard.set(new Map())
  patch.mockReset()
  patch.mockResolvedValue({ ok: true })
})

afterEach(() => {
  $sessions.set([])
  $unreadWriteGuard.set(new Map())
})

describe('markSessionUnread', () => {
  it('optimistically paints the row, then PATCHes with the owning profile', async () => {
    $sessions.set([row('a', { profile: 'work', unread: false })])

    await markSessionUnread('a', true)

    expect(patch).toHaveBeenCalledWith('a', true, 'work')
    expect($sessions.get().find(s => s.id === 'a')?.unread).toBe(true)
  })

  it('keeps the stale-page guard after its own optimistic paint', async () => {
    let resolvePatch!: (value: { ok: boolean }) => void

    patch.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolvePatch = resolve
        })
    )
    const guarded = row('a', { profile: 'work', unread: false })

    watchUnreadWriteGuard()
    $sessions.set([guarded])
    const pending = markSessionUnread('a', true)

    expect($unreadWriteGuard.get().get(unreadWriteKey(guarded))?.value).toBe(true)
    await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(1))
    resolvePatch({ ok: true })
    await pending
  })

  it('keeps a registry row route binding on the unread PATCH', async () => {
    $sessions.set([
      row('a', {
        _desktop_route_binding: 'binding-a',
        connection_id: 'same-id',
        profile: 'work',
        unread: false
      })
    ])

    await markSessionUnread('a', true)

    expect(patch).toHaveBeenCalledWith('a', true, {
      connectionId: 'same-id',
      profile: 'work',
      sessionRouteBinding: 'binding-a',
      sessionRouteId: 'a'
    })
  })

  it('no-ops for a runtime-only session with no persisted row', async () => {
    await markSessionUnread('ghost', true)

    expect(patch).not.toHaveBeenCalled()
  })

  it('rolls back the row and rethrows when the PATCH fails', async () => {
    $sessions.set([row('a', { unread: false })])
    patch.mockImplementationOnce(() => Promise.reject(new Error('offline')))

    await expect(markSessionUnread('a', true)).rejects.toThrow('offline')

    // The backend kept the old value, so the optimistic flip is undone and
    // the guard is released (nothing to fence a page about).
    expect($sessions.get().find(s => s.id === 'a')?.unread).toBe(false)
    expect($unreadWriteGuard.get().has(unreadWriteKey(row('a')))).toBe(false)
  })

  it('does not let an older failure roll back a newer successful toggle', async () => {
    let rejectFirst!: (error: Error) => void
    patch
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejectFirst = reject
          })
      )
      .mockResolvedValueOnce({ ok: true })
    $sessions.set([row('a', { unread: false })])

    const older = markSessionUnread('a', true)
    const olderResult = older.catch(error => error)
    const newer = markSessionUnread('a', false)
    await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(1))
    rejectFirst(new Error('older write failed'))

    expect(await olderResult).toEqual(expect.objectContaining({ message: 'older write failed' }))
    await newer
    expect($sessions.get().find(s => s.id === 'a')?.unread).toBe(false)
  })

  it('restores the last confirmed value when two queued toggles both fail', async () => {
    patch.mockRejectedValue(new Error('offline'))
    $sessions.set([row('a', { unread: false })])

    const older = markSessionUnread('a', true)
    const newer = markSessionUnread('a', false)
    const olderResult = older.catch(error => error)
    const newerResult = newer.catch(error => error)

    expect(await olderResult).toEqual(expect.objectContaining({ message: 'offline' }))
    expect(await newerResult).toEqual(expect.objectContaining({ message: 'offline' }))
    expect($sessions.get().find(s => s.id === 'a')?.unread).toBe(false)
  })

  it('rolls a failed newer toggle back to an older confirmed write', async () => {
    patch.mockResolvedValueOnce({ ok: true }).mockRejectedValueOnce(new Error('offline'))
    $sessions.set([row('a', { unread: false })])

    const older = markSessionUnread('a', true)
    const newer = markSessionUnread('a', false)
    const newerResult = newer.catch(error => error)

    await older
    expect(await newerResult).toEqual(expect.objectContaining({ message: 'offline' }))
    expect($sessions.get().find(s => s.id === 'a')?.unread).toBe(true)
  })

  it('does not accept a stale page matching the latest intent while queued writes are pending', async () => {
    let resolveFirst!: (value: { ok: boolean }) => void

    patch
      .mockImplementationOnce(
        () =>
          new Promise(resolve => {
            resolveFirst = resolve
          })
      )
      .mockRejectedValueOnce(new Error('offline'))

    watchUnreadWriteGuard()
    const original = row('a', { connection_id: 'gateway-a', unread: false })

    $sessions.set([original])
    const older = markSessionUnread(original, true)
    const newer = markSessionUnread(original, false).catch(error => error)

    await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(1))

    // This page predates both PATCH settlements. Its value happens to equal
    // the queued intent, so it must not be mistaken for confirmation.
    $sessions.set([{ ...original, unread: false }])
    resolveFirst({ ok: true })

    await older
    expect(await newer).toEqual(expect.objectContaining({ message: 'offline' }))
    expect($sessions.get()[0]?.unread).toBe(true)
  })

  it('keeps same-id writes and rollback state isolated by gateway owner', async () => {
    let resolveOwnerA!: (value: { ok: boolean }) => void

    patch
      .mockImplementationOnce(
        () =>
          new Promise(resolve => {
            resolveOwnerA = resolve
          })
      )
      .mockRejectedValueOnce(new Error('owner B offline'))

    const ownerA = row('same', {
      _desktop_route_binding: 'binding-a',
      connection_id: 'gateway-a',
      profile: 'work',
      unread: false
    })

    const ownerB = row('same', {
      _desktop_route_binding: 'binding-b',
      connection_id: 'gateway-b',
      profile: 'work',
      unread: false
    })

    $sessions.set([ownerA])
    const writeA = markSessionUnread('same', true)

    $sessions.set([ownerB])
    const writeB = markSessionUnread('same', true)
    const resultB = writeB.catch(error => error)

    await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(2))
    resolveOwnerA({ ok: true })
    await writeA
    expect(await resultB).toEqual(expect.objectContaining({ message: 'owner B offline' }))
    expect($sessions.get()[0]).toMatchObject({ connection_id: 'gateway-b', unread: false })
  })

  it('uses the clicked row owner when same-id rows coexist', async () => {
    const ownerA = row('same', { connection_id: 'gateway-a', profile: 'work', unread: false })
    const ownerB = row('same', { connection_id: 'gateway-b', profile: 'work', unread: false })

    $sessions.set([ownerA, ownerB])
    await markSessionUnread(ownerB, true)

    expect(patch).toHaveBeenCalledWith('same', true, { connectionId: 'gateway-b', profile: 'work' })
    expect($sessions.get()).toEqual([ownerA, { ...ownerB, unread: true }])
  })

  it('does not resolve a legacy string caller across duplicate owners', async () => {
    $sessions.set([
      row('same', { connection_id: 'gateway-a', unread: false }),
      row('same', { connection_id: 'gateway-b', unread: false })
    ])

    await markSessionUnread('same', true)

    expect(patch).not.toHaveBeenCalled()
    expect($sessions.get().every(session => session.unread === false)).toBe(true)
  })

  it('does not let a rejected write repaint a newer confirming list page', async () => {
    let rejectPatch!: (error: Error) => void

    patch.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectPatch = reject
        })
    )
    watchUnreadWriteGuard()
    const original = row('a', { connection_id: 'gateway-a', unread: false })

    $sessions.set([original])
    const pending = markSessionUnread(original, true)
    const result = pending.catch(error => error)

    await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(1))
    $sessions.set([{ ...original, unread: true }])
    rejectPatch(new Error('late failure'))

    expect(await result).toEqual(expect.objectContaining({ message: 'late failure' }))
    expect($sessions.get()[0]?.unread).toBe(true)
  })
})

describe('clearUnreadOnOpen', () => {
  it('no-ops for a session that is already read', async () => {
    $sessions.set([row('a', { unread: false })])

    await clearUnreadOnOpen('a')

    expect(patch).not.toHaveBeenCalled()
  })

  it('PATCHes read for an unread session, using its owning profile', async () => {
    $sessions.set([row('a', { profile: 'p2', unread: true })])

    await clearUnreadOnOpen('a')

    expect(patch).toHaveBeenCalledWith('a', false, 'p2')
    expect($sessions.get().find(s => s.id === 'a')?.unread).toBe(false)
  })

  it('clears only the clicked owner when duplicate ids coexist', async () => {
    const ownerA = row('same', { connection_id: 'gateway-a', profile: 'work', unread: true })
    const ownerB = row('same', { connection_id: 'gateway-b', profile: 'work', unread: true })

    $sessions.set([ownerA, ownerB])
    await clearUnreadOnOpen(ownerB)

    expect(patch).toHaveBeenCalledWith('same', false, { connectionId: 'gateway-b', profile: 'work' })
    expect($sessions.get()).toEqual([ownerA, { ...ownerB, unread: false }])
  })

  it('swallows a failed PATCH (the next honest refresh heals the dot)', async () => {
    $sessions.set([row('a', { unread: true })])
    patch.mockImplementationOnce(() => Promise.reject(new Error('offline')))

    await expect(clearUnreadOnOpen('a')).resolves.toBeUndefined()
  })
})

describe('watchUnreadWriteGuard', () => {
  it('drops a guard entry once a list page confirms the value we wrote', () => {
    watchUnreadWriteGuard()
    const guard = new Map<string, { at: number; value: boolean }>()
    const guarded = row('a', { unread: true })

    guard.set(unreadWriteKey(guarded), { at: Date.now(), value: true })
    $unreadWriteGuard.set(guard)

    // The server caught up and echoes our value back.
    $sessions.set([guarded])

    expect($unreadWriteGuard.get().has(unreadWriteKey(guarded))).toBe(false)
  })

  it('keeps the guard while a page contradicts a write still in flight', () => {
    watchUnreadWriteGuard()
    const guard = new Map<string, { at: number; value: boolean }>()
    const guarded = row('a', { unread: false })

    guard.set(unreadWriteKey(guarded), { at: Date.now(), value: true })
    $unreadWriteGuard.set(guard)

    // A list request issued before the PATCH still says read. Honouring it
    // would silently undo the mark the user just made.
    $sessions.set([guarded])

    expect($unreadWriteGuard.get().has(unreadWriteKey(guarded))).toBe(true)
  })
})
