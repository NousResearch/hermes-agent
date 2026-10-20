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

import { $unreadWriteGuard, clearUnreadOnOpen, markSessionUnread, watchUnreadWriteGuard } from './session-unread-remote'

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
    expect($unreadWriteGuard.get().has('a')).toBe(false)
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
    guard.set('a', { at: Date.now(), value: true })
    $unreadWriteGuard.set(guard)

    // The server caught up and echoes our value back.
    $sessions.set([row('a', { unread: true })])

    expect($unreadWriteGuard.get().has('a')).toBe(false)
  })

  it('keeps the guard while a page contradicts a write still in flight', () => {
    watchUnreadWriteGuard()
    const guard = new Map<string, { at: number; value: boolean }>()
    guard.set('a', { at: Date.now(), value: true })
    $unreadWriteGuard.set(guard)

    // A list request issued before the PATCH still says read. Honouring it
    // would silently undo the mark the user just made.
    $sessions.set([row('a', { unread: false })])

    expect($unreadWriteGuard.get().has('a')).toBe(true)
  })
})
