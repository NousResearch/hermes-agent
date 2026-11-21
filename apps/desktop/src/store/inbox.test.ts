import { atom, type WritableAtom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { dispatchNativeNotification, notify, requestGatewayForAgent } = vi.hoisted(() => ({
  dispatchNativeNotification: vi.fn(),
  notify: vi.fn(),
  requestGatewayForAgent: vi.fn()
}))

vi.mock('@/app/routes', () => ({ INBOX_ROUTE: '/inbox' }))
vi.mock('@/store/connections', () => ({ $activeConnectionId: atom<null | string>('local') }))
vi.mock('@/store/gateway', () => ({ requestGatewayForAgent }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/session', () => ({ $gatewayState: atom('open') }))
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification }))
vi.mock('@/store/notifications', () => ({ notify }))

import { $activeConnectionId as activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'

import {
  $inboxAvailable,
  $inboxItems,
  $inboxLoaded,
  $inboxLoadError,
  $inboxUnreadCount,
  _resetInboxForTests,
  type InboxItem,
  markInboxItemRead,
  refreshInbox,
  watchInbox
} from './inbox'

// The real atom is read-only; the mock above is writable so tests can switch source.
const $activeConnectionId = activeConnectionId as WritableAtom<null | string>

const item = (id: string, over: Partial<InboxItem> = {}): InboxItem => ({
  delivered_at: 1_700_000_000,
  job_id: `job-${id}`,
  job_name: `Job ${id}`,
  preview: `result ${id}`,
  running: false,
  session_id: id,
  unread: true,
  ...over
})

function serveList(items: InboxItem[]) {
  requestGatewayForAgent.mockImplementationOnce(async () => ({
    items,
    unread_count: items.filter(entry => entry.unread).length
  }))
}

beforeEach(() => {
  $activeConnectionId.set('local')
  $activeGatewayProfile.set('default')
  $gatewayState.set('open')
  _resetInboxForTests()
  dispatchNativeNotification.mockReset()
  notify.mockReset()
  requestGatewayForAgent.mockReset()
})

describe('inbox store', () => {
  it('notifies only for unread runs that finish after the first list (the baseline)', async () => {
    // Launch: two unread results already waiting, one run still going.
    serveList([item('a'), item('b'), item('c', { running: true })])
    await refreshInbox()

    expect($inboxAvailable.get()).toBe(true)
    expect($inboxUnreadCount.get()).toBe(3)
    expect(notify).not.toHaveBeenCalled()
    expect(dispatchNativeNotification).not.toHaveBeenCalled()

    // c finishes and d lands; a and b are still listed and must not re-announce.
    serveList([item('d'), item('c'), item('a'), item('b')])
    await refreshInbox()

    expect(notify).toHaveBeenCalledTimes(1)
    expect(notify.mock.calls[0][0].message).toContain('Open the Inbox')
    expect(dispatchNativeNotification).toHaveBeenCalledTimes(1)
    expect(dispatchNativeNotification.mock.calls[0][0]).toMatchObject({ activate: '/inbox', global: true, kind: 'inbox' })

    // The same list again is not news.
    serveList([item('d'), item('c'), item('a'), item('b')])
    await refreshInbox()

    expect(notify).toHaveBeenCalledTimes(1)
  })

  it('rolls an optimistic read back when inbox.read fails, and keeps it through a stale page', async () => {
    serveList([item('a'), item('b')])
    await refreshInbox()

    requestGatewayForAgent.mockRejectedValueOnce(new Error('boom'))
    await expect(markInboxItemRead('a')).rejects.toThrow('boom')

    expect($inboxItems.get().find(entry => entry.session_id === 'a')?.unread).toBe(true)
    expect($inboxUnreadCount.get()).toBe(2)

    // A successful read whose ack races an older list page that still says unread.
    let ack!: () => void
    requestGatewayForAgent.mockImplementationOnce(() => new Promise(resolve => (ack = () => resolve({}))))
    const write = markInboxItemRead('a')

    expect($inboxUnreadCount.get()).toBe(1)

    serveList([item('a'), item('b')])
    await refreshInbox()

    expect($inboxItems.get().find(entry => entry.session_id === 'a')?.unread).toBe(false)
    expect($inboxUnreadCount.get()).toBe(1)
    ack()
    await write
    expect(requestGatewayForAgent).toHaveBeenCalledWith('local', 'default', 'inbox.read', {
      read: true,
      session_id: 'a'
    })
  })

  it('hides the inbox on a backend without inbox.list, with no error toast', async () => {
    serveList([item('a')])
    await refreshInbox()
    expect($inboxAvailable.get()).toBe(true)

    requestGatewayForAgent.mockRejectedValueOnce(Object.assign(new Error('unknown method'), { code: -32601 }))
    await refreshInbox()

    expect($inboxAvailable.get()).toBe(false)
    expect($inboxItems.get()).toEqual([])
    expect($inboxUnreadCount.get()).toBe(0)
    expect(notify).not.toHaveBeenCalled()
  })

  it('coalesces refreshes requested while one is in flight into a single follow-up', async () => {
    let release!: () => void
    requestGatewayForAgent.mockImplementationOnce(
      () => new Promise(resolve => (release = () => resolve({ items: [], unread_count: 0 })))
    )
    serveList([item('a')])

    const first = refreshInbox()
    void refreshInbox()
    void refreshInbox()
    release()
    await first

    expect(requestGatewayForAgent).toHaveBeenCalledTimes(2)
    expect($inboxItems.get().map(entry => entry.session_id)).toEqual(['a'])
  })

  it('routes inbox.list and inbox.read through the active connection, not the profile name alone', async () => {
    $activeConnectionId.set('homelab')
    $activeGatewayProfile.set('marina')
    serveList([item('a')])
    await refreshInbox()
    requestGatewayForAgent.mockResolvedValueOnce({})
    await markInboxItemRead('a')

    expect(requestGatewayForAgent).toHaveBeenNthCalledWith(1, 'homelab', 'marina', 'inbox.list', { limit: 200 })
    expect(requestGatewayForAgent).toHaveBeenNthCalledWith(2, 'homelab', 'marina', 'inbox.read', {
      read: true,
      session_id: 'a'
    })
  })

  it('drops the old scope at once on a switch, and its late reply neither paints nor announces', async () => {
    const stop = watchInbox()
    serveList([item('a')])
    await refreshInbox()
    expect($inboxItems.get()).toHaveLength(1)

    let releaseA!: () => void
    requestGatewayForAgent.mockImplementationOnce(
      () => new Promise(resolve => (releaseA = () => resolve({ items: [item('late'), item('a')], unread_count: 2 })))
    )
    const pending = refreshInbox()

    $activeGatewayProfile.set('work')

    expect($inboxItems.get()).toEqual([])
    expect($inboxUnreadCount.get()).toBe(0)

    serveList([item('b')])
    releaseA()
    await pending

    expect(notify).not.toHaveBeenCalled()
    expect($inboxItems.get().map(entry => entry.session_id)).toEqual(['b'])
    stop()
  })

  it('asks for the whole server window so every counted unread run is listed', async () => {
    serveList([item('a')])
    await refreshInbox()

    expect(requestGatewayForAgent).toHaveBeenCalledWith('local', 'default', 'inbox.list', { limit: 200 })
  })

  it('flags a failed first load (not a missing method) and clears it once a list lands', async () => {
    requestGatewayForAgent.mockRejectedValueOnce(new Error('socket hang up'))
    await refreshInbox()

    expect($inboxLoaded.get()).toBe(false)
    expect($inboxLoadError.get()).toBe(true)

    serveList([item('a')])
    await refreshInbox()

    expect($inboxLoaded.get()).toBe(true)
    expect($inboxLoadError.get()).toBe(false)

    // A failure after a landed list keeps the list and raises no error state.
    requestGatewayForAgent.mockRejectedValueOnce(new Error('socket hang up'))
    await refreshInbox()
    expect($inboxLoadError.get()).toBe(false)
    expect($inboxItems.get()).toHaveLength(1)
  })
})
