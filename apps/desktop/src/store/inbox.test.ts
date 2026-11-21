import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { dispatchNativeNotification, notify, requestGatewayForProfile } = vi.hoisted(() => ({
  dispatchNativeNotification: vi.fn(),
  notify: vi.fn(),
  requestGatewayForProfile: vi.fn()
}))

vi.mock('@/app/routes', () => ({ INBOX_ROUTE: '/inbox' }))
vi.mock('@/store/connections', () => ({ $activeConnectionId: atom<null | string>('local') }))
vi.mock('@/store/gateway', () => ({ requestGatewayForProfile }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/session', () => ({ $gatewayState: atom('open') }))
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification }))
vi.mock('@/store/notifications', () => ({ notify }))

import {
  $inboxAvailable,
  $inboxItems,
  $inboxUnreadCount,
  _resetInboxForTests,
  type InboxItem,
  markInboxItemRead,
  refreshInbox
} from './inbox'

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
  requestGatewayForProfile.mockImplementationOnce(async () => ({
    items,
    unread_count: items.filter(entry => entry.unread).length
  }))
}

beforeEach(() => {
  _resetInboxForTests()
  dispatchNativeNotification.mockReset()
  notify.mockReset()
  requestGatewayForProfile.mockReset()
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

    requestGatewayForProfile.mockRejectedValueOnce(new Error('boom'))
    await expect(markInboxItemRead('a')).rejects.toThrow('boom')

    expect($inboxItems.get().find(entry => entry.session_id === 'a')?.unread).toBe(true)
    expect($inboxUnreadCount.get()).toBe(2)

    // A successful read whose ack races an older list page that still says unread.
    let ack!: () => void
    requestGatewayForProfile.mockImplementationOnce(() => new Promise(resolve => (ack = () => resolve({}))))
    const write = markInboxItemRead('a')

    expect($inboxUnreadCount.get()).toBe(1)

    serveList([item('a'), item('b')])
    await refreshInbox()

    expect($inboxItems.get().find(entry => entry.session_id === 'a')?.unread).toBe(false)
    expect($inboxUnreadCount.get()).toBe(1)
    ack()
    await write
    expect(requestGatewayForProfile).toHaveBeenCalledWith(expect.any(String), 'inbox.read', {
      read: true,
      session_id: 'a'
    })
  })

  it('hides the inbox on a backend without inbox.list, with no error toast', async () => {
    serveList([item('a')])
    await refreshInbox()
    expect($inboxAvailable.get()).toBe(true)

    requestGatewayForProfile.mockRejectedValueOnce(Object.assign(new Error('unknown method'), { code: -32601 }))
    await refreshInbox()

    expect($inboxAvailable.get()).toBe(false)
    expect($inboxItems.get()).toEqual([])
    expect($inboxUnreadCount.get()).toBe(0)
    expect(notify).not.toHaveBeenCalled()
  })

  it('coalesces refreshes requested while one is in flight into a single follow-up', async () => {
    let release!: () => void
    requestGatewayForProfile.mockImplementationOnce(
      () => new Promise(resolve => (release = () => resolve({ items: [], unread_count: 0 })))
    )
    serveList([item('a')])

    const first = refreshInbox()
    void refreshInbox()
    void refreshInbox()
    release()
    await first

    expect(requestGatewayForProfile).toHaveBeenCalledTimes(2)
    expect($inboxItems.get().map(entry => entry.session_id)).toEqual(['a'])
  })
})
