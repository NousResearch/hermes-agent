import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'

vi.mock('@/lib/media', () => ({
  downloadGatewayMediaFile: vi.fn()
}))

const openDesktopPathWithDefaultApp = vi.hoisted(() => vi.fn(async () => {}))

vi.mock('@/lib/desktop-fs', () => ({
  copyTextToClipboard: vi.fn(),
  isDesktopFsRemoteMode: () => false,
  openDesktopPathWithDefaultApp,
  renameDesktopPath: vi.fn(),
  revealDesktopPath: vi.fn(),
  trashDesktopPath: vi.fn()
}))

const media = await import('@/lib/media')
const downloadGatewayMediaFile = vi.mocked(media.downloadGatewayMediaFile)

const { downloadRemoteFile, isRecentOpenWithDefaultApp, openFileWithDefaultApp, shouldOfferRemoteFileDownload } =
  await import('./file-actions')

describe('shouldOfferRemoteFileDownload', () => {
  it('is only for files on a remote backend', () => {
    expect(shouldOfferRemoteFileDownload(false, true)).toBe(true)
    expect(shouldOfferRemoteFileDownload(true, true)).toBe(false)
    expect(shouldOfferRemoteFileDownload(false, false)).toBe(false)
    expect(shouldOfferRemoteFileDownload(true, false)).toBe(false)
  })
})

describe('downloadRemoteFile', () => {
  beforeEach(() => {
    clearNotifications()
    downloadGatewayMediaFile.mockReset()
  })

  afterEach(() => {
    clearNotifications()
  })

  it('saves a remote gateway file through the native download bridge', async () => {
    downloadGatewayMediaFile.mockResolvedValue({ path: '/Users/me/Downloads/notes.md', saved: true })

    await downloadRemoteFile('/home/linux/project/notes.md')

    expect(downloadGatewayMediaFile).toHaveBeenCalledWith('/home/linux/project/notes.md')
    expect($notifications.get()[0]?.message).toBe('Saved')
  })

  it('stays quiet when the save dialog is canceled', async () => {
    downloadGatewayMediaFile.mockResolvedValue({ canceled: true, saved: false })

    await downloadRemoteFile('/home/linux/project/notes.md')

    expect($notifications.get()).toEqual([])
  })

  it('toasts when the gateway download fails', async () => {
    downloadGatewayMediaFile.mockRejectedValue(new Error('Desktop file download bridge is unavailable'))

    await downloadRemoteFile('/home/linux/project/notes.md')

    expect($notifications.get()[0]?.kind).toBe('error')
    expect($notifications.get()[0]?.title).toBe('Download failed')
  })
})

describe('openFileWithDefaultApp fall-through suppression', () => {
  let fakeNow = 1_000_000

  beforeEach(() => {
    fakeNow = 1_000_000
    vi.spyOn(Date, 'now').mockImplementation(() => fakeNow)
  })

  afterEach(() => {
    vi.mocked(Date.now).mockRestore()
    openDesktopPathWithDefaultApp.mockReset()
    openDesktopPathWithDefaultApp.mockImplementation(async () => {})
  })

  it('stamps the instant so the row-activation fall-through is suppressed', async () => {
    expect(isRecentOpenWithDefaultApp()).toBe(false)

    await openFileWithDefaultApp('/tmp/报告.xlsx')

    expect(openDesktopPathWithDefaultApp).toHaveBeenCalledWith('/tmp/报告.xlsx')
    // the menu-close click lands on the row within a frame or two of the item
    // handler — still suppressed
    fakeNow += 50
    expect(isRecentOpenWithDefaultApp()).toBe(true)

    // but it does not leak: a normal click later activates the preview
    fakeNow += 1000
    expect(isRecentOpenWithDefaultApp()).toBe(false)
  })

  it('still stamps when the OS open fails (the toast replaces the preview)', async () => {
    openDesktopPathWithDefaultApp.mockRejectedValueOnce(new Error('no association'))

    await openFileWithDefaultApp('/tmp/x.unknown')

    expect(isRecentOpenWithDefaultApp()).toBe(true)
  })
})
