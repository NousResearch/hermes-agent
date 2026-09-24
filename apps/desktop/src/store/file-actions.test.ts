import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'

vi.mock('@/lib/media', () => ({
  downloadGatewayMediaFile: vi.fn()
}))

const media = await import('@/lib/media')
const downloadGatewayMediaFile = vi.mocked(media.downloadGatewayMediaFile)

const { downloadRemoteFile, shouldOfferLocalReveal, shouldOfferRemoteFileDownload } = await import('./file-actions')

describe('shouldOfferRemoteFileDownload', () => {
  it('is only for files on a remote backend', () => {
    expect(shouldOfferRemoteFileDownload(false, true)).toBe(true)
    expect(shouldOfferRemoteFileDownload(true, true)).toBe(false)
    expect(shouldOfferRemoteFileDownload(false, false)).toBe(false)
    expect(shouldOfferRemoteFileDownload(true, false)).toBe(false)
  })
})

describe('shouldOfferLocalReveal', () => {
  // The OS file manager can only show what is on this computer (#115167): the
  // focused row's backend decides; the primary's mode only when it is untagged.
  it.each([
    ['', false, true],
    ['', true, false],
    ['local', true, true],
    ['mini', false, false],
    [undefined, true, false]
  ])('connection %s with primaryRemote=%s -> %s', (connectionId, primaryRemote, expected) => {
    expect(shouldOfferLocalReveal(connectionId, primaryRemote)).toBe(expected)
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
