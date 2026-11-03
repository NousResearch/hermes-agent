import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'

import { downloadRemoteFile, shouldOfferLocalReveal, shouldOfferRemoteFileDownload } from './file-actions'

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
  const saveGatewayFile = vi.fn()

  beforeEach(() => {
    clearNotifications()
    saveGatewayFile.mockReset()
    vi.stubGlobal('hermesDesktop', { saveGatewayFile })
  })

  afterEach(() => {
    clearNotifications()
    vi.unstubAllGlobals()
  })

  it('saves a remote gateway file through the native download bridge', async () => {
    saveGatewayFile.mockResolvedValue({ path: '/Users/me/Downloads/notes.md', saved: true })

    await downloadRemoteFile('/home/linux/project/notes.md')

    expect(saveGatewayFile).toHaveBeenCalledWith(expect.objectContaining({ path: '/home/linux/project/notes.md' }))
    expect($notifications.get()[0]?.message).toBe('Saved')
  })

  it('stays quiet when the save dialog is canceled', async () => {
    saveGatewayFile.mockResolvedValue({ canceled: true, saved: false })

    await downloadRemoteFile('/home/linux/project/notes.md')

    expect($notifications.get()).toEqual([])
  })

  it('toasts when the gateway download fails', async () => {
    vi.stubGlobal('hermesDesktop', {})

    await downloadRemoteFile('/home/linux/project/notes.md')

    expect($notifications.get()[0]?.kind).toBe('error')
    expect($notifications.get()[0]?.title).toBe('Download failed')
  })
})
