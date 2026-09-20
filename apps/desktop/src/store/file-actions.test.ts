import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'

vi.mock('@/lib/media', () => ({
  downloadGatewayMediaFile: vi.fn()
}))

const media = await import('@/lib/media')
const downloadGatewayMediaFile = vi.mocked(media.downloadGatewayMediaFile)

const { downloadRemoteFile, openFileInDefaultApp, shouldOfferRemoteFileDownload } = await import('./file-actions')

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

describe('openFileInDefaultApp', () => {
  const openExternal = vi.fn()

  beforeEach(() => {
    clearNotifications()
    openExternal.mockReset()
    openExternal.mockResolvedValue(undefined)
    vi.stubGlobal('hermesDesktop', { openExternal })
  })

  afterEach(() => {
    clearNotifications()
    vi.unstubAllGlobals()
  })

  it('hands a POSIX path to the bridge as a file:// URL', async () => {
    await openFileInDefaultApp('/home/me/report.xlsx')

    expect(openExternal).toHaveBeenCalledWith('file:///home/me/report.xlsx')
  })

  it('encodes spaces and Windows path separators per segment', async () => {
    await openFileInDefaultApp('C:\\Users\\me\\My Report.xlsx')

    // pathToFileUrl encodes each segment (same helper the artifacts panel
    // uses); the main process decodes it back via the file:// branch of
    // openExternalUrl.
    expect(openExternal).toHaveBeenCalledWith('file:///C%3A/Users/me/My%20Report.xlsx')
  })

  it('stays quiet when the desktop bridge is unavailable', async () => {
    vi.unstubAllGlobals()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop

    await openFileInDefaultApp('/home/me/report.xlsx')

    expect($notifications.get()).toEqual([])
  })

  it('toasts when the bridge call fails', async () => {
    openExternal.mockRejectedValue(new Error('openExternal failed'))

    await openFileInDefaultApp('/home/me/report.xlsx')

    expect($notifications.get()[0]?.kind).toBe('error')
  })
})
