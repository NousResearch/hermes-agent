import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'

vi.mock('@/lib/media', () => ({
  downloadGatewayMediaFile: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  copyTextToClipboard: vi.fn(),
  createDesktopDirectory: vi.fn(),
  isDesktopFsRemoteMode: vi.fn(() => false),
  renameDesktopPath: vi.fn(),
  revealDesktopPath: vi.fn(),
  trashDesktopPath: vi.fn()
}))

vi.mock('@/store/workspace-events', () => ({
  notifyWorkspaceChanged: vi.fn(),
  notifyWorkspaceDirectoryChanged: vi.fn()
}))

const media = await import('@/lib/media')
const downloadGatewayMediaFile = vi.mocked(media.downloadGatewayMediaFile)
const desktopFs = await import('@/lib/desktop-fs')
const createDesktopDirectory = vi.mocked(desktopFs.createDesktopDirectory)
const workspaceEvents = await import('@/store/workspace-events')

const notifyWorkspaceDirectoryChanged = vi.mocked(
  (workspaceEvents as unknown as { notifyWorkspaceDirectoryChanged: (path: string) => void })
    .notifyWorkspaceDirectoryChanged
)

const { downloadRemoteFile, executeNewFolder, shouldOfferNewFolder, shouldOfferRemoteFileDownload } =
  await import('./file-actions')

describe('shouldOfferNewFolder', () => {
  it('is only for local directories', () => {
    expect(shouldOfferNewFolder(true, false)).toBe(true)
    expect(shouldOfferNewFolder(false, false)).toBe(false)
    expect(shouldOfferNewFolder(true, true)).toBe(false)
  })
})

describe('executeNewFolder', () => {
  beforeEach(() => {
    createDesktopDirectory.mockReset()
    notifyWorkspaceDirectoryChanged.mockReset()
  })

  it('creates a folder through the native bridge', async () => {
    createDesktopDirectory.mockResolvedValue('/repo/docs')

    await expect(executeNewFolder('/repo', 'docs')).resolves.toBe('/repo/docs')
    expect(createDesktopDirectory).toHaveBeenCalledWith('/repo', 'docs')
  })

  it('refreshes the displayed WSL parent instead of the bridged Windows result path', async () => {
    createDesktopDirectory.mockResolvedValue('\\\\wsl.localhost\\Ubuntu\\home\\alex\\repo\\docs')

    await executeNewFolder('/home/alex/repo', 'docs')

    expect(notifyWorkspaceDirectoryChanged).toHaveBeenCalledWith('/home/alex/repo')
  })
})

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
