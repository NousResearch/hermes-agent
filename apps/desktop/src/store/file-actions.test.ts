import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $notifications, clearNotifications } from '@/store/notifications'

vi.mock('@/lib/media', () => ({
  downloadGatewayMediaFile: vi.fn()
}))

const media = await import('@/lib/media')
const downloadGatewayMediaFile = vi.mocked(media.downloadGatewayMediaFile)

const { downloadRemoteFile, shouldOfferRemoteFileDownload, $fileActionDialog, $renamingPath, $creatingEntry } =
  await import('./file-actions')

const { $connection } = await import('@/store/session')

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

describe('pending file actions cancel on connection-scope change', () => {
  function connection(
    overrides: Partial<Record<'connectionId' | 'profile' | 'baseUrl' | 'remoteIdentity', string>>
  ): HermesConnection {
    return {
      mode: 'remote' as const,
      baseUrl: overrides.baseUrl ?? 'https://gw.example',
      profile: overrides.profile,
      remoteIdentity: overrides.remoteIdentity,
      connectionId: overrides.connectionId,
      token: 't',
      wsUrl: 'wss://gw/ws',
      logs: [] as string[],
      isFullscreen: false,
      nativeOverlayWidth: 0,
      windowButtonPosition: null
    }
  }

  beforeEach(() => {
    $connection.set(null)
    $fileActionDialog.set(null)
    $renamingPath.set(null)
    $creatingEntry.set(null)
  })

  it('keeps a pending action alive when the connection is unchanged (no false cancel)', () => {
    $connection.set(connection({ connectionId: 'g1', profile: 'alice' }))
    $creatingEntry.set({ directory: false, parentDir: '/home/alice/p' })

    // Re-emit the SAME connection descriptor (e.g. a refresh) — same scope key.
    $connection.set(connection({ connectionId: 'g1', profile: 'alice' }))

    expect($creatingEntry.get()).toEqual({ directory: false, parentDir: '/home/alice/p' })
  })

  it('cancels pending create when the active PROFILE changes on the same gateway', () => {
    $connection.set(connection({ connectionId: 'g1', profile: 'alice' }))
    $creatingEntry.set({ directory: false, parentDir: '/home/alice/p' })

    // Same gateway, different profile: the filesystem root and auth differ, so
    // an old absolute path must not commit through the new profile.
    $connection.set(connection({ connectionId: 'g1', profile: 'bob' }))

    expect($creatingEntry.get()).toBeNull()
  })

  it('cancels pending rename and delete dialogs when the scope changes', () => {
    $connection.set(connection({ connectionId: 'g1', profile: 'alice' }))
    $renamingPath.set('/home/alice/a.txt')
    $fileActionDialog.set({ kind: 'delete', path: '/home/alice/a.txt', name: 'a.txt', isDirectory: false })

    $connection.set(connection({ baseUrl: 'https://other-gateway' }))

    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
  })
})
