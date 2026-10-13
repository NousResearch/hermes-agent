import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { desktopFsCacheKey } from '@/lib/desktop-fs'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $connection } from '@/store/session'

const {
  downloadRemoteFile,
  shouldOfferLocalReveal,
  shouldOfferRemoteFileDownload,
  $fileActionDialog,
  $renamingPath,
  $creatingEntry,
  requestFileDelete,
  beginInlineRename,
  requestNewEntry
} = await import('./file-actions')

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

describe('pending file actions cancel on connection-scope change', () => {
  function connection(
    overrides: Partial<Record<'connectionId' | 'mode' | 'profile' | 'baseUrl' | 'remoteIdentity', string>>
  ): HermesConnection {
    return {
      mode: (overrides.mode as 'remote' | 'local' | undefined) ?? 'remote',
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

  it('cancels pending create, rename, and delete when the registered connection changes mode', async () => {
    const remote = connection({ connectionId: 'g1', profile: 'alice', mode: 'remote' })
    const local = connection({ connectionId: 'g1', profile: 'alice', mode: 'local' })
    expect(desktopFsCacheKey(remote)).not.toBe(desktopFsCacheKey(local))

    $connection.set(remote)
    await requestNewEntry({ directory: false, parentDir: '/home/alice' })
    beginInlineRename('/home/alice/old.txt')
    requestFileDelete({ path: '/home/alice/gone.txt', name: 'gone.txt', isDirectory: false })

    $connection.set(local)

    expect($creatingEntry.get()).toBeNull()
    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
  })

  it('cancels pending actions when local/remote routing changes even if gateway fields match', () => {
    const remote = connection({ profile: 'alice', remoteIdentity: 'legacy-host', mode: 'remote' })
    const local = connection({ profile: 'alice', remoteIdentity: 'legacy-host', mode: 'local' })
    expect(desktopFsCacheKey(remote)).not.toBe(desktopFsCacheKey(local))
    $connection.set(remote)
    $creatingEntry.set({ directory: false, parentDir: '/home/alice/p' })

    // The active backend changed; a duplicated partial key must not preserve an editor.
    $connection.set(local)

    expect($creatingEntry.get()).toBeNull()
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
    requestFileDelete({ path: '/home/alice/a.txt', name: 'a.txt', isDirectory: false })

    $connection.set(connection({ baseUrl: 'https://other-gateway' }))

    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
  })
})
