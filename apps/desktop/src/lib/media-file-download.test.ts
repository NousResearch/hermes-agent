import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $connection } from '@/store/session'

import { captureGatewayFileDownload } from './media'

beforeEach(() => {
  clearNotifications()
  // The window's focused connection must never leak into a captured owner.
  $connection.set({ connectionId: 'focused-host', mode: 'remote', profile: 'focused' } as never)
})

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  $connection.set(null)
  clearNotifications()
  vi.unstubAllGlobals()
})

const lastToast = () => $notifications.get()[0]

describe('captured gateway file download', () => {
  it('reports an unavailable desktop bridge', async () => {
    vi.stubGlobal('hermesDesktop', {})

    await expect(captureGatewayFileDownload()('/persisted/file.md', 'file.md')).resolves.toBeUndefined()

    expect(lastToast()).toMatchObject({ kind: 'error', title: 'Download failed' })
    expect(lastToast()?.message).toContain('Desktop file download bridge is unavailable')
  })

  it('rejects an absent stored path before invoking the bridge', async () => {
    const saveGatewayFile = vi.fn()
    vi.stubGlobal('hermesDesktop', { saveGatewayFile })

    await captureGatewayFileDownload()('  ', 'file.md')

    expect(saveGatewayFile).not.toHaveBeenCalled()
    expect(lastToast()?.message).toContain('Missing gateway file path')
  })

  it('confirms a completed save and stays quiet on cancel', async () => {
    const saveGatewayFile = vi.fn().mockResolvedValueOnce({ path: '/Users/me/Downloads/file.md', saved: true })
    vi.stubGlobal('hermesDesktop', { saveGatewayFile })
    const download = captureGatewayFileDownload()

    await download('/persisted/file.md', 'file.md')
    expect(lastToast()).toMatchObject({ kind: 'info', message: 'Saved' })

    clearNotifications()
    saveGatewayFile.mockResolvedValueOnce({ canceled: true, saved: false })
    await download('/persisted/file.md', 'file.md')
    expect($notifications.get()).toEqual([])
  })

  it('preserves legacy primary routing instead of inventing a default profile', async () => {
    const saveGatewayFile = vi.fn().mockResolvedValue({ saved: true })
    vi.stubGlobal('hermesDesktop', { saveGatewayFile })
    const download = captureGatewayFileDownload()
    setApiRequestConnection('remote')
    setApiRequestProfile('work')

    await download('/persisted/file.md', 'file.md')

    expect(saveGatewayFile).toHaveBeenCalledWith({ path: '/persisted/file.md', suggestedName: 'file.md' })
  })

  it('keeps explicit local ownership even after switching to a remote', async () => {
    const saveGatewayFile = vi.fn().mockResolvedValue({ canceled: true, saved: false })
    vi.stubGlobal('hermesDesktop', { saveGatewayFile })
    setApiRequestConnection('local')
    setApiRequestProfile('personal')
    const download = captureGatewayFileDownload()
    setApiRequestConnection('remote')
    setApiRequestProfile('work')

    await download('/persisted/file.md', 'file.md')

    expect(saveGatewayFile).toHaveBeenCalledWith({
      connectionId: 'local',
      profile: 'personal',
      path: '/persisted/file.md',
      suggestedName: 'file.md'
    })
  })
})
