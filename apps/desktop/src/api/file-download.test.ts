import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import { captureGatewayFileDownload } from './file-download'

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

describe('captured gateway file download', () => {
  it('reports an unavailable desktop bridge', async () => {
    vi.stubGlobal('hermesDesktop', {})
    await expect(captureGatewayFileDownload()('/persisted/file.md', 'file.md')).rejects.toThrow(
      'Desktop file download bridge is unavailable'
    )
  })

  it('rejects an absent stored path before invoking the bridge', async () => {
    const saveGatewayFile = vi.fn()
    vi.stubGlobal('hermesDesktop', { saveGatewayFile })
    await expect(captureGatewayFileDownload()('  ', 'file.md')).rejects.toThrow('Missing gateway file path')
    expect(saveGatewayFile).not.toHaveBeenCalled()
  })

  it('rejects an unsuccessful non-canceled save', async () => {
    vi.stubGlobal('hermesDesktop', { saveGatewayFile: vi.fn().mockResolvedValue({ saved: false }) })
    await expect(captureGatewayFileDownload()('/persisted/file.md', 'file.md')).rejects.toThrow('File download failed')
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
    await expect(download('/persisted/file.md', 'file.md')).resolves.toEqual({ canceled: true, saved: false })
    expect(saveGatewayFile).toHaveBeenCalledWith({
      connectionId: 'local',
      profile: 'personal',
      path: '/persisted/file.md',
      suggestedName: 'file.md'
    })
  })
})
