import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { localPreviewTarget, normalizeOrLocalPreviewTarget } from './local-preview'

describe('normalizeOrLocalPreviewTarget', () => {
  afterEach(() => {
    $connection.set(null)
    vi.unstubAllGlobals()
  })

  it('preserves Windows drive-letter and UNC artifact paths and file URLs instead of joining them to the remote cwd', () => {
    const drivePath = String.raw`C:\Users\Rohit\brief.md`
    const uncPath = String.raw`\\server\share\brief.md`
    const cwd = String.raw`C:\work`

    expect(localPreviewTarget(drivePath, cwd)).toMatchObject({ path: drivePath, url: 'file:///C:/Users/Rohit/brief.md' })
    expect(localPreviewTarget(uncPath, cwd)).toMatchObject({ path: uncPath, url: 'file://server/share/brief.md' })
    expect(localPreviewTarget('file:///C:/Users/Rohit/brief.md', cwd)).toMatchObject({
      path: 'C:/Users/Rohit/brief.md',
      url: 'file:///C:/Users/Rohit/brief.md'
    })
    expect(localPreviewTarget('file://server/share/brief.md', cwd)).toMatchObject({
      path: String.raw`\\server\share\brief.md`,
      url: 'file://server/share/brief.md'
    })
  })

  it('keeps gateway paths remote instead of resolving them on the desktop machine', async () => {
    const normalizePreviewTarget = vi.fn(async () => ({
      kind: 'file' as const,
      label: 'wrong-local-file.md',
      path: '/Users/local/wrong-local-file.md',
      previewKind: 'text' as const,
      source: './report.md',
      url: 'file:///Users/local/wrong-local-file.md'
    }))

    const api = vi.fn(async () => ({
      byteSize: 9,
      language: 'markdown',
      path: '/srv/project/report.md',
      text: '# Remote'
    }))

    vi.stubGlobal('window', { hermesDesktop: { api, normalizePreviewTarget } })
    $connection.set({ mode: 'remote' } as never)

    await expect(normalizeOrLocalPreviewTarget('./report.md', '/srv/project')).resolves.toMatchObject({
      path: '/srv/project/report.md',
      source: './report.md'
    })
    expect(normalizePreviewTarget).not.toHaveBeenCalled()
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/read-text?path=%2Fsrv%2Fproject%2Freport.md' })
  })

  it('promotes the backend-resolved absolute path when renderer context is missing', async () => {
    const api = vi.fn(async () => ({
      byteSize: 9,
      language: 'markdown',
      path: '/srv/project/report.md',
      text: '# Remote'
    }))

    vi.stubGlobal('window', { hermesDesktop: { api } })
    $connection.set({ mode: 'remote' } as never)

    await expect(normalizeOrLocalPreviewTarget('./report.md')).resolves.toMatchObject({
      path: '/srv/project/report.md',
      url: 'file:///srv/project/report.md'
    })
  })
})
