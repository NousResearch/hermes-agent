import { beforeEach, describe, expect, it, vi } from 'vitest'

const { isDesktopFsRemoteMode, readDesktopDir } = vi.hoisted(() => ({
  isDesktopFsRemoteMode: vi.fn(() => false),
  readDesktopDir: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  isDesktopFsRemoteMode,
  readDesktopDir,
  readDesktopFileDataUrl: vi.fn(),
  readDesktopFileText: vi.fn()
}))

import { existingLocalDirectoryPath } from './local-directory'

describe('existingLocalDirectoryPath', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    isDesktopFsRemoteMode.mockReturnValue(false)
  })

  it('returns an existing local directory resolved against the session cwd', async () => {
    readDesktopDir.mockResolvedValue({ entries: [] })

    await expect(existingLocalDirectoryPath('historical', '/work/reports')).resolves.toBe('/work/reports/historical')
    expect(readDesktopDir).toHaveBeenCalledWith('/work/reports/historical')
  })

  it('normalizes a Windows file URL before probing the directory', async () => {
    readDesktopDir.mockResolvedValue({ entries: [] })

    await expect(existingLocalDirectoryPath('file:///C:/Users/E/reports/historical')).resolves.toBe(
      'C:/Users/E/reports/historical'
    )
    expect(readDesktopDir).toHaveBeenCalledWith('C:/Users/E/reports/historical')
  })

  it('returns null when the target is not a directory', async () => {
    readDesktopDir.mockRejectedValue(new Error('not a directory'))

    await expect(existingLocalDirectoryPath('/work/report.md')).resolves.toBeNull()
  })

  it('does not probe remote-backend paths on the Electron host', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)

    await expect(existingLocalDirectoryPath('/srv/reports')).resolves.toBeNull()
    expect(readDesktopDir).not.toHaveBeenCalled()
  })
})
