import { afterEach, describe, expect, it, vi } from 'vitest'

import { compactProjectPath, fetchResolvedProjectDir } from './project-dir'

describe('compactProjectPath', () => {
  it('returns short paths unchanged', () => {
    expect(compactProjectPath('/Users/me/proj')).toBe('/Users/me/proj')
  })

  it('middle-truncates long paths', () => {
    const long = '/Users/someone/very/deep/nested/project/folder/with/many/segments'
    const out = compactProjectPath(long, 40)

    expect(out.length).toBeLessThanOrEqual(40)
    expect(out).toContain('…')
  })
})

describe('fetchResolvedProjectDir', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('returns trimmed dir from settings IPC', async () => {
    vi.stubGlobal('window', {
      hermesDesktop: {
        settings: {
          getResolvedProjectDir: vi.fn().mockResolvedValue({ dir: '  /tmp/proj  ' })
        }
      }
    })

    await expect(fetchResolvedProjectDir()).resolves.toBe('/tmp/proj')
  })

  it('returns empty string when IPC is unavailable', async () => {
    vi.stubGlobal('window', {})

    await expect(fetchResolvedProjectDir()).resolves.toBe('')
  })
})