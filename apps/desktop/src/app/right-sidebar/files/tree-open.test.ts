import { beforeEach, describe, expect, it, vi } from 'vitest'

import { openFileWithDefaultApp } from './tree'

describe('openFileWithDefaultApp', () => {
  beforeEach(() => {
    vi.stubGlobal('hermesDesktop', undefined)
  })

  it('opens the file via a file:// URL (default app)', async () => {
    const openExternal = vi.fn(async () => undefined)
    const revealPath = vi.fn(async () => true)
    vi.stubGlobal('hermesDesktop', { openExternal, revealPath })

    await openFileWithDefaultApp('/Users/echo/notes/my notes.md')

    expect(openExternal).toHaveBeenCalledWith('file:///Users/echo/notes/my%20notes.md')
    expect(revealPath).not.toHaveBeenCalled()
  })

  it('falls back to revealing in the file manager when opening fails', async () => {
    const openExternal = vi.fn(async () => {
      throw new Error('no default app')
    })
    const revealPath = vi.fn(async () => true)
    vi.stubGlobal('hermesDesktop', { openExternal, revealPath })

    await openFileWithDefaultApp('/tmp/a.txt')

    expect(openExternal).toHaveBeenCalledWith('file:///tmp/a.txt')
    expect(revealPath).toHaveBeenCalledWith('/tmp/a.txt')
  })

  it('tolerates a missing bridge (no-op)', async () => {
    await expect(openFileWithDefaultApp('/tmp/a.txt')).resolves.toBeUndefined()
  })
})
