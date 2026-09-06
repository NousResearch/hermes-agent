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

  it('percent-encodes # ? & in the filename (not treated as URL fragment/query)', async () => {
    const openExternal = vi.fn(async () => undefined)
    const revealPath = vi.fn(async () => true)
    vi.stubGlobal('hermesDesktop', { openExternal, revealPath })

    await openFileWithDefaultApp('/Users/echo/report#1.md')

    // `#` must be %23, not left as a fragment delimiter.
    expect(openExternal).toHaveBeenCalledWith('file:///Users/echo/report%231.md')
    expect(revealPath).not.toHaveBeenCalled()
  })

  it('normalizes a Windows path into a valid file:// URL', async () => {
    // pathToFileURL uses platform-specific path semantics: on macOS a
    // `C:\...` string is a relative path, so this only holds on Windows.
    if (process.platform !== 'win32') {
      return
    }
    const openExternal = vi.fn(async () => undefined)
    const revealPath = vi.fn(async () => true)
    vi.stubGlobal('hermesDesktop', { openExternal, revealPath })

    await openFileWithDefaultApp('C:\\Users\\echo\\notes\\a b.txt')

    // Backslashes → forward slashes, drive letter prefixed, space encoded.
    expect(openExternal).toHaveBeenCalledWith('file:///C:/Users/echo/notes/a%20b.txt')
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
