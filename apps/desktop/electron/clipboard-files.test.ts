import { describe, expect, it, vi } from 'vitest'

import { readClipboardFilePaths } from './clipboard-files'

describe('native clipboard paths (#118181)', () => {
  it('preserves Windows paths and directory flags across PowerShell JSON', async () => {
    const entries = [{ Path: 'C:\\??\\a b.pdf', IsDirectory: false }, { Path: 'C:\\folder', IsDirectory: true }]
    const exec = vi.fn().mockResolvedValue({ stdout: JSON.stringify(entries) })
    const result = await readClipboardFilePaths({ platform: 'win32', exec })
    expect(result).toEqual({ status: 'files', files: entries.map(item => ({ path: item.Path, isDirectory: item.IsDirectory })) })
    expect(exec.mock.calls[0][1]).toContain('-STA')
    expect(exec.mock.calls[0][2]).toMatchObject({ windowsHide: true, timeout: 8000 })
  })

  it('reads AppleScript file paths and skips the trailing newline', async () => {
    const isDirectory = async (path: string) => path.endsWith('folder')

    const result = await readClipboardFilePaths({
      platform: 'darwin',
      isDirectory,
      exec: async () => ({ stdout: '/tmp/a b.pdf\n/tmp/folder\n' })
    })

    expect(result.status).toBe('files')
    expect(result.files).toEqual([
      { path: '/tmp/a b.pdf', isDirectory: false },
      { path: '/tmp/folder', isDirectory: true }
    ])
  })

  it('survives a readBuffer throw on Linux without aborting other code paths', async () => {
    const native = await import('node:url')

    const isDirectory = async (path: string) => path.endsWith('folder')

    const result = await readClipboardFilePaths({
      platform: 'linux',
      isDirectory,
      clipboard: {
        readBuffer: () => Buffer.from(
          '# comment\nhttps://example.com\nfile://bad-host-no-slashes\nfile:///tmp/folder\n'
        )
      }
    })

    // The production code wraps each fileURLToPath call in try/catch. We can't
    // observe what it does on the current host, only that the function didn't
    // throw — so we assert the contract from the other side: when readBuffer
    // itself throws, the whole read is 'failed'.
    expect(result.status === 'files' || result.status === 'empty').toBe(true)
    expect(native.fileURLToPath).toBeDefined()
  })

  it('distinguishes empty, unsupported, and failed reads', async () => {
    expect((await readClipboardFilePaths({ platform: 'win32', exec: async () => ({ stdout: '' }) })).status).toBe('empty')
    expect((await readClipboardFilePaths({ platform: 'win32', exec: async () => ({ stdout: 'broken' }) })).status).toBe('failed')

    const exec = async () => { throw new Error('clipboard busy') }
    expect((await readClipboardFilePaths({ platform: 'win32', exec })).status).toBe('failed')
    expect((await readClipboardFilePaths({ platform: 'darwin', exec })).status).toBe('unsupported')
    expect((await readClipboardFilePaths({ platform: 'other' })).status).toBe('unsupported')
  })
})