import { afterEach, describe, expect, it, vi } from 'vitest'

const preview = vi.hoisted(() => ({
  target: {
    kind: 'file' as const,
    label: 'today.md',
    path: '/Users/test/repo/today.md',
    source: '/Users/test/repo/today.md',
    url: 'file:///Users/test/repo/today.md'
  }
}))

vi.mock('@/store/preview', () => ({ $filePreviewTarget: { get: () => preview.target } }))

import { host } from './index'

describe('host.openCodeClient', () => {
  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('delegates only to the curated desktop bridge', async () => {
    const openCodeClient = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openCodeClient }
    })
    const input = { client: 'codex' as const, cwd: '/Users/test/repo', prompt: 'Review notes/today.md' }

    await host.openCodeClient(input)

    expect(openCodeClient).toHaveBeenCalledWith(input)
  })

  it('fails safely outside the desktop shell', async () => {
    await expect(
      host.openCodeClient({ client: 'claude-code', cwd: '/Users/test/repo', prompt: 'Review notes/today.md' })
    ).rejects.toThrow('Code-client launch is unavailable')
  })

  it('returns a frozen copy of the active preview target', () => {
    const target = host.previewTarget()

    expect(target).not.toBeNull()
    expect(target).not.toBe(preview.target)
    expect(Object.isFrozen(target)).toBe(true)
  })
})
