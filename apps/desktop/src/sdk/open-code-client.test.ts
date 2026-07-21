import { afterEach, describe, expect, it, vi } from 'vitest'

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
})
