import { afterEach, describe, expect, it, vi } from 'vitest'

import { startWebAccess } from './webapp-access'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('startWebAccess', () => {
  it('waits as long as a Webapp build can take', async () => {
    // Start builds the bundle first, as `hermes webapp` does; that build can wait up to 30 minutes
    // for another one, far past the default request timeout.
    const api = vi.fn(async () => ({}))
    vi.stubGlobal('hermesDesktop', { api })

    await startWebAccess('lan')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ method: 'POST', path: '/api/webapp-access/start' }))
    expect((api.mock.calls[0] as unknown as [{ timeoutMs?: number }])[0].timeoutMs).toBeGreaterThan(30 * 60_000)
  })
})
