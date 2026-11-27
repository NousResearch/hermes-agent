import { afterEach, describe, expect, it, type Mock, vi } from 'vitest'

import { hermesApi, setApiRequestConnection } from './client'

describe('hermesApi in-flight GET coalescing', () => {
  const originalDesktop = window.hermesDesktop

  afterEach(() => {
    setApiRequestConnection(null)
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop, writable: true })
  })

  function installMockApi(api: Mock) {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api },
      writable: true
    })
  }

  it('coalesces concurrent identical GET requests and issues a fresh call after settlement', async () => {
    const first = Promise.withResolvers<{ ok: boolean }>()
    const second = Promise.withResolvers<{ ok: boolean }>()
    const api = vi.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    installMockApi(api)

    const a = hermesApi<{ ok: boolean }>({ path: '/api/profiles' })
    const b = hermesApi<{ ok: boolean }>({ path: '/api/profiles' })

    expect(api).toHaveBeenCalledTimes(1)
    first.resolve({ ok: true })
    expect(await Promise.all([a, b])).toEqual([{ ok: true }, { ok: true }])

    const c = hermesApi<{ ok: boolean }>({ path: '/api/profiles' })
    expect(api).toHaveBeenCalledTimes(2)
    second.resolve({ ok: false })
    expect(await c).toEqual({ ok: false })
  })

  it('never coalesces mutations or distinct profile/connection scopes', async () => {
    const api = vi.fn(() => Promise.withResolvers<unknown>().promise)
    installMockApi(api)

    void hermesApi({ path: '/api/config', profile: 'default' })
    void hermesApi({ path: '/api/config', profile: 'ops' })
    void hermesApi({ connectionId: 'remote-1', path: '/api/config', profile: 'default' })
    void hermesApi({ body: { a: 1 }, method: 'POST', path: '/api/config' })
    void hermesApi({ body: { a: 1 }, method: 'POST', path: '/api/config' })

    expect(api).toHaveBeenCalledTimes(5)
  })
})
