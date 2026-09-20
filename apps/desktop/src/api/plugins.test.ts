import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { activeConnection, pluginRest } from './plugins'

// desktop.getConnection/getConnectionFor are IPC round-trips into the main
// process with no timeout of their own (#93454). A wedged main-process
// round-trip must reject instead of hanging pluginSocket's connect() forever.
describe('activeConnection connection timeout (#93454)', () => {
  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.useRealTimers()
  })

  it('rejects instead of hanging forever when getConnection() wedges', async () => {
    vi.useFakeTimers()
    setApiRequestProfile('coder')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(() => new Promise(() => undefined)) }
    })

    const pending = expect(activeConnection()).rejects.toThrow('Timed out connecting to profile "coder"')

    await vi.advanceTimersByTimeAsync(20_000)
    await pending
  })

  it('rejects instead of hanging forever when getConnectionFor() wedges', async () => {
    vi.useFakeTimers()
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('research')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        getConnection: vi.fn(() => new Promise(() => undefined)),
        getConnectionFor: vi.fn(() => new Promise(() => undefined))
      }
    })

    const pending = expect(activeConnection()).rejects.toThrow('Timed out connecting to profile "research"')

    await vi.advanceTimersByTimeAsync(20_000)
    await pending
  })
})

// A plugin call with `target: 'local-primary'` must reach the local machine's
// primary backend regardless of which profile/connection the renderer is
// ambiently scoped to (named profile, registry remote, …) — the concrete
// consumer is the external Model Usage Status plugin, which must show the
// same local usage no matter which Hermes profile/connection is active.
describe('pluginRest target: local-primary', () => {
  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('forces connectionId "local" with no profile pin, overriding ambient scope', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('coder')

    await pluginRest('model-usage', '/status', { target: 'local-primary' })

    expect(api).toHaveBeenCalledTimes(1)
    const [request] = api.mock.calls[0] as [Record<string, unknown>]
    expect(request.path).toBe('/api/plugins/model-usage/status')
    expect(request.connectionId).toBe('local')
    expect(request.profile).toBeUndefined()
  })

  // Backwards-compat guard: every existing `ctx.rest` caller omits `target`,
  // so it must keep following the renderer's ambient connection/profile
  // scope byte-for-byte — the override above must not leak into the default
  // path.
  it('preserves the ambient connectionId and profile when target is omitted', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('coder')

    await pluginRest('model-usage', '/status')

    expect(api).toHaveBeenCalledTimes(1)
    const [request] = api.mock.calls[0] as [Record<string, unknown>]
    expect(request.path).toBe('/api/plugins/model-usage/status')
    expect(request.connectionId).toBe('gw-tailscale')
    expect(request.profile).toBe('coder')
  })
})
