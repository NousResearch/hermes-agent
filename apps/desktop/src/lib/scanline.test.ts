import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('scanline overlay lifecycle', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('lights on a capture start and hides when the matching completion lands', async () => {
    const setState = vi.fn()
    vi.stubGlobal('window', { hermesDesktop: { scanline: { setState } } })
    const { scanlineStart, scanlineComplete } = await import('./scanline')

    scanlineStart('cap-1')
    scanlineComplete('cap-1')

    expect(setState.mock.calls.map(([state]) => state)).toEqual(['active', 'hidden'])
  })

  it('ignores completions that were never registered as a capture', async () => {
    const setState = vi.fn()
    vi.stubGlobal('window', { hermesDesktop: { scanline: { setState } } })
    const { scanlineStart, scanlineComplete } = await import('./scanline')

    // A non-capture tool completing must not retire the sweep.
    scanlineStart('cap-1')
    scanlineComplete('other-tool')
    expect(setState.mock.calls.map(([state]) => state)).toEqual(['active'])

    // The real completion then hides it.
    scanlineComplete('cap-1')
    expect(setState.mock.calls.map(([state]) => state)).toEqual(['active', 'hidden'])
  })

  it('stays lit while multiple captures overlap, hiding only when all retire', async () => {
    const setState = vi.fn()
    vi.stubGlobal('window', { hermesDesktop: { scanline: { setState } } })
    const { scanlineStart, scanlineComplete } = await import('./scanline')

    scanlineStart('cap-1')
    scanlineStart('cap-2')
    scanlineComplete('cap-1') // one retires; the other is still in flight
    scanlineComplete('cap-2')

    // active, active(deduped), hidden
    expect(setState.mock.calls.map(([state]) => state)).toEqual(['active', 'hidden'])
  })

  it('deduplicates repeated states and tolerates a missing bridge', async () => {
    const setState = vi.fn()
    vi.stubGlobal('window', { hermesDesktop: { scanline: { setState } } })
    const { scanlineStart, scanlineComplete } = await import('./scanline')

    scanlineStart('cap-1')
    scanlineStart('cap-1') // same id again — no second push
    scanlineComplete('cap-1')
    scanlineComplete('cap-1') // already retired — no push

    expect(setState.mock.calls.map(([state]) => state)).toEqual(['active', 'hidden'])
  })

  it('no-ops when the bridge is absent (window without hermesDesktop)', async () => {
    vi.stubGlobal('window', {})
    const { scanlineStart, scanlineComplete } = await import('./scanline')

    expect(() => {
      scanlineStart('cap-1')
      scanlineComplete('cap-1')
    }).not.toThrow()
  })
})
