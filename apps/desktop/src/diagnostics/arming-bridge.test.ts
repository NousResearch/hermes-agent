import { afterEach, describe, expect, it, vi } from 'vitest'

// The bridge memoises its teardown at module scope, so each test imports a
// fresh copy after installing its own fake preload. `capture` comes from the
// SAME fresh registry — the bridge closes over that instance's module-level
// `armed`, so a stale import would arm a different module.
async function loadBridge() {
  vi.resetModules()

  const [bridge, capture] = await Promise.all([import('./arming-bridge'), import('./capture')])

  return { ...bridge, ...capture }
}

function installBridge(overrides: Record<string, unknown> = {}) {
  const state: {
    arm?: (payload: unknown) => void
    disarm?: () => void
    provider?: (captureId: string) => unknown
    unsubscribed: string[]
  } = { unsubscribed: [] }

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      diagnostics: {
        onArm: (callback: (payload: unknown) => void) => {
          state.arm = callback

          return () => state.unsubscribed.push('arm')
        },
        onDisarm: (callback: () => void) => {
          state.disarm = callback

          return () => state.unsubscribed.push('disarm')
        },
        setSnapshotProvider: (provider: (captureId: string) => unknown) => {
          state.provider = provider

          return () => state.unsubscribed.push('snapshot')
        },
        ...overrides
      }
    }
  })

  return state
}

afterEach(() => {
  Reflect.deleteProperty(window as never, 'hermesDesktop')
  vi.resetModules()
})

describe('initDiagnosticsSnapshots', () => {
  it('answers main with the running capture snapshot', async () => {
    const state = installBridge()
    const { armDiagnostics, initDiagnosticsSnapshots } = await loadBridge()

    initDiagnosticsSnapshots()
    expect(state.provider).toBeTypeOf('function')

    armDiagnostics('cap-abc', 1_700_000_000_000)

    const snapshot = state.provider!('cap-abc') as { captureId: string; events: unknown[] }

    expect(snapshot.captureId).toBe('cap-abc')
    expect(Array.isArray(snapshot.events)).toBe(true)
  })

  it('answers null while disarmed', async () => {
    const state = installBridge()
    const { initDiagnosticsSnapshots } = await loadBridge()

    initDiagnosticsSnapshots()

    expect(state.provider!('cap-abc')).toBeNull()
  })

  it('refuses to answer for a different capture id', async () => {
    const state = installBridge()
    const { armDiagnostics, initDiagnosticsSnapshots } = await loadBridge()

    initDiagnosticsSnapshots()
    armDiagnostics('cap-abc', 1_700_000_000_000)

    // A stale snapshot filed under the wrong capture would break the KTD3
    // alignment silently; contributing nothing is the safe answer.
    expect(state.provider!('cap-other')).toBeNull()
    expect((state.provider!('cap-abc') as { captureId: string }).captureId).toBe('cap-abc')
  })

  it('is idempotent and unsubscribes on teardown', async () => {
    const state = installBridge()
    const { initDiagnosticsSnapshots } = await loadBridge()

    const first = initDiagnosticsSnapshots()

    expect(initDiagnosticsSnapshots()).toBe(first)

    first()
    expect(state.unsubscribed).toContain('snapshot')
  })

  it('is a no-op against a preload without setSnapshotProvider', async () => {
    installBridge({ setSnapshotProvider: undefined })
    const { initDiagnosticsSnapshots } = await loadBridge()

    expect(() => initDiagnosticsSnapshots()()).not.toThrow()
  })

  it('is a no-op with no diagnostics bridge at all', async () => {
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {} })
    const { initDiagnosticsSnapshots } = await loadBridge()

    expect(() => initDiagnosticsSnapshots()()).not.toThrow()
  })
})
