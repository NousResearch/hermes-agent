import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { PreviewSigninsResult } from '@/global'

// Fresh module per test so the module-level `lastSynced` map does not leak across cases.
async function load() {
  vi.resetModules()

  return import('./preview-signins')
}

function setBridge(bridge: unknown) {
  ;(globalThis as { window: { hermesDesktop?: unknown } }).window = bridge
    ? { hermesDesktop: { syncPreviewSignins: bridge } }
    : {}
}

const ok = (status: PreviewSigninsResult['status']): PreviewSigninsResult => ({
  status,
  imported: 0,
  skipped: 0,
  failed: 0,
  cleared: 0
})

beforeEach(() => {
  setBridge(null)
})

afterEach(() => {
  vi.restoreAllMocks()
  delete (globalThis as { window?: unknown }).window
})

describe('syncPreviewSignins lastSynced (P2-8 renderer)', () => {
  it('does NOT mark the scope synced when the bridge replies with status error, so a retry still runs', async () => {
    const bridge = vi
      .fn<(r: unknown) => Promise<PreviewSigninsResult>>()
      .mockResolvedValueOnce(ok('error'))
      .mockResolvedValueOnce(ok('imported'))

    setBridge(bridge)

    const mod = await load()
    const scope = { connectionId: null, profile: 'work' }

    const first = await mod.syncPreviewSignins(true, scope)
    expect(first?.status).toBe('error')

    // The failed sync must NOT have left a permanent "synced" mark: the next if-changed call retries.
    const retry = mod.syncPreviewSigninsIfChanged(true, scope)
    expect(retry).not.toBeNull()
    await retry
    expect(bridge).toHaveBeenCalledTimes(2)
  })

  it('marks the scope synced after a successful reply, so if-changed is a no-op', async () => {
    const bridge = vi.fn<(r: unknown) => Promise<PreviewSigninsResult>>().mockResolvedValue(ok('imported'))

    setBridge(bridge)

    const mod = await load()
    const scope = { connectionId: null, profile: 'work' }

    await mod.syncPreviewSignins(true, scope)
    const again = mod.syncPreviewSigninsIfChanged(true, scope)

    expect(again).toBeNull()
    expect(bridge).toHaveBeenCalledTimes(1)
  })

  it('a successful clear (cleared/disabled) still marks synced', async () => {
    const bridge = vi.fn<(r: unknown) => Promise<PreviewSigninsResult>>().mockResolvedValue(ok('cleared'))

    setBridge(bridge)

    const mod = await load()
    const scope = { connectionId: null, profile: 'work' }

    await mod.syncPreviewSignins(false, scope)

    expect(mod.syncPreviewSigninsIfChanged(false, scope)).toBeNull()
  })

  it('off-Desktop (no bridge) returns null and marks nothing', async () => {
    setBridge(null)

    const mod = await load()
    const scope = { connectionId: null, profile: 'work' }

    expect(await mod.syncPreviewSignins(true, scope)).toBeNull()
    // Nothing marked, so a later real bridge sync is not suppressed.
  })
})
