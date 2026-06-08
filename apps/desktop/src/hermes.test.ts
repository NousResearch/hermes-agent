import { afterEach, describe, expect, it, vi } from 'vitest'

import { bulkArchiveSessions } from './hermes'

describe('bulkArchiveSessions', () => {
  const originalHermesDesktop = window.hermesDesktop

  afterEach(() => {
    vi.restoreAllMocks()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalHermesDesktop,
      writable: true
    })
  })

  it('posts deduped preserve ids to the manual bulk archive endpoint', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true, archived: 12 })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api },
      writable: true
    })

    await expect(bulkArchiveSessions(['pin', '', 'current', 'pin'])).resolves.toEqual({ ok: true, archived: 12 })

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/bulk-archive',
      method: 'POST',
      body: { preserve_ids: ['pin', 'current'] }
    })
  })
})
