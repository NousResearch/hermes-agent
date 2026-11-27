import { afterEach, describe, expect, it, vi } from 'vitest'

import { createProfile, PROFILE_CREATE_TIMEOUT_MS } from './profiles'

describe('createProfile request timeout', () => {
  const originalDesktop = window.hermesDesktop

  afterEach(() => {
    window.hermesDesktop = originalDesktop
  })

  it('passes a timeout exceeding default-plugins preflight (120s) + skill seeding (60s)', async () => {
    const api = vi.fn(async () => ({ name: 'work', ok: true, path: '/tmp/work' }))
    window.hermesDesktop = { api } as unknown as Window['hermesDesktop']

    await createProfile({ name: 'work' })

    expect(PROFILE_CREATE_TIMEOUT_MS).toBeGreaterThanOrEqual(180_000)
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/profiles',
        method: 'POST',
        body: { name: 'work' },
        timeoutMs: PROFILE_CREATE_TIMEOUT_MS
      })
    )
  })
})
