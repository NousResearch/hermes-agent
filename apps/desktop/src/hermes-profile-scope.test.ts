import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getActionStatus,
  getStatus,
  setApiRequestProfile
} from './hermes'

// Contract: every backend-targeted read/action-status helper must carry the
// active gateway profile, so multi-profile remote users query the backend
// profile they are actually on.
describe('backend status helpers are profile-scoped', () => {
  const api = vi.fn(async (_req: { path: string; profile?: string }) => ({}) as never)

  beforeEach(() => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api }
    api.mockClear()
  })

  afterEach(() => {
    setApiRequestProfile(null)
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  const lastProfile = () => api.mock.calls.at(-1)?.[0].profile

  it('omits profile when none is active (single-profile users unaffected)', () => {
    void getStatus()
    expect(lastProfile()).toBeUndefined()
  })

  it('forwards the active profile to backend status helpers', () => {
    setApiRequestProfile('coder')

    void getStatus()
    void getActionStatus('gateway-restart')

    for (const call of api.mock.calls) {
      expect(call[0].profile).toBe('coder')
    }
  })
})
