import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  activateCustomEndpoint,
  checkHermesUpdate,
  deleteCustomEndpoint,
  getActionStatus,
  getCustomEndpoints,
  getMemoryProviderConfig,
  getStatus,
  restartGateway,
  saveCustomEndpoint,
  saveMemoryProviderConfig,
  setApiRequestProfile,
  updateHermes,
  validateCustomEndpoint
} from './hermes'

// Contract: every backend-targeted action helper must carry the active gateway
// profile, so a multi-profile / global-remote user's restart, status poll, and
// update hit the backend they're actually on — not the primary/default. The
// System-panel "restart does nothing" bug was these helpers dropping it.
describe('backend action helpers are profile-scoped', () => {
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

  it('forwards the active profile to memory provider config calls', () => {
    setApiRequestProfile('coder')

    void getMemoryProviderConfig('honcho')
    void saveMemoryProviderConfig('honcho', { workspace: 'w' })

    for (const call of api.mock.calls) {
      expect(call[0].profile).toBe('coder')
    }
  })

  it('forwards the active profile to every custom endpoint request', () => {
    setApiRequestProfile('coder')

    const endpoint = {
      name: 'Local endpoint',
      base_url: 'http://127.0.0.1:11434/v1',
      model: 'local-model'
    }

    void getCustomEndpoints()
    void saveCustomEndpoint(endpoint)
    void validateCustomEndpoint(endpoint)
    void activateCustomEndpoint('local-endpoint')
    void deleteCustomEndpoint('local-endpoint')

    expect(api).toHaveBeenCalledTimes(5)

    for (const call of api.mock.calls) {
      expect(call[0].profile, call[0].path).toBe('coder')
    }
  })

  it('forwards the active profile to every backend action', () => {
    setApiRequestProfile('coder')

    void getStatus()
    void restartGateway()
    void updateHermes()
    void checkHermesUpdate()
    void getActionStatus('gateway-restart')

    for (const call of api.mock.calls) {
      expect(call[0].profile).toBe('coder')
    }
  })
})
