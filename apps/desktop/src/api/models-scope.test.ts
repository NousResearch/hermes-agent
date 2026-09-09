import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import { getHermesConfigSchema, saveHermesConfigRecord } from './config'
import { getGlobalModelOptions } from './models'

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

describe('model options owner scope', () => {
  it('uses the same explicit owner for capability checks and delegation writes', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true, fields: {} })
    vi.stubGlobal('window', { hermesDesktop: { api } })
    setApiRequestConnection('ambient-gateway')
    setApiRequestProfile('ambient-profile')
    const scope = { connectionId: 'local', profile: 'worker-profile' }
    await getHermesConfigSchema(scope)
    await saveHermesConfigRecord({ delegation: { fallback_providers: [] } }, scope)
    expect(api.mock.calls[0][0]).toMatchObject({ path: '/api/config/schema', ...scope })
    expect(api.mock.calls[1][0]).toMatchObject({ path: '/api/config', method: 'PUT', ...scope })
  })

  it('pins the catalog to the same connection and profile as the configuration', async () => {
    const api = vi.fn().mockResolvedValue({ providers: [] })
    vi.stubGlobal('window', { hermesDesktop: { api } })
    setApiRequestConnection('ambient-gateway')
    setApiRequestProfile('ambient-profile')
    await getGlobalModelOptions(undefined, { connectionId: 'worker-gateway', profile: 'worker-profile' })
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'worker-gateway', profile: 'worker-profile' }))
  })

  it('preserves legacy profile-only callers', async () => {
    const api = vi.fn().mockResolvedValue({ providers: [] })
    vi.stubGlobal('window', { hermesDesktop: { api } })
    setApiRequestConnection('ambient-gateway')
    await getGlobalModelOptions(undefined, 'worker-profile')
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'ambient-gateway', profile: 'worker-profile' }))
  })
})
