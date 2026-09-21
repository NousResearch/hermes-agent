import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./client', () => ({
  STARTUP_REQUEST_TIMEOUT_MS: 60_000,
  capabilityScoped: vi.fn(scope =>
    scope && typeof scope === 'object'
      ? {
          ...(scope.connectionId ? { connectionId: scope.connectionId } : {}),
          ...(scope.profile ? { profile: scope.profile } : {})
        }
      : {}
  ),
  hermesApi: vi.fn()
}))

const client = await import('./client')
const { getGlobalModelOptions, setModelAssignment } = await import('./models')

const hermesApi = vi.mocked(client.hermesApi)

beforeEach(() => {
  vi.clearAllMocks()
  hermesApi.mockResolvedValue({ ok: true } as never)
})

describe('model API explicit capability scope', () => {
  it('routes model options to the explicit connection and profile', async () => {
    await getGlobalModelOptions(
      { refresh: true, includeUnconfigured: true, explicitOnly: false },
      { connectionId: 'remote-a', profile: 'research' }
    )

    expect(hermesApi).toHaveBeenCalledWith({
      connectionId: 'remote-a',
      path: '/api/model/options?refresh=1&include_unconfigured=1',
      profile: 'research',
      timeoutMs: 60_000
    })
  })

  it('routes model assignment to the explicit connection and profile', async () => {
    await setModelAssignment(
      { model: 'llama-3', provider: 'fireworks', scope: 'main' },
      { connectionId: 'remote-a', profile: 'research' }
    )

    expect(hermesApi).toHaveBeenCalledWith({
      body: { model: 'llama-3', provider: 'fireworks', scope: 'main' },
      connectionId: 'remote-a',
      method: 'POST',
      path: '/api/model/set',
      profile: 'research'
    })
  })
})
