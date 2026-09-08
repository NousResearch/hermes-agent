import { beforeEach, expect, it, vi } from 'vitest'

vi.mock('./client', () => ({
  capabilityScoped: vi.fn(scope => scope),
  hermesApi: vi.fn(),
  STARTUP_REQUEST_TIMEOUT_MS: 60_000
}))

const client = await import('./client')
const { createProfile, getProfilesForScope } = await import('./profiles')
const hermesApi = vi.mocked(client.hermesApi)

beforeEach(() => vi.clearAllMocks())

it('routes profile reads and creates to an explicitly selected gateway', async () => {
  hermesApi.mockResolvedValue({ profiles: [] } as never)
  const scope = { connectionId: 'cloud' }

  await getProfilesForScope(scope)
  await createProfile({ clone_from: null, name: 'worker' }, scope)

  expect(hermesApi.mock.calls.map(([request]) => request)).toEqual([
    expect.objectContaining({ connectionId: 'cloud', path: '/api/profiles' }),
    expect.objectContaining({ connectionId: 'cloud', method: 'POST', path: '/api/profiles' })
  ])
})
