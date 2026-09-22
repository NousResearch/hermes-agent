import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./client', () => ({
  capabilityScoped: vi.fn(),
  hermesApi: vi.fn(),
  ownerScoped: vi.fn(),
  profileScoped: vi.fn()
}))

const client = await import('./client')
const { restartGateway } = await import('./system')

describe('restartGateway', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(client.capabilityScoped).mockReturnValue({
      connectionId: 'gateway-a',
      priority: 'foreground',
      profile: 'worker'
    })
    vi.mocked(client.hermesApi).mockResolvedValue({ name: 'gateway-restart' })
  })

  it('pins the restart request to the supplied gateway owner', async () => {
    const owner = { connectionId: 'gateway-a', profile: 'worker' }

    await restartGateway(owner)

    expect(client.capabilityScoped).toHaveBeenCalledWith(owner)
    expect(client.hermesApi).toHaveBeenCalledWith({
      connectionId: 'gateway-a',
      priority: 'foreground',
      profile: 'worker',
      path: '/api/gateway/restart',
      method: 'POST'
    })
  })
})
