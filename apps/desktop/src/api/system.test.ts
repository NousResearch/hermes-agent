import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./client', () => ({
  capabilityScoped: vi.fn(),
  hermesApi: vi.fn(),
  ownerScoped: vi.fn(),
  profileScoped: vi.fn()
}))

const client = await import('./client')
const { getMemoryStatus, restartGateway } = await import('./system')

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

describe('getMemoryStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(client.capabilityScoped).mockReturnValue({
      connectionId: 'gateway-b',
      priority: 'foreground',
      profile: 'memory-owner'
    })
    vi.mocked(client.hermesApi).mockResolvedValue({ active: '', builtin_files: {}, providers: [] })
  })

  it('pins the status read to the supplied Settings owner', async () => {
    const owner = { connectionId: 'gateway-b', profile: 'memory-owner' }

    await getMemoryStatus(owner)

    expect(client.capabilityScoped).toHaveBeenCalledWith(owner)
    expect(client.hermesApi).toHaveBeenCalledWith({
      connectionId: 'gateway-b',
      priority: 'foreground',
      profile: 'memory-owner',
      path: '/api/memory'
    })
  })
})
