import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./client', () => ({
  capabilityScoped: vi.fn(),
  hermesApi: vi.fn(),
  profileScoped: vi.fn()
}))

const client = await import('./client')
const { applyTelegramOnboarding, getMessagingPlatforms, updateMessagingPlatform } = await import('./messaging')

const owner = { connectionId: 'gateway-b', profile: 'default' }

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(client.capabilityScoped).mockReturnValue({
    connectionId: owner.connectionId,
    priority: 'foreground',
    profile: owner.profile
  })
})

describe('messaging owner routing', () => {
  it('pins reads and writes to the immutable gateway/profile owner', async () => {
    vi.mocked(client.hermesApi).mockResolvedValue({ platforms: [] } as never)

    await getMessagingPlatforms(owner)
    await updateMessagingPlatform('telegram', { enabled: true }, owner)

    expect(client.capabilityScoped).toHaveBeenNthCalledWith(1, owner)
    expect(client.capabilityScoped).toHaveBeenNthCalledWith(2, owner)
    expect(client.hermesApi).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ connectionId: 'gateway-b', path: '/api/messaging/platforms', profile: 'default' })
    )
    expect(client.hermesApi).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        body: { enabled: true },
        connectionId: 'gateway-b',
        method: 'PUT',
        path: '/api/messaging/platforms/telegram',
        profile: 'default'
      })
    )
  })

  it('keeps the owner profile in Telegram apply bodies', async () => {
    vi.mocked(client.hermesApi).mockResolvedValue({ ok: true } as never)

    await applyTelegramOnboarding('pair-1', ['123456'], owner)

    expect(client.hermesApi).toHaveBeenCalledWith(
      expect.objectContaining({
        body: { allowed_user_ids: ['123456'], profile: 'default' },
        connectionId: 'gateway-b',
        profile: 'default'
      })
    )
  })
})
