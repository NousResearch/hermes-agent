import { describe, expect, it, vi } from 'vitest'

const request = vi.fn(async (_profile: string, _method: string, _params?: Record<string, unknown>) => undefined)

vi.mock('@/store/gateway', () => ({
  activeGatewayProfileKey: () => 'work',
  requestGatewayForProfile: request
}))

vi.mock('@/lib/voice-live', () => ({
  fetchVoiceLiveStatus: async () => null
}))

const { setVoiceChatMode } = await import('./voice-live')

describe('setVoiceChatMode', () => {
  it('routes the write through the viewed profile, not the bare active socket', async () => {
    await setVoiceChatMode('gpt-live')

    // `config.set` is profile-scoped on the backend: an unscoped write on the
    // shared-primary route edits the LAUNCH profile's config.yaml (#125969 class).
    expect(request).toHaveBeenCalledWith('work', 'config.set', { key: 'voice.voice_chat_mode', value: 'gpt-live' })
  })
})
