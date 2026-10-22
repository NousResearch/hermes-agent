import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { VoiceLiveStatus } from '@/lib/voice-live'

const { fetches, request, route } = vi.hoisted(() => ({
  fetches: [] as Array<{ resolve: (status: null | VoiceLiveStatus) => void }>,
  request: vi.fn(async (_profile: string, _method: string, _params?: Record<string, unknown>) => undefined),
  route: { connectionId: 'conn-a', profile: 'work' }
}))

vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: () => route.connectionId,
  activeGatewayProfileKey: () => route.profile,
  requestGatewayForProfile: request
}))

vi.mock('@/lib/voice-live', () => ({
  fetchVoiceLiveStatus: () =>
    new Promise<null | VoiceLiveStatus>(resolve => {
      fetches.push({ resolve })
    })
}))

const { $voiceLiveStatus, refreshVoiceLiveStatus, setVoiceChatMode } = await import('./voice-live')

function status(mode: VoiceLiveStatus['mode']): VoiceLiveStatus {
  return { available: true, mode, model: 'gpt-live-1', reason: null, voice: 'alloy' }
}

// Settle every outstanding fetch so no in-flight slot leaks into the next test.
async function drainFetches(): Promise<void> {
  while (fetches.length) {
    fetches.shift()?.resolve(null)
    await new Promise(resolve => setTimeout(resolve, 0))
  }
}

beforeEach(async () => {
  await drainFetches()
  route.connectionId = 'conn-a'
  route.profile = 'work'
  $voiceLiveStatus.set(null)
})

describe('setVoiceChatMode', () => {
  it('routes the write through the viewed profile, not the bare active socket', async () => {
    const pending = setVoiceChatMode('gpt-live')

    await vi.waitFor(() => expect(fetches).toHaveLength(1))
    fetches[0].resolve(null)
    await pending

    // `config.set` is profile-scoped on the backend: an unscoped write on the
    // shared-primary route edits the LAUNCH profile's config.yaml (#125969 class).
    expect(request).toHaveBeenCalledWith('work', 'config.set', { key: 'voice.voice_chat_mode', value: 'gpt-live' })
  })

  it('re-reads the status after the write instead of joining a pre-write fetch', async () => {
    void refreshVoiceLiveStatus()
    const pending = setVoiceChatMode('gpt-live')

    // Let the mocked config.set settle so the post-write re-read is issued.
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(fetches).toHaveLength(2)

    fetches[0].resolve(status('chained'))
    fetches[1].resolve(status('gpt-live'))

    expect((await pending)?.mode).toBe('gpt-live')
    expect($voiceLiveStatus.get()?.mode).toBe('gpt-live')
  })
})

describe('refreshVoiceLiveStatus', () => {
  it('does not let the previous profile publish its status after a switch', async () => {
    const previous = refreshVoiceLiveStatus()
    route.profile = 'home'
    const current = refreshVoiceLiveStatus()

    expect(fetches).toHaveLength(2)
    fetches[1].resolve(status('chained'))
    await current
    fetches[0].resolve(status('gpt-live'))
    await previous

    expect($voiceLiveStatus.get()?.mode).toBe('chained')
  })

  it('shares one fetch between callers on the same scope', async () => {
    const first = refreshVoiceLiveStatus()
    const second = refreshVoiceLiveStatus()

    expect(fetches).toHaveLength(1)
    fetches[0].resolve(status('gpt-live'))

    expect(await first).toEqual(await second)
    expect($voiceLiveStatus.get()?.mode).toBe('gpt-live')
  })
})
