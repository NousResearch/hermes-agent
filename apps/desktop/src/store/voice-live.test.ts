import { describe, expect, it, vi } from 'vitest'

const request = vi.fn(async (_profile: string, _method: string, _params?: Record<string, unknown>) => undefined)

vi.mock('@/store/gateway', () => ({
  activeGatewayProfileKey: () => 'work',
  requestGatewayForProfile: request
}))

vi.mock('@/lib/voice-live', () => ({
  fetchVoiceLiveStatus: async () => null
}))

const { $voiceLiveGrokStatus, $voiceLiveStatus, selectedVoiceChatMode, setVoiceChatMode } = await import('./voice-live')

describe('setVoiceChatMode', () => {
  it('routes the write through the viewed profile, not the bare active socket', async () => {
    await setVoiceChatMode('gpt-live')

    // `config.set` is profile-scoped on the backend: an unscoped write on the
    // shared-primary route edits the LAUNCH profile's config.yaml (#125969 class).
    expect(request).toHaveBeenCalledWith('work', 'config.set', { key: 'voice.voice_chat_mode', value: 'gpt-live' })
  })
})

describe('selectedVoiceChatMode', () => {
  it('falls back to chained when the backend has not answered', () => {
    expect(selectedVoiceChatMode(null)).toBe('chained')
  })

  it('falls back to chained when the backend predates the mode', () => {
    expect(
      selectedVoiceChatMode({ available: false, mode: 'unknown-future-mode' as never, model: '', reason: null, voice: '' })
    ).toBe('chained')
  })

  it('reports gpt-live when the status says so', () => {
    expect(selectedVoiceChatMode({ available: true, mode: 'gpt-live', model: 'gpt-live-1', reason: null, voice: 'marin' })).toBe(
      'gpt-live'
    )
  })

  it('reports grok-live when the status says so — the new third engine', () => {
    expect(
      selectedVoiceChatMode({ available: true, mode: 'grok-live', model: 'grok-voice-latest', reason: null, voice: 'eve' })
    ).toBe('grok-live')
  })

  it('reads the ambient $voiceLiveStatus atom when no argument is given', () => {
    $voiceLiveStatus.set({ available: true, mode: 'grok-live', model: 'grok-voice-latest', reason: null, voice: 'eve' })

    expect(selectedVoiceChatMode()).toBe('grok-live')

    $voiceLiveStatus.set(null)
  })

  it('falls back to the grok-live atom when the gpt-live status is chained/null', () => {
    $voiceLiveStatus.set({ available: false, mode: 'chained', model: '', reason: 'no OpenAI API key', voice: '' })
    $voiceLiveGrokStatus.set({ available: true, mode: 'grok-live', model: 'grok-voice-latest', reason: null, voice: 'eve' })

    expect(selectedVoiceChatMode()).toBe('grok-live')

    $voiceLiveStatus.set(null)
    $voiceLiveGrokStatus.set(null)
  })
})

describe('$voiceLiveGrokStatus', () => {
  it('is a separate atom from $voiceLiveStatus (parallel status route, SPEC §11)', () => {
    $voiceLiveStatus.set({ available: true, mode: 'grok-live', model: 'x', reason: null, voice: 'x' })
    $voiceLiveGrokStatus.set(null)

    expect($voiceLiveGrokStatus.get()).toBeNull()
    expect($voiceLiveStatus.get()).not.toBeNull()

    $voiceLiveStatus.set(null)
  })
})
