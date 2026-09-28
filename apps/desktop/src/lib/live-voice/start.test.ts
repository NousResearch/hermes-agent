import { describe, expect, it, vi } from 'vitest'

const providers = vi.hoisted(() => ({
  gemini: vi.fn(async () => ({ setMuted: vi.fn(), stop: vi.fn() })),
  openai: vi.fn(async () => ({ setMuted: vi.fn(), stop: vi.fn() }))
}))

vi.mock('../gemini-live-voice', () => ({ startGeminiLiveVoice: providers.gemini }))
vi.mock('../realtime-voice', () => ({ startRealtimeVoice: providers.openai }))

import { startLiveVoice } from './start'

const handlers = { onAsk: vi.fn(), onDelegate: vi.fn(), onError: vi.fn(), onStatus: vi.fn() }

describe('startLiveVoice', () => {
  it('follows the provider the backend minted the session for', async () => {
    const gemini = {
      language: 'pl',
      model: 'gemini-3.8-live',
      provider: 'gemini',
      setup: {},
      token: 't',
      voice: 'Charon',
      ws_url: 'wss://x'
    } as const

    await startLiveVoice(handlers, { createSession: async () => gemini })
    expect(providers.gemini).toHaveBeenCalledWith(handlers, expect.objectContaining({ session: gemini }))
    expect(providers.openai).not.toHaveBeenCalled()
  })

  it('treats an answer without a provider (an older backend) as OpenAI', async () => {
    providers.gemini.mockClear()

    const legacy = {
      calls_url: 'https://x/calls',
      client_secret: 'ek',
      expires_at: null,
      model: 'gpt-realtime',
      voice: 'marin'
    }

    await startLiveVoice(handlers, { createSession: async () => legacy as never })
    expect(providers.openai).toHaveBeenCalledOnce()
    expect(providers.gemini).not.toHaveBeenCalled()
  })
})
