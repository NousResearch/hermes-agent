import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  AUDIO_SPEAK_MAX_REQUEST_TIMEOUT_MS,
  AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS,
  audioSpeakRequestTimeoutMs,
  getSessionMessages,
  listAllProfileSessions,
  listSessions,
  speakText
} from './hermes'

const emptySessionsResponse = {
  limit: 0,
  offset: 0,
  sessions: [],
  total: 0
}

describe('Hermes REST helpers', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue(emptySessionsResponse)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('uses a longer timeout for the single-profile session list', async () => {
    await listSessions(50, 1)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/sessions?limit=50&offset=0&min_messages=1&archived=exclude&order=recent',
        timeoutMs: 60_000
      })
    )
  })

  it('uses a longer timeout for the all-profile session list', async () => {
    await listAllProfileSessions(50, 1)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/profiles/sessions?limit=50&offset=0&min_messages=1&archived=exclude&order=recent&profile=all',
        timeoutMs: 60_000
      })
    )
  })

  it('tags cross-profile message reads for Electron routing and backend lookup', async () => {
    api.mockResolvedValue({ messages: [], session_id: 'session-1' })

    await getSessionMessages('session-1', 'xiaoxuxu')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu',
      profile: 'xiaoxuxu'
    })
  })

  it('bounds blocking TTS synthesis timeouts by text length', () => {
    expect(audioSpeakRequestTimeoutMs('short message')).toBe(AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS)
    expect(audioSpeakRequestTimeoutMs('x'.repeat(8_000))).toBe(280_000)
    expect(audioSpeakRequestTimeoutMs('x'.repeat(100_000))).toBe(AUDIO_SPEAK_MAX_REQUEST_TIMEOUT_MS)
  })

  it('uses an extended timeout for blocking TTS synthesis', async () => {
    api.mockResolvedValueOnce({
      data_url: 'data:audio/mpeg;base64,AA==',
      mime_type: 'audio/mpeg',
      ok: true,
      provider: 'openai'
    })

    await expect(speakText('Read this aloud')).resolves.toEqual({
      data_url: 'data:audio/mpeg;base64,AA==',
      mime_type: 'audio/mpeg',
      ok: true,
      provider: 'openai'
    })

    expect(api).toHaveBeenCalledWith({
      body: { text: 'Read this aloud' },
      method: 'POST',
      path: '/api/audio/speak',
      timeoutMs: AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS
    })
  })
})
