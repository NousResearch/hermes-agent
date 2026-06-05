import { afterEach, describe, expect, it, vi } from 'vitest'

import { transcribeAudio } from './hermes'

describe('Hermes REST audio helpers', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('uses an extended timeout for desktop audio transcription', async () => {
    const api = vi.fn(async <T,>() => ({ ok: true, transcript: 'hello', provider: 'local' }) as T)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })

    await expect(transcribeAudio('data:audio/webm;base64,AAAA', 'audio/webm')).resolves.toEqual({
      ok: true,
      transcript: 'hello',
      provider: 'local'
    })

    expect(api).toHaveBeenCalledWith({
      path: '/api/audio/transcribe',
      method: 'POST',
      timeoutMs: 300_000,
      body: {
        data_url: 'data:audio/webm;base64,AAAA',
        mime_type: 'audio/webm'
      }
    })
  })
})
