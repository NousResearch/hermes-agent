import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import {
  clearVoiceClientConfigCache,
  cutSentences,
  type DirectTtsConfig,
  fetchVoiceClientConfig,
  synthesizeSpeechClientDirect,
  transcribeAudioClientDirect,
  transcriptFromOpenAiMultipartBody
} from './voice-client-direct'

const directStt = {
  mode: 'direct',
  wire: 'openai-multipart',
  provider: 'groq',
  base_url: 'https://api.groq.com/openai/v1',
  api_key: 'gsk_test',
  model: 'whisper-large-v3-turbo',
  language: 'en'
} as const

const relay = { mode: 'relay', reason: 'local provider' } as const

function mockDesktopApi(response: unknown) {
  const api = vi.fn(async (_request: unknown) => response)

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api }
  })

  return api
}

describe('fetchVoiceClientConfig', () => {
  beforeEach(() => clearVoiceClientConfigCache())

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.restoreAllMocks()
  })

  it('fetches from /api/audio/voice-config with the ambient scope and caches per scope', async () => {
    const api = mockDesktopApi({ ok: true, stt: directStt, tts: relay })
    setApiRequestConnection('gw-remote')
    setApiRequestProfile('research')

    const first = await fetchVoiceClientConfig()
    const second = await fetchVoiceClientConfig()

    expect(first?.stt).toEqual(directStt)
    expect(second).toBe(first)
    // One fetch — the second call served from the scope cache.
    expect(api).toHaveBeenCalledTimes(1)

    const request = api.mock.calls[0][0] as { connectionId?: string; path: string; profile?: string }
    expect(request.path).toBe('/api/audio/voice-config')
    expect(request.profile).toBe('research')
    // hermesApi carries the ambient registry connection tag — the config
    // must come from the backend the user is talking to.
    expect(request.connectionId).toBe('gw-remote')
  })

  it("a scope switch never reuses another scope's credentials", async () => {
    const api = mockDesktopApi({ ok: true, stt: directStt, tts: relay })
    setApiRequestProfile('alpha')
    await fetchVoiceClientConfig()

    setApiRequestProfile('beta')
    await fetchVoiceClientConfig()

    expect(api).toHaveBeenCalledTimes(2)
  })

  it('resolves null on an older backend without the endpoint', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api: vi.fn(async () => Promise.reject(new Error('404'))) }
    })

    expect(await fetchVoiceClientConfig()).toBeNull()
  })
})

describe('transcribeAudioClientDirect', () => {
  beforeEach(() => clearVoiceClientConfigCache())

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('POSTs multipart to the provider and returns the transcript', async () => {
    mockDesktopApi({ ok: true, stt: directStt, tts: relay })

    const fetchMock = vi.fn(async () => new Response('  hello world  ', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const transcript = await transcribeAudioClientDirect(new Blob(['x'], { type: 'audio/webm' }))

    expect(transcript).toBe('hello world')

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('https://api.groq.com/openai/v1/audio/transcriptions')
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer gsk_test')
    // A hung provider must not stall dictation forever: every STT upload carries a timeout signal.
    expect(init.signal).toBeInstanceOf(AbortSignal)

    const form = init.body as FormData
    expect(form.get('model')).toBe('whisper-large-v3-turbo')
    expect(form.get('language')).toBe('en')
    expect(form.get('response_format')).toBe('text')
  })

  it('unwraps Mistral Voxtral JSON instead of dumping it into the composer', async () => {
    mockDesktopApi({
      ok: true,
      stt: {
        ...directStt,
        provider: 'mistral',
        base_url: 'https://api.mistral.ai/v1',
        model: 'voxtral-mini-latest',
        language: null
      },
      tts: relay
    })

    const voxtral = {
      model: 'voxtral-mini-latest',
      text: 'Hallo, bist du da?',
      language: null,
      segments: [],
      usage: { prompt_audio_seconds: 4, total_tokens: 388 },
      finish_reason: null
    }

    const fetchMock = vi.fn(async () => new Response(JSON.stringify(voxtral), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(await transcribeAudioClientDirect(new Blob(['x'], { type: 'audio/webm' }))).toBe('Hallo, bist du da?')
  })

  it('returns null (relay) when the provider is not client-callable', async () => {
    mockDesktopApi({ ok: true, stt: relay, tts: relay })
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    expect(await transcribeAudioClientDirect(new Blob(['x']))).toBeNull()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('surfaces provider rejections instead of silently relaying', async () => {
    mockDesktopApi({ ok: true, stt: directStt, tts: relay })
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ error: { message: 'invalid api key' } }), { status: 401 }))
    )

    await expect(transcribeAudioClientDirect(new Blob(['x']))).rejects.toThrow(/groq STT error.*invalid api key/)
  })

  it('speaks the xai wire shape', async () => {
    mockDesktopApi({
      ok: true,
      stt: {
        ...directStt,
        wire: 'xai-stt',
        provider: 'xai',
        base_url: 'https://api.x.ai/v1',
        model: 'grok-voice-transcribe-2.0'
      },
      tts: relay
    })

    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ text: 'grok heard this' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(await transcribeAudioClientDirect(new Blob(['x']))).toBe('grok heard this')

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('https://api.x.ai/v1/stt')
    expect((init.body as FormData).get('format')).toBe('true')
    expect((init.body as FormData).get('model')).toBe('grok-voice-transcribe-2.0')
  })

  it('drops format=true on the xai wire when no language is set (xAI answers HTTP 400 otherwise)', async () => {
    mockDesktopApi({
      ok: true,
      stt: {
        ...directStt,
        wire: 'xai-stt',
        provider: 'xai',
        base_url: 'https://api.x.ai/v1',
        model: 'grok-voice-transcribe-2.0',
        language: ''
      },
      tts: relay
    })

    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ text: 'auto detected' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(await transcribeAudioClientDirect(new Blob(['x']))).toBe('auto detected')

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect((init.body as FormData).has('format')).toBe(false)
    expect((init.body as FormData).has('language')).toBe(false)
  })

  it('speaks the elevenlabs wire shape with xi-api-key auth', async () => {
    mockDesktopApi({
      ok: true,
      stt: {
        ...directStt,
        wire: 'elevenlabs-stt',
        provider: 'elevenlabs',
        base_url: 'https://api.elevenlabs.io/v1',
        model: 'scribe_v2'
      },
      tts: relay
    })

    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ text: 'scribe text' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(await transcribeAudioClientDirect(new Blob(['x']))).toBe('scribe text')

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('https://api.elevenlabs.io/v1/speech-to-text')
    expect((init.headers as Record<string, string>)['xi-api-key']).toBe('gsk_test')
    expect((init.body as FormData).get('model_id')).toBe('scribe_v2')
  })

  /** A fetch that only settles when its AbortSignal fires — a wedged STT endpoint. */
  function hangingFetch() {
    return vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
        })
    )
  }

  it('aborts a hanging transcription at the default 60 s instead of transcribing forever', async () => {
    vi.useFakeTimers()

    try {
      mockDesktopApi({ ok: true, stt: directStt, tts: relay })
      const fetchMock = hangingFetch()
      vi.stubGlobal('fetch', fetchMock)

      const pending = transcribeAudioClientDirect(new Blob(['x'], { type: 'audio/webm' }))
      const settled = vi.fn()

      pending.then(settled, settled)
      await vi.advanceTimersByTimeAsync(0)

      const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
      expect(init.signal).toBeInstanceOf(AbortSignal)

      await vi.advanceTimersByTimeAsync(59_000)
      expect(settled).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(1_000)
      await expect(pending).rejects.toThrow(/Transcription timed out after 60s/)
    } finally {
      vi.useRealTimers()
    }
  })

  it('honours the gateway-resolved stt.openai.timeout for the direct request', async () => {
    vi.useFakeTimers()

    try {
      mockDesktopApi({ ok: true, stt: { ...directStt, timeout_s: 5 }, tts: relay })
      vi.stubGlobal('fetch', hangingFetch())

      const pending = transcribeAudioClientDirect(new Blob(['x'], { type: 'audio/webm' }))
      const settled = vi.fn()

      pending.then(settled, settled)
      await vi.advanceTimersByTimeAsync(4_900)
      expect(settled).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(200)
      await expect(pending).rejects.toThrow(/Transcription timed out after 5s/)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('synthesizeSpeechClientDirect', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  const openaiTts: DirectTtsConfig = {
    mode: 'direct',
    wire: 'openai-speech',
    provider: 'openai',
    base_url: 'https://api.openai.com/v1',
    api_key: 'sk_tts',
    model: 'gpt-4o-mini-tts',
    voice: 'nova',
    speed: null
  }

  it('POSTs the openai speech shape and returns audio bytes', async () => {
    const bytes = new Uint8Array([1, 2, 3]).buffer

    const fetchMock = vi.fn(async () => new Response(bytes, { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const audio = await synthesizeSpeechClientDirect(
      { ...openaiTts, extra_body: { consent_attestation: 'I own this voice' } },
      'Hello there.'
    )

    expect(new Uint8Array(audio)).toEqual(new Uint8Array([1, 2, 3]))

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('https://api.openai.com/v1/audio/speech')

    const body = JSON.parse(String(init.body)) as Record<string, unknown>
    expect(body.model).toBe('gpt-4o-mini-tts')
    expect(body.voice).toBe('nova')
    expect(body.input).toBe('Hello there.')
    expect(body.speed).toBeUndefined()
    // Server-resolved tts.openai extras (consent_attestation for cloned voices) reach the wire.
    expect(body.consent_attestation).toBe('I own this voice')
  })

  it('speaks the elevenlabs tts shape with the voice in the path', async () => {
    const fetchMock = vi.fn(async () => new Response(new ArrayBuffer(4), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await synthesizeSpeechClientDirect(
      {
        ...openaiTts,
        wire: 'elevenlabs-tts',
        provider: 'elevenlabs',
        base_url: 'https://api.elevenlabs.io/v1',
        model: 'eleven_turbo_v2',
        voice: 'voice123'
      },
      'Hi.'
    )

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('https://api.elevenlabs.io/v1/text-to-speech/voice123')
    expect((init.headers as Record<string, string>)['xi-api-key']).toBe('sk_tts')
  })

  it('throws on provider rejection', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('quota exceeded', { status: 429 }))
    )

    await expect(synthesizeSpeechClientDirect(openaiTts, 'Hi.')).rejects.toThrow(/openai TTS error.*429/)
  })
})

describe('transcriptFromOpenAiMultipartBody', () => {
  it('keeps Groq/OpenAI plain-text bodies', () => {
    expect(transcriptFromOpenAiMultipartBody('  hello world  ')).toBe('hello world')
  })

  it('extracts .text from a Voxtral JSON envelope', () => {
    expect(
      transcriptFromOpenAiMultipartBody(
        JSON.stringify({
          model: 'voxtral-mini-latest',
          text: 'Hallo, bist du da?',
          language: null,
          segments: [],
          usage: { prompt_audio_seconds: 4 }
        })
      )
    ).toBe('Hallo, bist du da?')
  })

  it('treats a JSON envelope with empty text as silence', () => {
    expect(transcriptFromOpenAiMultipartBody(JSON.stringify({ text: '  ', model: 'voxtral-mini-latest' }))).toBe('')
  })

  it('leaves a non-envelope JSON object intact', () => {
    expect(transcriptFromOpenAiMultipartBody('{"error":"nope"}')).toBe('{"error":"nope"}')
  })
})

describe('cutSentences', () => {
  it('emits complete sentences and holds the incomplete tail', () => {
    const { sentences, rest } = cutSentences('This is the first full sentence. And then it keeps goi', false)

    expect(sentences).toEqual(['This is the first full sentence.'])
    expect(rest).toBe('And then it keeps goi')
  })

  it('buffers too-short fragments instead of firing per abbreviation', () => {
    const { sentences, rest } = cutSentences('e.g. it continues', false)

    expect(sentences).toEqual([])
    expect(rest).toBe('e.g. it continues')
  })

  it('flush drains everything including the tail', () => {
    const { sentences, rest } = cutSentences('First complete sentence right here. tail bit', true)

    expect(sentences).toEqual(['First complete sentence right here.', 'tail bit'])
    expect(rest).toBe('')
  })

  it('handles CJK terminators', () => {
    const { sentences } = cutSentences(
      '这是一个完整的中文句子，它的长度足够超过最小句子门槛，所以会被切分出来。 下一句',
      true
    )

    expect(sentences[0]).toContain('。')
    expect(sentences).toHaveLength(2)
  })

  it('cuts a short CJK opener alone when the backend sends tts.streaming.min_len', () => {
    const text = '记得，叫团团。 然后我们再说第二句话，这一句要长一些才行。 '

    // Historical 24-char floor (older backend, no key): the opener rides with sentence two.
    expect(cutSentences(text, false).sentences).toEqual(['记得，叫团团。 然后我们再说第二句话，这一句要长一些才行。'])
    // tts.streaming.min_len = 6 (the CJK voice setup from #96927): spoken on its own.
    expect(cutSentences(text, false, 6).sentences).toEqual([
      '记得，叫团团。',
      '然后我们再说第二句话，这一句要长一些才行。'
    ])
  })
})
