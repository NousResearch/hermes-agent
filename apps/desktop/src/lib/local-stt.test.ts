import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $localStt,
  applyLocalSttFromConfig,
  audioFileFromDataUrl,
  LOCAL_STT_DEFAULT_MODEL,
  LOCAL_STT_DEFAULT_PORT,
  LOCAL_STT_DEFAULTS,
  LOCAL_STT_MAX_TIMEOUT_MS,
  LOCAL_STT_MIN_TIMEOUT_MS,
  localSttEndpoint,
  localSttSettingsFromConfig,
  localSttTimeoutMs,
  transcribeWithLocalStt
} from './local-stt'

const WEBM_DATA_URL = 'data:audio/webm;base64,AAECAw=='

const localSettings = (over: Partial<typeof LOCAL_STT_DEFAULTS> = {}) => ({
  ...LOCAL_STT_DEFAULTS,
  mode: 'local' as const,
  ...over
})

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

afterEach(() => {
  vi.unstubAllGlobals()
  $localStt.set(LOCAL_STT_DEFAULTS)
})

describe('localSttSettingsFromConfig', () => {
  it('defaults to the backend route when nothing is configured', () => {
    for (const config of [undefined, null, {}, { voice: {} }, { voice: { dictation: null } }]) {
      expect(localSttSettingsFromConfig(config)).toEqual(LOCAL_STT_DEFAULTS)
    }
  })

  it('only switches to local for the exact opt-in value', () => {
    expect(localSttSettingsFromConfig({ voice: { dictation: { stt: 'local' } } }).mode).toBe('local')
    expect(localSttSettingsFromConfig({ voice: { dictation: { stt: '  Local  ' } } }).mode).toBe('local')

    // Anything else — including a typo or a truthy non-string — stays on the
    // untouched backend path rather than silently routing to loopback.
    for (const value of ['backend', 'locale', 'whisper', '', true, 1, null]) {
      expect(localSttSettingsFromConfig({ voice: { dictation: { stt: value } } }).mode, String(value)).toBe('backend')
    }
  })

  it('accepts a hand-edited port as a number or a string, and rejects nonsense', () => {
    expect(localSttSettingsFromConfig({ voice: { dictation: { local_stt_port: 9000 } } }).port).toBe(9000)
    expect(localSttSettingsFromConfig({ voice: { dictation: { local_stt_port: ' 9000 ' } } }).port).toBe(9000)

    for (const value of [0, -1, 65_536, 1.5, 'abc', '', null, {}]) {
      expect(localSttSettingsFromConfig({ voice: { dictation: { local_stt_port: value } } }).port, String(value)).toBe(
        LOCAL_STT_DEFAULT_PORT
      )
    }
  })

  it('validates a quoted port exactly like an unquoted one', () => {
    // parseInt() would read "1.5" as port 1 (privileged) and "9000.5" as 9000,
    // so a fractional port would be accepted when quoted and rejected when not.
    for (const value of [1.5, '1.5', 9000.5, '9000.5', '65536', ' 12 abc', '1e999']) {
      expect(localSttSettingsFromConfig({ voice: { dictation: { local_stt_port: value } } }).port, String(value)).toBe(
        LOCAL_STT_DEFAULT_PORT
      )
    }
  })

  it('resolves the model id, falling back rather than sending a blank part', () => {
    // Servers that route by model id 400 on a blank part; those that ignore it
    // are unaffected either way — so an unusable value resolves to the default.
    expect(
      localSttSettingsFromConfig({ voice: { dictation: { local_stt_model: ' Systran/faster-whisper-base ' } } }).model
    ).toBe('Systran/faster-whisper-base')

    for (const value of ['', '   ', null, 7, {}]) {
      expect(
        localSttSettingsFromConfig({ voice: { dictation: { local_stt_model: value } } }).model,
        String(value)
      ).toBe(LOCAL_STT_DEFAULT_MODEL)
    }
  })

  it('carries the global stt.language hint, trimmed', () => {
    expect(localSttSettingsFromConfig({ stt: { language: ' pl ' } }).language).toBe('pl')
    expect(localSttSettingsFromConfig({ stt: { language: 7 as unknown as string } }).language).toBe('')
  })
})

describe('applyLocalSttFromConfig', () => {
  it('publishes the resolved settings for the synchronous read in transcribeAudio', () => {
    applyLocalSttFromConfig({
      stt: { language: 'pl' },
      voice: { dictation: { stt: 'local', local_stt_port: 9001, local_stt_model: 'tiny' } }
    })
    expect($localStt.get()).toEqual({ mode: 'local', port: 9001, model: 'tiny', language: 'pl' })

    // A profile whose config drops the block must fall back, not inherit.
    applyLocalSttFromConfig({})
    expect($localStt.get()).toEqual(LOCAL_STT_DEFAULTS)
  })
})

describe('audioFileFromDataUrl', () => {
  it('decodes base64 audio into real bytes for the multipart upload', async () => {
    const file = audioFileFromDataUrl(WEBM_DATA_URL, 'audio/webm')

    expect(file.type).toBe('audio/webm')
    expect([...new Uint8Array(await file.arrayBuffer())]).toEqual([0, 1, 2, 3])
  })

  it('names the part by mime type so ffmpeg-backed servers pick the right decoder', () => {
    expect(audioFileFromDataUrl(WEBM_DATA_URL, 'audio/webm').name).toBe('dictation.webm')
    expect(audioFileFromDataUrl('data:audio/wav;base64,AAA=', 'audio/wav').name).toBe('dictation.wav')
    expect(audioFileFromDataUrl('data:audio/ogg;base64,AAA=', 'audio/ogg').name).toBe('dictation.ogg')
    // Unknown container → the recorder's own default, never an extensionless blob.
    expect(audioFileFromDataUrl('data:audio/x-weird;base64,AAA=', 'audio/x-weird').name).toBe('dictation.webm')
  })

  it('falls back to the data-URL header when the recorder reports no mime type', () => {
    expect(audioFileFromDataUrl(WEBM_DATA_URL).type).toBe('audio/webm')
    expect(audioFileFromDataUrl(WEBM_DATA_URL, '   ').type).toBe('audio/webm')
  })

  it('rejects payloads that are not data URLs', () => {
    expect(() => audioFileFromDataUrl('https://example.com/clip.webm')).toThrow(/unsupported audio payload/i)
    expect(() => audioFileFromDataUrl('data:audio/webm;base64')).toThrow(/unsupported audio payload/i)
  })
})

describe('localSttTimeoutMs', () => {
  it('holds a short clip at the floor so an absent server is detected fast', () => {
    for (const dataUrl of ['', WEBM_DATA_URL, `data:audio/webm;base64,${'A'.repeat(1_000)}`]) {
      expect(localSttTimeoutMs(dataUrl)).toBe(LOCAL_STT_MIN_TIMEOUT_MS)
    }
  })

  it('grows the budget with clip length so a slow-but-alive server can finish', () => {
    // A flat floor bounds the transcription itself, so a long clip on a local
    // server would abort every time and never beat the backend it falls to.
    const long = localSttTimeoutMs(`data:audio/webm;base64,${'A'.repeat(200_000)}`)

    expect(long).toBeGreaterThan(LOCAL_STT_MIN_TIMEOUT_MS)
    expect(long).toBeLessThanOrEqual(LOCAL_STT_MAX_TIMEOUT_MS)
  })

  it('caps the wait so a wedged server never out-costs the backend rung', () => {
    expect(localSttTimeoutMs(`data:audio/webm;base64,${'A'.repeat(50_000_000)}`)).toBe(LOCAL_STT_MAX_TIMEOUT_MS)
  })
})

describe('transcribeWithLocalStt', () => {
  it('posts OpenAI-compatible multipart to loopback and returns the text', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ text: 'hello there' }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(transcribeWithLocalStt(WEBM_DATA_URL, 'audio/webm', localSettings({ port: 9001 }))).resolves.toBe(
      'hello there'
    )

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe(localSttEndpoint(9001))
    expect(url).toBe('http://127.0.0.1:9001/v1/audio/transcriptions')
    expect(init.method).toBe('POST')

    const form = init.body as FormData
    expect((form.get('file') as File).name).toBe('dictation.webm')
    expect(form.get('model')).toBe(LOCAL_STT_DEFAULT_MODEL)
    // Blank stt.language means auto-detect — sending '' would force it.
    expect(form.has('language')).toBe(false)
  })

  it('forwards the configured model id for servers that route on it', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ text: 'ok' }))
    vi.stubGlobal('fetch', fetchMock)

    await transcribeWithLocalStt(WEBM_DATA_URL, 'audio/webm', localSettings({ model: 'Systran/faster-whisper-base' }))

    expect((fetchMock.mock.calls[0][1].body as FormData).get('model')).toBe('Systran/faster-whisper-base')
  })

  it('forwards the configured language hint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ text: 'dzień dobry' }))
    vi.stubGlobal('fetch', fetchMock)

    await transcribeWithLocalStt(WEBM_DATA_URL, 'audio/webm', localSettings({ language: 'pl' }))

    expect((fetchMock.mock.calls[0][1].body as FormData).get('language')).toBe('pl')
  })

  it('throws on a non-2xx response so the caller can fall back', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ error: 'nope' }, 503)))

    await expect(transcribeWithLocalStt(WEBM_DATA_URL, 'audio/webm', localSettings())).rejects.toThrow(/503/)
  })

  it('throws when the body carries no usable transcript', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ segments: [] })))

    await expect(transcribeWithLocalStt(WEBM_DATA_URL, 'audio/webm', localSettings())).rejects.toThrow(/no transcript/i)
  })

  it('aborts a hung server instead of blocking dictation', async () => {
    vi.useFakeTimers()

    try {
      const fetchMock = vi.fn(
        (_url: string, init: { signal: AbortSignal }) =>
          new Promise((_resolve, reject) => {
            init.signal.addEventListener('abort', () => reject(new Error('aborted')))
          })
      )

      vi.stubGlobal('fetch', fetchMock)

      const pending = transcribeWithLocalStt(WEBM_DATA_URL, 'audio/webm', localSettings())
      const assertion = expect(pending).rejects.toThrow(/aborted/)

      await vi.advanceTimersByTimeAsync(3_000)
      await assertion
    } finally {
      vi.useRealTimers()
    }
  })

  it('lets a long clip transcribe past the short-clip floor', async () => {
    vi.useFakeTimers()

    try {
      let settle = (_response: Response) => {}

      const responded = new Promise<Response>(resolve => {
        settle = resolve
      })

      vi.stubGlobal(
        'fetch',
        vi.fn(
          (_url: string, init: { signal: AbortSignal }) =>
            new Promise<Response>((resolve, reject) => {
              void responded.then(resolve)
              init.signal.addEventListener('abort', () => reject(new Error('aborted')))
            })
        )
      )

      const longClip = `data:audio/webm;base64,${'A'.repeat(200_000)}`
      const pending = transcribeWithLocalStt(longClip, 'audio/webm', localSettings())

      // Still alive well past the floor that used to bound the whole roundtrip.
      await vi.advanceTimersByTimeAsync(LOCAL_STT_MIN_TIMEOUT_MS * 2)
      settle(jsonResponse({ text: 'a long dictation' }))

      await expect(pending).resolves.toBe('a long dictation')
    } finally {
      vi.useRealTimers()
    }
  })
})
