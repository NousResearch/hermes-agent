import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { clearVoiceClientConfigCache, type DirectSttConfig, transcribeAudioClientDirect } from './voice-client-direct'

const wires: DirectSttConfig['wire'][] = ['openai-multipart', 'xai-stt', 'elevenlabs-stt']

function configure(wire: DirectSttConfig['wire']) {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api: vi.fn(async () => ({
        ok: true,
        stt: {
          mode: 'direct',
          wire,
          provider: 'synthetic',
          base_url: 'https://speech.example.test/v1',
          api_key: 'synthetic-key',
          model: 'synthetic-model',
          language: null,
          timeout_s: 5
        },
        tts: { mode: 'relay' }
      }))
    }
  })
}

describe('direct STT response boundary', () => {
  beforeEach(() => {
    clearVoiceClientConfigCache()
    vi.useFakeTimers()
  })

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.useRealTimers()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it.each(wires.flatMap(wire => [200, 503].map(status => ({ wire, status }))))(
    'keeps one deadline through delayed headers and a stalled $status body on $wire',
    async ({ wire, status }) => {
      configure(wire)
      let stream: ReadableStreamDefaultController<Uint8Array> | undefined
      let signal: AbortSignal | undefined
      let outcome: unknown
      vi.stubGlobal(
        'fetch',
        vi.fn(
          (_url: string, init: RequestInit) =>
            new Promise<Response>(resolve => {
              signal = init.signal as AbortSignal
              setTimeout(() => {
                const body = new ReadableStream<Uint8Array>({
                  start(controller) {
                    stream = controller
                  }
                })

                signal?.addEventListener('abort', () => stream?.error(new DOMException('Aborted', 'AbortError')), {
                  once: true
                })
                resolve(new Response(body, { status }))
              }, 2_000)
            })
        )
      )
      const pending = transcribeAudioClientDirect(new Blob(['synthetic audio']))
      void pending.then(
        value => {
          outcome = value
        },
        error => {
          outcome = error
        }
      )

      try {
        await vi.advanceTimersByTimeAsync(4_999)
        expect(stream).toBeDefined()
        expect(outcome).toBeUndefined()
        await vi.advanceTimersByTimeAsync(1)
        expect(outcome).toBeInstanceOf(Error)
        expect(String(outcome)).toContain('Transcription timed out after 5s')
        expect(signal?.aborted).toBe(true)
      } finally {
        // Release the real Response reader even when proving the unfixed implementation.
        try {
          stream?.close()
        } catch {
          /* An aborted stream is already closed. */
        }

        await pending.catch(() => undefined)
      }
    }
  )

  it.each(
    wires.flatMap(wire =>
      [
        { error: { message: 'Synthetic failure' } },
        { detail: 'Synthetic failure' },
        { text: 'must not be submitted', error: 'Synthetic failure' }
      ].map(payload => ({ wire, payload }))
    )
  )('rejects an HTTP 200 error envelope on $wire: $payload', async ({ wire, payload }) => {
    configure(wire)
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(payload)))
    )
    await expect(transcribeAudioClientDirect(new Blob(['synthetic audio']))).rejects.toThrow(/transcription.*response/i)
  })

  it.each(wires)('finishes a valid %s body and clears its deadline', async wire => {
    configure(wire)
    let signal: AbortSignal | undefined
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init: RequestInit) => {
        signal = init.signal as AbortSignal

        return new Response(wire === 'openai-multipart' ? ' spoken words ' : JSON.stringify({ text: ' spoken words ' }))
      })
    )
    expect(await transcribeAudioClientDirect(new Blob(['synthetic audio']))).toBe('spoken words')
    await vi.advanceTimersByTimeAsync(6_000)
    expect(signal?.aborted).toBe(false)
  })

  it.each(wires)('preserves a non-timeout body-read failure on %s', async wire => {
    configure(wire)
    const failure = new Error('Synthetic body read failure')
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            new ReadableStream({
              start(controller) {
                controller.error(failure)
              }
            })
          )
      )
    )
    await expect(transcribeAudioClientDirect(new Blob(['synthetic audio']))).rejects.toBe(failure)
  })
})
