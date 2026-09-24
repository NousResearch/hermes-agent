import { profileScoped } from '@/api/client'
import { getApiRequestConnection, getApiRequestProfile, hermesApi } from '@/hermes'
import { resolveSiblingWsUrl } from '@/lib/sibling-ws-url'
import { withTimeout } from '@/lib/with-timeout'

/**
 * Streaming speech-to-text for the composer.
 *
 * While the user records, the desktop streams 16 kHz mono PCM to the gateway's
 * `/api/audio/transcribe-stream` WebSocket and renders the provider's `partial`
 * transcripts live, instead of waiting for the whole utterance to be uploaded
 * and transcribed in one shot. The endpoint is provider-neutral: it resolves
 * the active profile's STT provider (same `/api/audio/voice-config` seam as the
 * one-shot path) and the desktop carries no provider-specific logic.
 *
 * The gateway is the single source of truth for whether streaming is even
 * offered (`stt_streaming` on the voice config). Older backends, profiles whose
 * provider cannot stream, and any transport failure all resolve to `null` so
 * callers keep the existing file-based transcription path untouched.
 */

export interface TranscriptionStream {
  attach: (stream: MediaStream) => Promise<void>
  finish: () => Promise<string>
  cancel: () => void
}

/** Resolve `ws(s)://<gateway>/api/audio/transcribe-stream` for the ACTIVE
 *  (connection, profile) route. Rides the shared sibling-URL resolver so a
 *  registry-scoped remote mints its own credential (a fresh ticket) rather than
 *  answering for the local primary backend. Returns null when unavailable. */
export async function resolveTranscriptionStreamUrl(): Promise<null | string> {
  try {
    const profile = getApiRequestProfile()

    const url = new URL(
      await resolveSiblingWsUrl(
        { connectionId: getApiRequestConnection(), profile },
        '/api/audio/transcribe-stream'
      )
    )

    // The backend resolves the STT provider from this profile's config/.env
    // (same seam as /api/pty?profile=). A registry-minted URL may already carry
    // the BACKEND-namespace profile — never overwrite it with the desktop-side
    // routing alias.
    if (profile && !url.searchParams.has('profile')) {
      url.searchParams.set('profile', profile)
    }

    return url.toString()
  } catch {
    return null
  }
}

/** True when the connected gateway advertises the streaming STT capability.
 *  Absent endpoint or `stt_streaming: false` → false (use the file path). */
async function streamingAvailable(): Promise<boolean> {
  try {
    const capability = await hermesApi<{ stt_streaming?: boolean }>({
      ...profileScoped(),
      path: '/api/audio/voice-config'
    })

    return Boolean(capability?.stt_streaming)
  } catch {
    return false
  }
}

// The AudioContext performs resampling; the worklet batches 16 kHz mono PCM.
const processor = `
class Capture extends AudioWorkletProcessor {
  constructor() {
    super(); this.samples = [];
    this.port.onmessage = () => { this.flush(); this.port.postMessage('flushed'); };
  }
  flush() {
    if (!this.samples.length) return;
    const pcm = new Int16Array(this.samples); this.samples = [];
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
  }
  process(inputs) {
    const mono = inputs[0]?.[0];
    if (mono) for (const x of mono) this.samples.push(Math.round(Math.max(-1, Math.min(1, x)) * 32767));
    if (this.samples.length >= 4000) this.flush();
    return true;
  }
}
registerProcessor('asr-capture', Capture);`

/**
 * Open a streaming transcription session, or null when streaming is not
 * available (no capability, no route) so the caller falls back to the existing
 * file-based path. `onPartial` receives live partial text while recording.
 */
export async function openTranscriptionStream(onPartial: (text: string) => void): Promise<TranscriptionStream | null> {
  // Older backends and non-streaming providers retain the existing file path.
  if (!(await streamingAvailable())) {
    return null
  }

  const url = await resolveTranscriptionStreamUrl()

  if (!url) {
    return null
  }

  const ws = new WebSocket(url)
  let context: AudioContext | null = null
  let source: MediaStreamAudioSourceNode | null = null
  let node: AudioWorkletNode | null = null
  let error: Error | null = null
  let done = false
  let readyToSend = false
  let pending: ArrayBuffer[] = []
  let pendingBytes = 0
  let rejectReady: (error: Error) => void = () => undefined
  let resolveFinal: (text: string) => void = () => undefined
  let rejectFinal: (error: Error) => void = () => undefined
  let flushed: (() => void) | null = null

  const final = new Promise<string>((resolve, reject) => {
    resolveFinal = resolve
    rejectFinal = reject
  })

  // An error may arrive while recording, before finish() starts awaiting final.
  void final.catch(() => undefined)

  const fail = (reason: string) => {
    error = new Error(reason)
    rejectFinal(error)
  }

  const cancel = () => {
    done = true
    pending = []
    pendingBytes = 0
    rejectReady(new Error('ASR recording cancelled'))
    void context?.close()
    context = null

    if (ws.readyState === WebSocket.OPEN) {
      ws.send('cancel')
    }

    ws.close()
  }

  // Capture starts while the model warms. Queue a bounded amount of PCM so
  // the beginning of the utterance is preserved even on a cold start.
  const ready = withTimeout(
    new Promise<void>((resolve, reject) => {
      rejectReady = reject

      ws.onerror = () => {
        fail('ASR connection failed')
        reject(error)
      }

      ws.onclose = () => {
        if (!done) {
          fail('ASR connection closed')
          reject(error)
        }
      }

      ws.onmessage = event => {
        try {
          const message = JSON.parse(String(event.data))

          if (message.type === 'unsupported') {
            fail('ASR provider changed during recording')
            reject(error)
          }

          if (message.type === 'ready') {
            for (const pcm of pending) {
              ws.send(pcm)
            }

            pending = []
            pendingBytes = 0
            readyToSend = true
            resolve()
          }

          if (message.type === 'partial') {
            onPartial(message.text)
          }

          if (message.type === 'final') {
            done = true
            resolveFinal(message.text)
          }

          if (message.type === 'error') {
            fail(message.error)
            reject(error)
          }
        } catch {
          fail('Invalid ASR response')
          reject(error)
        }
      }
    }),
    60_000,
    'ASR model loading timed out'
  )

  void ready.catch(reason => {
    if (!done) {
      fail(reason instanceof Error ? reason.message : String(reason))
      cancel()
    }
  })

  return {
    attach: async stream => {
      if (error) {
        throw error
      }

      context = new AudioContext({ sampleRate: 16000 })
      const moduleUrl = URL.createObjectURL(new Blob([processor], { type: 'text/javascript' }))

      try {
        await context.audioWorklet.addModule(moduleUrl)
      } finally {
        URL.revokeObjectURL(moduleUrl)
      }

      node = new AudioWorkletNode(context, 'asr-capture')

      node.port.onmessage = event => {
        if (event.data === 'flushed') {
          flushed?.()

          return
        }

        if (done) {
          return
        }

        if (!readyToSend) {
          pendingBytes += (event.data as ArrayBuffer).byteLength

          if (pendingBytes > 32000 * 15) {
            fail('ASR model did not load within the recording buffer limit')
            cancel()
          } else {
            pending.push(event.data as ArrayBuffer)
          }

          return
        }

        if (ws.readyState !== WebSocket.OPEN || ws.bufferedAmount > 32000 * 15) {
          fail('ASR cannot keep up with recording')
          cancel()

          return
        }

        ws.send(event.data)
      }

      source = context.createMediaStreamSource(stream)
      source.connect(node)
      // The worklet emits silence; connecting keeps it processing without mic feedback.
      node.connect(context.destination)
      await context.resume()
    },
    cancel,
    finish: async () => {
      try {
        if (error) {
          throw error
        }

        source?.disconnect()

        if (node) {
          await withTimeout(
            new Promise<void>(resolve => {
              flushed = resolve
              node!.port.postMessage('flush')
            }),
            3000,
            'ASR capture flush timed out'
          )
        }

        await context?.close()
        context = null

        await ready

        if (error) {
          throw error
        }

        ws.send('finish')

        return await withTimeout(final, 60_000, 'ASR final transcription timed out')
      } finally {
        cancel()
      }
    }
  }
}
