import { resolveGatewayWsUrl } from '@hermes/shared'

import { getApiRequestConnection, getApiRequestProfile } from '@/hermes'

/**
 * Live dictation STT: stream mic PCM to the gateway's
 * /api/audio/transcribe-stream WebSocket WHILE the user is speaking.
 *
 * The gateway feeds the audio to the active streaming-capable STT provider
 * (plugin-registered, e.g. soniox-stt) and returns the transcript right when
 * the recording ends — no "record the whole utterance, then transcribe the
 * file" round-trip. Direction of travel:
 *
 *   mic → PCM chunks (this module) → gateway → provider (live) → text → composer
 *
 * Protocol (server contract in hermes_cli/web_server.py):
 *   client → {"sample_rate": 16000}   then binary s16le mono PCM frames,
 *            {"eos": true}            recording finished
 *   server → {"type":"partial","text":...}, {"type":"final","transcript":...},
 *            {"type":"error","message":...}
 *
 * Any failure (older backend, no streaming provider, socket drop) surfaces as
 * a rejected stop()/null handle — callers fall back to the recorded-blob
 * relay, so dictation keeps working exactly as before.
 */

export async function resolveTranscribeStreamUrl(): Promise<null | string> {
  const desktop = window.hermesDesktop

  if (!desktop?.getConnection) {
    return null
  }

  try {
    // Same (connection, profile) routing as resolveSpeakStreamUrl: mint a
    // fresh credential for the ACTIVE backend and swap the gateway endpoint
    // for the STT one — auth is shared across WS routes.
    const profile = getApiRequestProfile()
    const connectionId = getApiRequestConnection()

    const conn =
      connectionId && desktop.getConnectionFor
        ? await desktop.getConnectionFor({ connectionId, profile })
        : await desktop.getConnection(profile)

    const wsDeps =
      connectionId && desktop.getGatewayWsUrlFor
        ? { getGatewayWsUrl: () => desktop.getGatewayWsUrlFor!({ connectionId, profile }) }
        : connectionId
          ? {}
          : desktop

    const wsUrl = await resolveGatewayWsUrl(wsDeps, conn)
    const url = new URL(wsUrl)

    if (!url.pathname.endsWith('/api/ws')) {
      return null
    }

    url.pathname = url.pathname.replace(/\/api\/ws$/, '/api/audio/transcribe-stream')

    if (profile && !url.searchParams.has('profile')) {
      url.searchParams.set('profile', profile)
    }

    return url.toString()
  } catch {
    return null
  }
}

export interface DictationStreamSession {
  /** Feed one PCM chunk (16 kHz mono s16le). No-op once finished/closed. */
  pushAudio: (chunk: ArrayBuffer) => void
  /** End the recording and resolve with the final transcript (null = silence). */
  stop: () => Promise<null | string>
  /** Abort without a transcript (barge-in / cancel). */
  cancel: () => void
  /** Latest non-final text the provider has recognized so far. */
  lastPartial: () => string
}

export function openDictationStream(options?: {
  onPartial?: (text: string) => void
}): Promise<DictationStreamSession | null> {
  const onPartial = options?.onPartial

  return (async () => {
    const url = await resolveTranscribeStreamUrl()

    if (!url) {
      return null
    }

    let ws: WebSocket
    let opened = false
    let settled = false
    let partial = ''
    let pending: { resolve: (t: null | string) => void; reject: (e: Error) => void } | null = null
    let errorMessage: null | string = null

    const settle = (resolve: (t: null | string) => void, reject: (e: Error) => void) => {
      pending = { resolve, reject }
    }

    ws = new WebSocket(url)

    ws.onmessage = event => {
      let message: { type?: string; transcript?: string; text?: string; message?: string } = {}

      try {
        message = JSON.parse(String(event.data)) as typeof message
      } catch {
        return
      }

      if (message.type === 'partial' && typeof message.text === 'string') {
        partial = message.text
        onPartial?.(partial)
      } else if (message.type === 'final') {
        settled = true
        pending?.resolve((message.transcript ?? '').trim() || null)
        pending = null
        ws.close()
      } else if (message.type === 'error') {
        errorMessage = message.message ?? 'STT streaming error'
        settled = true
        pending?.reject(new Error(errorMessage))
        pending = null
        ws.close()
      }
    }

    ws.onclose = () => {
      if (!settled) {
        settled = true

        if (errorMessage) {
          console.warn(`[voice-stream] closed without final: ${errorMessage}`)
          pending?.reject(new Error(errorMessage))
        } else {
          pending?.resolve(null)
        }

        pending = null
      }
    }

    ws.onerror = () => {
      if (!settled && !errorMessage) {
        errorMessage = 'STT streaming connection failed'
      }
    }

    try {
      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => {
          opened = true
          resolve()
        }

        ws.onerror = () => {
          if (!opened) {
            reject(new Error(errorMessage ?? 'STT streaming connection failed'))
          }
        }
      })
    } catch (error) {
      ws.close()

      return null
    }

    ws.send(JSON.stringify({ sample_rate: 16000 }))
    console.info('[voice-stream] transcribe-stream connected (16 kHz s16le)')

    return {
      pushAudio: (chunk: ArrayBuffer) => {
        if (ws.readyState === WebSocket.OPEN && !settled) {
          try {
            ws.send(chunk)
          } catch {
            // Socket died mid-recording — stop() will surface the failure and
            // the caller falls back to the recorded blob.
          }
        }
      },
      stop: () =>
        new Promise<null | string>((resolve, reject) => {
          if (settled) {
            resolve(null)

            return
          }

          settle(resolve, reject)

          try {
            ws.send(JSON.stringify({ eos: true }))
          } catch (error) {
            reject(error instanceof Error ? error : new Error('STT streaming send failed'))
          }
        }),
      cancel: () => {
        if (!settled) {
          settled = true
          pending?.resolve(null)
          pending = null
        }

        ws.close()
      },
      lastPartial: () => partial
    }
  })()
}
