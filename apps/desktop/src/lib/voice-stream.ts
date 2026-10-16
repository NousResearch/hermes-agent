import type { ResolvedOwner } from '@/hermes'
import { resolveSiblingWsUrl } from '@/lib/sibling-ws-url'

/**
 * Live dictation STT: mic PCM streams to the host's `/api/audio/transcribe-stream` WebSocket
 * WHILE the user speaks; partial text arrives as it is recognized and the transcript lands right
 * at end of recording (server contract: `hermes_cli/web_routers/audio.py`). Provider keys never
 * leave the host, and the host resamples, so the client sends the AudioContext's own rate.
 *
 *   client → {"sample_rate": N}, then binary s16le mono PCM frames, {"eos": true}
 *   server → {"type":"partial","text"}, {"type":"final","transcript"}, {"type":"error","message"}
 *
 * Any failure (older backend, provider without a live wire, socket drop) resolves to `null` or
 * rejects stop(); the caller then transcribes its recorded blob exactly as before.
 */

export interface DictationStreamSession {
  /** Feed one PCM chunk (s16le mono at the announced rate). No-op once settled. */
  pushAudio: (chunk: ArrayBuffer) => void
  /** End the recording; resolves with the final transcript ('' = heard nothing). */
  stop: () => Promise<string>
  /** Abort without a transcript (cancel / unmount). */
  cancel: () => void
}

const OPEN_TIMEOUT_MS = 5_000

export async function openDictationStream(
  owner: ResolvedOwner,
  sampleRate: number,
  onPartial?: (text: string) => void
): Promise<DictationStreamSession | null> {
  let url: URL

  try {
    url = new URL(await resolveSiblingWsUrl(owner, '/api/audio/transcribe-stream'))
  } catch {
    return null
  }

  // A registry-minted URL may already carry the backend-namespace profile; keep it.
  if (owner.profile && !url.searchParams.has('profile')) {
    url.searchParams.set('profile', owner.profile)
  }

  const ws = new WebSocket(url.toString())
  ws.binaryType = 'arraybuffer'

  let settled = false
  let failure: Error | null = null
  let pending: { reject: (error: Error) => void; resolve: (text: string) => void } | null = null

  const settle = (outcome: Error | string) => {
    if (settled) {
      return
    }

    settled = true

    if (typeof outcome !== 'string') {
      failure = outcome
      pending?.reject(outcome)
    } else {
      pending?.resolve(outcome)
    }

    pending = null
  }

  ws.onmessage = event => {
    let message: { message?: string; text?: string; transcript?: string; type?: string }

    try {
      message = JSON.parse(String(event.data)) as typeof message
    } catch {
      return
    }

    if (message.type === 'partial' && typeof message.text === 'string') {
      onPartial?.(message.text)
    } else if (message.type === 'final') {
      settle((message.transcript ?? '').trim())
      ws.close()
    } else if (message.type === 'error') {
      settle(new Error(message.message || 'live transcription failed'))
      ws.close()
    }
  }

  const opened = await new Promise<boolean>(resolve => {
    const timer = window.setTimeout(() => resolve(false), OPEN_TIMEOUT_MS)

    ws.onopen = () => {
      window.clearTimeout(timer)
      resolve(true)
    }

    ws.onerror = () => {
      window.clearTimeout(timer)
      resolve(false)
    }
  })

  ws.onclose = () => settle(new Error('live transcription connection closed'))

  if (!opened || settled) {
    ws.close()

    return null
  }

  ws.send(JSON.stringify({ sample_rate: sampleRate }))

  return {
    pushAudio: chunk => {
      if (!settled && ws.readyState === WebSocket.OPEN) {
        ws.send(chunk)
      }
    },
    stop: () =>
      new Promise<string>((resolve, reject) => {
        if (settled) {
          reject(failure ?? new Error('live transcription already ended'))

          return
        }

        pending = { reject, resolve }
        ws.send(JSON.stringify({ eos: true }))
      }),
    cancel: () => {
      settled = true
      pending = null
      ws.close()
    }
  }
}
