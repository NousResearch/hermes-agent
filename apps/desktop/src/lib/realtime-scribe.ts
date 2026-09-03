import { CommitStrategy, RealtimeEvents, Scribe } from '@elevenlabs/client'

import { profileScoped } from '@/api/client'
import { hermesApi } from '@/hermes'

interface TokenResponse {
  ok: boolean
  language?: null | string
  model: string
  token: string
  websocket_url: string
}

export interface RealtimeScribeSession {
  close: () => void
  commit: () => void
  mute: () => void
  unmute: () => void
}

export interface RealtimeScribeOptions {
  onCommitted: (text: string) => void
  onError?: (error: Error) => void
  onPartial?: (text: string) => void
  silenceSeconds?: number
}

/**
 * Open one persistent ElevenLabs Scribe connection for a voice conversation.
 * Returns null when realtime isn't available so callers retain the existing
 * record/upload transcription path as a compatibility fallback.
 */
export async function startRealtimeScribe({
  onCommitted,
  onError,
  onPartial,
  silenceSeconds = 1
}: RealtimeScribeOptions): Promise<RealtimeScribeSession | null> {
  let token: TokenResponse

  try {
    token = await hermesApi<TokenResponse>({
      ...profileScoped(),
      method: 'POST',
      path: '/api/audio/scribe-token'
    })
  } catch {
    return null
  }

  if (!token?.ok || !token.token || !token.websocket_url) {
    return null
  }

  let connection: ReturnType<typeof Scribe.connect>

  try {
    connection = Scribe.connect({
      token: token.token,
      modelId: token.model || 'scribe_v2_realtime',
      baseUri: token.websocket_url,
      languageCode: token.language || undefined,
      commitStrategy: CommitStrategy.VAD,
      vadSilenceThresholdSecs: silenceSeconds,
      vadThreshold: 0.4,
      minSpeechDurationMs: 100,
      minSilenceDurationMs: 100,
      noVerbatim: true,
      microphone: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      }
    })
  } catch {
    return null
  }

  let openedSuccessfully = false
  let closing = false
  let terminalNotified = false

  const notifyTerminal = (error: Error) => {
    if (openedSuccessfully && !closing && !terminalNotified) {
      terminalNotified = true
      onError?.(error)
    }
  }

  connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, data => onPartial?.(data.text.trim()))
  connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, data => {
    const text = data.text.trim()

    if (text) {
      onCommitted(text)
    }
  })
  connection.on(RealtimeEvents.ERROR, data => {
    notifyTerminal(new Error(data.error || 'Realtime transcription failed'))
  })
  connection.on(RealtimeEvents.CLOSE, () => notifyTerminal(new Error('Realtime transcription closed')))

  // A single-use token is consumed while the WebSocket opens. Do not report a
  // ready session until that succeeds; callers may safely fall back on failure.
  const opened = new Promise<boolean>(resolve => {
    let settled = false

    const finish = (value: boolean) => {
      if (!settled) {
        settled = true
        resolve(value)
      }
    }

    connection.on(RealtimeEvents.OPEN, () => finish(true))
    connection.on(RealtimeEvents.CLOSE, () => finish(false))
    connection.on(RealtimeEvents.ERROR, () => finish(false))
    window.setTimeout(() => finish(false), 10_000)
  })

  if (!(await opened)) {
    connection.close()

    return null
  }

  openedSuccessfully = true

  return {
    close: () => {
      closing = true
      connection.close()
    },
    commit: () => connection.commit(),
    mute: () => {
      if (!connection.isMuted) {
        connection.mute()
      }
    },
    unmute: () => {
      if (connection.isMuted) {
        connection.unmute()
      }
    }
  }
}
