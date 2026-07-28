import { resolveGatewayWsUrl } from '@hermes/shared'

import { speakText } from '@/hermes'
import {
  $voicePlayback,
  setVoicePlaybackState,
  type VoicePlaybackSource,
  type VoicePlaybackState
} from '@/store/voice-playback'

import { sanitizeTextForSpeech } from './speech-text'

// Free Edge TTS occasionally hands back audio that never fires `playing`/`ended`
// nor `error` — leaving voice mode stuck "speaking" forever. Reject if playback
// fails to start or stalls mid-stream for this long (rearmed on each progress
// tick, so legitimately long speech is never cut off).
const PLAYBACK_STALL_MS = 15_000

let currentAudio: HTMLAudioElement | null = null
let currentStop: (() => void) | null = null
let sequence = 0

function currentState(
  status: VoicePlaybackState['status'],
  options?: VoicePlaybackOptions,
  audioElement: HTMLAudioElement | null = null
): VoicePlaybackState {
  return {
    audioElement,
    messageId: options?.messageId ?? null,
    sequence,
    source: options?.source ?? null,
    status
  }
}

export interface VoicePlaybackOptions {
  messageId?: string | null
  source: VoicePlaybackSource
}

export function stopVoicePlayback() {
  sequence += 1
  currentStop?.()
  currentStop = null

  if (currentAudio) {
    currentAudio.pause()
    currentAudio.src = ''
    currentAudio.load()
    currentAudio = null
  }

  setVoicePlaybackState({
    audioElement: null,
    messageId: null,
    sequence,
    source: null,
    status: 'idle'
  })
}

// ---------------------------------------------------------------------------
// Streaming path — /api/audio/speak-stream WebSocket, raw int16 PCM frames
// scheduled through Web Audio. Speech starts on the provider's first chunk
// instead of after full synthesis + base64 transfer.
// ---------------------------------------------------------------------------

async function resolveSpeakStreamUrl(): Promise<null | string> {
  const desktop = window.hermesDesktop

  if (!desktop?.getConnection) {
    return null
  }

  try {
    // Mint a fresh credential (single-use ticket in OAuth mode), then swap the
    // gateway endpoint for the PCM one — auth is shared across WS routes.
    const wsUrl = await resolveGatewayWsUrl(desktop, await desktop.getConnection())
    const url = new URL(wsUrl)

    if (!url.pathname.endsWith('/api/ws')) {
      return null
    }

    url.pathname = url.pathname.replace(/\/api\/ws$/, '/api/audio/speak-stream')

    return url.toString()
  } catch {
    return null
  }
}

export interface SpeechStreamSession {
  /** Feed more reply text as it streams in. Safe after `finish` (no-op). */
  append: (text: string) => void
  /** No more text coming — resolves `done` once the audio drains. */
  finish: () => void
  /**
   * 'done'    — audio fully played (or barged via stopVoicePlayback)
   * 'fallback'— no audio ever produced; caller should speak the accumulated
   *             text through `playSpeechText` instead.
   */
  done: Promise<'done' | 'fallback'>
}

/**
 * Open a live speech session: one WebSocket + one AudioContext for a whole
 * reply. Text is appended as LLM deltas arrive; the server cuts sentences and
 * streams PCM back while generation continues, so speech overlaps the text
 * stream (ChatGPT-style) with no per-sentence connection or synthesis gaps.
 */
function openSpeechStream(wsUrl: string, options: VoicePlaybackOptions): SpeechStreamSession {
  const ws = new WebSocket(wsUrl)
  ws.binaryType = 'arraybuffer'

  let context: AudioContext | null = null
  let streamRate = 24_000
  let nextStartAt = 0
  let carry: null | Uint8Array = null
  let started = false
  let settled = false
  let finished = false
  const pendingSends: string[] = []

  let settle: (value: 'done' | 'fallback') => void = () => undefined

  const done = new Promise<'done' | 'fallback'>(resolve => {
    settle = value => {
      if (settled) {
        return
      }

      settled = true
      currentStop = null

      try {
        ws.close()
      } catch {
        // already closed
      }

      void context?.close().catch(() => undefined)
      context = null
      resolve(value)
    }
  })

  const send = (frame: object) => {
    const data = JSON.stringify(frame)

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(data)
    } else if (ws.readyState === WebSocket.CONNECTING) {
      pendingSends.push(data)
    }
  }

  // stopVoicePlayback() → immediate barge-in: kill the socket (the server
  // aborts synthesis on disconnect) and the audio context (cuts sound now).
  currentStop = () => settle('done')

  const finishWhenDrained = () => {
    const remainingMs = context ? Math.max(0, nextStartAt - context.currentTime) * 1_000 : 0
    window.setTimeout(() => settle('done'), remainingMs + 100)
  }

  const schedule = (data: ArrayBuffer) => {
    if (!context) {
      return
    }

    // Provider chunks are not sample-aligned — carry any odd byte over.
    let bytes = new Uint8Array(data)

    if (carry) {
      const joined = new Uint8Array(carry.length + bytes.length)
      joined.set(carry)
      joined.set(bytes, carry.length)
      bytes = joined
      carry = null
    }

    const usable = bytes.length - (bytes.length % 2)

    if (bytes.length !== usable) {
      carry = bytes.slice(usable)
    }

    if (!usable) {
      return
    }

    const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, usable / 2)
    const buffer = context.createBuffer(1, pcm.length, streamRate)
    const channel = buffer.getChannelData(0)

    for (let index = 0; index < pcm.length; index += 1) {
      channel[index] = pcm[index] / 32_768
    }

    const source = context.createBufferSource()
    source.buffer = buffer
    source.connect(context.destination)

    const startAt = Math.max(context.currentTime + 0.05, nextStartAt)
    source.start(startAt)
    nextStartAt = startAt + buffer.duration

    if (!started) {
      started = true
      setVoicePlaybackState(currentState('speaking', options))
    }
  }

  ws.onopen = () => {
    pendingSends.splice(0).forEach(data => ws.send(data))
  }

  ws.onmessage = event => {
    if (typeof event.data !== 'string') {
      schedule(event.data as ArrayBuffer)

      return
    }

    let frame: { channels?: number; sample_rate?: number; type?: string }

    try {
      frame = JSON.parse(event.data) as typeof frame
    } catch {
      return
    }

    if (frame.type === 'start') {
      streamRate = frame.sample_rate || 24_000
      context = new AudioContext()
      nextStartAt = 0
    } else if (frame.type === 'end') {
      finishWhenDrained()
    } else if (frame.type === 'fallback') {
      settle(started ? 'done' : 'fallback')
    }
  }

  // A drop before any audio means the endpoint is unavailable (old backend,
  // auth, network) → fall back. After audio started, replaying the whole
  // message via POST would stutter — treat what played as the playback.
  ws.onerror = () => settle(started ? 'done' : 'fallback')
  ws.onclose = () => (started ? finishWhenDrained() : settle('fallback'))

  return {
    // Raw deltas — the server strips markdown/emoji per *sentence*, which is
    // the only safe granularity when constructs span delta boundaries.
    append: text => {
      if (text && !finished && !settled) {
        send({ text })
      }
    },
    finish: () => {
      if (!finished && !settled) {
        finished = true
        send({ done: true })
      }
    },
    done
  }
}

/**
 * Live-speak an in-progress reply: open a session, then `append` deltas and
 * `finish` when generation completes. Resolves null when streaming is
 * unavailable (old backend / non-chunked provider) — the caller falls back to
 * whole-text `playSpeechText`.
 */
export async function startSpeechStream(options: VoicePlaybackOptions): Promise<null | SpeechStreamSession> {
  const wsUrl = await resolveSpeakStreamUrl()

  if (!wsUrl) {
    return null
  }

  stopVoicePlayback()
  setVoicePlaybackState(currentState('preparing', options))

  const session = openSpeechStream(wsUrl, options)

  void session.done.then(outcome => {
    if (outcome === 'done') {
      setVoicePlaybackState(currentState('idle'))
    }
  })

  return session
}

/** One-shot playback of complete text over the streaming WS. */
function playSpeechStream(wsUrl: string, text: string, options: VoicePlaybackOptions): Promise<'fallback' | 'played'> {
  const session = openSpeechStream(wsUrl, options)
  session.append(text)
  session.finish()

  return session.done.then(outcome => (outcome === 'done' ? 'played' : 'fallback'))
}

async function playSpeechDataUrl(
  speakableText: string,
  options: VoicePlaybackOptions,
  isCurrent: () => boolean
): Promise<boolean> {
  const response = await speakText(speakableText)

  if (!isCurrent()) {
    return false
  }

  const audio = new Audio(response.data_url)
  currentAudio = audio
  setVoicePlaybackState(currentState('speaking', options, audio))

  await new Promise<void>((resolve, reject) => {
    let stall: number | null = null

    const cleanup = () => {
      if (stall !== null) {
        window.clearTimeout(stall)
        stall = null
      }

      audio.removeEventListener('ended', onEnded)
      audio.removeEventListener('error', onError)
      audio.removeEventListener('timeupdate', armStall)
      currentStop = null
    }

    const armStall = () => {
      if (stall !== null) {
        window.clearTimeout(stall)
      }

      stall = window.setTimeout(() => {
        cleanup()
        reject(new Error('Playback stalled'))
      }, PLAYBACK_STALL_MS)
    }

    const onEnded = () => {
      cleanup()
      resolve()
    }

    const onError = () => {
      cleanup()
      reject(new Error('Playback failed'))
    }

    currentStop = () => {
      cleanup()
      resolve()
    }

    audio.addEventListener('ended', onEnded, { once: true })
    audio.addEventListener('error', onError, { once: true })
    audio.addEventListener('timeupdate', armStall)
    armStall()
    void audio.play().catch(onError)
  })

  if (!isCurrent()) {
    return false
  }

  currentAudio = null

  return true
}

export async function playSpeechText(text: string, options: VoicePlaybackOptions): Promise<boolean> {
  stopVoicePlayback()

  const speakableText = sanitizeTextForSpeech(text)

  if (!speakableText) {
    return false
  }

  const ownSequence = sequence
  const isCurrent = () => ownSequence === sequence

  setVoicePlaybackState(currentState('preparing', options))

  try {
    // Streaming first; the POST data-URL path is the fallback for backends
    // without the WS endpoint or providers without a chunked API.
    const streamUrl = await resolveSpeakStreamUrl()

    if (streamUrl && isCurrent()) {
      const outcome = await playSpeechStream(streamUrl, speakableText, options)

      if (outcome === 'played') {
        if (!isCurrent()) {
          return false
        }

        setVoicePlaybackState(currentState('idle'))

        return true
      }
    }

    if (!isCurrent()) {
      return false
    }

    const played = await playSpeechDataUrl(speakableText, options, isCurrent)

    if (played) {
      setVoicePlaybackState(currentState('idle'))
    }

    return played
  } catch (error) {
    if (isCurrent()) {
      currentStop = null
      currentAudio = null
      setVoicePlaybackState(currentState('idle'))
    }

    throw error
  }
}

export function isVoicePlaybackActive() {
  return $voicePlayback.get().status !== 'idle'
}

// ---------------------------------------------------------------------------
// Interruption latch — the next prompt.submit carries `interrupted: true` so
// the model knows its spoken reply was cut off (it can react: "rude!").
// Marked by the barge-in paths (VAD, typing over playback); TTL'd so a stale
// barge never annotates an unrelated message minutes later.
// ---------------------------------------------------------------------------

const INTERRUPT_TTL_MS = 120_000
let interruptedAt: null | number = null

export function markVoicePlaybackInterrupted() {
  interruptedAt = Date.now()
}

export function takeVoicePlaybackInterrupted(): boolean {
  const at = interruptedAt
  interruptedAt = null

  return at !== null && Date.now() - at < INTERRUPT_TTL_MS
}
