import { profileScoped } from '@/api/client'
import { hermesApi } from '@/hermes'
import { activeGateway } from '@/store/gateway'

import { parseVoiceChatMode, type VoiceLiveStatus } from './voice-live'

/**
 * Grok-Live voice chat: the xAI realtime full-duplex engine that DELEGATES to
 * Hermes, same shape as GPT-Live (`voice-live.ts`) but a different transport.
 *
 * Architectural difference (docs/grok-live-voice/SPEC.md §2): the backend
 * (gateway process) holds the xAI websocket EXCLUSIVELY — there is no
 * renderer-negotiated WebRTC leg like GPT-Live's. Audio relays as base64
 * PCM16 chunks inside the existing JSON-RPC channel (`voice.grok.*` methods
 * and events, SPEC §4.1) — the same shape `wake.feed` already uses for
 * client-side mic capture, NOT raw binary WS frames (the shared
 * json-rpc-channel has no binary protocol support).
 *
 * This module owns the transport only: session lifecycle over
 * `voice.grok.start` / `voice.grok.stop`, mic capture → base64 chunks →
 * `voice.grok.audio`, playback of the downstream `voice.grok.audio` events,
 * and the event stream (`voice.grok.transcript`, `voice.grok.state`,
 * `voice.grok.delegation`) the conversation hook drives. The gateway
 * connection this module gets IS the desktop's live JSON-RPC channel — no
 * separate socket.
 */

const CAPTURE_SAMPLE_RATE = 24_000
// ~100ms per chunk per SPEC §4.1 (2,400 frames @ 24kHz).
const CHUNK_FRAMES = 2_400
const CLOSE_TIMEOUT_MS = 15_000

export interface GrokTranscriptFragment {
  speaker: 'assistant' | 'user'
  text: string
}

export interface GrokVoiceHandlers {
  /** The backend judged a spoken exchange a real request — same shape as
   *  GPT-Live's `onDelegation`, but the context comes from the SERVER (the
   *  bridge owns the xAI ws and accumulates transcript fragments itself). */
  onDelegation: (delegationId: string, context: string) => void
  /** `voice.grok.state` event: connecting|listening|speaking|thinking|error|reconnecting|degraded|idle. */
  onState: (state: string, reason?: string) => void
  /** `voice.grok.transcript` event, for captions / live UI. */
  onTranscript?: (fragment: GrokTranscriptFragment) => void
  /** Session ended (explicit stop, or a fatal error the backend gave up retrying). */
  onClosed: (reason: string) => void
}

export async function fetchVoiceLiveGrokStatus(): Promise<null | VoiceLiveStatus> {
  try {
    const response = await hermesApi<{ ok: boolean } & VoiceLiveStatus>({
      ...profileScoped(),
      path: '/api/audio/voice-live-grok/status'
    })

    if (!response?.ok) {
      return null
    }

    return {
      available: Boolean(response.available),
      mode: parseVoiceChatMode(response.mode),
      model: response.model,
      reason: response.reason ?? null,
      voice: response.voice
    }
  } catch {
    // Older backend without the endpoint (predates grok-live) → chained.
    return null
  }
}

function bytesToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  let binary = ''
  const chunk = 0x8000

  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }

  return btoa(binary)
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }

  return bytes
}

function downsampleTo24k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === CAPTURE_SAMPLE_RATE) {
    return input
  }

  if (inputRate <= 0) {
    return new Float32Array(0)
  }

  const ratio = inputRate / CAPTURE_SAMPLE_RATE
  const outLen = Math.max(1, Math.floor(input.length / ratio))
  const out = new Float32Array(outLen)

  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio)
    const end = Math.min(input.length, Math.floor((i + 1) * ratio))
    let sum = 0
    let count = 0

    for (let j = start; j < end; j++) {
      sum += input[j] ?? 0
      count++
    }

    out[i] = count > 0 ? sum / count : 0
  }

  return out
}

function floatToInt16LE(input: Float32Array): ArrayBuffer {
  const buf = new ArrayBuffer(input.length * 2)
  const view = new DataView(buf)

  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i] ?? 0))
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }

  return buf
}

function int16LEToFloat(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const out = new Float32Array(bytes.length / 2)

  for (let i = 0; i < out.length; i++) {
    out[i] = view.getInt16(i * 2, true) / 0x8000
  }

  return out
}

export class GrokVoiceSession {
  sessionId: string
  private microphone: null | MediaStream = null
  private audioContext: null | AudioContext = null
  private processor: null | ScriptProcessorNode = null
  private playbackContext: null | AudioContext = null
  private playbackCursor = 0
  private seq = 0
  private finalized = false
  private muted = false
  private offEvent: null | (() => void) = null
  private closeTimer: null | number = null

  constructor(private readonly handlers: GrokVoiceHandlers) {
    this.sessionId = `grok-live-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  }

  /**
   * Adopt the real Hermes chat session id. The backend routes every
   * `voice.grok.*` event by `params.session_id` through the owning session's
   * transport (server.write_json), and the delegation seam looks the same id
   * up in `_sessions` — a synthetic id never reaches the app and never
   * becomes a Hermes turn. The composer knows the open chat's id; call this
   * before `start()` (gpt-live gets the same guarantee server-side because
   * its session is created by the backend with the real sid).
   */
  useSessionId(sessionId: string | null | undefined): void {
    if (sessionId && sessionId !== this.sessionId) {
      this.sessionId = sessionId
    }
  }

  async start(): Promise<void> {
    const gateway = activeGateway()

    if (!gateway) {
      throw new Error('gateway not connected')
    }

    this.playbackContext = new (window.AudioContext ||
      (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext)()
    this.playbackCursor = this.playbackContext.currentTime

    this.offEvent = gateway.onAny(event => {
      if (event.payload && (event.payload as { session_id?: string }).session_id !== this.sessionId) {
        return
      }

      switch (event.type) {
        case 'voice.grok.state': {
          const payload = event.payload as { state?: string; reason?: string } | undefined

          this.handlers.onState(payload?.state ?? 'idle', payload?.reason)

          return
        }

        case 'voice.grok.transcript': {
          const payload = event.payload as { speaker?: string; text?: string } | undefined

          if (payload?.text) {
            this.handlers.onTranscript?.({
              speaker: payload.speaker === 'assistant' ? 'assistant' : 'user',
              text: payload.text
            })
          }

          return
        }

        case 'voice.grok.delegation': {
          const payload = event.payload as { delegation_id?: string; context?: string } | undefined

          if (payload?.delegation_id) {
            this.handlers.onDelegation(payload.delegation_id, payload.context ?? '')
          }

          return
        }

        case 'voice.grok.audio': {
          const payload = event.payload as { pcm_b64?: string } | undefined

          if (payload?.pcm_b64) {
            this.playChunk(payload.pcm_b64)
          }

          return
        }

        default:
          return
      }
    })

    await gateway.request('voice.grok.start', { session_id: this.sessionId })

    this.microphone = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, echoCancellation: true, noiseSuppression: true }
    })

    const audioWindow = window as Window & { webkitAudioContext?: typeof AudioContext }
    const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

    if (!AudioContextCtor) {
      throw new Error('AudioContext unavailable for Grok-Live mic capture')
    }

    const context = new AudioContextCtor()
    this.audioContext = context
    const source = context.createMediaStreamSource(this.microphone)
    const processor = context.createScriptProcessor(4096, 1, 1)
    this.processor = processor
    const mute = context.createGain()
    mute.gain.value = 0

    let pending = new Float32Array(0)

    processor.onaudioprocess = event => {
      if (this.finalized || this.muted) {
        return
      }

      const input = event.inputBuffer.getChannelData(0)
      const at24k = downsampleTo24k(input, context.sampleRate)
      const merged = new Float32Array(pending.length + at24k.length)
      merged.set(pending, 0)
      merged.set(at24k, pending.length)
      let offset = 0

      while (offset + CHUNK_FRAMES <= merged.length) {
        const frame = merged.subarray(offset, offset + CHUNK_FRAMES)
        offset += CHUNK_FRAMES
        this.sendChunk(frame)
      }

      pending = merged.subarray(offset)
    }

    source.connect(processor)
    processor.connect(mute)
    mute.connect(context.destination)

    if (context.state === 'suspended') {
      await context.resume().catch(() => undefined)
    }
  }

  private sendChunk(frame: Float32Array): void {
    const gateway = activeGateway()

    if (!gateway) {
      return
    }

    this.seq += 1
    const pcm = floatToInt16LE(frame)
    void gateway
      .request('voice.grok.audio', {
        pcm_b64: bytesToBase64(pcm),
        seq: this.seq,
        session_id: this.sessionId
      })
      .catch(() => undefined)
  }

  private playChunk(pcmB64: string): void {
    const context = this.playbackContext

    if (!context) {
      return
    }

    const samples = int16LEToFloat(base64ToBytes(pcmB64))
    const buffer = context.createBuffer(1, samples.length, CAPTURE_SAMPLE_RATE)
    buffer.getChannelData(0).set(samples)
    const source = context.createBufferSource()
    source.buffer = buffer
    source.connect(context.destination)

    const startAt = Math.max(this.playbackCursor, context.currentTime)
    source.start(startAt)
    this.playbackCursor = startAt + buffer.duration
  }

  setMuted(muted: boolean): void {
    this.muted = muted
    const gateway = activeGateway()

    void gateway?.request('voice.grok.mute', { muted, session_id: this.sessionId }).catch(() => undefined)
  }

  /** Graceful close: ask the backend to end the xAI session, tear down local
   *  capture/playback after it (or a timeout). */
  close(): void {
    if (this.finalized) {
      return
    }

    const gateway = activeGateway()
    const finish = () => this.finish('close_requested')

    if (!gateway) {
      finish()

      return
    }

    void gateway
      .request('voice.grok.stop', { session_id: this.sessionId })
      .catch(() => undefined)
      .finally(finish)

    this.closeTimer = window.setTimeout(finish, CLOSE_TIMEOUT_MS)
  }

  private finish(reason: string): void {
    if (this.finalized) {
      return
    }

    this.finalized = true

    if (this.closeTimer) {
      window.clearTimeout(this.closeTimer)
      this.closeTimer = null
    }

    this.offEvent?.()
    this.offEvent = null
    this.processor?.disconnect()
    this.processor = null
    void this.audioContext?.close().catch(() => undefined)
    this.audioContext = null
    void this.playbackContext?.close().catch(() => undefined)
    this.playbackContext = null
    this.microphone?.getTracks().forEach(track => track.stop())
    this.microphone = null
    this.handlers.onClosed(reason)
  }
}
