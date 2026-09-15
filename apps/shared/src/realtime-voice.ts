/**
 * Framework-agnostic xAI realtime voice client for browser surfaces.
 *
 * Desktop and dashboard renderers hold the S2S WebSocket directly: the
 * backend mints an ephemeral token + the ready-made session.update payload
 * (RPC `voice.realtime_token`), and the browser supplies echo-cancelled mic
 * capture — which is why these surfaces get full duplex where the CLI needs
 * half-duplex gating.
 *
 * The supervisor tool contract (consult_hermes / steer_hermes, instant
 * force_message acknowledgments, deferred response.create after tool output)
 * mirrors the Python controller in agent/voice_supervisor.py — the surface
 * only implements TurnRunner-equivalent callbacks.
 */

export const CONSULT_TOOL_NAME = 'consult_hermes'
export const STEER_TOOL_NAME = 'steer_hermes'

export const REALTIME_INPUT_SAMPLE_RATE = 16000
export const REALTIME_OUTPUT_SAMPLE_RATE = 24000

/** Spoken instantly (force_message — no model turn) when the model calls a
 *  tool without its mandated filler. Mirrors _ACK_PHRASES in
 *  tools/voice_realtime_config.py. */
export const ACK_PHRASES: readonly string[] = [
  'On it — give me a moment.',
  'Sure, let me check that.',
  'Okay, working on it.',
  'Alright, one moment.',
  'Got it — digging in now.',
  'Let me have Hermes look at that.',
  'On it. This might take a bit.',
  'Sure thing — checking now.',
  'Okay, let me find out.',
  'Alright, Hermes is on it.'
]

/** How long sendFunctionOutput waits for current speech to finish before
 *  requesting the follow-up response (xAI no-overlap best practice). */
const QUIET_WAIT_TIMEOUT_MS = 20_000
const QUIET_POLL_MS = 200
/** Mic frames ~100 ms apiece at 16 kHz (xAI best practice). */
const MIC_FRAME_SAMPLES = 1600
/** WebSocket OPEN readyState without touching the global (absent in jsdom). */
const WS_OPEN = 1

/** Linear resample — browsers often ignore AudioContext({ sampleRate }). */
export function resampleFloat32(
  input: Float32Array,
  fromRate: number,
  toRate: number
): Float32Array {
  if (fromRate === toRate || input.length === 0) {
    return input
  }

  const ratio = fromRate / toRate
  const outLength = Math.max(1, Math.floor(input.length / ratio))
  const out = new Float32Array(outLength)

  for (let i = 0; i < outLength; i++) {
    const src = i * ratio
    const i0 = Math.floor(src)
    const i1 = Math.min(i0 + 1, input.length - 1)
    const frac = src - i0
    out[i] = input[i0] * (1 - frac) + input[i1] * frac
  }

  return out
}

/** Drop OpenAI-compat extras so a rejected session.update can be retried. */
export function minimalSessionUpdate(payload: Record<string, unknown>): Record<string, unknown> {
  const rawSession = payload.session

  const session: Record<string, unknown> =
    rawSession && typeof rawSession === 'object' ? { ...(rawSession as Record<string, unknown>) } : {}

  delete session.reasoning
  const turnDetection = session.turn_detection

  if (turnDetection && typeof turnDetection === 'object') {
    const next = { ...(turnDetection as Record<string, unknown>) }
    delete next.create_response
    session.turn_detection = next
  }

  return { ...payload, type: 'session.update', session }
}

export type RealtimeVoiceStatus =
  | 'connecting'
  | 'listening'
  | 'speaking'
  | 'closed'
  | 'error'

export interface RealtimeTokenGrant {
  /** Ephemeral token from `voice.realtime_token` (never a raw API key). */
  token: string
  /** Full wss URL including `?model=`. */
  url: string
  /** session.update payload built server-side — sent verbatim. */
  session_update: Record<string, unknown>
}

export interface RealtimeFunctionCall {
  name: string
  callId: string
  args: Record<string, unknown>
}

export interface RealtimeVoiceCallbacks {
  /** Supervisor tool call (consult/steer) — the surface runs the turn. */
  onFunctionCall: (call: RealtimeFunctionCall) => void | Promise<void>
  onStatus?: (status: RealtimeVoiceStatus, detail?: string) => void
  /** Sidecar ASR of the user — kept solely for spoken stop-word detection
   *  (approximate text; never displayed or fed to the model). */
  onUserTranscript?: (text: string) => void
  /** Server VAD heard the user start talking (assistant audio was cut). */
  onUserSpeechStarted?: () => void
  /** Mic input level 0..1 at frame cadence (UI meters). */
  onLevel?: (level: number) => void
}

interface WebSocketFactory {
  (url: string, protocols: string[]): WebSocket
}

export interface RealtimeVoiceClientOptions {
  /** Test seam — defaults to `new WebSocket(url, protocols)`. */
  createSocket?: WebSocketFactory
  /** Test seam — skip mic/speaker plumbing entirely. */
  disableAudio?: boolean
}

interface ScheduledPlayback {
  context: AudioContext
  nextStartAt: number
  sources: Set<AudioBufferSourceNode>
  carry: Uint8Array | null
}

export class RealtimeVoiceClient {
  private socket: WebSocket | null = null
  private callbacks: RealtimeVoiceCallbacks | null = null
  private readonly createSocket: WebSocketFactory
  private readonly disableAudio: boolean

  private micStream: MediaStream | null = null
  private micContext: AudioContext | null = null
  private micNode: ScriptProcessorNode | null = null
  private pendingMicFrames: string[] = []

  private playback: ScheduledPlayback | null = null
  private activeResponse = false
  private responseHadAudio = false
  private closed = false
  private muted = false
  private sessionUpdate: Record<string, unknown> | null = null
  private minimalRetryDone = false
  private responseCreatePending = false

  constructor(options: RealtimeVoiceClientOptions = {}) {
    this.createSocket =
      options.createSocket ?? ((url, protocols) => new WebSocket(url, protocols))
    this.disableAudio = options.disableAudio ?? false
  }

  get speaking(): boolean {
    return this.playback !== null && this.playback.sources.size > 0
  }

  get alive(): boolean {
    return !this.closed
  }

  async connect(
    grant: RealtimeTokenGrant,
    callbacks: RealtimeVoiceCallbacks,
    audio?: { deviceId?: null | string }
  ): Promise<void> {
    if (this.socket) {throw new Error('realtime voice client already connected')}
    this.callbacks = callbacks
    this.closed = false
    this.minimalRetryDone = false
    this.sessionUpdate = grant.session_update
    callbacks.onStatus?.('connecting')

    // Open mic first (permission prompt) and buffer early frames — audio
    // starts flowing the instant the socket opens (xAI best practice).
    if (!this.disableAudio) {await this.openMic(audio?.deviceId)}

    // close() during the permission prompt found nothing to tear down and
    // every later close() is a no-op — without this check the mic stays
    // hot and a socket is dialed that can never be closed.
    if (this.closed) {
      this.teardownAudio()

      return
    }

    // Browsers cannot set headers on WebSocket upgrade; the ephemeral token
    // travels in the subprotocol (never reused across dials).
    const socket = this.createSocket(grant.url, [`xai-client-secret.${grant.token}`])
    this.socket = socket

    // Attach onmessage before open — the server can emit events during
    // handshake (session.updated, early response.created).
    socket.onmessage = ev => {
      if (typeof ev.data !== 'string') {return}
      let event: Record<string, unknown>

      try {
        event = JSON.parse(ev.data) as Record<string, unknown>
      } catch {
        return
      }

      this.handleServerEvent(event)
    }

    try {
      await new Promise<void>((resolve, reject) => {
        const fail = (message: string) => {
          callbacks.onStatus?.('error', message)
          reject(new Error(message))
        }

        socket.onopen = () => {
          this.send(grant.session_update)

          for (const frame of this.pendingMicFrames) {
            this.send({ type: 'input_audio_buffer.append', audio: frame })
          }

          this.pendingMicFrames = []
          callbacks.onStatus?.('listening')
          resolve()
        }

        socket.onerror = () => fail('realtime socket error')

        socket.onclose = ev => {
          if (this.closed) {return}
          fail(ev.reason || 'realtime socket closed during connect')
        }
      })
    } catch (error) {
      this.closed = true
      this.teardownAudio()

      try {
        socket.close()
      } catch {
        // already closing
      }

      this.socket = null
      throw error
    }

    socket.onerror = () => {
      if (this.closed) {return}
      callbacks.onStatus?.('error', 'realtime socket error')
    }

    socket.onclose = ev => {
      if (this.closed) {return}
      this.closed = true
      this.teardownAudio()
      this.socket = null
      callbacks.onStatus?.('closed', ev.reason || undefined)
    }
  }

  close(): void {
    if (this.closed) {return}
    this.closed = true
    this.teardownAudio()

    try {
      this.socket?.close()
    } catch {
      // already closing
    }

    this.socket = null
    this.callbacks?.onStatus?.('closed')
  }

  /** Return a tool result; the follow-up response is requested only once
   *  current speech finished playing (mirrors the Python controller). Only
   *  one deferred response.create is ever in flight — outputs landing while
   *  one is pending (steer result + consult result) are covered by it, so
   *  two poll loops can never race concurrent response.create sends. */
  sendFunctionOutput(callId: string, output: string): void {
    this.send({
      type: 'conversation.item.create',
      item: { type: 'function_call_output', call_id: callId, output }
    })

    if (this.responseCreatePending) {return}
    this.responseCreatePending = true
    const startedAt = Date.now()

    const tick = () => {
      if (this.closed) {
        this.responseCreatePending = false

        return
      }

      const quiet = !this.speaking && !this.activeResponse

      if (quiet || Date.now() - startedAt > QUIET_WAIT_TIMEOUT_MS) {
        this.responseCreatePending = false
        this.send({ type: 'response.create' })

        return
      }

      setTimeout(tick, QUIET_POLL_MS)
    }

    tick()
  }

  /** Speak exact text with no model turn (xAI force_message). */
  speakVerbatim(text: string, interruptible = true): void {
    const trimmed = text.trim()

    if (!trimmed) {return}
    this.send({
      type: 'conversation.item.create',
      item: {
        type: 'force_message',
        role: 'assistant',
        interruptible,
        content: [{ type: 'output_text', text: trimmed }]
      }
    })
  }

  /** Instant rotating "on it" when the model called a tool silently. */
  speakAcknowledgment(): void {
    const phrase = ACK_PHRASES[Math.floor(Math.random() * ACK_PHRASES.length)]
    this.speakVerbatim(phrase, true)
  }

  get lastResponseHadAudio(): boolean {
    return this.responseHadAudio
  }

  /** Muted drops mic frames client-side; the session stays connected. */
  setMuted(muted: boolean): void {
    this.muted = muted
  }

  get isMuted(): boolean {
    return this.muted
  }

  /** OS label of the capture track, if the mic is open. */
  get captureLabel(): string {
    return this.micStream?.getAudioTracks()[0]?.label ?? ''
  }

  // -- server events ---------------------------------------------------------

  handleServerEvent(event: Record<string, unknown>): void {
    const type = String(event.type ?? '')

    switch (type) {
      case 'response.created':
        this.activeResponse = true
        this.responseHadAudio = false

        break

      case 'response.done':

      case 'response.completed':

      case 'response.cancelled':
        this.activeResponse = false

        if (!this.speaking) {this.callbacks?.onStatus?.('listening')}

        break

      case 'response.output_audio.delta':
      case 'response.audio.delta': {
        this.responseHadAudio = true
        const b64 = String(event.delta ?? event.audio ?? '')

        if (b64) {this.schedulePlayback(b64)}

        break
      }

      case 'conversation.item.input_audio_transcription.completed': {
        const transcript = String(event.transcript ?? '').trim()

        if (transcript) {this.callbacks?.onUserTranscript?.(transcript)}

        break
      }

      case 'input_audio_buffer.speech_started':
        // Server interrupts its own response in VAD mode; drop the locally
        // queued remainder to match. Browser AEC keeps this echo-safe.
        this.clearPlayback()
        this.callbacks?.onUserSpeechStarted?.()

        break
      case 'response.function_call_arguments.done': {
        const name = String(event.name ?? '')
        const callId = String(event.call_id ?? '')

        if (!name || !callId) {break}
        let args: Record<string, unknown> = {}

        try {
          const raw = event.arguments
          args = typeof raw === 'string' ? JSON.parse(raw) : ((raw as Record<string, unknown>) ?? {})
        } catch {
          args = {}
        }

        void this.callbacks?.onFunctionCall({ name, callId, args })

        break
      }

      case 'error': {
        const detail = event.error ?? event.message ?? 'unknown realtime error'
        const message = typeof detail === 'string' ? detail : JSON.stringify(detail)

        if (!this.minimalRetryDone && this.sessionUpdate) {
          // The retry may be masking an unrelated error (rate limit, bad
          // tool output) — keep the detail visible instead of eating it.
          console.warn('realtime voice server error (retrying with minimal session config):', message)
          this.minimalRetryDone = true
          this.send(minimalSessionUpdate(this.sessionUpdate))

          break
        }

        this.callbacks?.onStatus?.('error', message)

        break
      }

      default:
        break
    }
  }

  // -- outgoing --------------------------------------------------------------

  private send(payload: Record<string, unknown>): void {
    const socket = this.socket

    if (!socket || socket.readyState !== WS_OPEN) {return}

    try {
      socket.send(JSON.stringify(payload))
    } catch {
      // socket died between readyState check and send — close path handles it
    }
  }

  // -- mic capture -------------------------------------------------------------

  private async openMic(deviceId?: null | string): Promise<void> {
    const audio: MediaTrackConstraints = { echoCancellation: true, noiseSuppression: true }

    if (deviceId && deviceId !== 'default') {
      audio.deviceId = { exact: deviceId }
    }

    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio })
    } catch {
      if (!audio.deviceId) {
        throw new Error('microphone unavailable')
      }

      delete audio.deviceId
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio })
    }

    this.micContext = new AudioContext({ sampleRate: REALTIME_INPUT_SAMPLE_RATE })
    const captureRate = this.micContext.sampleRate
    const source = this.micContext.createMediaStreamSource(this.micStream)
    const node = this.micContext.createScriptProcessor(4096, 1, 1)
    this.micNode = node
    let pcmCarry = new Float32Array(0)

    node.onaudioprocess = e => {
      const raw = e.inputBuffer.getChannelData(0)
      const input = resampleFloat32(raw, captureRate, REALTIME_INPUT_SAMPLE_RATE)
      const merged = new Float32Array(pcmCarry.length + input.length)
      merged.set(pcmCarry)
      merged.set(input, pcmCarry.length)
      let offset = 0

      while (merged.length - offset >= MIC_FRAME_SAMPLES) {
        const frame = merged.subarray(offset, offset + MIC_FRAME_SAMPLES)
        offset += MIC_FRAME_SAMPLES
        this.emitMicFrame(frame)
      }

      pcmCarry = merged.slice(offset)
    }

    source.connect(node)
    // Keep the processor alive without audible mic monitor.
    const silent = this.micContext.createGain()
    silent.gain.value = 0
    node.connect(silent)
    silent.connect(this.micContext.destination)
  }

  private emitMicFrame(frame: Float32Array): void {
    if (this.muted) {
      this.callbacks?.onLevel?.(0)

      return
    }

    let sumSquares = 0
    const pcm = new Int16Array(frame.length)

    for (let i = 0; i < frame.length; i++) {
      const s = Math.max(-1, Math.min(1, frame[i]))
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff
      sumSquares += s * s
    }

    this.callbacks?.onLevel?.(Math.sqrt(sumSquares / frame.length))
    const bytes = new Uint8Array(pcm.buffer)
    let binary = ''

    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
    }

    const b64 = btoa(binary)

    if (!this.socket || this.socket.readyState !== WS_OPEN) {
      // Buffer pre-connect audio; bounded so a dead socket can't grow it.
      if (this.pendingMicFrames.length < 50) {this.pendingMicFrames.push(b64)}

      return
    }

    this.send({ type: 'input_audio_buffer.append', audio: b64 })
  }

  // -- playback ---------------------------------------------------------------

  private schedulePlayback(b64: string): void {
    if (this.disableAudio) {return}

    if (!this.playback) {
      this.playback = {
        context: new AudioContext({ sampleRate: REALTIME_OUTPUT_SAMPLE_RATE }),
        nextStartAt: 0,
        sources: new Set(),
        carry: null
      }
    }

    const pb = this.playback
    const binary = atob(b64)
    let bytes = new Uint8Array(binary.length)

    for (let i = 0; i < binary.length; i++) {bytes[i] = binary.charCodeAt(i)}

    if (pb.carry) {
      const merged = new Uint8Array(pb.carry.length + bytes.length)
      merged.set(pb.carry)
      merged.set(bytes, pb.carry.length)
      bytes = merged
      pb.carry = null
    }

    if (bytes.length % 2 === 1) {
      // int16 alignment: hold the odd byte for the next delta.
      pb.carry = bytes.slice(bytes.length - 1)
      bytes = bytes.subarray(0, bytes.length - 1)
    }

    if (bytes.length === 0) {return}
    const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2)
    const float24k = new Float32Array(pcm.length)

    for (let i = 0; i < pcm.length; i++) {float24k[i] = pcm[i] / 32768}

    // Provider audio is 24 kHz; resample when the context ignored that rate.
    const playRate = pb.context.sampleRate
    const samples = resampleFloat32(float24k, REALTIME_OUTPUT_SAMPLE_RATE, playRate)
    const buffer = pb.context.createBuffer(1, samples.length, playRate)
    buffer.getChannelData(0).set(samples)
    const source = pb.context.createBufferSource()
    source.buffer = buffer
    source.connect(pb.context.destination)
    const startAt = Math.max(pb.context.currentTime + 0.02, pb.nextStartAt)
    pb.nextStartAt = startAt + buffer.duration
    pb.sources.add(source)

    source.onended = () => {
      pb.sources.delete(source)

      if (pb.sources.size === 0 && !this.activeResponse) {
        this.callbacks?.onStatus?.('listening')
      }
    }

    source.start(startAt)
    this.callbacks?.onStatus?.('speaking')
  }

  private clearPlayback(): void {
    const pb = this.playback

    if (!pb) {return}

    for (const source of pb.sources) {
      try {
        source.stop()
      } catch {
        // already stopped
      }
    }

    pb.sources.clear()
    pb.nextStartAt = 0
    pb.carry = null
  }

  private teardownAudio(): void {
    this.clearPlayback()

    if (this.playback) {
      void this.playback.context.close().catch(() => undefined)
      this.playback = null
    }

    if (this.micNode) {
      try {
        this.micNode.disconnect()
      } catch {
        // context already closed
      }

      this.micNode = null
    }

    if (this.micContext) {
      void this.micContext.close().catch(() => undefined)
      this.micContext = null
    }

    if (this.micStream) {
      for (const track of this.micStream.getTracks()) {track.stop()}
      this.micStream = null
    }

    this.pendingMicFrames = []
  }
}
