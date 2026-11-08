import { openGatewaySocket } from '@/api/canonical-protocol'
import { type OwnerScope, resolveOwnerNow } from '@/api/client'
import { resolveSiblingWsUrl } from '@/lib/sibling-ws-url'
import {
  chunkForCommentary,
  type LiveHistoryMessage,
  type LiveSession,
  type LiveTranscriptFragment,
  transcriptContextWindow,
  type VoiceLiveHandlers
} from '@/lib/voice-live'

/**
 * GPT-Live on an Azure OpenAI resource: the same full-duplex voice frontend that
 * DELEGATES to Hermes, but over the gateway's Azure Realtime relay WebSocket
 * instead of WebRTC. Azure Realtime has no SDP exchange, so there is no
 * RTCPeerConnection and no `/api/audio/voice-live/session` POST here — the
 * renderer dials the ticket-gated sibling socket `/api/audio/voice-live-stream`
 * (same authenticated origin the route's `/api/ws` uses, via
 * `resolveSiblingWsUrl`/`openGatewaySocket`, so the owner (connection, profile)
 * routing matches chat) and speaks the relay protocol:
 *
 *   client → {"type":"session.start","history":[...]}   once the socket is open
 *   server → {"type":"session.started","session":{"id":...}}  (synthetic, after the
 *            backend's Azure session.updated), then standard Azure Realtime events
 *   client → {"type":"input_audio_buffer.append","audio":<base64 PCM16 mono 24kHz>}
 *   client → session.commentary.append / session.commentary.complete {delegation_id}
 *            (existing commentary messages; complete seals the accumulated text and
 *            the backend answers with ONE function_call_output + response.create)
 *   client → session.thinking.append / session.instructions.append (existing messages)
 *   client → {"type":"session.close"}
 *
 *   server → response.function_call_arguments.done {call_id, name:'delegate_to_hermes'}
 *            maps to the existing onDelegation(call_id, contextWindow())
 *   server → conversation.item.input_audio_transcription.completed (user transcript)
 *            and response.audio_transcript.delta (assistant transcript)
 *   server → response.audio.delta <base64 PCM16 mono 24kHz> → played via Web Audio
 *
 * Provider credentials never reach the renderer: the backend holds the Azure
 * resource key and relays the vendor session.
 */

/** The ticket-gated relay socket, beside `/api/ws` (socket name `voice-live-stream`). */
export const AZURE_LIVE_SOCKET_PATH = '/api/audio/voice-live-stream'

/** Azure Realtime PCM is pinned server-side; the client must match it. */
export const AZURE_PCM_SAMPLE_RATE = 24_000

const OPEN_TIMEOUT_MS = 15_000
const START_TIMEOUT_MS = 30_000
const CLOSE_TIMEOUT_MS = 15_000
// Vendor cap on commentary appends is ~500 tokens; keep the same headroom as the
// OpenAI transport (~4 chars/token).
const APPEND_CHAR_LIMIT = 1_400
const CAPTURE_BUFFER_SIZE = 4_096

interface AzureServerEvent {
  type: string
  delta?: string
  transcript?: string
  call_id?: string
  name?: string
  session?: { id?: string }
  item?: { transcript?: string }
  error?: { type?: string; code?: null | string; message?: string }
  usage?: { seconds?: number }
  reason?: string
  message?: string
}

interface ResampleState {
  /** Fractional position, within the current input buffer, of the next output sample. */
  phase: number
}

/** Linear-interpolation resampler to PCM16 mono at `outRate`; the AudioContext's
 *  own rate is the input rate (the vendor pins 24 kHz, the device usually runs
 *  at 48 kHz). */
function resampleToPcm16(input: Float32Array, inRate: number, outRate: number, state: ResampleState): Int16Array {
  const ratio = inRate / outRate
  const count = Math.max(0, Math.floor((input.length - 1 - state.phase) / ratio) + 1)
  const out = new Int16Array(count)

  for (let n = 0; n < count; n += 1) {
    const pos = state.phase + n * ratio
    const index = Math.floor(pos)
    const frac = pos - index
    const before = input[index]
    const after = index + 1 < input.length ? input[index + 1] : before
    const sample = before + (after - before) * frac

    out[n] = sample <= -1 ? -32_768 : sample >= 1 ? 32_767 : Math.round(sample * 32_768)
  }

  if (input.length > 0) {
    state.phase = count > 0 ? state.phase + count * ratio - input.length : state.phase

    if (state.phase < 0) {
      state.phase = 0
    }
  }

  return out
}

function base64FromBytes(bytes: Uint8Array): string {
  let binary = ''

  const CHUNK = 0x8000

  for (let index = 0; index < bytes.length; index += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(index, index + CHUNK))
  }

  return btoa(binary)
}

function audioContextConstructor(): typeof AudioContext {
  const ctor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext

  if (!ctor) {
    throw new DOMException('Web Audio is unavailable here', 'NotSupportedError')
  }

  return ctor
}

export class VoiceLiveAzureSession implements LiveSession {
  /** Whose (connection, profile) this session dials; null → the active scope.
   *  Same owner rule as the OpenAI transport: a Bot chat's voice session runs
   *  on the Bot's own (connection, profile). */
  private readonly owner: null | OwnerScope
  private readonly handlers: VoiceLiveHandlers
  private ws: null | WebSocket = null
  private microphone: null | MediaStream = null
  private captureContext: null | AudioContext = null
  private captureSource: null | MediaStreamAudioSourceNode = null
  private processor: null | ScriptProcessorNode = null
  private playbackContext: null | AudioContext = null
  private playbackSources = new Set<AudioBufferSourceNode>()
  private nextPlayTime = 0
  private resampleState: ResampleState = { phase: 0 }
  private closeTimer: null | number = null
  private startTimer: null | number = null
  private startSettle: null | { reject: (error: Error) => void; resolve: () => void } = null
  private finalized = false
  private started = false
  private starting = false
  private closeRequested = false
  private muted = false
  private eventCounter = 0
  private transcript: LiveTranscriptFragment[] = []
  private lastSpeaking = false
  sessionId: null | string = null
  /** The delegation currently being answered by Hermes (the function call_id). */
  activeDelegationId: null | string = null
  private readonly eventHandlers: Record<string, (event: AzureServerEvent) => void> = {
    'conversation.item.input_audio_transcription.completed': event => {
      this.pushTranscript('user', event.item?.transcript ?? event.transcript ?? '')
    },
    error: event => this.handleError(event),
    'response.audio.delta': event => {
      if (event.delta) {
        this.playAudioDelta(event.delta)
      }
    },
    'response.audio_transcript.delta': event => this.pushTranscript('assistant', event.delta ?? ''),
    'response.function_call_arguments.done': event => this.handleDelegation(event),
    'session.closed': event => {
      this.finish(this.closeRequested ? 'close_requested' : event.reason ?? 'closed', event.usage?.seconds ?? null)
    },
    'session.started': event => this.handleStarted(event)
  }

  constructor(handlers: VoiceLiveHandlers, owner: null | OwnerScope = null) {
    this.handlers = handlers
    this.owner = owner
  }

  get connected(): boolean {
    return this.started && this.ws?.readyState === WebSocket.OPEN
  }

  private nextEventId(prefix: string): string {
    this.eventCounter += 1

    return `${prefix}_${this.eventCounter}`
  }

  private send(event: Record<string, unknown>): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false
    }

    this.ws.send(JSON.stringify(event))

    return true
  }

  /** Recent conversation, oldest first, bounded exactly like the OpenAI transport. */
  contextWindow(): LiveTranscriptFragment[] {
    return transcriptContextWindow(this.transcript)
  }

  async start(history: LiveHistoryMessage[]): Promise<void> {
    if (this.ws) {
      throw new Error('Azure live session already started')
    }

    // Same guard as the WebRTC transport: an insecure origin has no mediaDevices,
    // and the recorder's copy maps this to "microphone unsupported".
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new DOMException('Microphone capture is unavailable here', 'NotSupportedError')
    }

    this.starting = true

    // The relay socket rides the owner's (connection, profile) route exactly the
    // way chat's `/api/ws` does — registry `*For` bridges for a named connection,
    // the plain pair otherwise; a browser host mints a `voice-live-stream` ticket.
    // A null owner means the active scope, resolved once, now.
    const resolved = resolveOwnerNow(this.owner ?? undefined)
    const route = { connectionId: resolved.connectionId, profile: resolved.profile }
    const url = new URL(await resolveSiblingWsUrl(route, AZURE_LIVE_SOCKET_PATH))

    // A registry-minted URL may already carry the backend-namespace profile; keep it.
    if (route.profile && !url.searchParams.has('profile')) {
      url.searchParams.set('profile', route.profile)
    }

    const ws = openGatewaySocket(url.toString())
    this.ws = ws
    ws.onmessage = event => this.handleEvent(String(event.data))

    ws.onclose = () => {
      if (!this.finalized) {
        // A drop during start() is reported by start()'s own rejection instead.
        this.finish(this.closeRequested ? 'close_requested' : 'connection_lost', null, !this.starting)
      }
    }

    const opened = await this.waitForOpen(ws)

    if (!opened) {
      this.finalized = true
      this.starting = false
      ws.close()
      throw new Error('Could not open the Azure live stream connection')
    }

    try {
      this.send({ type: 'session.start', history })
      await this.waitForStarted()
      this.microphone = await navigator.mediaDevices.getUserMedia({
        audio: { autoGainControl: true, echoCancellation: true, noiseSuppression: true }
      })
      this.startCapture()
    } catch (error) {
      // The hook's start() catch reports this once; keep the close silent here.
      this.finish(this.closeRequested ? 'close_requested' : 'connection_lost', null, false)
      throw error
    } finally {
      this.starting = false
    }
  }

  private waitForOpen(ws: WebSocket): Promise<boolean> {
    return new Promise(resolve => {
      const timer = window.setTimeout(() => {
        detach()
        resolve(false)
      }, OPEN_TIMEOUT_MS)

      const detach = () => {
        window.clearTimeout(timer)
        ws.removeEventListener('open', onOpen)
        ws.removeEventListener('error', onFail)
        ws.removeEventListener('close', onFail)
      }

      const onOpen = () => {
        detach()
        resolve(true)
      }

      const onFail = () => {
        detach()
        resolve(false)
      }

      ws.addEventListener('open', onOpen)
      ws.addEventListener('error', onFail)
      ws.addEventListener('close', onFail)
    })
  }

  private waitForStarted(): Promise<void> {
    if (this.started) {
      return Promise.resolve()
    }

    return new Promise<void>((resolve, reject) => {
      this.startSettle = { reject, resolve }
      this.startTimer = window.setTimeout(() => {
        this.startSettle = null
        reject(new Error('Azure live session did not start'))
      }, START_TIMEOUT_MS)
    })
  }

  /** Mic → PCM16 mono 24 kHz → base64 `input_audio_buffer.append`. */
  private startCapture(): void {
    const context = new (audioContextConstructor())()
    this.captureContext = context
    const source = context.createMediaStreamSource(this.microphone as MediaStream)
    const processor = context.createScriptProcessor(CAPTURE_BUFFER_SIZE, 1, 1)

    processor.onaudioprocess = event => {
      if (this.finalized || this.muted) {
        return
      }

      const pcm = resampleToPcm16(event.inputBuffer.getChannelData(0), context.sampleRate, AZURE_PCM_SAMPLE_RATE, this.resampleState)

      if (pcm.length > 0) {
        this.send({ audio: base64FromBytes(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength)), type: 'input_audio_buffer.append' })
      }
    }

    source.connect(processor)
    processor.connect(context.destination)
    this.captureSource = source
    this.processor = processor
  }

  /** `response.audio.delta` → scheduled Web Audio playback at 24 kHz. */
  private playAudioDelta(base64: string): void {
    try {
      const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0))
      const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2))

      if (pcm.length === 0) {
        return
      }

      if (!this.playbackContext) {
        this.playbackContext = new (audioContextConstructor())()
        this.nextPlayTime = 0
      }

      const context = this.playbackContext
      const buffer = context.createBuffer(1, pcm.length, AZURE_PCM_SAMPLE_RATE)
      const channel = buffer.getChannelData(0)

      for (let index = 0; index < pcm.length; index += 1) {
        channel[index] = pcm[index] / 32_768
      }

      const source = context.createBufferSource()
      source.buffer = buffer
      source.connect(context.destination)

      const now = context.currentTime

      if (this.nextPlayTime < now) {
        this.nextPlayTime = now
      }

      source.start(this.nextPlayTime)
      this.nextPlayTime += buffer.duration
      this.playbackSources.add(source)

      source.onended = () => {
        this.playbackSources.delete(source)

        if (this.playbackSources.size === 0) {
          this.setSpeaking(false)
        }
      }

      this.setSpeaking(true)
    } catch {
      // Playback is best-effort: a decode/schedule failure must not kill the session.
    }
  }

  private setSpeaking(speaking: boolean): void {
    if (speaking !== this.lastSpeaking) {
      this.lastSpeaking = speaking
      this.handlers.onSpeakingChange?.(speaking)
    }
  }

  private pushTranscript(speaker: 'assistant' | 'user', text: string): void {
    if (!text) {
      return
    }

    const now = Date.now()
    this.transcript.push({ endMs: now, speaker, startMs: now, text })

    if (this.transcript.length > 2_000) {
      this.transcript.splice(0, this.transcript.length - 1_500)
    }

    this.handlers.onTranscript?.({ endMs: now, speaker, startMs: now, text })
  }

  private handleEvent(raw: string): void {
    let event: AzureServerEvent

    try {
      event = JSON.parse(raw) as AzureServerEvent
    } catch {
      return
    }

    this.eventHandlers[event.type]?.(event)
  }

  private handleStarted(event: AzureServerEvent): void {
    this.started = true
    this.sessionId = event.session?.id ?? this.sessionId

    if (this.startTimer) {
      window.clearTimeout(this.startTimer)
      this.startTimer = null
    }

    this.startSettle?.resolve()
    this.startSettle = null
  }

  private handleDelegation(event: AzureServerEvent): void {
    // The backend's delegate_to_hermes function call: a Hermes turn is
    // requested with the call_id as the delegation id.
    if (event.name === 'delegate_to_hermes' && event.call_id) {
      this.activeDelegationId = event.call_id
      this.handlers.onDelegation(event.call_id, this.contextWindow())
    }
  }

  private handleError(event: AzureServerEvent): void {
    const message = event.error?.message ?? event.message ?? 'Azure live error'

    if (!this.started) {
      this.startSettle?.reject(new Error(message))
      this.startSettle = null
    }

    this.handlers.onError(message, false)
  }

  /** Quiet progress for the live model ("Hermes is running the tests…"). The
   *  relay keeps the existing message; the backend drives it from its fallback. */
  think(delegationId: null | string, content: string): void {
    const text = content.replace(/\s+/g, ' ').trim().slice(0, APPEND_CHAR_LIMIT)

    if (text) {
      this.send({
        content: text,
        delegation_id: delegationId,
        event_id: this.nextEventId('think'),
        type: 'session.thinking.append'
      })
    }
  }

  /** A result the voice should say aloud (paraphrased); the relay accumulates
   *  these per delegation_id until `completeCommentary` seals them. */
  speak(delegationId: null | string, content: string): void {
    for (const chunk of chunkForCommentary(content)) {
      this.send({
        content: chunk,
        delegation_id: delegationId,
        event_id: this.nextEventId('say'),
        type: 'session.commentary.append'
      })
    }
  }

  /** Seal the accumulated commentary: the backend answers the delegation with
   *  ONE function_call_output(call_id) + response.create. */
  completeCommentary(delegationId: string): void {
    this.send({
      delegation_id: delegationId,
      event_id: this.nextEventId('complete'),
      type: 'session.commentary.complete'
    })
  }

  /** Steer the live persona mid-conversation (session-wide); the relay may
   *  transform it backend-side. */
  instruct(content: string): void {
    const text = content.trim().slice(0, APPEND_CHAR_LIMIT)

    if (text) {
      this.send({
        content: text,
        delegation_id: null,
        event_id: this.nextEventId('instr'),
        type: 'session.instructions.append'
      })
    }
  }

  setMuted(muted: boolean): void {
    this.muted = muted

    for (const track of this.microphone?.getAudioTracks() ?? []) {
      track.enabled = !muted
    }
  }

  /** Graceful close: ask for `session.close`, tear down on the reply (or timeout). */
  close(): void {
    if (this.finalized) {
      return
    }

    this.closeRequested = true

    if (!this.send({ type: 'session.close' })) {
      this.finish('close_requested', null)

      return
    }

    this.closeTimer = window.setTimeout(() => this.finish('close_requested', null), CLOSE_TIMEOUT_MS)
  }

  private finish(reason: string, usageSeconds: null | number, notify = true): void {
    if (this.finalized) {
      return
    }

    this.finalized = true

    if (this.closeTimer) {
      window.clearTimeout(this.closeTimer)
      this.closeTimer = null
    }

    if (this.startTimer) {
      window.clearTimeout(this.startTimer)
      this.startTimer = null
    }

    if (this.startSettle) {
      this.startSettle.reject(new Error(reason === 'close_requested' ? 'Azure live session closed during start' : 'Azure live stream connection lost'))
      this.startSettle = null
    }

    if (this.processor) {
      this.processor.onaudioprocess = null
      this.processor.disconnect()
      this.processor = null
    }

    this.captureSource?.disconnect()
    this.captureSource = null
    void this.captureContext?.close().catch(() => undefined)
    this.captureContext = null

    for (const source of this.playbackSources) {
      source.onended = null
      source.stop()
      source.disconnect()
    }

    this.playbackSources.clear()
    void this.playbackContext?.close().catch(() => undefined)
    this.playbackContext = null
    this.setSpeaking(false)

    this.microphone?.getTracks().forEach(track => track.stop())
    this.microphone = null
    this.ws?.close()
    this.ws = null

    if (notify) {
      this.handlers.onClosed(reason, usageSeconds)
    }
  }
}
