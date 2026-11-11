import { type OwnerScope, ownerScoped } from '@/api/client'
import { hermesApi } from '@/hermes'

/**
 * Grok Voice transport. The gateway mints a 300s client secret
 * (`POST /api/audio/grok-voice/session`); this module opens
 * `wss://api.x.ai/v1/realtime` with it and never sees the long-lived bearer.
 *
 * Real work is a Hermes turn: `hermes_task` becomes the composer's voice-live
 * submit, and the final text returns as a `[Hermes result]` user item.
 * Protocol shape follows the hermes-xai voice plugin (MIT), not a vendored copy.
 */

export const SAMPLE_RATE = 24_000
export const FRAME_SAMPLES = SAMPLE_RATE / 10
export const HERMES_RESULT_PREFIX = '[Hermes result] '
export const HERMES_RESULT_CAP = 3_000
export const CONTEXT_TURNS = 12
const HANG_UP_MS = 1_500

const WORKLET = `
class Pcm24k extends AudioWorkletProcessor {
  process (inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (ch && ch.length) this.port.postMessage(ch.slice(0))
    return true
  }
}
registerProcessor('hermes-grok-pcm', Pcm24k)
`

export interface DialogState {
  hangUp: boolean
  owedTurn: boolean
  playing: boolean
  responding: boolean
  results: string[]
  spoke: boolean
  turns: string[]
}

export interface GrokEvent {
  arguments?: string
  call_id?: string
  delta?: string
  error?: { message?: string }
  name?: string
  transcript?: string
  type: string
}

export interface DialogDecision {
  assistantTranscript?: string
  delegate?: { context: string; request: string }
  error?: string
  flushPlayback?: boolean
  hangUp?: boolean
  play?: string
  send: object[]
  state: DialogState
  userTranscript?: string
}

export function freshDialog(): DialogState {
  return {
    hangUp: false,
    owedTurn: false,
    playing: false,
    responding: false,
    results: [],
    spoke: false,
    turns: []
  }
}

/** Joined Hermes text, capped, then prefixed once. Empty becomes a spoken fallback. */
export function hermesResultText(parts: string[]): string {
  const joined = parts
    .map(part => part.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .join('\n\n')

  const body = (joined || 'Hermes finished without a spoken result.').slice(0, HERMES_RESULT_CAP)

  return `${HERMES_RESULT_PREFIX}${body}`
}

function remember(state: DialogState, role: 'assistant' | 'user', text: string): DialogState {
  const turns = [...state.turns, `${role}: ${text.trim()}`]

  return { ...state, turns: turns.length > CONTEXT_TURNS ? turns.slice(-CONTEXT_TURNS) : turns }
}

function resultItem(text: string): object {
  return {
    item: { content: [{ text, type: 'input_text' }], role: 'user', type: 'message' },
    type: 'conversation.item.create'
  }
}

function toolOutput(callId: string, output: object): object {
  return {
    item: { call_id: callId, output: JSON.stringify(output), type: 'function_call_output' },
    type: 'conversation.item.create'
  }
}

/** Speak queued results only when the voice is quiet. */
function flushed(state: DialogState): DialogDecision {
  if (!state.results.length || state.responding || state.playing) {
    return { send: [], state }
  }

  return {
    send: [resultItem(hermesResultText(state.results)), { type: 'response.create' }],
    state: { ...state, responding: true, results: [] }
  }
}

export function enqueueResult(state: DialogState, text: string): DialogDecision {
  return flushed({ ...state, results: [...state.results, text] })
}

function onResponseCreated(state: DialogState): DialogDecision {
  return { send: [], state: { ...state, responding: true, spoke: false } }
}

function onAudio(state: DialogState, event: GrokEvent): DialogDecision {
  const delta = typeof event.delta === 'string' ? event.delta : ''

  if (!delta) {
    return { send: [], state }
  }

  return { play: delta, send: [], state: { ...state, playing: true, spoke: true } }
}

function onSpeechStarted(state: DialogState): DialogDecision {
  return { flushPlayback: true, send: [], state: { ...state, playing: false } }
}

function onPlaybackIdle(state: DialogState): DialogDecision {
  return flushed({ ...state, playing: false })
}

function onTranscript(state: DialogState, event: GrokEvent, role: 'assistant' | 'user'): DialogDecision {
  const text = String(event.transcript || '').trim()

  if (!text) {
    return { send: [], state }
  }

  const next = remember(state, role, text)

  return role === 'user'
    ? { send: [], state: next, userTranscript: text }
    : { assistantTranscript: text, send: [], state: next }
}

function onTool(state: DialogState, event: GrokEvent): DialogDecision {
  const name = String(event.name || '')
  const callId = String(event.call_id || '')
  let args: { request?: string } = {}

  try {
    args = JSON.parse(String(event.arguments || '{}')) as { request?: string }
  } catch {
    args = {}
  }

  if (name === 'hermes_task') {
    const request = String(args.request || '').trim()

    return {
      delegate: request ? { context: state.turns.join('\n'), request } : undefined,
      send: [
        toolOutput(callId, {
          note: 'The result will arrive in a "[Hermes result]" message.',
          receipt: 'started'
        })
      ],
      state: { ...state, owedTurn: true }
    }
  }

  if (name === 'end_call') {
    return { send: [toolOutput(callId, { ok: true })], state: { ...state, hangUp: true } }
  }

  return { send: [toolOutput(callId, { error: 'unknown tool', ok: false })], state }
}

function onResponseDone(state: DialogState): DialogDecision {
  if (state.hangUp) {
    return { hangUp: true, send: [], state: { ...state, responding: false } }
  }

  let next: DialogState = { ...state, responding: false }
  const send: object[] = []

  if (next.owedTurn && !next.spoke && next.results.length === 0) {
    send.push({ type: 'response.create' })
    next = { ...next, responding: true }
  }

  next = { ...next, owedTurn: false }
  const spoken = flushed(next)

  return { send: [...send, ...spoken.send], state: spoken.state }
}

function onError(state: DialogState, event: GrokEvent): DialogDecision {
  return { error: event.error?.message || 'Grok Voice error', send: [], state }
}

const HANDLERS: Record<string, (state: DialogState, event: GrokEvent) => DialogDecision> = {
  'conversation.item.input_audio_transcription.completed': (state, event) => onTranscript(state, event, 'user'),
  error: onError,
  'input_audio_buffer.speech_started': state => onSpeechStarted(state),
  'playback-idle': state => onPlaybackIdle(state),
  'response.audio.delta': onAudio,
  'response.created': state => onResponseCreated(state),
  'response.done': state => onResponseDone(state),
  'response.function_call_arguments.done': onTool,
  'response.output_audio.delta': onAudio,
  'response.output_audio_transcript.done': (state, event) => onTranscript(state, event, 'assistant')
}

export function reduceDialog(state: DialogState, event: GrokEvent): DialogDecision {
  const handler = HANDLERS[event.type]

  return handler ? handler(state, event) : { send: [], state }
}

export function floatToBase64(float32: Float32Array): string {
  const pcm = new Int16Array(float32.length)

  for (let i = 0; i < float32.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32[i]))
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
  }

  const bytes = new Uint8Array(pcm.buffer)
  let binary = ''

  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }

  return btoa(binary)
}

export function base64ToPcm(text: string): Int16Array {
  const binary = atob(text)
  const bytes = new Uint8Array(binary.length)

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }

  return new Int16Array(bytes.buffer, 0, bytes.length >> 1)
}

interface MintedSession {
  client_secret?: { expires_at?: null | number; value?: string }
  model?: string
  protocol_prefix?: string
  session?: object
  url?: string
}

export interface GrokVoiceHandlers {
  onAssistantTranscript?: (text: string) => void
  onClosed: (reason: string) => void
  onDelegate: (request: string, context: string) => void
  onError: (message: string, fatal: boolean) => void
  onLevel?: (level: number) => void
  onSpeakingChange?: (speaking: boolean) => void
  onUserTranscript: (text: string) => void
}

interface Player {
  close: () => void
  flush: () => void
  play: (pcm: Int16Array) => void
}

function createPlayer(onIdle: () => void): Player {
  const context = new AudioContext({ sampleRate: SAMPLE_RATE })
  const live = new Set<AudioBufferSourceNode>()
  let next = 0
  let suppressIdle = false

  return {
    close() {
      this.flush()
      void context.close().catch(() => undefined)
    },
    flush() {
      // Barge-in stops buffers without treating that as "the voice finished",
      // so a queued Hermes result stays queued until the model turn ends.
      suppressIdle = true

      for (const node of live) {
        node.onended = null

        try {
          node.stop()
        } catch {
          // Already ended.
        }
      }

      live.clear()
      next = 0
      suppressIdle = false
    },
    play(pcm) {
      const buffer = context.createBuffer(1, pcm.length, SAMPLE_RATE)
      const channel = buffer.getChannelData(0)

      for (let i = 0; i < pcm.length; i++) {
        channel[i] = pcm[i] / 0x8000
      }

      const node = context.createBufferSource()
      node.buffer = buffer
      node.connect(context.destination)
      next = Math.max(next, context.currentTime)
      node.start(next)
      next += buffer.duration
      live.add(node)

      node.onended = () => {
        live.delete(node)

        if (!suppressIdle && live.size === 0) {
          onIdle()
        }
      }
    }
  }
}

function frameLevel(frame: Float32Array): number {
  let sum = 0

  for (let i = 0; i < frame.length; i++) {
    sum += frame[i] * frame[i]
  }

  return Math.min(1, Math.sqrt(sum / frame.length) * 4)
}

export class GrokVoiceSession {
  private dialog = freshDialog()
  private finalized = false
  private hangUpTimer: null | number = null
  private muted = false
  private player: null | Player = null
  private started = false
  private stopMic: (() => void) | null = null
  private ws: null | WebSocket = null

  constructor(
    private readonly handlers: GrokVoiceHandlers,
    private readonly owner: null | OwnerScope
  ) {}

  async start(): Promise<void> {
    if (this.started) {
      throw new Error('Grok Voice session already started')
    }

    this.started = true

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new DOMException('Microphone capture is unavailable here', 'NotSupportedError')
    }

    const minted = await hermesApi<MintedSession>({
      ...ownerScoped(this.owner ?? undefined),
      method: 'POST',
      path: '/api/audio/grok-voice/session',
      timeoutMs: 30_000
    })

    const secret = minted?.client_secret?.value
    const url = minted?.url

    if (!secret || !url || !minted?.session) {
      throw new Error('Grok Voice session creation failed')
    }

    this.player = createPlayer(() => this.notePlaybackIdle())
    await this.openSocket(url, `${minted.protocol_prefix ?? 'xai-client-secret.'}${secret}`, minted.session)
  }

  deliverResult(text: string): void {
    if (this.finalized) {
      return
    }

    this.apply(enqueueResult(this.dialog, text))
  }

  /** Ask the voice to answer the audio it already has. */
  nudge(): void {
    this.send({ type: 'response.create' })
  }

  setMuted(muted: boolean): void {
    this.muted = muted
  }

  close(reason = 'close_requested'): void {
    if (this.finalized) {
      return
    }

    this.finalized = true

    if (this.hangUpTimer != null) {
      window.clearTimeout(this.hangUpTimer)
      this.hangUpTimer = null
    }

    this.stopMic?.()
    this.stopMic = null
    this.player?.close()
    this.player = null

    const socket = this.ws
    this.ws = null
    socket?.close()
    this.handlers.onClosed(reason)
  }

  private async openSocket(url: string, protocol: string, session: object): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      let settled = false
      const ws = new WebSocket(url, [protocol])
      this.ws = ws

      const fail = (error: unknown) => {
        if (settled) {
          return
        }

        settled = true
        reject(error instanceof Error ? error : new Error('Grok Voice connection failed'))
      }

      ws.onopen = () => {
        this.send({ session, type: 'session.update' })
        void this.openMic()
          .then(() => {
            if (!settled) {
              settled = true
              resolve()
            }
          })
          .catch(fail)
      }

      ws.onmessage = ({ data }) => this.receive(String(data))
      ws.onerror = () => fail(new Error('Grok Voice connection failed'))

      ws.onclose = () => {
        if (!settled) {
          this.finalized = true
          this.stopMic?.()
          this.stopMic = null
          this.ws = null
          fail(new Error('Grok Voice connection failed'))

          return
        }

        if (!this.finalized) {
          this.close('connection_lost')
        }
      }
    })
  }

  private async openMic(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, echoCancellation: true, noiseSuppression: true }
    })

    if (this.finalized) {
      for (const track of stream.getTracks()) {
        track.stop()
      }

      throw new Error('Grok Voice connection failed')
    }

    const context = new AudioContext({ sampleRate: SAMPLE_RATE })
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' }))

    try {
      await context.audioWorklet.addModule(url)
    } finally {
      URL.revokeObjectURL(url)
    }

    const node = new AudioWorkletNode(context, 'hermes-grok-pcm')
    const pending: Float32Array[] = []
    let size = 0

    node.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
      pending.push(data)
      size += data.length

      if (size < FRAME_SAMPLES) {
        return
      }

      const frame = new Float32Array(size)
      let at = 0

      for (const part of pending) {
        frame.set(part, at)
        at += part.length
      }

      pending.length = 0
      size = 0
      this.handlers.onLevel?.(this.muted ? 0 : frameLevel(frame))

      if (!this.muted) {
        this.send({ audio: floatToBase64(frame), type: 'input_audio_buffer.append' })
      }
    }

    context.createMediaStreamSource(stream).connect(node)

    this.stopMic = () => {
      for (const track of stream.getTracks()) {
        track.stop()
      }

      void context.close().catch(() => undefined)
    }

    if (this.finalized) {
      this.stopMic()
      this.stopMic = null
    }
  }

  private receive(data: string): void {
    let event: GrokEvent

    try {
      event = JSON.parse(data) as GrokEvent
    } catch {
      return
    }

    this.apply(reduceDialog(this.dialog, event))
  }

  private apply(decision: DialogDecision): void {
    this.dialog = decision.state

    for (const message of decision.send) {
      this.send(message)
    }

    if (decision.flushPlayback) {
      this.player?.flush()
      this.handlers.onSpeakingChange?.(false)
    }

    if (decision.play) {
      this.player?.play(base64ToPcm(decision.play))
      this.handlers.onSpeakingChange?.(true)
    }

    if (decision.delegate) {
      this.handlers.onDelegate(decision.delegate.request, decision.delegate.context)
    }

    if (decision.userTranscript) {
      this.handlers.onUserTranscript(decision.userTranscript)
    }

    if (decision.assistantTranscript) {
      this.handlers.onAssistantTranscript?.(decision.assistantTranscript)
    }

    if (decision.error) {
      this.handlers.onError(decision.error, false)
    }

    if (decision.hangUp) {
      this.scheduleHangUp()
    }
  }

  private notePlaybackIdle(): void {
    if (this.finalized) {
      return
    }

    this.apply(reduceDialog(this.dialog, { type: 'playback-idle' }))

    if (!this.dialog.playing) {
      this.handlers.onSpeakingChange?.(false)
    }
  }

  private scheduleHangUp(): void {
    if (this.hangUpTimer != null || this.finalized) {
      return
    }

    this.hangUpTimer = window.setTimeout(() => {
      this.hangUpTimer = null
      this.close('end_call')
    }, HANG_UP_MS)
  }

  private send(event: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(event))
    }
  }
}
