import { profileScoped } from '@/api/client'
import { hermesApi } from '@/hermes'

/**
 * GPT-Live voice chat: the full-duplex voice frontend that DELEGATES to Hermes.
 *
 * `voice.voice_chat_mode: gpt-live` swaps the chained mic → STT → turn → TTS
 * loop for one OpenAI voice model (`gpt-live-1`) that listens and speaks at
 * the same time over WebRTC and has no tools of its own. Whenever the user
 * asks for real work it emits `session.delegation.created`; the desktop turns
 * that into an ordinary Hermes turn on the open session and streams the reply
 * back with `session.commentary.append`, which the voice paraphrases aloud.
 * Hermes keeps every capability — model choice, tools, memory, approvals.
 *
 * This module owns the transport only: session creation via the gateway
 * (the OpenAI key never reaches the renderer), the RTCPeerConnection, the
 * `oai-events` data channel, transcript accumulation and the command
 * surface the conversation hook drives. Vendor contract:
 * https://developers.openai.com/api/docs/guides/live-delegation
 */

export type VoiceChatMode = 'chained' | 'gpt-live'

export interface VoiceLiveStatus {
  mode: VoiceChatMode
  available: boolean
  reason: null | string
  model: string
  voice: string
}

export interface LiveHistoryMessage {
  type: 'message'
  role: 'assistant' | 'developer' | 'user'
  content: Array<{ type: 'input_text' | 'output_text'; text: string }>
}

interface LiveServerEvent {
  type: string
  event_id?: string
  client_event_id?: string
  delta?: string
  start_ms?: number
  end_ms?: number
  delegation?: { id: string; type: string; target: string }
  error?: { type?: string; code?: null | string; message?: string; client_event_id?: string }
  usage?: { seconds?: number }
  reason?: string
  session?: { id: string }
}

export interface LiveTranscriptFragment {
  speaker: 'assistant' | 'user'
  text: string
  startMs: number
  endMs: number
}

export interface VoiceLiveHandlers {
  /** GPT-Live asked the backend (Hermes) for help. `context` is the recent
   *  transcript window, newest last — the delegation itself carries no text. */
  onDelegation: (delegationId: string, context: LiveTranscriptFragment[]) => void
  /** Vendor-side error. `fatal` when the session is gone. */
  onError: (message: string, fatal: boolean) => void
  /** `session.closed` arrived (or the transport dropped without it). */
  onClosed: (reason: string, usageSeconds: null | number) => void
  /** Transcript deltas, for captions / live UI. */
  onTranscript?: (fragment: LiveTranscriptFragment) => void
  /** Assistant audio output level hint: the remote track is speaking. */
  onSpeakingChange?: (speaking: boolean) => void
}

const CLOSE_TIMEOUT_MS = 15_000
const ICE_GATHER_TIMEOUT_MS = 10_000
// Vendor cap: 500 tokens per append. ~4 chars/token, keep headroom.
const APPEND_CHAR_LIMIT = 1_400
// How much conversation the backend receives per delegation.
const CONTEXT_WINDOW_MS = 5 * 60_000
const CONTEXT_MAX_FRAGMENTS = 80

export async function fetchVoiceLiveStatus(): Promise<null | VoiceLiveStatus> {
  try {
    const response = await hermesApi<{ ok: boolean } & VoiceLiveStatus>({
      ...profileScoped(),
      path: '/api/audio/voice-live/status'
    })

    if (!response?.ok) {
      return null
    }

    return {
      available: Boolean(response.available),
      mode: response.mode === 'gpt-live' ? 'gpt-live' : 'chained',
      model: response.model,
      reason: response.reason ?? null,
      voice: response.voice
    }
  } catch {
    // Older backend without the endpoint → chained.
    return null
  }
}

/** Split a reply into append-sized chunks on sentence boundaries. */
export function chunkForCommentary(text: string, limit = APPEND_CHAR_LIMIT): string[] {
  const clean = text.replace(/\s+/g, ' ').trim()

  if (!clean) {
    return []
  }

  if (clean.length <= limit) {
    return [clean]
  }

  const chunks: string[] = []
  let current = ''

  for (const sentence of clean.split(/(?<=[.!?])\s+/)) {
    if (sentence.length > limit) {
      if (current) {
        chunks.push(current)
        current = ''
      }

      for (let index = 0; index < sentence.length; index += limit) {
        chunks.push(sentence.slice(index, index + limit))
      }

      continue
    }

    const candidate = current ? `${current} ${sentence}` : sentence

    if (candidate.length > limit) {
      chunks.push(current)
      current = sentence
    } else {
      current = candidate
    }
  }

  if (current) {
    chunks.push(current)
  }

  return chunks
}

/** Seed history for a new Live session from the chat transcript (text turns only). */
export function toLiveHistory(
  turns: Array<{ role: 'assistant' | 'user'; text: string }>,
  maxMessages = 24,
  maxChars = 6_000
): LiveHistoryMessage[] {
  const out: LiveHistoryMessage[] = []
  let budget = maxChars

  for (const turn of [...turns].reverse()) {
    const text = turn.text.replace(/\s+/g, ' ').trim().slice(0, 1_200)

    if (!text) {
      continue
    }

    if (out.length >= maxMessages || budget - text.length < 0) {
      break
    }

    budget -= text.length
    out.unshift({
      content: [{ text, type: turn.role === 'assistant' ? 'output_text' : 'input_text' }],
      role: turn.role,
      type: 'message'
    })
  }

  return out
}

async function waitForIceGathering(connection: RTCPeerConnection): Promise<void> {
  if (connection.iceGatheringState === 'complete') {
    return
  }

  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      connection.removeEventListener('icegatheringstatechange', onState)
      // Trickle is fine: the vendor answers with the candidates it has.
      resolve()
    }, ICE_GATHER_TIMEOUT_MS)

    function onState() {
      if (connection.iceGatheringState !== 'complete') {
        return
      }

      window.clearTimeout(timeout)
      connection.removeEventListener('icegatheringstatechange', onState)
      resolve()
    }

    connection.addEventListener('icegatheringstatechange', onState)
    connection.addEventListener('connectionstatechange', () => {
      if (connection.connectionState === 'failed') {
        window.clearTimeout(timeout)
        reject(new Error('WebRTC connection failed'))
      }
    })
  })
}

export class VoiceLiveSession {
  readonly audio: HTMLAudioElement
  private peer: null | RTCPeerConnection = null
  private events: null | RTCDataChannel = null
  private microphone: null | MediaStream = null
  private closeTimer: null | number = null
  private finalized = false
  private started = false
  private eventCounter = 0
  private transcript: LiveTranscriptFragment[] = []
  private speakingProbe: null | number = null
  private analyser: null | AnalyserNode = null
  private audioContext: null | AudioContext = null
  private lastSpeaking = false
  sessionId: null | string = null
  /** The delegation currently being answered by Hermes; late results for an
   *  older id are dropped by the conversation hook. */
  activeDelegationId: null | string = null

  constructor(private readonly handlers: VoiceLiveHandlers) {
    this.audio = new Audio()
    this.audio.autoplay = true
  }

  get connected(): boolean {
    return this.started && this.events?.readyState === 'open'
  }

  private nextEventId(prefix: string): string {
    this.eventCounter += 1

    return `${prefix}_${this.eventCounter}`
  }

  private send(event: Record<string, unknown>): boolean {
    if (!this.events || this.events.readyState !== 'open') {
      return false
    }

    this.events.send(JSON.stringify(event))

    return true
  }

  /** Recent conversation, oldest first, bounded by time and count. */
  contextWindow(): LiveTranscriptFragment[] {
    const last = this.transcript.at(-1)

    if (!last) {
      return []
    }

    const floor = last.endMs - CONTEXT_WINDOW_MS

    return this.transcript.filter(fragment => fragment.endMs >= floor).slice(-CONTEXT_MAX_FRAGMENTS)
  }

  async start(history: LiveHistoryMessage[]): Promise<void> {
    if (this.peer) {
      throw new Error('GPT-Live session already started')
    }

    const connection = new RTCPeerConnection()
    this.peer = connection

    connection.addEventListener('track', event => {
      const stream = new MediaStream([event.track])
      this.audio.srcObject = stream
      void this.audio.play().catch(() => undefined)
      this.armSpeakingProbe(stream)
    })
    connection.addEventListener('connectionstatechange', () => {
      if (connection.connectionState === 'failed' || connection.connectionState === 'disconnected') {
        this.finish('connection_lost', null)
      }
    })

    this.microphone = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, echoCancellation: true, noiseSuppression: true }
    })

    for (const track of this.microphone.getAudioTracks()) {
      connection.addTrack(track, this.microphone)
    }

    // Register the data channel before the offer so its m-line is negotiated.
    const events = connection.createDataChannel('oai-events')
    this.events = events
    events.addEventListener('message', ({ data }) => this.handleEvent(String(data)))
    events.addEventListener('close', () => {
      if (!this.finalized) {
        this.finish('connection_lost', null)
      }
    })

    const offer = await connection.createOffer()
    await connection.setLocalDescription(offer)
    await waitForIceGathering(connection)

    const sdp = connection.localDescription?.sdp

    if (!sdp) {
      throw new Error('Missing local SDP offer')
    }

    const response = await hermesApi<{
      ok: boolean
      session?: { id: string }
      transport?: { sdp: string; type: string }
    }>({
      ...profileScoped(),
      body: { history, sdp },
      method: 'POST',
      path: '/api/audio/voice-live/session',
      timeoutMs: 45_000
    })

    if (!response?.ok || !response.transport?.sdp) {
      throw new Error('GPT-Live session creation failed')
    }

    this.sessionId = response.session?.id ?? null
    await connection.setRemoteDescription({ sdp: response.transport.sdp, type: 'answer' })
  }

  private armSpeakingProbe(stream: MediaStream): void {
    try {
      const context = new AudioContext()
      const source = context.createMediaStreamSource(stream)
      const analyser = context.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)
      this.audioContext = context
      this.analyser = analyser
      const buffer = new Uint8Array(analyser.frequencyBinCount)
      let quietFrames = 0

      this.speakingProbe = window.setInterval(() => {
        analyser.getByteTimeDomainData(buffer)
        let peak = 0

        for (const sample of buffer) {
          peak = Math.max(peak, Math.abs(sample - 128))
        }

        const loud = peak > 6
        quietFrames = loud ? 0 : quietFrames + 1
        const speaking = loud || quietFrames < 4

        if (speaking !== this.lastSpeaking) {
          this.lastSpeaking = speaking
          this.handlers.onSpeakingChange?.(speaking)
        }
      }, 100)
    } catch {
      // No analyser → no speaking indicator; the conversation still works.
    }
  }

  private handleEvent(raw: string): void {
    let event: LiveServerEvent

    try {
      event = JSON.parse(raw) as LiveServerEvent
    } catch {
      return
    }

    switch (event.type) {
      case 'session.started':
        this.started = true
        this.sessionId = event.session?.id ?? this.sessionId

        return

      case 'session.input_transcript.delta':
      case 'session.output_transcript.delta': {
        const fragment: LiveTranscriptFragment = {
          endMs: event.end_ms ?? 0,
          speaker: event.type === 'session.input_transcript.delta' ? 'user' : 'assistant',
          startMs: event.start_ms ?? 0,
          text: event.delta ?? ''
        }

        this.transcript.push(fragment)

        if (this.transcript.length > 2_000) {
          this.transcript.splice(0, this.transcript.length - 1_500)
        }

        this.handlers.onTranscript?.(fragment)

        return
      }

      case 'session.delegation.created': {
        const id = event.delegation?.id

        if (id) {
          this.activeDelegationId = id
          this.handlers.onDelegation(id, this.contextWindow())
        }

        return
      }

      case 'error': {
        const code = event.error?.code ?? ''

        // Late appends after our own close are expected noise.
        if (code === 'context_injection_incomplete') {
          return
        }

        this.handlers.onError(event.error?.message ?? 'GPT-Live error', false)

        return
      }

      case 'session.closed':
        this.finish(event.reason ?? 'closed', event.usage?.seconds ?? null)

        return

      default:
        return
    }
  }

  /** Quiet progress for the live model ("Hermes is running the tests…"). */
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

  /** A result the voice should say aloud (paraphrased). */
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

  /** Steer the live persona mid-conversation (session-wide). */
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
    for (const track of this.microphone?.getAudioTracks() ?? []) {
      track.enabled = !muted
    }

    this.send({ event_id: this.nextEventId(muted ? 'mute' : 'unmute'), type: muted ? 'session.input_audio.mute' : 'session.input_audio.unmute' })
  }

  /** Graceful close: ask for `session.closed`, tear down after it (or a timeout). */
  close(): void {
    if (this.finalized) {
      return
    }

    if (!this.send({ type: 'session.close' })) {
      this.finish('close_requested', null)

      return
    }

    this.closeTimer = window.setTimeout(() => this.finish('close_requested', null), CLOSE_TIMEOUT_MS)
  }

  private finish(reason: string, usageSeconds: null | number): void {
    if (this.finalized) {
      return
    }

    this.finalized = true

    if (this.closeTimer) {
      window.clearTimeout(this.closeTimer)
      this.closeTimer = null
    }

    if (this.speakingProbe) {
      window.clearInterval(this.speakingProbe)
      this.speakingProbe = null
    }

    this.analyser?.disconnect()
    void this.audioContext?.close().catch(() => undefined)
    this.microphone?.getTracks().forEach(track => track.stop())
    this.events?.close()
    this.peer?.close()
    this.audio.srcObject = null
    this.audio.pause()
    this.handlers.onClosed(reason, usageSeconds)
  }
}
