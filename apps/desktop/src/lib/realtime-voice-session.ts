import { createRealtimeVoiceSession } from '@/hermes'

const REALTIME_CALLS_URL = 'https://api.openai.com/v1/realtime/calls'

export type RealtimeVoiceStatus = 'idle' | 'connecting' | 'listening' | 'transcribing' | 'error'

export interface RealtimeTranscript {
  id: string
  text: string
}

export interface RealtimeVoiceEventAction {
  error?: string
  speechStarted?: boolean
  status?: RealtimeVoiceStatus
  transcript?: RealtimeTranscript
}

export function reduceRealtimeVoiceEvent(event: unknown): RealtimeVoiceEventAction {
  if (!event || typeof event !== 'object') {
    return { error: 'Invalid Realtime voice event', status: 'error' }
  }

  const record = event as Record<string, unknown>
  const type = typeof record.type === 'string' ? record.type : ''

  if (type.startsWith('response.') || type.includes('function_call')) {
    return { error: 'Unexpected Realtime response-generation event', status: 'error' }
  }

  switch (type) {
    case 'input_audio_buffer.speech_started':
      return { speechStarted: true, status: 'listening' }

    case 'input_audio_buffer.speech_stopped':
      return { status: 'transcribing' }
    case 'conversation.item.input_audio_transcription.completed': {
      const id = typeof record.item_id === 'string' ? record.item_id.trim() : ''
      const text = typeof record.transcript === 'string' ? record.transcript.trim() : ''

      return id && text ? { status: 'listening', transcript: { id, text } } : {}
    }

    case 'conversation.item.input_audio_transcription.failed':
      return { error: 'Realtime transcription failed', status: 'error' }

    case 'error':
      return { error: 'Realtime voice session failed', status: 'error' }

    default:
      return {}
  }
}

interface RealtimeVoiceCallbacks {
  onError?: (error: Error) => void
  onSpeechStarted?: () => void
  onStatus?: (status: RealtimeVoiceStatus) => void
  onTranscript: (transcript: RealtimeTranscript) => void
}

interface RealtimeVoiceDependencies {
  createPeerConnection?: () => RTCPeerConnection
  createSession?: typeof createRealtimeVoiceSession
  fetch?: typeof fetch
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>
}

export interface RealtimeVoiceConnectOptions {
  language?: string
  sessionId: string
}

export class RealtimeVoiceSession {
  private channel: RTCDataChannel | null = null
  private generation = 0
  private readonly seenTranscriptIds = new Set<string>()
  private peer: RTCPeerConnection | null = null
  private status: RealtimeVoiceStatus = 'idle'
  private stream: MediaStream | null = null

  private readonly createPeerConnection: () => RTCPeerConnection
  private readonly createSession: typeof createRealtimeVoiceSession
  private readonly fetchImpl: typeof fetch
  private readonly getUserMedia: (constraints: MediaStreamConstraints) => Promise<MediaStream>

  constructor(
    private readonly callbacks: RealtimeVoiceCallbacks,
    dependencies: RealtimeVoiceDependencies = {}
  ) {
    this.createPeerConnection = dependencies.createPeerConnection ?? (() => new RTCPeerConnection())
    this.createSession = dependencies.createSession ?? createRealtimeVoiceSession
    this.fetchImpl = dependencies.fetch ?? fetch
    this.getUserMedia = dependencies.getUserMedia ?? (constraints => navigator.mediaDevices.getUserMedia(constraints))
  }

  get currentStatus() {
    return this.status
  }

  async connect({ language, sessionId }: RealtimeVoiceConnectOptions): Promise<void> {
    this.disconnect()
    const ownGeneration = this.generation
    this.setStatus('connecting')

    try {
      const stream = await this.getUserMedia({
        audio: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true
        }
      })

      if (ownGeneration !== this.generation) {
        stream.getTracks().forEach(track => track.stop())

        return
      }

      const peer = this.createPeerConnection()
      const channel = peer.createDataChannel('oai-events')

      this.stream = stream
      this.peer = peer
      this.channel = channel
      stream.getAudioTracks().forEach(track => peer.addTrack(track, stream))
      channel.addEventListener('message', this.handleMessage)
      channel.addEventListener('close', this.handleChannelClose)
      peer.addEventListener('connectionstatechange', this.handleConnectionState)

      const offer = await peer.createOffer()
      await peer.setLocalDescription(offer)
      const session = await this.createSession(sessionId, language)

      if (ownGeneration !== this.generation) {
        return
      }

      if (!session.client_secret || session.expires_at * 1000 <= Date.now() + 5_000 || !session.session_binding) {
        throw new Error('Realtime voice session secret is invalid or expired')
      }

      const response = await this.fetchImpl(REALTIME_CALLS_URL, {
        method: 'POST',
        body: offer.sdp ?? '',
        headers: {
          Authorization: `Bearer ${session.client_secret}`,
          'Content-Type': 'application/sdp'
        }
      })

      if (!response.ok) {
        throw new Error(`Realtime WebRTC negotiation failed (${response.status})`)
      }

      const answerSdp = await response.text()

      if (!answerSdp || ownGeneration !== this.generation) {
        if (!answerSdp) {
          throw new Error('Realtime WebRTC negotiation returned an empty answer')
        }

        return
      }

      await peer.setRemoteDescription({ type: 'answer', sdp: answerSdp })

      if (peer.connectionState === 'connected') {
        this.setStatus('listening')
      }
    } catch (error) {
      if (ownGeneration !== this.generation) {
        return
      }

      this.fail(error)
    }
  }

  cancelInput() {
    if (this.channel?.readyState === 'open') {
      // Transcription sessions never send response.cancel/response.create.
      this.channel.send(JSON.stringify({ type: 'input_audio_buffer.clear' }))
    }
  }

  setMuted(muted: boolean) {
    this.stream?.getAudioTracks().forEach(track => {
      track.enabled = !muted
    })
  }

  disconnect() {
    this.generation += 1
    this.closeResources()
    this.seenTranscriptIds.clear()
    this.setStatus('idle')
  }

  private readonly handleMessage = (message: MessageEvent) => {
    let event: unknown

    try {
      event = JSON.parse(String(message.data))
    } catch {
      this.fail(new Error('Invalid Realtime voice event'))

      return
    }

    const action = reduceRealtimeVoiceEvent(event)

    if (action.error) {
      this.fail(new Error(action.error))

      return
    }

    if (action.status) {
      this.setStatus(action.status)
    }

    if (action.speechStarted) {
      this.callbacks.onSpeechStarted?.()
    }

    if (action.transcript && !this.seenTranscriptIds.has(action.transcript.id)) {
      this.seenTranscriptIds.add(action.transcript.id)
      this.callbacks.onTranscript(action.transcript)
    }
  }

  private readonly handleChannelClose = () => {
    if (this.status !== 'idle') {
      this.fail(new Error('Realtime voice data channel closed'))
    }
  }

  private readonly handleConnectionState = () => {
    const connectionState = this.peer?.connectionState

    if (connectionState === 'connected') {
      this.setStatus('listening')
    } else if (connectionState === 'failed' || connectionState === 'disconnected') {
      this.fail(new Error('Realtime voice connection was lost'))
    }
  }

  private fail(error: unknown) {
    const normalized = error instanceof Error ? error : new Error('Realtime voice session failed')

    this.closeResources()
    this.setStatus('error')
    this.callbacks.onError?.(normalized)
  }

  private closeResources() {
    if (this.channel) {
      this.channel.removeEventListener('message', this.handleMessage)
      this.channel.removeEventListener('close', this.handleChannelClose)
      this.channel.close()
      this.channel = null
    }

    if (this.peer) {
      this.peer.removeEventListener('connectionstatechange', this.handleConnectionState)
      this.peer.close()
      this.peer = null
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop())
      this.stream = null
    }
  }

  private setStatus(status: RealtimeVoiceStatus) {
    if (this.status === status) {
      return
    }

    this.status = status
    this.callbacks.onStatus?.(status)
  }
}
