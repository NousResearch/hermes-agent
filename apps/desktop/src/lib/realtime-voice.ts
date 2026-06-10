import { createRealtimeClientSecret } from '@/hermes'

const OPENAI_REALTIME_CALLS_URL = 'https://api.openai.com/v1/realtime/calls'
const OPENAI_REALTIME_LEGACY_URL = 'https://api.openai.com/v1/realtime'

export type RealtimeVoiceStatus = 'closed' | 'connecting' | 'listening' | 'speaking'

export interface RealtimeVoiceSessionOptions {
  instructions?: string
  model?: string
  onError?: (error: unknown) => void
  onLevel?: (level: number) => void
  onStatus?: (status: RealtimeVoiceStatus) => void
  voice?: string
}

export interface RealtimeVoiceSession {
  mute: (muted: boolean) => void
  stop: () => void
}

function emitStatus(options: RealtimeVoiceSessionOptions, status: RealtimeVoiceStatus) {
  options.onStatus?.(status)
}

async function postOfferToCallsApi(clientSecret: string, offer: RTCSessionDescriptionInit, model: string) {
  const form = new FormData()
  form.append('sdp', offer.sdp || '')
  form.append('session', JSON.stringify({ type: 'realtime', model }))

  const response = await fetch(OPENAI_REALTIME_CALLS_URL, {
    method: 'POST',
    headers: { Authorization: `Bearer ${clientSecret}` },
    body: form
  })

  if (!response.ok) {
    throw new Error(`Realtime calls SDP failed: HTTP ${response.status} ${await response.text()}`)
  }

  return response.text()
}

async function postOfferToLegacyApi(clientSecret: string, offer: RTCSessionDescriptionInit, model: string) {
  const response = await fetch(`${OPENAI_REALTIME_LEGACY_URL}?model=${encodeURIComponent(model)}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${clientSecret}`,
      'Content-Type': 'application/sdp'
    },
    body: offer.sdp || ''
  })

  if (!response.ok) {
    throw new Error(`Realtime legacy SDP failed: HTTP ${response.status} ${await response.text()}`)
  }

  return response.text()
}

export async function startRealtimeVoiceSession(options: RealtimeVoiceSessionOptions = {}): Promise<RealtimeVoiceSession> {
  if (typeof RTCPeerConnection === 'undefined') {
    throw new Error('Realtime voice requires WebRTC support')
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Realtime voice requires microphone capture support')
  }

  emitStatus(options, 'connecting')
  const secret = await createRealtimeClientSecret({
    instructions: options.instructions,
    model: options.model,
    voice: options.voice
  })

  const pc = new RTCPeerConnection()
  let stream: MediaStream | null = null
  let context: AudioContext | null = null
  let levelTimer: number | null = null
  const audio = new Audio()

  const cleanup = () => {
    if (levelTimer !== null) {
      window.clearInterval(levelTimer)
      levelTimer = null
    }
    options.onLevel?.(0)
    pc.getSenders().forEach(sender => sender.track?.stop())
    stream?.getTracks().forEach(track => track.stop())
    audio.pause()
    audio.srcObject = null
    void context?.close().catch(() => undefined)
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const [track] = stream.getAudioTracks()
    if (!track) {
      throw new Error('No microphone audio track available')
    }

    pc.addTrack(track, stream)

    audio.autoplay = true

    pc.ontrack = event => {
      const [remoteStream] = event.streams
      if (remoteStream) {
        audio.srcObject = remoteStream
        void audio.play().catch(error => options.onError?.(error))
      }
    }

    const dc = pc.createDataChannel('oai-events')
    dc.onopen = () => emitStatus(options, 'listening')
    dc.onerror = event => options.onError?.(event)
    dc.onmessage = event => {
      try {
        const data = JSON.parse(String(event.data)) as { type?: string }
        switch (data.type) {
          case 'response.audio.started':
          case 'response.output_audio.started':
          case 'response.created':
            emitStatus(options, 'speaking')
            break
          case 'response.audio.done':
          case 'response.output_audio.done':
          case 'response.done':
          case 'input_audio_buffer.speech_started':
            emitStatus(options, 'listening')
            break
          default:
            break
        }
      } catch {
        // Ignore non-JSON diagnostic frames.
      }
    }

    context = new AudioContext()
    const source = context.createMediaStreamSource(stream)
    const analyser = context.createAnalyser()
    analyser.fftSize = 256
    source.connect(analyser)
    const samples = new Uint8Array(analyser.frequencyBinCount)
    levelTimer = window.setInterval(() => {
      analyser.getByteFrequencyData(samples)
      const average = samples.reduce((sum, value) => sum + value, 0) / Math.max(1, samples.length)
      options.onLevel?.(Math.min(1, average / 128))
    }, 100)

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    let answerSdp: string
    try {
      answerSdp = await postOfferToCallsApi(secret.client_secret, offer, secret.model)
    } catch (callsError) {
      console.warn('[hermes] Realtime calls SDP failed; trying legacy SDP endpoint', callsError)
      answerSdp = await postOfferToLegacyApi(secret.client_secret, offer, secret.model)
    }

    await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp })
    emitStatus(options, 'listening')

    return {
      mute(muted: boolean) {
        track.enabled = !muted
      },
      stop() {
        try {
          dc.close()
        } catch {
          // already closed
        }
        cleanup()
        pc.close()
        emitStatus(options, 'closed')
      }
    }
  } catch (error) {
    cleanup()
    pc.close()
    emitStatus(options, 'closed')
    throw error
  }
}
