import { speakText } from '@/hermes'
import {
  $voicePlayback,
  setVoicePlaybackState,
  type VoicePlaybackSource,
  type VoicePlaybackState
} from '@/store/voice-playback'

import { sanitizeTextForSpeech } from './speech-text'

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

interface SpeechAudioResponse {
  audio_bytes?: null | number
  data_url: string
  mime_type?: string
  ok?: boolean
  playback?: string
  provider?: string
  request_id?: string
  timings_ms?: Record<string, unknown>
  transport?: string
}

function voiceRequestId(options: VoicePlaybackOptions): string {
  const suffix = options.messageId ? `-${options.messageId}` : ''
  return `desktop-${Date.now().toString(36)}-${sequence}-${options.source}${suffix}`.replace(/[^A-Za-z0-9_.:-]/g, '').slice(0, 80)
}

async function getSpeechAudio(text: string, options: VoicePlaybackOptions, requestId: string): Promise<SpeechAudioResponse> {
  return speakText(text, { requestId, source: options.source })
}

export function stopVoicePlayback() {
  sequence += 1
  currentStop?.()
  currentStop = null

  if (window.hermesDesktop.stopNativeAudioPlayback) {
    void window.hermesDesktop.stopNativeAudioPlayback().catch(() => undefined)
  }

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

function shouldUseNativeAudio(_options: VoicePlaybackOptions) {
  // Prefer the native player for every Desktop speech path. Chromium/WebAudio
  // can enter a bad device state on Joe's CachyOS/PipeWire stack while native
  // players (`mpv`/`afplay`) continue routing cleanly through the OS default
  // sink. Browser audio remains available as the fallback below.
  return typeof window.hermesDesktop.playAudioDataUrl === 'function'
}

async function playNativeSpeechAudio(
  response: SpeechAudioResponse,
  options: VoicePlaybackOptions,
  isCurrent: () => boolean
): Promise<boolean> {
  const playAudioDataUrl = window.hermesDesktop.playAudioDataUrl
  const requestId = response.request_id

  if (!playAudioDataUrl) {
    return false
  }

  setVoicePlaybackState(currentState('speaking', options))

  const playbackResult = await new Promise<{ code?: null | number; durationMs?: number; ok: boolean; signal?: null | string }>((resolve, reject) => {
    currentStop = () => {
      currentStop = null
      void window.hermesDesktop.stopNativeAudioPlayback?.().catch(() => undefined)
      resolve({ ok: true, signal: 'SIGTERM' })
    }

    playAudioDataUrl(response.data_url, {
      audioBytes: response.audio_bytes ?? null,
      mimeType: response.mime_type,
      provider: response.provider,
      requestId,
      source: options.source,
      transport: response.transport
    })
      .then(resolve)
      .catch(reject)
  })

  console.info('[voice] native playback result', {
    request_id: requestId,
    result: playbackResult,
    timings_ms: response.timings_ms,
    transport: response.transport
  })

  currentStop = null

  if (!isCurrent()) {
    return false
  }

  if (!playbackResult.ok) {
    throw new Error(`Native playback failed (code ${playbackResult.code ?? 'unknown'}, signal ${playbackResult.signal ?? 'none'})`)
  }

  setVoicePlaybackState(currentState('idle'))
  return true
}

async function playBrowserSpeechAudio(
  response: SpeechAudioResponse,
  options: VoicePlaybackOptions,
  isCurrent: () => boolean
): Promise<boolean> {
  const audio = new Audio(response.data_url)
  currentAudio = audio
  setVoicePlaybackState(currentState('speaking', options, audio))

  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      audio.removeEventListener('ended', onEnded)
      audio.removeEventListener('error', onError)
      currentStop = null
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
    void audio.play().catch(reject)
  })

  if (!isCurrent()) {
    return false
  }

  currentAudio = null
  setVoicePlaybackState(currentState('idle'))

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

  const requestId = voiceRequestId(options)

  setVoicePlaybackState(currentState('preparing', options))

  try {
    const response = await getSpeechAudio(speakableText, options, requestId)
    response.request_id = response.request_id || requestId

    if (!isCurrent()) {
      return false
    }

    if (response.playback === 'livekit') {
      // The sidecar already published this audio into the LiveKit room;
      // playing the data_url locally too would double-speak for anyone at the desk.
      setVoicePlaybackState(currentState('idle'))
      return true
    }

    if (shouldUseNativeAudio(options)) {
      try {
        return await playNativeSpeechAudio(response, options, isCurrent)
      } catch (nativeError) {
        console.warn('Native voice playback failed; falling back to browser audio.', nativeError)

        if (!isCurrent()) {
          return false
        }
      }
    }

    return await playBrowserSpeechAudio(response, options, isCurrent)
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
