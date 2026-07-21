import { speakText } from '@/hermes'
import {
  $voicePlayback,
  setVoicePlaybackState,
  type VoicePlaybackSource,
  type VoicePlaybackState
} from '@/store/voice-playback'

import { sanitizeTextForSpeech, splitSpeechText } from './speech-text'

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

export function playSelectedSpeechText(text: string): Promise<boolean> {
  return playSpeechText(text, {
    messageId: 'selection-read-aloud',
    source: 'read-aloud'
  })
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

  const chunks = splitSpeechText(speakableText)

  try {
    for (const chunk of chunks) {
      if (!isCurrent()) {
        return false
      }

      setVoicePlaybackState(currentState('preparing', options))
      const response = await speakText(chunk)

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
    }

    setVoicePlaybackState(currentState('idle'))

    return true
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
