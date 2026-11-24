import { atom } from 'nanostores'

export type VoicePlaybackSource = 'read-aloud' | 'voice-conversation'
export type VoicePlaybackStatus = 'idle' | 'preparing' | 'speaking'

export interface VoicePlaybackState {
  audioElement: HTMLAudioElement | null
  messageId: string | null
  sequence: number
  source: VoicePlaybackSource | null
  status: VoicePlaybackStatus
}

export const $voicePlayback = atom<VoicePlaybackState>({
  audioElement: null,
  messageId: null,
  sequence: 0,
  source: null,
  status: 'idle'
})

/** What plugins see: either voice (built-in or GPT-Live), without the audio. */
export interface VoiceState {
  /** Why speech last ended: it played out, or it was cut short. `null` while speaking or on a pause. */
  endReason: 'finished' | 'stopped' | null
  messageId: string | null
  source: 'live' | VoicePlaybackSource | null
  status: VoicePlaybackStatus
}

export const $voice = atom<VoiceState>({ endReason: null, messageId: null, source: null, status: 'idle' })

export function setVoicePlaybackState(next: VoicePlaybackState, endReason: 'finished' | 'stopped' = 'stopped') {
  const { messageId, source, status } = next
  const wasIdle = $voicePlayback.get().status === 'idle'

  $voicePlayback.set(next)

  // Stopping from silence ends nothing: it must not overwrite the last reason or a live session.
  if (status !== 'idle' || !wasIdle) {
    $voice.set({ endReason: status === 'idle' ? endReason : null, messageId, source, status })
  }
}
