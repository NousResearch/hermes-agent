import { persistentAtom } from '@/lib/persisted'

// Device-level speech playback speed. Choosing a rate on any read-aloud or
// voice-conversation playback persists it as the rate every later playback
// starts from, so a listener who prefers 1.5x doesn't re-pick it per reply.
// Applied client-side (audio element playbackRate / AudioBufferSource
// playbackRate) rather than at synthesis: it works for every provider
// including the ones that ignore `speed`, retunes the reply that is already
// speaking, and never double-applies with a synthesis-side rate.

const STORAGE_KEY = 'hermes.desktop.voicePlaybackSpeed'

export const DEFAULT_VOICE_PLAYBACK_SPEED = 1

// Same window as the text_to_speech tool's `speed` (0.25–4.0).
const MIN_SPEED = 0.25
const MAX_SPEED = 4

/** True for rates speech playback can sensibly run at. */
export function isVoicePlaybackSpeed(speed: number): boolean {
  return Number.isFinite(speed) && speed >= MIN_SPEED && speed <= MAX_SPEED
}

// The menu's presets. 0.75× for following dense or unfamiliar speech, 1.25×
// and 1.5× for review, 2× for skimming a long read-aloud.
export const VOICE_PLAYBACK_SPEED_PRESETS = [0.75, 1, 1.25, 1.5, 2] as const

export const $voicePlaybackSpeed = persistentAtom<number>(STORAGE_KEY, DEFAULT_VOICE_PLAYBACK_SPEED, {
  decode: raw => {
    const parsed = Number(raw)

    return isVoicePlaybackSpeed(parsed) ? parsed : DEFAULT_VOICE_PLAYBACK_SPEED
  },
  // The default doesn't need a stored record; encoding null removes the key.
  encode: value => (value === DEFAULT_VOICE_PLAYBACK_SPEED ? null : String(value))
})

/** Persist a user-chosen rate; out-of-range values are ignored, not clamped. */
export function setVoicePlaybackSpeed(speed: number) {
  if (isVoicePlaybackSpeed(speed) && speed !== $voicePlaybackSpeed.get()) {
    $voicePlaybackSpeed.set(speed)
  }
}
