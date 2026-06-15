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

export function setVoicePlaybackState(next: VoicePlaybackState) {
  $voicePlayback.set(next)
}

// ── Auto-TTS preference ──────────────────────────────────────────────
const AUTO_TTS_STORAGE_KEY = 'hermes.desktop.autoTts'

function loadAutoTts(): boolean {
  if (typeof window === 'undefined') return false
  try {
    const raw = window.localStorage.getItem(AUTO_TTS_STORAGE_KEY)
    if (raw !== null) return JSON.parse(raw) === true
  } catch { /* treat unparseable as unset */ }
  return false
}

export const $autoTts = atom(loadAutoTts())

/** Set auto-TTS preference.  Persists to localStorage so config refreshes
 *  do not override the user's explicit toggle. */
export function setAutoTts(value: boolean) {
  $autoTts.set(value)
  try {
    window.localStorage.setItem(AUTO_TTS_STORAGE_KEY, JSON.stringify(value))
  } catch { /* localStorage may be unavailable */ }
}

/** Apply auto_tts from config ONLY when the user has not manually toggled
 *  (i.e. no localStorage entry exists).  Called from use-hermes-config. */
export function setAutoTtsFromConfig(value: boolean) {
  if (typeof window === 'undefined') return
  if (window.localStorage.getItem(AUTO_TTS_STORAGE_KEY) !== null) return
  $autoTts.set(value)
}
