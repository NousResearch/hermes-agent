/**
 * Soft interface sounds: a glassy tick for a tap, a rising pair for opening or
 * success, a falling one for closing or cancel. Synthesised with WebAudio (no
 * asset to ship, same approach as the wake and completion cues) and kept quiet
 * on purpose — they confirm a press, they never announce it.
 *
 * They ride on the app's existing haptic seam: every `triggerHaptic(intent)`
 * call site already marks a deliberate press, so one hook there sounds all of
 * them, and the title bar's mute button (`$hapticsMuted`) silences both.
 * Settings → Appearance switches just the sounds off (`$uiSoundsEnabled`).
 */

import { atom } from 'nanostores'

import { getAudioContext } from '@/lib/audio-context'
import { persistBoolean, storedBoolean } from '@/lib/storage'
import { $hapticsMuted } from '@/store/haptics'

const STORAGE_KEY = 'czesiek.ui-sounds.v1'

export const $uiSoundsEnabled = atom(storedBoolean(STORAGE_KEY, true))
$uiSoundsEnabled.subscribe(on => persistBoolean(STORAGE_KEY, on))

export type UiSound =
  | 'cancel'
  | 'close'
  | 'crisp'
  | 'error'
  | 'grow'
  | 'open'
  | 'selection'
  | 'shrink'
  | 'snap'
  | 'submit'
  | 'success'
  | 'tap'
  | 'warning'

interface Voice {
  /** Seconds after the sound starts. */
  at: number
  dur: number
  freq: number
  gain: number
  type?: OscillatorType
}

// Tuned around a pentatonic set so any two sounds close together never clash.
const VOICES: Record<UiSound, Voice[]> = {
  cancel: [
    { at: 0, dur: 0.09, freq: 523.25, gain: 0.05 },
    { at: 0.07, dur: 0.12, freq: 392, gain: 0.045 }
  ],
  close: [
    { at: 0, dur: 0.08, freq: 659.25, gain: 0.04 },
    { at: 0.06, dur: 0.1, freq: 523.25, gain: 0.035 }
  ],
  crisp: [{ at: 0, dur: 0.05, freq: 1567.98, gain: 0.035, type: 'triangle' }],
  error: [
    { at: 0, dur: 0.1, freq: 233.08, gain: 0.05, type: 'triangle' },
    { at: 0.09, dur: 0.16, freq: 196, gain: 0.05, type: 'triangle' }
  ],
  grow: [{ at: 0, dur: 0.07, freq: 880, gain: 0.03 }],
  open: [
    { at: 0, dur: 0.08, freq: 523.25, gain: 0.04 },
    { at: 0.06, dur: 0.12, freq: 783.99, gain: 0.045 }
  ],
  selection: [{ at: 0, dur: 0.04, freq: 1318.51, gain: 0.025 }],
  shrink: [{ at: 0, dur: 0.07, freq: 659.25, gain: 0.03 }],
  snap: [
    { at: 0, dur: 0.05, freq: 392, gain: 0.05, type: 'triangle' },
    { at: 0.03, dur: 0.09, freq: 587.33, gain: 0.03 }
  ],
  submit: [
    { at: 0, dur: 0.07, freq: 659.25, gain: 0.04 },
    { at: 0.06, dur: 0.14, freq: 987.77, gain: 0.05 }
  ],
  success: [
    { at: 0, dur: 0.09, freq: 587.33, gain: 0.04 },
    { at: 0.08, dur: 0.1, freq: 783.99, gain: 0.045 },
    { at: 0.16, dur: 0.2, freq: 1174.66, gain: 0.05 }
  ],
  tap: [{ at: 0, dur: 0.06, freq: 1046.5, gain: 0.035 }],
  warning: [
    { at: 0, dur: 0.1, freq: 440, gain: 0.045 },
    { at: 0.12, dur: 0.1, freq: 440, gain: 0.04 }
  ]
}

/** Longest a human-paced UI plays before more is machine-gunning: at most this many per window. */
export const SOUND_RATE_LIMIT = 6
export const SOUND_RATE_WINDOW_MS = 1000

/** Pure rate limiter: true (and records) when another sound may play at `now`. */
export function admitSound(recent: number[], now: number): boolean {
  while (recent.length && now - recent[0]! >= SOUND_RATE_WINDOW_MS) {
    recent.shift()
  }

  if (recent.length >= SOUND_RATE_LIMIT) {
    return false
  }

  recent.push(now)

  return true
}

const recentSounds: number[] = []

export function playUiSound(sound: UiSound): void {
  if (!$uiSoundsEnabled.get() || $hapticsMuted.get() || !admitSound(recentSounds, performance.now())) {
    return
  }

  const ac = getAudioContext()

  if (!ac) {
    return
  }

  try {
    const master = ac.createGain()
    master.gain.setValueAtTime(1, ac.currentTime)
    master.connect(ac.destination)
    const t0 = ac.currentTime + 0.005

    for (const voice of VOICES[sound]) {
      const osc = ac.createOscillator()
      const env = ac.createGain()
      const start = t0 + voice.at
      const end = start + voice.dur

      osc.type = voice.type ?? 'sine'
      osc.frequency.setValueAtTime(voice.freq, start)
      // Fast attack into an exponential decay: no click at either end.
      env.gain.setValueAtTime(0.0001, start)
      env.gain.exponentialRampToValueAtTime(Math.max(voice.gain, 0.0002), start + 0.006)
      env.gain.exponentialRampToValueAtTime(0.0001, end)
      osc.connect(env)
      env.connect(master)
      osc.start(start)
      osc.stop(end + 0.02)
    }
  } catch {
    // A dead audio context must never break the press that triggered the sound.
  }
}
