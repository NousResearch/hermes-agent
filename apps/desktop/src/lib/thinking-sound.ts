// Ambient "thinking" sound for the desktop voice conversation. While the agent
// works (status === 'thinking') no audio flows, which reads as "it died" during
// long thinking/tool stretches. A calm, quiet, repeating pair of soft bubble
// blips fills the gap — same WebAudio oscillator synthesis approach as
// wake-sound.ts / completion-sound.ts (no asset to ship), mirroring the
// backend's numpy-synthesized blips in tools/voice_mode.py so CLI and desktop
// sound alike.
//
// Honours the shared sound-mute toggle ($hapticsMuted) and the
// voice.thinking_sound config gate ($thinkingSoundEnabled). Stops instantly on
// stopThinkingSound() — callers fire it the moment TTS starts, the mic re-arms,
// or the conversation ends.

import { getAudioContext } from '@/lib/audio-context'
import { $hapticsMuted } from '@/store/haptics'
import { $thinkingSoundEnabled } from '@/store/voice-prefs'

let timer: number | null = null
let blipIndex = 0

// One soft "blub": short sine with a gentle downward pitch glide and a smooth
// attack into an exponential decay — no clicks, deliberately quiet.
function blub(ac: AudioContext, freq: number) {
  const t0 = ac.currentTime + 0.01
  const dur = 0.16
  const osc = ac.createOscillator()
  const env = ac.createGain()

  osc.type = 'sine'
  osc.frequency.setValueAtTime(freq, t0)
  osc.frequency.exponentialRampToValueAtTime(freq * 0.72, t0 + dur)

  env.gain.setValueAtTime(0.0001, t0)
  env.gain.exponentialRampToValueAtTime(0.08, t0 + 0.02)
  env.gain.exponentialRampToValueAtTime(0.0001, t0 + dur)

  osc.connect(env)
  env.connect(ac.destination)
  osc.start(t0)
  osc.stop(t0 + dur + 0.02)
}

export function isThinkingSoundActive(): boolean {
  return timer !== null
}

/** Start the repeating thinking blips (idempotent). Best-effort, never throws. */
export function startThinkingSound(): void {
  if (timer !== null || !$thinkingSoundEnabled.get()) {
    return
  }

  const tick = () => {
    if ($hapticsMuted.get() === false) {
      const ac = getAudioContext()

      if (ac) {
        try {
          // Alternate two calm pitches (G4 / E4), matching the backend blips.
          blub(ac, blipIndex % 2 === 0 ? 392 : 329.6)
        } catch {
          // Audio backend unavailable — stay silent, keep the loop harmless.
        }
      }
    }

    blipIndex += 1
    // ~0.8-1.2s spacing with slight randomization so it reads organic.
    timer = window.setTimeout(tick, 800 + Math.random() * 400)
  }

  timer = window.setTimeout(tick, 400)
}

/** Stop the thinking blips instantly (idempotent). */
export function stopThinkingSound(): void {
  if (timer !== null) {
    window.clearTimeout(timer)
    timer = null
  }
}
