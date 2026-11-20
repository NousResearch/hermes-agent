// Wake-word activation chime. A short, bright, rising two-note "ding" that
// plays the moment "Hey Hermes" is detected, so it's obvious the wake
// registered before voice capture starts. Deliberately distinct from the
// turn-end completion cue (completion-sound.ts): this one RISES (open/ready),
// the completion cue settles (done). Reuses the same lightweight WebAudio
// synthesis approach — no asset file to ship.

import { getAudioContext } from '@/lib/audio-context'
import { $hapticsMuted } from '@/store/haptics'

// One enveloped sine voice → master. Linear-ish attack into an exponential
// decay keeps the tail smooth and avoids the click you get ramping to zero.
function ding(ac: AudioContext, master: GainNode, t0: number, freq: number, dur: number, gain: number) {
  const osc = ac.createOscillator()
  const env = ac.createGain()
  const end = t0 + dur

  osc.type = 'sine'
  osc.frequency.setValueAtTime(freq, t0)

  env.gain.setValueAtTime(0.0001, t0)
  env.gain.exponentialRampToValueAtTime(Math.max(gain, 0.0002), t0 + 0.008)
  env.gain.exponentialRampToValueAtTime(0.0001, end)

  osc.connect(env)
  env.connect(master)
  osc.start(t0)
  osc.stop(end + 0.02)
}

// Play the wake chime. Honours the shared sound-mute toggle ($hapticsMuted),
// the same gate the completion cue uses, so muting turn-end sounds also
// silences this. Best-effort: never throws into the wake-event handler.
export function playWakeSound(): void {
  if ($hapticsMuted.get()) {
    return
  }

  const ac = getAudioContext()

  if (!ac) {
    return
  }

  try {
    const master = ac.createGain()
    master.gain.setValueAtTime(0.5, ac.currentTime)
    master.connect(ac.destination)

    const t0 = ac.currentTime + 0.01
    // Rising perfect-fourth: G5 -> C6. Short and bright — "listening".
    ding(ac, master, t0, 783.99, 0.12, 0.06)
    ding(ac, master, t0 + 0.1, 1046.5, 0.28, 0.07)
  } catch {
    // WebAudio can throw if the context died mid-call; a missed chime must
    // never break wake handling.
  }
}
