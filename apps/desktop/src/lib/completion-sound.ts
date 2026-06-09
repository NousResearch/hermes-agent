// Completion sound bank for agent turn-end cues.
// Runtime playback is pinned to a curated default (currently variant 8).

import { $hapticsMuted } from '@/store/haptics'

type OscType = OscillatorType

interface ToneSpec {
  attack?: number
  dur: number
  freq: number
  gain?: number
  start?: number
  type?: OscType
}

let ctx: AudioContext | null = null

function getCtx(): AudioContext | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    if (!ctx) {
      const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext

      if (!Ctor) {
        return null
      }

      ctx = new Ctor()
    }

    // Autoplay policies can leave the context suspended until a gesture; a
    // resume() here recovers it once the user has interacted with the window.
    if (ctx.state === 'suspended') {
      void ctx.resume().catch(() => undefined)
    }

    return ctx
  } catch {
    return null
  }
}

// One enveloped oscillator voice → master. Linear attack into an exponential
// decay keeps the tail smooth and avoids the click you get ramping to zero.
function voice(ac: AudioContext, master: GainNode, t0: number, spec: ToneSpec) {
  const osc = ac.createOscillator()
  const env = ac.createGain()
  const start = t0 + (spec.start ?? 0)
  const peak = spec.gain ?? 0.5
  const attack = spec.attack ?? 0.006
  const end = start + spec.dur

  osc.type = spec.type ?? 'sine'
  osc.frequency.setValueAtTime(spec.freq, start)

  env.gain.setValueAtTime(0.0001, start)
  env.gain.exponentialRampToValueAtTime(Math.max(peak, 0.0002), start + attack)
  env.gain.exponentialRampToValueAtTime(0.0001, end)

  osc.connect(env)
  env.connect(master)
  osc.start(start)
  osc.stop(end + 0.02)
}

let reverbImpulse: AudioBuffer | null = null

// A short, synthetic reverb tail (decaying noise impulse). Used as a subtle wet
// send so the chimes feel like they sit in a room rather than a tin can — the
// detail that nudges them from "arcade beep" toward "polished app". The impulse
// buffer is generated once and cached; each play gets a fresh, disposable
// convolver so connections never pile up on a shared node.
function makeReverb(ac: AudioContext): ConvolverNode {
  if (!reverbImpulse) {
    const seconds = 1.1
    const length = Math.floor(ac.sampleRate * seconds)
    reverbImpulse = ac.createBuffer(2, length, ac.sampleRate)

    for (let channel = 0; channel < 2; channel += 1) {
      const data = reverbImpulse.getChannelData(channel)

      for (let i = 0; i < length; i += 1) {
        // White noise with a steep exponential decay → smooth, short tail.
        data[i] = (Math.random() * 2 - 1) * (1 - i / length) ** 3.2
      }
    }
  }

  const convolver = ac.createConvolver()
  convolver.buffer = reverbImpulse

  return convolver
}

export interface CompletionSoundVariant {
  id: number
  name: string
  // `master` is warm (runs through low-pass + room tail).
  play: (ac: AudioContext, master: GainNode, t0: number) => void
}

// Note frequencies (equal temperament). Everything lives in a low-mid register
// (C3–C5) so the chimes feel warm and "appy" rather than bright and arcade-y.
const C3 = 130.81
const C4 = 261.63
const E4 = 329.63
const G4 = 392
const C5 = 523.25
const D5 = 587.33
const E5 = 659.25
const G5 = 783.99
const C6 = 1046.5

export const COMPLETION_SOUND_VARIANTS: readonly CompletionSoundVariant[] = [
  {
    id: 1,
    name: 'Tiks success (MIT)',
    play: (ac, master, t0) => {
      // Ported from rexa-developer/tiks success(): tonic then fifth.
      voice(ac, master, t0, { freq: C5, dur: 0.11, gain: 0.12, attack: 0.008, type: 'sine' })
      voice(ac, master, t0 + 0.085, { freq: G5, dur: 0.16, gain: 0.12, attack: 0.008, type: 'sine' })
    }
  },
  {
    id: 2,
    name: 'Seslen message (MIT)',
    play: (ac, master, t0) => {
      // Ported from productdevbook/seslen message(): soft two-tone bell.
      voice(ac, master, t0, { freq: 880, dur: 0.28, gain: 0.1, attack: 0.01, type: 'sine' })
      voice(ac, master, t0 + 0.08, { freq: 1320, dur: 0.34, gain: 0.085, attack: 0.01, type: 'sine' })
    }
  },
  {
    id: 3,
    name: 'Seslen success chirp (MIT)',
    play: (ac, master, t0) => {
      // Ported from productdevbook/seslen success(): 660→990→1320 triangle.
      const osc = ac.createOscillator()
      const env = ac.createGain()
      osc.type = 'triangle'
      osc.frequency.setValueAtTime(660, t0)
      osc.frequency.linearRampToValueAtTime(990, t0 + 0.08)
      osc.frequency.linearRampToValueAtTime(1320, t0 + 0.18)
      env.gain.setValueAtTime(0.0001, t0)
      env.gain.linearRampToValueAtTime(0.11, t0 + 0.01)
      env.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.32)
      osc.connect(env)
      env.connect(master)
      osc.start(t0)
      osc.stop(t0 + 0.34)
    }
  },
  {
    id: 4,
    name: 'Tiks notify (MIT)',
    play: (ac, master, t0) => {
      // Ported from rexa-developer/tiks notify(): two-note rising figure.
      voice(ac, master, t0, { freq: 880, dur: 0.18, gain: 0.1, attack: 0.008, type: 'sine' })
      voice(ac, master, t0 + 0.1, { freq: 1320, dur: 0.3, gain: 0.1, attack: 0.008, type: 'sine' })
    }
  },
  {
    id: 5,
    name: 'Seslen notify (MIT)',
    play: (ac, master, t0) => {
      // Ported from productdevbook/seslen notify(): 660-880-1320 sequence.
      const notes = [660, 880, 1320]
      notes.forEach((frequency, i) => {
        const start = t0 + i * 0.1
        voice(ac, master, start, { freq: frequency, dur: 0.16, gain: 0.095, attack: 0.01, type: 'sine' })
      })
    }
  },
  {
    id: 6,
    name: 'Seslen victory arpeggio (MIT)',
    play: (ac, master, t0) => {
      const notes = [C5, E5, G5, C6]
      notes.forEach((frequency, i) => {
        voice(ac, master, t0 + i * 0.09, { freq: frequency, dur: 0.24, gain: 0.12, attack: 0.008, type: 'triangle' })
      })
    }
  },
  {
    id: 7,
    name: 'Seslen level-up arpeggio (MIT)',
    play: (ac, master, t0) => {
      const notes = [C5, D5, E5, G5, C6]
      notes.forEach((frequency, i) => {
        voice(ac, master, t0 + i * 0.09, { freq: frequency, dur: 0.22, gain: 0.11, attack: 0.008, type: 'triangle' })
      })
    }
  },
  {
    id: 8,
    name: 'Two-note comfort (minimal)',
    play: (ac, master, t0) => {
      voice(ac, master, t0, { freq: E4, dur: 0.22, gain: 0.05, attack: 0.03, type: 'sine' })
      voice(ac, master, t0 + 0.08, { freq: C4, dur: 0.52, gain: 0.07, attack: 0.08, type: 'sine' })
      voice(ac, master, t0 + 0.08, { freq: C3, dur: 0.46, gain: 0.02, attack: 0.1, type: 'sine' })
    }
  }
] as const

const DEFAULT_COMPLETION_VARIANT_ID = 8

export function playVariant(variantId: number) {
  const variant = COMPLETION_SOUND_VARIANTS.find(v => v.id === variantId)

  if (!variant) {
    return
  }

  const ac = getCtx()

  if (!ac) {
    return
  }

  // Signal path: voices → master → low-pass → (dry + reverb send) → out.
  // The low-pass sits low to keep things warm, and a small wet send adds the
  // sense of space that makes the chime feel like part of a polished app.
  const master = ac.createGain()
  const tone = ac.createBiquadFilter()
  tone.type = 'lowpass'
  tone.frequency.setValueAtTime(3400, ac.currentTime)
  tone.Q.setValueAtTime(0.4, ac.currentTime)
  master.gain.setValueAtTime(0.7, ac.currentTime)
  master.connect(tone)

  const dry = ac.createGain()
  dry.gain.setValueAtTime(0.92, ac.currentTime)
  tone.connect(dry)
  dry.connect(ac.destination)

  const reverb = makeReverb(ac)
  const wet = ac.createGain()
  wet.gain.setValueAtTime(0.22, ac.currentTime)
  tone.connect(reverb)
  reverb.connect(wet)
  wet.connect(ac.destination)

  variant.play(ac, master, ac.currentTime + 0.01)
}

// Plays the fixed completion cue on any `message.complete`.
export function playCompletionSound() {
  if ($hapticsMuted.get()) {
    return
  }

  playVariant(DEFAULT_COMPLETION_VARIANT_ID)
}
