/**
 * Synthesized sound bed for the intro reveal — no audio assets.
 *
 * Musical, not textural: a slow pad in D (D3+A3+D4) swells under the assemble
 * phase, particle beats land as soft plucked fifths, the mark locking is a
 * warm two-voice latch, and the wordmark resolves on a rising triad arp with a
 * long tail. Everything sits well under system volume — a score, not a sting.
 *
 * All scheduling is beat-driven from the sequence timeline so sound, type, and
 * particles share one clock. Autoplay-safe: the context resumes lazily and
 * every call is a no-op when audio is unavailable or muted.
 */

import { $hapticsMuted } from '@/store/haptics'

let ctx: AudioContext | null = null
let master: GainNode | null = null

function getCtx(): AudioContext | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    if (!ctx) {
      const Ctor =
        window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext

      if (!Ctor) {
        return null
      }

      ctx = new Ctor()
      master = ctx.createGain()
      master.gain.value = 0.55
      master.connect(ctx.destination)
    }

    if (ctx.state === 'suspended') {
      void ctx.resume().catch(() => undefined)
    }

    return ctx
  } catch {
    return null
  }
}

function muted(): boolean {
  try {
    return $hapticsMuted.get()
  } catch {
    return false
  }
}

function env(g: GainNode, t0: number, peak: number, attack: number, decay: number): void {
  g.gain.setValueAtTime(0.0001, t0)
  g.gain.exponentialRampToValueAtTime(Math.max(peak, 0.0002), t0 + attack)
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + attack + decay)
}

/** The pad: three detuned triangle voices on a D root (D3, A3, D4) behind a
 *  gentle lowpass. `setLevel` follows the choreography — the chord swells as
 *  the streams flow and holds warm under the formed badge. */
export function startPad(): { setLevel: (v: number) => void; stop: () => void } {
  const ac = getCtx()

  if (!ac || !master || muted()) {
    return { setLevel: () => undefined, stop: () => undefined }
  }

  const lp = ac.createBiquadFilter()

  lp.type = 'lowpass'
  lp.frequency.value = 600
  lp.Q.value = 0.4

  const g = ac.createGain()

  g.gain.value = 0

  lp.connect(g)
  g.connect(master)

  // D3 / A3 / D4 with slight detune per voice for width.
  const oscs: OscillatorNode[] = []

  for (const [freq, detune, gain] of [
    [146.83, -4, 0.5],
    [220.0, 3, 0.35],
    [293.66, -2, 0.28]
  ] as const) {
    const osc = ac.createOscillator()
    const vg = ac.createGain()

    osc.type = 'triangle'
    osc.frequency.value = freq
    osc.detune.value = detune
    vg.gain.value = gain
    osc.connect(vg)
    vg.connect(lp)
    osc.start()
    oscs.push(osc)
  }

  return {
    setLevel: v => {
      const t = ac.currentTime

      // The filter opens with the swell so the chord brightens as it rises.
      lp.frequency.cancelScheduledValues(t)
      lp.frequency.setTargetAtTime(600 + v * 900, t, 0.5)
      g.gain.setTargetAtTime(v * 0.11, t, 0.35)
    },
    stop: () => {
      const t = ac.currentTime

      g.gain.setTargetAtTime(0, t, 0.4)
      window.setTimeout(() => {
        for (const osc of oscs) {
          try {
            osc.stop()
          } catch {
            // already stopped
          }
        }
      }, 1600)
    }
  }
}

/** Soft plucked fifth (D5→A4 glide) for text beats — melodic, dry, brief. */
export function playTick(pitch = 1): void {
  const ac = getCtx()

  if (!ac || !master || muted()) {
    return
  }

  const t0 = ac.currentTime
  const osc = ac.createOscillator()
  const g = ac.createGain()

  osc.type = 'triangle'
  osc.frequency.setValueAtTime(587.33 * pitch, t0)
  osc.frequency.exponentialRampToValueAtTime(440 * pitch, t0 + 0.12)

  env(g, t0, 0.09, 0.004, 0.22)
  osc.connect(g)
  g.connect(master)
  osc.start(t0)
  osc.stop(t0 + 0.3)
}

/** Rising swell for the flow phase — a low A that opens upward as the field
 *  gathers the motes. */
export function playSwell(): void {
  const ac = getCtx()

  if (!ac || !master || muted()) {
    return
  }

  const t0 = ac.currentTime
  const osc = ac.createOscillator()
  const g = ac.createGain()

  osc.type = 'sine'
  osc.frequency.setValueAtTime(110, t0)
  osc.frequency.exponentialRampToValueAtTime(220, t0 + 2.4)

  g.gain.setValueAtTime(0.0001, t0)
  g.gain.exponentialRampToValueAtTime(0.07, t0 + 1.2)
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + 3.2)

  osc.connect(g)
  g.connect(master)
  osc.start(t0)
  osc.stop(t0 + 3.4)
}

/** Warm two-voice latch when the mark locks — root + fifth (D4+A4). */
export function playLatch(): void {
  const ac = getCtx()

  if (!ac || !master || muted()) {
    return
  }

  const t0 = ac.currentTime

  for (const [freq, gain, delay] of [
    [293.66, 0.13, 0],
    [440.0, 0.09, 0.03]
  ] as const) {
    const osc = ac.createOscillator()
    const g = ac.createGain()

    osc.type = 'sine'
    osc.frequency.value = freq
    env(g, t0 + delay, gain, 0.014, 0.7)
    osc.connect(g)
    g.connect(master)
    osc.start(t0 + delay)
    osc.stop(t0 + delay + 0.8)
  }
}

/** Resolving arp for the wordmark — D5, F#5, A5, D6 staggered with long
 *  tails: the major-third resolution the pad has been withholding. */
export function playResolve(): void {
  const ac = getCtx()

  if (!ac || !master || muted()) {
    return
  }

  const t0 = ac.currentTime

  const partials: Array<readonly [number, number, number]> = [
    [587.33, 0.08, 0],
    [739.99, 0.07, 0.1],
    [880.0, 0.06, 0.2],
    [1174.66, 0.05, 0.32]
  ]

  for (const [freq, gain, delay] of partials) {
    const osc = ac.createOscillator()
    const g = ac.createGain()

    osc.type = 'triangle'
    osc.frequency.value = freq
    env(g, t0 + delay, gain, 0.02, 1.1)
    osc.connect(g)
    g.connect(master)
    osc.start(t0 + delay)
    osc.stop(t0 + delay + 1.2)
  }
}
