/** Synthesized cues for the intro. The clock in use-intro-clock.ts calls them on the score's beats. */

import { $hapticsMuted } from '@/store/haptics'

interface IntroAudioWindow extends Window {
  webkitAudioContext?: typeof AudioContext
}

interface IntroPad {
  setLevel: (level: number) => void
  stop: () => void
}

let ctx: AudioContext | null = null
let master: GainNode | null = null

function getCtx(): AudioContext | null {
  if (globalThis.window === undefined) {
    return null
  }

  try {
    if (!ctx) {
      const audioWindow: IntroAudioWindow = window
      const Ctor = window.AudioContext || audioWindow.webkitAudioContext

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

function env(g: GainNode, t0: number, peak: number, attack: number, decay: number): void {
  g.gain.setValueAtTime(0.0001, t0)
  g.gain.exponentialRampToValueAtTime(Math.max(peak, 0.0002), t0 + attack)
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + attack + decay)
}

export function startPad(): IntroPad {
  const ac = getCtx()

  if (!ac || !master || $hapticsMuted.get()) {
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

      // The lowpass cutoff rises with the level, so the chord brightens as it swells.
      lp.frequency.cancelScheduledValues(t)
      lp.frequency.setTargetAtTime(600 + v * 900, t, 0.5)
      g.gain.setTargetAtTime(v * 0.11, t, 0.35)
    },
    stop: () => {
      const t = ac.currentTime

      g.gain.setTargetAtTime(0, t, 0.4)
      window.setTimeout(() => {
        for (const osc of oscs) {
          osc.stop()
        }
      }, 1600)
    }
  }
}

export function playTick(pitch = 1): void {
  const ac = getCtx()

  if (!ac || !master || $hapticsMuted.get()) {
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

export function playSwell(): void {
  const ac = getCtx()

  if (!ac || !master || $hapticsMuted.get()) {
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

export function playLatch(): void {
  const ac = getCtx()

  if (!ac || !master || $hapticsMuted.get()) {
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

export function playResolve(): void {
  const ac = getCtx()

  if (!ac || !master || $hapticsMuted.get()) {
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
