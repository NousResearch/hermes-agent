/** Score time stretches animation and sound together through INTRO_PACE.
 * Real timers use wall time so recovery still works if the frame clock stalls. */

/** Playback rate for the score. >1 plays slower. */
export const INTRO_PACE = 1.25

export interface IntroBeat {
  cue?: 'latch' | 'resolve' | 'swell' | 'tick'
  id: string
  /** ms from sequence start. */
  t: number
}

export const INTRO_BEATS: IntroBeat[] = [
  { id: 'ask', cue: 'tick', t: 0 },
  { id: 'send', cue: 'tick', t: 3900 },
  { id: 'working', cue: 'swell', t: 4500 },
  { id: 'reply', cue: 'latch', t: 7600 },
  { id: 'everywhere', t: 10600 },
  { id: 'brand', cue: 'resolve', t: 13600 },
  { id: 'dissolve', t: 16400 }
]

export const INTRO_TOTAL_MS = 17600

export const INTRO_WALL_MS = Math.round(INTRO_TOTAL_MS * INTRO_PACE)

/** Exit dissolve window after the sequence. index.tsx uses it as a wall-time delay before closing the overlay.
 *  use-intro-clock.ts adds it to score time for the brand fade and the loop cutoff, so INTRO_PACE scales it there. */
export const INTRO_EXIT_MS = 900

/** Overlay deadman margin: the surface force-closes its own window this long after the nominal end, even if
 *  the clock stalls. The main process holds an independent watchdog above this. */
export const INTRO_DEADMAN_MS = INTRO_WALL_MS + 4000

export function sampleCurves(t: number) {
  const ramp = (from: number, to: number) => {
    if (to <= from) {
      return t >= to ? 1 : 0
    }

    const f = Math.min(1, Math.max(0, (t - from) / (to - from)))

    return f * f * (3 - 2 * f)
  }

  const brandT = INTRO_BEATS.find(b => b.id === 'brand')!.t
  const dissolveT = INTRO_BEATS.find(b => b.id === 'dissolve')!.t

  return {
    glow: ramp(brandT - 300, brandT + 1400) * (1 - ramp(dissolveT, dissolveT + 900)),
    scatter: ramp(dissolveT, dissolveT + 1000)
  }
}

export const INTRO_PROMPT = 'Model a hero cube in Blender and cycle it through some materials'

export const INTRO_REPLY_WORDS =
  'Done — materials compiled and previewed on the cube. Want a turntable render exported?'.split(' ')

/** Tool activity rows that appear during the `working` beat. `doneAt` changes the trailing status from
 *  running to done. Times are absolute sequence ms, so the whole piece stays on one clock. */
export interface IntroToolRow {
  at: number
  doneAt: number
  doneText: string
  icon: 'browser' | 'cron' | 'terminal'
  label: string
  runningText: string
}

export const INTRO_TOOL_ROWS: IntroToolRow[] = [
  {
    at: 4700,
    doneAt: 6100,
    doneText: 'scene linked',
    icon: 'browser',
    label: 'blender-mcp',
    runningText: 'connecting to Blender…'
  },
  {
    at: 5350,
    doneAt: 6800,
    doneText: 'metal · rough 0.2',
    icon: 'terminal',
    label: 'metal',
    runningText: 'compiling metal…'
  },
  {
    at: 6000,
    doneAt: 7300,
    doneText: 'glass · ior 1.45',
    icon: 'cron',
    label: 'glass',
    runningText: 'compiling glass…'
  }
]

/** Per-character reveal times for the typed prompt. Delays vary between keys, with longer pauses after
 *  spaces and punctuation, and come from a seeded LCG so every run is identical. */
export function typingSchedule(text: string, startMs: number, endMs: number): number[] {
  let seed = 1337

  const rand = () => {
    seed = (seed * 48271) % 2147483647

    return seed / 2147483647
  }

  const weights = Array.from(text, ch => {
    const base = 1 + rand() * 1.1

    // The extra weight on a space or a punctuation mark is the delay before that character appears.
    if (ch === ' ') {
      return base + 0.9
    }

    if (/[,.!?]/.test(ch)) {
      return base + 1.4
    }

    return base
  })

  const total = weights.reduce((a, b) => a + b, 0)
  const span = endMs - startMs
  const times: number[] = []
  let acc = 0

  for (const w of weights) {
    acc += w
    times.push(startMs + (acc / total) * span)
  }

  return times
}

/** Word reveal times for the streaming reply. easeOutQuad on the index, so early words arrive quicker than
 *  late ones. */
export function streamingSchedule(wordCount: number, startMs: number, endMs: number): number[] {
  const times: number[] = []
  const span = endMs - startMs

  for (let i = 0; i < wordCount; i += 1) {
    const f = (i + 1) / wordCount

    times.push(startMs + (1 - (1 - f) * (1 - f)) * span)
  }

  return times
}

/** Beats in the half-open range (prevT, t]. Callers fire each sound cue once even when the rAF cadence is
 *  irregular. Pass prevT = -1 on the first frame so the beat at t = 0 fires. */
export function beatsBetween(prevT: number, t: number): IntroBeat[] {
  return INTRO_BEATS.filter(b => b.t > prevT && b.t <= t)
}
