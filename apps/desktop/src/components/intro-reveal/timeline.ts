/**
 * Beat timeline for the intro reveal. One declarative table + continuous
 * curves own the whole sequence; the surface and sound derive from here.
 *
 * v4 arc — the product story, not an abstraction:
 *   ask        — an enlarged pristine composer; a request types itself
 *   send       — the message commits
 *   working    — real tool activity materializes (browser, terminal, cron)
 *   reply      — the agent's answer streams in
 *   everywhere — the chat becomes one of several live agents, on every surface
 *   brand      — badge + wordmark close
 *   dissolve   — release back to the desktop
 *
 * Times are milliseconds from sequence start — SCORE time, not wall time. The
 * surface divides its clock by `INTRO_PACE` before reading anything here, so
 * every schedule in the piece (beats, typing, streaming, tool rows, the cube's
 * materials, rotation and glitch) stretches together off one number. Retiming
 * the cinematic means moving that knob, never re-deriving the tables.
 *
 * Anything on a REAL timer — the deadman, the watchdogs, the conductor's
 * backstop — measures `INTRO_WALL_MS` instead.
 */

/** Playback rate for the score. >1 plays slower. */
export const INTRO_PACE = 1.25

export interface IntroBeat {
  /** Sound cue fired when the beat lands. */
  cue?: 'latch' | 'resolve' | 'swell' | 'tick'
  /** Stable id for tests + the surface's scene switches. */
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

/** How long the piece actually takes to watch. */
export const INTRO_WALL_MS = Math.round(INTRO_TOTAL_MS * INTRO_PACE)

/** Exit dissolve window after the sequence — kept in one place so the surface
 *  fade and the window self-close agree. A real CSS transition, so it is wall
 *  time and the pace does not touch it. */
export const INTRO_EXIT_MS = 900

/** Overlay deadman margin: the surface force-closes its own window this long
 *  after the nominal end even if the clock stalls, and the main process holds
 *  an independent watchdog above that. The screen ALWAYS comes back. */
export const INTRO_DEADMAN_MS = INTRO_WALL_MS + 4000

/** Continuous curves sampled every frame (0..1, smooth, pure). */
export function sampleCurves(t: number): { glow: number; scatter: number } {
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

// ── The demo script (deterministic typing + streaming) ──────────────────────

export const INTRO_PROMPT = 'Model a hero cube in Blender and cycle it through some materials'

export const INTRO_REPLY_WORDS =
  'Done — materials compiled and previewed on the cube. Want a turntable render exported?'.split(' ')

/** Tool activity rows that materialize during `working`. `doneAt` flips the
 *  trailing status from running to the check state. Times are absolute
 *  sequence ms so the whole piece stays on one clock. */
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

/** Per-character reveal times for the typed prompt: human cadence (variable
 *  inter-key delays, tiny pauses after spaces), deterministic via a seeded
 *  LCG so every run is identical and there is nothing to jitter. */
export function typingSchedule(text: string, startMs: number, endMs: number): number[] {
  let seed = 1337

  const rand = () => {
    seed = (seed * 48271) % 2147483647

    return seed / 2147483647
  }

  const weights = Array.from(text, ch => {
    const base = 1 + rand() * 1.1

    // Breathe after word boundaries; hesitate slightly on punctuation.
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

/** Word reveal times for the streaming reply — front-loaded like real token
 *  streaming (fast burst, gentle tail). */
export function streamingSchedule(wordCount: number, startMs: number, endMs: number): number[] {
  const times: number[] = []
  const span = endMs - startMs

  for (let i = 0; i < wordCount; i += 1) {
    const f = (i + 1) / wordCount

    // easeOutQuad on the index → early words arrive quicker.
    times.push(startMs + (1 - (1 - f) * (1 - f)) * span)
  }

  return times
}

/** Beats that land within (prevT, t] — used to fire sound cues exactly once
 *  even when rAF cadence is irregular. Pass prevT = -1 on the first frame so
 *  the t=0 beat fires. */
export function beatsBetween(prevT: number, t: number): IntroBeat[] {
  return INTRO_BEATS.filter(b => b.t > prevT && b.t <= t)
}
