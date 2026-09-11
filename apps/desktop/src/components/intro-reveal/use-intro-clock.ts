import { useCallback, useEffect, useRef, useState } from 'react'

import { playLatch, playResolve, playSwell, playTick, startPad } from './sound'
import {
  beatsBetween,
  INTRO_BEATS,
  INTRO_DEADMAN_MS,
  INTRO_EXIT_MS,
  INTRO_PACE,
  INTRO_PROMPT,
  INTRO_REPLY_WORDS,
  INTRO_TOOL_ROWS,
  INTRO_TOTAL_MS,
  type IntroBeat,
  sampleCurves,
  streamingSchedule,
  typingSchedule
} from './timeline'
import { drawViewport, VIEWPORT_END_MS } from './viewport-cube'

const SOUND_CUES = {
  tick: (beat: string) => playTick(beat === 'send' ? 1.35 : 1),
  swell: playSwell,
  latch: playLatch,
  resolve: playResolve
} satisfies Record<NonNullable<IntroBeat['cue']>, (beat: string) => void>

const INTRO_BEAT_INDEX: Record<string, number> = Object.fromEntries(INTRO_BEATS.map((b, i) => [b.id, i]))

const SEND_T = INTRO_BEATS.find(b => b.id === 'send')!.t
const REPLY_T = INTRO_BEATS.find(b => b.id === 'reply')!.t
const EVERYWHERE_T = INTRO_BEATS.find(b => b.id === 'everywhere')!.t
const BRAND_T = INTRO_BEATS.find(b => b.id === 'brand')!.t
const TYPE_TIMES = typingSchedule(INTRO_PROMPT, 700, SEND_T - 450)
const WORD_TIMES = streamingSchedule(INTRO_REPLY_WORDS.length, REPLY_T + 150, REPLY_T + 2400)

interface Frame {
  beat: number
  replyWords: number
  /** 45ms quantized clock — drives braille spinners + scramble decodes. */
  tick: number
  toolDone: number // bitmask
  toolShown: number // bitmask
  typed: number
}

const INITIAL_FRAME: Frame = { beat: 0, replyWords: 0, tick: 0, toolDone: 0, toolShown: 0, typed: 0 }

function frameAt(t: number, beat: number): Frame {
  let typed = 0

  while (typed < TYPE_TIMES.length && TYPE_TIMES[typed] <= t) {
    typed += 1
  }

  let replyWords = 0

  while (replyWords < WORD_TIMES.length && WORD_TIMES[replyWords] <= t) {
    replyWords += 1
  }

  let toolShown = 0
  let toolDone = 0

  for (let i = 0; i < INTRO_TOOL_ROWS.length; i += 1) {
    if (t >= INTRO_TOOL_ROWS[i].at) {
      toolShown |= 1 << i
    }

    if (t >= INTRO_TOOL_ROWS[i].doneAt) {
      toolDone |= 1 << i
    }
  }

  return { beat, replyWords, tick: Math.floor(t / 45), toolDone, toolShown, typed }
}

export function useIntroClock(onSkip?: () => void) {
  const glowRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const brandRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = useState<Frame>(INITIAL_FRAME)
  const [clockLeaving, setClockLeaving] = useState(false)
  const [faded, setFaded] = useState(false)
  const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

  const skip = useCallback(() => {
    setClockLeaving(true)

    if (onSkip) {
      onSkip()

      return
    }

    window.hermesDesktop?.introReveal?.skip?.()
    window.setTimeout(() => {
      void window.hermesDesktop?.introReveal?.close?.({ showMain: true }).catch(() => undefined)
    }, 1200)
  }, [onSkip])

  useEffect(() => {
    const id = window.setTimeout(() => {
      void window.hermesDesktop?.introReveal?.close?.({ showMain: true }).catch(() => undefined)
    }, INTRO_DEADMAN_MS)

    return () => window.clearTimeout(id)
  }, [])

  // The native window keeps the clock running while the main app is hidden.
  useEffect(() => {
    if (reduceMotion) {
      setFrame({ ...INITIAL_FRAME, beat: INTRO_BEAT_INDEX.brand })

      // This branch has no frame loop to reveal the brand or hide the demo.
      for (const element of [glowRef.current, brandRef.current]) {
        if (element) {
          element.style.opacity = '1'
        }
      }

      if (stageRef.current) {
        stageRef.current.style.opacity = '0'
      }

      playLatch()

      const id = window.setTimeout(() => {
        skip()
      }, 2600)

      return () => window.clearTimeout(id)
    }

    const pad = startPad()
    const start = performance.now()
    let prevT = -1
    // `start` is wall time; everything downstream of `elapsed` is score time.
    // Dividing once, here, is what makes the whole piece — beats, schedules,
    // the cube's rotation and tear — play at INTRO_PACE with nothing else to
    // keep in step.
    const elapsed = () => (performance.now() - start) / INTRO_PACE
    let raf = 0
    let currentBeat = 0
    let reportedDone = false
    let lastFrameKey = ''

    const tick = () => {
      const t = elapsed()

      for (const b of beatsBetween(prevT, t)) {
        currentBeat = INTRO_BEAT_INDEX[b.id] ?? currentBeat

        if (b.cue) {
          SOUND_CUES[b.cue](b.id)
        }
      }

      prevT = t

      drawViewportFrame(viewportRef.current, t)

      const next = frameAt(t, currentBeat)
      const key = `${next.beat}:${next.typed}:${next.replyWords}:${next.toolShown}:${next.toolDone}`

      if (key !== lastFrameKey) {
        lastFrameKey = key
        setFrame(next)
      }

      const curves = sampleCurves(t)

      pad.setLevel(Math.max(curves.glow, next.beat >= INTRO_BEAT_INDEX.working ? 0.45 : 0.2))

      const ss = (from: number, to: number) => {
        const f = Math.min(1, Math.max(0, (t - from) / (to - from)))

        return f * f * (3 - 2 * f)
      }

      // The stage never sits still: a slow drift-up across the whole piece,
      // a gentle scale breath, and a lateral ease as the constellation opens
      // (hero sits slightly left once the side agents arrive — asymmetric,
      // not centered). All one transform, compositor-only.
      if (stageRef.current) {
        const rise = -10 - ss(0, INTRO_TOTAL_MS) * 26
        const breathe = 1 + Math.sin(t / 2600) * 0.004
        const openScale = 1 - ss(EVERYWHERE_T - 600, EVERYWHERE_T + 1200) * 0.06
        const lateral = ss(EVERYWHERE_T - 600, EVERYWHERE_T + 1400) * -18
        const brandPush = ss(BRAND_T - 300, BRAND_T + 1200)

        stageRef.current.style.transform = `translate(${lateral}px, ${rise + brandPush * -14}px) scale(${breathe * openScale * (1 - brandPush * 0.05)})`
        stageRef.current.style.opacity = String(1 - brandPush)
      }

      // The ENTIRE brand close (glow + badge + wordmark + tagline) rides ONE
      // alpha so nothing is ever readable against a half-faded bloom. It
      // rises in with the glow and the whole group breathes out together
      // through the exit window.
      const brandIn = ss(BRAND_T - 200, BRAND_T + 1300)
      const brandOut = 1 - ss(INTRO_TOTAL_MS - 500, INTRO_TOTAL_MS + INTRO_EXIT_MS - 100)
      const brandAlpha = brandIn * brandOut

      if (glowRef.current) {
        glowRef.current.style.opacity = String(brandAlpha)
        glowRef.current.style.transform = `translate(-50%, -30%) scale(${0.9 + brandIn * 0.14})`
      }

      if (brandRef.current) {
        brandRef.current.style.opacity = String(brandAlpha)
        brandRef.current.style.transform = `translateY(${(1 - brandIn) * 26 - brandIn * 6}px) scale(${0.94 + brandIn * 0.06})`
      }

      if (t >= INTRO_TOTAL_MS && !reportedDone) {
        reportedDone = true
        skip()
      }

      if (t < INTRO_TOTAL_MS + INTRO_EXIT_MS) {
        raf = requestAnimationFrame(tick)
      }
    }

    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      pad.stop()
    }
  }, [reduceMotion, skip])

  // ── Esc to skip (local — never depends on the main renderer). ───────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        skip()
      }
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [skip])

  useEffect(() => {
    const id = requestAnimationFrame(() => setFaded(true))

    return () => cancelAnimationFrame(id)
  }, [])

  return { frame, leaving: clockLeaving, faded, skip, glowRef, stageRef, brandRef, viewportRef }
}

function drawViewportFrame(canvas: HTMLCanvasElement | null, t: number) {
  if (canvas && t < VIEWPORT_END_MS) {
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const cw = canvas.clientWidth
    const ch = canvas.clientHeight

    if (cw > 0 && ch > 0) {
      if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
        canvas.width = cw * dpr
        canvas.height = ch * dpr
      }

      const ctx2d = canvas.getContext('2d')

      if (ctx2d) {
        ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0)
        drawViewport(ctx2d, cw, ch, t)
      }
    }
  }
}
