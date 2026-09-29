import { useEffect, useRef } from 'react'

import { $micLevel, clampMicLevel } from '@/store/voice-level'

/** Bars around the ring — even, so the two halves mirror exactly. */
export const AURA_BARS = 72

/**
 * Per-bar targets for one frame of the ring. Pure, so the shape is testable
 * without a canvas: a measured level swells the whole ring, and a slow wave
 * of `phase` gives it the organic, breathing outline instead of a flat pulse.
 * Silence (level 0) always yields zeros — the ring never fakes activity.
 */
export function auraTargets(level: number, phase: number, bars = AURA_BARS): number[] {
  const energy = clampMicLevel(level)

  return Array.from({ length: bars }, (_, i) => {
    // Mirror around the vertical axis so left and right move together.
    const angle = (Math.min(i, bars - i) / (bars / 2)) * Math.PI
    const wave = 0.55 + 0.45 * Math.sin(angle * 3 + phase) * Math.cos(angle * 2 - phase * 0.7)

    return energy * wave
  })
}

/**
 * The voice made visible: a spectrum ring hugging the orb. Cyan runs into
 * violet round the circle, every bar is driven by the real microphone level,
 * and the ring dissolves when the conversation is closed. It sits over the
 * orb's edge, sized from the core, and draws nothing at all while inactive.
 */
export function VoiceAura({ active, reducedMotion = false }: { active: boolean; reducedMotion?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext?.('2d') ?? null

    if (!canvas || !ctx) {
      return undefined
    }

    const bars = new Array<number>(AURA_BARS).fill(0)
    let opacity = 0
    let phase = 0
    let smoothLevel = 0
    let frame = 0

    const draw = () => {
      const rect = canvas.getBoundingClientRect()
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      const size = Math.max(1, Math.min(rect.width, rect.height))

      if (canvas.width !== Math.round(size * ratio)) {
        canvas.width = Math.round(size * ratio)
        canvas.height = Math.round(size * ratio)
      }

      ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
      ctx.clearRect(0, 0, size, size)

      if (opacity < 0.01) {
        return
      }

      const centre = size / 2
      // The orb's glass edge sits at 38% of the core; the canvas is 1.32× it.
      const inner = size * 0.315
      const reach = size * 0.15

      ctx.lineCap = 'round'
      ctx.lineWidth = Math.max(2, size * 0.008)
      ctx.globalAlpha = opacity

      for (let i = 0; i < AURA_BARS; i += 1) {
        const angle = (i / AURA_BARS) * Math.PI * 2 - Math.PI / 2
        const length = 2 + bars[i]! * reach
        const hue = 188 + 82 * (0.5 - 0.5 * Math.cos((i / AURA_BARS) * Math.PI * 2))
        const cos = Math.cos(angle)
        const sin = Math.sin(angle)

        ctx.strokeStyle = `hsl(${hue}, 92%, ${62 + bars[i]! * 10}%)`
        ctx.shadowColor = `hsla(${hue}, 95%, 60%, ${0.25 + bars[i]! * 0.5})`
        ctx.shadowBlur = 4 + bars[i]! * 10
        ctx.beginPath()
        ctx.moveTo(centre + cos * inner, centre + sin * inner)
        ctx.lineTo(centre + cos * (inner + length), centre + sin * (inner + length))
        ctx.stroke()
      }

      ctx.globalAlpha = 1
      ctx.shadowBlur = 0
    }

    const loop = () => {
      const target = active ? 1 : 0
      opacity += (target - opacity) * 0.12
      smoothLevel += (($micLevel.get() || 0) - smoothLevel) * (reducedMotion ? 0.05 : 0.35)
      phase += 0.03

      const goals = auraTargets(smoothLevel, reducedMotion ? 0 : phase)

      for (let i = 0; i < AURA_BARS; i += 1) {
        bars[i] = bars[i]! + (goals[i]! - bars[i]!) * 0.4
      }

      draw()

      // Once closed and faded out the loop stops; the next open restarts it.
      if (active || opacity > 0.01) {
        frame = requestAnimationFrame(loop)
      } else {
        draw()
      }
    }

    frame = requestAnimationFrame(loop)

    return () => cancelAnimationFrame(frame)
  }, [active, reducedMotion])

  return (
    <canvas
      aria-hidden="true"
      className="jarvis-core__aura"
      data-testid="jarvis-voice-aura"
      ref={canvasRef}
    />
  )
}
