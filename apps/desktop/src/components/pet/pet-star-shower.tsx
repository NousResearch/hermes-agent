import { useEffect, useRef } from 'react'

/**
 * Canvas hatch celebration layered over a freshly revealed pet: a one-shot
 * sunburst of rotating god-rays, a fast radial star burst (confetti physics —
 * velocity + decay + gravity + spin), and a light trickle of rising twinkle
 * motes. Additive (`lighter`) so the sparkles bloom. No glow-halo flash.
 *
 * Sized to its container (absolute inset-0, pointer-events: none) and disabled
 * under `prefers-reduced-motion`.
 */

const GOLD = '#ffd76a'
const BURST = 15
const VELOCITY = 500
const DECAY = 0.9
const GRAVITY = 90
const RAY_COUNT = 24
const GOLD_MIX = 0.6
const MOTE_MS = 333 // ~3 / sec

interface Star {
  x: number
  y: number
  vx: number
  vy: number
  size: number
  rot: number
  vrot: number
  phase: number
  twinkle: number
  life: number
  ttl: number
  color: string
  rise: boolean
}

function readAccent(el: HTMLElement): string {
  return getComputedStyle(el).getPropertyValue('--ui-accent').trim() || '#9aa0ff'
}

function sparkle(ctx: CanvasRenderingContext2D, size: number, rot: number, color: string): void {
  ctx.rotate(rot)
  ctx.fillStyle = color
  for (const [rx, ry] of [
    [size, size * 0.26],
    [size * 0.26, size]
  ]) {
    ctx.beginPath()
    ctx.moveTo(0, -ry)
    ctx.lineTo(rx, 0)
    ctx.lineTo(0, ry)
    ctx.lineTo(-rx, 0)
    ctx.closePath()
    ctx.fill()
  }
  const core = Math.max(1, Math.round(size * 0.4))
  ctx.fillStyle = '#fff'
  ctx.fillRect(-core / 2, -core / 2, core, core)
}

export function PetStarShower() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    const parent = canvas?.parentElement
    if (!canvas || !ctx || !parent) {
      return
    }
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      return
    }

    const accent = readAccent(canvas)
    const dpr = Math.min(window.devicePixelRatio || 1, 3)
    let w = 0
    let h = 0
    let cx = 0
    let cy = 0
    const resize = () => {
      const r = parent.getBoundingClientRect()
      w = r.width
      h = r.height
      cx = w / 2
      cy = h * 0.54
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(parent)

    const pick = () => (Math.random() < GOLD_MIX ? GOLD : Math.random() < 0.5 ? accent : '#ffffff')
    const stars: Star[] = []
    for (let i = 0; i < BURST; i++) {
      const a = Math.random() * Math.PI * 2
      const sp = VELOCITY * (0.4 + Math.random() * 0.7)
      stars.push({
        x: cx, y: cy, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
        size: 3.5 + Math.random() * 5.5, rot: Math.random() * 6.28, vrot: (Math.random() - 0.5) * 8,
        phase: 0, twinkle: 0, life: 0, ttl: 0.8 + Math.random() * 0.7, color: pick(), rise: false
      })
    }
    const rays = { life: 0, ttl: 0.9, rot: Math.random() * 6.28 }

    let raf = 0
    let last = performance.now()
    let acc = 0
    let raysAlive = true

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      const ms = now - last
      last = now
      const dt = Math.min(0.05, ms / 1000)
      const decay = Math.pow(DECAY, dt * 60)
      acc += ms
      if (acc >= MOTE_MS && stars.length < 40) {
        acc = 0
        stars.push({
          x: cx + (Math.random() - 0.5) * w * 0.85, y: cy + Math.random() * h * 0.25,
          vx: (Math.random() - 0.5) * 14, vy: -(14 + Math.random() * 26),
          size: 2.5 + Math.random() * 3.5, rot: Math.random() * 6.28, vrot: (Math.random() - 0.5) * 2,
          phase: Math.random() * 6.28, twinkle: 5 + Math.random() * 4, life: 0, ttl: 1.2 + Math.random(),
          color: pick(), rise: true
        })
      }

      ctx.clearRect(0, 0, w, h)
      ctx.globalCompositeOperation = 'lighter'

      // Sunburst god-rays — one-shot bloom + slow spin.
      if (raysAlive) {
        rays.life += dt
        rays.rot += dt * 0.6
        const t = rays.life / rays.ttl
        if (t >= 1) {
          raysAlive = false
        } else {
          const len = Math.max(w, h) * 0.62 * (1 - (1 - t) ** 2)
          ctx.save()
          ctx.translate(cx, cy)
          ctx.rotate(rays.rot)
          for (let i = 0; i < RAY_COUNT; i++) {
            ctx.rotate((Math.PI * 2) / RAY_COUNT)
            const a = (1 - t) * 0.3 * (i % 2 ? 0.65 : 1)
            const wd = len * 0.05
            const g = ctx.createLinearGradient(0, 0, 0, -len)
            g.addColorStop(0, `rgba(255,255,255,${a})`)
            g.addColorStop(1, 'rgba(255,255,255,0)')
            ctx.fillStyle = g
            ctx.beginPath()
            ctx.moveTo(-wd, 0)
            ctx.lineTo(wd, 0)
            ctx.lineTo(0, -len)
            ctx.closePath()
            ctx.fill()
          }
          ctx.restore()
        }
      }

      for (let i = stars.length - 1; i >= 0; i--) {
        const s = stars[i]
        s.life += dt
        if (s.rise) {
          s.vy += 7 * dt
          s.phase += s.twinkle * dt
        } else {
          s.vx *= decay
          s.vy = s.vy * decay + GRAVITY * dt
        }
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.rot += s.vrot * dt
        if (s.life >= s.ttl || s.y < -12) {
          stars.splice(i, 1)
          continue
        }
        const fade = s.rise
          ? Math.min(1, s.life * 5, (s.ttl - s.life) * 3) * (0.45 + 0.55 * Math.abs(Math.sin(s.phase)))
          : Math.min(1, (s.ttl - s.life) * 3)
        ctx.save()
        ctx.globalAlpha = fade
        ctx.translate(Math.round(s.x), Math.round(s.y))
        sparkle(ctx, s.size, s.rot, s.color)
        ctx.restore()
      }

      ctx.globalCompositeOperation = 'source-over'
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  return <canvas className="pointer-events-none absolute inset-0 z-10 h-full w-full" ref={canvasRef} />
}
