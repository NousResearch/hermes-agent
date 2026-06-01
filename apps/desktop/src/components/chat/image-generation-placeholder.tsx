import { type FC, useCallback, useEffect, useRef } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'

type Rgb = { r: number; g: number; b: number }

const RAMP = ' .,:;-=+*#%@'

const FALLBACKS = {
  card: { r: 255, g: 255, b: 255 },
  muted: { r: 240, g: 240, b: 239 },
  foreground: { r: 36, g: 36, b: 36 },
  primary: { r: 207, g: 128, b: 109 },
  ring: { r: 185, g: 121, b: 105 }
} satisfies Record<string, Rgb>

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const smoothstep = (edge0: number, edge1: number, value: number) => {
  const t = clamp((value - edge0) / (edge1 - edge0), 0, 1)

  return t * t * (3 - 2 * t)
}

const parseColor = (value: string, fallback: Rgb): Rgb => {
  const hex = value.trim().match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i)

  if (hex) {
    return {
      r: Number.parseInt(hex[1], 16),
      g: Number.parseInt(hex[2], 16),
      b: Number.parseInt(hex[3], 16)
    }
  }

  const rgb = value.trim().match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/i)

  return rgb ? { r: Number(rgb[1]), g: Number(rgb[2]), b: Number(rgb[3]) } : fallback
}

const mix = (a: Rgb, b: Rgb, amount: number): Rgb => ({
  r: Math.round(a.r + (b.r - a.r) * amount),
  g: Math.round(a.g + (b.g - a.g) * amount),
  b: Math.round(a.b + (b.b - a.b) * amount)
})

const rgba = ({ r, g, b }: Rgb, alpha: number) => `rgba(${r}, ${g}, ${b}, ${alpha})`

const hash2 = (x: number, y: number) => {
  const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453

  return n - Math.floor(n)
}

const noise2 = (x: number, y: number) => {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const xf = x - xi
  const yf = y - yi
  const u = xf * xf * (3 - 2 * xf)
  const v = yf * yf * (3 - 2 * yf)
  const a = hash2(xi, yi)
  const b = hash2(xi + 1, yi)
  const c = hash2(xi, yi + 1)
  const d = hash2(xi + 1, yi + 1)

  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

const fbm = (x: number, y: number) => {
  let value = 0
  let amplitude = 0.5
  let frequency = 1

  for (let i = 0; i < 4; i += 1) {
    value += amplitude * noise2(x * frequency, y * frequency)
    frequency *= 2.04
    amplitude *= 0.52
  }

  return value
}

const readTheme = () => {
  const styles = getComputedStyle(document.documentElement)

  return {
    card: parseColor(styles.getPropertyValue('--dt-card'), FALLBACKS.card),
    muted: parseColor(styles.getPropertyValue('--dt-muted'), FALLBACKS.muted),
    foreground: parseColor(styles.getPropertyValue('--dt-foreground'), FALLBACKS.foreground),
    primary: parseColor(styles.getPropertyValue('--dt-primary'), FALLBACKS.primary),
    ring: parseColor(styles.getPropertyValue('--dt-ring'), FALLBACKS.ring)
  }
}

const fitCanvas = (canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) => {
  const rect = canvas.getBoundingClientRect()
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const width = Math.max(1, rect.width)
  const height = Math.max(1, rect.height)

  canvas.width = Math.round(width * dpr)
  canvas.height = Math.round(height * dpr)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  return { width, height }
}

const drawAsciiDiffusion = (ctx: CanvasRenderingContext2D, width: number, height: number, time: number) => {
  const theme = readTheme()
  const bg = ctx.createLinearGradient(0, 0, width, height)
  bg.addColorStop(0, rgba(mix(theme.card, theme.primary, 0.08), 1))
  bg.addColorStop(0.54, rgba(mix(theme.card, theme.muted, 0.68), 1))
  bg.addColorStop(1, rgba(mix(theme.muted, theme.ring, 0.12), 1))
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, width, height)

  const cycle = (time * 0.028) % 1

  const denoise = cycle < 0.82 ? smoothstep(0.02, 0.82, cycle) : 1 - smoothstep(0.82, 1, cycle)

  const fontSize = clamp(width / 58, 8, 13)
  const cellWidth = fontSize * 0.78
  const cellHeight = fontSize * 1.28
  const cols = Math.ceil(width / cellWidth)
  const rows = Math.ceil(height / cellHeight)
  const centerX = 0.53 + Math.sin(time * 0.055) * 0.02
  const centerY = 0.5 + Math.cos(time * 0.048) * 0.02
  const timestep = Math.floor(time * 1.15)
  const timestepBlend = smoothstep(0, 1, time * 1.15 - timestep)

  ctx.font = `${fontSize}px "SF Mono", "Cascadia Code", Menlo, Consolas, monospace`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  for (let row = -1; row <= rows + 1; row += 1) {
    for (let col = -1; col <= cols + 1; col += 1) {
      const x = col * cellWidth + cellWidth * 0.5
      const y = row * cellHeight + cellHeight * 0.5
      const nx = x / width
      const ny = y / height
      const dx = (nx - centerX) * 1.2
      const dy = (ny - centerY) * 0.95
      const radius = Math.hypot(dx, dy)
      const angle = Math.atan2(dy, dx)

      const bloom =
        Math.exp(-(radius * radius) / 0.075) * 0.72 +
        Math.exp(-((radius - (0.28 + Math.sin(angle * 5 + time * 0.16) * 0.035)) ** 2) / 0.0028) * 0.8

      const contour =
        Math.exp(-((Math.sin(angle * 3 + radius * 17 - time * 0.17) * 0.5 + 0.5 - radius) ** 2) / 0.016) * 0.38

      const stem = Math.exp(-((nx - centerX + 0.05) ** 2 / 0.004 + (ny - centerY - 0.25) ** 2 / 0.08)) * 0.46

      const latent = clamp(bloom + contour + stem, 0, 1)
      const staticA = hash2(col + timestep * 19, row - timestep * 11)

      const staticB = hash2(col + (timestep + 1) * 19, row - (timestep + 1) * 11)

      const staticNoise = staticA + (staticB - staticA) * timestepBlend
      const livingNoise = fbm(col * 0.12 + time * 0.024, row * 0.12 - time * 0.018)
      const denoiseWave = Math.exp(-((radius - denoise * 0.62) ** 2) / 0.006)

      const signal = clamp(
        staticNoise * (1 - denoise) +
          latent * denoise +
          (livingNoise - 0.45) * (0.45 - denoise * 0.26) +
          denoiseWave * 0.3,
        0,
        1
      )

      const dropoutA = hash2(col - timestep * 7, row + timestep * 13)

      const dropoutB = hash2(col - (timestep + 1) * 7, row + (timestep + 1) * 13)

      const dropout = dropoutA + (dropoutB - dropoutA) * timestepBlend

      if (dropout > 0.35 + signal * 0.68) {
        continue
      }

      const glyph = RAMP[clamp(Math.floor(signal * (RAMP.length - 1)), 0, RAMP.length - 1)]

      if (glyph === ' ') {
        continue
      }

      const jitter = (1 - denoise) * 1.35 + (1 - latent) * 0.45
      const jx = (noise2(col * 0.31, row * 0.31 + time * 0.09) - 0.5) * jitter
      const jy = (noise2(col * 0.27 - time * 0.085, row * 0.27) - 0.5) * jitter
      const tintAmount = clamp(latent * 0.7 + denoiseWave * 0.4, 0, 1)
      const warm = mix(theme.primary, theme.ring, hash2(col, row))
      const tint = mix(theme.foreground, warm, tintAmount)
      const alpha = clamp(0.12 + signal * 0.68 + denoiseWave * 0.16, 0, 0.86)

      if (signal > 0.58 && denoise > 0.34) {
        ctx.fillStyle = rgba(theme.ring, alpha * 0.2)
        ctx.fillText(glyph, x + jx + 0.75, y + jy - 0.45)
        ctx.fillStyle = rgba(theme.primary, alpha * 0.18)
        ctx.fillText(glyph, x + jx - 0.75, y + jy + 0.45)
      }

      ctx.fillStyle = rgba(tint, alpha)
      ctx.fillText(glyph, x + jx, y + jy)
    }
  }

  const veil = ctx.createRadialGradient(
    width * centerX,
    height * centerY,
    0,
    width * centerX,
    height * centerY,
    Math.min(width, height) * (0.35 + denoise * 0.3)
  )

  veil.addColorStop(0, rgba(theme.card, 0.08 + denoise * 0.12))
  veil.addColorStop(0.52, rgba(theme.card, 0.05))
  veil.addColorStop(1, rgba(theme.card, 0))
  ctx.fillStyle = veil
  ctx.fillRect(0, 0, width, height)
}

const DiffusionCanvas: FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const sizeRef = useRef({ width: 0, height: 0 })

  const fitToContainer = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')

    if (!canvas || !ctx) {
      return
    }

    sizeRef.current = fitCanvas(canvas, ctx)
  }, [])

  useResizeObserver(fitToContainer, canvasRef)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')

    if (!canvas || !ctx) {
      return
    }

    sizeRef.current = fitCanvas(canvas, ctx)

    let frame = requestAnimationFrame(function draw(now) {
      const { width, height } = sizeRef.current
      ctx.clearRect(0, 0, width, height)
      drawAsciiDiffusion(ctx, width, height, now / 1000)
      frame = requestAnimationFrame(draw)
    })

    return () => {
      cancelAnimationFrame(frame)
    }
  }, [])

  return <canvas className="absolute inset-0 h-full w-full" ref={canvasRef} />
}

export const ImageGenerationPlaceholder: FC = () => {
  return (
    <div aria-label="Rendering image" aria-live="polite" className="w-full max-w-136 self-start" role="status">
      <div className="relative h-(--image-preview-height) overflow-hidden rounded-4xl border border-border/55 shadow-[inset_0_0.0625rem_0_color-mix(in_srgb,white_45%,transparent),inset_0_0_0_0.0625rem_color-mix(in_srgb,var(--dt-border)_34%,transparent),inset_0_-0.75rem_1.75rem_color-mix(in_srgb,var(--dt-primary)_5%,transparent)]">
        <DiffusionCanvas />
      </div>
    </div>
  )
}
