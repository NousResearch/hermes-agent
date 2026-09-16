import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import {
  PORTRAIT_CYCLE_MS,
  PORTRAIT_FRAME_MS,
  shapeLuminance,
  warmColor
} from '@/lib/sidebar-portrait-math'
import { $onBattery, batteryPollInterval } from '@/store/power'
import { $sidebarPortrait, ensureSidebarPortrait } from '@/store/sidebar-portrait'
import type { SidebarPortraitData } from '@/global'

function paint(canvas: HTMLCanvasElement, portrait: SidebarPortraitData, elapsed: number) {
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }

  const { width, height, luminance } = portrait
  const image = ctx.createImageData(width, height)

  for (let i = 0; i < luminance.length; i++) {
    const [r, g, b] = warmColor(shapeLuminance(luminance[i]!, elapsed))
    const offset = i * 4
    image.data[offset] = r
    image.data[offset + 1] = g
    image.data[offset + 2] = b
    image.data[offset + 3] = 255
  }

  ctx.putImageData(image, 0, 0)
}

/**
 * Fixed-height footer box, below the agents/sessions list. Renders at the
 * source resolution and lets CSS scale it to the sidebar's width so it never
 * distorts regardless of how many tabs/sessions push the list above it.
 *
 * The repaint loop stops while the window is hidden and stretches its cadence
 * on battery, matching the visibility-gated timer policy in main.ts.
 */
export function SidebarPortrait() {
  const portrait = useStore($sidebarPortrait)
  const onBattery = useStore($onBattery)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    ensureSidebarPortrait()
  }, [])

  useEffect(() => {
    if (!portrait) {
      return
    }

    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    canvas.width = portrait.width
    canvas.height = portrait.height

    let timer = 0
    let cycleStart = performance.now()

    const tick = () => {
      if (document.hidden) {
        // Hidden window: skip the frame entirely; the next visible tick
        // repaints with the correct elapsed time.
        timer = window.setTimeout(tick, PORTRAIT_CYCLE_MS)
        return
      }

      const elapsed = performance.now() - cycleStart
      if (elapsed >= PORTRAIT_CYCLE_MS) {
        cycleStart = performance.now()
      }

      paint(canvas, portrait, elapsed >= PORTRAIT_CYCLE_MS ? 0 : elapsed)
      timer = window.setTimeout(tick, batteryPollInterval(PORTRAIT_FRAME_MS, onBattery))
    }

    tick()

    return () => window.clearTimeout(timer)
  }, [portrait, onBattery])

  if (!portrait) {
    return null
  }

  return (
    <div className="shrink-0 px-2 pb-1.5">
      <canvas
        className="block w-full rounded-md"
        ref={canvasRef}
        style={{ aspectRatio: `${portrait.width} / ${portrait.height}` }}
      />
    </div>
  )
}
