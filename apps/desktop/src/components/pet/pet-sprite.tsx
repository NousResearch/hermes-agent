import { memo, useEffect, useMemo, useRef } from 'react'

import { $petState, type PetInfo, type PetState } from '@/store/pet'

const DEFAULT_FRAME_W = 192
const DEFAULT_FRAME_H = 208
const DEFAULT_FRAMES = 6
const DEFAULT_LOOP_MS = 1100
// Mirrors agent.pet.constants.DEFAULT_SCALE — fallback only; the gateway sends
// the configured scale.
const DEFAULT_SCALE = 0.33

// Mirrors agent.pet.constants.CODEX_STATE_ROWS (Petdex current taxonomy).
export const DEFAULT_STATE_ROWS = [
  'idle',
  'running-right',
  'running-left',
  'waving',
  'jumping',
  'failed',
  'waiting',
  'running',
  'review'
]

const STATE_ALIASES: Record<PetState, string[]> = {
  idle: ['idle'],
  wave: ['wave', 'waving'],
  jump: ['jump', 'jumping'],
  run: ['run', 'running'],
  failed: ['failed'],
  review: ['review'],
  waiting: ['waiting']
}

const ROW_TO_STATE: Record<string, PetState> = {
  idle: 'idle',
  wave: 'wave',
  waving: 'wave',
  jump: 'jump',
  jumping: 'jump',
  run: 'run',
  running: 'run',
  'running-right': 'run',
  'running-left': 'run',
  failed: 'failed',
  review: 'review',
  waiting: 'waiting'
}

/**
 * Pick the running row + mirror for a horizontal travel direction.
 *
 * Codex sheets ship dedicated `running-left` / `running-right` locomotion rows
 * (already facing their way → no flip). Pets without them fall back to the
 * in-place `running`/`run` row, which faces left by convention, so rightward
 * travel is mirrored. Returns no `row` in that fallback case so the caller lets
 * `$petState` resolve it (and applies `mirror`).
 */
export function roamWalkRow(dir: -1 | 0 | 1, stateRows?: string[]): { row?: string; mirror: boolean } {
  if (dir === 0) {
    return { mirror: false }
  }

  const rows = stateRows ?? DEFAULT_STATE_ROWS
  const hasLeft = rows.includes('running-left')
  const hasRight = rows.includes('running-right')

  if (dir > 0) {
    if (hasRight) {
      return { mirror: false, row: 'running-right' }
    }

    if (hasLeft) {
      return { mirror: true, row: 'running-left' }
    }

    return { mirror: true }
  }

  if (hasLeft) {
    return { mirror: false, row: 'running-left' }
  }

  if (hasRight) {
    return { mirror: true, row: 'running-right' }
  }

  return { mirror: false }
}

interface PetSpriteProps {
  info: PetInfo
  /** On-screen scale multiplier applied on top of the pet's native scale. */
  zoom?: number
  /**
   * Force a specific animation state instead of reading the live `$petState`.
   * Used by the generate-flow preview to showcase every row without driving (or
   * being driven by) the real agent activity that moves the floating mascot.
   */
  stateOverride?: PetState
  /** Force a concrete row name from `info.stateRows` (e.g. `running-right`). */
  rowOverride?: string
}

/**
 * Canvas renderer for a petdex spritesheet — the one piece that must be
 * TypeScript (the engine's decode/encode is Python). Draws the row matching the
 * live `$petState`, stepping `framesPerState` frames across a `loopMs` loop.
 *
 * State is read from `$petState` via a ref + subscription rather than a prop,
 * so the frequent activity-driven state changes during an agent turn update the
 * canvas (inside its RAF loop) WITHOUT triggering a React re-render. Combined
 * with `memo`, this component effectively never re-renders after mount until
 * the pet itself changes.
 */
function PetSpriteImpl({ info, zoom = 1, stateOverride, rowOverride }: PetSpriteProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const stateRef = useRef<PetState>($petState.get())
  const overrideRef = useRef<PetState | undefined>(stateOverride)
  const rowOverrideRef = useRef<string | undefined>(rowOverride)

  // Keep the override current without re-running the RAF setup effect.
  useEffect(() => {
    overrideRef.current = stateOverride
  }, [stateOverride])

  useEffect(() => {
    rowOverrideRef.current = rowOverride
  }, [rowOverride])

  const frameW = info.frameW ?? DEFAULT_FRAME_W
  const frameH = info.frameH ?? DEFAULT_FRAME_H
  const frames = info.framesPerState ?? DEFAULT_FRAMES
  const framesByState = info.framesByState
  const framesByRow = info.framesByRow
  const loopMs = info.loopMs ?? DEFAULT_LOOP_MS
  const scale = (info.scale ?? DEFAULT_SCALE) * zoom
  const rows = info.stateRows ?? DEFAULT_STATE_ROWS

  const drawW = Math.round(frameW * scale)
  const drawH = Math.round(frameH * scale)

  const image = useMemo(() => {
    if (!info.spritesheetBase64) {
      return null
    }

    const img = new Image()
    img.src = `data:${info.mime ?? 'image/webp'};base64,${info.spritesheetBase64}`

    return img
  }, [info.spritesheetBase64, info.mime])

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas || !image) {
      return
    }

    // willReadFrequently: the pop-out overlay samples this canvas's alpha under
    // the cursor (per-pixel click-through), so opt into the CPU-readback path.
    const ctx = canvas.getContext('2d', { willReadFrequently: true })

    if (!ctx) {
      return
    }

    // Track state via subscription, not a prop — no re-render on activity ticks.
    stateRef.current = $petState.get()

    const unsubState = $petState.listen(next => {
      stateRef.current = next
    })

    let raf = 0
    let frame = 0
    let lastStep = performance.now()
    let drawnFrame = -1
    let drawnRow = -1
    let activeRow = -1
    let activeCount = -1

    const rowIndexForState = (s: PetState): number => {
      for (const key of STATE_ALIASES[s] ?? [s]) {
        const idx = rows.indexOf(key)

        if (idx >= 0) {
          return idx
        }
      }

      return 0
    }

    // Resolve a state to the row it draws and its real frame count. A state
    // with no real frames (ragged sheet, empty row) falls back to idle rather
    // than flashing blank padding.
    const resolve = (s: PetState): { row: number; count: number } => {
      const real = framesByState?.[s] ?? frames

      if (real > 0) {
        return { row: rowIndexForState(s), count: real }
      }

      return { row: rowIndexForState('idle'), count: Math.max(1, framesByState?.idle ?? frames) }
    }

    const resolveRow = (rowName: string): { row: number; count: number } => {
      const row = rows.indexOf(rowName)
      const state = ROW_TO_STATE[rowName]

      const count = Math.max(
        1,
        framesByRow?.[rowName] ?? framesByState?.[rowName] ?? (state ? framesByState?.[state] : 0) ?? frames
      )

      return { row: row >= 0 ? row : rowIndexForState(state ?? 'idle'), count }
    }

    const render = (now: number) => {
      const forcedRow = rowOverrideRef.current
      const { row, count } = forcedRow ? resolveRow(forcedRow) : resolve(overrideRef.current ?? stateRef.current)

      if (row !== activeRow || count !== activeCount) {
        activeRow = row
        activeCount = count
        frame = 0
        lastStep = now
        drawnFrame = -1
      }

      // Per-state step keeps every state's loop ~loopMs even when frame counts
      // differ; counts vary per row so derive the cadence here, not once.
      const stepMs = loopMs / count

      if (now - lastStep >= stepMs) {
        frame += 1
        lastStep = now
      }

      frame %= count

      // Only touch the canvas when the visible cell actually changes. The RAF
      // ticks at ~60Hz but the sprite only steps ~5Hz, so this skips ~90% of
      // the clear+draw work and keeps the main thread free.
      if ((frame !== drawnFrame || row !== drawnRow) && image.complete && image.naturalWidth > 0) {
        const sx = frame * frameW
        const sy = row * frameH
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.imageSmoothingEnabled = false
        ctx.drawImage(image, sx, sy, frameW, frameH, 0, 0, drawW, drawH)
        drawnFrame = frame
        drawnRow = row
      }

      raf = requestAnimationFrame(render)
    }

    raf = requestAnimationFrame(render)

    return () => {
      cancelAnimationFrame(raf)
      unsubState()
    }
  }, [image, frameW, frameH, frames, framesByState, framesByRow, loopMs, drawW, drawH, rows])

  return (
    <canvas
      aria-label={info.displayName ? `${info.displayName} pet` : 'pet'}
      height={drawH}
      ref={canvasRef}
      style={{ height: drawH, width: drawW }}
      width={drawW}
    />
  )
}

/**
 * Memoized so a parent re-render (e.g. a position commit on drag-end) doesn't
 * re-run the canvas setup. Props change only when the pet itself changes.
 */
export const PetSprite = memo(PetSpriteImpl)
