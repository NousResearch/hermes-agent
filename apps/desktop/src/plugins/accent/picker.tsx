// Dev-only: an OKLCH accent picker in the statusbar.
//
// Rebuilt from an HSV picker, which was the wrong instrument for the job: HSV
// crushes the entire blue family into a narrow band of its hue rail, so
// dragging "to blue" lands you on ~266° — pure sRGB blue, which reads violet to
// the eye. Every blue that actually looks blue (GitHub 257°, Tailwind 260°,
// Nous 263°) sits in a few degrees you cannot reliably hit in HSV. In OKLCH the
// rail is perceptually even, so those degrees get their fair share of track.
//
// The field is a CANVAS, not a CSS gradient, because a hue slice of OKLCH is a
// curved wedge in sRGB rather than a rectangle. Drawing it per-pixel through
// the real conversion means every pixel is a color the display can actually
// show, and the wedge's edge IS the gamut boundary — no silent clamping.
//
// It drives `$accentOverride`, which the theme context feeds through
// `retintTheme`, so the whole app repaints against the REAL derivation on every
// pointer move. Nothing persists: reload and you're back on the authored theme.
//
// Sizing note: the statusbar is `h-5` (20px). The TRIGGER must live inside that
// band or it clips to a sliver; the panel is a popover and can be any size.

import {
  $accentOverride,
  contrastRatio,
  hexToOklch,
  maxChroma,
  type Oklch,
  oklchToHex,
  oklchToSrgb255,
  Popover,
  PopoverContent,
  PopoverTrigger,
  setAccentOverride,
  useTheme,
  useValue
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

// Named reference points, with their OKLCH hue — the ones worth comparing while
// judging an accent. The hue spread across the blues is the whole reason this
// control exists: they look far apart and are 6° apart.
const SWATCHES: ReadonlyArray<{ hex: string; name: string }> = [
  { hex: '#0053FD', name: 'Nous blue · 263° (light seed)' },
  { hex: '#4a84fe', name: 'Nous blue · 263° (dark seed)' },
  { hex: '#1540B1', name: 'Psyche blue · 264°' },
  { hex: '#0969da', name: 'GitHub blue · 257°' },
  { hex: '#196d31', name: 'GitHub green · 148°' },
  { hex: '#8250df', name: 'GitHub purple · 303°' },
  { hex: '#bf3989', name: 'GitHub pink · 354°' },
  { hex: '#bc4c00', name: 'GitHub orange · 52°' }
]

const FIELD_W = 224
const FIELD_H = 130
const RAIL_H = 12

// Statusbar label width, in mono characters. The chip's text CHANGES as you
// drag, and the bar lays out right-to-left from the right edge — so an unpinned
// width slides the picker out from under the pointer mid-drag.
const HEX_CH = 7 // '#rrggbb'

/** Chroma axis top — past this is out of gamut at every hue, so it'd be dead space. */
const CHROMA_MAX = 0.33

const clamp01 = (n: number) => Math.min(1, Math.max(0, n))

/** Pointer drag over a box, reported as 0..1 on each axis. */
function useDrag(onMove: (x: number, y: number) => void) {
  const ref = useRef<HTMLElement>(null)
  const moveRef = useRef(onMove)
  moveRef.current = onMove

  const onPointerDown = useCallback((event: React.PointerEvent) => {
    const box = ref.current

    if (!box) {
      return
    }

    const emit = (clientX: number, clientY: number) => {
      const r = box.getBoundingClientRect()
      moveRef.current(clamp01((clientX - r.left) / r.width), clamp01((clientY - r.top) / r.height))
    }

    emit(event.clientX, event.clientY)

    const onPointerMove = (e: PointerEvent) => emit(e.clientX, e.clientY)

    const stop = () => {
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', stop)
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', stop)
  }, [])

  return { onPointerDown, ref }
}

/** Lightness (y, 1→0) × chroma (x, 0→CHROMA_MAX) at one hue. */
function ChromaLightnessField({
  hue,
  lch,
  onPick
}: {
  hue: number
  lch: Oklch
  onPick: (l: number, c: number) => void
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const drag = useDrag((x, y) => onPick(1 - y, x * CHROMA_MAX))

  useEffect(() => {
    const el = canvas.current

    if (!el) {
      return
    }

    const ctx = el.getContext('2d')

    if (!ctx) {
      return
    }

    const img = ctx.createImageData(FIELD_W, FIELD_H)

    for (let y = 0; y < FIELD_H; y += 1) {
      const l = 1 - y / (FIELD_H - 1)

      for (let x = 0; x < FIELD_W; x += 1) {
        const rgb = oklchToSrgb255({ l, c: (x / (FIELD_W - 1)) * CHROMA_MAX, h: hue })
        const i = (y * FIELD_W + x) * 4

        if (rgb) {
          img.data[i] = rgb[0]
          img.data[i + 1] = rgb[1]
          img.data[i + 2] = rgb[2]
          img.data[i + 3] = 255
        } else {
          // Outside sRGB: leave it transparent so the wedge's real edge shows.
          img.data[i + 3] = 0
        }
      }
    }

    ctx.clearRect(0, 0, FIELD_W, FIELD_H)
    ctx.putImageData(img, 0, 0)
  }, [hue])

  return (
    <div
      className="relative cursor-crosshair overflow-hidden rounded-sm"
      onPointerDown={drag.onPointerDown}
      ref={drag.ref as React.RefObject<HTMLDivElement>}
      style={{ height: FIELD_H, width: FIELD_W }}
    >
      <canvas className="block size-full" height={FIELD_H} ref={canvas} width={FIELD_W} />
      <div
        className="pointer-events-none absolute size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,.6)]"
        style={{ left: `${(lch.c / CHROMA_MAX) * 100}%`, top: `${(1 - lch.l) * 100}%` }}
      />
    </div>
  )
}

/** Hue rail, sampled through the real conversion at the current L and C. */
function HueRail({ lch, onPick }: { lch: Oklch; onPick: (h: number) => void }) {
  const drag = useDrag(x => onPick(x * 360))

  // Hold the current lightness/chroma so the rail previews THIS color at every
  // hue, rather than a generic rainbow that lies about where you'll land.
  const gradient = useMemo(() => {
    const stops: string[] = []

    for (let i = 0; i <= 36; i += 1) {
      const h = (i / 36) * 360
      const c = Math.min(lch.c, maxChroma(lch.l, h))
      stops.push(`${oklchToHex({ l: lch.l, c, h })} ${(i / 36) * 100}%`)
    }

    return `linear-gradient(to right, ${stops.join(', ')})`
  }, [lch.l, lch.c])

  return (
    <div
      className="relative cursor-ew-resize rounded-full"
      onPointerDown={drag.onPointerDown}
      ref={drag.ref as React.RefObject<HTMLDivElement>}
      style={{ background: gradient, height: RAIL_H, width: FIELD_W }}
    >
      <div
        className="pointer-events-none absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,.6)]"
        style={{ left: `${(lch.h / 360) * 100}%` }}
      />
    </div>
  )
}

function AccentPicker() {
  const { theme, renderedMode } = useTheme()
  const override = useValue($accentOverride)
  const painted = theme.colors.primary
  const [lch, setLch] = useState<Oklch>(() => hexToOklch(painted) ?? { l: 0.5, c: 0.15, h: 260 })
  const [text, setText] = useState(painted)

  // Follow the painted accent when it changes from outside this control — but
  // never mid-drag, which would fight the user's own pointer.
  const dragging = useRef(false)

  useEffect(() => {
    if (!dragging.current) {
      setLch(hexToOklch(painted) ?? { l: 0.5, c: 0.15, h: 260 })
      setText(painted)
    }
  }, [painted])

  const commit = (next: Oklch) => {
    dragging.current = true
    setLch(next)
    const hex = oklchToHex(next)
    setText(hex)
    setAccentOverride(hex)
    requestAnimationFrame(() => {
      dragging.current = false
    })
  }

  const surface = (renderedMode === 'dark' ? theme.darkColors : theme.colors) ?? theme.colors
  const ratio = contrastRatio(painted, surface.sidebarBackground ?? surface.background)

  return (
    <div className="flex flex-col gap-2 p-2" style={{ width: FIELD_W + 16 }}>
      <ChromaLightnessField hue={lch.h} lch={lch} onPick={(l, c) => commit({ ...lch, c, l })} />
      <HueRail lch={lch} onPick={h => commit({ ...lch, h })} />

      <div className="flex items-center gap-1.5">
        <span className="size-5 shrink-0 rounded-sm border border-(--dt-border)" style={{ background: painted }} />
        <input
          className="min-w-0 flex-1 rounded-sm border border-(--dt-border) bg-transparent px-1.5 py-0.5 font-mono text-[11px] uppercase"
          onChange={event => {
            setText(event.target.value)
            setAccentOverride(event.target.value)
          }}
          spellCheck={false}
          value={text}
        />
        <button
          className="shrink-0 rounded-sm border border-(--dt-border) px-1.5 py-0.5 text-[11px]"
          onClick={() => setAccentOverride(null)}
          type="button"
        >
          reset
        </button>
      </div>

      <div className="grid grid-cols-8 gap-1">
        {SWATCHES.map(swatch => (
          <button
            className="size-5 rounded-sm border border-(--dt-border)"
            key={swatch.hex}
            onClick={() => setAccentOverride(swatch.hex)}
            style={{ background: swatch.hex }}
            title={`${swatch.name} · ${swatch.hex}`}
            type="button"
          />
        ))}
      </div>

      {/* OKLCH readout + what the theme RESOLVED to for the painted mode, which
          is usually not the hex you clicked (dark adapts for contrast). Digits
          are tabular and the hue/L/C fields are fixed-width: an unpinned
          readout re-wraps as you drag past 100°, which resizes the popover
          under the pointer. */}
      <div className="flex items-center justify-between font-mono text-[10px] tabular-nums text-(--ui-text-tertiary)">
        <span>
          H<span className="inline-block w-[3ch] text-right">{Math.round(lch.h)}</span> L{lch.l.toFixed(2)} C
          {lch.c.toFixed(3)}
        </span>
        <span className={ratio >= 4.5 ? '' : 'text-(--dt-destructive)'}>
          {renderedMode} {ratio.toFixed(1)}:1
        </span>
      </div>
      {/* Always occupies its row, even when empty — this line appears exactly
          when a pick needs adapting, which is mid-drag, and a mounting <p>
          would grow the panel while the pointer is inside it. */}
      <p className="h-3 font-mono text-[10px] leading-tight text-(--ui-text-tertiary)">
        {override && override.toLowerCase() !== painted.toLowerCase()
          ? `picked ${override} → ${painted} for contrast`
          : '\u00a0'}
      </p>
    </div>
  )
}

export function AccentPickerTrigger() {
  const { theme } = useTheme()

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          className="inline-flex h-full items-center gap-1.5 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
          title="Accent color (dev)"
          type="button"
        >
          <span
            className="size-2.5 shrink-0 rounded-full border border-(--dt-border)"
            style={{ background: theme.colors.primary }}
          />
          {/* Always the hex, never a word: a label that alternated between
              `accent` and `#rrggbb` changed width on the first drag, which
              reflowed the bar under the cursor. Seven mono characters, always. */}
          <span className="text-center font-mono tabular-nums" style={{ width: `${HEX_CH}ch` }}>
            {theme.colors.primary}
          </span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-auto p-0" side="top">
        <AccentPicker />
      </PopoverContent>
    </Popover>
  )
}
