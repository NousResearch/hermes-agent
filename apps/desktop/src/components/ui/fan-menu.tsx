import {
  type CSSProperties,
  memo,
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState
} from 'react'
import { createPortal } from 'react-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

export type FanDirection = 'horizontal' | 'vertical' | 'arc'

export interface FanMenuItem {
  id: string
  icon: ReactNode
  label: string
  /** A toggle that is ON. Solid primary disc, so it reads from across the room. */
  active?: boolean
  disabled?: boolean
  onSelect: () => void
}

export interface FanMenuProps {
  /** The always-visible control. Hovering or focusing it fans `items` out. */
  hub: FanMenuItem & { className?: string }
  items: readonly FanMenuItem[]
  direction?: FanDirection
  /** Accessible name for the fanned group. */
  label: string
  /** Gap between discs, px. Discs are the hub's measured size; the step
   *  between them is that size plus this. */
  gap?: number
  /** Override the outward-facing tooltip side derived from the fan geometry. */
  tipAnchor?: 'top' | 'left' | 'right' | 'bottom'
  /** The hairline caret on the hub that says "there is more". */
  hint?: boolean
}

interface Point {
  x: number
  y: number
}

/** Quick with a touch of overshoot: the discs pop into place, not float. */
const POP = 'cubic-bezier(0.2, 1.25, 0.4, 1)'
const OPEN_MS = 160
const CLOSE_MS = 100
const STAGGER_MS = 20
/** Grace while the pointer crosses the gap between the hub and a disc. */
const HIDE_DELAY_MS = 150
/** Widest an arc will spread before it grows its radius instead. */
const ARC_MAX_SPAN_DEG = 150

/**
 * Disc centres relative to the hub, in px. `step` is one disc plus one gap,
 * so neighbours never touch in any direction.
 *
 * - vertical: a column straight up, in order.
 * - horizontal: reading order, split around the hub — the first half to the
 *   left, the rest to the right, the extra one (odd counts) on the right.
 * - arc: a semicircle above, spread so adjacent discs sit one `step` apart
 *   along the chord; past 150° the radius grows to fit instead.
 */
export function fanOffsets(direction: FanDirection, count: number, step: number): Point[] {
  const out: Point[] = []

  if (direction === 'vertical') {
    for (let i = 0; i < count; i++) {
      out.push({ x: 0, y: -(i + 1) * step })
    }

    return out
  }

  if (direction === 'horizontal') {
    const left = Math.floor(count / 2)

    for (let i = 0; i < count; i++) {
      out.push({ x: i < left ? -(left - i) * step : (i - left + 1) * step, y: 0 })
    }

    return out
  }

  if (count === 1) {
    return [{ x: 0, y: -step }]
  }

  let radius = step * 1.15
  let delta = 2 * Math.asin(Math.min(1, step / (2 * radius)))
  const maxSpan = (ARC_MAX_SPAN_DEG * Math.PI) / 180

  if (delta * (count - 1) > maxSpan) {
    delta = maxSpan / (count - 1)
    radius = step / (2 * Math.sin(delta / 2))
  }

  const start = Math.PI / 2 + (delta * (count - 1)) / 2

  for (let i = 0; i < count; i++) {
    const a = start - i * delta

    out.push({ x: Math.cos(a) * radius, y: -Math.sin(a) * radius })
  }

  return out
}

const sameRect = (a: DOMRect | null, b: DOMRect) =>
  a !== null && a.left === b.left && a.top === b.top && a.width === b.width && a.height === b.height

const OPEN_HUB_TIP_SIDE = { vertical: 'left', horizontal: 'top', arc: 'bottom' } as const

function fanTipSide(direction: FanDirection, offset: Point): NonNullable<FanMenuProps['tipAnchor']> {
  if (direction === 'vertical') {
    return 'left'
  }

  if (direction === 'horizontal' || Math.abs(offset.x) < Math.abs(offset.y)) {
    return 'top'
  }

  return offset.x < 0 ? 'left' : 'right'
}

/**
 * One control that fans its siblings out on hover — a column, a row centred
 * on it, or an arc. The hub stays in the flow where the consumer puts it; the
 * discs portal to `body` so an `overflow-hidden` parent cannot clip them, and
 * are re-anchored on scroll and resize.
 *
 * Hover, geometry and open/close timing live here. The consumer only
 * re-renders when its own items change; pointer traffic never reaches it.
 */
export function FanMenu({ direction = 'vertical', gap = 4, hint = true, hub, items, label, tipAnchor }: FanMenuProps) {
  const anchorRef = useRef<HTMLSpanElement | null>(null)
  const [rect, setRect] = useState<DOMRect | null>(null)
  const [open, setOpen] = useState(false)
  const hideTimer = useRef<number | undefined>(undefined)
  const unmountTimer = useRef<number | undefined>(undefined)

  const clearTimers = useCallback(() => {
    window.clearTimeout(hideTimer.current)
    window.clearTimeout(unmountTimer.current)
    hideTimer.current = undefined
  }, [])

  const measure = useCallback(() => {
    const el = anchorRef.current

    if (el) {
      const next = el.getBoundingClientRect()

      setRect(current => (sameRect(current, next) ? current : next))
    }
  }, [])

  const show = useCallback(() => {
    clearTimers()
    measure()
    setOpen(true)
  }, [clearTimers, measure])

  const hide = useCallback(() => {
    clearTimers()
    setOpen(false)
    unmountTimer.current = window.setTimeout(() => setRect(null), CLOSE_MS + STAGGER_MS * items.length)
  }, [clearTimers, items.length])

  const hideSoon = useCallback(() => {
    window.clearTimeout(hideTimer.current)
    hideTimer.current = window.setTimeout(hide, HIDE_DELAY_MS)
  }, [hide])

  const stayOpen = useCallback(() => {
    window.clearTimeout(hideTimer.current)
    hideTimer.current = undefined
  }, [])

  useEffect(() => clearTimers, [clearTimers])

  // The hub can move under a parked fan: its container grows, the window
  // resizes, an ancestor scrolls.
  const anchored = rect !== null

  useEffect(() => {
    if (!anchored) {
      return
    }

    window.addEventListener('scroll', measure, true)
    window.addEventListener('resize', measure)

    return () => {
      window.removeEventListener('scroll', measure, true)
      window.removeEventListener('resize', measure)
    }
  }, [anchored, measure])

  // Enter/leave alone can strand the fan open. A disc that goes `disabled`
  // under the pointer (a toggle turning pending right after the click) stops
  // receiving pointer events, so its `pointerleave` never fires; a fast
  // flick can skip the leave the same way. Watch the document while open and
  // close as soon as the pointer is over neither the hub nor a disc.
  useEffect(() => {
    if (!open) {
      return
    }

    const onMove = (event: PointerEvent) => {
      const target = event.target instanceof Element ? event.target : null

      const inside =
        target !== null &&
        (anchorRef.current?.contains(target) === true || target.closest('[data-slot="fan-menu"]') !== null)

      if (inside) {
        stayOpen()
      } else if (hideTimer.current === undefined) {
        hideSoon()
      }
    }

    document.addEventListener('pointermove', onMove, true)

    return () => document.removeEventListener('pointermove', onMove, true)
  }, [hideSoon, open, stayOpen])

  const leaveUnlessWithin = (event: React.FocusEvent<HTMLElement>, then: () => void) => {
    if (!(event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget))) {
      then()
    }
  }

  return (
    <>
      {/* The wrapper takes the hover, not the button: a disabled hub swallows
          pointer events, and the fan must still open around it. */}
      <span
        className="relative inline-flex shrink-0"
        data-slot="fan-menu-anchor"
        onBlur={event => leaveUnlessWithin(event, hideSoon)}
        onFocus={show}
        onPointerEnter={show}
        onPointerLeave={hideSoon}
        ref={anchorRef}
      >
        <Tip label={hub.label} side={tipAnchor ?? (open ? OPEN_HUB_TIP_SIDE[direction] : 'top')}>
          <Button
            aria-expanded={open}
            aria-haspopup="true"
            aria-label={hub.label}
            aria-pressed={hub.active}
            className={hub.className}
            disabled={hub.disabled}
            onClick={hub.onSelect}
            size="icon"
            type="button"
            variant="ghost"
          >
            {hub.icon}
          </Button>
        </Tip>
        {hint ? (
          <Codicon
            className={cn(
              'pointer-events-none absolute -top-px right-0 text-(--ui-text-quaternary) transition-opacity duration-100',
              open && 'opacity-0'
            )}
            name="chevron-up"
            size="0.5rem"
          />
        ) : null}
      </span>
      {rect
        ? createPortal(
            <FanCluster
              direction={direction}
              items={items}
              label={label}
              onClose={hide}
              onPointerEnter={stayOpen}
              onPointerLeave={hideSoon}
              open={open}
              rect={rect}
              step={rect.width + gap}
              tipAnchor={tipAnchor}
            />,
            document.body
          )
        : null}
    </>
  )
}

function discStyle(target: Point, index: number, entered: boolean): CSSProperties {
  // Folded, a disc sits halfway to its slot rather than under the hub: one
  // that mounts beneath a parked pointer picks up :hover and keeps it (a
  // transform-only move never re-hit-tests), so it would pop out pre-lit.
  const k = entered ? 1 : 0.5
  const delay = index * STAGGER_MS

  return {
    opacity: entered ? 1 : 0,
    scale: entered ? '1' : '0.5',
    translate: `${(target.x * k).toFixed(2)}px ${(target.y * k).toFixed(2)}px`,
    transition: entered
      ? `translate ${OPEN_MS}ms ${POP} ${delay}ms, scale ${OPEN_MS}ms ${POP} ${delay}ms, opacity 80ms ease-out ${delay}ms`
      : `translate ${CLOSE_MS}ms ease-in, scale ${CLOSE_MS}ms ease-in, opacity ${CLOSE_MS}ms ease-in`
  }
}

const FanCluster = memo(function FanCluster({
  direction,
  items,
  label,
  onClose,
  onPointerEnter,
  onPointerLeave,
  open,
  rect,
  step,
  tipAnchor
}: {
  direction: FanDirection
  items: readonly FanMenuItem[]
  label: string
  onClose: () => void
  onPointerEnter: () => void
  onPointerLeave: () => void
  open: boolean
  rect: DOMRect
  step: number
  tipAnchor: FanMenuProps['tipAnchor']
}) {
  const rootRef = useRef<HTMLDivElement | null>(null)
  // Mount folded, then unfold on the next style pass so the translate
  // actually transitions instead of landing in place. The layout read forces
  // Chromium to resolve the folded style before the change.
  const [entered, setEntered] = useState(false)

  useLayoutEffect(() => {
    if (open) {
      void rootRef.current?.offsetWidth
    }

    setEntered(open)
  }, [open])

  const offsets = fanOffsets(direction, items.length, step)

  return (
    // The root is the hub's footprint and takes no pointer events itself, so
    // the real hub underneath keeps its hover; only the discs are hittable.
    <div
      aria-label={label}
      className="pointer-events-none fixed z-(--z-over-modal)"
      data-direction={direction}
      data-slot="fan-menu"
      data-state={entered ? 'open' : 'closed'}
      onKeyDown={event => {
        if (event.key === 'Escape') {
          event.stopPropagation()
          onClose()
        }
      }}
      ref={rootRef}
      role="group"
      style={{ height: rect.height, left: rect.left, top: rect.top, width: rect.width }}
    >
      {items.map((item, index) => (
        <Tip key={item.id} label={item.label} side={tipAnchor ?? fanTipSide(direction, offsets[index])}>
          <Button
            aria-label={item.label}
            aria-pressed={item.active}
            className={cn(
              'absolute inset-0 rounded-full shadow-md',
              entered ? 'pointer-events-auto' : 'pointer-events-none'
            )}
            disabled={item.disabled}
            onClick={item.onSelect}
            onPointerEnter={onPointerEnter}
            onPointerLeave={onPointerLeave}
            size="inline"
            style={discStyle(offsets[index], index, entered)}
            type="button"
            variant={item.active ? 'default' : 'floating'}
          >
            {item.icon}
          </Button>
        </Tip>
      ))}
    </div>
  )
})
