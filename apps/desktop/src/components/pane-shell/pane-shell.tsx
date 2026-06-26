import { useStore } from '@nanostores/react'
import {
  Children,
  type CSSProperties,
  isValidElement,
  type ReactElement,
  type ReactNode,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import { cn } from '@/lib/utils'
import { $paneStates, ensurePaneRegistered, setPaneHeightOverride, setPaneWidthOverride } from '@/store/panes'

import { PaneShellContext, type PaneShellContextValue, type PaneSlot } from './context'

type PaneSide = 'left' | 'right'
type WidthValue = string | number

interface PaneRoleMarker {
  __paneShellRole?: 'pane' | 'main'
}

export interface PaneProps {
  children?: ReactNode
  className?: string
  defaultOpen?: boolean
  /** Paints a persistent hairline on the resize edge (not just the hover sash) so the pane boundary is always visible. */
  divider?: boolean
  /** Forces the pane closed (track→0, aria-hidden) without writing to the store — for transient route gates. */
  disabled?: boolean
  /** Like disabled, but keeps hoverReveal alive — collapses the track without writing to the store (e.g. narrow window). */
  forceCollapsed?: boolean
  /** When collapsed, float the contents over the main column on hover/focus instead of hiding them (track stays 0px). */
  hoverReveal?: boolean
  /**
   * Lay the pane out as a horizontal row beneath its rail (spanning every column on
   * its `side`) instead of as a vertical column. The pane then resizes on the Y axis.
   * Used to drop the terminal under a crowded rail rather than squeezing another column in.
   */
  bottomRow?: boolean
  /** Default height of a `bottomRow` pane. */
  height?: WidthValue
  /** Min/max height clamps for a `bottomRow` pane's vertical resize. */
  maxHeight?: WidthValue
  minHeight?: WidthValue
  /** Width of the collapsed-overlay panel. Defaults to the docked width (or its resize override); set this to render a narrower overlay than the docked pane (e.g. min width on mobile). */
  overlayWidth?: WidthValue
  /** Called with true while the pane is a collapsed hover-reveal overlay, so the consumer can keep contents mounted (ready to slide). */
  onOverlayActiveChange?: (overlayActive: boolean) => void
  id: string
  maxWidth?: WidthValue
  minWidth?: WidthValue
  resizable?: boolean
  side: PaneSide
  width?: WidthValue
}

export interface PaneMainProps {
  children?: ReactNode
  className?: string
}

export interface PaneShellProps {
  children?: ReactNode
  className?: string
  style?: CSSProperties
}

interface CollectedPane {
  bottomRow: boolean
  defaultOpen: boolean
  disabled: boolean
  forceCollapsed: boolean
  height: string
  id: string
  resizable: boolean
  side: PaneSide
  width: string
}

const DEFAULT_WIDTH = '16rem'
const DEFAULT_HEIGHT = '18rem'
const DEFAULT_RESIZE_MIN_WIDTH = 160
const DEFAULT_RESIZE_MIN_HEIGHT = 120

// Resize-sash geometry per axis: `x` is a vertical bar on the inner edge of a
// column; `y` is a horizontal bar on the top edge of a bottom row.
const SASH = {
  x: {
    orientation: 'vertical',
    bar: 'bottom-0 top-0 w-1 cursor-col-resize',
    line: 'inset-y-0 left-1/2 w-px -translate-x-1/2',
    hover: 'inset-y-0 left-1/2 w-(--vscode-sash-hover-size,0.25rem) -translate-x-1/2'
  },
  y: {
    orientation: 'horizontal',
    bar: 'inset-x-0 top-0 h-1 -translate-y-1/2 cursor-row-resize',
    line: 'inset-x-0 top-1/2 h-px -translate-y-1/2',
    hover: 'inset-x-0 top-1/2 h-(--vscode-sash-hover-size,0.25rem) -translate-y-1/2'
  }
} as const

// Hover-reveal slide. The enter delay is a pure-CSS hover-intent gate: a fast
// pass-by doesn't dwell on the trigger long enough for the delay to elapse.
const HOVER_REVEAL_SLIDE_MS = 220
const HOVER_REVEAL_ENTER_DELAY_MS = 130
const HOVER_REVEAL_EASE = 'cubic-bezier(0.32,0.72,0,1)'
// Offset shadow lifting the revealed panel off the content (same both sides;
// the mirror axis is offset-x, which is 0). Same color on light + dark.
const HOVER_REVEAL_SHADOW = '0px -18px 18px -5px #00000012'
// Edge trigger strip, inset past the OS window-resize grab area AND the
// adjacent pane's scrollbar (0.5rem, .scrollbar-dt) — the strip overlays the
// neighboring scroller's edge, so any overlap makes the scrollbar reveal the
// pane on hover and swallow its clicks (#44140).
const HOVER_REVEAL_TRIGGER_WIDTH = 14
const HOVER_REVEAL_EDGE_GUTTER = 'calc(0.5rem + 2px)'

// Fired (window CustomEvent<{ id }>) to toggle a force-collapsed pane's reveal
// from the keyboard, since its store-open toggle is a no-op while collapsed.
export const PANE_TOGGLE_REVEAL_EVENT = 'hermes:pane-toggle-reveal'

const widthToCss = (value: WidthValue | undefined, fallback: string) =>
  value === undefined ? fallback : typeof value === 'number' ? `${value}px` : value

const remPx = () =>
  typeof window === 'undefined'
    ? 16
    : Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16

const viewportPx = () => (typeof window === 'undefined' ? 1280 : window.innerWidth)
const viewportHeightPx = () => (typeof window === 'undefined' ? 800 : window.innerHeight)

// Resolves PaneProps min/max (number | "Npx" | "Nrem" | "Nvw" | "Nvh" | "N%") to
// pixels for drag clamping. vw/% resolve against window width, vh against height.
function widthToPx(value: WidthValue | undefined) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined
  }

  const match = value?.trim().match(/^(-?\d*\.?\d+)(px|rem|vw|vh|%)?$/)

  if (!match) {
    return undefined
  }

  const n = Number.parseFloat(match[1])

  switch (match[2]) {
    case 'rem':
      return n * remPx()

    case 'vh':
      return (n * viewportHeightPx()) / 100

    case 'vw':

    case '%':
      return (n * viewportPx()) / 100

    default:
      return n
  }
}

function isRole(child: unknown, role: 'pane' | 'main'): child is ReactElement {
  return isValidElement(child) && (child.type as PaneRoleMarker)?.__paneShellRole === role
}

function collectPanes(children: ReactNode) {
  const left: CollectedPane[] = []
  const right: CollectedPane[] = []
  let mainCount = 0

  Children.forEach(children, child => {
    if (isRole(child, 'main')) {
      mainCount++

      return
    }

    if (!isRole(child, 'pane')) {
      return
    }

    const props = child.props as PaneProps

    const entry: CollectedPane = {
      bottomRow: props.bottomRow ?? false,
      defaultOpen: props.defaultOpen ?? true,
      disabled: props.disabled ?? false,
      forceCollapsed: props.forceCollapsed ?? false,
      height: widthToCss(props.height, DEFAULT_HEIGHT),
      id: props.id,
      resizable: props.resizable ?? false,
      side: props.side,
      width: widthToCss(props.width, DEFAULT_WIDTH)
    }

    ;(props.side === 'left' ? left : right).push(entry)
  })

  return { left, mainCount, right }
}

type PaneStoreState = Record<string, { open: boolean; widthOverride?: number; heightOverride?: number }>

function paneIsOpen(pane: CollectedPane, states: PaneStoreState) {
  const stateOpen = states[pane.id]?.open ?? pane.defaultOpen

  return !pane.disabled && !pane.forceCollapsed && stateOpen
}

function trackForPane(pane: CollectedPane, states: PaneStoreState) {
  const open = paneIsOpen(pane, states)

  if (!open) {
    return { open: false, track: '0px' }
  }

  const override = pane.resizable ? states[pane.id]?.widthOverride : undefined

  return { open: true, track: override !== undefined ? `${override}px` : pane.width }
}

function heightTrackForPane(pane: CollectedPane, states: PaneStoreState) {
  const override = pane.resizable ? states[pane.id]?.heightOverride : undefined

  return override !== undefined ? `${override}px` : pane.height
}

export function PaneShell({ children, className, style }: PaneShellProps) {
  const paneStates = useStore($paneStates)
  const { left, mainCount, right } = useMemo(() => collectPanes(children), [children])

  if (import.meta.env.DEV && mainCount > 1) {
    console.warn('[PaneShell] expected at most one <PaneMain>, got', mainCount)
  }

  const ctxValue = useMemo(() => {
    const paneById = new Map<string, PaneSlot>()
    const tracks: string[] = []
    const cssVars: Record<string, string> = {}
    let column = 1

    // A bottom-row pane drops out of its rail's column flow and instead spans
    // every column on its side as a new row below them. The first open one wins
    // and decides which rail gets split into two rows.
    const leftCols = left.filter(pane => !pane.bottomRow)
    const rightCols = right.filter(pane => !pane.bottomRow)
    const bottomRowPanes = [...left, ...right].filter(pane => pane.bottomRow)
    const activeBottomRow = bottomRowPanes.find(pane => paneIsOpen(pane, paneStates)) ?? null
    const bottomRailSide = activeBottomRow?.side ?? null

    // Open column panes on the bottom row's side shrink to the top row; everything
    // else (main, the other rail, closed / hover-reveal panes) stays full height.
    const addColumn = (pane: CollectedPane, paneSide: PaneSide) => {
      const { open, track } = trackForPane(pane, paneStates)
      tracks.push(track)
      cssVars[`--pane-${pane.id}-width`] = track
      const gridRow = open && paneSide === bottomRailSide ? '1 / 2' : '1 / -1'
      paneById.set(pane.id, { open, side: paneSide, gridColumn: `${column} / ${column + 1}`, gridRow, bottomRow: false })
      column++
    }

    for (const pane of leftCols) {
      addColumn(pane, 'left')
    }

    tracks.push('minmax(0,1fr)')
    const mainColumn = column++

    for (const pane of rightCols) {
      addColumn(pane, 'right')
    }

    // Place every bottom-row pane: span its rail's columns on the second row.
    for (const pane of bottomRowPanes) {
      const gridColumn = pane.side === 'left' ? `1 / ${mainColumn}` : `${mainColumn + 1} / -1`
      paneById.set(pane.id, { open: pane === activeBottomRow, side: pane.side, gridColumn, gridRow: '2 / 3', bottomRow: true })
    }

    // Always emit explicit rows so `grid-row: 1 / -1` (full-height) resolves
    // against a known last line. With a bottom row active there are two tracks;
    // otherwise a single 1fr track behaves exactly like the old single-row grid.
    const gridTemplateRows = activeBottomRow
      ? `minmax(0,1fr) ${heightTrackForPane(activeBottomRow, paneStates)}`
      : 'minmax(0,1fr)'

    return { cssVars, gridTemplate: tracks.join(' '), gridTemplateRows, mainColumn, paneById } satisfies PaneShellContextValue & {
      cssVars: Record<string, string>
      gridTemplate: string
      gridTemplateRows: string
    }
  }, [left, paneStates, right])

  const composedStyle = useMemo<CSSProperties>(
    () => ({
      ...ctxValue.cssVars,
      ...style,
      gridTemplateColumns: ctxValue.gridTemplate,
      gridTemplateRows: ctxValue.gridTemplateRows
    }),
    [ctxValue.cssVars, ctxValue.gridTemplate, ctxValue.gridTemplateRows, style]
  )

  return (
    <PaneShellContext.Provider value={{ mainColumn: ctxValue.mainColumn, paneById: ctxValue.paneById }}>
      <div className={cn('relative grid h-full min-h-0', className)} data-pane-shell="" style={composedStyle}>
        {children}
      </div>
    </PaneShellContext.Provider>
  )
}

export function Pane({
  children,
  className,
  defaultOpen = true,
  divider = false,
  disabled = false,
  hoverReveal = false,
  maxHeight,
  minHeight,
  overlayWidth: overlayWidthProp,
  id,
  maxWidth,
  minWidth,
  onOverlayActiveChange,
  resizable = false,
  width
}: PaneProps) {
  const ctx = useContext(PaneShellContext)
  const paneStates = useStore($paneStates)
  const registered = useRef(false)
  const paneRef = useRef<HTMLDivElement | null>(null)
  // Keyboard (mod+b / mod+j) pins the reveal open while collapsed; hover is CSS.
  const [forced, setForced] = useState(false)

  const slot = ctx?.paneById.get(id)
  const open = Boolean(slot?.open && !disabled)
  const side = slot?.side ?? 'left'
  // Collapsed + hoverReveal: float the pane contents over the main column on
  // hover/focus instead of hiding them. Honors any persisted resize width.
  const overlayActive = !open && hoverReveal && !disabled
  const override = resizable ? paneStates[id]?.widthOverride : undefined
  // Overlay width: an explicit `overlayWidth` (e.g. min width on mobile) wins,
  // else the persisted resize override, else the docked width.
  const overlayWidth =
    overlayWidthProp !== undefined
      ? widthToCss(overlayWidthProp, DEFAULT_WIDTH)
      : override !== undefined
        ? `${override}px`
        : widthToCss(width, DEFAULT_WIDTH)

  useEffect(() => {
    if (registered.current) {
      return
    }

    registered.current = true
    ensurePaneRegistered(id, { open: defaultOpen })
  }, [defaultOpen, id])

  // Keyboard toggle pins/unpins the reveal while collapsed; clear when no longer
  // a collapsed overlay (reopened / widened).
  useEffect(() => {
    if (typeof window === 'undefined' || !overlayActive) {
      setForced(false)

      return
    }

    const onToggle = (e: Event) => {
      if ((e as CustomEvent<{ id: string }>).detail?.id === id) {
        setForced(v => !v)
      }
    }

    window.addEventListener(PANE_TOGGLE_REVEAL_EVENT, onToggle)

    return () => window.removeEventListener(PANE_TOGGLE_REVEAL_EVENT, onToggle)
  }, [id, overlayActive])

  // Keep contents mounted while collapsed so reveal is a pure CSS transform.
  useEffect(() => {
    onOverlayActiveChange?.(overlayActive)
  }, [onOverlayActiveChange, overlayActive])

  const isBottomRow = Boolean(slot?.bottomRow)
  const axis = isBottomRow ? 'y' : 'x'
  const sash = SASH[axis]
  const canResize = open && resizable
  const lo = widthToPx(minWidth) ?? DEFAULT_RESIZE_MIN_WIDTH
  const hi = widthToPx(maxWidth) ?? Number.POSITIVE_INFINITY
  const loH = widthToPx(minHeight) ?? DEFAULT_RESIZE_MIN_HEIGHT
  const hiH = widthToPx(maxHeight) ?? Number.POSITIVE_INFINITY

  // One pointer-drag for both axes. Columns grow toward the main column (left
  // rail → right, right rail → left); the bottom row grows up from its top edge.
  const startResize = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, axis: 'x' | 'y') => {
      const rect = paneRef.current?.getBoundingClientRect()
      const base = (axis === 'x' ? rect?.width : rect?.height) ?? 0

      if (!canResize || base <= 0) {
        return
      }

      event.preventDefault()

      const handle = event.currentTarget
      const { pointerId } = event
      const start = axis === 'x' ? event.clientX : event.clientY
      const dir = axis === 'x' ? (side === 'left' ? 1 : -1) : -1
      const [min, max] = axis === 'x' ? [lo, hi] : [loH, hiH]
      const apply = axis === 'x' ? setPaneWidthOverride : setPaneHeightOverride
      const restoreCursor = document.body.style.cursor
      const restoreSelect = document.body.style.userSelect

      handle.setPointerCapture?.(pointerId)
      document.body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize'
      document.body.style.userSelect = 'none'

      const onMove = (e: PointerEvent) => {
        const next = base + ((axis === 'x' ? e.clientX : e.clientY) - start) * dir
        apply(id, Math.round(Math.min(max, Math.max(min, next))))
      }

      const cleanup = () => {
        document.body.style.cursor = restoreCursor
        document.body.style.userSelect = restoreSelect
        handle.releasePointerCapture?.(pointerId)
        window.removeEventListener('pointermove', onMove, true)
        window.removeEventListener('pointerup', cleanup, true)
        window.removeEventListener('pointercancel', cleanup, true)
        window.removeEventListener('blur', cleanup)
      }

      window.addEventListener('pointermove', onMove, true)
      window.addEventListener('pointerup', cleanup, true)
      window.addEventListener('pointercancel', cleanup, true)
      window.addEventListener('blur', cleanup)
    },
    [canResize, hi, hiH, id, lo, loH, side]
  )

  if (!ctx) {
    if (import.meta.env.DEV) {
      console.warn(`[Pane:${id}] must be rendered inside <PaneShell>`)
    }

    return null
  }

  if (!slot) {
    return null
  }

  // Collapsed hover-reveal track: a 0px, pointer-transparent grid cell holding a
  // thin edge trigger + the floating panel (both absolute, escaping the zero
  // box). group-hover (or data-forced from the keyboard) drives the slide; the
  // enter-delay is the hover-intent gate. No JS pointer math.
  if (overlayActive) {
    const edge = side === 'left' ? 'left' : 'right'
    const offscreen = side === 'left' ? '-translate-x-[calc(100%+1rem)]' : 'translate-x-[calc(100%+1rem)]'

    return (
      <div
        className={cn('group/reveal pointer-events-none relative min-w-0', className)}
        data-forced={forced ? '' : undefined}
        data-pane-hover-reveal={forced ? 'open' : 'closed'}
        data-pane-id={id}
        data-pane-open="false"
        data-pane-side={side}
        ref={paneRef}
        style={{ gridColumn: slot.gridColumn, gridRow: slot.gridRow }}
      >
        <div
          aria-hidden="true"
          className="pointer-events-auto absolute inset-y-0 z-30 [-webkit-app-region:no-drag]"
          data-pane-reveal-trigger=""
          style={{ [edge]: HOVER_REVEAL_EDGE_GUTTER, width: HOVER_REVEAL_TRIGGER_WIDTH }}
        />

        {/* Keyed on side so flipping panes remounts off-screen on the new edge
            instead of transitioning the transform across the viewport. */}
        <div
          className={cn(
            'pointer-events-none absolute inset-y-0 z-30 overflow-hidden transition-transform delay-0',
            offscreen,
            'group-hover/reveal:pointer-events-auto group-hover/reveal:translate-x-0 group-hover/reveal:delay-[var(--reveal-enter-delay)] group-hover/reveal:shadow-[var(--reveal-shadow)]',
            'group-data-[forced]/reveal:pointer-events-auto group-data-[forced]/reveal:translate-x-0 group-data-[forced]/reveal:delay-0 group-data-[forced]/reveal:shadow-[var(--reveal-shadow)]'
          )}
          key={edge}
          style={
            {
              [edge]: 0,
              width: overlayWidth,
              '--reveal-shadow': HOVER_REVEAL_SHADOW,
              transitionDuration: `${HOVER_REVEAL_SLIDE_MS}ms`,
              transitionTimingFunction: HOVER_REVEAL_EASE,
              '--reveal-enter-delay': `${HOVER_REVEAL_ENTER_DELAY_MS}ms`
            } as CSSProperties
          }
        >
          <div className="flex h-full w-full flex-col">{children}</div>
        </div>
      </div>
    )
  }

  return (
    <div
      aria-hidden={!open}
      className={cn('relative min-h-0 min-w-0 overflow-hidden', !open && 'pointer-events-none', className)}
      data-pane-id={id}
      data-pane-open={open ? 'true' : 'false'}
      data-pane-side={slot.side}
      ref={paneRef}
      style={{ gridColumn: slot.gridColumn, gridRow: slot.gridRow }}
    >
      {canResize && (
        <div
          aria-label={`Resize ${id}`}
          aria-orientation={sash.orientation}
          className={cn(
            'group absolute z-20 [-webkit-app-region:no-drag]',
            sash.bar,
            !isBottomRow && (slot.side === 'left' ? 'right-0 translate-x-1/2' : 'left-0 -translate-x-1/2')
          )}
          onPointerDown={e => startResize(e, axis)}
          role="separator"
          tabIndex={0}
        >
          {divider && <span className={cn('absolute bg-(--ui-stroke-secondary)', sash.line)} />}
          <span
            className={cn(
              'absolute bg-(--ui-sash-hover-border) opacity-0 transition-opacity duration-100 group-hover:opacity-100 group-focus-visible:opacity-100',
              sash.hover
            )}
          />
        </div>
      )}
      {children}
    </div>
  )
}

;(Pane as unknown as PaneRoleMarker).__paneShellRole = 'pane'

export function PaneMain({ children, className }: PaneMainProps) {
  const ctx = useContext(PaneShellContext)

  if (!ctx) {
    if (import.meta.env.DEV) {
      console.warn('[PaneMain] must be rendered inside <PaneShell>')
    }

    return null
  }

  return (
    <div
      className={cn('flex min-h-0 min-w-0 flex-col overflow-hidden', className)}
      data-pane-main="true"
      style={{ gridColumn: `${ctx.mainColumn} / ${ctx.mainColumn + 1}`, gridRow: '1 / -1' }}
    >
      {children}
    </div>
  )
}

;(PaneMain as unknown as PaneRoleMarker).__paneShellRole = 'main'
