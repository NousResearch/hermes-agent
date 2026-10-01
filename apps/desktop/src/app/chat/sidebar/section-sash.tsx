import {
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useState
} from 'react'

import { cn } from '@/lib/utils'
import { setPaneHeightOverride } from '@/store/panes'

import {
  KEYBOARD_STEP_PX,
  resizeSessionsSeam,
  type SeamHeights,
  SIDEBAR_BOTTOM_SECTION_ID,
  SIDEBAR_PINNED_SECTION_ID,
  SIDEBAR_SESSIONS_SECTION_ID
} from './section-resize'

export const PINNED_H_VAR = '--sidebar-pinned-h'
export const BOTTOM_H_VAR = '--sidebar-bottom-h'
export const SESSIONS_MIN_H_VAR = '--sidebar-sessions-min-h'

export type SectionSashEdge = 'bottom-seam' | 'pinned-seam'

interface SeamSpec {
  /** The neighbour's scrolling body, inside the section list. */
  selector: string
  cssVar: string
  storeId: string
  /** +1: dragging down grows the neighbour (it's above). -1: it shrinks (below). */
  direction: 1 | -1
}

const SEAMS: Record<SectionSashEdge, SeamSpec> = {
  'pinned-seam': {
    selector: '[data-sidebar-section="pinned"] [data-sidebar="group-content"]',
    cssVar: PINNED_H_VAR,
    storeId: SIDEBAR_PINNED_SECTION_ID,
    direction: 1
  },
  'bottom-seam': {
    selector: '[data-sidebar-section="bottom"]',
    cssVar: BOTTOM_H_VAR,
    storeId: SIDEBAR_BOTTOM_SECTION_ID,
    direction: -1
  }
}

interface SectionSashProps {
  /** The sidebar's section list; it carries the height CSS variables. */
  containerRef: RefObject<HTMLDivElement | null>
  edge: SectionSashEdge
  label: string
}

const px = (value: number | undefined) => (value === undefined ? '' : `${value}px`)

function measure(container: HTMLElement, seam: SeamSpec) {
  const neighbour = container.querySelector<HTMLElement>(seam.selector)
  const sessions = container.querySelector<HTMLElement>('[data-sidebar-section="sessions"]')

  if (!neighbour || !sessions) {
    return null
  }

  return {
    neighbour: neighbour.getBoundingClientRect().height,
    sessions: sessions.getBoundingClientRect().height
  }
}

function commit(seam: SeamSpec, heights: SeamHeights) {
  setPaneHeightOverride(seam.storeId, heights.neighbour)
  setPaneHeightOverride(SIDEBAR_SESSIONS_SECTION_ID, heights.sessions)
}

/**
 * A horizontal resize sash on one of Sessions' seams. Mid-drag it writes the
 * CSS variables straight onto the list element — the sidebar is a large tree
 * and must not re-render per pointer frame — and persists once on release.
 * Double-click returns every section to its natural size.
 */
export function SectionSash({ containerRef, edge, label }: SectionSashProps) {
  const [dragging, setDragging] = useState(false)
  const seam = SEAMS[edge]

  const startDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const container = containerRef.current
    const start = container && measure(container, seam)

    if (event.button !== 0 || !container || !start) {
      return
    }

    event.preventDefault()
    const startY = event.clientY
    // What the store last rendered, to put back if the OS cancels the gesture.
    const before = [seam.cssVar, SESSIONS_MIN_H_VAR].map(name => [name, container.style.getPropertyValue(name)])
    let latest: SeamHeights | null = null
    setDragging(true)
    document.body.style.cursor = 'row-resize'

    const onMove = (move: PointerEvent) => {
      if (!latest && move.clientY === startY) {
        return
      }

      latest = resizeSessionsSeam(start, seam.direction * (move.clientY - startY))
      container.style.setProperty(seam.cssVar, px(latest.neighbour))
      container.style.setProperty(SESSIONS_MIN_H_VAR, px(latest.sessions))
    }

    const finish = (end: PointerEvent) => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', finish)
      window.removeEventListener('pointercancel', finish)
      document.body.style.cursor = ''
      setDragging(false)

      if (end.type === 'pointercancel') {
        for (const [name, value] of before) {
          container.style.setProperty(name, value)
        }
      } else if (latest) {
        // Only a real drag persists. A press that never moved (including each
        // half of a double-click) must not freeze today's natural heights
        // into fixed overrides.
        commit(seam, latest)
      }
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', finish)
    window.addEventListener('pointercancel', finish)
  }

  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = event.key === 'ArrowUp' ? -1 : event.key === 'ArrowDown' ? 1 : 0
    const container = containerRef.current
    const start = container && measure(container, seam)

    if (!step || !start) {
      return
    }

    event.preventDefault()
    commit(seam, resizeSessionsSeam(start, seam.direction * step * KEYBOARD_STEP_PX))
  }

  const reset = () => {
    for (const id of [SIDEBAR_PINNED_SECTION_ID, SIDEBAR_BOTTOM_SECTION_ID, SIDEBAR_SESSIONS_SECTION_ID]) {
      setPaneHeightOverride(id, undefined)
    }
  }

  return (
    // Zero-height in the flow so it never shifts the sections it sits between;
    // the hit area straddles the seam. Compact viewports flatten the sidebar
    // into one scroll, where per-section heights don't apply.
    <div className="relative h-0 shrink-0 compact:hidden">
      <div
        aria-label={label}
        aria-orientation="horizontal"
        className="group/sash absolute inset-x-0 top-0 z-10 h-2 -translate-y-1/2 cursor-row-resize outline-none"
        onDoubleClick={reset}
        onKeyDown={onKeyDown}
        onPointerDown={startDrag}
        role="separator"
        tabIndex={0}
      >
        {/* Drawn exactly like the pane-shell sash (TreeSplit) so every seam in
            the app reads the same: a hairline that recedes at 0.1, and on
            hover an accent-tinted band. The faint stroke alone measured
            1.0–1.4:1 across the built-in skins (invisible on nous-alt dark);
            the accent mix in --ui-sash-hover-border is what carries it. */}
        <span
          className={cn(
            'absolute inset-x-2 top-1/2 h-px -translate-y-1/2 bg-(--ui-stroke-secondary) transition-opacity duration-100',
            dragging ? 'opacity-100' : 'opacity-10 group-hover/sash:opacity-100 group-focus-visible/sash:opacity-100'
          )}
        />
        <span
          className={cn(
            'absolute inset-x-2 top-1/2 h-(--vscode-sash-hover-size,0.25rem) -translate-y-1/2 bg-(--ui-sash-hover-border) transition-opacity duration-100',
            dragging ? 'opacity-100' : 'opacity-0 group-hover/sash:opacity-100 group-focus-visible/sash:opacity-100'
          )}
        />
      </div>
    </div>
  )
}
