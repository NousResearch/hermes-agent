/**
 * Floating panes — the tree's non-tiling placement.
 *
 * `placement: 'floating'` opts a pane OUT of the layout tree: it never becomes
 * a track, never takes width from a zone, and never appears in a tab strip.
 * The tree renders it as a fixed card above itself, draggable by its header,
 * with position + collapse persisted per pane id. The geometry rules live in
 * floating-rect.ts.
 */

import { useStore } from '@nanostores/react'
import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useRef, useState } from 'react'

import { HUD_SURFACE } from '@/app/floating-hud'
import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { Codicon } from '@/components/ui/codicon'
import { ContribBoundary } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import type { Contribution } from '@/contrib/types'
import { readJson, writeJson } from '@/lib/storage'
import { cn } from '@/lib/utils'

import { $hiddenTreePanes } from '../store'

import {
  anchoredRect,
  clampFloatingRect,
  FLOATING_PLACEMENT,
  floatingPx,
  type FloatingRect,
  type FloatingViewport,
  reflowRect
} from './floating-rect'
import { paneChrome } from './track-model'

const POSITIONS_KEY = 'hermes.desktop.floatingPanes.v1'

const DEFAULT_SIZE = { width: 240, height: 180 }

interface StoredRect {
  x: number
  y: number
  collapsed?: boolean
}

const readStored = (): Record<string, StoredRect> => readJson<Record<string, StoredRect>>(POSITIONS_KEY) ?? {}

const viewportNow = (): FloatingViewport => ({
  width: window.innerWidth,
  height: window.innerHeight,
  top: TITLEBAR_HEIGHT
})

function FloatingPane({ pane }: { pane: Contribution }) {
  const chrome = paneChrome(pane)
  const anchor = chrome.anchor ?? 'top-right'

  const size = {
    width: floatingPx(chrome.width, DEFAULT_SIZE.width),
    height: floatingPx(chrome.height, DEFAULT_SIZE.height)
  }

  const [rect, setRect] = useState<FloatingRect>(() => {
    const stored = readStored()[pane.id]
    const spawned = anchoredRect(anchor, size, viewportNow())

    return stored ? { ...spawned, x: stored.x, y: stored.y } : spawned
  })

  const [collapsed, setCollapsed] = useState(() => readStored()[pane.id]?.collapsed ?? false)

  const drag = useRef<{ x: number; y: number } | null>(null)
  const viewport = useRef<FloatingViewport>(viewportNow())

  const persist = useCallback(
    (next: FloatingRect, nextCollapsed: boolean) => {
      writeJson(POSITIONS_KEY, { ...readStored(), [pane.id]: { x: next.x, y: next.y, collapsed: nextCollapsed } })
    },
    [pane.id]
  )

  // Track the viewport so an edge-anchored pane rides its edge on resize.
  // The previous-size read lives in the handler (not a useEffect body): it's
  // window geometry, not a mirrored reactive value.
  const handleResize = useCallback(() => {
    const next = viewportNow()

    setRect(current => reflowRect(current, anchor, viewport.current, next))
    viewport.current = next
  }, [anchor])

  useEffect(() => {
    window.addEventListener('resize', handleResize)

    return () => window.removeEventListener('resize', handleResize)
  }, [handleResize])

  const onPointerDown = useCallback((event: ReactPointerEvent<HTMLElement>) => {
    if ((event.target as HTMLElement).closest('[data-floating-no-drag]')) {
      return
    }

    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = { x: event.clientX, y: event.clientY }
    event.preventDefault()
  }, [])

  const onPointerMove = useCallback((event: ReactPointerEvent<HTMLElement>) => {
    const from = drag.current

    if (!from) {
      return
    }

    drag.current = { x: event.clientX, y: event.clientY }

    setRect(current =>
      clampFloatingRect(
        { ...current, x: current.x + event.clientX - from.x, y: current.y + event.clientY - from.y },
        viewport.current
      )
    )
  }, [])

  const onPointerUp = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      if (!drag.current) {
        return
      }

      drag.current = null
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      setRect(current => {
        persist(current, collapsed)

        return current
      })
    },
    [collapsed, persist]
  )

  const toggleCollapsed = () =>
    setCollapsed(current => {
      persist(rect, !current)

      return !current
    })

  return (
    <div
      className={cn('pointer-events-auto fixed z-45 flex flex-col overflow-hidden', HUD_SURFACE)}
      data-floating-pane={pane.id}
      style={{
        left: rect.x,
        top: rect.y,
        width: size.width,
        height: collapsed ? undefined : size.height
      }}
    >
      {/* Header IS the drag handle — the floating equivalent of a tab strip. */}
      <header
        className="flex shrink-0 cursor-grab items-center justify-between gap-2 px-2.5 py-1.5 text-[0.6875rem] text-(--ui-text-secondary) select-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        style={{ touchAction: 'none' }}
      >
        <span className="truncate font-medium">{pane.title ?? pane.id}</span>
        <button
          className="rounded p-0.5 text-(--ui-text-quaternary) transition-colors hover:text-(--ui-text-primary)"
          data-floating-no-drag=""
          onClick={toggleCollapsed}
          type="button"
        >
          <Codicon name={collapsed ? 'chevron-down' : 'chevron-up'} size="0.75rem" />
        </button>
      </header>

      {!collapsed && (
        <div className="min-h-0 flex-1 overflow-auto">
          <ContribBoundary id={pane.id}>{pane.render?.()}</ContribBoundary>
        </div>
      )}
    </div>
  )
}

/** Every `placement: 'floating'` contribution, rendered above the tree. */
export function FloatingPanes() {
  const panes = useContributions('panes')
  const hidden = useStore($hiddenTreePanes)

  const floating = panes.filter(pane => paneChrome(pane).placement === FLOATING_PLACEMENT && !hidden.has(pane.id))

  if (floating.length === 0) {
    return null
  }

  return (
    <>
      {floating.map(pane => (
        <FloatingPane key={`${pane.source ?? 'core'}:${pane.id}`} pane={pane} />
      ))}
    </>
  )
}
