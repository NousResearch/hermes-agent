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
import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useRef } from 'react'

import { HUD_SURFACE } from '@/app/floating-hud'
import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { useOnboardingChatActive } from '@/components/onboarding-chat/assembly'
import { Codicon } from '@/components/ui/codicon'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import type { Contribution } from '@/contrib/types'
import { LAYOUT_KEYS } from '@/lib/layout-persistence'
import { Codecs } from '@/lib/persisted'
import { cn } from '@/lib/utils'
import { modeLayout } from '@/store/interface-mode'

import { hiddenPaneProps, PaneLifecycleContext, PaneVisibleContext } from '../../pane-visibility'
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
import { PaneBody } from './pane-body'
import { paneChrome } from './track-model'

const DEFAULT_SIZE = { width: 240, height: 180 }

interface StoredRect {
  x: number
  y: number
  collapsed?: boolean
}

const $positions = modeLayout.atom<Record<string, StoredRect>>(
  LAYOUT_KEYS.floating,
  () => ({}),
  Codecs.json(value =>
    value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, StoredRect>) : {}
  ),
  true
)

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

  const stored = useStore($positions)[pane.id]
  const rect = { ...anchoredRect(anchor, size, viewportNow()), ...stored }
  const collapsed = stored?.collapsed ?? false
  const drag = useRef<{ x: number; y: number } | null>(null)
  const viewport = useRef<FloatingViewport>(viewportNow())

  const setRect = useCallback(
    (update: (current: FloatingRect) => FloatingRect) => {
      const positions = $positions.get()
      const current = positions[pane.id]
      const next = update({
        ...anchoredRect(anchor, { width: size.width, height: size.height }, viewport.current),
        ...current
      })
      $positions.set({ ...positions, [pane.id]: { x: next.x, y: next.y, collapsed: current?.collapsed } })
    },
    [pane.id, anchor, size.width, size.height]
  )

  const persist = useCallback(
    (next: FloatingRect, nextCollapsed: boolean) => {
      const positions = { ...$positions.get(), [pane.id]: { x: next.x, y: next.y, collapsed: nextCollapsed } }
      $positions.set(positions)
      modeLayout.write(LAYOUT_KEYS.floating, JSON.stringify(positions))
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
  }, [anchor, setRect])

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
  }, [setRect])

  const onPointerUp = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      if (!drag.current) {
        return
      }

      drag.current = null
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      persist({ ...rect, ...$positions.get()[pane.id] }, collapsed)
    },
    [pane.id, rect, collapsed, persist]
  )

  const toggleCollapsed = () => persist(rect, !collapsed)

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
          <Codicon name={collapsed ? 'chevron-up' : 'chevron-down'} size="0.75rem" />
        </button>
      </header>

      {(!collapsed || chrome.lifecycleKeepAlive) && (
        <PaneBody hidden={collapsed}>
          <div className="h-full overflow-auto" inert={collapsed || undefined} {...hiddenPaneProps(collapsed)}>
            <PaneLifecycleContext value={collapsed ? 'hot-hidden' : 'visible'}>
              <PaneVisibleContext value={!collapsed}>
                <ContribBoundary id={pane.id}>{pane.render && <ContribRender render={pane.render} />}</ContribBoundary>
              </PaneVisibleContext>
            </PaneLifecycleContext>
          </div>
        </PaneBody>
      )}
    </div>
  )
}

/** Every `placement: 'floating'` contribution, rendered above the tree. */
export function FloatingPanes() {
  const panes = useContributions('panes')
  const hidden = useStore($hiddenTreePanes)

  const onboardingActive = useOnboardingChatActive()

  const floating = onboardingActive
    ? []
    : panes.filter(pane => paneChrome(pane).placement === FLOATING_PLACEMENT && !hidden.has(pane.id))

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
