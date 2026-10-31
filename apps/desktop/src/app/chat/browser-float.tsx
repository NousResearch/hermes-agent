import { useStore } from '@nanostores/react'
import { type PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { PaneStripGlyph } from '@/components/ui/pane-tab'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $browserFloat,
  clampFloatRect,
  clearBrowserFloat,
  dockBrowserFloat,
  dockSideForPointer,
  FLOAT_MARGIN,
  floatPillOffset,
  type FloatRect,
  setBrowserFloatMinimized,
  setBrowserFloatRect,
  snapFloatRect
} from '@/store/browser-float'
import { $browserPages, $floatingBrowserTabId, $previewTabs } from '@/store/preview'

import { browserTabLabel } from './preview-tile'
import { PreviewTilePane } from './right-rail/preview'

type Gesture = { kind: 'move' | 'resize'; pointerX: number; pointerY: number; start: FloatRect }

const viewport = () => ({ height: window.innerHeight, width: window.innerWidth })

/** Fallback for `--titlebar-height` when the shell hasn't defined it. */
const TITLEBAR_FALLBACK = 34
const PILL_HEIGHT_FALLBACK = 32
/** `max-w-64` on the pill, for when it hasn't been measured yet. */
const PILL_WIDTH_FALLBACK = 256

/** Where the pill goes now: measured against the status bar, the title bar, and
 *  the composer — whose send button sits at the bottom-right of the chat column,
 *  under the pill. */
function pillOffset(pill: HTMLElement | null) {
  const statusbar = document.querySelector<HTMLElement>('[data-slot="statusbar"]')
  const bottom = statusbar ? Math.max(0, window.innerHeight - statusbar.getBoundingClientRect().top) : 0
  const titlebar = parseFloat(getComputedStyle(statusbar ?? document.documentElement).getPropertyValue('--titlebar-height'))

  // Only clear the composer when it actually reaches under the pill's corner; a
  // narrow chat column beside an open rail leaves the corner free.
  const dock = document.querySelector<HTMLElement>('[data-slot="composer-dock"]')?.getBoundingClientRect()
  const reach = (pill?.offsetWidth || PILL_WIDTH_FALLBACK) + FLOAT_MARGIN
  const composer = dock && dock.right >= window.innerWidth - reach ? Math.max(0, window.innerHeight - dock.top) : 0

  return floatPillOffset(
    viewport(),
    { bottom, composer, top: Number.isFinite(titlebar) ? titlebar : TITLEBAR_FALLBACK },
    pill?.offsetHeight || PILL_HEIGHT_FALLBACK
  )
}

/**
 * The floating Browser window (see `store/browser-float.ts`). Renders the same
 * preview pane the layout tree does, in a window over the app, or — minimised —
 * as a pill bottom-right, just above the status bar.
 */
export function BrowserFloat() {
  const { t } = useI18n()
  const copy = t.preview.float
  const tabId = useStore($floatingBrowserTabId)
  const tabs = useStore($previewTabs)
  const pages = useStore($browserPages)
  const { minimized, rect } = useStore($browserFloat)
  const [gesture, setGesture] = useState<Gesture | null>(null)
  const [live, setLive] = useState<FloatRect | null>(null)
  const gestureRef = useRef<Gesture | null>(null)
  const pillRef = useRef<HTMLButtonElement | null>(null)
  const [pill, setPill] = useState(() => pillOffset(null))
  const tab = tabId ? tabs.find(item => item.id === tabId) : undefined

  // The tab closed while floating: nothing is left to float.
  useEffect(() => {
    if (tabId && !tab) {
      clearBrowserFloat()
    }
  }, [tab, tabId])

  // Seat the pill above the composer and the status bar, and keep it there as
  // the window or the composer changes height.
  useEffect(() => {
    if (!minimized) {
      return
    }

    const place = () => setPill(pillOffset(pillRef.current))
    const dock = document.querySelector<HTMLElement>('[data-slot="composer-dock"]')
    const watch = dock && typeof ResizeObserver === 'function' ? new ResizeObserver(place) : null

    place()

    if (watch && dock) {
      watch.observe(dock)
    }

    window.addEventListener('resize', place)

    return () => {
      watch?.disconnect()
      window.removeEventListener('resize', place)
    }
  }, [minimized])

  // Keep the float on screen when the window shrinks.
  useEffect(() => {
    if (!tabId) {
      return
    }

    const onResize = () => {
      const current = $browserFloat.get().rect

      if (current) {
        setBrowserFloatRect(clampFloatRect(current, viewport()))
      }
    }

    window.addEventListener('resize', onResize)

    return () => window.removeEventListener('resize', onResize)
  }, [tabId])

  if (!tabId || !tab || tab.target.kind !== 'url' || !rect) {
    return null
  }

  const title = browserTabLabel(tab.target, pages[tabId])
  const shown = live ?? rect

  const begin = (kind: Gesture['kind']) => (event: ReactPointerEvent<HTMLElement>) => {
    // Title-bar buttons keep their own clicks; only the bar itself drags.
    if (event.button !== 0 || (event.target as Element).closest('button')) {
      return
    }

    event.preventDefault()
    event.currentTarget.setPointerCapture?.(event.pointerId)
    const next = { kind, pointerX: event.clientX, pointerY: event.clientY, start: rect }

    gestureRef.current = next
    setGesture(next)
    setLive(rect)
  }

  const move = (event: ReactPointerEvent<HTMLElement>) => {
    const active = gestureRef.current

    if (!active) {
      return
    }

    const dx = event.clientX - active.pointerX
    const dy = event.clientY - active.pointerY
    const { start } = active

    setLive(
      clampFloatRect(
        active.kind === 'move'
          ? { ...start, x: start.x + dx, y: start.y + dy }
          : { ...start, height: start.height + dy, width: start.width + dx },
        viewport()
      )
    )
  }

  const end = (event: ReactPointerEvent<HTMLElement>) => {
    const active = gestureRef.current

    gestureRef.current = null
    setGesture(null)

    if (!active || !live) {
      setLive(null)

      return
    }

    const side = active.kind === 'move' ? dockSideForPointer(event.clientX, viewport()) : null

    setLive(null)

    if (side) {
      dockBrowserFloat(side)

      return
    }

    setBrowserFloatRect(active.kind === 'move' ? snapFloatRect(live, viewport()) : live)
  }

  if (minimized) {
    return (
      <Tip label={copy.restore}>
        <button
          aria-label={copy.restore}
          className={cn(
            'fixed z-40 flex max-w-64 items-center gap-2 rounded-full border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) py-1.5 pr-3 pl-2.5 text-xs text-(--ui-text-primary) shadow-lg hover:bg-(--ui-control-hover-background) [-webkit-app-region:no-drag]'
          )}
          data-browser-float-pill=""
          onClick={() => setBrowserFloatMinimized(false)}
          ref={pillRef}
          style={pill}
          type="button"
        >
          <span aria-hidden className="relative flex size-2">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-(--ui-green) opacity-60 motion-reduce:animate-none" />
            <span className="relative inline-flex size-2 rounded-full bg-(--ui-green)" />
          </span>
          <span className="min-w-0 truncate">{title}</span>
        </button>
      </Tip>
    )
  }

  return (
    <section
      aria-label={copy.label}
      className={cn(
        'fixed top-0 left-0 z-40 flex flex-col overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) shadow-2xl',
        !gesture && 'transition-[transform,width,height] duration-200 ease-out motion-reduce:transition-none'
      )}
      data-browser-float=""
      style={{ height: shown.height, transform: `translate3d(${shown.x}px, ${shown.y}px, 0)`, width: shown.width }}
    >
      <header
        className="flex h-8 shrink-0 cursor-grab items-center gap-1 border-b border-(--ui-stroke-tertiary) pr-1 pl-2.5 select-none active:cursor-grabbing"
        data-browser-float-title=""
        onPointerCancel={end}
        onPointerDown={begin('move')}
        onPointerMove={move}
        onPointerUp={end}
      >
        <Codicon aria-hidden className="text-(--ui-text-tertiary)" name="globe" size="0.75rem" />
        <span className="min-w-0 flex-1 truncate text-xs text-(--ui-text-secondary)">{title}</span>
        <PaneStripGlyph
          icon={<Codicon name="layout-sidebar-left" size="0.8125rem" />}
          label={copy.dockLeft}
          onSelect={() => dockBrowserFloat('left')}
        />
        <PaneStripGlyph
          icon={<Codicon name="layout-sidebar-right" size="0.8125rem" />}
          label={copy.dockRight}
          onSelect={() => dockBrowserFloat('right')}
        />
        <PaneStripGlyph
          icon={<Codicon name="chrome-minimize" size="0.8125rem" />}
          label={copy.minimize}
          onSelect={() => setBrowserFloatMinimized(true)}
        />
      </header>
      <div className="relative flex min-h-0 flex-1 flex-col">
        <PreviewTilePane tabId={tabId} />
        {/* The page's frame swallows pointer events mid-gesture; a shield keeps
            the drag or resize with the float. */}
        {gesture && <div aria-hidden className="absolute inset-0 z-10" data-browser-float-shield="" />}
      </div>
      <div
        aria-hidden
        className="absolute right-0 bottom-0 z-20 size-3 cursor-nwse-resize"
        data-browser-float-resize=""
        onPointerCancel={end}
        onPointerDown={begin('resize')}
        onPointerMove={move}
        onPointerUp={end}
      />
    </section>
  )
}
