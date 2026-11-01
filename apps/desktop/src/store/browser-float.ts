import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

import { $rightRailActiveTabId, selectRightRailTab } from './layout'
import { setPaneWidthOverride } from './panes'
import { $floatingBrowserTabId, $previewTabs, openBrowserTab, toggleBrowserTab } from './preview'
import { PREVIEW_TILE_PREFIX } from './preview-explicit'

/**
 * The floating Browser: one in-app Browser tab lifted out of the layout tree
 * into a window that floats over the app. It drags by its title, resizes from
 * its corner, snaps to the nearest corner on release, docks into the tree on
 * either side, and minimises to a pill (Esc). Geometry persists per install.
 *
 * The page itself is the same preview pane the tree renders; the docked tree
 * leaves the tab out while it floats (`$dockedVisiblePreviewTabs`).
 */

export interface FloatRect {
  height: number
  width: number
  x: number
  y: number
}

export interface Viewport {
  height: number
  width: number
}

export type FloatDockSide = 'left' | 'right'

export interface BrowserFloatState {
  minimized: boolean
  rect: FloatRect | null
}

const STORAGE_KEY = 'hermes.desktop.browserFloat.v1'

/** Edge gap the float keeps from the window, and the snap reach of a corner. */
export const FLOAT_MARGIN = 12
export const FLOAT_SNAP_DISTANCE = 48
/** A drag released this close to a side edge docks into the tree there. */
export const FLOAT_DOCK_DISTANCE = 6
export const FLOAT_MIN_WIDTH = 320
export const FLOAT_MIN_HEIGHT = 220

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

function decodeRect(raw: unknown): FloatRect | null {
  const rect = raw as Partial<FloatRect> | null

  return rect && finite(rect.x) && finite(rect.y) && finite(rect.width) && finite(rect.height)
    ? { height: rect.height, width: rect.width, x: rect.x, y: rect.y }
    : null
}

function loadState(): BrowserFloatState {
  const raw = readJson<Partial<BrowserFloatState>>(STORAGE_KEY)

  return { minimized: false, rect: decodeRect(raw?.rect) }
}

export const $browserFloat = atom<BrowserFloatState>(loadState())

function setFloatState(next: BrowserFloatState) {
  $browserFloat.set(next)
  writeJson(STORAGE_KEY, { rect: next.rect })
}

/** Where a fresh float opens: the bottom-right corner, about a third of the window. */
export function defaultFloatRect(viewport: Viewport): FloatRect {
  const width = Math.max(FLOAT_MIN_WIDTH, Math.round(viewport.width * 0.42))
  const height = Math.max(FLOAT_MIN_HEIGHT, Math.round(viewport.height * 0.52))

  return clampFloatRect(
    { height, width, x: viewport.width - width - FLOAT_MARGIN, y: viewport.height - height - FLOAT_MARGIN },
    viewport
  )
}

/** Keep the float fully on screen and no smaller than its minimum. */
export function clampFloatRect(rect: FloatRect, viewport: Viewport): FloatRect {
  const maxWidth = Math.max(FLOAT_MIN_WIDTH, viewport.width - FLOAT_MARGIN * 2)
  const maxHeight = Math.max(FLOAT_MIN_HEIGHT, viewport.height - FLOAT_MARGIN * 2)
  const width = Math.min(Math.max(rect.width, FLOAT_MIN_WIDTH), maxWidth)
  const height = Math.min(Math.max(rect.height, FLOAT_MIN_HEIGHT), maxHeight)
  const x = Math.min(Math.max(rect.x, FLOAT_MARGIN), Math.max(FLOAT_MARGIN, viewport.width - width - FLOAT_MARGIN))
  const y = Math.min(Math.max(rect.y, FLOAT_MARGIN), Math.max(FLOAT_MARGIN, viewport.height - height - FLOAT_MARGIN))

  return { height, width, x, y }
}

/** On release, pull the float into a corner it was dropped near. Each axis
 *  snaps on its own, so a float dropped near one edge settles against it. */
export function snapFloatRect(rect: FloatRect, viewport: Viewport): FloatRect {
  const clamped = clampFloatRect(rect, viewport)
  const left = FLOAT_MARGIN
  const right = viewport.width - clamped.width - FLOAT_MARGIN
  const top = FLOAT_MARGIN
  const bottom = viewport.height - clamped.height - FLOAT_MARGIN

  const snap = (value: number, a: number, b: number) =>
    Math.abs(value - a) <= FLOAT_SNAP_DISTANCE ? a : Math.abs(value - b) <= FLOAT_SNAP_DISTANCE ? b : value

  return { ...clamped, x: snap(clamped.x, left, right), y: snap(clamped.y, top, bottom) }
}

/** What the minimised pill keeps clear of: the title bar, the status bar, and
 *  the composer when that reaches under the pill. */
export interface FloatPillInsets {
  bottom: number
  /** How much of the bottom edge the composer takes up at the pill's corner. */
  composer?: number
  top: number
}

/** The minimised pill sits bottom-right of the content area, just above the
 *  status bar and clear of the composer's send button. In a window too short
 *  for that it stays under the title bar. */
export function floatPillOffset(
  viewport: Viewport,
  insets: FloatPillInsets,
  pillHeight: number
): { bottom: number; right: number } {
  const lowest = Math.max(insets.bottom, insets.composer ?? 0) + FLOAT_MARGIN
  const highest = viewport.height - insets.top - FLOAT_MARGIN - pillHeight

  return { bottom: Math.max(0, Math.min(lowest, highest)), right: FLOAT_MARGIN }
}

/** The side a drag docks to when the pointer is released against that edge. */
export function dockSideForPointer(pointerX: number, viewport: Viewport): FloatDockSide | null {
  if (pointerX <= FLOAT_DOCK_DISTANCE) {
    return 'left'
  }

  if (pointerX >= viewport.width - FLOAT_DOCK_DISTANCE) {
    return 'right'
  }

  return null
}

function viewport(): Viewport {
  return { height: window.innerHeight, width: window.innerWidth }
}

/** Side each docked Browser tab returns to; read by the preview-tile mirror. */
const dockSides = new Map<string, FloatDockSide>()

export function browserDockSide(tabId: string): FloatDockSide | undefined {
  return dockSides.get(tabId)
}

/** Lift a Browser tab out of the tree into the float (the active Browser, or a
 *  fresh one when none is open). */
export function floatBrowserTab(tabId?: string) {
  let id = tabId

  if (!id) {
    openBrowserTab()
    id = $rightRailActiveTabId.get() ?? undefined
  }

  const tab = id ? $previewTabs.get().find(item => item.id === id) : undefined

  if (!tab || tab.target.kind !== 'url') {
    return
  }

  const current = $browserFloat.get()

  setFloatState({ minimized: false, rect: clampFloatRect(current.rect ?? defaultFloatRect(viewport()), viewport()) })
  $floatingBrowserTabId.set(tab.id)
}

export function setBrowserFloatRect(rect: FloatRect) {
  setFloatState({ ...$browserFloat.get(), rect })
}

export function setBrowserFloatMinimized(minimized: boolean) {
  if ($browserFloat.get().minimized !== minimized) {
    setFloatState({ ...$browserFloat.get(), minimized })
  }
}

/** Return the floating tab to the tree on `side`, sized to the float's width
 *  through the pane shell's own width override (the store its sash writes). */
export function dockBrowserFloat(side: FloatDockSide = 'right') {
  const tabId = $floatingBrowserTabId.get()

  if (!tabId) {
    return
  }

  const width = $browserFloat.get().rect?.width

  dockSides.set(tabId, side)

  if (width) {
    setPaneWidthOverride(`${PREVIEW_TILE_PREFIX}:${tabId}`, Math.round(width))
  }

  setBrowserFloatMinimized(false)
  $floatingBrowserTabId.set(null)
  selectRightRailTab(tabId)
}

/** The tab left (closed, or the profile changed): drop the float quietly. */
export function clearBrowserFloat() {
  setBrowserFloatMinimized(false)
  $floatingBrowserTabId.set(null)
}

/** ⌥⌘B: float the Browser; with one floating, minimise it to the pill or
 *  bring it back. */
export function toggleBrowserFloat() {
  if (!$floatingBrowserTabId.get()) {
    floatBrowserTab()

    return
  }

  setBrowserFloatMinimized(!$browserFloat.get().minimized)
}

/** Escape inside the open float minimises it. Returns whether it claimed the
 *  key, so the global dispatcher yields (Escape is also composer.cancel). */
export function minimizeBrowserFloatFrom(target: EventTarget | null): boolean {
  if (!$floatingBrowserTabId.get() || $browserFloat.get().minimized) {
    return false
  }

  const node = target instanceof Element ? target : null

  if (!node?.closest('[data-browser-float]')) {
    return false
  }

  setBrowserFloatMinimized(true)

  return true
}

/** ⌘⇧L while the Browser floats docks it back where it was, rather than
 *  toggling a tree pane the float is standing in for. */
export function toggleBrowserOrDockFloat() {
  const tabId = $floatingBrowserTabId.get()

  if (tabId) {
    dockBrowserFloat(browserDockSide(tabId) ?? 'right')

    return
  }

  toggleBrowserTab()
}
