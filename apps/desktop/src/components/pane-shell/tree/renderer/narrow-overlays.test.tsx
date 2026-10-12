import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { registry } from '@/contrib/registry'
import { $paneStates, setPaneWidthOverride } from '@/store/panes'
import { $connection } from '@/store/session'
import { stubResizeObserver } from '@/test/jsdom'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree, $narrowViewport, declareDefaultTree } from '../store'

import { NarrowOverlays, narrowOverlayWidth } from './narrow-overlays'

// Ground truth for "the Bots tab is still visible when the sessions sidebar
// collapses on a narrow window". A collapsible pane DOCKED into the sessions
// zone (SESSIONS | BOTS) must leave the grid with the zone, and the narrow
// edge overlay must mirror the zone's tab strip so the docked pane stays
// reachable — not just the zone's first pane.

beforeAll(() => {
  stubResizeObserver()
})

const disposers: (() => void)[] = []

const registerPane = (id: string, title: string, data: Record<string, unknown>, body: string) => {
  disposers.push(
    registry.register({
      area: 'panes',
      data,
      id,
      render: () => <div data-testid={`${id}-body`}>{body}</div>,
      title
    })
  )
}

beforeEach(() => {
  window.localStorage.clear()
  $hiddenTreePanes.set(new Set())

  registerPane('sessions', 'sessions', { collapsible: true, placement: 'left', width: '237px' }, 'session rows')
  registerPane('bots', 'Bots', { collapsible: true, placement: 'left', width: '260px' }, 'bot roster')
  registerPane('workspace', 'workspace', { placement: 'main', uncloseable: true }, 'chat')

  declareDefaultTree(split('row', [group(['sessions', 'bots']), group(['workspace'])]))
  $narrowViewport.set(true)
})

afterEach(() => {
  cleanup()
  $narrowViewport.set(false)
  $layoutTree.set(null)
  $connection.set(null)
  vi.restoreAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  disposers.splice(0).forEach(dispose => dispose())
})

const revealPane = (id: string) => {
  act(() => {
    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode: 'open' } }))
  })
}

const overlayTab = (paneId: string) => document.querySelector<HTMLElement>(`[data-narrow-overlay-tab="${paneId}"]`)

describe('narrow overlay of a stacked zone', () => {
  it('mirrors the zone tab strip so every stacked collapsible stays reachable', () => {
    const { getByTestId, queryByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    // Both zone-mates surface as tabs; the revealed pane's body is on screen.
    expect(overlayTab('sessions')).toBeTruthy()
    expect(overlayTab('bots')).toBeTruthy()
    expect(getByTestId('sessions-body')).toBeTruthy()
    expect(queryByTestId('bots-body')).toBeNull()

    // Clicking the BOTS tab swaps the overlay to the docked pane.
    fireEvent.pointerDown(overlayTab('bots')!, { button: 0 })
    expect(getByTestId('bots-body')).toBeTruthy()
    expect(queryByTestId('sessions-body')).toBeNull()
  })

  it('keeps the stripless form for a zone with a single collapsible', () => {
    // Direct set: declareDefaultTree only ADOPTS into an existing tree — it
    // would keep the beforeEach zone (with bots) instead of replacing it.
    $layoutTree.set(split('row', [group(['sessions']), group(['workspace'])]))

    const { getByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    expect(getByTestId('sessions-body')).toBeTruthy()
    expect(overlayTab('sessions')).toBeNull()
  })

  it('pads the overlay below the native window controls on macOS', () => {
    // Ground truth for "the SESSIONS tab strip slides under the traffic
    // lights on a narrow window" (#110033). The overlay starts at the
    // viewport's top edge, so without a reservation its tab strip sits under
    // the native controls.
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {} })
    $connection.set({ windowButtonPosition: { x: 24, y: 0 } } as never)
    // jsdom has no layout: the overlay fills the viewport's top-left corner.
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 800,
      height: 800,
      left: 0,
      right: 237,
      toJSON: () => ({}),
      top: 0,
      width: 237,
      x: 0,
      y: 0
    } as DOMRect)

    render(<NarrowOverlays />)
    revealPane('sessions')

    // Controls rect = { x: 0, y: 0, width: 24 + 58, height: 34 }; the
    // reservation is the overlap's bottom edge, so the strip starts below it.
    const overlay = document.querySelector<HTMLElement>('[data-narrow-overlay="sessions"]')
    expect(overlay?.style.paddingTop).toBe('34px')
    // The reserved band stays a window-drag target, as in a docked TreeGroup.
    const dragSpacer = overlay?.querySelector<HTMLElement>('[aria-hidden="true"]')
    expect(dragSpacer).not.toBeNull()
    expect(dragSpacer?.style.height).toBe('34px')
  })

  it('honors the user drag width, not the declared width, when the zone collapsed', () => {
    // The sash writes a widthOverride per shown pane of the zone; the overlay
    // must size from the same resolution the docked zone uses (fixedTrackSize
    // = declared max() refined by overrides), not from data.width alone.
    // jsdom's CSSOM drops the min() wrapper from style.width, so the rendered
    // width is unobservable here — assert the overlay's own resolution
    // (narrowOverlayWidth, the pure helper the component styles from) against
    // the live tree + store the mounted overlay saw.
    setPaneWidthOverride('sessions', 170)
    setPaneWidthOverride('bots', 170)

    const { container } = render(<NarrowOverlays />)

    revealPane('bots')

    const overlay = container.querySelector<HTMLElement>('[data-narrow-overlay]')
    expect(overlay).toBeTruthy()

    const tree = $layoutTree.get()!
    const bots = registry.getArea('panes').find(p => p.id === 'bots')!

    const width = narrowOverlayWidth(
      {
        paneFor: id => registry.getArea('panes').find(p => p.id === id),
        paneGone: () => false,
        overrides: $paneStates.get()
      },
      tree,
      bots
    )

    // The user dragged the zone to 170px; the declared 260px must lose.
    expect(width).toBe('170px')
    expect(width).not.toBe('260px')

    // And the seamless fallback: without a tree the declared width still
    // sizes the overlay.
    expect(narrowOverlayWidth({ paneFor: () => undefined, paneGone: () => false, overrides: {} }, null, bots)).toBe(
      '260px'
    )
  })
})

// The `display.hover_reveal_file_browser` toggle (#45928): `pointerHoverReveal:
// false` takes a pane out of the pointer-hover path while manual reveal
// (PANE_TOGGLE_REVEAL_EVENT) keeps working.
describe('pointer hover reveal toggle', () => {
  it.each([
    ['static', false as const],
    ['live getter', () => false]
  ])('keeps manual reveal when pointer hover is disabled (%s)', (_form, pointerHoverReveal) => {
    registerPane('files', 'files', { collapsible: true, placement: 'right', pointerHoverReveal, width: '240px' }, 'file tree')
    $layoutTree.set(split('row', [group(['sessions']), group(['workspace']), group(['files'])]))

    const { container } = render(<NarrowOverlays />)
    const hoverStrip = (side: 'left' | 'right') => container.querySelector<HTMLElement>(`[data-pane-hover-strip="${side}"]`)
    const overlay = (id: string) => container.querySelector(`[data-narrow-overlay="${id}"]`)

    // The right edge's only collapsible opted out — no hover-intent strip for
    // it at all (the old edge trigger stays unrendered); the left edge keeps
    // its strip for the default panes.
    expect(hoverStrip('right')).toBeNull()
    expect(hoverStrip('left')).toBeTruthy()

    // Pointer hover reveals only the hoverable pane — never the opted-out one.
    fireEvent.mouseEnter(hoverStrip('left')!)
    expect(overlay('sessions')).toBeTruthy()
    expect(overlay('files')).toBeNull()

    // Manual/event reveal still opens the opted-out pane.
    revealPane('files')
    expect(overlay('files')).toBeTruthy()
    expect(overlay('sessions')).toBeNull()
  })
})
