import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { registry } from '@/contrib/registry'
import { stubResizeObserver } from '@/test/jsdom'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree, $narrowViewport, declareDefaultTree } from '../store'

import { NarrowOverlays } from './narrow-overlays'

// Ground truth for "the Bots tab is still visible when the sessions sidebar
// collapses on a narrow window". A collapsible pane DOCKED into the sessions
// zone (SESSIONS | BOTS) must leave the grid with the zone, and the narrow
// edge overlay must mirror the zone's tab strip so the docked pane stays
// reachable — not just the zone's first pane.

beforeAll(() => {
  stubResizeObserver()
})

const disposers: (() => void)[] = []
const originalInnerWidth = globalThis.innerWidth

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
  Object.defineProperty(globalThis, 'innerWidth', { configurable: true, value: 400 })
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
  globalThis.document.body.style.touchAction = ''
  Object.defineProperty(globalThis, 'innerWidth', { configurable: true, value: originalInnerWidth })
  $narrowViewport.set(false)
  $layoutTree.set(null)
  disposers.splice(0).forEach(dispose => dispose())
})

const revealPane = (id: string) => {
  act(() => {
    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode: 'open' } }))
  })
}

const overlayTab = (paneId: string) =>
  globalThis.document.querySelector<HTMLElement>(`[data-narrow-overlay-tab="${paneId}"]`)

const swipe = (
  target: Window | Document | Node | Element,
  from: { x: number; y: number },
  to: { x: number; y: number },
  pointerId = 7,
  pointerType = 'touch'
) => {
  fireEvent.pointerDown(target, {
    button: 0,
    clientX: from.x,
    clientY: from.y,
    pointerId,
    pointerType
  })
  fireEvent.pointerMove(target, {
    button: 0,
    clientX: to.x,
    clientY: to.y,
    pointerId,
    pointerType
  })
  fireEvent.pointerUp(target, {
    button: 0,
    clientX: to.x,
    clientY: to.y,
    pointerId,
    pointerType
  })
}

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

  it('opens Sessions with an inward swipe from anywhere in the left half', () => {
    const { getByTestId, queryByTestId } = render(<NarrowOverlays />)

    expect(queryByTestId('sessions-body')).toBeNull()
    expect(globalThis.document.body.style.touchAction).toBe('pan-y')

    // Starts well away from the edge, inside the left half of the viewport.
    swipe(globalThis.document.body, { x: 180, y: 240 }, { x: 236, y: 240 })

    expect(getByTestId('sessions-body')).toBeTruthy()
    expect(globalThis.document.querySelector('[data-sessions-swipe-close="left"]')).toBeTruthy()
  })

  it('opens Sessions for a pen swipe from the left half as well as touch', () => {
    const { getByTestId } = render(<NarrowOverlays />)

    swipe(globalThis.document.body, { x: 120, y: 240 }, { x: 188, y: 240 }, 8, 'pen')

    expect(getByTestId('sessions-body')).toBeTruthy()
  })

  it('ignores right-half, short, vertical, cancelled, mouse and unknown gestures', () => {
    const { queryByTestId } = render(<NarrowOverlays />)
    const screen = globalThis.document.body

    swipe(screen, { x: 220, y: 240 }, { x: 290, y: 240 }, 7)
    expect(queryByTestId('sessions-body')).toBeNull()

    swipe(screen, { x: 120, y: 240 }, { x: 164, y: 243 }, 8)
    expect(queryByTestId('sessions-body')).toBeNull()

    swipe(screen, { x: 120, y: 240 }, { x: 184, y: 340 }, 9)
    expect(queryByTestId('sessions-body')).toBeNull()

    fireEvent.pointerDown(screen, { button: 0, clientX: 120, clientY: 240, pointerId: 10, pointerType: 'touch' })
    fireEvent.pointerCancel(screen, { pointerId: 10, pointerType: 'touch' })
    fireEvent.pointerMove(screen, { button: 0, clientX: 196, clientY: 240, pointerId: 10, pointerType: 'touch' })
    expect(queryByTestId('sessions-body')).toBeNull()

    swipe(screen, { x: 120, y: 240 }, { x: 196, y: 240 }, 11, 'mouse')
    expect(queryByTestId('sessions-body')).toBeNull()

    swipe(screen, { x: 120, y: 240 }, { x: 196, y: 240 }, 12, '')
    expect(queryByTestId('sessions-body')).toBeNull()

    swipe(screen, { x: 120, y: 240 }, { x: 196, y: 240 }, 13, 'unknown')
    expect(queryByTestId('sessions-body')).toBeNull()
  })

  it('keeps the left-half gesture narrow-only and restores the page touch policy on unmount', () => {
    globalThis.document.body.style.touchAction = 'manipulation'
    const { unmount } = render(<NarrowOverlays />)

    expect(globalThis.document.body.style.touchAction).toBe('pan-y')
    unmount()
    expect(globalThis.document.body.style.touchAction).toBe('manipulation')

    $narrowViewport.set(false)
    render(<NarrowOverlays />)
    swipe(globalThis.document.body, { x: 120, y: 240 }, { x: 196, y: 240 }, 14)

    expect(globalThis.document.querySelector('[data-testid="sessions-body"]')).toBeNull()
    expect(globalThis.document.body.style.touchAction).toBe('manipulation')
    globalThis.document.body.style.touchAction = ''
  })

  it('closes Sessions with a swipe back toward the left edge', () => {
    const { getByTestId, queryByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    const pane = globalThis.document.querySelector<HTMLElement>('[data-sessions-swipe-close="left"]')
    expect(pane).toBeTruthy()
    expect(getByTestId('sessions-body')).toBeTruthy()

    swipe(pane!, { x: 190, y: 240 }, { x: 118, y: 245 })

    expect(queryByTestId('sessions-body')).toBeNull()
  })

  it('does not close Sessions for the wrong or vertical swipe direction', () => {
    const { getByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    const pane = globalThis.document.querySelector<HTMLElement>('[data-sessions-swipe-close="left"]')!

    swipe(pane, { x: 120, y: 240 }, { x: 198, y: 242 }, 11)
    expect(getByTestId('sessions-body')).toBeTruthy()

    swipe(pane, { x: 190, y: 240 }, { x: 120, y: 340 }, 12)
    expect(getByTestId('sessions-body')).toBeTruthy()
  })

  it('closes a pinned sessions overlay when the visible chat pane is tapped', () => {
    const { getByTestId, queryByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    expect(getByTestId('sessions-body')).toBeTruthy()

    const dismissTarget = globalThis.document.querySelector<HTMLElement>('[data-narrow-overlay-dismiss]')
    const pane = globalThis.document.querySelector<HTMLElement>('[data-sessions-swipe-close]')
    expect(dismissTarget).toBeTruthy()
    expect(dismissTarget!.tagName).toBe('DIV')
    expect(dismissTarget!.getAttribute('aria-hidden')).toBe('true')
    expect(dismissTarget!.tabIndex).toBe(-1)
    expect(dismissTarget!.classList.contains('z-[var(--z-narrow-overlay-backdrop)]')).toBe(true)
    expect(pane!.classList.contains('z-[var(--z-narrow-overlay)]')).toBe(true)
    expect(pane!.getAttribute('role')).toBe('dialog')
    expect(pane!.getAttribute('aria-modal')).toBe('true')
    expect(pane!.getAttribute('aria-label')).toBe('sessions')
    expect(globalThis.document.activeElement).toBe(pane)

    fireEvent.click(dismissTarget!)

    expect(queryByTestId('sessions-body')).toBeNull()
    expect(globalThis.document.querySelector('[data-narrow-overlay-dismiss]')).toBeNull()
  })

  it('makes background controls inert and restores their focus when the overlay closes', () => {
    const { getByTestId } = render(
      <>
        <button data-testid="background-action" type="button">
          Chat action
        </button>
        <NarrowOverlays />
      </>
    )

    const backgroundAction = getByTestId('background-action') as HTMLButtonElement
    const previousInert = backgroundAction.inert

    backgroundAction.focus()

    revealPane('sessions')

    const pane = globalThis.document.querySelector<HTMLElement>('[data-narrow-overlay-pane]')
    const dismissTarget = globalThis.document.querySelector<HTMLElement>('[data-narrow-overlay-dismiss]')
    expect(backgroundAction.inert).toBe(true)
    expect(globalThis.document.activeElement).toBe(pane)

    fireEvent.click(dismissTarget!)

    expect(backgroundAction.inert).toBe(previousInert)
    expect(globalThis.document.activeElement).toBe(backgroundAction)
  })

  it('keeps desktop hover previews nonmodal without stealing focus', () => {
    const { getByTestId } = render(
      <>
        <button data-testid="background-action" type="button">
          Chat action
        </button>
        <NarrowOverlays />
      </>
    )

    const backgroundAction = getByTestId('background-action') as HTMLButtonElement
    const previousInert = backgroundAction.inert

    backgroundAction.focus()
    fireEvent.mouseEnter(globalThis.document.querySelector<HTMLElement>('[data-narrow-overlay-edge="left"]')!)

    const pane = globalThis.document.querySelector<HTMLElement>('[data-narrow-overlay-pane]')
    expect(pane).toBeTruthy()
    expect(pane!.getAttribute('role')).toBeNull()
    expect(pane!.getAttribute('aria-modal')).toBeNull()
    expect(globalThis.document.querySelector('[data-narrow-overlay-dismiss]')).toBeNull()
    expect(backgroundAction.inert).toBe(previousInert)
    expect(globalThis.document.activeElement).toBe(backgroundAction)
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
})
