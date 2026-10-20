import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $paneStates, setPaneHeightOverride } from '@/store/panes'

import {
  KEYBOARD_STEP_PX,
  SIDEBAR_BOTTOM_SECTION_ID,
  SIDEBAR_PINNED_SECTION_ID,
  SIDEBAR_SESSIONS_SECTION_ID
} from './section-resize'
import { BOTTOM_H_VAR, PINNED_H_VAR, SectionSash, type SectionSashEdge, SESSIONS_MIN_H_VAR } from './section-sash'

const PINNED_LABEL = 'Resize Pinned and Sessions'
const BOTTOM_LABEL = 'Resize Sessions'

// jsdom has no layout; each section reports the height a real sidebar would.
const HEIGHTS = { bottom: 100, pinned: 150, sessions: 300 }

function SectionList() {
  const ref = useRef<HTMLDivElement>(null)

  return (
    <div data-testid="list" ref={ref}>
      <div data-sidebar-section="pinned">
        <div data-sidebar="group-content" data-testid="pinned" />
      </div>
      <SectionSash containerRef={ref} edge="pinned-seam" label={PINNED_LABEL} />
      <div data-sidebar-section="sessions" data-testid="sessions" />
      <SectionSash containerRef={ref} edge="bottom-seam" label={BOTTOM_LABEL} />
      <div data-sidebar-section="bottom" data-testid="bottom" />
    </div>
  )
}

function renderList() {
  render(<SectionList />)

  for (const [id, height] of Object.entries(HEIGHTS)) {
    Object.defineProperty(screen.getByTestId(id), 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ bottom: height, height, left: 0, right: 240, toJSON: () => ({}), top: 0, width: 240, x: 0, y: 0 })
    })
  }

  return screen.getByTestId('list')
}

const cssPx = (list: HTMLElement, name: string) => Number.parseFloat(list.style.getPropertyValue(name))
const override = (id: string) => $paneStates.get()[id]?.heightOverride

function press(sash: HTMLElement, clientY = 500) {
  fireEvent.pointerDown(sash, { button: 0, clientY, pointerId: 1, pointerType: 'mouse' })
}

function moveTo(clientY: number) {
  fireEvent.pointerMove(window, { clientY, pointerId: 1, pointerType: 'mouse' })
}

function release(type: 'pointerCancel' | 'pointerUp' = 'pointerUp') {
  fireEvent[type](window, { pointerId: 1, pointerType: 'mouse' })
}

interface SeamCase {
  edge: SectionSashEdge
  label: string
  neighbour: keyof typeof HEIGHTS
  neighbourVar: string
  storeId: string
  /** +1 when dragging down grows the neighbour (it sits above the seam). */
  direction: 1 | -1
}

const SEAMS: SeamCase[] = [
  {
    direction: 1,
    edge: 'pinned-seam',
    label: PINNED_LABEL,
    neighbour: 'pinned',
    neighbourVar: PINNED_H_VAR,
    storeId: SIDEBAR_PINNED_SECTION_ID
  },
  {
    direction: -1,
    edge: 'bottom-seam',
    label: BOTTOM_LABEL,
    neighbour: 'bottom',
    neighbourVar: BOTTOM_H_VAR,
    storeId: SIDEBAR_BOTTOM_SECTION_ID
  }
]

beforeEach(() => {
  window.localStorage.clear()
  $paneStates.set({})
})

afterEach(() => {
  cleanup()
  $paneStates.set({})
  window.document.body.style.cursor = ''
})

describe.each(SEAMS)('SectionSash $edge', seam => {
  it.each([40, -40])('trades height with Sessions while dragging %ipx, and persists it once on release', dy => {
    const list = renderList()
    const total = HEIGHTS[seam.neighbour] + HEIGHTS.sessions

    press(screen.getByRole('separator', { name: seam.label }))
    moveTo(500 + dy)

    // Mid-drag the list repaints from CSS variables alone; the store (and the
    // big sidebar tree subscribed to it) must not churn per pointer frame.
    expect(cssPx(list, seam.neighbourVar)).toBe(HEIGHTS[seam.neighbour] + seam.direction * dy)
    expect(cssPx(list, seam.neighbourVar) + cssPx(list, SESSIONS_MIN_H_VAR)).toBe(total)
    expect(override(seam.storeId)).toBeUndefined()

    release()

    expect(override(seam.storeId)).toBe(cssPx(list, seam.neighbourVar))
    expect(override(SIDEBAR_SESSIONS_SECTION_ID)).toBe(cssPx(list, SESSIONS_MIN_H_VAR))
  })

  it('moves the seam one keyboard step per arrow key, by the same rules as a drag', () => {
    renderList()

    fireEvent.keyDown(screen.getByRole('separator', { name: seam.label }), { key: 'ArrowDown' })

    expect(override(seam.storeId)).toBe(HEIGHTS[seam.neighbour] + seam.direction * KEYBOARD_STEP_PX)
    expect(override(seam.storeId)! + override(SIDEBAR_SESSIONS_SECTION_ID)!).toBe(
      HEIGHTS[seam.neighbour] + HEIGHTS.sessions
    )
  })
})

describe('SectionSash gestures that must not resize', () => {
  it('persists nothing for a press that never moves, including both halves of a double-click', () => {
    renderList()
    const sash = screen.getByRole('separator', { name: BOTTOM_LABEL })

    press(sash)
    moveTo(500) // a jittery device re-sending the press position is still not a drag
    release()
    press(sash)
    release()

    expect($paneStates.get()).toEqual({})
  })

  it('puts back the previous heights and persists nothing when the OS cancels the drag', () => {
    const list = renderList()
    // What ChatSidebar would have rendered from an earlier, saved drag.
    list.style.setProperty(BOTTOM_H_VAR, '90px')

    press(screen.getByRole('separator', { name: BOTTOM_LABEL }))
    moveTo(560)

    expect(list.style.getPropertyValue(BOTTOM_H_VAR)).not.toBe('90px')

    release('pointerCancel')

    expect(list.style.getPropertyValue(BOTTOM_H_VAR)).toBe('90px')
    expect(list.style.getPropertyValue(SESSIONS_MIN_H_VAR)).toBe('')
    expect($paneStates.get()).toEqual({})
    // The gesture is over: later pointer movement no longer resizes anything.
    moveTo(700)
    expect(list.style.getPropertyValue(BOTTOM_H_VAR)).toBe('90px')
  })

  it('ignores presses from buttons other than the primary one', () => {
    const list = renderList()

    fireEvent.pointerDown(screen.getByRole('separator', { name: BOTTOM_LABEL }), { button: 2, clientY: 500 })
    moveTo(560)
    release()

    expect(list.style.getPropertyValue(BOTTOM_H_VAR)).toBe('')
    expect($paneStates.get()).toEqual({})
  })
})

describe('SectionSash double-click', () => {
  it('returns every section to its natural size, whichever seam is clicked', () => {
    for (const label of [PINNED_LABEL, BOTTOM_LABEL]) {
      renderList()

      for (const id of [SIDEBAR_PINNED_SECTION_ID, SIDEBAR_SESSIONS_SECTION_ID, SIDEBAR_BOTTOM_SECTION_ID]) {
        setPaneHeightOverride(id, 200)
      }

      fireEvent.doubleClick(screen.getByRole('separator', { name: label }))

      for (const id of [SIDEBAR_PINNED_SECTION_ID, SIDEBAR_SESSIONS_SECTION_ID, SIDEBAR_BOTTOM_SECTION_ID]) {
        expect(override(id)).toBeUndefined()
      }

      cleanup()
    }
  })
})
