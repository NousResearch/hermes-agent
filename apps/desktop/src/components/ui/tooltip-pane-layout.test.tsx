import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { RootTooltipProvider, Tip } from './tooltip'

const rect = (x: number, y: number, width: number, height: number): DOMRect =>
  ({ top: y, left: x, right: x + width, bottom: y + height, width, height, x, y, toJSON: () => ({}) }) as DOMRect

/** Chromium-like geometry, installed BEFORE render because the boundary is
 *  resolved in a layout effect during the mount commit: a zero-rect
 *  [data-tree-group] host, a real trigger inside it, a full viewport around. */
function mockGeometry(triggerLaidOut = () => true) {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    if (this.hasAttribute('data-tree-group')) {
      return rect(0, 0, 0, 0)
    }

    if (this.tagName === 'BUTTON') {
      return triggerLaidOut() ? rect(100, 100, 40, 30) : rect(0, 0, 0, 0)
    }

    return rect(0, 0, 1024, 768)
  })

  // floating-ui reads a boundary's inner size from clientWidth/clientHeight,
  // which jsdom always reports as 0 — mirror the mocked rects there too.
  for (const [name, size] of [
    ['clientWidth', 1024],
    ['clientHeight', 768]
  ] as const) {
    Object.defineProperty(Element.prototype, name, {
      configurable: true,
      get(this: Element) {
        return this.hasAttribute('data-tree-group') ? 0 : size
      }
    })
  }
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  delete (Element.prototype as { clientWidth?: number }).clientWidth
  delete (Element.prototype as { clientHeight?: number }).clientHeight
})

// The floating-composer host (floating-surface.tsx) carries a data-tree-group
// of its own while being display:contents, so composer controls resolve it as
// their pane. A zero-rect boundary clips every side of a fully visible
// trigger and the tip mounts straight into visibility:hidden.
it('opens a tip whose nearest tree-group host has no layout', async () => {
  mockGeometry()

  render(
    <RootTooltipProvider>
      <div data-tree-group="floating-host">
        <Tip label="Model · custom:test: model-x">
          <button>Trigger</button>
        </Tip>
      </div>
    </RootTooltipProvider>
  )

  const trigger = screen.getByRole('button')
  fireEvent.pointerEnter(trigger)
  fireEvent.pointerMove(trigger, { pointerType: 'mouse' })

  await vi.waitFor(
    () => {
      // eslint-disable-next-line no-restricted-globals -- the portal mounts on the live document
      expect(document.querySelector('[data-slot="tooltip-content"]')).not.toBeNull()
    },
    { timeout: 2000 }
  )

  // eslint-disable-next-line no-restricted-globals -- the portal mounts on the live document
  const wrapper = document.querySelector<HTMLElement>('[data-radix-popper-content-wrapper]')

  expect(wrapper).not.toBeNull()
  expect(wrapper?.style.visibility).not.toBe('hidden')
})

// The live app mounts composer `Tip`s before their trigger has geometry (the
// floating surface is laid out later). The pane must be resolved when the tip
// OPENS — a once-at-mount resolution keeps the zero-rect host forever and every
// composer tip stays visibility:hidden.
it('resolves the pane at open time, not at mount', async () => {
  let laidOut = false
  mockGeometry(() => laidOut)

  render(
    <RootTooltipProvider>
      <div data-tree-group="floating-host">
        <Tip label="Add context">
          <button>Trigger</button>
        </Tip>
      </div>
    </RootTooltipProvider>
  )

  laidOut = true
  const trigger = screen.getByRole('button')
  fireEvent.pointerEnter(trigger)
  fireEvent.pointerMove(trigger, { pointerType: 'mouse' })

  await vi.waitFor(
    () => {
      // eslint-disable-next-line no-restricted-globals -- the portal mounts on the live document
      expect(document.querySelector('[data-radix-popper-content-wrapper]')).not.toBeNull()
    },
    { timeout: 2000 }
  )

  await vi.waitFor(() => {
    // eslint-disable-next-line no-restricted-globals -- the portal mounts on the live document
    expect(document.querySelector<HTMLElement>('[data-radix-popper-content-wrapper]')?.style.visibility).not.toBe(
      'hidden'
    )
  })
})
