import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RootTooltipProvider, Tip } from './tooltip'

const zeroRect = (): DOMRect =>
  ({ top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect

const triggerRect = (): DOMRect =>
  ({ top: 100, left: 100, right: 140, bottom: 130, width: 40, height: 30, x: 100, y: 100, toJSON: () => ({}) }) as DOMRect

const paneRect = (): DOMRect =>
  ({ top: 0, left: 0, right: 1024, bottom: 768, width: 1024, height: 768, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect

/** Install geometry BEFORE render: the boundary-resolving layout effect runs
 *  during the mount commit, so it must already see Chromium-like rects. The
 *  pane host is the [data-tree-group] element itself; the trigger is the
 *  button inside it; everything else is a full-viewport ancestor. */
function mockGeometry(hostRect: () => DOMRect) {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    if (this.closest('[data-tree-group]') === this) {
      return hostRect()
    }

    return this.tagName === 'BUTTON' ? triggerRect() : paneRect()
  })

  // floating-ui reads a boundary's inner rect from clientWidth/clientHeight,
  // which jsdom always reports as 0 — mirror the mocked rect there too.
  Object.defineProperty(Element.prototype, 'clientWidth', {
    configurable: true,
    get(this: Element) {
      return this.closest('[data-tree-group]') === this ? hostRect().width : 1024
    }
  })
  Object.defineProperty(Element.prototype, 'clientHeight', {
    configurable: true,
    get(this: Element) {
      return this.closest('[data-tree-group]') === this ? hostRect().height : 768
    }
  })
}

async function hoverOpen() {
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
  return document.querySelector<HTMLElement>('[data-radix-popper-content-wrapper]')
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  delete (Element.prototype as any).clientWidth
  delete (Element.prototype as any).clientHeight
})

// The floating-composer host (floating-surface.tsx) carries a data-tree-group
// of its own while being display:contents, so composer controls resolve it as
// their pane. A zero-rect boundary clips every side of a fully visible
// trigger and the tip mounts straight into visibility:hidden (#114602).
describe('Tip collision boundary resolution', () => {
  it('ignores a pane without layout instead of hiding the bubble', async () => {
    mockGeometry(zeroRect)

    render(
      <RootTooltipProvider>
        <div data-tree-group="floating-host">
          <Tip label="Model · custom:test: model-x">
            <button>Trigger</button>
          </Tip>
        </div>
      </RootTooltipProvider>
    )

    const wrapper = await hoverOpen()

    expect(wrapper).not.toBeNull()
    expect(wrapper?.style.visibility).not.toBe('hidden')
  })

  it('still clips against a pane that has layout', async () => {
    mockGeometry(paneRect)

    render(
      <RootTooltipProvider>
        <div data-tree-group="pane">
          <Tip label="Details">
            <button>Trigger</button>
          </Tip>
        </div>
      </RootTooltipProvider>
    )

    const wrapper = await hoverOpen()

    expect(wrapper).not.toBeNull()
    expect(wrapper?.style.visibility).not.toBe('hidden')
  })
})
