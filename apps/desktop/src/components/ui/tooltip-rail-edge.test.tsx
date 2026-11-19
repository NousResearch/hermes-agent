import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { RootTooltipProvider, Tip } from './tooltip'

const rect = (x: number, y: number, width: number, height: number): DOMRect =>
  ({ top: y, left: x, right: x + width, bottom: y + height, width, height, x, y, toJSON: () => ({}) }) as DOMRect

/** The rail's real shape: `.thread-timeline-ticks` is a 48x300 scrolling strip
 *  floating mid-window (the rail is `top: 50%; translateY(-50%)`), its
 *  `.thread-timeline-track` is `position: relative`, and `.thread-timeline-tick`
 *  is a 7px absolutely positioned button that sits flush against the strip's clip
 *  box at both ends. Chromium-like geometry, installed BEFORE render. */
function mockRailGeometry(tickTop: number) {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    if (this.hasAttribute('data-rail')) {
      return rect(800, 200, 48, 300)
    }

    if (this.tagName === 'BUTTON') {
      return rect(800, 200 + tickTop, 48, 7)
    }

    return rect(0, 0, 1024, 768)
  })

  // floating-ui reads a clipping ancestor's inner size from clientWidth/Height,
  // which jsdom always reports as 0 — mirror the mocked rects there too. The
  // viewport boundary takes the same path, so the fallback is the window size.
  // Spied on the prototype's own getters, so `vi.restoreAllMocks()` puts the real
  // ones back — no descriptor to clean up by hand.
  for (const [name, size, viewport] of [
    ['clientWidth', 48, 1024],
    ['clientHeight', 300, 768]
  ] as const) {
    vi.spyOn(Element.prototype, name, 'get').mockImplementation(function (this: Element) {
      return this.hasAttribute('data-rail') ? size : viewport
    })
  }
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

// A tick at either end of the rail is flush with the strip's clip box. Radix
// feeds `collisionPadding` (12) into the hide (`referenceHidden`) middleware,
// which insets that clip box: a trigger shorter than the padding, sitting on the
// edge, reads as fully scrolled out and its bubble mounts into `visibility:
// hidden` — a mark that looks dead on hover. Middle ticks have room, which is
// why only the first and last marks were reported.
it.each([
  ['first', 0],
  ['middle', 140],
  ['last', 293]
])('opens a visible bubble on the %s mark of a scrolling rail', async (_position, tickTop) => {
  mockRailGeometry(tickTop)

  render(
    <RootTooltipProvider>
      <div data-rail style={{ left: 800, overflowY: 'auto', position: 'absolute', top: 200 }}>
        <div style={{ position: 'relative' }}>
          <Tip label={`Prompt at ${tickTop}`} placement="right-rail">
            <button style={{ height: 7, position: 'absolute', top: tickTop }}>tick</button>
          </Tip>
        </div>
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
