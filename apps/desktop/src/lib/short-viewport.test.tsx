import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { useShortViewport } from './short-viewport'

class FakeVisualViewport extends EventTarget {
  height = 900
  offsetTop = 0
  scale = 1
}

function Shortness() {
  return <span data-testid="short">{String(useShortViewport())}</span>
}

const short = () => screen.getByTestId('short').textContent

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

/** Layout height and the visible height are separate on purpose: `interactive-widget=resizes-content` shrinks
 *  the layout with the keyboard, but iOS Safari and other hosts only shrink the visual viewport. */
function stubViewport({
  height,
  layout = 900,
  scale = 1
}: { height?: number; layout?: number; scale?: number } = {}) {
  const viewport = new FakeVisualViewport()

  viewport.height = height ?? layout
  viewport.scale = scale
  // No layout media query matches: only the visible height can make this short.
  vi.stubGlobal('matchMedia', (query: string) => ({
    addEventListener: () => {},
    matches: false,
    media: query,
    removeEventListener: () => {}
  }))
  vi.stubGlobal('innerHeight', layout)
  vi.stubGlobal('visualViewport', viewport)

  return viewport
}

function resize(viewport: FakeVisualViewport) {
  act(() => {
    viewport.dispatchEvent(new Event('resize'))
  })
}

it('reports a short viewport when only the visual viewport shrinks, with layout height unchanged', () => {
  const viewport = stubViewport()

  render(<Shortness />)
  expect(short()).toBe('false')

  viewport.height = 380  // the keyboard came up; window.innerHeight did not move
  resize(viewport)

  expect(short()).toBe('true')

  viewport.height = 900  // keyboard dismissed
  resize(viewport)

  expect(short()).toBe('false')
})

it('does not read a pinch as a short viewport', () => {
  const viewport = stubViewport()

  render(<Shortness />)
  viewport.height = 300  // zoomed in: 300 x 3 still fills the 900px layout
  viewport.scale = 3
  resize(viewport)

  expect(short()).toBe('false')
})

it('still reports short from the layout media query alone', () => {
  stubViewport({ layout: 900 })
  vi.stubGlobal('matchMedia', (query: string) => ({
    addEventListener: () => {},
    matches: query === '(max-height: 520px)',
    media: query,
    removeEventListener: () => {}
  }))

  render(<Shortness />)

  expect(short()).toBe('true')
})
