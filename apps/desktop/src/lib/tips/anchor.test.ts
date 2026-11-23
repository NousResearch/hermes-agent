import { afterEach, expect, it, vi } from 'vitest'

import { resolveTipAnchor } from './anchor'

const box = (left: number, top: number, width: number, height: number) =>
  ({ left, top, right: left + width, bottom: top + height, width, height }) as DOMRect

function mount(html: string) {
  window.document.body.innerHTML = html
  const anchor = window.document.querySelector<HTMLElement>('[data-tour="composer"]')!
  vi.spyOn(anchor, 'getBoundingClientRect').mockReturnValue(box(400, 600, 640, 120))

  return anchor
}

/** What the browser reports on top at a point. */
function onTop(element: Element | null) {
  window.document.elementFromPoint = () => element
}

afterEach(() => {
  window.document.body.innerHTML = ''
  vi.restoreAllMocks()
})

it('points at an anchor that is on screen with nothing over it', () => {
  const anchor = mount('<main><div data-tour="composer"><textarea></textarea></div></main>')
  onTop(anchor.querySelector('textarea'))

  expect(resolveTipAnchor(window.document, ['[data-tour="composer"]'])).toBe(anchor)
})

it('skips an anchor that something else covers', () => {
  mount('<main><div data-tour="composer"></div></main><section id="page"></section>')
  onTop(window.document.querySelector('#page'))

  expect(resolveTipAnchor(window.document, ['[data-tour="composer"]'])).toBeNull()
})

it('skips an anchor in a region taken out of use, whatever is on top', () => {
  for (const attribute of ['inert', 'aria-hidden="true"']) {
    const anchor = mount(`<main ${attribute}><div data-tour="composer"></div></main>`)
    onTop(anchor)

    expect(resolveTipAnchor(window.document, ['[data-tour="composer"]'])).toBeNull()
  }
})

it('falls through to the next target when the first is covered', () => {
  mount('<main><div data-tour="composer"></div><button data-tour="model-pill"></button></main>')
  const pill = window.document.querySelector<HTMLElement>('[data-tour="model-pill"]')!
  vi.spyOn(pill, 'getBoundingClientRect').mockReturnValue(box(900, 560, 80, 24))
  onTop(pill)

  expect(resolveTipAnchor(window.document, ['[data-tour="composer"]', '[data-tour="model-pill"]'])).toBe(pill)
})
