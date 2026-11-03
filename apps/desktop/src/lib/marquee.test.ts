import { describe, expect, it } from 'vitest'

import { armMarquee, disarmMarquee } from './marquee'

function pointerEvent(el: HTMLElement) {
  return { currentTarget: el } as unknown as React.PointerEvent<HTMLElement>
}

function setWidths(el: HTMLElement, scrollWidth: number, clientWidth: number) {
  Object.defineProperty(el, 'scrollWidth', { configurable: true, value: scrollWidth })
  Object.defineProperty(el, 'clientWidth', { configurable: true, value: clientWidth })
}

describe('marquee helpers', () => {
  it('arms the CSS animation with the measured distance when the text overflows', () => {
    const el = document.createElement('span')

    setWidths(el, 300, 100)
    armMarquee(pointerEvent(el))

    expect(el.dataset.marquee).toBe('true')
    expect(el.style.getPropertyValue('--marquee-d')).toBe('200px')
    expect(el.style.getPropertyValue('--marquee-t')).toMatch(/s$/)
  })

  it('leaves short text alone so it never wiggles', () => {
    const el = document.createElement('span')

    setWidths(el, 100, 100)
    armMarquee(pointerEvent(el))

    expect(el.dataset.marquee).toBeUndefined()
    expect(el.style.getPropertyValue('--marquee-d')).toBe('')
  })

  it('tolerates 2px of sub-pixel rounding slack', () => {
    const el = document.createElement('span')

    setWidths(el, 102, 100)
    armMarquee(pointerEvent(el))

    expect(el.dataset.marquee).toBeUndefined()
  })

  it('disarms on pointer leave', () => {
    const el = document.createElement('span')

    setWidths(el, 300, 100)
    armMarquee(pointerEvent(el))
    expect(el.dataset.marquee).toBe('true')

    disarmMarquee(pointerEvent(el))
    expect(el.dataset.marquee).toBeUndefined()
  })
})
