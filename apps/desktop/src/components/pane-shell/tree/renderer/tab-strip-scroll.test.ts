import { describe, expect, it } from 'vitest'

import { tabStripScrollLeft } from './tab-strip-scroll'

describe('tabStripScrollLeft', () => {
  // 300px window over 900px of tabs + "+" → 600px of scroll range.
  const strip = (over: Partial<Parameters<typeof tabStripScrollLeft>[0]> = {}) =>
    tabStripScrollLeft({
      clientWidth: 300,
      last: false,
      scrollLeft: 0,
      scrollWidth: 900,
      tabEnd: 0,
      tabStart: 0,
      ...over
    })

  it('scrolls to the end for the last tab so the "+" comes with it', () => {
    expect(strip({ last: true, scrollLeft: 0, tabEnd: 880, tabStart: 780 })).toBe(600)
  })

  it('leaves the offset alone when the tab is already fully visible', () => {
    expect(strip({ scrollLeft: 200, tabEnd: 400, tabStart: 300 })).toBe(200)
  })

  it('scrolls left to the tab start when it sits before the window', () => {
    expect(strip({ scrollLeft: 400, tabEnd: 200, tabStart: 100 })).toBe(100)
  })

  it('scrolls right just far enough to reveal a tab past the window', () => {
    expect(strip({ scrollLeft: 0, tabEnd: 450, tabStart: 350 })).toBe(150)
  })

  it('never exceeds the scrollable range', () => {
    expect(strip({ last: true, scrollWidth: 250 })).toBe(0)
    expect(strip({ scrollLeft: 0, tabEnd: 1200, tabStart: 1100 })).toBe(600)
  })
})
