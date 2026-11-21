import { describe, expect, it } from 'vitest'

import { canGoBack, canGoForward, nextHistoryPosition } from './history-position'

describe('history position', () => {
  it('starts with nowhere to go on a fresh window', () => {
    const start = nextHistoryPosition({ index: 0, last: 0 }, 'POP', 0)

    expect(canGoBack(start)).toBe(false)
    expect(canGoForward(start)).toBe(false)
  })

  it('offers Back after a push and Forward after going back', () => {
    const pushed = nextHistoryPosition({ index: 0, last: 0 }, 'PUSH', 1)
    expect([canGoBack(pushed), canGoForward(pushed)]).toEqual([true, false])

    const back = nextHistoryPosition(pushed, 'POP', 0)
    expect([canGoBack(back), canGoForward(back)]).toEqual([false, true])
  })

  it('drops the forward entries when a new page is pushed from the middle', () => {
    let position = nextHistoryPosition({ index: 0, last: 0 }, 'PUSH', 1)
    position = nextHistoryPosition(position, 'PUSH', 2)
    position = nextHistoryPosition(position, 'POP', 1)
    position = nextHistoryPosition(position, 'PUSH', 2)

    expect(canGoForward(position)).toBe(false)
  })

  it('keeps Forward through a replace, which only swaps the current entry', () => {
    let position = nextHistoryPosition({ index: 0, last: 0 }, 'PUSH', 1)
    position = nextHistoryPosition(position, 'POP', 0)
    position = nextHistoryPosition(position, 'REPLACE', 0)

    expect(canGoForward(position)).toBe(true)
  })

  it('knows the entries behind a reloaded window but not the ones ahead of it', () => {
    const reloaded = nextHistoryPosition({ index: 0, last: 0 }, 'POP', 4)

    expect([canGoBack(reloaded), canGoForward(reloaded)]).toEqual([true, false])
  })
})
