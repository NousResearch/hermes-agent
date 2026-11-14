import { describe, expect, it } from 'vitest'

import { THREAD_SCROLL_BOTTOM } from '@/store/thread-scroll'

import { shouldPersistLanding } from './list'

// A mounted window of 5000px with a 600px viewport: the scroll range is
// 4400px, so a remembered offset beyond that can never be shown — the
// restore clamps at the top of the window instead (#131165).
const base = { readerOwnsViewport: false, scrollHeight: 5000, clientHeight: 600 }

describe('shouldPersistLanding', () => {
  it('refuses a landing whose target is still beyond the scroll range', () => {
    expect(shouldPersistLanding({ ...base, target: { fromBottom: 7416.47, kind: 'offset' } })).toBe(false)
    expect(shouldPersistLanding({ ...base, target: { fromBottom: 4401, kind: 'offset' } })).toBe(false)
  })

  it('keeps a landing the transcript could actually show', () => {
    expect(shouldPersistLanding({ ...base, target: { fromBottom: 4400, kind: 'offset' } })).toBe(true)
    expect(shouldPersistLanding({ ...base, target: { fromBottom: 800, kind: 'offset' } })).toBe(true)
  })

  it('keeps a reader-owned position even while the load is unfinished', () => {
    expect(
      shouldPersistLanding({ ...base, readerOwnsViewport: true, target: { fromBottom: 7416, kind: 'offset' } })
    ).toBe(true)
  })

  it('never refuses a bottom target', () => {
    expect(shouldPersistLanding({ ...base, target: THREAD_SCROLL_BOTTOM })).toBe(true)
  })

  it('refuses any offset while the scroll range is empty', () => {
    expect(
      shouldPersistLanding({
        readerOwnsViewport: false,
        target: { fromBottom: 1, kind: 'offset' },
        scrollHeight: 600,
        clientHeight: 600
      })
    ).toBe(false)
  })
})
