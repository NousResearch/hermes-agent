import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { visibleClock } from './visible-clock'

const noteVisibleTime = vi.hoisted(() => vi.fn())

vi.mock('@/store/desktop-metrics', () => ({ noteVisibleTime }))

let visibility: DocumentVisibilityState = 'visible'

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(0)
  visibility = 'visible'
  vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() => visibility)
  noteVisibleTime.mockClear()
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('visibleClock', () => {
  it('credits only spans the window was visible', () => {
    const clock = visibleClock(11 * 60_000)

    vi.setSystemTime(60_000)
    visibility = 'hidden'
    clock.credit() // visible → hidden: the visible minute counts
    vi.setSystemTime(5 * 60_000)
    clock.credit() // a tick while hidden: nothing
    visibility = 'visible'
    clock.credit() // hidden → visible: nothing
    vi.setSystemTime(7 * 60_000)
    clock.credit()

    expect(noteVisibleTime.mock.calls).toEqual([[60_000], [2 * 60_000]])
  })

  it('a suspend with the window up (no ticks for hours) credits nothing for the gap', () => {
    const clock = visibleClock(11 * 60_000)

    vi.setSystemTime(10 * 60_000)
    clock.credit()
    vi.setSystemTime(10 * 60_000 + 8 * 3_600_000) // resumed: the first tick after the sleep
    clock.credit()
    vi.setSystemTime(10 * 60_000 + 8 * 3_600_000 + 10 * 60_000)
    clock.credit()

    expect(noteVisibleTime.mock.calls).toEqual([[10 * 60_000], [10 * 60_000]])
  })
})
