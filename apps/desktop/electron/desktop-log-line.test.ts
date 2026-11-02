import { describe, expect, it } from 'vitest'

import { formatDesktopLogLine, formatLogStamp } from './desktop-log-line'

describe('formatDesktopLogLine', () => {
  it('stamps lines with the local wall-clock time, in the shape Python logs use', () => {
    // Built from local components, so the assertion holds in every time zone:
    // the stamp must name the same local moment agent.log would print for it.
    const moment = new Date(2026, 8, 3, 7, 5, 9, 42)
    const line = formatDesktopLogLine('[boot] Resolving Hermes backend', formatLogStamp(moment))

    const match = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3}) \[hermes\] (.*)$/.exec(line)

    expect(match).not.toBeNull()
    const [, y, mo, d, h, mi, s, ms, text] = match!
    expect(new Date(+y, +mo - 1, +d, +h, +mi, +s, +ms).getTime()).toBe(moment.getTime())
    expect(text).toBe('[boot] Resolving Hermes backend')
  })
})
