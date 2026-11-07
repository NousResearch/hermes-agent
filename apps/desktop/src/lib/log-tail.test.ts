import { describe, expect, it } from 'vitest'

import { newestLogTimestamp, pickFreshestLogTail } from './log-tail'

// #146682: gui.log is written only in mode="gui", so a panel that always tails it shows
// frozen, days-old lines as current activity when the Desktop is attached to the messaging
// gateway. These tests pin the selection rule, not the panel wiring.
describe('newestLogTimestamp', () => {
  it('parses asctime (comma milliseconds) and picks the newest line', () => {
    const lines = [
      '2026-10-31 18:58:49,627 WARNING older',
      '2026-10-31 19:04:26,001 INFO newer',
      'Traceback (most recent call last):' // continuation lines carry no stamp
    ]

    expect(newestLogTimestamp(lines)).toBe(Date.UTC(2026, 9, 31, 19, 4, 26, 1))
  })

  it('returns null when nothing carries a timestamp', () => {
    expect(newestLogTimestamp(['no stamp here', '   '])).toBeNull()
  })

  it('tolerates dotted fractions and a T separator', () => {
    expect(newestLogTimestamp(['2026-10-31T19:04:26.5 INFO x'])).toBe(Date.UTC(2026, 9, 31, 19, 4, 26, 500))
  })
})

describe('pickFreshestLogTail', () => {
  it('prefers the tail with the newest line over file order', () => {
    const gui = { file: 'gui', lines: ['2026-10-27 09:00:00,000 WARNING stale asctime'] }
    const gateway = { file: 'gateway', lines: ['2026-10-31 19:04:26,001 INFO live'] }

    expect(pickFreshestLogTail([gui, gateway])?.file).toBe('gateway')
    expect(pickFreshestLogTail([gateway, gui])?.file).toBe('gateway')
  })

  it('keeps the old behaviour when only one tail has content', () => {
    const gui = { file: 'gui', lines: ['2026-10-31 19:04:26,001 INFO live'] }
    const gateway = { file: 'gateway', lines: [] as string[] }

    expect(pickFreshestLogTail([gui, gateway])?.file).toBe('gui')
  })

  it('falls back to the first non-empty tail when no line is parseable', () => {
    const gui = { file: 'gui', lines: ['plain text'] }
    const gateway = { file: 'gateway', lines: ['more plain text'] }

    expect(pickFreshestLogTail([gui, gateway])?.file).toBe('gui')
  })

  it('returns null when every tail is empty', () => {
    expect(pickFreshestLogTail([{ file: 'gui', lines: [] }, { file: 'gateway', lines: [] }])).toBeNull()
    expect(pickFreshestLogTail([])).toBeNull()
  })
})
