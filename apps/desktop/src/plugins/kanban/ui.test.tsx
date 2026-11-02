// Regression tests for the board-page crash of 2026-08-24: two task_comments
// rows written by a raw-SQL bypass carried created_at as the STRING
// '2026-08-24 06:39:32' instead of epoch seconds. ago() fed the string into
// relativeTime() ('…' * 1000 -> NaN) and Intl.RelativeTimeFormat.format threw
// "Value need to be finite number", taking the whole kanban page down
// ("plugin:kanban:kanban:page failed to render"). The guard contract: ago()
// returns null for anything that is not a finite number of seconds; every
// call site already renders null as nothing.
import { describe, expect, it } from 'vitest'

import { ago } from './ui'

describe('ago() guards against non-finite backend timestamps', () => {
  it('formats normal epoch seconds', () => {
    const out = ago(Math.floor(Date.now() / 1000) - 120)

    expect(typeof out).toBe('string')
    expect(out!.length).toBeGreaterThan(0)
  })

  it('returns null for the corrupt datetime-string shape seen in the wild', () => {
    // Pre-fix this line THROWS: RangeError: Value need to be finite number
    // for Intl.RelativeTimeFormat.prototype.format().
    expect(ago('2026-08-24 06:39:32' as unknown as number)).toBeNull()
  })

  it('returns null for NaN, Infinity, 0, null, undefined', () => {
    expect(ago(Number.NaN)).toBeNull()
    expect(ago(Number.POSITIVE_INFINITY)).toBeNull()
    expect(ago(0)).toBeNull()
    expect(ago(null)).toBeNull()
    expect(ago(undefined)).toBeNull()
  })
})
