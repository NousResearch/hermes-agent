import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $displayTimestamps, setDisplayTimestampsFromConfig } from '@/store/display-timestamps'

import { TimelineTimestamp } from './timeline-timestamp'
import { formatClockTimestamp } from './timestamp'

afterEach(cleanup)

beforeEach(() => {
  $displayTimestamps.set(false)
})

describe('setDisplayTimestampsFromConfig', () => {
  it('accepts boolean and string forms, defaulting off', () => {
    setDisplayTimestampsFromConfig(true)
    expect($displayTimestamps.get()).toBe(true)

    setDisplayTimestampsFromConfig(false)
    expect($displayTimestamps.get()).toBe(false)

    setDisplayTimestampsFromConfig('true')
    expect($displayTimestamps.get()).toBe(true)

    setDisplayTimestampsFromConfig(undefined)
    expect($displayTimestamps.get()).toBe(false)
  })
})

describe('TimelineTimestamp display.timestamps gate', () => {
  const timestamp = new Date('2026-05-01T00:00:00.000Z').getTime() / 1000

  it('renders nothing while display.timestamps is off (the default)', () => {
    const { container } = render(<TimelineTimestamp timestamp={timestamp} />)

    expect(container.querySelector('[data-slot="timeline-timestamp"]')).toBeNull()
  })

  it('renders the stamp once display.timestamps is on', () => {
    $displayTimestamps.set(true)

    const { container } = render(<TimelineTimestamp timestamp={timestamp} />)

    expect(container.querySelector('[data-slot="timeline-timestamp"]')).toBeTruthy()
  })
})

describe('TimelineTimestamp precision', () => {
  const started = new Date(2026, 4, 1, 16, 30, 3, 456).getTime() / 1000
  const finished = new Date(2026, 4, 1, 16, 32, 9, 789).getTime() / 1000

  // Mirrors the component's own hover formatter, so the assertion survives a
  // locale that renders digits differently (e.g. ar-SA).
  const preciseFormat = (d: Date) =>
    new Intl.DateTimeFormat(undefined, {
      day: 'numeric',
      fractionalSecondDigits: 3,
      hour: 'numeric',
      minute: '2-digit',
      month: 'short',
      second: '2-digit',
      year: 'numeric'
    }).format(d)

  beforeEach(() => {
    $displayTimestamps.set(true)
  })

  it('defaults to the precise range for activity boundaries', () => {
    const { container } = render(<TimelineTimestamp completedAt={finished} timestamp={started} />)
    const text = container.querySelector('[data-slot="timeline-timestamp"]')?.textContent ?? ''

    expect(text).toContain('→')
    // Sub-minute detail is exactly what an activity row must keep: two
    // instants in the same minute stay distinguishable here.
    expect(text).not.toBe(formatClockTimestamp(started))
    expect(text.split('→')[0]?.trim()).not.toBe(formatClockTimestamp(started))
  })

  it('collapses to a single landing clock with no seconds when precision is clock', () => {
    const { container } = render(<TimelineTimestamp completedAt={finished} precision="clock" timestamp={started} />)
    const node = container.querySelector('[data-slot="timeline-timestamp"]')
    const text = node?.textContent ?? ''

    expect(text).not.toContain('→')
    expect(text).toBe(formatClockTimestamp(finished))
    // Full precision is still reachable on hover.
    expect(node?.getAttribute('title') ?? '').toBe(
      `${preciseFormat(new Date(started * 1000))} → ${preciseFormat(new Date(finished * 1000))}`
    )
  })

  it('falls back to the send time in clock mode when the turn never completed', () => {
    const { container } = render(<TimelineTimestamp precision="clock" timestamp={started} />)

    expect(container.querySelector('[data-slot="timeline-timestamp"]')?.textContent).toBe(formatClockTimestamp(started))
  })
})
