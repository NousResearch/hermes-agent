import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'
import { $displayTimestamps, setDisplayTimestampsFromConfig } from '@/store/display-timestamps'

import { TimelineTimestamp } from './timeline-timestamp'
import { formatMessageTimestamp } from './timestamp'

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

// #103608: the default label is friendly/duration-shaped. Millisecond wall-clock
// text (and the start → end range that used to carry it) survives only in the
// hover tooltip and the precise formatters.
describe('TimelineTimestamp default labels', () => {
  // Local wall clock, so the assertions don't depend on the runner's timezone.
  const started = new Date(2026, 4, 1, 13, 2, 3, 456)
  const start = started.getTime() / 1000
  const MILLISECOND_CLOCK = /\d{1,2}:\d{2}:\d{2}\.\d{3}/
  const PRECISE_RANGE = /\d{1,2}:\d{2}:\d{2}\.\d{3}.*→.*\d{1,2}:\d{2}:\d{2}\.\d{3}/

  const stampText = (container: HTMLElement) =>
    container.querySelector('[data-slot="timeline-timestamp"]')?.textContent?.trim()

  beforeEach(() => {
    $displayTimestamps.set(true)
  })

  it('shows a duration for a settled event instead of its millisecond range', () => {
    const { container } = render(<TimelineTimestamp completedAt={start + 21} timestamp={start} />)

    expect(stampText(container)).toBe('21s')
  })

  it('collapses a sub-second step to "<1s"', () => {
    const { container } = render(<TimelineTimestamp completedAt={start + 0.277} timestamp={start} />)

    expect(stampText(container)).toBe('<1s')
  })

  it('keeps the precise start → end range on the tooltip', () => {
    const { container } = render(<TimelineTimestamp completedAt={start + 21} timestamp={start} />)
    const stamp = container.querySelector('[data-slot="timeline-timestamp"]')

    expect(stamp?.getAttribute('title')).toMatch(PRECISE_RANGE)
    expect(stampText(container)).not.toMatch(MILLISECOND_CLOCK)
  })

  it('renders a friendly day-and-clock label for a message row', () => {
    const { container } = render(<TimelineTimestamp friendly timestamp={start} />)

    expect(stampText(container)).toBe(formatMessageTimestamp(started, TRANSLATIONS.en.assistant.thread))
    expect(stampText(container)).not.toMatch(MILLISECOND_CLOCK)
  })

  it('falls back to the plain clock for an event with no completion time', () => {
    const { container } = render(<TimelineTimestamp timestamp={start} />)

    expect(stampText(container)).toBe(
      new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(started)
    )
  })

  it('renders the label inside a real <time> element carrying the instant', () => {
    const { container } = render(<TimelineTimestamp completedAt={start + 21} timestamp={start} />)

    expect(container.querySelector('time')?.getAttribute('datetime')).toBe(started.toISOString())
  })
})
