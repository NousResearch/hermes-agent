import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { setTimestampFormatFromConfig } from '@/lib/time'
import { $displayTimestamps, setDisplayTimestampsFromConfig } from '@/store/display-timestamps'

import { TimelineTimestamp } from './timeline-timestamp'

afterEach(() => {
  cleanup()
  setTimestampFormatFromConfig(undefined)
})

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

  it('reacts to the configured hour cycle in both the label and precise title', () => {
    $displayTimestamps.set(true)
    const local = new Date(2026, 4, 1, 20, 2, 3, 456)
    const seconds = local.getTime() / 1000
    setTimestampFormatFromConfig('%H:%M')

    const { container } = render(<TimelineTimestamp timestamp={seconds} />)
    const stamp = container.querySelector('[data-slot="timeline-timestamp"]')

    const clockOptions: Intl.DateTimeFormatOptions = {
      fractionalSecondDigits: 3,
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit'
    }

    const titleOptions: Intl.DateTimeFormatOptions = {
      day: 'numeric',
      fractionalSecondDigits: 3,
      hour: 'numeric',
      minute: '2-digit',
      month: 'short',
      second: '2-digit',
      year: 'numeric'
    }

    expect(stamp?.textContent).toBe(new Intl.DateTimeFormat(undefined, { ...clockOptions, hourCycle: 'h23' }).format(local))
    expect(stamp?.getAttribute('title')).toBe(
      new Intl.DateTimeFormat(undefined, { ...titleOptions, hourCycle: 'h23' }).format(local)
    )

    act(() => setTimestampFormatFromConfig('%I:%M %p'))

    expect(stamp?.textContent).toBe(new Intl.DateTimeFormat(undefined, { ...clockOptions, hourCycle: 'h12' }).format(local))
    expect(stamp?.getAttribute('title')).toBe(
      new Intl.DateTimeFormat(undefined, { ...titleOptions, hourCycle: 'h12' }).format(local)
    )
  })
})
