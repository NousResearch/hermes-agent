import { describe, expect, it } from 'vitest'

import { type FirstRunPick, firstRunText, nextFirstRun } from './routine-first-run'

// Thursday 19 November 2026, in the test's local zone.
const at = (day: number, hour: number, minute = 0) => new Date(2026, 10, day, hour, minute).getTime()
const NOW = at(19, 10, 30)

const pick = (patch: Partial<FirstRunPick>): FirstRunPick => ({
  date: '2026-11-25',
  frequency: 'daily',
  monthday: '1',
  time: '09:00',
  weekday: '1',
  ...patch
})

describe('the first run of a routine still being set up', () => {
  it('says nothing before a frequency is picked, or for words the page cannot read', () => {
    expect(nextFirstRun(null, NOW)).toBeNull()
    expect(nextFirstRun(pick({ frequency: 'custom' }), NOW)).toBeNull()
    expect(nextFirstRun(pick({ time: '' }), NOW)).toBeNull()
  })

  it('runs a daily routine today while its time is still ahead, else tomorrow', () => {
    expect(nextFirstRun(pick({ time: '14:00' }), NOW)).toEqual({ at: new Date(at(19, 14)), style: 'relative' })
    expect(nextFirstRun(pick({ time: '09:00' }), NOW)).toEqual({ at: new Date(at(20, 9)), style: 'relative' })
    expect(firstRunText(nextFirstRun(pick({ time: '09:00' }), NOW)!, 'en-US', NOW)).toBe('tomorrow, 9:00 AM')
    expect(firstRunText(nextFirstRun(pick({ time: '14:00' }), NOW)!, 'en-US', NOW)).toBe('today, 2:00 PM')
  })

  it('runs an hourly routine at the next top of the hour', () => {
    expect(nextFirstRun(pick({ frequency: 'hourly' }), NOW)).toEqual({ at: new Date(at(19, 11)), style: 'relative' })
  })

  it('skips the weekend for a weekday routine, and dates a start past tomorrow', () => {
    const friday = at(20, 18)
    const next = nextFirstRun(pick({ frequency: 'weekdays' }), friday)!

    expect(next.at).toEqual(new Date(at(23, 9)))
    expect(next.style).toBe('date')
    expect(firstRunText(next, 'en-US', friday)).toBe('Mon, Nov 23, 9:00 AM')
    expect(nextFirstRun(pick({ frequency: 'weekdays' }), NOW)?.style).toBe('relative')
  })

  it('always dates a weekly or monthly routine', () => {
    const weekly = nextFirstRun(pick({ frequency: 'weekly', weekday: '5' }), NOW)!

    expect(weekly).toEqual({ at: new Date(at(20, 9)), style: 'date' })
    expect(firstRunText(weekly, 'en-US', NOW)).toBe('Fri, Nov 20, 9:00 AM')
    expect(nextFirstRun(pick({ frequency: 'weekly', weekday: '4', time: '08:00' }), NOW)?.at).toEqual(new Date(at(26, 8)))
    expect(nextFirstRun(pick({ frequency: 'monthly', monthday: '19', time: '11:00' }), NOW)).toEqual({
      at: new Date(at(19, 11)),
      style: 'date'
    })
    expect(nextFirstRun(pick({ frequency: 'monthly', monthday: '1' }), NOW)?.at).toEqual(new Date(2026, 11, 1, 9, 0))
  })

  it('skips months without the day, as the scheduler does', () => {
    const january = new Date(2027, 0, 31, 12).getTime()

    expect(nextFirstRun(pick({ frequency: 'monthly', monthday: '31' }), january)?.at).toEqual(new Date(2027, 2, 31, 9, 0))
  })

  it('names the one run of a one-off routine, with the year once it is not this year', () => {
    const once = nextFirstRun(pick({ date: '2026-11-25', frequency: 'once', time: '16:15' }), NOW)!

    expect(once).toEqual({ at: new Date(at(25, 16, 15)), style: 'once' })
    expect(firstRunText(once, 'en-US', NOW)).toBe('Wed, Nov 25, 4:15 PM')
    expect(firstRunText(nextFirstRun(pick({ date: '2027-01-04', frequency: 'once' }), NOW)!, 'en-US', NOW)).toBe(
      'Mon, Jan 4, 2027, 9:00 AM'
    )
    expect(nextFirstRun(pick({ date: '', frequency: 'once' }), NOW)).toBeNull()
  })
})
