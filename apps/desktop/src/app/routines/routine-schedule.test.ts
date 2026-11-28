import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'

import {
  composeSchedule,
  defaultPick,
  describePick,
  describeSchedule,
  formatMoment,
  machineZoneNote,
  pickForSchedule,
  runDay,
  type SchedulePick
} from './routine-schedule'
import { routinesCopy } from './routines-copy'

const NOW = new Date(2026, 10, 12, 12, 0).getTime()

const words = {
  copy: routinesCopy('en'),
  dayName: (day: string) => TRANSLATIONS.en.cron.days[day] ?? day,
  locale: 'en-US'
}

const pick = (patch: Partial<SchedulePick>): SchedulePick => ({ ...defaultPick(NOW), ...patch })

describe('When to run: the picker composes what the scheduler accepts', () => {
  it('turns each preset into its cron expression, Once into a local time, Custom into its words', () => {
    expect(composeSchedule(pick({ frequency: 'daily', time: '07:00' }))).toBe('0 7 * * *')
    expect(composeSchedule(pick({ frequency: 'weekdays', time: '07:30' }))).toBe('30 7 * * 1-5')
    expect(composeSchedule(pick({ frequency: 'weekly', time: '16:00', weekday: '5' }))).toBe('0 16 * * 5')
    expect(composeSchedule(pick({ frequency: 'monthly', monthday: '15', time: '09:00' }))).toBe('0 9 15 * *')
    expect(composeSchedule(pick({ frequency: 'hourly' }))).toBe('0 * * * *')
    expect(composeSchedule(pick({ date: '2026-11-14', frequency: 'once', time: '08:15' }))).toBe('2026-11-14T08:15')
    expect(composeSchedule(pick({ custom: '  every 2h ', frequency: 'custom' }))).toBe('every 2h')
    expect(composeSchedule(pick({ custom: '', frequency: 'custom' }))).toBe('')
  })

  it('composes nothing, and reads nothing back, for a frequency whose time was cleared', () => {
    const cleared = [
      pick({ frequency: 'daily', time: '' }),
      pick({ frequency: 'weekdays', time: '' }),
      pick({ frequency: 'weekly', time: '' }),
      pick({ frequency: 'monthly', time: '' }),
      pick({ date: '2026-11-14', frequency: 'once', time: '' })
    ]

    for (const entry of cleared) {
      expect(composeSchedule(entry)).toBe('')
      expect(describePick(entry, words)).toBeNull()
    }
  })

  it('starts a new routine on every day at 09:00, with Once on tomorrow', () => {
    expect(defaultPick(NOW)).toMatchObject({ date: '2026-11-13', frequency: 'daily', time: '09:00' })
  })

  it('reopens a stored schedule in the preset it came from, round trip', () => {
    for (const expr of ['0 7 * * *', '30 7 * * 1-5', '0 16 * * 5', '0 9 15 * *', '0 * * * *']) {
      expect(composeSchedule(pickForSchedule({ expr, kind: 'cron' }, expr, NOW))).toBe(expr)
    }

    expect(pickForSchedule({ expr: '0 16 * * 7', kind: 'cron' }, '', NOW)).toMatchObject({ frequency: 'weekly', weekday: '0' })
    expect(pickForSchedule({ kind: 'once', run_at: '2026-11-14T08:15:00-06:00' }, '', NOW)).toMatchObject({
      date: '2026-11-14',
      frequency: 'once',
      time: '08:15'
    })
  })

  it('keeps anything the presets cannot show as Custom, in words the scheduler reads back the same', () => {
    expect(pickForSchedule({ expr: '*/15 * * * *', kind: 'cron' }, '', NOW)).toMatchObject({
      custom: '*/15 * * * *',
      frequency: 'custom'
    })
    expect(pickForSchedule({ kind: 'interval', minutes: 2880 }, 'every 2880m', NOW)).toMatchObject({
      custom: 'every 2d',
      frequency: 'custom'
    })
    expect(pickForSchedule({ kind: 'interval', minutes: 90 }, '', NOW).custom).toBe('every 90m')
  })
})

describe('the plain-words read-back', () => {
  it('says each preset in words', () => {
    expect(describePick(pick({ frequency: 'daily', time: '07:00' }), words)).toBe('Every day at 7:00 AM')
    expect(describePick(pick({ frequency: 'weekdays', time: '07:00' }), words)).toBe('Monday to Friday at 7:00 AM')
    expect(describePick(pick({ frequency: 'weekly', time: '16:00', weekday: '5' }), words)).toBe('Every Friday at 4:00 PM')
    expect(describePick(pick({ frequency: 'monthly', monthday: '1', time: '09:00' }), words)).toBe(
      'Day 1 of every month at 9:00 AM'
    )
    expect(describePick(pick({ date: '2026-11-14', frequency: 'once', time: '08:15' }), words)).toBe(
      'Once, on Nov 14, 2026 at 8:15 AM'
    )
  })

  it('reads stored intervals and stepped cron in words, and never shows a raw expression', () => {
    expect(describeSchedule({ kind: 'interval', minutes: 2880 }, words)).toBe('Every 2 days')
    expect(describeSchedule({ kind: 'interval', minutes: 30 }, words)).toBe('Every 30 minutes')
    expect(describeSchedule({ expr: '*/15 * * * *', kind: 'cron' }, words)).toBe('Every 15 minutes')
    expect(describeSchedule({ expr: '0 */2 * * *', kind: 'cron' }, words)).toBe('Every 2 hours')
    expect(describeSchedule({ expr: '15 * * * *', kind: 'cron' }, words)).toBe('Every hour at :15')
    expect(describeSchedule({ expr: '0 9 1,15 * *', kind: 'cron' }, words)).toBe('On a custom schedule')
    expect(describeSchedule(undefined, words)).toBe('On a custom schedule')
  })

  it('leaves Custom words unread rather than guessing', () => {
    expect(describePick(pick({ custom: 'every 2h', frequency: 'custom' }), words)).toBeNull()
  })
})

describe('the day a run happened', () => {
  it('is today, yesterday, or its date, with the year', () => {
    expect(runDay(new Date(2026, 10, 12, 7).getTime(), 'en-US', NOW)).toEqual({ date: 'Nov 12, 2026', kind: 'today' })
    expect(runDay(new Date(2026, 10, 11, 23).getTime(), 'en-US', NOW)).toEqual({
      date: 'Nov 11, 2026',
      kind: 'yesterday'
    })
    expect(runDay(new Date(2026, 10, 3, 7).getTime(), 'en-US', NOW)).toEqual({ date: 'Nov 3, 2026', kind: 'date' })
    expect(runDay(new Date(2025, 11, 30, 7).getTime(), 'en-US', NOW)).toEqual({ date: 'Dec 30, 2025', kind: 'date' })
    expect(runDay('not a date', 'en-US', NOW)).toBeNull()
  })
})

describe('a moment named in the reader’s locale', () => {
  it('reads today and tomorrow relatively, in the app locale, and keeps an off-year’s year', () => {
    expect(formatMoment(new Date(2026, 10, 12, 7).getTime(), 'en-US', NOW)).toBe('today, 7:00 AM')
    expect(formatMoment(new Date(2026, 10, 13, 9).getTime(), 'en-US', NOW)).toBe('tomorrow, 9:00 AM')
    // A moment in another calendar year names it, so "Nov 6" never reads as this year's.
    expect(formatMoment(new Date(2026, 10, 3, 7).getTime(), 'en-US', NOW)).toBe('Nov 3, 7:00 AM')
    expect(formatMoment(new Date(2025, 11, 30, 7).getTime(), 'en-US', NOW)).toBe('Dec 30, 2025, 7:00 AM')
  })

  it('never throws on a malformed date, in the moment or the day', () => {
    expect(formatMoment('not a date', 'en-US', NOW)).toBe('')
    expect(formatMoment(Number.NaN, 'en-US', NOW)).toBe('')
    expect(runDay('not a date', 'en-US', NOW)).toBeNull()
  })
})

describe('A schedule read on a machine in another time zone', () => {
  // A stamp for one instant, written with an offset `shift` minutes away from the viewer's own.
  const stamp = (shift: number) => {
    const at = new Date(2026, 10, 13, 7, 0)
    const offset = -at.getTimezoneOffset() + shift
    const wall = new Date(at.getTime() + offset * 60_000).toISOString().slice(0, 19)
    const abs = Math.abs(offset)

    return `${wall}${offset < 0 ? '-' : '+'}${String(Math.floor(abs / 60)).padStart(2, '0')}:${String(abs % 60).padStart(2, '0')}`
  }

  const daily = { expr: '0 7 * * *', kind: 'cron' as const }

  it('reads plainly when the machine keeps the viewer’s time', () => {
    expect(describeSchedule(daily, words, stamp(0))).toBe('Every day at 7:00 AM')
    expect(describeSchedule(daily, words, null)).toBe('Every day at 7:00 AM')
  })

  it('names the machine’s zone beside a clock time that is not the viewer’s', () => {
    expect(describeSchedule(daily, words, stamp(-60))).toMatch(/^Every day at 7:00 AM \(UTC[+−]\d+(:\d\d)?\)$/)
    expect(machineZoneNote('2026-11-13T07:00:00-06:00')).toBe(
      new Date('2026-11-13T07:00:00-06:00').getTimezoneOffset() === 360 ? null : 'UTC−6'
    )
    expect(machineZoneNote('2026-11-13T07:00:00+05:30')).toBe(
      new Date('2026-11-13T07:00:00+05:30').getTimezoneOffset() === -330 ? null : 'UTC+5:30'
    )
  })

  it('names the machine’s zone beside a Once time written in another zone, as for the presets', () => {
    expect(describeSchedule({ kind: 'once', run_at: stamp(0) }, words)).toBe('Once, on Nov 13, 2026 at 7:00 AM')
    expect(describeSchedule({ kind: 'once', run_at: stamp(-60) }, words)).toMatch(
      /^Once, on Nov 13, 2026 at 6:00 AM \(UTC[+−]\d+(:\d\d)?\)$/
    )
  })

  it('leaves schedules without a wall-clock time alone', () => {
    expect(describeSchedule({ kind: 'interval', minutes: 30 }, words, stamp(-60))).toBe('Every 30 minutes')
    expect(describeSchedule({ expr: '*/15 * * * *', kind: 'cron' }, words, stamp(-60))).toBe('Every 15 minutes')
  })
})
