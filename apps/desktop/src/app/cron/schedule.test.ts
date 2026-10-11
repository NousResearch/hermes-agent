import { describe, expect, it } from 'vitest'

import { partsWithTime, presetScheduleExpr, scheduleOptionForExpr, scheduleParts } from './schedule'

describe('cron schedule parts', () => {
  it('rebuilds a preset expression with the time the user picked', () => {
    const parts = partsWithTime(scheduleParts('0 9 * * 1'), '17:45')

    expect(presetScheduleExpr('weekly', parts)).toBe('45 17 * * 1')
    expect(scheduleOptionForExpr('45 17 * * 1').value).toBe('weekly')
  })

  it('keeps the picked time and day when switching presets', () => {
    const parts = scheduleParts('30 7 15 * *')

    expect(presetScheduleExpr('weekdays', parts)).toBe('30 7 * * 1-5')
    expect(presetScheduleExpr('monthly', parts)).toBe('30 7 15 * *')
    expect(presetScheduleExpr('hourly', parts)).toBe('30 * * * *')
    expect(presetScheduleExpr('custom', parts)).toBeUndefined()
  })

  it('never writes an expression from a cleared or out-of-range time', () => {
    const parts = scheduleParts('15 6 * * *')

    expect(partsWithTime(parts, '')).toEqual(parts)
    expect(partsWithTime(parts, '25:00')).toEqual(parts)
  })

  it('reads cron Sunday as 7 the same as 0', () => {
    expect(scheduleParts('0 9 * * 7').dayOfWeek).toBe(0)
  })
})
