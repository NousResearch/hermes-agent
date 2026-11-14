import type { CronJobSchedule } from '@/hermes'

import type { RoutinesCopy } from './routines-copy'

// "When to run" (proposal point 1: one schedule picker, read back in plain words). The picker composes
// the schedule strings the scheduler already accepts (cron/jobs.py parse_schedule): a 5-field cron
// expression for the recurring presets, an ISO local time for Once, and the user's own words for Custom.

export type Frequency = 'custom' | 'daily' | 'hourly' | 'monthly' | 'once' | 'weekdays' | 'weekly'

export const FREQUENCIES: readonly Frequency[] = ['once', 'hourly', 'daily', 'weekdays', 'weekly', 'monthly', 'custom']

export interface SchedulePick {
  frequency: Frequency
  /** HH:MM, 24h. */
  time: string
  /** Cron day of week, 0 (Sunday) to 6. */
  weekday: string
  /** Day of the month, 1–31. */
  monthday: string
  /** YYYY-MM-DD, for Once. */
  date: string
  /** Custom: plain words ("every 2 hours") or a cron expression. */
  custom: string
}

const INT = /^\d+$/
const STEP = /^\*\/(\d+)$/
const ISO_LOCAL = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/

const pad = (value: number | string) => String(value).padStart(2, '0')

function isoDate(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
}

/** A new routine's picker: every day at 09:00; Once defaults to tomorrow. */
export function defaultPick(nowMs = Date.now()): SchedulePick {
  const tomorrow = new Date(nowMs)
  tomorrow.setDate(tomorrow.getDate() + 1)

  return { custom: '', date: isoDate(tomorrow), frequency: 'daily', monthday: '1', time: '09:00', weekday: '1' }
}

function splitTime(time: string): [number, number] {
  const [hour = '9', minute = '0'] = time.split(':')

  return [Number(hour) || 0, Number(minute) || 0]
}

const COMPOSE: Record<Frequency, (pick: SchedulePick) => string> = {
  custom: pick => pick.custom.trim(),
  daily: pick => cron(pick, '*', '*'),
  hourly: () => '0 * * * *',
  monthly: pick => cron(pick, pick.monthday || '1', '*'),
  once: pick => (pick.date ? `${pick.date}T${pick.time || '09:00'}` : ''),
  weekdays: pick => cron(pick, '*', '1-5'),
  weekly: pick => cron(pick, '*', pick.weekday || '1')
}

function cron(pick: SchedulePick, dayOfMonth: string, dayOfWeek: string): string {
  const [hour, minute] = splitTime(pick.time)

  return `${minute} ${hour} ${dayOfMonth} * ${dayOfWeek}`
}

/** The schedule string the backend stores for this pick ('' when it is not complete). */
export function composeSchedule(pick: SchedulePick): string {
  return COMPOSE[pick.frequency](pick)
}

/** A cron expression's fields, when it is one the presets can show. */
interface PresetMatch {
  frequency: Frequency
  time?: string
  weekday?: string
  monthday?: string
}

function presetForCron(expr: string): null | PresetMatch {
  const parts = expr.trim().split(/\s+/)

  if (parts.length !== 5) {
    return null
  }

  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts

  if (expr.trim().replace(/\s+/g, ' ') === '0 * * * *') {
    return { frequency: 'hourly' }
  }

  if (!INT.test(minute) || !INT.test(hour) || month !== '*') {
    return null
  }

  const time = `${pad(hour)}:${pad(minute)}`

  if (dayOfMonth === '*') {
    if (dayOfWeek === '*') {
      return { frequency: 'daily', time }
    }

    if (dayOfWeek === '1-5') {
      return { frequency: 'weekdays', time }
    }

    if (INT.test(dayOfWeek) && Number(dayOfWeek) <= 7) {
      return { frequency: 'weekly', time, weekday: String(Number(dayOfWeek) % 7) }
    }

    return null
  }

  return INT.test(dayOfMonth) && dayOfWeek === '*' ? { frequency: 'monthly', monthday: dayOfMonth, time } : null
}

/** An interval in the scheduler's own words ("every 2d"), so editing a routine keeps its cadence. */
export function intervalPhrase(minutes: number): string {
  if (minutes % 1440 === 0) {
    return `every ${minutes / 1440}d`
  }

  return minutes % 60 === 0 ? `every ${minutes / 60}h` : `every ${minutes}m`
}

/** The stored schedule as the text Custom would hold. */
export function scheduleText(schedule: CronJobSchedule | null | undefined, fallback = ''): string {
  if (schedule?.kind === 'interval' && schedule.minutes) {
    return intervalPhrase(schedule.minutes)
  }

  return schedule?.expr || schedule?.run_at || schedule?.display || fallback
}

/** The picker for an existing routine: a preset when its schedule is one, else Custom with its text. */
export function pickForSchedule(
  schedule: CronJobSchedule | null | undefined,
  fallback: string,
  nowMs = Date.now()
): SchedulePick {
  const base = defaultPick(nowMs)

  if (schedule?.kind === 'once' && schedule.run_at) {
    const match = ISO_LOCAL.exec(schedule.run_at)

    if (match) {
      return { ...base, date: `${match[1]}-${match[2]}-${match[3]}`, frequency: 'once', time: `${match[4]}:${match[5]}` }
    }
  }

  const preset = schedule?.kind === 'cron' && schedule.expr ? presetForCron(schedule.expr) : null

  if (preset) {
    return { ...base, ...preset, time: preset.time ?? base.time }
  }

  return { ...base, custom: scheduleText(schedule, fallback), frequency: 'custom' }
}

// ── Plain-words read-back ─────────────────────────────────────────────────────────────────────────

export interface ScheduleWords {
  copy: RoutinesCopy
  locale: string
  /** Weekday names by cron day ('0'–'6'), from the catalog. */
  dayName: (day: string) => string
}

export function formatClock(time: string, locale: string): string {
  const [hour, minute] = splitTime(time)

  return new Date(2000, 0, 1, hour, minute).toLocaleTimeString(locale, { hour: 'numeric', minute: '2-digit' })
}

function formatDay(date: string, locale: string): string {
  const [year, month, day] = date.split('-').map(Number)

  return new Date(year, (month || 1) - 1, day || 1).toLocaleDateString(locale, {
    day: 'numeric',
    month: 'short',
    year: 'numeric'
  })
}

function describeInterval(minutes: number, copy: RoutinesCopy): string {
  if (minutes % 1440 === 0) {
    return copy.schedule.everyDays(minutes / 1440)
  }

  return minutes % 60 === 0 ? copy.schedule.everyHours(minutes / 60) : copy.schedule.everyMinutes(minutes)
}

/** Cron shapes beyond the presets that still read naturally: every N minutes or hours, hourly at :MM. */
function describeCronSteps(expr: string, copy: RoutinesCopy): null | string {
  const [minute, hour, dayOfMonth, month, dayOfWeek] = expr.trim().split(/\s+/)

  if (dayOfMonth !== '*' || month !== '*' || dayOfWeek !== '*') {
    return null
  }

  const minuteStep = STEP.exec(minute)

  if (minuteStep && hour === '*') {
    return copy.schedule.everyMinutes(Number(minuteStep[1]))
  }

  const hourStep = STEP.exec(hour)

  if (hourStep && minute === '0') {
    return copy.schedule.everyHours(Number(hourStep[1]))
  }

  return INT.test(minute) && hour === '*' ? copy.schedule.hourlyAt(pad(minute)) : null
}

const DESCRIBE: Record<Exclude<Frequency, 'custom'>, (pick: SchedulePick, words: ScheduleWords) => string> = {
  daily: (pick, { copy, locale }) => copy.schedule.daily(formatClock(pick.time, locale)),
  hourly: (_pick, { copy }) => copy.schedule.hourly,
  monthly: (pick, { copy, locale }) => copy.schedule.monthly(pick.monthday, formatClock(pick.time, locale)),
  once: (pick, { copy, locale }) => copy.schedule.once(formatDay(pick.date, locale), formatClock(pick.time, locale)),
  weekdays: (pick, { copy, locale }) => copy.schedule.weekdays(formatClock(pick.time, locale)),
  weekly: (pick, { copy, dayName, locale }) => copy.schedule.weekly(dayName(pick.weekday), formatClock(pick.time, locale))
}

/** A picker state in plain words; null for Custom text the page cannot read (shown as typed). */
export function describePick(pick: SchedulePick, words: ScheduleWords): null | string {
  if (pick.frequency !== 'custom') {
    return pick.frequency === 'once' && !pick.date ? null : DESCRIBE[pick.frequency](pick, words)
  }

  return null
}

// Presets whose sentence names a wall-clock time on the machine. Once is an instant, hourly has no clock.
const WALL_CLOCK: ReadonlySet<Frequency> = new Set(['daily', 'weekdays', 'weekly', 'monthly'])
const STAMP_OFFSET = /(Z|[+-]\d{2}:?\d{2})$/

/** "UTC−6" when a machine stamp's own offset is not the viewer's at that instant, else null. A
 *  schedule's clock is the machine's wall time; every moment beside it is shown in the viewer's. */
export function machineZoneNote(stamp: null | string | undefined): null | string {
  const raw = stamp?.match(STAMP_OFFSET)?.[1]
  const at = raw ? new Date(stamp!) : null

  if (!raw || !at || Number.isNaN(at.valueOf())) {
    return null
  }

  const digits = raw === 'Z' ? '+0000' : raw.replace(':', '')
  const minutes = (digits.startsWith('-') ? -1 : 1) * (Number(digits.slice(1, 3)) * 60 + Number(digits.slice(3)))

  if (minutes === -at.getTimezoneOffset()) {
    return null
  }

  const rest = Math.abs(minutes) % 60

  return `UTC${minutes < 0 ? '−' : '+'}${Math.floor(Math.abs(minutes) / 60)}${rest ? `:${String(rest).padStart(2, '0')}` : ''}`
}

/** A stored schedule in plain words. Falls back to the generic "custom schedule" phrase, so a simple
 *  surface never shows a raw expression (developer mode shows it beside this). `machineStamp` is any
 *  stamp the machine wrote for this job (its next run): a clock time is marked with the machine's
 *  zone when that is not the viewer's. */
export function describeSchedule(
  schedule: CronJobSchedule | null | undefined,
  words: ScheduleWords,
  machineStamp?: null | string
): string {
  if (schedule?.kind === 'interval' && schedule.minutes) {
    return describeInterval(schedule.minutes, words.copy)
  }

  const pick = pickForSchedule(schedule, '')

  if (pick.frequency !== 'custom') {
    const text = describePick(pick, words)
    const zone = text && WALL_CLOCK.has(pick.frequency) ? machineZoneNote(machineStamp) : null

    return text ? (zone ? `${text} (${zone})` : text) : words.copy.schedule.custom
  }

  if (schedule?.kind === 'cron' && schedule.expr) {
    return describeCronSteps(schedule.expr, words.copy) ?? words.copy.schedule.custom
  }

  return words.copy.schedule.custom
}

// ── Moments ───────────────────────────────────────────────────────────────────────────────────────

const DAY_MS = 86_400_000

function startOfDay(ms: number): number {
  const date = new Date(ms)
  date.setHours(0, 0, 0, 0)

  return date.getTime()
}

/** "today, 7:00 AM" / "tomorrow, 9:00 AM" / "Nov 6, 4:00 PM" for a run or the next run. */
export function formatMoment(at: Date | number | string, locale: string, nowMs = Date.now()): string {
  const date = at instanceof Date ? at : new Date(at)

  if (Number.isNaN(date.valueOf())) {
    return ''
  }

  const time = date.toLocaleTimeString(locale, { hour: 'numeric', minute: '2-digit' })
  const days = Math.round((startOfDay(date.getTime()) - startOfDay(nowMs)) / DAY_MS)

  if (Math.abs(days) <= 1) {
    const relative = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' }).format(days, 'day')

    return `${relative}, ${time}`
  }

  const day = date.toLocaleDateString(locale, {
    day: 'numeric',
    month: 'short',
    ...(new Date(nowMs).getFullYear() === date.getFullYear() ? {} : { year: 'numeric' })
  })

  return `${day}, ${time}`
}

/** Sentence-initial form of a moment (a table cell, a run row's heading). */
export function capitalized(text: string, locale: string): string {
  return text ? text.charAt(0).toLocaleUpperCase(locale) + text.slice(1) : text
}

export function formatDuration(seconds: number, locale: string): string {
  const [value, unit] =
    seconds < 60
      ? [Math.max(1, Math.round(seconds)), 'second']
      : seconds < 3600
        ? [Math.round(seconds / 60), 'minute']
        : [Math.round(seconds / 360) / 10, 'hour']

  return new Intl.NumberFormat(locale, { style: 'unit', unit, unitDisplay: 'narrow' }).format(value)
}
