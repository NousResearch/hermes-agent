import type { Translations } from '@/i18n'

// Schedule presets for the cron editor. A preset's `expr` is its 9:00 AM seed
// for a blank job; once picked, the editable parts (time, weekday, day of
// month, minute) come from the job's own expression, not from this seed.
export interface ScheduleOption {
  expr?: string
  value: string
}

export const SCHEDULE_OPTIONS: ReadonlyArray<ScheduleOption> = [
  { expr: '0 9 * * *', value: 'daily' },
  { expr: '0 9 * * 1-5', value: 'weekdays' },
  { expr: '0 9 * * 1', value: 'weekly' },
  { expr: '0 9 1 * *', value: 'monthly' },
  { expr: '0 * * * *', value: 'hourly' },
  { expr: '*/15 * * * *', value: 'every-15-minutes' },
  { value: 'custom' }
]

export type ScheduleField = 'dayOfMonth' | 'dayOfWeek' | 'minute' | 'time'

// Which parts of the expression each preset lets the user pick. Presets not
// listed here (every-15-minutes, custom) have no part controls.
export const PRESET_FIELDS: Readonly<Record<string, readonly ScheduleField[]>> = {
  daily: ['time'],
  weekdays: ['time'],
  weekly: ['dayOfWeek', 'time'],
  monthly: ['dayOfMonth', 'time'],
  hourly: ['minute']
}

// Monday-first, matching how people read a week; cron numbers Sunday 0.
export const WEEKDAY_ORDER = ['1', '2', '3', '4', '5', '6', '0'] as const

// Months shorter than this skip a monthly job pinned past it.
export const SHORTEST_MONTH_DAYS = 28

export interface ScheduleParts {
  dayOfMonth: number
  dayOfWeek: number
  hour: number
  minute: number
}

const DEFAULT_PARTS: ScheduleParts = { dayOfMonth: 1, dayOfWeek: 1, hour: 9, minute: 0 }

const PRESET_EXPR: Readonly<Record<string, (parts: ScheduleParts) => string>> = {
  daily: p => `${p.minute} ${p.hour} * * *`,
  weekdays: p => `${p.minute} ${p.hour} * * 1-5`,
  weekly: p => `${p.minute} ${p.hour} * * ${p.dayOfWeek}`,
  monthly: p => `${p.minute} ${p.hour} ${p.dayOfMonth} * *`,
  hourly: p => `${p.minute} * * * *`,
  'every-15-minutes': () => '*/15 * * * *'
}

export function cronParts(expr: string): null | string[] {
  const parts = expr.trim().replace(/\s+/g, ' ').split(' ')

  return parts.length === 5 ? parts : null
}

function isIntegerToken(value: string): boolean {
  return /^\d+$/.test(value)
}

function boundedToken(value: string, min: number, max: number, fallback: number): number {
  if (!isIntegerToken(value)) {
    return fallback
  }

  const parsed = Number(value)

  return parsed >= min && parsed <= max ? parsed : fallback
}

// The editable parts of an expression. A field the expression doesn't pin to
// one integer (a range, a step, a wildcard) keeps its default, so switching
// presets still lands on a sensible time.
export function scheduleParts(expr: string): ScheduleParts {
  const parts = cronParts(expr)

  if (!parts) {
    return DEFAULT_PARTS
  }

  const [minute, hour, dayOfMonth, , dayOfWeek] = parts

  return {
    dayOfMonth: boundedToken(dayOfMonth, 1, 31, DEFAULT_PARTS.dayOfMonth),
    // Cron accepts 7 for Sunday too; normalize so the weekday picker matches.
    dayOfWeek: boundedToken(dayOfWeek, 0, 7, DEFAULT_PARTS.dayOfWeek) % 7,
    hour: boundedToken(hour, 0, 23, DEFAULT_PARTS.hour),
    minute: boundedToken(minute, 0, 59, DEFAULT_PARTS.minute)
  }
}

// The expression a preset fires on with these parts; undefined for "custom",
// whose expression the user types.
export function presetScheduleExpr(preset: string, parts: ScheduleParts): string | undefined {
  return PRESET_EXPR[preset]?.(parts)
}

const pad2 = (value: number): string => String(value).padStart(2, '0')

// `HH:MM` for a native time input.
export function timeInputValue(parts: ScheduleParts): string {
  return `${pad2(parts.hour)}:${pad2(parts.minute)}`
}

// Apply a native time input's `HH:MM`. A cleared or partial value keeps the
// current time rather than writing an expression the scheduler would reject.
export function partsWithTime(parts: ScheduleParts, value: string): ScheduleParts {
  const match = /^(\d{1,2}):(\d{2})/.exec(value)

  if (!match) {
    return parts
  }

  const hour = Number(match[1])
  const minute = Number(match[2])

  return hour <= 23 && minute <= 59 ? { ...parts, hour, minute } : parts
}

const isTimed = ([minute, hour, , month]: string[]): boolean =>
  isIntegerToken(minute) && isIntegerToken(hour) && month === '*'

// Which preset owns an expression; first match wins, anything else is custom.
const PRESET_MATCHERS: ReadonlyArray<readonly [string, (parts: string[]) => boolean]> = [
  ['daily', parts => isTimed(parts) && parts[2] === '*' && parts[4] === '*'],
  ['weekdays', parts => isTimed(parts) && parts[2] === '*' && parts[4] === '1-5'],
  ['weekly', parts => isTimed(parts) && parts[2] === '*' && isIntegerToken(parts[4])],
  ['monthly', parts => isTimed(parts) && parts[4] === '*' && isIntegerToken(parts[2])],
  ['hourly', ([minute, ...rest]) => isIntegerToken(minute) && rest.every(token => token === '*')]
]

export function scheduleOptionForExpr(expr: string): ScheduleOption {
  const normalized = expr.trim().replace(/\s+/g, ' ')
  const custom = SCHEDULE_OPTIONS[SCHEDULE_OPTIONS.length - 1]
  const exactMatch = SCHEDULE_OPTIONS.find(option => option.expr === normalized)
  const parts = cronParts(normalized)

  if (exactMatch || !parts) {
    return exactMatch ?? custom
  }

  const preset = PRESET_MATCHERS.find(([, matches]) => matches(parts))?.[0]

  return SCHEDULE_OPTIONS.find(option => option.value === preset) ?? custom
}

function dayName(value: string, c: Translations['cron']): string {
  return c.days[value] ?? c.dayFallback(value)
}

export function formatCronTime(minute: string, hour: string): string {
  const numericHour = Number(hour)
  const numericMinute = Number(minute)

  if (!Number.isInteger(numericHour) || !Number.isInteger(numericMinute)) {
    return `${hour}:${minute}`
  }

  return new Date(2000, 0, 1, numericHour, numericMinute).toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit'
  })
}

const SUMMARIES: Readonly<Record<string, (parts: string[], time: string, c: Translations['cron']) => string>> = {
  daily: (_, time, c) => c.everyDayAt(time),
  weekdays: (_, time, c) => c.weekdaysAt(time),
  weekly: ([, , , , dayOfWeek], time, c) => c.everyDayOfWeekAt(dayName(dayOfWeek, c), time),
  monthly: ([, , dayOfMonth], time, c) => c.monthlyOnDayAt(dayOfMonth, time),
  hourly: ([minute], _, c) => (minute === '0' ? c.topOfHour : c.everyHourAt(minute.padStart(2, '0')))
}

export function scheduleSummary(option: ScheduleOption, expr: string, c: Translations['cron']): string {
  const parts = cronParts(expr)
  const summarize = SUMMARIES[option.value]

  if (!parts || !summarize) {
    return c.scheduleHints[option.value] ?? ''
  }

  return summarize(parts, formatCronTime(parts[0], parts[1]), c)
}
