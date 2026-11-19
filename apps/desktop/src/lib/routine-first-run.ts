// When a routine that is still being set up would first run (the create sheet's "First run: …" line
// above its Create button). Pure: the picker's state and a clock in, a moment and how to say it out.
// The schedule runs on the machine's wall clock; the line reads it on the viewer's, which is the
// same clock for every machine that shares the viewer's zone.

export type FirstRunFrequency = 'custom' | 'daily' | 'hourly' | 'monthly' | 'once' | 'weekdays' | 'weekly'

export interface FirstRunPick {
  frequency: FirstRunFrequency
  /** HH:MM, 24h. */
  time: string
  /** Cron day of week, 0 (Sunday) to 6. */
  weekday: string
  /** Day of the month, 1–31. */
  monthday: string
  /** YYYY-MM-DD, for Once. */
  date: string
}

/** `relative`: today or tomorrow, said so. `date`: a weekday and date. `once`: the one run. */
export interface FirstRun {
  at: Date
  style: 'date' | 'once' | 'relative'
}

const CLOCK = /^(\d{1,2}):(\d{2})$/
const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})$/
const DAY_MS = 86_400_000
/** Enough days to reach a day-31 in a run of short months (Jan 31 → Mar 31 is 59). */
const SEARCH_DAYS = 400

function clockOf(time: string): [number, number] | null {
  const match = CLOCK.exec(time)

  return match && Number(match[1]) < 24 ? [Number(match[1]), Number(match[2])] : null
}

/** The first day from today (at `hour:minute`) that `matches` and is still ahead of `nowMs`. */
function nextDay(nowMs: number, clock: [number, number], matches: (day: Date) => boolean): Date | null {
  for (let offset = 0; offset < SEARCH_DAYS; offset += 1) {
    const day = new Date(nowMs)
    day.setDate(day.getDate() + offset)
    day.setHours(clock[0], clock[1], 0, 0)

    if (day.getTime() > nowMs && matches(day)) {
      return day
    }
  }

  return null
}

function startOfDay(ms: number): number {
  const day = new Date(ms)
  day.setHours(0, 0, 0, 0)

  return day.getTime()
}

/** Today or tomorrow is said so; anything later is a date. */
function relativeOrDate(at: Date | null, nowMs: number): FirstRun | null {
  if (!at) {
    return null
  }

  const days = Math.round((startOfDay(at.getTime()) - startOfDay(nowMs)) / DAY_MS)

  return { at, style: days <= 1 ? 'relative' : 'date' }
}

const dated = (at: Date | null): FirstRun | null => (at ? { at, style: 'date' } : null)

type Rule = (pick: FirstRunPick, clock: [number, number], nowMs: number) => FirstRun | null

const RULES: Record<Exclude<FirstRunFrequency, 'custom' | 'hourly' | 'once'>, Rule> = {
  daily: (_pick, clock, nowMs) => relativeOrDate(nextDay(nowMs, clock, () => true), nowMs),
  weekdays: (_pick, clock, nowMs) => relativeOrDate(nextDay(nowMs, clock, day => day.getDay() % 6 !== 0), nowMs),
  weekly: (pick, clock, nowMs) => dated(nextDay(nowMs, clock, day => day.getDay() === Number(pick.weekday) % 7)),
  monthly: (pick, clock, nowMs) => dated(nextDay(nowMs, clock, day => day.getDate() === Number(pick.monthday)))
}

function nextHour(nowMs: number): FirstRun {
  const at = new Date(nowMs)
  at.setHours(at.getHours() + 1, 0, 0, 0)

  return { at, style: 'relative' }
}

function once(pick: FirstRunPick, clock: [number, number]): FirstRun | null {
  const match = ISO_DATE.exec(pick.date)

  return match
    ? { at: new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]), clock[0], clock[1]), style: 'once' }
    : null
}

/** The routine's first run, or null before a frequency is picked, for Custom words, and for a pick
 *  that would not save (a cleared time or date). */
export function nextFirstRun(pick: FirstRunPick | null, nowMs: number): FirstRun | null {
  if (!pick || pick.frequency === 'custom') {
    return null
  }

  if (pick.frequency === 'hourly') {
    return nextHour(nowMs)
  }

  const clock = clockOf(pick.time)

  if (!clock) {
    return null
  }

  return pick.frequency === 'once' ? once(pick, clock) : RULES[pick.frequency](pick, clock, nowMs)
}

/** "today, 9:00 AM" · "tomorrow, 9:00 AM" · "Mon, Nov 23, 9:00 AM" (with the year once it is not
 *  this year), in the locale's words and clock. */
export function firstRunText(run: FirstRun, locale: string, nowMs: number): string {
  const time = run.at.toLocaleTimeString(locale, { hour: 'numeric', minute: '2-digit' })

  if (run.style === 'relative') {
    const days = Math.round((startOfDay(run.at.getTime()) - startOfDay(nowMs)) / DAY_MS)

    return `${new Intl.RelativeTimeFormat(locale, { numeric: 'auto' }).format(days, 'day')}, ${time}`
  }

  const day = run.at.toLocaleDateString(locale, {
    day: 'numeric',
    month: 'short',
    weekday: 'short',
    ...(new Date(nowMs).getFullYear() === run.at.getFullYear() ? {} : { year: 'numeric' })
  })

  return `${day}, ${time}`
}
