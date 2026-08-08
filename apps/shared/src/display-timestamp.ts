export interface DisplayTimestampOptions {
  enabled: boolean
  format: string
}

const WEEKDAYS_SHORT = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as const
const WEEKDAYS_LONG = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'] as const
const MONTHS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] as const

const MONTHS_LONG = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December'
] as const

const pad = (value: number, width = 2) => String(value).padStart(width, '0')

const dayOfYear = (date: Date): number => {
  // UTC calendar arithmetic avoids an off-by-one after local daylight-saving
  // transitions, where consecutive local midnights can be 23 or 25 hours apart.
  const start = Date.UTC(date.getFullYear(), 0, 1)
  const current = Date.UTC(date.getFullYear(), date.getMonth(), date.getDate())

  return Math.floor((current - start) / 86_400_000) + 1
}

const weekNumber = (date: Date, firstWeekday: 0 | 1): number => {
  const firstDay = new Date(date.getFullYear(), 0, 1).getDay()
  const firstOccurrence = (7 + firstWeekday - firstDay) % 7
  const day = dayOfYear(date) - 1

  return day < firstOccurrence ? 0 : Math.floor((day - firstOccurrence) / 7) + 1
}

const isoWeekParts = (date: Date): { year: number; week: number; weekday: number } => {
  const weekday = ((date.getDay() + 6) % 7) + 1
  const thursday = new Date(date.getFullYear(), date.getMonth(), date.getDate() + (4 - weekday))
  const year = thursday.getFullYear()
  const januaryFirstWeekday = ((new Date(year, 0, 1).getDay() + 6) % 7) + 1
  const firstThursday = new Date(year, 0, 1 + ((4 - januaryFirstWeekday + 7) % 7))

  const days = Math.round(
    (Date.UTC(thursday.getFullYear(), thursday.getMonth(), thursday.getDate()) -
      Date.UTC(firstThursday.getFullYear(), firstThursday.getMonth(), firstThursday.getDate())) /
      86_400_000
  )

  return { year, week: Math.floor(days / 7) + 1, weekday }
}

const timezoneOffset = (date: Date): string => {
  const total = -date.getTimezoneOffset()
  const sign = total < 0 ? '-' : '+'
  const absolute = Math.abs(total)

  return `${sign}${pad(Math.floor(absolute / 60))}${pad(absolute % 60)}`
}

const timezoneName = (date: Date): string => {
  const part = new Intl.DateTimeFormat(undefined, { timeZoneName: 'short' })
    .formatToParts(date)
    .find(candidate => candidate.type === 'timeZoneName')

  return part?.value ?? ''
}

/**
 * Format a display-only timestamp using the shared Python ``strftime``-style
 * contract from ``display.timestamp_format``. The formatter intentionally
 * returns only the label; renderers own brackets, separators, and styling, so
 * timestamp text never enters model context or machine-readable payloads.
 */
export function formatDisplayTimestamp(value: Date | number | undefined, options: DisplayTimestampOptions): string {
  if (!options.enabled || value === undefined) {
    return ''
  }

  // Backend transcript timestamps are Unix seconds. Keep Date values as-is so
  // browser-native message dates remain unambiguous at the call site.
  const date = value instanceof Date ? value : new Date(value * 1000)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  const hour12 = date.getHours() % 12 || 12
  const iso = isoWeekParts(date)
  const year = date.getFullYear()
  const yearShort = pad(year % 100)
  const month = pad(date.getMonth() + 1)
  const day = pad(date.getDate())
  const hour = pad(date.getHours())
  const minute = pad(date.getMinutes())
  const second = pad(date.getSeconds())
  const meridiem = date.getHours() < 12 ? 'AM' : 'PM'

  const replacements: Record<string, string> = {
    '%': '%',
    a: WEEKDAYS_SHORT[date.getDay()],
    A: WEEKDAYS_LONG[date.getDay()],
    b: MONTHS_SHORT[date.getMonth()],
    B: MONTHS_LONG[date.getMonth()],
    h: MONTHS_SHORT[date.getMonth()],
    c: `${WEEKDAYS_SHORT[date.getDay()]} ${MONTHS_SHORT[date.getMonth()]} ${String(date.getDate()).padStart(2, ' ')} ${hour}:${minute}:${second} ${pad(year, 4)}`,
    C: pad(Math.floor(year / 100)),
    d: day,
    D: `${month}/${day}/${yearShort}`,
    e: String(date.getDate()).padStart(2, ' '),
    F: `${pad(year, 4)}-${month}-${day}`,
    f: pad(date.getMilliseconds() * 1000, 6),
    G: pad(iso.year, 4),
    g: pad(iso.year % 100),
    H: hour,
    I: pad(hour12),
    j: pad(dayOfYear(date), 3),
    k: String(date.getHours()).padStart(2, ' '),
    l: String(hour12).padStart(2, ' '),
    m: month,
    M: minute,
    n: '\n',
    p: meridiem,
    P: meridiem.toLowerCase(),
    r: `${pad(hour12)}:${minute}:${second} ${meridiem}`,
    R: `${hour}:${minute}`,
    s: String(Math.floor(date.getTime() / 1000)),
    S: second,
    t: '\t',
    T: `${hour}:${minute}:${second}`,
    u: String(iso.weekday),
    U: pad(weekNumber(date, 0)),
    V: pad(iso.week),
    w: String(date.getDay()),
    W: pad(weekNumber(date, 1)),
    x: `${month}/${day}/${yearShort}`,
    X: `${hour}:${minute}:${second}`,
    y: yearShort,
    Y: pad(year, 4),
    z: timezoneOffset(date),
    Z: timezoneName(date)
  }

  return String(options.format || '%H:%M').replace(/%([%a-zA-Z])/g, (token, directive: string) =>
    Object.prototype.hasOwnProperty.call(replacements, directive) ? replacements[directive] : token
  )
}
