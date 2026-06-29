// Heuristic JSON → human summary for tool results. Default view; technical
// mode still gets the raw JSON section.

const WRAPPER_KEYS = ['data', 'result', 'output', 'response', 'payload'] as const

const PRIORITY_KEYS = [
  'title',
  'name',
  'path',
  'file',
  'filepath',
  'url',
  'href',
  'link',
  'status',
  'id',
  'message',
  'summary',
  'description'
] as const

const ERROR_KEYS = ['error', 'errors', 'failure', 'exception'] as const
// 'stderr' deliberately excluded: many CLIs emit informational lines on
// stderr (npm progress, git's hint:, gcc's `In file included from`) that
// aren't errors. Treating those as error signal flipped tool cards into
// destructive styling for healthy commands.
const ERROR_MSG_KEYS = ['message', 'reason', 'detail'] as const
const NON_ERROR_TEXT = new Set(['', '0', 'false', 'none', 'null', 'nil', 'ok', 'success', 'n/a', 'na'])

type Json = Record<string, unknown>

const isRecord = (v: unknown): v is Json => Boolean(v && typeof v === 'object' && !Array.isArray(v))

function tryJson(value: string): unknown {
  const t = value.trim()

  if (!t) {
    return ''
  }

  if (!/^[{[]|^"/.test(t)) {
    return value
  }

  try {
    return JSON.parse(t)
  } catch {
    return value
  }
}

const norm = (v: unknown): unknown => (typeof v === 'string' ? tryJson(v) : v)

const titleCase = (k: string) =>
  k
    .split(/[_\-.]+/)
    .filter(Boolean)
    .map(p => `${p[0]?.toUpperCase() ?? ''}${p.slice(1)}`)
    .join(' ')

const pluralize = (n: number, noun: string) => `${n} ${noun}${n === 1 ? '' : 's'}`

function clipInline(value: string, max = 180): string {
  const c = value.replace(/\s+/g, ' ').trim()

  return c.length > max ? `${c.slice(0, max - 1)}…` : c
}

function clipBlock(value: string, maxChars = 1800, maxLines = 18): string {
  const t = value.trim()

  if (!t) {
    return ''
  }

  const lines = t.split('\n')
  let text = lines.slice(0, maxLines).join('\n')
  const clipped = lines.length > maxLines || text.length > maxChars

  if (text.length > maxChars) {
    text = text.slice(0, maxChars - 1).trimEnd()
  }

  return clipped && !text.endsWith('…') ? `${text}…` : text
}

function firstString(record: Json, keys: readonly string[]): string {
  for (const k of keys) {
    const v = record[k]

    if (typeof v === 'string' && v.trim()) {
      return v.trim()
    }
  }

  return ''
}

function orderedKeys(keys: string[]): string[] {
  const priority = PRIORITY_KEYS.filter(k => keys.includes(k))
  const rest = keys.filter(k => !priority.includes(k as never))

  return [...priority, ...rest]
}

const isWrapperKey = (k: string) => (WRAPPER_KEYS as readonly string[]).includes(k)
const skipField = (k: string, v: unknown) => isWrapperKey(k) || ((k === 'success' || k === 'ok') && v === true)

function summarizeScalar(v: unknown): string {
  if (typeof v === 'string') {
    return clipInline(v)
  }

  if (typeof v === 'number' || typeof v === 'boolean') {
    return String(v)
  }

  return ''
}

function summarizeRecordInline(record: Json, depth: number): string {
  if (depth > 3) {
    return pluralize(Object.keys(record).length, 'field')
  }

  const title = firstString(record, ['title', 'name', 'path', 'file', 'filepath', 'url', 'href', 'link', 'id'])
  const status = firstString(record, ['status', 'category', 'type'])
  const message = firstString(record, ['snippet', 'summary', 'description', 'message'])

  if (title && status) {
    return `${clipInline(title, 110)} (${clipInline(status, 54)})`
  }

  if (title && message && title !== message) {
    return `${clipInline(title, 90)} - ${clipInline(message, 84)}`
  }

  if (title) {
    return clipInline(title, 150)
  }

  const pairs = orderedKeys(Object.keys(record))
    .filter(k => !skipField(k, record[k]))
    .map(k => {
      const s = summarizeScalar(record[k])

      return s ? `${titleCase(k)}: ${s}` : ''
    })
    .filter(Boolean)
    .slice(0, 2)

  return pairs.length ? pairs.join(' · ') : pluralize(Object.keys(record).length, 'field')
}

function summarizeListItem(item: unknown, depth: number): string {
  const v = norm(item)

  if (typeof v === 'string') {
    return clipInline(v)
  }

  if (typeof v === 'number' || typeof v === 'boolean') {
    return String(v)
  }

  if (v == null) {
    return ''
  }

  if (Array.isArray(v)) {
    return pluralize(v.length, 'item')
  }

  if (isRecord(v)) {
    return summarizeRecordInline(v, depth + 1)
  }

  return clipInline(String(v))
}

function formatFieldValue(value: unknown, depth: number): string {
  const v = norm(value)
  const scalar = summarizeScalar(v)

  if (scalar) {
    return scalar
  }

  if (v == null) {
    return ''
  }

  if (Array.isArray(v)) {
    if (!v.length) {
      return ''
    }

    const scalars = v.map(summarizeScalar).filter(Boolean)

    if (scalars.length === v.length && v.length <= 4) {
      return clipInline(scalars.join(', '))
    }

    const first = summarizeListItem(v[0], depth + 1)

    return first ? `${pluralize(v.length, 'item')} (${first})` : pluralize(v.length, 'item')
  }

  if (isRecord(v)) {
    return summarizeRecordInline(v, depth + 1)
  }

  return clipInline(String(v))
}

// "Returned N items" / "0 items" / "Returned an empty object" are all
// noise — better to render nothing and let the title carry the signal.
function formatArraySummary(value: unknown[], depth: number): string {
  if (!value.length) {
    return ''
  }

  const max = 6

  const lines = value
    .slice(0, max)
    .map(item => summarizeListItem(item, depth + 1))
    .filter(Boolean)
    .map(l => `- ${l}`)

  if (!lines.length) {
    return ''
  }

  if (value.length > max) {
    const remaining = value.length - max
    lines.push(`- … ${remaining} more ${remaining === 1 ? 'item' : 'items'}`)
  }

  return lines.join('\n')
}

function formatRecordSummary(record: Json, depth: number): string {
  const keys = Object.keys(record)

  if (!keys.length) {
    return ''
  }

  if (depth <= 2) {
    const direct = firstString(record, ['message', 'summary', 'description', 'preview', 'text', 'content'])
    const meaningful = keys.filter(k => !skipField(k, record[k]) && !isWrapperKey(k))

    if (direct && meaningful.length <= 1) {
      return clipBlock(direct)
    }
  }

  const candidates = orderedKeys(keys).filter(k => !skipField(k, record[k]))
  const max = 8
  const lines: string[] = []
  let consumed = 0

  for (const k of candidates) {
    consumed += 1

    const v = formatFieldValue(record[k], depth + 1)

    if (!v) {
      continue
    }

    lines.push(`- ${titleCase(k)}: ${v}`)

    if (lines.length >= max) {
      break
    }
  }

  if (!lines.length) {
    return ''
  }

  // Only candidates we never examined (because the `max` cap broke the loop
  // early) are genuinely hidden. Candidates whose value rendered empty were
  // consumed and intentionally produced nothing — counting them here would
  // inflate the tail with fields that don't exist.
  const remaining = candidates.length - consumed

  if (remaining > 0) {
    lines.push(`- … ${remaining} more ${remaining === 1 ? 'field' : 'fields'}`)
  }

  return lines.join('\n')
}

function formatSummaryValue(value: unknown, depth: number): string {
  if (depth > 4) {
    return ''
  }

  const v = norm(value)

  if (typeof v === 'string') {
    return clipBlock(v)
  }

  if (typeof v === 'number' || typeof v === 'boolean') {
    return String(v)
  }

  if (v == null) {
    return ''
  }

  if (Array.isArray(v)) {
    return formatArraySummary(v, depth + 1)
  }

  if (isRecord(v)) {
    return formatRecordSummary(v, depth + 1)
  }

  return clipInline(String(v))
}

function unwrapPayload(value: unknown): unknown {
  let cur: unknown = norm(value)

  for (let i = 0; i < 4; i += 1) {
    if (!isRecord(cur)) {
      return cur
    }

    const record = cur
    const key = WRAPPER_KEYS.find(k => record[k] != null)

    if (!key) {
      return record
    }

    cur = norm(record[key])
  }

  return cur
}

function hasMeaningfulErrorValue(value: unknown): boolean {
  const v = norm(value)

  if (v == null) {
    return false
  }

  if (typeof v === 'string') {
    return !NON_ERROR_TEXT.has(v.trim().toLowerCase())
  }

  if (typeof v === 'boolean') {
    return v
  }

  if (typeof v === 'number') {
    return v !== 0
  }

  if (Array.isArray(v)) {
    return v.some(hasMeaningfulErrorValue)
  }

  if (isRecord(v)) {
    return Object.keys(v).length > 0
  }

  return true
}

function hasErrorSignal(record: Json): boolean {
  const status = typeof record.status === 'string' ? record.status : ''

  return (
    record.success === false ||
    record.ok === false ||
    /\b(error|failed|failure|fatal|exception)\b/i.test(status) ||
    ERROR_KEYS.some(k => hasMeaningfulErrorValue(record[k]))
  )
}

function valueErrorText(value: unknown): string {
  const v = norm(value)

  if (typeof v === 'string') {
    return hasMeaningfulErrorValue(v) ? clipBlock(v, 700, 12) : ''
  }

  if (Array.isArray(v)) {
    return clipBlock(v.map(valueErrorText).filter(Boolean).slice(0, 3).join('; '), 700, 12)
  }

  if (isRecord(v)) {
    const direct = firstString(v, ERROR_MSG_KEYS)

    if (direct) {
      return clipBlock(direct, 700, 12)
    }
  }

  return ''
}

function findNestedError(value: unknown, depth: number, seen: Set<unknown>): string {
  if (depth > 5) {
    return ''
  }

  const v = norm(value)

  if (!v || typeof v !== 'object' || seen.has(v)) {
    return ''
  }

  seen.add(v)

  if (Array.isArray(v)) {
    for (const item of v) {
      const nested = findNestedError(item, depth + 1, seen)

      if (nested) {
        return nested
      }
    }

    return ''
  }

  const record = v as Json

  for (const k of ERROR_KEYS) {
    if (!hasMeaningfulErrorValue(record[k])) {
      continue
    }

    const text = valueErrorText(record[k])

    if (text) {
      return text
    }
  }

  if (hasErrorSignal(record)) {
    const direct = firstString(record, ERROR_MSG_KEYS)

    if (direct) {
      return clipBlock(direct, 700, 12)
    }
  }

  for (const k of [...ERROR_KEYS, ...WRAPPER_KEYS, 'details', 'meta']) {
    const nested = findNestedError(record[k], depth + 1, seen)

    if (nested) {
      return nested
    }
  }

  return ''
}

export function formatToolResultSummary(value: unknown): string {
  return formatSummaryValue(unwrapPayload(value), 0) || formatSummaryValue(value, 0)
}

export function extractToolErrorMessage(value: unknown): string {
  return findNestedError(value, 0, new Set())
}
